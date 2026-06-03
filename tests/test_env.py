"""Acceptance tests for the single-cube env (env/cube_env.py)."""

from __future__ import annotations

import random

import pytest

from core.cube import SOLVED
from core.render_state import render_state
from core.scramble import scramble_with_solution
from env.cube_env import OBS_SIZE, CubeEnv, one_hot_observation


def _action_index(env: CubeEnv, name: str) -> int:
    return env.action_names.index(name)


def test_observation_is_valid_one_hot():
    env = CubeEnv(seed=0)
    obs = env.reset(scramble_depth=15)
    assert len(obs) == OBS_SIZE == 324
    assert sum(obs) == 54  # exactly one hot bit per facelet
    # Each facelet's 6-block has exactly one 1.0.
    for f in range(54):
        block = obs[f * 6 : f * 6 + 6]
        assert sum(block) == 1.0 and set(block) <= {0.0, 1.0}


def test_depth_zero_reset_is_solved_observation():
    env = CubeEnv(seed=1)
    obs = env.reset(scramble_depth=0)
    assert env.is_solved()
    assert obs == one_hot_observation(SOLVED)


def test_solution_via_steps_solves_and_rewards():
    # Build a known scramble + its free solution, then drive the env with it.
    res = scramble_with_solution(20, random.Random(42))
    env = CubeEnv(seed=0, max_steps=100)
    # Put the env into exactly this scrambled state by reusing the same seed path:
    # simpler -- reset to solved then check we can solve an independently scrambled
    # env via its own state. Here we instead verify the env solves the SAME state
    # by reconstructing it through reset with a matching scramble.
    # (Direct state injection isn't part of the public API; we drive a fresh env.)
    env.reset(scramble_depth=0)
    # Manually scramble through the public step API using res.scramble, then solve.
    for name in res.scramble:
        env.step(_action_index(env, name))
    assert not env.is_solved()

    done = False
    reward = 0.0
    info: dict = {}
    for name in res.solution:
        _obs, reward, done, info = env.step(_action_index(env, name))
    assert done and info["solved"]
    assert reward == 1.0


def test_step_limit_times_out():
    env = CubeEnv(seed=0, max_steps=3)
    env.reset(scramble_depth=20)
    last_info = {}
    done = False
    for _ in range(3):
        _obs, _r, done, last_info = env.step(0)
    assert done
    assert last_info["steps"] == 3
    assert last_info["timed_out"] is True
    assert last_info["solved"] is False


def test_step_before_reset_raises():
    env = CubeEnv(seed=0)
    with pytest.raises(RuntimeError):
        env.step(0)


def test_action_out_of_range_raises():
    env = CubeEnv(seed=0)
    env.reset(scramble_depth=1)
    with pytest.raises(ValueError):
        env.step(18)
    with pytest.raises(ValueError):
        env.step(-1)


def test_reward_is_config_driven():
    # Override reward weights via config; env must honor them, not hard-code.
    cfg = {
        "env": {
            "max_steps": 5,
            "reward": {"solved": 7.0, "per_step": -0.5, "timeout": -3.0},
        }
    }
    env = CubeEnv(config=cfg, seed=0)
    env.reset(scramble_depth=20)
    # First (non-solving) step pays per_step; eventual timeout pays timeout.
    _obs, r0, done0, _ = env.step(0)  # step 1 of 5
    assert r0 == -0.5 and not done0
    for _ in range(4):  # steps 2..5; the 5th hits max_steps
        _obs, r, done, info = env.step(0)
    assert done and info["timed_out"]
    assert info["steps"] == 5
    assert r == -3.0


def test_render_consistency_between_obs_and_state():
    env = CubeEnv(seed=3)
    env.reset(scramble_depth=12)
    for a in range(5):
        obs, _r, _d, _i = env.step(a)
        # Decode the one-hot back to colors and compare to a direct render.
        decoded = [obs[f * 6 : f * 6 + 6].index(1.0) for f in range(54)]
        assert decoded == render_state(env.state)
