"""Phase-2 gate: the tensor env must be BIT-IDENTICAL to the single-cube reference.

The vectorized env reuses the same move/facelet tables but applies them with
torch gathers. The only way to trust that is to prove that, for arbitrary states
and arbitrary per-cube actions, one vectorized step equals looping the pure
``core.moves.apply_move`` over the batch -- exactly, not approximately. Every
other vec_env test is secondary to this one.
"""

from __future__ import annotations

import random

import pytest

torch = pytest.importorskip("torch")  # vec_env is the one torch-dependent module

from core.cube import SOLVED
from core.moves import apply_move, apply_sequence, load_registry
from env.cube_env import CubeEnv, one_hot_observation
from env.vec_env import OBS_SIZE, VecCubeEnv

REG = load_registry()


def _random_states(n: int, seed: int):
    rng = random.Random(seed)
    states = []
    for _ in range(n):
        s = SOLVED
        for _ in range(rng.randint(0, 40)):
            s = apply_move(s, REG[rng.choice(REG.action_names)])
        states.append(s)
    return states


# --------------------------------------------------------------------------- #
# THE gate test.
# --------------------------------------------------------------------------- #


def test_step_is_bit_identical_to_looped_single_for_all_actions():
    # For each of the 18 actions, apply it to a batch of random states in the
    # tensor env and assert the resulting states equal the per-cube reference.
    for action in range(18):
        states = _random_states(64, seed=2000 + action)
        env = VecCubeEnv.from_states(states, max_steps=10_000)
        env.step([action] * len(states))
        got = env.current_states()
        want = [apply_move(s, REG[REG.action_names[action]]) for s in states]
        assert got == want, f"vec step != single for action {action}"


def test_step_bit_identical_with_random_per_cube_actions():
    # The realistic case: every cube does a *different* move in the same step.
    rng = random.Random(99)
    states = _random_states(256, seed=7)
    actions = [rng.randrange(18) for _ in states]

    env = VecCubeEnv.from_states(states, max_steps=10_000)
    env.step(actions)
    got = env.current_states()
    want = [apply_move(s, REG[REG.action_names[a]]) for s, a in zip(states, actions)]
    assert got == want


def test_observation_matches_single_env_layout():
    states = _random_states(128, seed=11)
    env = VecCubeEnv.from_states(states)
    obs = env.observation()  # [N,324] float32
    assert obs.shape == (128, OBS_SIZE)
    assert obs.dtype == torch.float32
    for n, s in enumerate(states):
        assert obs[n].tolist() == one_hot_observation(s)


def test_multi_step_trajectory_matches_reference():
    # Drive 30 random steps and compare the whole trajectory of states.
    rng = random.Random(5)
    states = _random_states(48, seed=3)
    env = VecCubeEnv.from_states(states, max_steps=10_000)
    ref = list(states)
    for _ in range(30):
        actions = [rng.randrange(18) for _ in states]
        env.step(actions)
        ref = [apply_move(s, REG[REG.action_names[a]]) for s, a in zip(ref, actions)]
        assert env.current_states() == ref


# --------------------------------------------------------------------------- #
# reset / solved behavior matches the single env.
# --------------------------------------------------------------------------- #


def test_depth_zero_reset_is_solved_and_matches_single():
    env = VecCubeEnv(16, seed=0)
    obs = env.reset(depths=0)
    assert env.is_solved().all()
    solved_obs = one_hot_observation(SOLVED)
    for n in range(16):
        assert obs[n].tolist() == solved_obs


def test_reset_scrambles_are_valid_and_unsolved():
    env = VecCubeEnv(512, seed=1234)
    env.reset(depths=20)
    # Valid cubes (invariants hold) -- read back and validate via the pure core.
    for s in env.current_states():
        s.validate()
    # Depth-20 scrambles are essentially never already solved.
    assert env.is_solved().sum().item() == 0


def test_per_cube_depths_are_honored():
    # Cube 0 gets depth 0 (stays solved); the rest get depth 15.
    n = 8
    depths = torch.tensor([0] + [15] * (n - 1))
    env = VecCubeEnv(n, seed=7)
    env.reset(depths=depths)
    solved = env.is_solved()
    assert solved[0].item() is True
    assert solved[1:].sum().item() == 0


def test_is_solved_matches_single_env():
    states = _random_states(200, seed=321)
    env = VecCubeEnv.from_states(states)
    vec_solved = env.is_solved().tolist()
    single_solved = [s.is_solved() for s in states]
    assert vec_solved == single_solved


def test_vectorized_solution_solves_batch():
    # Build per-cube scrambles, then feed each cube its own free solution.
    rng = random.Random(8)
    scrambles = []
    states = []
    for _ in range(32):
        seq = [rng.choice(REG.action_names) for _ in range(12)]
        scrambles.append(seq)
        states.append(apply_sequence(SOLVED, [REG[m] for m in seq]))

    env = VecCubeEnv.from_states(states, max_steps=10_000)
    solutions = [[_inv(m) for m in reversed(seq)] for seq in scrambles]
    for t in range(12):
        actions = [REG.action_names.index(solutions[i][t]) for i in range(32)]
        _obs, _r, done, info = env.step(actions)
    assert env.is_solved().all()
    assert done.all() and info["solved"].all()


def test_reward_and_done_semantics():
    # One cube depth-1, solved by its inverse; rewards sparse by default (+1 solved).
    env = VecCubeEnv(4, seed=2, max_steps=3)
    env.reset(depths=0)  # all solved already; step a move then undo
    _obs, _r, _d, _i = env.step([0, 0, 0, 0])          # R on all -> unsolved
    obs, reward, done, info = env.step([1, 1, 1, 1])   # R' on all -> solved
    assert env.is_solved().all()
    assert torch.allclose(reward, torch.ones(4))
    assert done.all() and info["solved"].all()


def _inv(name: str) -> str:
    base = name[0]
    if name.endswith("2"):
        return name
    if name.endswith("'"):
        return base
    return base + "'"
