"""Single-cube environment (contract D4). Pure stdlib -- the readable reference.

This is the clarity-first env: one cube, plain Python, no torch. It exists to be
easy to read and to serve as the ground truth that the vectorized tensor env
(env/vec_env.py) is checked against, step for step.

Interface (custom, NOT Gymnasium):
    reset(scramble_depth) -> obs
    step(action: int)     -> (obs, reward, done, info)
    observation()         -> 324-dim one-hot of the 54-sticker view

Action space: the 18 face turns, indexed 0..17 in the registry's canonical order
(env.action_names gives the mapping). Nothing else -- no slices/wide/rotations.

Reward: a CONFIG-DRIVEN placeholder (config/default.json -> env.reward), sparse by
default (+1 on solved, else 0). This deliberately does NOT shape toward "closer to
solved": measuring closeness is the whole hard problem and the ADI learner gets
its signal from the scramble-depth label, not from this number. Reward weights
are data so a research pass can tune them without touching the architecture.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from core.cube import CubeState
from core.moves import MoveRegistry, apply_move, load_registry
from core.render_state import NUM_COLORS, NUM_FACELETS, render_state
from core.scramble import scramble

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "default.json"

# Observation size: each of the 54 facelets is one-hot over 6 colors.
OBS_SIZE = NUM_FACELETS * NUM_COLORS  # 324


def load_config(path: Path | str | None = None) -> dict:
    """Load a JSON config (defaults to config/default.json)."""
    path = Path(path) if path is not None else _DEFAULT_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def one_hot_observation(state: CubeState) -> list[float]:
    """54-sticker view -> 324-dim one-hot, as ``obs[facelet*6 + color] = 1.0``.

    This exact layout (facelet-major, color the fast axis) is the contract the
    network encoder and the vectorized env must agree on. Returned as Python
    floats here; env/vec_env.py returns the identical layout as a float32 tensor.
    """
    facelets = render_state(state)
    obs = [0.0] * OBS_SIZE
    for f, color in enumerate(facelets):
        obs[f * NUM_COLORS + color] = 1.0
    return obs


class CubeEnv:
    """One scrambled cube the agent acts on until solved or out of steps."""

    def __init__(
        self,
        *,
        config: dict | None = None,
        registry: MoveRegistry | None = None,
        max_steps: int | None = None,
        seed: int | None = None,
    ):
        self.config = config if config is not None else load_config()
        self.registry = registry if registry is not None else load_registry()
        self.action_names = self.registry.action_names
        self.num_actions = len(self.action_names)  # 18

        env_cfg = self.config.get("env", {})
        # Explicit arg overrides config; config overrides a sane fallback.
        self.max_steps = max_steps if max_steps is not None else env_cfg.get("max_steps", 50)
        reward_cfg = env_cfg.get("reward", {})
        self._r_solved = float(reward_cfg.get("solved", 1.0))
        self._r_step = float(reward_cfg.get("per_step", 0.0))
        self._r_timeout = float(reward_cfg.get("timeout", 0.0))

        self.rng = random.Random(seed)

        # Episode state (set on reset).
        self._state: CubeState | None = None
        self._steps = 0
        self._scramble_depth = 0

    # --- core API -----------------------------------------------------------

    def reset(self, scramble_depth: int, *, seed: int | None = None) -> list[float]:
        """Start a new episode from a fresh depth-``scramble_depth`` scramble."""
        if seed is not None:
            self.rng = random.Random(seed)
        self._scramble_depth = scramble_depth
        self._state = scramble(scramble_depth, self.rng, registry=self.registry)
        self._steps = 0
        return self.observation()

    def step(self, action: int) -> tuple[list[float], float, bool, dict]:
        """Apply action (0..17); return (obs, reward, done, info)."""
        if self._state is None:
            raise RuntimeError("step() called before reset()")
        if not 0 <= action < self.num_actions:
            raise ValueError(f"action {action} out of range [0,{self.num_actions})")

        move = self.registry[self.action_names[action]]
        self._state = apply_move(self._state, move)
        self._steps += 1

        solved = self._state.is_solved()
        timed_out = self._steps >= self.max_steps
        done = solved or timed_out

        # Sparse-by-default reward; see module docstring on why we don't shape.
        if solved:
            reward = self._r_solved
        elif timed_out:
            reward = self._r_timeout
        else:
            reward = self._r_step

        info = {
            "depth": self._scramble_depth,
            "steps": self._steps,
            "solved": solved,
            "timed_out": timed_out and not solved,
        }
        return self.observation(), reward, done, info

    def observation(self) -> list[float]:
        """Current 324-dim one-hot observation."""
        assert self._state is not None, "no state; call reset() first"
        return one_hot_observation(self._state)

    # --- introspection (handy for tests / recording) ------------------------

    @property
    def state(self) -> CubeState:
        assert self._state is not None, "no state; call reset() first"
        return self._state

    @property
    def steps(self) -> int:
        return self._steps

    def is_solved(self) -> bool:
        return self._state is not None and self._state.is_solved()
