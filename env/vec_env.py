"""Vectorized environment: N cubes in one tensor (contract D4, the speed path).

This is the ONLY place torch is allowed inside core+env. It holds N cubes as
integer tensors and turns a (possibly different) face on every cube with a single
``torch.gather`` per state array -- no Python loop over cubes. That is what lets
the trainer run 1 -> tens of thousands of agents on one device at near-flat
per-cube cost.

It is NOT a re-implementation of the cube. It loads the SAME move tables from
config/moves.json (via core.moves.MoveRegistry) and the SAME facelet tables from
core.render_state, converts them to tensors once at init, and applies them. The
pure single-cube path (core.moves.apply_move + env.cube_env) is the reference;
tests/test_vec_env.py asserts this tensor path is BIT-IDENTICAL to looping that
reference over the batch. If they ever disagree, the reference wins.

The "gather" convention (docs/move_notation.md) is exactly a gather:
    new.cp[:, i] = cp[:, perm[i]]   ->   torch.gather(cp, 1, perm_expanded)
with orientation deltas added mod 3 (corners) / mod 2 (edges).
"""

from __future__ import annotations

import torch

from core.cube import NUM_CORNERS, NUM_EDGES, CubeState
from core.moves import MoveRegistry, load_registry
from core.render_state import (
    CENTER_COLOR,
    CENTER_FACELET,
    CORNER_COLOR,
    CORNER_SOURCES,
    EDGE_COLOR,
    EDGE_SOURCES,
    NUM_COLORS,
    NUM_FACELETS,
)

OBS_SIZE = NUM_FACELETS * NUM_COLORS  # 324


class VecCubeEnv:
    """A batch of ``num_envs`` cubes stepped together as tensors."""

    def __init__(
        self,
        num_envs: int,
        *,
        config: dict | None = None,
        registry: MoveRegistry | None = None,
        device: str | torch.device = "cpu",
        max_steps: int = 50,
        seed: int | None = None,
    ):
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.registry = registry if registry is not None else load_registry()
        self.action_names = self.registry.action_names
        self.num_actions = len(self.action_names)  # 18
        self.max_steps = max_steps

        if config is not None:
            reward_cfg = config.get("env", {}).get("reward", {})
        else:
            reward_cfg = {}
        self._r_solved = float(reward_cfg.get("solved", 1.0))
        self._r_step = float(reward_cfg.get("per_step", 0.0))
        self._r_timeout = float(reward_cfg.get("timeout", 0.0))

        # Reproducible RNG living on the same device as the tensors.
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(seed if seed is not None else torch.seed())

        self._build_move_tables()
        self._build_render_tables()
        self._identity = self._make_identity()

        # Per-cube bookkeeping; populated by reset()/set_states().
        self.cp = self._identity[0].clone()
        self.co = torch.zeros(num_envs, NUM_CORNERS, dtype=torch.long, device=self.device)
        self.ep = self._identity[1].clone()
        self.eo = torch.zeros(num_envs, NUM_EDGES, dtype=torch.long, device=self.device)
        self.steps = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.depths = torch.zeros(num_envs, dtype=torch.long, device=self.device)

    # --- one-time table construction ---------------------------------------

    def _build_move_tables(self) -> None:
        """Stack the 18 actions' permutations into [18, 8] / [18, 12] tensors.

        Indexing these by a per-cube action vector ([N]) yields the per-cube
        permutation ([N, 8]) so a single gather serves a heterogeneous batch.
        """
        space = self.registry.action_space  # 18 Move objects, canonical order
        long, dev = torch.long, self.device
        self.CP_PERM = torch.tensor([m.cp for m in space], dtype=long, device=dev)  # [18,8]
        self.CO_ADD = torch.tensor([m.co for m in space], dtype=long, device=dev)   # [18,8]
        self.EP_PERM = torch.tensor([m.ep for m in space], dtype=long, device=dev)  # [18,12]
        self.EO_ADD = torch.tensor([m.eo for m in space], dtype=long, device=dev)   # [18,12]

    def _build_render_tables(self) -> None:
        """Tensorize the inverted facelet tables from core.render_state.

        For each facelet we precompute which slot/position feeds it, so rendering
        the whole batch is a handful of gathers (no per-cube Python).
        """
        long, dev = torch.long, self.device
        cf, cs, cq = zip(*CORNER_SOURCES)  # facelet, slot, pos for 24 corner facelets
        ef, es, eq = zip(*EDGE_SOURCES)
        self._cf = torch.tensor(cf, dtype=long, device=dev)
        self._cs = torch.tensor(cs, dtype=long, device=dev)
        self._cq = torch.tensor(cq, dtype=long, device=dev)
        self._ef = torch.tensor(ef, dtype=long, device=dev)
        self._es = torch.tensor(es, dtype=long, device=dev)
        self._eq = torch.tensor(eq, dtype=long, device=dev)
        self._corner_color = torch.tensor(CORNER_COLOR, dtype=long, device=dev)  # [8,3]
        self._edge_color = torch.tensor(EDGE_COLOR, dtype=long, device=dev)      # [12,2]
        self._center_facelet = torch.tensor(CENTER_FACELET, dtype=long, device=dev)  # [6]
        self._center_color = torch.tensor(CENTER_COLOR, dtype=long, device=dev)      # [6]

    def _make_identity(self) -> tuple[torch.Tensor, torch.Tensor]:
        cp = torch.arange(NUM_CORNERS, device=self.device).expand(self.num_envs, -1).clone()
        ep = torch.arange(NUM_EDGES, device=self.device).expand(self.num_envs, -1).clone()
        return cp, ep

    # --- state get/set ------------------------------------------------------

    def set_states(self, states: list[CubeState]) -> None:
        """Load an explicit list of CubeStates (must be len == num_envs)."""
        if len(states) != self.num_envs:
            raise ValueError(f"expected {self.num_envs} states, got {len(states)}")
        long, dev = torch.long, self.device
        self.cp = torch.tensor([s.cp for s in states], dtype=long, device=dev)
        self.co = torch.tensor([s.co for s in states], dtype=long, device=dev)
        self.ep = torch.tensor([s.ep for s in states], dtype=long, device=dev)
        self.eo = torch.tensor([s.eo for s in states], dtype=long, device=dev)
        self.steps = torch.zeros(self.num_envs, dtype=long, device=dev)
        self.depths = torch.zeros(self.num_envs, dtype=long, device=dev)

    def current_states(self) -> list[CubeState]:
        """Read the batch back as a list of CubeStates (for tests / recording)."""
        cp = self.cp.tolist()
        co = self.co.tolist()
        ep = self.ep.tolist()
        eo = self.eo.tolist()
        return [
            CubeState(cp=tuple(cp[n]), co=tuple(co[n]), ep=tuple(ep[n]), eo=tuple(eo[n]))
            for n in range(self.num_envs)
        ]

    @classmethod
    def from_states(cls, states: list[CubeState], **kwargs) -> "VecCubeEnv":
        env = cls(len(states), **kwargs)
        env.set_states(states)
        return env

    # --- core API -----------------------------------------------------------

    def reset(self, depths: int | torch.Tensor) -> torch.Tensor:
        """Reset all cubes to fresh scrambles of the given depth(s) -> [N, 324]."""
        self.cp, self.ep = (t.clone() for t in self._make_identity())
        self.co = torch.zeros(self.num_envs, NUM_CORNERS, dtype=torch.long, device=self.device)
        self.eo = torch.zeros(self.num_envs, NUM_EDGES, dtype=torch.long, device=self.device)
        self.steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.depths = self._as_depth_vector(depths)
        self._scramble(self.depths)
        return self.observation()

    def step(
        self, actions: torch.Tensor | list[int]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Apply per-cube ``actions`` ([N], each 0..17). Returns obs, reward, done, info."""
        actions = self._as_action_vector(actions)
        self.cp, self.co, self.ep, self.eo = self._apply(self.cp, self.co, self.ep, self.eo, actions)
        self.steps = self.steps + 1

        solved = self.is_solved()
        timed_out = self.steps >= self.max_steps
        done = solved | timed_out

        # Sparse-by-default reward, computed without a Python loop. Same semantics
        # as the single env: solved > timeout > per-step.
        reward = torch.full((self.num_envs,), self._r_step, dtype=torch.float32, device=self.device)
        reward = torch.where(timed_out & ~solved, torch.tensor(self._r_timeout, device=self.device), reward)
        reward = torch.where(solved, torch.tensor(self._r_solved, device=self.device), reward)

        info = {"solved": solved, "timed_out": timed_out & ~solved, "steps": self.steps.clone(), "depth": self.depths.clone()}
        return self.observation(), reward, done, info

    def observation(self) -> torch.Tensor:
        """Current batch observation: [N, 324] float32 one-hot, facelet-major.

        Layout matches the single env exactly: obs[n, f*6 + c] == 1 iff facelet f
        of cube n shows color c.
        """
        facelets = self._render()  # [N, 54] long
        onehot = torch.nn.functional.one_hot(facelets, num_classes=NUM_COLORS)  # [N,54,6]
        return onehot.reshape(self.num_envs, OBS_SIZE).to(torch.float32)

    def is_solved(self) -> torch.Tensor:
        """[N] bool: identity permutation and zero orientation, all O(1) per cube."""
        ident_c = torch.arange(NUM_CORNERS, device=self.device)
        ident_e = torch.arange(NUM_EDGES, device=self.device)
        return (
            (self.cp == ident_c).all(dim=1)
            & (self.ep == ident_e).all(dim=1)
            & (self.co == 0).all(dim=1)
            & (self.eo == 0).all(dim=1)
        )

    # --- internals ----------------------------------------------------------

    def _apply(
        self,
        cp: torch.Tensor,
        co: torch.Tensor,
        ep: torch.Tensor,
        eo: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """One move per cube via gather; the heart of the speed path."""
        cp_perm = self.CP_PERM[actions]  # [N,8] per-cube source indices
        co_add = self.CO_ADD[actions]
        ep_perm = self.EP_PERM[actions]  # [N,12]
        eo_add = self.EO_ADD[actions]
        new_cp = torch.gather(cp, 1, cp_perm)
        new_co = (torch.gather(co, 1, cp_perm) + co_add) % 3
        new_ep = torch.gather(ep, 1, ep_perm)
        new_eo = (torch.gather(eo, 1, ep_perm) + eo_add) % 2
        return new_cp, new_co, new_ep, new_eo

    def _render(self) -> torch.Tensor:
        """Project the batch to [N, 54] color ints (inverse facelet tables)."""
        n = self.num_envs
        facelets = torch.empty(n, NUM_FACELETS, dtype=torch.long, device=self.device)

        # Corners: color at facelet = CORNER_COLOR[cp[slot]][(q - co[slot]) % 3].
        cp_sel = self.cp[:, self._cs]               # [N,24]
        co_sel = self.co[:, self._cs]               # [N,24]
        c_n = (self._cq.unsqueeze(0) - co_sel) % 3  # [N,24]
        facelets[:, self._cf] = self._corner_color[cp_sel, c_n]

        # Edges: analogous, mod 2.
        ep_sel = self.ep[:, self._es]
        eo_sel = self.eo[:, self._es]
        e_n = (self._eq.unsqueeze(0) - eo_sel) % 2
        facelets[:, self._ef] = self._edge_color[ep_sel, e_n]

        # Centers are constant across the batch.
        facelets[:, self._center_facelet] = self._center_color
        return facelets

    def _scramble(self, depths: torch.Tensor) -> None:
        """Vectorized backward scramble to per-cube depths.

        Mirrors core.scramble's hygiene (no two consecutive moves on the same
        face) so depth stays a meaningful distance, but does it for all cubes at
        once. It does NOT reproduce the single env's exact RNG stream -- only its
        statistics and validity. Per-cube depths are honored by masking: once a
        cube has taken `depth` moves, further steps are no-ops for it.
        """
        if self.num_envs == 0:
            return
        max_depth = int(depths.max().item())
        # Face group of an action is action // 3 (3 actions per face, canonical order).
        prev_group = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        for t in range(max_depth):
            active = t < depths  # [N] bool: this cube still scrambling

            rand_group = torch.randint(0, 6, (self.num_envs,), generator=self.generator, device=self.device)
            offset = torch.randint(0, 5, (self.num_envs,), generator=self.generator, device=self.device)
            # A face different from the previous one, uniform over the other 5.
            alt_group = (prev_group + 1 + offset) % 6
            group = torch.where(prev_group < 0, rand_group, alt_group)
            suffix = torch.randint(0, 3, (self.num_envs,), generator=self.generator, device=self.device)
            actions = group * 3 + suffix

            new_cp, new_co, new_ep, new_eo = self._apply(self.cp, self.co, self.ep, self.eo, actions)
            m_c = active.unsqueeze(1)  # broadcast mask over the 8/12 columns
            self.cp = torch.where(m_c, new_cp, self.cp)
            self.co = torch.where(m_c, new_co, self.co)
            self.ep = torch.where(m_c, new_ep, self.ep)
            self.eo = torch.where(m_c, new_eo, self.eo)
            prev_group = torch.where(active, group, prev_group)

    # --- input coercion -----------------------------------------------------

    def _as_action_vector(self, actions: torch.Tensor | list[int]) -> torch.Tensor:
        t = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        if t.shape != (self.num_envs,):
            raise ValueError(f"actions must have shape ({self.num_envs},), got {tuple(t.shape)}")
        if (t < 0).any() or (t >= self.num_actions).any():
            raise ValueError(f"actions must be in [0,{self.num_actions})")
        return t

    def _as_depth_vector(self, depths: int | torch.Tensor) -> torch.Tensor:
        if isinstance(depths, int):
            return torch.full((self.num_envs,), depths, dtype=torch.long, device=self.device)
        t = torch.as_tensor(depths, dtype=torch.long, device=self.device)
        if t.shape != (self.num_envs,):
            raise ValueError(f"depths must be a scalar or shape ({self.num_envs},)")
        return t
