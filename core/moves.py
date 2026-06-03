"""Moves as permutations (contract D2): load, derive, apply.

This module loads the 6 base face-turn generators from ``config/moves.json`` and
DERIVES everything else from them with pure group theory:

  * the prime  R'  = the inverse of R   (undoes it),
  * the double R2  = R applied twice,

so the 18-move agent action space (6 faces x {quarter, prime, double}) is built
from 6 hand-authored tables and three small, testable operations -- compose and
invert. You never hand-write an inverse; that is exactly the kind of table you'd
get subtly wrong.

Application uses the 'gather' convention (see docs/move_notation.md and the
header of config/moves.json): a move pulls each slot's new contents from some
source slot and adds an orientation delta. That single rule covers corners and
edges, single states and batches, and -- crucially -- it is literally a
``gather`` in tensor terms, which is how the Phase-2 torch env (env/vec_env.py)
will run thousands of cubes at once. This file stays in the PURE core: standard
library only, no torch. The batched path here is a pure-Python cross-check that
the same tables vectorize correctly; the tensor speed-path reuses these tables.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from core.cube import (
    CORNER_ORI_MOD,
    EDGE_ORI_MOD,
    NUM_CORNERS,
    NUM_EDGES,
    CubeState,
)

# config/moves.json lives at <repo>/config/moves.json; this file is <repo>/core/moves.py.
_DEFAULT_REGISTRY_PATH = Path(__file__).resolve().parent.parent / "config" / "moves.json"


@dataclass(frozen=True)
class Move:
    """One move as a 'gather + twist' rule (contract D2).

    For every slot ``i`` of the new state:
        new.cp[i] = old.cp[cp[i]];  new.co[i] = (old.co[cp[i]] + co[i]) % 3
        new.ep[i] = old.ep[ep[i]];  new.eo[i] = (old.eo[ep[i]] + eo[i]) % 2

    ``cp``/``ep`` are *source* indices (where slot i pulls from); ``co``/``eo``
    are orientation deltas applied after the pull. Applying a Move to the solved
    state simply reads the tables out, so a Move's own tables equal the cube
    state it produces from solved -- handy for sanity-checking by eye.
    """

    name: str
    cp: tuple[int, ...]  # corner source indices, len 8
    co: tuple[int, ...]  # corner twist deltas (mod 3), len 8
    ep: tuple[int, ...]  # edge source indices, len 12
    eo: tuple[int, ...]  # edge flip deltas (mod 2), len 12


# --------------------------------------------------------------------------- #
# Applying a move
# --------------------------------------------------------------------------- #


def apply_move(state: CubeState, move: Move) -> CubeState:
    """Apply ``move`` to a single ``state`` and return the new state.

    This is the one true definition of what a move *does*; the batched version
    below must agree with looping this over a batch (proved in test_moves.py).
    """
    cp, co, ep, eo = state.cp, state.co, state.ep, state.eo
    m_cp, m_co, m_ep, m_eo = move.cp, move.co, move.ep, move.eo

    new_cp = tuple(cp[m_cp[i]] for i in range(NUM_CORNERS))
    new_co = tuple((co[m_cp[i]] + m_co[i]) % CORNER_ORI_MOD for i in range(NUM_CORNERS))
    new_ep = tuple(ep[m_ep[i]] for i in range(NUM_EDGES))
    new_eo = tuple((eo[m_ep[i]] + m_eo[i]) % EDGE_ORI_MOD for i in range(NUM_EDGES))
    return CubeState(cp=new_cp, co=new_co, ep=new_ep, eo=new_eo)


def apply_sequence(state: CubeState, moves: list[Move]) -> CubeState:
    """Apply a list of moves left-to-right (moves[0] first)."""
    for m in moves:
        state = apply_move(state, m)
    return state


def apply_move_batch(states: list[CubeState], move: Move) -> list[CubeState]:
    """Apply ``move`` to a batch of states, structure-of-arrays style.

    WHY this exists in the pure core: the Phase-2 environment will hold N cubes
    in one tensor and turn a face by a single ``gather`` -- the speed path. Before
    we get there (and before torch is even a dependency), this proves the *same*
    permutation tables vectorize correctly: we lay the batch out column-major
    (one list per slot across all cubes) and gather per slot, rather than looping
    whole cubes. test_moves.py asserts this equals the looped single-state path,
    so any indexing bug in the vectorized layout is caught here, in plain Python,
    long before the tensor version inherits the same tables.
    """
    n = len(states)
    if n == 0:
        return []

    m_cp, m_co, m_ep, m_eo = move.cp, move.co, move.ep, move.eo

    # Column-major view: cp_cols[i][k] = corner-perm of slot i for cube k.
    cp_cols = [[s.cp[i] for s in states] for i in range(NUM_CORNERS)]
    co_cols = [[s.co[i] for s in states] for i in range(NUM_CORNERS)]
    ep_cols = [[s.ep[i] for s in states] for i in range(NUM_EDGES)]
    eo_cols = [[s.eo[i] for s in states] for i in range(NUM_EDGES)]

    # Gather each output slot from its source column, across the whole batch.
    new_cp_cols = [cp_cols[m_cp[i]] for i in range(NUM_CORNERS)]
    new_co_cols = [
        [(c + m_co[i]) % CORNER_ORI_MOD for c in co_cols[m_cp[i]]]
        for i in range(NUM_CORNERS)
    ]
    new_ep_cols = [ep_cols[m_ep[i]] for i in range(NUM_EDGES)]
    new_eo_cols = [
        [(e + m_eo[i]) % EDGE_ORI_MOD for e in eo_cols[m_ep[i]]]
        for i in range(NUM_EDGES)
    ]

    # Re-assemble row-major (one CubeState per cube).
    out: list[CubeState] = []
    for k in range(n):
        out.append(
            CubeState(
                cp=tuple(new_cp_cols[i][k] for i in range(NUM_CORNERS)),
                co=tuple(new_co_cols[i][k] for i in range(NUM_CORNERS)),
                ep=tuple(new_ep_cols[i][k] for i in range(NUM_EDGES)),
                eo=tuple(new_eo_cols[i][k] for i in range(NUM_EDGES)),
            )
        )
    return out


# --------------------------------------------------------------------------- #
# Deriving moves: compose (square) and invert (prime)
# --------------------------------------------------------------------------- #


def compose(first: Move, second: Move, name: str) -> Move:
    """Return the move equal to applying ``first`` then ``second``.

    Under the gather rule, composing two gathers is itself a gather:
        (second o first).cp[i] = first.cp[ second.cp[i] ]
    and the twist deltas add (the inner move's delta at the pulled slot, plus the
    outer move's delta at slot i). We compute this purely from the tables, so a
    double R2 = compose(R, R) is derived, never transcribed.
    """
    cp = tuple(first.cp[second.cp[i]] for i in range(NUM_CORNERS))
    co = tuple(
        (first.co[second.cp[i]] + second.co[i]) % CORNER_ORI_MOD for i in range(NUM_CORNERS)
    )
    ep = tuple(first.ep[second.ep[i]] for i in range(NUM_EDGES))
    eo = tuple(
        (first.eo[second.ep[i]] + second.eo[i]) % EDGE_ORI_MOD for i in range(NUM_EDGES)
    )
    return Move(name=name, cp=cp, co=co, ep=ep, eo=eo)


def invert(move: Move, name: str) -> Move:
    """Return the inverse move (R' from R): the move that undoes ``move``.

    If ``move`` pulls slot i from slot p[i] and adds twist a[i], then the inverse
    must pull back the other way -- p_inv[p[i]] = i -- and apply the *negated*
    twist at the destination so the two cancel:
        invert.co[i] = (-move.co[p_inv[i]]) % 3
    (For edges, mod 2, negation is a no-op, but we keep the formula uniform.)
    Verified by the identity ``apply(apply(s, M), M') == s`` for every move.
    """
    cp_inv = _invert_perm(move.cp)
    ep_inv = _invert_perm(move.ep)
    co = tuple((-move.co[cp_inv[i]]) % CORNER_ORI_MOD for i in range(NUM_CORNERS))
    eo = tuple((-move.eo[ep_inv[i]]) % EDGE_ORI_MOD for i in range(NUM_EDGES))
    return Move(name=name, cp=cp_inv, co=co, ep=ep_inv, eo=eo)


def _invert_perm(perm: tuple[int, ...]) -> tuple[int, ...]:
    """Inverse of a permutation given as source indices: out[perm[i]] = i."""
    out = [0] * len(perm)
    for i, src in enumerate(perm):
        out[src] = i
    return tuple(out)


def inverse_name(name: str) -> str:
    """Name of the move that undoes ``name`` (R->R', R'->R, R2->R2).

    Works on the standard face-turn notation (one face letter + optional ' or 2).
    Used by the scramble generator and episode recorder to turn a scramble into
    its free solution -- the reverse sequence of inverse moves.
    """
    face = name[0]
    if name.endswith("2"):
        return name  # a half turn is its own inverse
    if name.endswith("'"):
        return face
    return face + "'"


# --------------------------------------------------------------------------- #
# The registry: load 6 bases, derive 18 actions
# --------------------------------------------------------------------------- #


class MoveRegistry:
    """All moves the system knows, built from the 6 base generators.

    The agent action space is the 18 face turns (6 faces x quarter/prime/double),
    exposed in a fixed, documented order via ``action_names`` / ``action_space``.
    The order is data-driven (``face_order`` + ``suffixes`` in the JSON) so the
    network's output head and the move list never drift apart.
    """

    def __init__(self, config: dict):
        bases = config["base_generators"]
        face_order = config["face_order"]
        suffixes = config["suffixes"]

        self.moves: dict[str, Move] = {}
        self.action_names: list[str] = []

        for face in face_order:
            table = bases[face]
            quarter = Move(
                name=face,
                cp=tuple(table["cp"]),
                co=tuple(table["co"]),
                ep=tuple(table["ep"]),
                eo=tuple(table["eo"]),
            )
            prime = invert(quarter, face + suffixes["prime"])
            double = compose(quarter, quarter, face + suffixes["double"])

            for mv in (quarter, prime, double):
                self.moves[mv.name] = mv
            # Canonical action order: quarter, prime, double -- per face, in face_order.
            self.action_names.extend(
                [face + suffixes["quarter"], prime.name, double.name]
            )

        # Defensive: the action space must be exactly the 18 distinct face turns.
        assert len(self.action_names) == 18, self.action_names
        assert len(set(self.action_names)) == 18, "duplicate action names"

    @property
    def action_space(self) -> list[Move]:
        """The 18 agent actions as Move objects, in canonical order."""
        return [self.moves[name] for name in self.action_names]

    def __getitem__(self, name: str) -> Move:
        return self.moves[name]

    def __contains__(self, name: str) -> bool:
        return name in self.moves

    def __len__(self) -> int:
        return len(self.moves)


def load_registry(path: Path | str | None = None) -> MoveRegistry:
    """Load the move registry from config/moves.json (or a custom path)."""
    path = Path(path) if path is not None else _DEFAULT_REGISTRY_PATH
    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return MoveRegistry(config)
