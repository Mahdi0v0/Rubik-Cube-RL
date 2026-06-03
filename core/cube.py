"""The cube's internal state -- the cubie model (contract D1).

WHY a cubie model and not a 54-sticker grid?
    A Rubik's cube is, mathematically, a permutation group acting on 20 movable
    pieces: 8 corners and 12 edges. (The 6 centers are fixed -- they define the
    color scheme and never move relative to each other, so we don't model them.)
    Tracking the 20 pieces directly -- *which* piece sits in each slot and *how*
    it is twisted -- is the smallest faithful description of the state. A move is
    then a permutation with orientation bookkeeping, the solved-check is O(1)
    (is every piece home and untwisted?), and the whole thing needs no graphics.
    The 54-sticker color grid the UI and the neural net consume is a *derived
    view* of this state (see core/render_state.py in Phase 2), never the source
    of truth.

This module is part of the PURE core: standard library only, no torch, no web.
You can read and test all of the cube logic with zero ML dependencies installed.
"""

from __future__ import annotations

from dataclasses import dataclass

# --- Sizes of the two piece classes. Fixed for a 3x3; named to kill magic numbers.
NUM_CORNERS = 8
NUM_EDGES = 12

# Orientation moduli: a corner can be twisted 3 ways, an edge flipped 2 ways.
CORNER_ORI_MOD = 3
EDGE_ORI_MOD = 2

# Human-readable slot labels, in index order. These define the numbering that the
# move tables in config/moves.json are written against; see docs/move_notation.md.
CORNER_LABELS = ("URF", "UFL", "ULB", "UBR", "DFR", "DLF", "DBL", "DRB")
EDGE_LABELS = ("UR", "UF", "UL", "UB", "DR", "DF", "DL", "DB", "FR", "FL", "BL", "BR")


@dataclass(frozen=True)
class CubeState:
    """One full cube configuration as four integer tuples (contract D1).

    Fields (all positional within their tuple = the slot index):
        cp -- corner permutation: cp[i] is the *id* of the corner currently in
              slot i. Identity (cp[i] == i) means "the right corner is home".
        co -- corner orientation in slot i, in {0,1,2} (twist, mod 3).
        ep -- edge permutation: ep[i] is the id of the edge in slot i.
        eo -- edge orientation in slot i, in {0,1} (flip, mod 2).

    Tuples (not lists) so a state is immutable and hashable -- a move returns a
    *new* state rather than mutating in place, which makes the algebra easy to
    reason about and lets states be used as dict keys / set members in search.

    This same four-array shape is the JSON wire format (see ``serialize``); the
    browser and checkpoints speak exactly this.
    """

    cp: tuple[int, ...]
    co: tuple[int, ...]
    ep: tuple[int, ...]
    eo: tuple[int, ...]

    # --- Construction -------------------------------------------------------

    @classmethod
    def solved(cls) -> "CubeState":
        """The goal state: every piece home (identity perm) and untwisted."""
        return cls(
            cp=tuple(range(NUM_CORNERS)),
            co=(0,) * NUM_CORNERS,
            ep=tuple(range(NUM_EDGES)),
            eo=(0,) * NUM_EDGES,
        )

    @classmethod
    def from_dict(cls, d: dict) -> "CubeState":
        """Rebuild a state from the JSON wire form ``{cp, co, ep, eo}``."""
        state = cls(
            cp=tuple(d["cp"]),
            co=tuple(d["co"]),
            ep=tuple(d["ep"]),
            eo=tuple(d["eo"]),
        )
        state.validate()  # never trust data coming off the wire / disk
        return state

    # --- Queries ------------------------------------------------------------

    def is_solved(self) -> bool:
        """True iff this is the goal state.

        O(1): the comparison touches a fixed 8+8+12+12 = 40 integers regardless
        of how scrambled the cube is. We never search, never compare stickers --
        a cube is solved exactly when both permutations are the identity and both
        orientation arrays are all zero. (Comparing to the shared SOLVED singleton
        below short-circuits on the first mismatch.)
        """
        return self == SOLVED

    def serialize(self) -> dict:
        """Return the JSON-ready dict ``{cp, co, ep, eo}`` (lists, not tuples).

        This is the canonical wire format shared by the episode files, the
        checkpoint ``start_state``, and the browser playground.
        """
        return {
            "cp": list(self.cp),
            "co": list(self.co),
            "ep": list(self.ep),
            "eo": list(self.eo),
        }

    def validate(self) -> None:
        """Assert the three physical invariants of a *reachable* cube state.

        A random fill of these arrays is almost never a real cube. Only states
        reachable by face turns satisfy all three:
          1. cp and ep are genuine permutations (each id appears exactly once).
          2. Total corner twist is 0 mod 3  -- you cannot twist a single corner
             in isolation on a real cube.
          3. Total edge flip is 0 mod 2     -- you cannot flip a single edge.
        (There is also a permutation-parity tie between corners and edges; we
        don't enforce it here because every state we *construct* -- solved, or
        solved + legal moves -- satisfies it by construction. Moves preserve all
        of these, so this is mainly a guard on externally supplied data.)
        """
        if sorted(self.cp) != list(range(NUM_CORNERS)):
            raise ValueError(f"cp is not a permutation of 0..{NUM_CORNERS - 1}: {self.cp}")
        if sorted(self.ep) != list(range(NUM_EDGES)):
            raise ValueError(f"ep is not a permutation of 0..{NUM_EDGES - 1}: {self.ep}")
        if len(self.co) != NUM_CORNERS or any(c not in (0, 1, 2) for c in self.co):
            raise ValueError(f"co must be {NUM_CORNERS} values in {{0,1,2}}: {self.co}")
        if len(self.eo) != NUM_EDGES or any(e not in (0, 1) for e in self.eo):
            raise ValueError(f"eo must be {NUM_EDGES} values in {{0,1}}: {self.eo}")
        if sum(self.co) % CORNER_ORI_MOD != 0:
            raise ValueError(f"corner twist not conserved (sum(co) % 3 != 0): {self.co}")
        if sum(self.eo) % EDGE_ORI_MOD != 0:
            raise ValueError(f"edge flip not conserved (sum(eo) % 2 != 0): {self.eo}")

    # --- Debugging niceties -------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        tag = "SOLVED" if self == SOLVED else "scrambled"
        return f"CubeState({tag}, cp={self.cp}, co={self.co}, ep={self.ep}, eo={self.eo})"


# A single shared solved instance. ``is_solved`` compares against this; it is also
# the natural starting point for backward scrambling (Phase 2) and for any test.
SOLVED = CubeState.solved()
