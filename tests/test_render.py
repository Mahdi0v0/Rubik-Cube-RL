"""Acceptance tests for the 54-sticker projection (core/render_state.py).

The crown jewel here is the GOLD-STANDARD test: we build the sticker-level
permutation of each face turn from *independent geometry* -- a facelet ->
(position, normal) table plus 3D rotation -- and assert it agrees with
``render(apply(state, move))``. The geometric generator never touches
render_state's CORNER_FACELET/CORNER_COLOR tables, so this is two independent
encodings of the same physical cube agreeing, which is what actually proves the
facelet tables are right (not merely self-consistent).
"""

from __future__ import annotations

import random

import pytest

from core.cube import SOLVED
from core.moves import apply_move, apply_sequence, load_registry
from core.render_state import (
    FACE_NAMES,
    NUM_COLORS,
    NUM_FACELETS,
    render_state,
)

REG = load_registry()


def _scrambled(seed: int, depth: int):
    rng = random.Random(seed)
    return apply_sequence(SOLVED, [REG[rng.choice(REG.action_names)] for _ in range(depth)])


# --------------------------------------------------------------------------- #
# Cheap structural checks.
# --------------------------------------------------------------------------- #


def test_solved_has_nine_of_each_color_in_solid_faces():
    f = render_state(SOLVED)
    assert len(f) == NUM_FACELETS
    for color in range(NUM_COLORS):
        assert f.count(color) == 9
    # Each face is a solid block of its center's color, in face order U,R,F,D,L,B.
    for k in range(NUM_COLORS):
        face = f[9 * k : 9 * k + 9]
        assert face == [k] * 9, f"face {FACE_NAMES[k]} not solid: {face}"


def test_centers_are_invariant_under_every_move():
    # Centers never move; this is cheap but a strong guard on the projection and
    # on the move tables (a slice move leaking in would break it instantly). The
    # center facelet of face k is index 9*k+4 and always shows color k.
    for name in REG.action_names:
        s = _scrambled(seed=7, depth=20)
        before = render_state(s)
        after = render_state(apply_move(s, REG[name]))
        for k in range(NUM_COLORS):
            c = 9 * k + 4
            assert after[c] == before[c] == k


def test_render_is_injective_on_random_states():
    # Distinct *legal* cube states must map to distinct 54-vectors (the sticker
    # view is a faithful encoding). We map render -> state; a collision to two
    # different states would mean the projection loses information.
    seen: dict[tuple[int, ...], object] = {}
    rng = random.Random(2024)
    for _ in range(8000):
        s = apply_sequence(
            SOLVED, [REG[rng.choice(REG.action_names)] for _ in range(rng.randint(1, 25))]
        )
        key = tuple(render_state(s))
        if key in seen:
            assert seen[key] == s, "two different states rendered to the same stickers"
        else:
            seen[key] = s


# --------------------------------------------------------------------------- #
# GOLD STANDARD: geometric sticker permutations vs. render(apply(...)).
#
# Independent geometry: each facelet has a 3D position (cubie center, coords in
# {-1,0,1}) and an outward normal (the face it is on). A clockwise face turn is a
# -90 deg rotation about the outward normal applied to every facelet on that
# layer. We build the resulting facelet permutation from scratch and compare.
# --------------------------------------------------------------------------- #

# (r,c) -> world (x,y,z) for each face, derived from the standard net geometry
# (documented in docs/move_notation.md). Facelet index = base + r*3 + c.
_FACE_POS_RULE = {
    "U": lambda r, c: (c - 1, 1, r - 1),
    "R": lambda r, c: (1, 1 - r, 1 - c),
    "F": lambda r, c: (c - 1, 1 - r, 1),
    "D": lambda r, c: (c - 1, -1, 1 - r),
    "L": lambda r, c: (-1, 1 - r, c - 1),
    "B": lambda r, c: (1 - c, 1 - r, -1),
}
_FACE_NORMAL = {
    "U": (0, 1, 0),
    "R": (1, 0, 0),
    "F": (0, 0, 1),
    "D": (0, -1, 0),
    "L": (-1, 0, 0),
    "B": (0, 0, -1),
}

# Build per-facelet position & normal, and the inverse lookup (pos,normal)->index.
_POS: list[tuple[int, int, int]] = [None] * NUM_FACELETS  # type: ignore
_NRM: list[tuple[int, int, int]] = [None] * NUM_FACELETS  # type: ignore
for _k, _name in enumerate(FACE_NAMES):
    _base = 9 * _k
    for _r in range(3):
        for _c in range(3):
            _idx = _base + _r * 3 + _c
            _POS[_idx] = _FACE_POS_RULE[_name](_r, _c)
            _NRM[_idx] = _FACE_NORMAL[_name]
_INDEX_AT = {(_POS[i], _NRM[i]): i for i in range(NUM_FACELETS)}

# Clockwise (-90 deg about the outward normal) rotations, verified against the
# trusted cubie tables in core (e.g. under R the DFR corner moves to URF).
_ROT = {
    "U": lambda x, y, z: (-z, y, x),
    "D": lambda x, y, z: (z, y, -x),
    "R": lambda x, y, z: (x, z, -y),
    "L": lambda x, y, z: (x, -z, y),
    "F": lambda x, y, z: (y, -x, z),
    "B": lambda x, y, z: (-y, x, z),
}
# Which coordinate component selects a face's turning layer, and its value.
_LAYER = {
    "U": (1, 1),
    "D": (1, -1),
    "R": (0, 1),
    "L": (0, -1),
    "F": (2, 1),
    "B": (2, -1),
}


def _geometric_quarter_perm(face: str) -> list[int]:
    """Gather perm P for a clockwise quarter turn: new_render[x] = old_render[P[x]]."""
    rot = _ROT[face]
    axis, val = _LAYER[face]
    perm = list(range(NUM_FACELETS))
    for a in range(NUM_FACELETS):
        if _POS[a][axis] != val:
            continue  # not on this turning layer
        new_pos = rot(*_POS[a])
        new_nrm = rot(*_NRM[a])
        b = _INDEX_AT[(new_pos, new_nrm)]  # facelet the sticker at `a` moves to
        perm[b] = a
    return perm


def _gather(vec: list[int], perm: list[int]) -> list[int]:
    return [vec[perm[x]] for x in range(NUM_FACELETS)]


def _compose(p_first: list[int], p_second: list[int]) -> list[int]:
    # Apply p_first then p_second, both gather perms: result[x] = p_first[p_second[x]].
    return [p_first[p_second[x]] for x in range(NUM_FACELETS)]


def _invert(perm: list[int]) -> list[int]:
    inv = [0] * NUM_FACELETS
    for x, src in enumerate(perm):
        inv[src] = x
    return inv


def _build_geometric_perms() -> dict[str, list[int]]:
    """All 18 sticker perms, derived purely geometrically (no render tables)."""
    perms: dict[str, list[int]] = {}
    for face in ["U", "R", "F", "D", "L", "B"]:
        q = _geometric_quarter_perm(face)
        perms[face] = q
        perms[face + "'"] = _invert(q)
        perms[face + "2"] = _compose(q, q)
    return perms


_GEO_PERMS = _build_geometric_perms()


def test_geometric_quarter_turns_have_order_four():
    # Sanity on the independent generator itself before we trust it as an oracle.
    for face in ["U", "R", "F", "D", "L", "B"]:
        p = _GEO_PERMS[face]
        composed = list(range(NUM_FACELETS))
        for _ in range(4):
            composed = _compose(p, composed)
        assert composed == list(range(NUM_FACELETS)), f"geometric {face}^4 != identity"


@pytest.mark.parametrize("name", REG.action_names)
def test_render_matches_geometric_sticker_permutation(name):
    geo = _GEO_PERMS[name]
    for seed in range(6):
        s = _scrambled(seed=seed * 13 + 1, depth=18)
        expected = _gather(render_state(s), geo)         # geometry says stickers move thus
        actual = render_state(apply_move(s, REG[name]))  # cubie move + projection
        assert actual == expected, f"render disagrees with geometry for move {name}"
