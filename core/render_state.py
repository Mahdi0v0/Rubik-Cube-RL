"""Project the cubie state onto the 54-sticker color grid (the derived view).

The cubie model (core/cube.py) is the source of truth, but neither the neural
net nor the browser want to reason about corner twists -- they want *colors*.
This module is the one-way projection cubie-state -> 54 stickers, each an integer
color in 0..5. It is the shared input form for:
  * the NN encoder (one-hot of these 54 ints -> 324-dim, built in env/),
  * the browser playground (the episode JSON carries exactly this 54-vector),
  * the vectorized env (env/vec_env.py reuses the tables below as tensors).

Pure core: standard library only. No torch, no web.

------------------------------------------------------------------------------
Facelet numbering (see docs/move_notation.md for the picture)
------------------------------------------------------------------------------
54 facelets, grouped by face in this fixed order, 9 each, row-major (top row
left->right, then middle, then bottom) as you look straight at the face:

    U: 0..8     R: 9..17    F: 18..26    D: 27..35    L: 36..44    B: 45..53

Center colors define the scheme and never move: U=0 R=1 F=2 D=3 L=4 B=5.
This is the standard Kociemba facelet layout, which is why the corner/edge
facelet tables below match the well-known reference values.

------------------------------------------------------------------------------
How a sticker gets its color
------------------------------------------------------------------------------
Each movable piece carries its colors with it. For the corner sitting in slot i:
its three stickers live at the facelet positions CORNER_FACELET[i], and the
colors painted there are those of the corner *cubie* currently in that slot,
CORNER_COLOR[cp[i]], cyclically shifted by the slot's orientation co[i]. Twist
co=1 means "rotate the colors one facelet clockwise about the corner". Edges are
the same idea with two facelets and a flip (eo in {0,1}). This is exactly
Kociemba's `toFaceCube`, just written for our gather-convention state.
"""

from __future__ import annotations

from core.cube import CubeState

# --- Colors, named to kill magic numbers. Values double as the center facelet
#     colors and as the face order of the 54-vector.
U, R, F, D, L, B = 0, 1, 2, 3, 4, 5
FACE_NAMES = ("U", "R", "F", "D", "L", "B")
NUM_FACELETS = 54
NUM_COLORS = 6

# Index of the first facelet of each face (face k starts at 9*k).
FACE_BASE = {name: 9 * k for k, name in enumerate(FACE_NAMES)}


# --------------------------------------------------------------------------- #
# The facelet tables (standard Kociemba numbering).
#
# Slot order matches core.cube: corners URF,UFL,ULB,UBR,DFR,DLF,DBL,DRB and
# edges UR,UF,UL,UB,DR,DF,DL,DB,FR,FL,BL,BR.
#
# CORNER_FACELET[i] = the 3 facelet indices of corner slot i, listed so that
#   index 0 is the U/D-axis facelet (the orientation reference). CORNER_COLOR[j]
#   = the 3 colors of corner cubie j in the SAME order.  For corner slot i
#   holding cubie cp[i] with twist co[i], the color shown at CORNER_FACELET[i][q]
#   is CORNER_COLOR[cp[i]][(q - co[i]) % 3].
# --------------------------------------------------------------------------- #

CORNER_FACELET: tuple[tuple[int, int, int], ...] = (
    (8, 9, 20),    # URF : U9, R1, F3
    (6, 18, 38),   # UFL : U7, F1, L3
    (0, 36, 47),   # ULB : U1, L1, B3
    (2, 45, 11),   # UBR : U3, B1, R3
    (29, 26, 15),  # DFR : D3, F9, R7
    (27, 44, 24),  # DLF : D1, L9, F7
    (33, 53, 42),  # DBL : D7, B9, L7
    (35, 17, 51),  # DRB : D9, R9, B7
)
CORNER_COLOR: tuple[tuple[int, int, int], ...] = (
    (U, R, F),  # URF
    (U, F, L),  # UFL
    (U, L, B),  # ULB
    (U, B, R),  # UBR
    (D, F, R),  # DFR
    (D, L, F),  # DLF
    (D, B, L),  # DBL
    (D, R, B),  # DRB
)

EDGE_FACELET: tuple[tuple[int, int], ...] = (
    (5, 10),   # UR : U6, R2
    (7, 19),   # UF : U8, F2
    (3, 37),   # UL : U4, L2
    (1, 46),   # UB : U2, B2
    (32, 16),  # DR : D6, R8
    (28, 25),  # DF : D2, F8
    (30, 43),  # DL : D4, L8
    (34, 52),  # DB : D8, B8
    (23, 12),  # FR : F6, R4
    (21, 41),  # FL : F4, L6
    (50, 39),  # BL : B6, L4
    (48, 14),  # BR : B4, R6
)
EDGE_COLOR: tuple[tuple[int, int], ...] = (
    (U, R),  # UR
    (U, F),  # UF
    (U, L),  # UL
    (U, B),  # UB
    (D, R),  # DR
    (D, F),  # DF
    (D, L),  # DL
    (D, B),  # DB
    (F, R),  # FR
    (F, L),  # FL
    (B, L),  # BL
    (B, R),  # BR
)

# Centers: facelet index -> fixed color. Position 4 within each face is the center.
CENTER_FACELET: tuple[int, ...] = tuple(FACE_BASE[name] + 4 for name in FACE_NAMES)
CENTER_COLOR: tuple[int, ...] = (U, R, F, D, L, B)


# --------------------------------------------------------------------------- #
# Precomputed "per-facelet source" decomposition.
#
# render() writes colors piece-by-piece (the readable forward direction). For the
# vectorized tensor render we instead want, for each *facelet*, a rule to read its
# color directly: which slot and which slot-position it belongs to. We invert the
# facelet tables ONCE here so env/vec_env.py can gather without re-deriving.
#
#   CORNER_SOURCES[k] = (facelet, slot_i, pos_q)  for the 24 corner facelets
#   EDGE_SOURCES[k]   = (facelet, slot_i, pos_q)  for the 24 edge facelets
# color(facelet) = CORNER_COLOR[cp[slot_i]][(pos_q - co[slot_i]) % 3]   (corners)
#               = EDGE_COLOR[ep[slot_i]][(pos_q - eo[slot_i]) % 2]      (edges)
# --------------------------------------------------------------------------- #

CORNER_SOURCES: tuple[tuple[int, int, int], ...] = tuple(
    (CORNER_FACELET[i][q], i, q) for i in range(8) for q in range(3)
)
EDGE_SOURCES: tuple[tuple[int, int, int], ...] = tuple(
    (EDGE_FACELET[i][q], i, q) for i in range(12) for q in range(2)
)


def render_state(state: CubeState) -> list[int]:
    """Return the 54-sticker color grid (ints 0..5) for ``state``.

    Forward / piece-by-piece direction (mirrors Kociemba's toFaceCube), which is
    the easiest to read: for each slot, lay the cubie's colors onto its facelets,
    rotated by the slot's orientation. The vectorized env produces the identical
    vector via the inverted SOURCES tables; test_vec_env cross-checks the two.
    """
    facelets = [0] * NUM_FACELETS

    # Corners: twist co shifts which facelet shows which color (mod 3).
    for i in range(8):
        cubie = state.cp[i]
        twist = state.co[i]
        for n in range(3):
            facelets[CORNER_FACELET[i][(n + twist) % 3]] = CORNER_COLOR[cubie][n]

    # Edges: flip eo swaps the two facelets (mod 2).
    for i in range(12):
        cubie = state.ep[i]
        flip = state.eo[i]
        for n in range(2):
            facelets[EDGE_FACELET[i][(n + flip) % 2]] = EDGE_COLOR[cubie][n]

    # Centers are fixed.
    for f, color in zip(CENTER_FACELET, CENTER_COLOR):
        facelets[f] = color

    return facelets
