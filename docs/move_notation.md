# Move notation & cubie numbering

This is the ground truth the move tables in [`config/moves.json`](../config/moves.json)
are written against. If you ever doubt a table, read this, then let
[`tests/test_moves.py`](../tests/test_moves.py) be the judge.

> Scope: this documents the **internal cubie model**, the **6 base generators**
> (Phase 1), and the **54-sticker projection** (Phase 2). Slices/wide moves and
> view rotations are later phases and are intentionally not defined here yet.

## The cubie model (state)

A 3x3 cube is a permutation of 8 corner pieces and 12 edge pieces, each with an
orientation. Centers are fixed (they *define* the colors) and are not modelled.
State is four integer arrays:

| array | len | meaning | orientation modulus |
|---|---|---|---|
| `cp` | 8 | which corner **id** sits in each corner slot | — |
| `co` | 8 | twist of the corner in each slot | mod 3 (`{0,1,2}`) |
| `ep` | 12 | which edge **id** sits in each edge slot | — |
| `eo` | 12 | flip of the edge in each slot | mod 2 (`{0,1}`) |

Solved = identity permutation + all-zero orientation. Solved-check is O(1).

### Corner numbering (slots **and** ids 0–7)

Letters are the faces a corner touches; `U/D` up/down, `F/B` front/back, `L/R` left/right.

| id | label | id | label |
|---|---|---|---|
| 0 | URF | 4 | DFR |
| 1 | UFL | 5 | DLF |
| 2 | ULB | 6 | DBL |
| 3 | UBR | 7 | DRB |

### Edge numbering (slots **and** ids 0–11)

| id | label | id | label | id | label |
|---|---|---|---|---|---|
| 0 | UR | 4 | DR | 8 | FR |
| 1 | UF | 5 | DF | 9 | FL |
| 2 | UL | 6 | DL | 10 | BL |
| 3 | UB | 7 | DB | 11 | BR |

This is the standard Kociemba numbering, which is why the base tables match the
well-known reference values exactly.

## How a move is applied — the "gather" convention

A move is four arrays with the **same shapes** as the state. They are *source
indices* plus *orientation deltas*. To apply move `M` to state `S`, for every
slot `i`:

```
S'.cp[i] = S.cp[ M.cp[i] ]                 # slot i pulls its corner from slot M.cp[i]
S'.co[i] = (S.co[ M.cp[i] ] + M.co[i]) % 3 # ...and is twisted by M.co[i]
S'.ep[i] = S.ep[ M.ep[i] ]
S'.eo[i] = (S.eo[ M.ep[i] ] + M.eo[i]) % 2
```

Two consequences make this convention worth its weight:

1. **A move table is also a cube state.** Applying `M` to *solved* (identity
   perm, zero orient) just copies the tables out, so `M.cp` etc. literally equal
   "the cube after one `M` from solved." You can read a table and picture it.
2. **It is a `gather`.** `new = old[index]` is exactly `torch.gather`. The pure
   core ([`core/moves.py`](../core/moves.py)) applies it with Python indexing;
   the Phase-2 tensor env will apply the identical tables to `[N, …]` tensors in
   one op. Single-cube and N-cube paths share one definition.

### Orientation: why corners are mod 3 and edges mod 2

Orientation counts how a piece is rotated *in place* relative to a reference
facelet. A corner has 3 facelets, so its twist is mod 3; an edge has 2, so its
flip is mod 2. `U`/`D` turns never change any orientation (orientation is defined
relative to the U/D axis). `R`/`L`/`F`/`B` twist corners and `F`/`B` flip edges,
and the deltas around each turned face always sum to `0 mod (3 or 2)` — you can
never twist one corner or flip one edge alone on a real cube. The tables encode
exactly those deltas.

## Deriving the rest (you only author 6 tables)

From each base quarter turn the loader derives:

- **prime** `R'` = `invert(R)` — pull back the other way, negate the twist deltas.
- **double** `R2` = `compose(R, R)` — apply the gather twice; deltas add.

So 6 hand-written tables + two tiny operations give the **18-move agent action
space** (`R R' R2 … B B'`, see `face_order` in the JSON). Inverses are never
transcribed by hand — that is the table most likely to be wrong.

## The proofs (Phase-1 gate)

The tables are trusted only because these all pass:

| identity | catches |
|---|---|
| `face^4 = I` | wrong permutation cycle |
| `(R U R' U')^6 = I` | composition / twist / flip bug ("sexy move", order 6) |
| `M · M' = I` | bad inverse |
| `M2 = M · M` | bad double |
| batched == looped | vectorized-indexing bug |
| scramble → reversed-inverse solves | round-trip end to end |

## The 54-sticker projection (Phase 2)

The cubie model is the source of truth; the **54-sticker color grid** is the
derived view that the neural net and the browser consume. It is produced by
[`core/render_state.py`](../core/render_state.py).

### Facelet ordering

54 facelets, grouped by face in this fixed order, **9 per face, row-major**
(top row left→right, then middle row, then bottom row, as you look straight at
the face):

```
U: 0..8     R: 9..17    F: 18..26    D: 27..35    L: 36..44    B: 45..53
```

So facelet index = `9 * face + (3 * row + col)`, with faces ordered
`U,R,F,D,L,B` (= color values `0..5`). This is the standard Kociemba facelet
layout. The picture:

```
            U0 U1 U2
            U3 U4 U5
            U6 U7 U8
   L36..    F18..    R9..     B45..
   L39..    F21..    R12..    B48..
   L42..    F24..    R15..    B51..
            D27 D28 D29
            D30 D31 D32
            D33 D34 D35
```

### Colors and centers

Color values equal the face order: `U=0 R=1 F=2 D=3 L=4 B=5`. Centers define the
scheme and never move — facelet `9*k + 4` always shows color `k`.

### How a sticker gets its color

Each movable piece carries its colors. For the corner in slot `i` (holding cubie
`cp[i]` with twist `co[i]`), the color at facelet `CORNER_FACELET[i][q]` is
`CORNER_COLOR[cp[i]][(q - co[i]) % 3]`; edges are the same with two facelets and
`mod 2`. This is exactly Kociemba's `toFaceCube`, written for our gather state.

### Observation encoding

The env one-hot-encodes the 54-vector into **324 dims**, facelet-major with color
as the fast axis: `obs[9*... ]` → precisely `obs[f*6 + c] = 1` iff facelet `f`
shows color `c`. The single env ([`env/cube_env.py`](../env/cube_env.py)) and the
vectorized env ([`env/vec_env.py`](../env/vec_env.py)) share this exact layout.

### Two independent proofs (Phase-2 gate)

| check | catches |
|---|---|
| geometric sticker-perm == `render(apply(s,M))` for all 18 | wrong facelet/color table |
| render injective on random states | projection loses information |
| centers invariant under every move | slice leakage / projection bug |
| vec `step` bit-identical to looped single `apply_move` | tensor path inherited tables wrong |
