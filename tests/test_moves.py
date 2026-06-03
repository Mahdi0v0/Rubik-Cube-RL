"""The Phase-1 gate: prove the move algebra (CLAUDE.md section 8).

These tests are the *only* reason to trust the hand-authored tables in
config/moves.json. Each one targets a specific class of bug:

    R^4 = I              wrong permutation cycle length
    (R U R' U')^6 = I    composition / corner-twist / edge-flip bugs ("sexy move")
    |R U| = 105, ...     EXTERNAL ground-truth element orders (see below)
    M . M' = I           bad inverse derivation
    M2 = M . M           bad double derivation
    batched == looped    vectorization / indexing bug in the SoA batch path
    scramble -> reverse  scramble/inverse round-trip end to end

A note on what actually proves the tables. Most identities here (M.M'=I, M2=M.M,
scramble round-trips) are SELF-CONSISTENCY: primes/doubles/inverses are all
derived from the same six base tables, so they would stay green even if those
tables were geometrically wrong. The real external anchors -- facts about the
cube group that do NOT depend on our derivation -- are the element orders:
    |R|=4, |R U R' U'|=6, |R U|=105, |F R|=105, |R U'|=63, |R U2|=30, |R L|=4.
|R U|=105 in particular is the classic catcher for a transposed move table.

If you change a table or a derivation and these stay green, you're fine.
"""

from __future__ import annotations

import random

import pytest

from core.cube import (
    EDGE_ORI_MOD,
    CORNER_ORI_MOD,
    SOLVED,
    CubeState,
)
from core.moves import (
    apply_move,
    apply_move_batch,
    apply_sequence,
    compose,
    load_registry,
)

REG = load_registry()
FACES = ["R", "L", "U", "D", "F", "B"]


# --------------------------------------------------------------------------- #
# Sanity: the registry shape
# --------------------------------------------------------------------------- #


def test_registry_has_18_actions():
    assert len(REG.action_names) == 18
    assert len(REG.action_space) == 18
    # 6 quarter + 6 prime + 6 double, all distinct.
    assert len(set(REG.action_names)) == 18


def test_solved_state_is_solved():
    assert SOLVED.is_solved()
    assert CubeState.solved().is_solved()


# --------------------------------------------------------------------------- #
# R^4 = I  (and every face^4 = I): a quarter turn has order 4.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("face", FACES)
def test_quarter_turn_has_order_four(face):
    s = SOLVED
    for _ in range(4):
        s = apply_move(s, REG[face])
    assert s.is_solved(), f"{face}^4 did not return to solved"


@pytest.mark.parametrize("face", FACES)
def test_quarter_turn_is_not_identity_too_early(face):
    # Guards against a degenerate table that is accidentally the identity.
    s = SOLVED
    for k in range(1, 4):
        s = apply_move(s, REG[face])
        assert not s.is_solved(), f"{face}^{k} should not be solved yet"


# --------------------------------------------------------------------------- #
# (R U R' U')^6 = I : the "sexy move" has order 6.
# --------------------------------------------------------------------------- #


def test_sexy_move_has_order_six():
    seq = [REG["R"], REG["U"], REG["R'"], REG["U'"]]
    s = SOLVED
    for rep in range(1, 7):
        s = apply_sequence(s, seq)
        if rep < 6:
            assert not s.is_solved(), f"sexy move solved too early at rep {rep}"
    assert s.is_solved(), "(R U R' U')^6 did not return to solved"


# --------------------------------------------------------------------------- #
# EXTERNAL GROUND TRUTH: element orders.
#
# These numbers are properties of the Rubik's-cube group itself -- you can look
# them up or compute them on a physical cube -- and they do NOT depend on how we
# authored or derived our tables. A table that is internally consistent but
# geometrically wrong (e.g. two facelets transposed) will fail these, where it
# would pass every self-referential identity. |R U| = 105 is the canonical one.
# --------------------------------------------------------------------------- #


def _order(seq_names: list[str]) -> int:
    """Smallest n >= 1 such that repeating the move sequence n times solves it."""
    s = SOLVED
    n = 0
    while True:
        s = apply_sequence(s, [REG[m] for m in seq_names])
        n += 1
        if s.is_solved():
            return n
        assert n < 10_000, f"sequence {seq_names} has no finite order?!"


@pytest.mark.parametrize(
    "seq,expected",
    [
        (["R"], 4),
        (["R", "U", "R'", "U'"], 6),
        (["R", "U"], 105),
        (["F", "R"], 105),
        (["R", "U'"], 63),
        (["R", "U2"], 30),
        (["R", "L"], 4),  # opposite faces commute -> small order
    ],
)
def test_known_element_orders(seq, expected):
    assert _order(seq) == expected, f"|{' '.join(seq)}| should be {expected}"


# --------------------------------------------------------------------------- #
# M . M' = I : each derived prime undoes its base (and vice versa).
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("face", FACES)
def test_move_then_inverse_is_identity(face):
    prime = face + "'"
    assert apply_sequence(SOLVED, [REG[face], REG[prime]]).is_solved()
    assert apply_sequence(SOLVED, [REG[prime], REG[face]]).is_solved()


@pytest.mark.parametrize("name", [f + s for f in FACES for s in ("", "'", "2")])
def test_every_action_has_a_working_inverse(name):
    # The inverse of any of the 18 actions is also one of the 18 actions.
    base = name[0]
    if name.endswith("2"):
        inv_name = name  # a double is its own inverse
    elif name.endswith("'"):
        inv_name = base
    else:
        inv_name = base + "'"
    assert apply_sequence(SOLVED, [REG[name], REG[inv_name]]).is_solved()


# --------------------------------------------------------------------------- #
# M2 = M . M : the derived double equals applying the quarter twice.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("face", FACES)
def test_double_equals_quarter_squared(face):
    via_double = apply_move(SOLVED, REG[face + "2"])
    via_twice = apply_sequence(SOLVED, [REG[face], REG[face]])
    assert via_double == via_twice
    # ...and a double has order 2.
    assert apply_sequence(SOLVED, [REG[face + "2"], REG[face + "2"]]).is_solved()


def test_double_derivation_matches_compose():
    for face in FACES:
        derived = compose(REG[face], REG[face], face + "2")
        assert derived.cp == REG[face + "2"].cp
        assert derived.co == REG[face + "2"].co
        assert derived.ep == REG[face + "2"].ep
        assert derived.eo == REG[face + "2"].eo


# --------------------------------------------------------------------------- #
# Every move keeps the cube physically valid (twist/flip conserved, perm valid).
# --------------------------------------------------------------------------- #


def test_all_moves_preserve_invariants():
    rng = random.Random(1234)
    s = SOLVED
    for _ in range(500):
        mv = REG[rng.choice(REG.action_names)]
        s = apply_move(s, mv)
        s.validate()  # raises if any invariant is broken
    assert sum(s.co) % CORNER_ORI_MOD == 0
    assert sum(s.eo) % EDGE_ORI_MOD == 0


# --------------------------------------------------------------------------- #
# batched == looped : the SoA batch path equals N single applications.
# --------------------------------------------------------------------------- #


def _random_states(n: int, seed: int) -> list[CubeState]:
    rng = random.Random(seed)
    states = []
    for _ in range(n):
        s = SOLVED
        for _ in range(rng.randint(0, 30)):
            s = apply_move(s, REG[rng.choice(REG.action_names)])
        states.append(s)
    return states


@pytest.mark.parametrize("name", REG.action_names)
def test_batched_matches_looped(name):
    mv = REG[name]
    # Deterministic seed derived from the action's fixed position, NOT hash():
    # Python salts str hashes per process (PYTHONHASHSEED), which would make the
    # generated states -- and therefore this test -- differ from run to run.
    seed = 1000 + REG.action_names.index(name)
    states = _random_states(37, seed=seed)
    batched = apply_move_batch(states, mv)
    looped = [apply_move(s, mv) for s in states]
    assert batched == looped


def test_batched_empty_is_empty():
    assert apply_move_batch([], REG["R"]) == []


# --------------------------------------------------------------------------- #
# depth-d solvable : a scramble's reverse solves it in <= d moves.
# --------------------------------------------------------------------------- #


def _inverse_name(name: str) -> str:
    base = name[0]
    if name.endswith("2"):
        return name
    if name.endswith("'"):
        return base
    return base + "'"


@pytest.mark.parametrize("depth", [1, 5, 10, 25, 50])
def test_scramble_reverse_solves(depth):
    """Scramble from solved, then apply the reversed inverse sequence.

    This is the heart of the ADI data generator (Phase 3): because we build the
    scramble ourselves, its solution is free -- the reverse of the scramble. If
    inverses and composition are correct, the reversed-inverse sequence must
    return any depth-d scramble to solved in exactly d moves.
    """
    rng = random.Random(depth * 99 + 1)
    scramble = [rng.choice(REG.action_names) for _ in range(depth)]

    scrambled = apply_sequence(SOLVED, [REG[n] for n in scramble])
    if depth > 0:
        assert not scrambled.is_solved()

    solution = [_inverse_name(n) for n in reversed(scramble)]
    solved_again = apply_sequence(scrambled, [REG[n] for n in solution])
    assert solved_again.is_solved()
    assert len(solution) == depth


# --------------------------------------------------------------------------- #
# Group facts that catch deeper table errors.
# --------------------------------------------------------------------------- #


def test_superflip_style_round_trip_many_random_sequences():
    """A random sequence followed by its exact inverse is always the identity."""
    rng = random.Random(2026)
    for _ in range(200):
        seq = [rng.choice(REG.action_names) for _ in range(rng.randint(1, 40))]
        s = apply_sequence(SOLVED, [REG[n] for n in seq])
        inv = [_inverse_name(n) for n in reversed(seq)]
        assert apply_sequence(s, [REG[n] for n in inv]).is_solved()
