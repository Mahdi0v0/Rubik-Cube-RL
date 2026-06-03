"""Acceptance tests for the backward scramble generator (core/scramble.py)."""

from __future__ import annotations

import random

import pytest

from core.cube import SOLVED
from core.moves import apply_sequence, load_registry
from core.scramble import (
    depth_labeled_states,
    random_scramble_moves,
    scramble,
    scramble_with_solution,
)

REG = load_registry()


def test_depth_zero_is_solved():
    assert scramble(0, random.Random(0)).is_solved()
    assert random_scramble_moves(0, random.Random(0)) == []


@pytest.mark.parametrize("depth", [1, 2, 5, 10, 25, 50, 100])
def test_solution_solves_in_exactly_depth(depth):
    # The free solution (reversed inverse) must return the scramble to solved,
    # in exactly `depth` moves -- the property the whole ADI data engine rests on.
    res = scramble_with_solution(depth, random.Random(depth * 7 + 3))
    assert len(res.scramble) == depth
    assert len(res.solution) == depth
    if depth > 0:
        assert not res.state.is_solved()
    solved = apply_sequence(res.state, [REG[n] for n in res.solution])
    assert solved.is_solved()


def test_same_seed_reproduces():
    a = scramble_with_solution(40, random.Random(12345))
    b = scramble_with_solution(40, random.Random(12345))
    assert a == b  # frozen dataclass equality: state, scramble, solution, depth
    # ...and different seeds (almost surely) differ.
    c = scramble_with_solution(40, random.Random(54321))
    assert a.scramble != c.scramble


@pytest.mark.parametrize("depth", [2, 3, 20, 100])
def test_no_two_consecutive_moves_share_a_face(depth):
    moves = random_scramble_moves(depth, random.Random(depth))
    faces = [m[0] for m in moves]
    for a, b in zip(faces, faces[1:]):
        assert a != b, f"consecutive moves share face {a}: {moves}"


def test_scramble_states_are_valid_cubes():
    rng = random.Random(99)
    for _ in range(200):
        s = scramble(rng.randint(0, 60), rng)
        s.validate()  # raises on broken invariants


def test_depth_labeled_states_shape_and_reproducibility():
    gen = list(depth_labeled_states(500, random.Random(7), min_depth=1, max_depth=30))
    assert len(gen) == 500
    for state, depth in gen:
        assert 1 <= depth <= 30
        state.validate()
    # Reproducible given the same seed.
    again = list(depth_labeled_states(500, random.Random(7), min_depth=1, max_depth=30))
    assert [(s, d) for s, d in gen] == [(s, d) for s, d in again]


def test_depth_labeled_states_rejects_bad_range():
    with pytest.raises(ValueError):
        list(depth_labeled_states(1, random.Random(0), min_depth=5, max_depth=3))
