"""Generate scrambles by walking BACKWARD from solved (the ADI data engine).

WHY backward, and why this matters for the whole project:
    Rewarding an agent only when it reaches a solved cube never works -- a deep
    random scramble is essentially never solved by chance, so the signal is
    always zero. The DeepCubeA / Autodidactic-Iteration fix is to manufacture
    training data whose distance-to-solved is *known by construction*: start from
    the solved state and apply k random moves. The result is a state we know is
    reachable in k moves, and -- for free -- the reverse of those k moves is a
    valid solution (a no-solver-needed expert trajectory). The network later
    regresses this k as cost-to-go.

    NOTE on "depth": k is the number of generating moves, i.e. an UPPER BOUND on
    the true (god) distance, not necessarily the optimal solution length. That is
    exactly what ADI uses; deeper scrambles are statistically farther, which is
    enough to learn a useful value heuristic.

Pure core: standard library only. No torch, no web.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import dataclass

from core.cube import SOLVED, CubeState
from core.moves import MoveRegistry, apply_sequence, inverse_name, load_registry

# One shared registry; the move tables are read-only.
_REGISTRY: MoveRegistry = load_registry()


def random_scramble_moves(
    depth: int,
    rng: random.Random,
    *,
    registry: MoveRegistry = _REGISTRY,
    avoid_redundant: bool = True,
) -> list[str]:
    """Pick ``depth`` random move names for a scramble.

    Hygiene (``avoid_redundant``): never choose a move on the SAME face as the
    immediately preceding move. That single rule removes the two ways a scramble
    would otherwise lie about its own depth:
      * cancellation -- ``R`` then ``R'`` undoes itself (net 0 moves),
      * merging / 3-in-a-row -- ``R R`` collapses to ``R2`` and ``R R R`` to
        ``R'`` (fewer net moves than the count).
    With no two consecutive moves sharing a face, a depth-k scramble is a genuine
    k-move walk. (We don't also dedupe the commuting-opposite-face case like
    ``R L R``; that's a smaller effect and a fine future refinement.)
    """
    if depth < 0:
        raise ValueError(f"scramble depth must be >= 0, got {depth}")

    names = registry.action_names
    moves: list[str] = []
    prev_face: str | None = None
    for _ in range(depth):
        if avoid_redundant and prev_face is not None:
            choices = [n for n in names if n[0] != prev_face]
        else:
            choices = names
        pick = rng.choice(choices)
        moves.append(pick)
        prev_face = pick[0]
    return moves


def scramble(depth: int, rng: random.Random, *, registry: MoveRegistry = _REGISTRY) -> CubeState:
    """Return a state that is ``depth`` random moves away from solved."""
    move_names = random_scramble_moves(depth, rng, registry=registry)
    return apply_sequence(SOLVED, [registry[n] for n in move_names])


@dataclass(frozen=True)
class ScrambleResult:
    """A scramble together with its free, solver-free solution.

    ``solution`` is the reverse sequence of inverse moves; applying it to
    ``state`` returns to solved in exactly ``depth`` moves. This is the optional
    behavior-cloning trajectory referenced in the ADI contract -- no external
    solver ever enters the gradient path.
    """

    state: CubeState
    scramble: tuple[str, ...]
    solution: tuple[str, ...]
    depth: int


def scramble_with_solution(
    depth: int, rng: random.Random, *, registry: MoveRegistry = _REGISTRY
) -> ScrambleResult:
    """Scramble from solved and also return the reversed-inverse solution."""
    move_names = random_scramble_moves(depth, rng, registry=registry)
    state = apply_sequence(SOLVED, [registry[n] for n in move_names])
    solution = [inverse_name(n) for n in reversed(move_names)]
    return ScrambleResult(
        state=state,
        scramble=tuple(move_names),
        solution=tuple(solution),
        depth=depth,
    )


def depth_labeled_states(
    num_samples: int,
    rng: random.Random,
    *,
    min_depth: int = 1,
    max_depth: int,
    registry: MoveRegistry = _REGISTRY,
) -> Iterator[tuple[CubeState, int]]:
    """Yield ``num_samples`` ``(state, depth)`` pairs for ADI training.

    Depth is sampled uniformly in ``[min_depth, max_depth]`` per sample. A
    curriculum (Phase 3+) will instead bias this distribution toward shallow
    scrambles early in training; that scheduling lives in ``rl/curriculum.py``,
    not here -- this generator stays a dumb, reproducible source given its rng.
    """
    if min_depth < 0 or max_depth < min_depth:
        raise ValueError(f"need 0 <= min_depth <= max_depth, got {min_depth}, {max_depth}")
    for _ in range(num_samples):
        depth = rng.randint(min_depth, max_depth)
        yield scramble(depth, rng, registry=registry), depth
