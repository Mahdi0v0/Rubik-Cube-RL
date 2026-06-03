<div align="center">

# 🧩 rubiks-rl

**An AI that learns to solve a 3×3 Rubik's Cube on its own — and a browser playground to watch it.**

No hand-coded solving algorithms. No black boxes. Built from scratch, heavily
commented, minimal dependencies — so it doubles as a teaching reference.

`Status: 🚧 in development — Phase 1 (cube core)` · `Learner: ADI / DeepCubeA` · `Hard deps: PyTorch only`

</div>

---

## What it is

`rubiks-rl` trains an agent to solve a scrambled Rubik's Cube through
reinforcement learning — it discovers a solving policy rather than executing
human methods like CFOP. The cube, its moves, and the environment are written
from scratch for total control, the learner is built on PyTorch, and the
playground is plain HTML/CSS/JS that anyone can open in a browser to watch a
solve animate.

The project is deliberately readable. The original attempt stalled on opaque
reference code, so here **clear documentation and comments are first-class
deliverables**, equal in importance to the model itself.

## Why it's interesting

Rewarding an agent only when it reaches a solved cube doesn't work: the state
space is ~4.3 × 10¹⁹ and a deep random scramble is essentially never solved by
chance, so the learning signal is always zero. Measuring "how close to solved"
*is the entire problem*.

The fix (the [DeepCubeA](#acknowledgments) approach, **Autodidactic Iteration**):
generate training data by scrambling **backward from the solved state**, so the
distance to solved is known *by construction*. A network learns to estimate that
distance, and a search uses the estimate as a heuristic to actually solve cubes.

## Features

- 🧠 **Self-taught solving** — learns a policy via RL, not scripted algorithms.
- ⚙️ **From-scratch, controllable environment** — cubie-level state, moves as
  permutations, O(1) solved-check.
- 🚀 **Massively parallel** — the env holds *N* cubes in one tensor; a move is a
  single batched op, so you can run 1 → thousands of agents on one device.
- ⏸️ **Pause / save / resume** — checkpoints capture the *whole* run (weights,
  optimizer, replay buffer, RNG, curriculum) for lossless, reproducible resumes.
- 🎞️ **Decoupled speed** — train flat-out; replay episodes at any speed, scrub,
  pause, step.
- 🎲 **A real cube in the browser** — colored stickers, free orbit to any angle,
  animated turns, manual scrambling. CSS 3D, no framework, no backend.

## How it works

Three strictly decoupled subsystems, with dependencies pointing **inward** toward
a pure core:

```
┌───────────────────────────┐   obs + step(action)   ┌──────────────────────┐   episode.json   ┌────────────────────────┐
│  CORE + ENV  (pure python) │ ─────────────────────▶ │  RL STACK  (torch)   │ ───────────────▶ │  WEB  (browser, no py) │
│  state · moves · scramble  │                        │  net · ADI · search  │                  │  CSS-3D cube · player  │
│  vectorized environment    │                        │  checkpoints         │                  │  (static, zero backend)│
└───────────────────────────┘                        └──────────────────────┘                  └────────────────────────┘
```

- **Core never imports torch or the web.** It's testable with zero ML deps.
- **PyTorch lives only in the RL stack.**
- **The browser consumes serialized JSON**, never Python logic — which is why the
  playground ships with no server.

The full design — every contract, data-flow, workflow, and milestone, with
diagrams — is in **[`docs/DRD.md`](docs/DRD.md)**.

## Project structure

```
rubiks-rl/
├── config/         # JSON config — hyperparams, curriculum, move registry
├── core/           # PURE python — cube state, moves, scramble, render  (no torch)
├── env/            # vectorized environment — N cubes as one tensor
├── rl/             # TORCH only — networks, ADI loop, search, replay buffer
├── training/       # entrypoint, checkpointing, run management
├── server/         # (Phase 6) FastAPI + WebSocket live dashboard
├── web/            # FROM SCRATCH — CSS-3D cube, episode player  (no framework)
├── tests/          # move-algebra proofs, env equivalence checks
├── scripts/        # record episodes, benchmark the env
└── docs/           # DRD.md (full design), architecture.md, move_notation.md
```

## Getting started

> **Status note:** the project is early (Phase 1). The cube core is being built
> first; training and the playground arrive in later phases (see [Roadmap](#roadmap)).
> The commands below describe intended usage as each phase lands.

### Run the tests (available now / Phase 1)

```bash
git clone https://github.com/<you>/rubiks-rl.git
cd rubiks-rl
pip install -e ".[dev]"
pytest tests/ -v          # proves the cube logic: R⁴=I, (R U R' U')⁶=I, etc.
```

The `core/` package has **no ML dependencies** — you can read and test the cube
logic without installing PyTorch.

### Train an agent (Phase 3+)

```bash
pip install -e .                                   # installs PyTorch
python -m training.train --config config/default.json
python -m training.train --resume <run_id>         # pause anytime, resume losslessly
```

### Watch a solve (Phase 5+)

```bash
# No install, no server — just open the file:
open web/index.html
```

Pick a recorded episode and watch the agent solve, step through frames, change
speed, rotate the cube, or scramble it yourself.

## Roadmap

| Phase | Deliverable | Exit gate | Status |
|---|---|---|---|
| 0 | Repo skeleton, config schemas, design doc | contracts agreed | ✅ done |
| 1 | `core/cube.py`, `core/moves.py`, move tests | move-algebra tests pass | 🚧 in progress |
| 2 | environment + scramble + render | `vec == single`, env benchmark | ⏳ |
| 3 | ADI core: networks, training loop, search | solves shallow scrambles | ⏳ |
| 4 | training infra: parallel actors, checkpoints | pause/resume reproduces | ⏳ |
| 5 | static playground (CSS-3D + player) | a non-technical user watches a solve | ⏳ |
| 6 | live dashboard (FastAPI + WebSocket) | real-time control + streaming | 🔭 later |
| 7 | Curriculum / HER alternate learners, PBT, Three.js | cross-strategy comparison | 🔭 future |

Each phase is a vertical slice that leaves the repo working and tested; no phase
starts before the previous gate is green.

## Tech stack

| Area | Choice | Why |
|---|---|---|
| Learning | **PyTorch** | autograd + CUDA; the one heavy dep, confined to `rl/` |
| Environment | custom, vectorized | full control; not Gymnasium (adapter is a fallback only) |
| Config | **JSON** (stdlib) | zero-dependency, version-controlled |
| Live server | FastAPI + uvicorn *(Phase 6, optional)* | async WebSocket + static serving |
| Frontend | **none** — CSS 3D transforms | a colored, animated, orbitable cube needs no framework |
| Tests | pytest | — |

The consumer playground requires **zero Python and zero npm** — it's static files.

## Contributing

Contributions welcome. The architecture is built to make two extensions cheap:

- **Add a move** — edit `config/moves.json` (a permutation table); the loader
  derives the prime/double and the tests prove it. No Python changes.
- **Add a learner** — implement the `Trainer` interface (e.g. Curriculum, HER);
  the training entrypoint depends on the interface, not your class.

Fork or branch, keep the [dependency rule](#how-it-works) intact, add tests, and
open a PR. Trained checkpoints are distributed via GitHub Releases (they're
git-ignored, not committed).

## Acknowledgments

This project's method follows the line of work that solved the cube with deep RL:
**Autodidactic Iteration** (McAleer et al., *Solving the Rubik's Cube Without
Human Knowledge*, 2018) and **DeepCubeA** (Agostinelli, McAleer, Shmakov, Baldi,
*Solving the Rubik's Cube with Deep Reinforcement Learning and Search*, Nature
Machine Intelligence, 2019). This is an independent, from-scratch implementation
for learning and experimentation.

## License

Intended to be open source. _(Add a `LICENSE` file — e.g. MIT — before publishing.)_
