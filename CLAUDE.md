# CLAUDE.md

> Project memory for Claude Code. Read at the start of every session. This is a
> **behavioral contract**, not documentation — every rule here is binding.
> Full design detail lives in [`docs/DRD.md`](docs/DRD.md) (not yet written);
> read it before any non-trivial change once it exists. When this file and a
> prompt conflict, **this file wins** unless the user explicitly overrides it.

---

## 1. What this project is

`rubiks-rl` is an open-source agent that **learns to solve a 3×3 Rubik's Cube on
its own** via reinforcement learning (no hand-coded human algorithms), plus a
framework-free browser playground to watch it solve.

It is built from scratch on purpose. The previous attempt failed because
reference projects were opaque. **Readability and comments are deliverables
equal to the model.** Optimize every file for the next human who reads it.

---

## 2. Current task — START HERE

Phase 1 (cube core) is complete and the move-algebra gate is green.
**You are implementing Phase 2: the environment.**

Build, in this order, and stop at the gate:

1. `core/render_state.py` — project cubie state → 54-sticker color grid (pure stdlib).
2. `core/scramble.py` — backward-from-solved, depth-labeled scramble generation (pure stdlib).
3. `env/cube_env.py` — single-cube env: `reset/step/observation` (pure stdlib).
4. `env/vec_env.py` — N cubes in one tensor; one move = one `gather` (the **only** torch in core+env).

**Phase-2 gate (definition of done):** `vec_env.step` is bit-identical to looping
`core.moves.apply_move` over the batch (all 18 actions, random states/actions);
render + scramble acceptance tests pass; `scripts/benchmark_env.py` shows near-flat
per-cube cost as N scales. Do not start Phase 3 until this is green.

> Phase 1, for reference (done): `core/cube.py` (cubie state + `is_solved()` +
> `serialize()`), `core/moves.py` (load 6 base generators from `config/moves.json`,
> derive primes/doubles, apply to **a single state and a pure-Python batched
> cross-check — NOT a torch tensor; torch is forbidden in `core/`**), and
> `tests/test_moves.py` (the algebraic identity proofs + external element-order
> anchors like |R U|=105). The 6 base-generator tables in `config/moves.json` are
> authored and proven.

---

## 3. The dependency rule — NON-NEGOTIABLE

Three subsystems. Dependencies point **inward toward the pure core only.**

| Layer | Dirs | May import | Must NOT import |
|---|---|---|---|
| **Core + Env** | `core/`, `env/` | Python stdlib (and torch only inside `env/vec_env.py` for tensors) | the web layer, the server |
| **RL stack** | `rl/`, `training/` | torch, core, env | the web layer |
| **Presentation** | `web/`, `server/` | nothing Python in `web/` | core/rl Python logic |

- `core/` is **pure** — no torch, no web, no network. Must be readable and
  testable with zero ML deps installed.
- **PyTorch appears only in `rl/`, `training/`, and `env/vec_env.py`.** Never in
  `core/cube.py`, `core/moves.py`, `core/scramble.py`, `core/render_state.py`.
- The browser never imports Python. It consumes **serialized JSON** (episodes /
  state messages). The RL→Presentation boundary is a file, not a function call.

If a change would make the core depend on torch or the web, **stop and flag it.**

---

## 4. Architecture (one line each)

- `core/cube.py` — cubie state, invariants, solved-check, serialize.
- `core/moves.py` — permutation load/derive; single + batched application.
- `core/scramble.py` — backward-from-solved, depth-labeled scramble generation.
- `core/render_state.py` — project cubie state → 54-sticker color grid (for NN + UI).
- `env/cube_env.py` — single-cube env: `reset/step/observation`.
- `env/vec_env.py` — N cubes in one tensor; one move = one `gather`. The speed path.
- `rl/networks.py` — value + policy net (DeepCubeA-style).
- `rl/adi.py` — Autodidactic Iteration training loop (the v1 learner).
- `rl/search.py` — weighted A* / MCTS over the learned value → solution move list.
- `rl/replay_buffer.py`, `rl/curriculum.py` — experience store, scramble-depth scheduler.
- `training/train.py` — entrypoint: N parallel actors + one shared learner.
- `training/checkpoint.py`, `training/run_manager.py` — full-state save/resume, experiment dirs.
- `web/` — CSS-3D cube renderer + episode player (static, zero backend).
- `server/` — (Phase 6) FastAPI + WebSocket live dashboard.

---

## 5. Contracts you must honor

Full specs + diagrams in `docs/DRD.md` (TBD). Summaries (do not silently change these):

**State (D1) — the cubie model.** Four integer arrays:
```
cp: int[8]   corner permutation   (which corner in each of 8 slots)
co: int[8]   corner orientation   ∈ {0,1,2}, mod 3
ep: int[12]  edge permutation     (which edge in each of 12 slots)
eo: int[12]  edge orientation     ∈ {0,1},   mod 2
```
Invariants: `sum(co) % 3 == 0`, `sum(eo) % 2 == 0`, perm is a valid permutation.
Solved = identity permutation + zero orientation. Solved-check is **O(1)**
(`perm == identity ∧ orient == 0`). Centers are fixed and not modelled.
This 4-array dict is also the JSON wire format.

**Moves (D2) — registry ≠ action space.**
- Six face turns `{R,L,U,D,F,B}` generate the whole group — that's all the agent needs.
- **Agent action space = the 18 face turns** (`R R' R2 … B B'`). Nothing else.
- Registry also holds UI-only moves: slices `M E S`, wide `Rw…`, view rotations
  `x y z`. **View rotations are camera-only and never mutate state.** Slices/wide
  are NOT agent actions (slices move centers and would break the O(1) check).
- Adding a move = a `config/moves.json` edit, not a code change.

**Env (D4).** Custom, not Gymnasium. `step(action) -> (obs, reward, done, info)`.
`obs` = one-hot of the 54-sticker view = 324-dim (or `[N,324]`). `done` when
solved or step-limit hit. Reward shaping is deferred to a research pass and lives
in `config/default.json` — it does not affect the architecture.

**RL (D3) — ADI.** Generate depth-labeled states by scrambling **backward** from
solved (distance known by construction). Train value (regress cost-to-go) +
policy. Solve with weighted A* over the value. The **reverse of a backward
scramble is a free expert trajectory** for optional, low-weight behavior cloning
— no external solver ever sits in the gradient path.

**Episode JSON** (Python→browser): `{schema_version, cube_size, scramble[],
start_state{cp,co,ep,eo}, solution[], frames[{move, state_54[54], value}],
meta{...}}`. Frames carry the 54-sticker view so the browser needs no cube logic.

**Checkpoint** (pause/resume must reproduce the run, not just reload weights):
bundle `{schema_version, model_state, optim_state, replay_buffer, rng{torch,
numpy,python}, curriculum{depth,schedule_pos}, meta{generation, global_step,
config_hash, wall_clock, git_sha}}`. On disk: `checkpoints/<run_id>/gen_NNNN.pt`
+ `latest` pointer + `config.json` + `metrics.csv`.

---

## 6. Coding conventions

- **Comment the *why*, not the *what*.** Explain the math behind permutations and
  the RL rationale behind a loss/target. Assume the reader knows Python; they may
  not know cube group theory or ADI.
- **Minimal, justified dependencies.** Before adding any dependency, justify why
  we can't build it ourselves in a comment / PR note. Defaults: stdlib + torch.
  No NumPy if torch tensors suffice. No frontend framework (CSS 3D only).
- **Config is data.** Hyperparameters, curriculum, reward weights, search weight,
  and the move registry live in `config/*.json`. No magic numbers in code.
- **Type-hint everything.** Public functions get docstrings stating shapes and units.
- **Vertical slices.** Each phase leaves the repo in a working, tested state.
  Don't scaffold ten empty modules; build one path end-to-end and test it.
- **Tests gate phases.** Write the test with (or before) the code. CI runs `pytest`.
- **Seedable RNG everywhere; deterministic tests.** No `hash()`-salted seeds.
- Python ≥ 3.11. Format with `ruff`/`black` defaults.

---

## 7. Commands

```bash
# install (dev / trainer)
pip install -e ".[dev]"          # pytest etc.; core tests need ZERO ML deps
pip install -e ".[train]"        # adds PyTorch (only needed for env/vec_env + rl/)

# run the test suite (the phase gate)
pytest tests/ -v

# run a single test while iterating
pytest tests/test_moves.py -v -k "sexy_move"

# (Phase 2+) benchmark env throughput
python scripts/benchmark_env.py --n 1024

# (Phase 3+) train
python -m training.train --config config/default.json

# (Phase 4+) resume a run
python -m training.train --resume <run_id>

# (Phase 5+) watch a solve — just open the static file, no server
open web/index.html
```

The **consumer playground has zero Python / zero npm** — it is static files.
Never add a build step or framework to `web/`.

---

## 8. Phase-1 test identities (all green)

| Test | Asserts | Catches |
|---|---|---|
| `R⁴ = I` | four quarter-turns of any face return to start | wrong permutation cycles |
| `(R U R' U')⁶ = I` | the "sexy move" has order 6 | composition / orientation bugs |
| `\|R U\| = 105` (and friends) | external element orders | geometrically wrong-but-consistent tables |
| `move · inverse = I` | each derived prime undoes its base | bad inverse derivation |
| `batched == looped` | one batched step == N single steps | vectorization errors |
| depth-d solvable | a depth-d scramble's reverse solves it in ≤ d | scramble/inverse mismatch |

---

## 9. Roadmap (don't skip a gate)

`P0 Foundations` → `P1 Cube core` → ⟨gate: move algebra ✅⟩ → `P2 Env`
→ ⟨gate: vec==single + benchmark⟩ → `P3 ADI core` → ⟨gate: solves shallow
scrambles⟩ → `P4 Training infra` → ⟨gate: pause/resume reproduces⟩ →
`P5 Static playground` → ⟨gate: user watches a solve⟩ → `P6 Live dashboard
(later)` → `P7 Curriculum/HER/PBT (future)`.

---

## 10. Decisions NOT to relitigate

Unless the user explicitly asks to revisit one, treat these as settled — do not
"helpfully" replace them:

- **Cubie model**, not a 54-sticker matrix, for the internal state. (Sticker grid
  is a derived view only.)
- **`core/` stays torch-free.** The batched path in `core/moves.py` is pure
  Python; the real `torch.gather` speed path is `env/vec_env.py` and reuses the
  same move tables. Do not add torch to `core/` to satisfy the word "tensor".
- **ADI / DeepCubeA**, not sparse PPO/DQN. Naive "reward closer to solved" does
  not converge — measuring distance-to-solved *is* the whole problem. If you find
  yourself adding a dense reward by measuring closeness, stop.
- **18-move face-turn action space.** No slices/wide/rotations in the policy.
- **Data-parallel** learning (N actors → one shared net), not population-based as
  the core learner. PBT, if ever, tunes hyperparameters only.
- **Custom env**, not Gymnasium (adapter is a fallback only).
- **CSS 3D** for rendering; Three.js only if photorealism is ever requested.
- **JSON** config; static-only playground.

---

## 11. Pointers

- Full design + diagrams: `docs/DRD.md` (not yet written)
- Move notation / axis conventions / facelet numbering: `docs/move_notation.md`
- Claude Code memory docs: https://code.claude.com/docs/en/memory
