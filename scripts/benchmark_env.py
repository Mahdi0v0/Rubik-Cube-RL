"""Benchmark the vectorized env throughput (Phase-2 gate evidence).

Measures moves/second as the batch size N grows. The whole point of the tensor
env is that one move is a single gather regardless of N, so per-cube cost should
fall sharply with N (amortized launch overhead) and then flatten -- i.e. the env
*scales*. This script reports both aggregate moves/sec and per-cube cost so you
can see that flattening directly.

Usage:
    python scripts/benchmark_env.py                 # default sweep, cpu
    python scripts/benchmark_env.py --n 4096        # single size
    python scripts/benchmark_env.py --device cuda    # if you have a GPU
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Allow running as a plain script (python scripts/benchmark_env.py) without install.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from env.vec_env import VecCubeEnv  # noqa: E402


def benchmark_one(n: int, steps: int, device: str, seed: int = 0) -> dict:
    """Time `steps` random batched moves over N cubes; return timing stats."""
    env = VecCubeEnv(n, device=device, max_steps=steps + 1, seed=seed)
    env.reset(depths=20)

    gen = torch.Generator(device=device).manual_seed(seed)
    # Pre-sample all action vectors so we time the env, not the RNG.
    action_batches = [
        torch.randint(0, 18, (n,), generator=gen, device=device) for _ in range(steps)
    ]

    # Warm up (kernel compilation / allocator), not counted.
    for a in action_batches[: min(5, steps)]:
        env.step(a)

    _sync(device)
    t0 = time.perf_counter()
    for a in action_batches:
        env.step(a)
    _sync(device)
    elapsed = time.perf_counter() - t0

    total_moves = n * steps
    return {
        "n": n,
        "steps": steps,
        "seconds": elapsed,
        "moves_per_sec": total_moves / elapsed,
        "ns_per_cube_move": elapsed / total_moves * 1e9,
    }


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=None, help="single batch size (overrides sweep)")
    parser.add_argument("--steps", type=int, default=200, help="moves to time per size")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to cpu.")
        args.device = "cpu"

    sizes = [args.n] if args.n is not None else [1, 256, 1024, 4096, 16384]

    print(f"device={args.device}  steps/size={args.steps}  torch={torch.__version__}")
    print(f"{'N':>8} {'moves/sec':>16} {'ns/cube-move':>14} {'wall (s)':>10}")
    print("-" * 52)
    for n in sizes:
        try:
            r = benchmark_one(n, args.steps, args.device)
        except RuntimeError as e:  # typically OOM at the largest sizes
            print(f"{n:>8}  skipped: {e}")
            continue
        print(f"{r['n']:>8} {r['moves_per_sec']:>16,.0f} {r['ns_per_cube_move']:>14.1f} {r['seconds']:>10.3f}")


if __name__ == "__main__":
    main()
