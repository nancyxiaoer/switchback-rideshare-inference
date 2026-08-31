from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from switchback_lab.inference import (
    draw_switchback_assignment,
    pure_history_linear,
    pure_history_naive,
)


def elapsed(function, *args) -> tuple[float, tuple[np.ndarray, np.ndarray]]:
    start = time.perf_counter()
    result = function(*args)
    return time.perf_counter() - start, result


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark O(Tm) versus O(T).")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/benchmarks/pure_history_runtime.csv"),
    )
    args = parser.parse_args()
    rows: list[dict[str, float | int]] = []

    # Keep the naive benchmark moderate; running it at T=500,000 and m=5,000
    # would deliberately repeat billions of comparisons.
    for n_events, memory in ((5000, 100), (10000, 250), (20000, 500)):
        w = draw_switchback_assignment(n_events, block_length=50, seed=n_events)
        naive_seconds, naive = elapsed(pure_history_naive, w, memory)
        linear_seconds, linear = elapsed(pure_history_linear, w, memory)
        if not all(np.array_equal(a, b) for a, b in zip(naive, linear)):
            raise AssertionError("naive and linear implementations disagree")
        rows.append(
            {
                "n_events": n_events,
                "memory": memory,
                "naive_seconds": naive_seconds,
                "linear_seconds": linear_seconds,
                "speedup": naive_seconds / max(linear_seconds, 1e-12),
            }
        )

    full_w = draw_switchback_assignment(500000, block_length=1000, seed=42)
    full_linear_seconds, _ = elapsed(pure_history_linear, full_w, 5000)
    rows.append(
        {
            "n_events": 500000,
            "memory": 5000,
            "naive_seconds": np.nan,
            "linear_seconds": full_linear_seconds,
            "speedup": np.nan,
        }
    )

    frame = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(frame.to_string(index=False))
    print(f"\nSaved benchmark to {args.output.resolve()}")


if __name__ == "__main__":
    main()

