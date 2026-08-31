"""Run resumable randomization inference over generated XP-Gym trajectories."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from switchback_lab.inference import (
    anticipation_pirt_greater_fast,
    carryover_crt,
    global_total_effect_crt,
    validate_switchback,
)


TRAJECTORY_PATTERN = re.compile(r"trajectory_L(\d+)_seed(\d+)\.npz$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run total-effect, carryover and anticipation tests by seed."
    )
    parser.add_argument("--input-dir", type=Path, default=Path("data/generated"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/multiseed"))
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--total-memory", type=int, default=5000)
    parser.add_argument(
        "--carryover-memories",
        type=int,
        nargs="+",
        default=[0, 1000, 2500, 5000, 10000],
    )
    parser.add_argument("--test-seed", type=int, default=50_000)
    parser.add_argument("--true-ate", type=float, default=1.278396906)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Optional subset for a smoke test, e.g. --seeds 1000.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Discard existing inference rows."
    )
    return parser.parse_args()


def discover_trajectories(input_dir: Path, selected_seeds: list[int] | None):
    found: list[tuple[int, int, Path]] = []
    selected = set(selected_seeds) if selected_seeds is not None else None
    for path in input_dir.rglob("trajectory_L*_seed*.npz"):
        match = TRAJECTORY_PATTERN.match(path.name)
        if not match:
            continue
        block_length, seed = map(int, match.groups())
        if selected is None or seed in selected:
            found.append((block_length, seed, path))
    if not found:
        raise FileNotFoundError(f"No generated trajectories found under {input_dir}")
    return sorted(found)


def derived_seed(base: int, *values: int) -> int:
    sequence = np.random.SeedSequence([base, *(int(value) + 1 for value in values)])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def wilson_interval(rejections: int, n: int) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    z = 1.959963984540054
    rate = rejections / n
    denominator = 1.0 + z**2 / n
    center = (rate + z**2 / (2.0 * n)) / denominator
    half = z * np.sqrt(rate * (1.0 - rate) / n + z**2 / (4.0 * n**2)) / denominator
    return float(center - half), float(center + half)


def test_key(row: dict[str, Any]) -> tuple[int, int, str, int]:
    return (
        int(row["block_length"]),
        int(row["seed"]),
        str(row["test"]),
        int(row["memory_events"]),
    )


def save_results(rows: dict[tuple[int, int, str, int], dict[str, Any]], path: Path):
    frame = pd.DataFrame(rows.values())
    if not frame.empty:
        frame = frame.sort_values(
            ["block_length", "seed", "test", "memory_events"]
        )
    frame.to_csv(path, index=False)


def result_row(
    *,
    block_length: int,
    seed: int,
    test: str,
    memory_events: int,
    alternative: str,
    alpha: float,
    n_resamples: int,
    randomization_seed: int,
    elapsed_seconds: float,
    result=None,
    error: str = "",
) -> dict[str, Any]:
    ok = result is not None
    p_value = float(result.p_value) if ok else float("nan")
    return {
        "block_length": block_length,
        "seed": seed,
        "test": test,
        "memory_events": memory_events,
        "alternative": alternative,
        "status": "ok" if ok else "unavailable",
        "error": error,
        "statistic": float(result.statistic) if ok else float("nan"),
        "p_value": p_value,
        "reject_at_alpha": bool(p_value <= alpha) if ok else False,
        "alpha": alpha,
        "n_resamples": n_resamples,
        "n_randomized_units": int(result.n_randomized_units) if ok else 0,
        "n_focal_events": int(result.n_focal_events) if ok else 0,
        "randomization_seed": randomization_seed,
        "elapsed_seconds": elapsed_seconds,
    }


def summarize_inference(results: pd.DataFrame) -> pd.DataFrame:
    summaries: list[dict[str, Any]] = []
    for (test, block_length, memory_events), group in results.groupby(
        ["test", "block_length", "memory_events"], dropna=False
    ):
        valid = group[group["status"] == "ok"]
        rejections = int(valid["reject_at_alpha"].sum())
        n_valid = len(valid)
        lower, upper = wilson_interval(rejections, n_valid)
        summaries.append(
            {
                "test": test,
                "block_length": int(block_length),
                "memory_events": int(memory_events),
                "trajectories_total": len(group),
                "valid_tests": n_valid,
                "unavailable_tests": len(group) - n_valid,
                "rejections": rejections,
                "rejection_rate": rejections / n_valid if n_valid else np.nan,
                "rejection_rate_ci_lower": lower,
                "rejection_rate_ci_upper": upper,
                "mean_p_value": valid["p_value"].mean(),
                "median_p_value": valid["p_value"].median(),
                "mean_randomized_units": valid["n_randomized_units"].mean(),
                "mean_focal_events": valid["n_focal_events"].mean(),
            }
        )
    return pd.DataFrame(summaries).sort_values(
        ["test", "block_length", "memory_events"]
    )


def summarize_estimation(manifest: pd.DataFrame, true_ate: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for block_length, group in manifest.groupby("block_length"):
        naive_error = group["naive_effect"] - true_ate
        ht_error = group["pure_history_ht"] - true_ate
        rows.append(
            {
                "block_length": int(block_length),
                "n_trajectories": len(group),
                "true_ate": true_ate,
                "naive_mean": group["naive_effect"].mean(),
                "naive_sd": group["naive_effect"].std(ddof=1),
                "naive_bias": naive_error.mean(),
                "naive_rmse": np.sqrt(np.mean(naive_error**2)),
                "pure_history_ht_mean": group["pure_history_ht"].mean(),
                "pure_history_ht_sd": group["pure_history_ht"].std(ddof=1),
                "pure_history_ht_bias": ht_error.mean(),
                "pure_history_ht_rmse": np.sqrt(np.mean(ht_error**2)),
                "pure_history_share_mean": group["pure_history_share"].mean(),
                "treated_block_share_mean": group["treated_block_share"].mean(),
                "median_generation_seconds": group["generation_seconds"].median(),
            }
        )
    return pd.DataFrame(rows).sort_values("block_length")


def save_estimation_plot(summary: pd.DataFrame, output_dir: Path) -> None:
    n_seeds = int(summary["n_trajectories"].min())
    x = np.arange(len(summary))
    labels = summary["block_length"].astype(str)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
    axes[0].errorbar(
        x - 0.06,
        summary["naive_mean"],
        yerr=summary["naive_sd"],
        marker="o",
        capsize=4,
        label="Naive",
    )
    axes[0].errorbar(
        x + 0.06,
        summary["pure_history_ht_mean"],
        yerr=summary["pure_history_ht_sd"],
        marker="s",
        capsize=4,
        label="Pure-history HT",
    )
    axes[0].axhline(summary["true_ate"].iloc[0], color="black", linestyle=":")
    axes[0].set_xticks(x, labels)
    axes[0].set_xlabel("Switchback interval L")
    axes[0].set_ylabel("Mean estimate ± 1 SD")
    axes[0].set_title(f"Estimator behavior across {n_seeds} seeds")
    axes[0].legend()

    width = 0.36
    axes[1].bar(x - width / 2, summary["naive_rmse"], width, label="Naive")
    axes[1].bar(
        x + width / 2,
        summary["pure_history_ht_rmse"],
        width,
        label="Pure-history HT",
    )
    axes[1].set_xticks(x, labels)
    axes[1].set_xlabel("Switchback interval L")
    axes[1].set_ylabel("RMSE")
    axes[1].set_title("Bias–variance cost of exposure filtering")
    axes[1].legend()
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "estimation_performance.png", dpi=180)
    plt.close(fig)


def save_inference_plot(summary: pd.DataFrame, output_dir: Path, alpha: float) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.2))
    block_lengths = sorted(summary["block_length"].unique())

    for axis, test, title in (
        (axes[0], "total_effect_crt", "Total-effect CRT"),
        (axes[1], "anticipation_pirt", "Anticipation PIRT"),
    ):
        part = summary[summary["test"] == test].set_index("block_length")
        rates = np.array([part.loc[length, "rejection_rate"] for length in block_lengths])
        lower = np.array(
            [part.loc[length, "rejection_rate_ci_lower"] for length in block_lengths]
        )
        upper = np.array(
            [part.loc[length, "rejection_rate_ci_upper"] for length in block_lengths]
        )
        x = np.arange(len(block_lengths))
        axis.errorbar(
            x,
            rates,
            yerr=np.vstack([rates - lower, upper - rates]),
            marker="o",
            capsize=4,
        )
        axis.set_xticks(x, [str(length) for length in block_lengths])
        axis.set_xlabel("Switchback interval L")
        axis.set_ylabel("Rejection rate")
        axis.set_ylim(0, 1.05)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    axes[1].axhline(alpha, color="black", linestyle=":", label="alpha")
    axes[1].legend()

    carry = summary[summary["test"] == "carryover_crt"]
    for block_length, group in carry.groupby("block_length"):
        group = group.sort_values("memory_events")
        axes[2].plot(
            group["memory_events"],
            group["rejection_rate"],
            marker="o",
            label=f"L={block_length}",
        )
    axes[2].axhline(alpha, color="black", linestyle=":")
    axes[2].set_xlabel("Tested carryover horizon m (events)")
    axes[2].set_ylabel("Rejection rate")
    axes[2].set_ylim(0, 1.05)
    axes[2].set_title("Carryover rejection profile")
    axes[2].grid(axis="y", alpha=0.25)
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "randomization_inference.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.resamples <= 0:
        raise SystemExit("--resamples must be positive")
    if not 0 < args.alpha < 1:
        raise SystemExit("--alpha must be between 0 and 1")
    if any(memory < 0 for memory in args.carryover_memories):
        raise SystemExit("carryover memories must be nonnegative event counts")

    trajectories = discover_trajectories(args.input_dir, args.seeds)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "inference_results.csv"
    completed: dict[tuple[int, int, str, int], dict[str, Any]] = {}
    if results_path.exists() and not args.overwrite:
        existing = pd.read_csv(results_path)
        existing_resamples = set(existing["n_resamples"].astype(int))
        if existing_resamples != {args.resamples}:
            raise SystemExit(
                "Existing results use a different --resamples value. Use a new "
                "output directory or add --overwrite."
            )
        for row in existing.to_dict(orient="records"):
            completed[test_key(row)] = row

    for trajectory_index, (block_length, seed, path) in enumerate(trajectories, 1):
        data = np.load(path)
        w = np.asarray(data["action"], dtype=np.int8)
        y = np.asarray(data["reward"], dtype=float)
        validate_switchback(w, block_length)
        print(
            f"[{trajectory_index}/{len(trajectories)}] L={block_length}, seed={seed}",
            flush=True,
        )

        specifications = [
            ("total_effect_crt", args.total_memory, "greater", 1),
            ("anticipation_pirt", -1, "greater", 2),
            *[
                ("carryover_crt", memory, "two-sided", 10 + index)
                for index, memory in enumerate(args.carryover_memories)
            ],
        ]
        for test, memory, alternative, test_code in specifications:
            key = (block_length, seed, test, memory)
            if key in completed and not args.overwrite:
                print(f"  SKIP {test}, memory={memory}")
                continue
            randomization_seed = derived_seed(
                args.test_seed, block_length, seed, test_code, max(memory, 0)
            )
            start = perf_counter()
            result = None
            error = ""
            try:
                if test == "total_effect_crt":
                    result = global_total_effect_crt(
                        y,
                        w,
                        block_length=block_length,
                        memory=args.total_memory,
                        n_resamples=args.resamples,
                        alternative="greater",
                        seed=randomization_seed,
                    )
                elif test == "anticipation_pirt":
                    result = anticipation_pirt_greater_fast(
                        y,
                        w,
                        block_length=block_length,
                        fixed_prefix_events=block_length,
                        n_resamples=args.resamples,
                        seed=randomization_seed,
                    )
                else:
                    result = carryover_crt(
                        y,
                        w,
                        block_length=block_length,
                        tested_memory=memory,
                        n_resamples=args.resamples,
                        alternative="two-sided",
                        seed=randomization_seed,
                    )
            except ValueError as exc:
                error = str(exc)
            elapsed = perf_counter() - start
            row = result_row(
                block_length=block_length,
                seed=seed,
                test=test,
                memory_events=memory,
                alternative=alternative,
                alpha=args.alpha,
                n_resamples=args.resamples,
                randomization_seed=randomization_seed,
                elapsed_seconds=elapsed,
                result=result,
                error=error,
            )
            completed[key] = row
            status = row["status"]
            p_text = f"{row['p_value']:.4g}" if status == "ok" else error
            print(f"  {test}, memory={memory}: {status}, p={p_text}", flush=True)

        save_results(completed, results_path)

    results = pd.DataFrame(completed.values())
    inference_summary = summarize_inference(results)
    inference_summary.to_csv(args.output_dir / "inference_summary.csv", index=False)

    manifest_path = args.input_dir / "manifest.csv"
    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
        if args.seeds is not None:
            manifest = manifest[manifest["seed"].isin(args.seeds)]
        estimation_summary = summarize_estimation(manifest, args.true_ate)
        estimation_summary.to_csv(
            args.output_dir / "estimation_summary.csv", index=False
        )
        save_estimation_plot(estimation_summary, args.output_dir)
    save_inference_plot(inference_summary, args.output_dir, args.alpha)

    print("\nInference summary")
    print(inference_summary.to_string(index=False))
    print(f"\nSaved outputs to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
