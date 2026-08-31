from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from switchback_lab.inference import (
    anticipation_pirt_greater_fast,
    carryover_crt,
    draw_switchback_assignment,
    global_total_effect_crt,
    synthetic_outcomes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quick size/power calibration on synthetic outcomes."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/synthetic"))
    parser.add_argument("--replications", type=int, default=30)
    parser.add_argument("--resamples", type=int, default=500)
    parser.add_argument("--n-events", type=int, default=20000)
    parser.add_argument("--block-length", type=int, default=200)
    parser.add_argument("--memory", type=int, default=100)
    parser.add_argument("--carryover-tested-memory", type=int, default=50)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | int | str]] = []

    # Under H0^m, all coefficients beyond tested m are exactly zero.
    null_carry = tuple([0.01] * args.carryover_tested_memory)
    # Under the alternative, only lags beyond tested m receive a positive tail
    # signal. This keeps the size/power interpretation auditable.
    alternative_carry = tuple(
        [0.0] * args.carryover_tested_memory
        + [0.05] * (2 * args.carryover_tested_memory)
    )

    for replication in range(args.replications):
        design_seed = args.seed + 100 * replication
        outcome_seed = design_seed + 1
        test_seed = design_seed + 2
        w = draw_switchback_assignment(
            args.n_events, args.block_length, seed=design_seed
        )

        scenarios = {
            "total_null": synthetic_outcomes(w, direct_effect=0.0, seed=outcome_seed),
            "total_alternative": synthetic_outcomes(
                w, direct_effect=0.6, seed=outcome_seed
            ),
            "anticipation_null": synthetic_outcomes(
                w, direct_effect=0.4, anticipation_effect=0.0, seed=outcome_seed
            ),
            "anticipation_alternative": synthetic_outcomes(
                w, direct_effect=0.4, anticipation_effect=4.0, seed=outcome_seed
            ),
            "carryover_null": synthetic_outcomes(
                w,
                direct_effect=0.0,
                carryover_coefficients=null_carry,
                seed=outcome_seed,
            ),
            "carryover_alternative": synthetic_outcomes(
                w,
                direct_effect=0.0,
                carryover_coefficients=alternative_carry,
                seed=outcome_seed,
            ),
        }

        for scenario_name in ("total_null", "total_alternative"):
            result = global_total_effect_crt(
                scenarios[scenario_name],
                w,
                block_length=args.block_length,
                memory=args.memory,
                n_resamples=args.resamples,
                seed=test_seed,
            )
            rows.append(
                {
                    "replication": replication,
                    "test": "total_effect_crt",
                    "scenario": scenario_name,
                    "p_value": result.p_value,
                }
            )

        for scenario_name in ("carryover_null", "carryover_alternative"):
            result = carryover_crt(
                scenarios[scenario_name],
                w,
                block_length=args.block_length,
                tested_memory=args.carryover_tested_memory,
                n_resamples=args.resamples,
                alternative="greater",
                seed=test_seed + 1,
            )
            rows.append(
                {
                    "replication": replication,
                    "test": "carryover_crt",
                    "scenario": scenario_name,
                    "p_value": result.p_value,
                }
            )

        for scenario_name in ("anticipation_null", "anticipation_alternative"):
            result = anticipation_pirt_greater_fast(
                scenarios[scenario_name],
                w,
                block_length=args.block_length,
                fixed_prefix_events=args.block_length,
                n_resamples=args.resamples,
                seed=test_seed + 2,
            )
            rows.append(
                {
                    "replication": replication,
                    "test": "anticipation_pirt",
                    "scenario": scenario_name,
                    "p_value": result.p_value,
                }
            )

        print(
            f"Completed replication {replication + 1}/{args.replications}", flush=True
        )

    pvalues = pd.DataFrame(rows)
    summary = (
        pvalues.assign(reject=lambda frame: frame["p_value"] <= args.alpha)
        .groupby(["test", "scenario"], as_index=False)
        .agg(
            replications=("reject", "size"),
            rejection_rate=("reject", "mean"),
            mean_p_value=("p_value", "mean"),
        )
    )
    pvalues.to_csv(args.output_dir / "synthetic_pvalues.csv", index=False)
    summary.to_csv(args.output_dir / "synthetic_summary.csv", index=False)
    print("\nSynthetic calibration summary")
    print(summary.to_string(index=False))
    print(f"\nSaved outputs to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
