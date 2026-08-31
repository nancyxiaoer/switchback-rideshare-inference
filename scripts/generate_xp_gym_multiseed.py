"""Generate resumable multi-seed XP-Gym switchback trajectories.

Each trajectory is validated and saved immediately as a compressed NPZ file.
The manifest is rewritten after every successful seed, so an interrupted run
can be resumed without regenerating completed trajectories.
"""

from __future__ import annotations

import argparse
import json
from math import ceil
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

try:
    import jax
    import jaxlib
    from xp_gym.designs.design import SwitchbackDesign
    from xp_gym.environments.rideshare import XPRidesharePricingEnv
except ImportError as exc:  # pragma: no cover - optional simulator dependency
    raise SystemExit(
        "JAX/XP-Gym is unavailable. Run this script in the Poetry environment "
        "that successfully passed check_xp_gym_rollout.py."
    ) from exc

from switchback_lab.inference import (
    pure_history_ht_estimate,
    pure_history_linear,
    validate_switchback,
)
from switchback_lab.xp_gym_rollout import build_compiled_rollout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate validated multi-seed XP-Gym trajectories."
    )
    parser.add_argument("--n-steps", type=int, default=500_000)
    parser.add_argument(
        "--block-lengths", type=int, nargs="+", default=[1000, 5000, 10000]
    )
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--seed-count", type=int, default=2)
    parser.add_argument("--memory", type=int, default=5000)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/generated")
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate files that already exist. The default is safe resume.",
    )
    return parser.parse_args()


def build_environment(n_steps: int, switch_every: int):
    env = XPRidesharePricingEnv(
        n_cars=300,
        n_events=n_steps,
        price_per_distance_A=0.01,
        price_per_distance_B=0.02,
    )
    inner_params = env.default_params.env_params.replace(
        w_price=-0.3,
        w_eta=-0.005,
        w_intercept=4,
    )
    env_params = env.default_params.replace(env_params=inner_params)
    design = SwitchbackDesign(
        p=0.5,
        switch_every=switch_every,
        time_attr="time",
    )
    return env, env_params, design


def wait_and_copy(tree: Any) -> Any:
    ready = jax.tree_util.tree_map(lambda x: x.block_until_ready(), tree)
    return jax.device_get(ready)


def validate_and_summarize(
    trajectory: dict[str, np.ndarray],
    *,
    block_length: int,
    seed: int,
    memory: int,
    generation_seconds: float,
    includes_compilation: bool,
    output_path: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, int | float | bool | str]]:
    action = np.asarray(trajectory["action"]).reshape(-1).astype(np.int8)
    reward = np.asarray(trajectory["reward"]).reshape(-1).astype(np.float32)
    if len(action) != len(reward):
        raise ValueError("action and reward lengths differ")
    if not np.isfinite(reward).all():
        raise ValueError("reward contains NaN or Inf")

    block_labels = validate_switchback(action, block_length)
    pure_one, pure_zero = pure_history_linear(action, memory)
    eligible = np.arange(memory, len(action))
    pure_share = float((pure_one[eligible] | pure_zero[eligible]).mean())

    treated = action == 1
    control = action == 0
    if not treated.any() or not control.any():
        raise ValueError(
            "trajectory has no treated or no control events; use more blocks or another seed"
        )
    treated_mean = float(reward[treated].mean())
    control_mean = float(reward[control].mean())
    naive_effect = treated_mean - control_mean
    pure_ht = pure_history_ht_estimate(
        reward,
        action,
        memory=memory,
        block_length=block_length,
    )

    summary: dict[str, int | float | bool | str] = {
        "block_length": block_length,
        "seed": seed,
        "n_events": len(action),
        "n_blocks": len(block_labels),
        "treated_block_share": float(block_labels.mean()),
        "treated_event_share": float(action.mean()),
        "mean_reward": float(reward.mean()),
        "mean_reward_control": control_mean,
        "mean_reward_treated": treated_mean,
        "naive_effect": naive_effect,
        "pure_history_memory": memory,
        "pure_history_share": pure_share,
        "pure_history_ht": pure_ht,
        "generation_seconds": generation_seconds,
        "includes_compilation": includes_compilation,
        "jax_backend": jax.default_backend(),
        "output_file": str(output_path),
    }
    return action, reward, summary


def upsert_manifest(
    manifest_path: Path,
    completed: dict[tuple[int, int], dict[str, Any]],
) -> None:
    frame = pd.DataFrame(completed.values())
    if not frame.empty:
        frame = frame.sort_values(["block_length", "seed"])
    frame.to_csv(manifest_path, index=False)


def main() -> None:
    args = parse_args()
    if args.n_steps <= 0 or args.seed_count <= 0:
        raise SystemExit("--n-steps and --seed-count must be positive")
    if not 0 <= args.memory < args.n_steps:
        raise SystemExit("--memory must satisfy 0 <= memory < n_steps")
    if any(length <= 0 for length in args.block_lengths):
        raise SystemExit("all block lengths must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.csv"
    completed: dict[tuple[int, int], dict[str, Any]] = {}
    if manifest_path.exists():
        existing = pd.read_csv(manifest_path)
        for row in existing.to_dict(orient="records"):
            completed[(int(row["block_length"]), int(row["seed"]))] = row

    seeds = list(range(args.seed_start, args.seed_start + args.seed_count))
    run_config = {
        "n_steps": args.n_steps,
        "block_lengths": args.block_lengths,
        "seed_start": args.seed_start,
        "seed_count": args.seed_count,
        "seeds": seeds,
        "memory": args.memory,
        "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__,
        "jax_backend": jax.default_backend(),
        "storage_format": "compressed npz containing action and reward",
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )

    generated_now = 0
    skipped = 0
    for block_length in args.block_lengths:
        block_dir = args.output_dir / f"L{block_length}"
        block_dir.mkdir(parents=True, exist_ok=True)
        pending = []
        for seed in seeds:
            path = block_dir / f"trajectory_L{block_length}_seed{seed}.npz"
            if path.exists() and not args.overwrite:
                print(f"SKIP existing L={block_length}, seed={seed}: {path}")
                skipped += 1
            else:
                pending.append((seed, path))

        if not pending:
            continue

        print(
            f"\nBuilding XP-Gym environment for L={block_length}; "
            f"{len(pending)} trajectory/trajectories pending.",
            flush=True,
        )
        env, env_params, design = build_environment(args.n_steps, block_length)
        rollout = build_compiled_rollout(env, design, args.n_steps)

        for pending_index, (seed, output_path) in enumerate(pending):
            print(
                f"RUN L={block_length}, seed={seed}, T={args.n_steps}", flush=True
            )
            start = perf_counter()
            trajectory = rollout(jax.random.PRNGKey(seed), env_params)
            trajectory = wait_and_copy(trajectory)
            elapsed = perf_counter() - start
            action, reward, summary = validate_and_summarize(
                trajectory,
                block_length=block_length,
                seed=seed,
                memory=args.memory,
                generation_seconds=elapsed,
                includes_compilation=pending_index == 0,
                output_path=output_path,
            )
            np.savez_compressed(output_path, action=action, reward=reward)
            summary["file_size_megabytes"] = output_path.stat().st_size / (1024**2)
            completed[(block_length, seed)] = summary
            upsert_manifest(manifest_path, completed)
            generated_now += 1
            print(
                f"SAVED {output_path} | {elapsed:.2f}s | "
                f"treated blocks={summary['treated_block_share']:.3f} | "
                f"naive effect={summary['naive_effect']:.4f}",
                flush=True,
            )

    print("\nMulti-seed generation complete")
    print(f"Generated now: {generated_now}")
    print(f"Skipped existing: {skipped}")
    print(f"Expected design cells: {len(args.block_lengths) * len(seeds)}")
    print(f"Manifest: {manifest_path.resolve()}")


if __name__ == "__main__":
    main()
