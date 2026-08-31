"""Validate and benchmark the optional JAX/XP-Gym trajectory generator.

Run this only in an environment where JAX and XP-Gym are already installed.
The first JAX call includes compilation time; the second call reuses the
compiled executable and is the relevant per-trajectory runtime estimate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from xp_gym.designs.design import SwitchbackDesign
    from xp_gym.environments.rideshare import XPRidesharePricingEnv
    from xp_gym.observation import Observation
except ImportError as exc:  # pragma: no cover - depends on optional simulator
    raise SystemExit(
        "JAX/XP-Gym is not installed in this environment. Activate the "
        "environment that previously ran your XP-Gym run.py, then install this "
        "starter project with: pip install -e <path-to-starter-project>"
    ) from exc

from switchback_lab.xp_gym_rollout import build_compiled_rollout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check reproducibility, block structure and runtime of a JAX rollout."
    )
    parser.add_argument("--n-steps", type=int, required=True)
    parser.add_argument("--switch-every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--compare-python-loop",
        action="store_true",
        help="Also run the slow original loop; use only at 1,000-5,000 steps.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/rollout_checks"))
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
    """Wait for asynchronous JAX work, then copy the result to host NumPy."""
    ready = jax.tree_util.tree_map(lambda x: x.block_until_ready(), tree)
    return jax.device_get(ready)


def timed_rollout(rollout, seed: int, env_params: Any):
    start = perf_counter()
    trajectory = rollout(jax.random.PRNGKey(seed), env_params)
    trajectory = wait_and_copy(trajectory)
    return trajectory, perf_counter() - start


def original_python_loop(
    env: Any,
    env_params: Any,
    design: Any,
    n_steps: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Small reference run whose RNG split order matches the scan rollout."""
    rng = jax.random.PRNGKey(seed)
    rng, reset_rng = jax.random.split(rng)
    rng, design_rng = jax.random.split(rng)
    obs, state = env.reset(reset_rng, env_params)
    design_state = design.reset(design_rng, env_params)

    records: dict[str, list[Any]] = {
        "action": [],
        "reward": [],
        "cluster_id": [],
        "time": [],
        "done": [],
    }
    for _ in range(n_steps):
        action, design_info = design.assign_treatment(design_state, state)
        rng, step_rng = jax.random.split(rng)
        next_obs, next_state, reward, done, _ = env.step(
            step_rng, state, action, env_params
        )
        xp_observation = Observation(
            obs=next_obs,
            action=action,
            reward=reward,
            design_info=design_info,
        )
        records["action"].append(action)
        records["reward"].append(reward)
        records["cluster_id"].append(design_info.cluster_id)
        records["time"].append(state.time)
        records["done"].append(done)
        design_state = design.update(design_state, xp_observation)
        obs, state = next_obs, next_state

    stacked = {name: jnp.stack(values) for name, values in records.items()}
    return wait_and_copy(stacked)


def validate_trajectory(
    trajectory: dict[str, np.ndarray], n_steps: int, switch_every: int
) -> dict[str, bool | int | float]:
    action = np.asarray(trajectory["action"]).reshape(-1).astype(int)
    reward = np.asarray(trajectory["reward"]).reshape(-1)
    cluster_id = np.asarray(trajectory["cluster_id"]).reshape(-1).astype(int)
    time_index = np.asarray(trajectory["time"]).reshape(-1).astype(int)

    same_cluster = cluster_id[1:] == cluster_id[:-1]
    within_block_violations = int(
        np.sum(same_cluster & (action[1:] != action[:-1]))
    )
    expected_clusters = time_index // switch_every
    device_bytes = int(
        sum(np.asarray(value).nbytes for value in trajectory.values())
    )

    return {
        "row_count_ok": len(action) == n_steps,
        "finite_rewards_ok": bool(np.isfinite(reward).all()),
        "binary_actions_ok": bool(np.isin(action, [0, 1]).all()),
        "time_index_ok": bool(np.array_equal(time_index, np.arange(n_steps))),
        "cluster_formula_ok": bool(np.array_equal(cluster_id, expected_clusters)),
        "within_block_action_violations": within_block_violations,
        "block_constancy_ok": within_block_violations == 0,
        "n_observed_blocks": int(np.unique(cluster_id).size),
        "treated_event_share": float(action.mean()),
        "mean_reward": float(reward.mean()),
        "saved_trajectory_megabytes": device_bytes / (1024**2),
    }


def trajectories_equal(left: dict[str, np.ndarray], right: dict[str, np.ndarray]) -> bool:
    return all(
        np.array_equal(np.asarray(left[key]), np.asarray(right[key]))
        for key in left
    )


def compare_scan_to_python_loop(
    scan: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    *,
    reward_atol: float = 1e-5,
    reward_rtol: float = 1e-6,
) -> tuple[bool, dict[str, bool | int | float | None]]:
    """Compare implementations without requiring bitwise-equal floats.

    Assignment, cluster, time and terminal flags are discrete and therefore
    must match exactly. Rewards are computed through differently compiled XLA
    programs, so harmless last-bit floating-point differences are accepted but
    measured and reported.
    """
    diagnostics: dict[str, bool | int | float | None] = {}
    discrete_ok = True
    for key in ("action", "cluster_id", "time", "done"):
        scan_value = np.asarray(scan[key])
        reference_value = np.asarray(reference[key])
        exact = bool(np.array_equal(scan_value, reference_value))
        same_shape = scan_value.shape == reference_value.shape
        if same_shape:
            mismatch = np.flatnonzero(
                scan_value.reshape(-1) != reference_value.reshape(-1)
            )
        else:
            mismatch = np.array([], dtype=int)
        diagnostics[f"scan_loop_{key}_exact"] = exact
        diagnostics[f"scan_loop_{key}_same_shape"] = same_shape
        diagnostics[f"scan_loop_{key}_mismatch_count"] = (
            int(mismatch.size) if same_shape else -1
        )
        diagnostics[f"scan_loop_{key}_first_mismatch_index"] = (
            int(mismatch[0]) if mismatch.size else None
        )
        discrete_ok = discrete_ok and exact

    scan_action = np.asarray(scan["action"]).reshape(-1).astype(int)
    reference_action = np.asarray(reference["action"]).reshape(-1).astype(int)
    diagnostics.update(
        {
            "scan_treated_event_share": float(scan_action.mean()),
            "python_loop_treated_event_share": float(reference_action.mean()),
            "scan_first_action": int(scan_action[0]),
            "python_loop_first_action": int(reference_action[0]),
        }
    )

    scan_reward = np.asarray(scan["reward"], dtype=float).reshape(-1)
    reference_reward = np.asarray(reference["reward"], dtype=float).reshape(-1)
    absolute_difference = np.abs(scan_reward - reference_reward)
    nonexact = np.flatnonzero(absolute_difference > 0)
    reward_close = bool(
        np.allclose(
            scan_reward,
            reference_reward,
            atol=reward_atol,
            rtol=reward_rtol,
            equal_nan=False,
        )
    )
    diagnostics.update(
        {
            "scan_loop_reward_close": reward_close,
            "scan_loop_reward_max_abs_diff": float(absolute_difference.max()),
            "scan_loop_reward_mean_abs_diff": float(absolute_difference.mean()),
            "scan_loop_reward_nonexact_count": int(nonexact.size),
            "scan_loop_reward_first_nonexact_index": (
                int(nonexact[0]) if nonexact.size else None
            ),
            "scan_first_reward": float(scan_reward[0]),
            "python_loop_first_reward": float(reference_reward[0]),
            "scan_loop_reward_atol": reward_atol,
            "scan_loop_reward_rtol": reward_rtol,
        }
    )
    return bool(discrete_ok and reward_close), diagnostics


def compare_initialization_under_jit(
    env: Any,
    env_params: Any,
    design: Any,
    seed: int,
) -> dict[str, bool | int | float | None]:
    """Locate whether eager and compiled execution first diverge at reset."""

    def initialize(root_rng, params):
        rng, reset_rng = jax.random.split(root_rng)
        rng, design_rng = jax.random.split(rng)
        obs, state = env.reset(reset_rng, params)
        design_state = design.reset(design_rng, params)
        action, design_info = design.assign_treatment(design_state, state)
        return {
            "reset_rng": reset_rng,
            "design_rng": design_rng,
            "obs": obs,
            "locations": state.locations,
            "times": state.times,
            "state_time": state.time,
            "event_t": state.event.t,
            "event_src": state.event.src,
            "event_dest": state.event.dest,
            "action": action,
            "cluster_id": design_info.cluster_id,
        }

    root_rng = jax.random.PRNGKey(seed)
    eager = wait_and_copy(initialize(root_rng, env_params))
    compiled = wait_and_copy(jax.jit(initialize)(root_rng, env_params))
    result: dict[str, bool | int | float | None] = {}
    all_exact = True
    for key in eager:
        eager_value = np.asarray(eager[key])
        compiled_value = np.asarray(compiled[key])
        exact = bool(np.array_equal(eager_value, compiled_value))
        result[f"initial_{key}_eager_vs_jit_exact"] = exact
        if eager_value.shape == compiled_value.shape:
            mismatch = np.flatnonzero(
                eager_value.reshape(-1) != compiled_value.reshape(-1)
            )
            result[f"initial_{key}_mismatch_count"] = int(mismatch.size)
            result[f"initial_{key}_first_mismatch_index"] = (
                int(mismatch[0]) if mismatch.size else None
            )
        all_exact = all_exact and exact
    result["initialization_eager_vs_jit_all_exact"] = bool(all_exact)
    result["initial_eager_action"] = int(np.asarray(eager["action"]).item())
    result["initial_jit_action"] = int(np.asarray(compiled["action"]).item())
    return result


def main() -> None:
    args = parse_args()
    if args.n_steps <= 0 or args.switch_every <= 0:
        raise SystemExit("--n-steps and --switch-every must both be positive.")
    if args.compare_python_loop and args.n_steps > 5000:
        raise SystemExit(
            "Do not use --compare-python-loop above 5,000 steps; the reference "
            "loop is intentionally slow."
        )

    env, env_params, design = build_environment(args.n_steps, args.switch_every)
    rollout = build_compiled_rollout(env, design, args.n_steps)

    # First call compiles; second call measures cached execution and also checks
    # same-seed reproducibility. A third call checks that a new seed changes data.
    first, compile_and_run_seconds = timed_rollout(rollout, args.seed, env_params)
    repeated, cached_run_seconds = timed_rollout(rollout, args.seed, env_params)
    different, different_seed_seconds = timed_rollout(
        rollout, args.seed + 1, env_params
    )

    checks = validate_trajectory(first, args.n_steps, args.switch_every)
    checks["same_seed_reproducible"] = trajectories_equal(first, repeated)
    checks["different_seed_changes_trajectory"] = not trajectories_equal(
        first, different
    )

    python_loop_seconds: float | None = None
    if args.compare_python_loop:
        initialization_checks = compare_initialization_under_jit(
            env, env_params, design, args.seed
        )
        checks.update(initialization_checks)
        start = perf_counter()
        reference = original_python_loop(
            env, env_params, design, args.n_steps, args.seed
        )
        python_loop_seconds = perf_counter() - start
        scan_matches, comparison = compare_scan_to_python_loop(first, reference)
        checks.update(comparison)
        checks["scan_matches_python_loop"] = scan_matches

    boolean_checks = [value for value in checks.values() if isinstance(value, bool)]
    all_checks_passed = bool(all(boolean_checks))
    summary = {
        "n_steps": args.n_steps,
        "switch_every": args.switch_every,
        "seed": args.seed,
        "next_seed_checked": args.seed + 1,
        "jax_backend": jax.default_backend(),
        "compile_and_first_run_seconds": compile_and_run_seconds,
        "cached_run_seconds": cached_run_seconds,
        "different_seed_run_seconds": different_seed_seconds,
        "python_loop_seconds": python_loop_seconds,
        "all_checks_passed": all_checks_passed,
        "checks": checks,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / (
        f"check_T{args.n_steps}_L{args.switch_every}_seed{args.seed}.json"
    )
    output_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved check report to {output_path.resolve()}")
    if not all_checks_passed:
        raise SystemExit("One or more rollout checks failed; do not scale up yet.")


if __name__ == "__main__":
    main()
