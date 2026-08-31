"""JAX rollout template for replacing the 500,000-step Python loop.

This module is optional: it requires JAX and XP-Gym. Keep it out of the package
imports until the simulator environment is installed and verified.
"""

from __future__ import annotations

from typing import Any, Callable

import jax

from xp_gym.observation import Observation


def build_compiled_rollout(
    env: Any, design: Any, n_steps: int
) -> Callable[[jax.Array, Any], dict[str, jax.Array]]:
    """Create one compiled trajectory generator.

    `env` and `design` are captured by the Python closure. `n_steps` is fixed so
    XLA can compile one `lax.scan` rather than executing `n_steps` Python calls.
    """

    def rollout(rng: jax.Array, env_params: Any) -> dict[str, jax.Array]:
        # Keep the same split order as the original Python loop.  This makes it
        # possible to compare the loop and scan implementations exactly under
        # one seed, rather than merely comparing their distributions.
        rng, reset_rng = jax.random.split(rng)
        rng, design_rng = jax.random.split(rng)
        obs, state = env.reset(reset_rng, env_params)
        design_state = design.reset(design_rng, env_params)

        def one_step(carry: tuple[Any, ...], _: None):
            step_rng_source, current_obs, current_state, current_design_state = carry
            action, design_info = design.assign_treatment(
                current_design_state, current_state
            )
            step_rng_source, step_rng = jax.random.split(step_rng_source)
            next_obs, next_state, reward, done, _ = env.step(
                step_rng, current_state, action, env_params
            )
            xp_observation = Observation(
                obs=next_obs,
                action=action,
                reward=reward,
                design_info=design_info,
            )
            next_design_state = design.update(
                current_design_state, xp_observation
            )
            next_carry = (
                step_rng_source,
                next_obs,
                next_state,
                next_design_state,
            )
            record = {
                "action": action,
                "reward": reward,
                "cluster_id": design_info.cluster_id,
                "time": current_state.time,
                "done": done,
            }
            return next_carry, record

        initial_carry = (rng, obs, state, design_state)
        _, trajectory = jax.lax.scan(
            one_step, initial_carry, xs=None, length=n_steps
        )
        return trajectory

    return jax.jit(rollout)


def build_batched_rollout(
    rollout: Callable[[jax.Array, Any], dict[str, jax.Array]],
) -> Callable[[jax.Array, Any], dict[str, jax.Array]]:
    """Vectorize a compiled rollout across a batch of independent RNG keys."""
    return jax.jit(jax.vmap(rollout, in_axes=(0, None)))
