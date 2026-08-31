from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import numpy as np


@dataclass(frozen=True)
class TestResult:
    statistic: float
    p_value: float
    n_randomized_units: int
    n_focal_events: int


def validate_switchback(w: np.ndarray, block_length: int) -> np.ndarray:
    """Validate an equal-event-length switchback path and return block labels."""
    w = np.asarray(w, dtype=np.int8)
    if w.ndim != 1 or len(w) == 0:
        raise ValueError("w must be a non-empty one-dimensional array")
    if block_length <= 0:
        raise ValueError("block_length must be positive")
    if not np.isin(w, (0, 1)).all():
        raise ValueError("w must contain only 0/1")

    n_blocks = ceil(len(w) / block_length)
    labels = w[np.arange(n_blocks) * block_length]
    reconstructed = np.repeat(labels, block_length)[: len(w)]
    if not np.array_equal(w, reconstructed):
        raise ValueError("assignment is not constant inside each switchback block")
    return labels


def make_predetermined_sections(
    n_events: int, block_length: int, min_section_length: int
) -> list[tuple[int, int]]:
    """Return sections as half-open block-index intervals [a, b)."""
    if min_section_length <= 0:
        raise ValueError("min_section_length must be positive")

    n_blocks = ceil(n_events / block_length)
    sections: list[tuple[int, int]] = []
    section_start = 0
    accumulated_length = 0

    for block in range(n_blocks):
        event_start = block * block_length
        event_end = min((block + 1) * block_length, n_events)
        accumulated_length += event_end - event_start

        if accumulated_length >= min_section_length:
            sections.append((section_start, block + 1))
            section_start = block + 1
            accumulated_length = 0

    # A short remainder cannot form a legal section, so merge it into the
    # preceding predetermined section. This rule is fixed before seeing w/y.
    if section_start < n_blocks:
        if not sections:
            raise ValueError("experiment is shorter than min_section_length")
        sections[-1] = (sections[-1][0], n_blocks)

    return sections


def pure_history_naive(w: np.ndarray, memory: int) -> tuple[np.ndarray, np.ndarray]:
    """Reference O(T * memory) implementation."""
    w = np.asarray(w, dtype=np.int8)
    if not 0 <= memory < len(w):
        raise ValueError("memory must satisfy 0 <= memory < len(w)")

    pure_one = np.zeros(len(w), dtype=bool)
    pure_zero = np.zeros(len(w), dtype=bool)
    for t in range(memory, len(w)):
        window = w[t - memory : t + 1]
        pure_one[t] = np.all(window == 1)
        pure_zero[t] = np.all(window == 0)
    return pure_one, pure_zero


def pure_history_linear(w: np.ndarray, memory: int) -> tuple[np.ndarray, np.ndarray]:
    """O(T) sliding-window implementation with O(T) output memory."""
    w = np.asarray(w, dtype=np.int8)
    if not 0 <= memory < len(w):
        raise ValueError("memory must satisfy 0 <= memory < len(w)")

    pure_one = np.zeros(len(w), dtype=bool)
    pure_zero = np.zeros(len(w), dtype=bool)
    window_length = memory + 1
    window_sum = int(w[:window_length].sum())

    pure_one[memory] = window_sum == window_length
    pure_zero[memory] = window_sum == 0

    for t in range(memory + 1, len(w)):
        window_sum += int(w[t]) - int(w[t - window_length])
        pure_one[t] = window_sum == window_length
        pure_zero[t] = window_sum == 0

    return pure_one, pure_zero


def exact_pure_history_probabilities(
    n_events: int,
    memory: int,
    block_length: int,
    treatment_probability: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact exposure probabilities for independent equal-length blocks."""
    if not 0.0 < treatment_probability < 1.0:
        raise ValueError("treatment_probability must be in (0, 1)")
    if not 0 <= memory < n_events:
        raise ValueError("memory must satisfy 0 <= memory < n_events")

    pi_one = np.zeros(n_events, dtype=float)
    pi_zero = np.zeros(n_events, dtype=float)
    t = np.arange(memory, n_events)
    first_block = (t - memory) // block_length
    last_block = t // block_length
    n_spanned_blocks = last_block - first_block + 1
    pi_one[t] = treatment_probability**n_spanned_blocks
    pi_zero[t] = (1.0 - treatment_probability) ** n_spanned_blocks
    return pi_one, pi_zero


def pure_history_ht_estimate(
    y: np.ndarray,
    w: np.ndarray,
    memory: int,
    block_length: int,
    treatment_probability: float = 0.5,
) -> float:
    """Horvitz-Thompson contrast based on pure treatment/control histories."""
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=np.int8)
    if len(y) != len(w):
        raise ValueError("y and w must have equal length")

    pure_one, pure_zero = pure_history_linear(w, memory)
    pi_one, pi_zero = exact_pure_history_probabilities(
        len(w), memory, block_length, treatment_probability
    )
    eligible = np.arange(memory, len(w))
    contribution = np.zeros(len(eligible), dtype=float)
    is_one = pure_one[eligible]
    is_zero = pure_zero[eligible]
    contribution[is_one] = y[eligible[is_one]] / pi_one[eligible[is_one]]
    contribution[is_zero] = -y[eligible[is_zero]] / pi_zero[eligible[is_zero]]
    return float(contribution.mean())


def draw_switchback_assignment(
    n_events: int,
    block_length: int,
    treatment_probability: float = 0.5,
    seed: int = 42,
) -> np.ndarray:
    """Draw independent Bernoulli labels at the block level and expand to events."""
    if n_events <= 0 or block_length <= 0:
        raise ValueError("n_events and block_length must be positive")
    if not 0.0 < treatment_probability < 1.0:
        raise ValueError("treatment_probability must be in (0, 1)")
    rng = np.random.default_rng(seed)
    labels = rng.binomial(
        1, treatment_probability, size=ceil(n_events / block_length)
    )
    return np.repeat(labels, block_length)[:n_events].astype(np.int8)


def _monte_carlo_p_value(
    simulated: np.ndarray, observed: float, alternative: str
) -> float:
    if alternative == "greater":
        count = np.count_nonzero(simulated >= observed)
    elif alternative == "less":
        count = np.count_nonzero(simulated <= observed)
    elif alternative == "two-sided":
        count = np.count_nonzero(np.abs(simulated) >= abs(observed))
    else:
        raise ValueError("alternative must be greater, less, or two-sided")
    return (count + 1.0) / (len(simulated) + 1.0)


def global_total_effect_crt(
    y: np.ndarray,
    w: np.ndarray,
    block_length: int,
    memory: int,
    treatment_probability: float = 0.5,
    n_resamples: int = 10_000,
    alternative: str = "greater",
    seed: int = 42,
) -> TestResult:
    """Conditional randomization test of the partially sharp total-effect null."""
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=np.int8)
    if len(y) != len(w):
        raise ValueError("y and w must have equal length")
    labels = validate_switchback(w, block_length)
    sections = make_predetermined_sections(len(w), block_length, memory + 1)

    focal_sums: list[float] = []
    focal_counts: list[int] = []
    observed_labels: list[int] = []
    conditional_probabilities: list[float] = []

    for first_block, last_block_exclusive in sections:
        section_labels = labels[first_block:last_block_exclusive]
        if not np.all(section_labels == section_labels[0]):
            continue

        event_start = first_block * block_length
        event_end = min(last_block_exclusive * block_length, len(w))
        focal = y[event_start + memory : event_end]
        if len(focal) == 0:
            continue

        n_merged_blocks = last_block_exclusive - first_block
        numerator = treatment_probability**n_merged_blocks
        denominator = numerator + (1.0 - treatment_probability) ** n_merged_blocks
        conditional_probability = numerator / denominator

        focal_sums.append(float(focal.sum()))
        focal_counts.append(len(focal))
        observed_labels.append(int(section_labels[0]))
        conditional_probabilities.append(conditional_probability)

    if not focal_sums:
        raise ValueError("no realized constant section contains focal events")

    sums = np.asarray(focal_sums)
    counts = np.asarray(focal_counts)
    z_obs = np.asarray(observed_labels)
    p = np.asarray(conditional_probabilities)
    n_focal = int(counts.sum())

    observed = float(np.sum(np.where(z_obs == 1, sums / p, -sums / (1.0 - p))) / n_focal)

    rng = np.random.default_rng(seed)
    z_star = rng.random((n_resamples, len(p))) < p
    simulated = (
        z_star * (sums / p) - (1 - z_star) * (sums / (1.0 - p))
    ).sum(axis=1) / n_focal

    return TestResult(
        statistic=observed,
        p_value=_monte_carlo_p_value(simulated, observed, alternative),
        n_randomized_units=len(p),
        n_focal_events=n_focal,
    )


def carryover_crt(
    y: np.ndarray,
    w: np.ndarray,
    block_length: int,
    tested_memory: int,
    treatment_probability: float = 0.5,
    n_resamples: int = 10_000,
    alternative: str = "two-sided",
    seed: int = 42,
) -> TestResult:
    """CRT of H0: treatment carryover lasts at most tested_memory events."""
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=np.int8)
    if len(y) != len(w):
        raise ValueError("y and w must have equal length")
    labels = validate_switchback(w, block_length)
    sections = make_predetermined_sections(len(w), block_length, tested_memory + 1)

    focal_means: list[float] = []
    preceding_labels: list[int] = []
    n_focal = 0

    # Odd section (0-based even index) supplies a randomized preceding label;
    # the next section is held fixed and supplies focal outcomes.
    for section_index in range(0, len(sections) - 1, 2):
        previous_first, previous_last = sections[section_index]
        focal_first, focal_last = sections[section_index + 1]
        focal_start = focal_first * block_length
        focal_end = min(focal_last * block_length, len(w))
        focal = y[focal_start + tested_memory : focal_end]
        if len(focal) == 0:
            continue
        focal_means.append(float(focal.mean()))
        preceding_labels.append(int(labels[previous_last - 1]))
        n_focal += len(focal)

    if not focal_means:
        raise ValueError("no focal section pairs are available")

    means = np.asarray(focal_means)
    z_obs = np.asarray(preceding_labels)
    q = treatment_probability
    observed = float(np.mean(z_obs * means / q - (1 - z_obs) * means / (1 - q)))

    rng = np.random.default_rng(seed)
    z_star = rng.random((n_resamples, len(means))) < q
    simulated = np.mean(
        z_star * means / q - (1 - z_star) * means / (1 - q), axis=1
    )

    return TestResult(
        statistic=observed,
        p_value=_monte_carlo_p_value(simulated, observed, alternative),
        n_randomized_units=len(means),
        n_focal_events=n_focal,
    )


def anticipation_pirt_greater_fast(
    y: np.ndarray,
    w: np.ndarray,
    block_length: int,
    fixed_prefix_events: int,
    treatment_probability: float = 0.5,
    n_resamples: int = 10_000,
    seed: int = 42,
) -> TestResult:
    """Fast one-sided PIRT for non-anticipation under equal-length blocks.

    For each alternative schedule, only the first block at which it differs
    from the observed schedule can make the two pairwise signed statistics
    differ. Prefix sums make each comparison O(1) after that block is found.
    """
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=np.int8)
    if len(y) != len(w):
        raise ValueError("y and w must have equal length")
    labels = validate_switchback(w, block_length)
    n_blocks = len(labels)
    n_fixed_blocks = ceil(fixed_prefix_events / block_length)
    if not 1 <= n_fixed_blocks < n_blocks:
        raise ValueError("fixed prefix must cover at least one but not all blocks")

    rng = np.random.default_rng(seed)
    alternatives = (
        rng.random((n_resamples, n_blocks)) < treatment_probability
    ).astype(np.int8)
    alternatives[:, :n_fixed_blocks] = labels[:n_fixed_blocks]

    differs = alternatives != labels[None, :]
    has_difference = differs.any(axis=1)
    first_different_block = np.argmax(differs, axis=1)
    prefix_sum_y = np.concatenate(([0.0], np.cumsum(y, dtype=float)))

    # A - B. No-difference draws are exact ties and therefore count as A >= B.
    pairwise_difference = np.zeros(n_resamples, dtype=float)
    rows = np.flatnonzero(has_difference)
    blocks = first_different_block[rows]
    n_common_events = np.minimum(blocks * block_length, len(y))
    boundary_events = n_common_events - 1
    centered_boundary_y = (
        y[boundary_events] - prefix_sum_y[n_common_events] / n_common_events
    )
    alternative_sign = 2 * alternatives[rows, blocks] - 1
    observed_sign = 2 * labels[blocks] - 1
    pairwise_difference[rows] = (
        (alternative_sign - observed_sign)
        * centered_boundary_y
        / n_common_events
    )

    p_value = (np.count_nonzero(pairwise_difference >= 0.0) + 1.0) / (
        n_resamples + 1.0
    )
    return TestResult(
        statistic=float(np.mean(pairwise_difference)),
        p_value=float(p_value),
        n_randomized_units=n_blocks - n_fixed_blocks,
        n_focal_events=int(np.median(first_different_block[has_difference]) * block_length)
        if has_difference.any()
        else len(y),
    )


def synthetic_outcomes(
    w: np.ndarray,
    direct_effect: float,
    carryover_coefficients: tuple[float, ...] = (),
    anticipation_effect: float = 0.0,
    noise_std: float = 1.0,
    seed: int = 42,
) -> np.ndarray:
    """Controlled DGP for size/power validation of the three tests."""
    w = np.asarray(w, dtype=float)
    rng = np.random.default_rng(seed)
    t = np.arange(len(w), dtype=float)
    y = 0.25 * np.sin(2.0 * np.pi * t / max(len(w), 1))
    y += direct_effect * w

    for lag, coefficient in enumerate(carryover_coefficients, start=1):
        y[lag:] += coefficient * w[:-lag]

    if anticipation_effect != 0.0:
        y[:-1] += anticipation_effect * w[1:]

    y += rng.normal(0.0, noise_std, size=len(w))
    return y
