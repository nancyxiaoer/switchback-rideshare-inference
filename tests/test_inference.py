from itertools import product

import numpy as np

from switchback_lab.inference import (
    exact_pure_history_probabilities,
    pure_history_linear,
    pure_history_naive,
    validate_switchback,
)


def test_linear_matches_naive() -> None:
    rng = np.random.default_rng(7)
    w = np.repeat(rng.binomial(1, 0.5, size=12), 5)
    for memory in (0, 1, 4, 5, 9):
        naive_one, naive_zero = pure_history_naive(w, memory)
        fast_one, fast_zero = pure_history_linear(w, memory)
        assert np.array_equal(naive_one, fast_one)
        assert np.array_equal(naive_zero, fast_zero)


def test_exact_probabilities_match_enumeration() -> None:
    n_events = 12
    block_length = 3
    memory = 4
    n_blocks = n_events // block_length
    pi_one, pi_zero = exact_pure_history_probabilities(
        n_events, memory, block_length, 0.5
    )

    count_one = np.zeros(n_events)
    count_zero = np.zeros(n_events)
    for labels in product((0, 1), repeat=n_blocks):
        w = np.repeat(labels, block_length)
        pure_one, pure_zero = pure_history_linear(w, memory)
        count_one += pure_one
        count_zero += pure_zero

    denominator = 2**n_blocks
    assert np.allclose(pi_one, count_one / denominator)
    assert np.allclose(pi_zero, count_zero / denominator)


def test_switchback_validation() -> None:
    w = np.repeat([1, 0, 1], 4)
    labels = validate_switchback(w, 4)
    assert np.array_equal(labels, np.array([1, 0, 1]))

