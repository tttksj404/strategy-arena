from __future__ import annotations

import numpy as np

import selection_bias_stats as stats


def test_dominant_candidate_has_zero_cscv_probability_of_backtest_overfitting() -> None:
    returns = np.asarray(
        [
            [0.02, 0.01, 0.02, 0.01, 0.02, 0.01, 0.02, 0.01],
            [0.00, 0.01, 0.00, -0.01, 0.00, 0.01, 0.00, -0.01],
        ],
        dtype=float,
    )

    result = stats.cscv_pbo(returns, bins=4)

    assert result.pbo == 0.0
    assert result.splits == 6


def test_deflated_sharpe_penalizes_more_trials() -> None:
    returns = np.asarray([0.02, -0.01, 0.015, -0.005, 0.01, 0.0, 0.012, -0.004], dtype=float)

    few = stats.deflated_sharpe(returns, trials=2)
    many = stats.deflated_sharpe(returns, trials=200)

    assert many.benchmark_sharpe > few.benchmark_sharpe
    assert many.probability < few.probability


def test_deflated_sharpe_has_a_finite_single_trial_baseline() -> None:
    returns = np.asarray([0.02, -0.01, 0.015, -0.005, 0.01, 0.0, 0.012, -0.004], dtype=float)

    result = stats.deflated_sharpe(returns, trials=1)

    assert np.isfinite(result.benchmark_sharpe)
    assert result.benchmark_sharpe == 0.0


def test_cscv_moments_match_the_direct_return_calculation() -> None:
    returns = np.asarray(
        [
            [0.01, 0.02, -0.01, 0.01, 0.02, 0.00, -0.01, 0.01],
            [0.00, 0.01, 0.00, -0.01, 0.01, -0.01, 0.00, 0.01],
        ],
        dtype=float,
    )

    sums, squares, counts = stats.bin_moments(returns, bins=4)

    assert stats.cscv_pbo_from_moments(sums, squares, counts) == stats.cscv_pbo(returns, bins=4)
