"""Causal end-to-end signal checks, real CLI errors, and candidate accounting."""

import subprocess
import sys
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from alt_search_models import MarketData
from run_x_discovery import archive_for, configs, lock_test_ids
from x_discovery_pairs import fit_pair
from x_discovery_signals import SignalFamily, SignalSpec, build_signal, specifications


def synthetic_market() -> MarketData:
    rng = np.random.default_rng(773)
    innovations = rng.normal(0, .01, (220, 14))
    innovations[:, 2:] += innovations[:, :1] * .8
    prices = np.exp(4 + np.cumsum(innovations, axis=0))
    return MarketData(tuple(datetime(2023, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(220)),
                      tuple(f"A{i}" for i in range(14)), prices.copy(), prices * 1.01,
                      prices * .99, prices.copy(), np.full_like(prices, 1e7))


@pytest.mark.parametrize("family", [SignalFamily.OU, SignalFamily.RESIDUAL, SignalFamily.DONCHIAN])
def test_future_prices_cannot_change_an_earlier_open_decision(family: SignalFamily) -> None:
    # Given: a price path whose suffix is replaced, without altering past observations.
    data = synthetic_market()
    modified = data.close_price.copy()
    modified[150:] *= np.linspace(1, 4, 70)[:, None]
    future = replace(data, close_price=modified)
    spec = SignalSpec(family, 60, horizon=3, direction=-1, entry=1.5)
    # When: both full-path computations independently build their training-selected signal.
    original, _ = build_signal(data, spec, bars_per_day=1, train_end=80)
    altered, _ = build_signal(future, spec, bars_per_day=1, train_end=80)
    # Then: the decision at open 150 has not seen close 150.
    np.testing.assert_array_equal(original[:151], altered[:151])


def test_factor_weights_never_exceed_one_unit_of_gross_exposure() -> None:
    # Given: factor-correlated synthetic returns and a two-sided residual portfolio.
    data = synthetic_market()
    # When: the estimate includes the BTC/ETH-equivalent hedge weights.
    weights, _ = build_signal(data, SignalSpec(SignalFamily.RESIDUAL, 30, 3), bars_per_day=1, train_end=80)
    # Then: the hedge is included in gross exposure, not hidden outside leverage.
    assert np.max(np.abs(weights).sum(axis=1)) <= 1 + 1e-12
    assert np.any(weights[:, :2] != 0)


def test_ou_fit_detects_a_seeded_mean_reverting_spread() -> None:
    # Given: log prices with beta 1.5 and a known stationary spread.
    rng = np.random.default_rng(911)
    right = np.cumsum(rng.normal(0, .02, 500)) + 4
    noise = rng.normal(0, .002, 500)
    spread = np.zeros(500)
    for index in range(1, 500):
        spread[index] = .85 * spread[index - 1] + noise[index]
    # When: fitting the full training-only sample.
    result = fit_pair(1.5 * right + spread, right, test=True)
    # Then: beta and half life agree with the generating mechanism within sampling error.
    assert result is not None
    assert result.beta == pytest.approx(1.5, abs=.02)
    assert 2 < result.half_life < 10
    assert result.p_value <= .05


def test_preregistered_candidate_count_is_245() -> None:
    # Given: three intervals and the explicit original-plus-addendum grid.
    # When: materializing registration without prices or outcomes.
    candidates = [item for interval in ("1H", "4H", "1D")
                  for spec in specifications(interval) for item in configs(interval, spec)]
    # Then: 195 original, 20 pressure policies, and 30 exit ablations are all unique.
    assert len(candidates) == len({item.identifier for item in candidates}) == 245


def test_locked_test_hash_matches_exact_disk_bytes_on_windows(tmp_path: Path) -> None:
    from hashlib import sha256

    lock_file = tmp_path / "test_ids_locked.txt"
    digest = lock_test_ids(lock_file, ("first", "second"))
    assert digest == sha256(lock_file.read_bytes()).hexdigest()
    assert lock_file.read_bytes() == b"first\nsecond\n"


def test_daily_archive_routing_does_not_use_the_intraday_partition_manifest() -> None:
    # Given: non-overlapping daily and intraday collections.
    # When: requesting the daily interval.
    selected = archive_for("1D", Path("intraday_v5"), Path("daily_v2"))
    # Then: select the actual daily manifest, not a missing intraday group.
    assert selected == Path("daily_v2")


def test_cli_rejects_missing_archive_without_writing_a_performance_report(tmp_path: Path) -> None:
    # Given: a genuine command-line invocation with no archive manifest.
    script = Path(__file__).with_name("run_x_discovery.py")
    # When: starting the actual adapter and runner.
    result = subprocess.run([sys.executable, str(script), "--archive", str(tmp_path / "missing"),
                             "--output", str(tmp_path / "result")], capture_output=True, timeout=30, check=False)
    # Then: failure is observable and no performance is invented.
    assert result.returncode != 0
    assert not (tmp_path / "result" / "report.json").exists()
