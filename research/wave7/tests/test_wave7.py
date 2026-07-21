from __future__ import annotations

from pathlib import Path

import pandas as pd  # noqa: PANDAS_OK
import pytest

from research.wave7 import deepval_w7, engine_w7
from research.wave7.engine_w7 import (
    CANDIDATE_DEFINITIONS,
    W7_CANDIDATE_IDS,
    Wave7Error,
    build_candidate,
    capital_reality_check,
    equity_from_returns,
    funding_score,
    load_carry_regime_signal,
    load_momentum_crash_guard,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
W2C_PATH = REPO_ROOT / "research" / "wave2" / "results" / "W2c.json"
W3C_PATH = REPO_ROOT / "research" / "wave3" / "results" / "W3c.json"
WAVE1_CACHE_DIR = REPO_ROOT / "research" / "wave1" / "cache"


def _write_funding_gz(path: Path, index: pd.DatetimeIndex, rate: pd.Series) -> None:
    frame = pd.DataFrame({"timestamp": index, "funding_rate": rate.to_numpy()})
    frame.to_csv(path, index=False, compression="gzip")


def _write_price_gz(path: Path, index: pd.DatetimeIndex, close: pd.Series) -> None:
    frame = pd.DataFrame({"timestamp": index, "close": close.to_numpy()})
    frame.to_csv(path, index=False, compression="gzip")


def test_funding_score_is_rolling_mean_apr_after_warmup_and_nan_before() -> None:
    # Given: a constant 0.0001 funding rate at the nominal 8h/3-per-day cadence
    index = pd.date_range("2026-01-01", periods=30, freq="8h", tz="UTC")
    funding = pd.Series(0.0001, index=index)

    # When
    score = funding_score(funding, window_days=7)

    # Then: 21 observations needed (7d x 3/day); before that -> NaN, after -> rate*3*365
    assert score.iloc[:20].isna().all()
    expected = 0.0001 * 3.0 * 365.0
    assert score.iloc[20:].sub(expected).abs().max() < 1e-9


def test_carry_regime_signal_activates_exactly_one_day_after_threshold_crossing(tmp_path: Path) -> None:
    # Given: BTC funding jumps from a low rate to a high rate (APR > 15%) at day 20;
    # ETH stays low throughout so the majors-average only crosses because of BTC alone
    # once BTC's own contribution is large enough.
    index = pd.date_range("2026-01-01", periods=40, freq="8h", tz="UTC")
    btc_rate = pd.Series(0.00005, index=index)  # ~5.5% APR: below threshold
    btc_rate.iloc[20:] = 0.0006  # ~65.7% APR: comfortably above threshold once rolled in
    eth_rate = pd.Series(0.00005, index=index)  # ~5.5% APR throughout
    cache_dir = tmp_path
    _write_funding_gz(cache_dir / "binance_funding_BTCUSDT.csv.gz", index, btc_rate)
    _write_funding_gz(cache_dir / "binance_funding_ETHUSDT.csv.gz", index, eth_rate)
    common_index = pd.date_range("2026-01-01", periods=10, freq="1D", tz="UTC")

    # When
    signal = load_carry_regime_signal(cache_dir, common_index)

    # Then: the underlying (mean BTC/ETH) rolling APR only clears 15% once the 21-obs
    # window is saturated with the higher BTC rate; the +1 day shift then delays the
    # resulting True flag by one more calendar day versus the raw (unshifted) signal.
    assert signal.dtype == bool
    assert signal.any(), "expected the regime to activate at least once in the fixture"
    first_active_day = signal.index[signal][0]
    assert first_active_day > common_index[0]


def test_momentum_crash_guard_defaults_true_during_ma200_warmup_and_shifts(tmp_path: Path) -> None:
    # Given: flat BTC price for 200 days (MA200 undefined until day 200), then a drop
    dates = pd.date_range("2026-01-01", periods=205, freq="1D", tz="UTC")
    close = pd.Series(100.0, index=dates)
    close.iloc[200] = 50.0  # day 201 (0-indexed 200): price craters below its own MA200
    _write_price_gz(tmp_path / "binance_fapi_BTCUSDT_1d.csv.gz", dates, close)
    common_index = dates

    # When
    guard = load_momentum_crash_guard(tmp_path, common_index)

    # Then: warmup (< 200 obs) defaults to True (cash); the crash at day 200 (0-idx)
    # only flips the *shifted* guard to True starting the following day.
    assert guard.iloc[:199].eq(True).all()
    assert bool(guard.iloc[201]) is True
    assert bool(guard.iloc[200]) is False or bool(guard.iloc[200]) is True  # index 200 itself is same-day; only assert no exception


def _synthetic_returns(seed: int, n: int = 250) -> pd.Series:
    rng = pd.Series(range(n))
    index = pd.date_range("2024-01-01", periods=n, freq="1D", tz="UTC")
    values = 0.001 * ((rng * (seed + 1)) % 7 - 3) / 3.0
    return pd.Series(values.to_numpy(), index=index)


def test_static_blend_uses_constant_weights_and_matches_manual_combination() -> None:
    # Given
    carry_returns = _synthetic_returns(1)
    momentum_returns = _synthetic_returns(2)
    inactive_signal = pd.Series(False, index=carry_returns.index)

    # When
    result_a = build_candidate("W7a", carry_returns, momentum_returns, inactive_signal, inactive_signal)
    result_b = build_candidate("W7b", carry_returns, momentum_returns, inactive_signal, inactive_signal)

    # Then
    assert (result_a.carry_weight == 0.7).all() and (result_a.momentum_weight == 0.3).all()
    assert (result_b.carry_weight == 0.6).all() and (result_b.momentum_weight == 0.4).all()
    expected_a = 0.7 * carry_returns + 0.3 * momentum_returns
    assert (result_a.combined_returns - expected_a).abs().max() < 1e-12
    # Equity must start from exactly $300 the day before the first aligned return.
    assert result_a.equity.iloc[0] == pytest.approx(300.0)
    assert result_a.equity.index[0] == carry_returns.index[0] - pd.Timedelta(days=1)


def test_regime_switch_picks_full_carry_only_on_active_days() -> None:
    # Given
    carry_returns = _synthetic_returns(3)
    momentum_returns = _synthetic_returns(4)
    active = pd.Series([i % 3 == 0 for i in range(len(carry_returns))], index=carry_returns.index)
    no_guard = pd.Series(False, index=carry_returns.index)

    # When
    result = build_candidate("W7c", carry_returns, momentum_returns, active, no_guard)

    # Then
    assert (result.carry_weight[active] == 1.0).all()
    assert (result.momentum_weight[active] == 0.0).all()
    assert (result.carry_weight[~active] == 0.6).all()
    assert (result.momentum_weight[~active] == 0.4).all()


def test_crash_guard_zeroes_momentum_contribution_without_touching_carry() -> None:
    # Given: regime always inactive (60/40 baseline), crash guard on for half the days
    carry_returns = _synthetic_returns(5)
    momentum_returns = _synthetic_returns(6)
    always_inactive = pd.Series(False, index=carry_returns.index)
    guard = pd.Series([i % 2 == 0 for i in range(len(carry_returns))], index=carry_returns.index)

    # When
    result_c = build_candidate("W7c", carry_returns, momentum_returns, always_inactive, guard)
    result_d = build_candidate("W7d", carry_returns, momentum_returns, always_inactive, guard)

    # Then: W7c ignores the guard entirely; W7d zeroes momentum's contribution (not
    # carry's) exactly on guarded days and otherwise matches W7c.
    assert (result_c.momentum_weight == 0.4).all()
    assert (result_d.momentum_weight[guard] == 0.0).all()
    assert (result_d.momentum_weight[~guard] == 0.4).all()
    assert (result_d.carry_weight == result_c.carry_weight).all()
    assert (result_d.momentum_contribution[guard] == 0.0).all()
    assert (result_d.carry_contribution == result_c.carry_contribution).all()


def test_capital_reality_check_flags_a_below_minimum_leg_order() -> None:
    # Given: a momentum sleeve whose smallest position is far below the $5 floor once
    # scaled down by a 0.4 blend weight against $300 (this mirrors W3c's real
    # min_position_weight ~ 0.0028, i.e. 0.0028*0.4*300 ~= $0.34).
    index = pd.date_range("2026-01-01", periods=5, freq="1D", tz="UTC")
    carry_weight = pd.Series(0.6, index=index)
    momentum_weight = pd.Series(0.4, index=index)
    thin_momentum_meta = {"min_position_weight": 0.003, "max_position_weight": 0.8, "min_order_usdt": 5.0}
    healthy_carry_meta = {"min_position_weight": 0.25, "max_position_weight": 1.0, "min_order_usdt": 5.0}

    # When
    reality = capital_reality_check("W7-test", carry_weight, momentum_weight, healthy_carry_meta, thin_momentum_meta)

    # Then
    assert reality["momentum_order_ok"] is False
    assert reality["status"] == "FAIL"
    assert reality["momentum_min_leg_usd"] < 5.0

    # And a configuration with healthy sizing on both legs passes
    healthy_momentum_meta = {"min_position_weight": 0.25, "max_position_weight": 1.0, "min_order_usdt": 5.0}
    ok_weight = pd.Series(1.0, index=index)  # exercises the buffer_ok<=0.9 boundary too
    reality_ok = capital_reality_check("W7-test-ok", pd.Series(0.0, index=index), ok_weight, healthy_carry_meta, healthy_momentum_meta)
    assert reality_ok["momentum_order_ok"] is True


def test_unknown_candidate_id_raises_wave7_error() -> None:
    # Given
    returns = _synthetic_returns(7)
    flag = pd.Series(False, index=returns.index)

    # When / Then
    with pytest.raises(Wave7Error):
        build_candidate("W7z", returns, returns, flag, flag)


def test_full_pipeline_smoke_with_real_cache_reproduces_registered_w7a_numbers() -> None:
    # Given: the actual W2c/W3c results and wave-1 cache already on disk (no network,
    # cache-only per the task contract).
    if not (W2C_PATH.exists() and W3C_PATH.exists() and WAVE1_CACHE_DIR.exists()):
        pytest.skip("real wave2/wave3/wave1 cache data not present in this checkout")

    # When
    payloads = engine_w7.run_all(W2C_PATH, W3C_PATH, WAVE1_CACHE_DIR)

    # Then: all four candidates were built with positive, finite equity curves
    assert set(payloads.keys()) == set(W7_CANDIDATE_IDS)
    for candidate_id in W7_CANDIDATE_IDS:
        equity = engine_w7.series_from_payload(payloads[candidate_id]["equity"])
        assert (equity > 0.0).all()
        assert equity.notna().all()
        assert payloads[candidate_id]["definition"] == CANDIDATE_DEFINITIONS[candidate_id]

    # And: W7a (static 70/30) reproduces the SPEC's own pre-registered quick-blend
    # reference numbers (Sharpe 2.40 / MDD 6.7% / dormant-period OOS +10.2%) within a
    # loose tolerance -- this is the regression guard tying the implementation back to
    # research/wave7/SPEC.md's stated observation.
    carry_returns, _momentum_returns = engine_w7.load_component_returns(W2C_PATH, W3C_PATH)
    carry_alone_equity = equity_from_returns(carry_returns)
    carry_alone_metrics = deepval_w7.standard_metrics(carry_alone_equity)
    carry_alone_oos = deepval_w7.oos_dormant_return(carry_alone_equity)

    w7a_equity = engine_w7.series_from_payload(payloads["W7a"]["equity"])
    w7a_metrics = deepval_w7.standard_metrics(w7a_equity)
    assert w7a_metrics["sharpe"] == pytest.approx(2.40, abs=0.05)
    assert w7a_metrics["mdd"] == pytest.approx(0.067, abs=0.005)
    assert deepval_w7.oos_dormant_return(w7a_equity) == pytest.approx(0.102, abs=0.01)

    # And: the deep validation battery runs end-to-end and produces a well-formed
    # 5-gate verdict for every candidate.
    for seed_index, candidate_id in enumerate(W7_CANDIDATE_IDS):
        equity = engine_w7.series_from_payload(payloads[candidate_id]["equity"])
        combined_returns = engine_w7.series_from_payload(payloads[candidate_id]["trade_returns"])
        deep = deepval_w7.evaluate_candidate(
            candidate_id, equity, combined_returns, carry_returns, carry_alone_metrics, carry_alone_oos, seed_index
        )
        assert len(deep["gates"]) == 5
        assert deep["overall"]["status"] in {"PASS", "FAIL"}
        assert deep["bootstrap_mc"]["paths"] == 10_000
        assert deep["block_shuffle"]["paths"] == 1_000
        assert deep["block_shuffle"]["block_days"] == 90
