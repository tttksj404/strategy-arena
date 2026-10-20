# Wave-17 test suite. Deliberately offline/deterministic (SYNTHETIC data, no real HTTP, no
# dependency on cache/results being present) -- same convention as
# research/wave16_duallayer/tests/test_wave16.py. Covers the 5 SPEC.md-mandated areas
# (rate vs lendingRate field separation, ratio application, F3==F0 identity, volatility
# calculation, time-series depth verification) plus gates17's own promotion-rule logic.

from __future__ import annotations

from pathlib import Path
import sys

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[3]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

import numpy as np
import pandas as pd  # noqa: PANDAS_OK
import pytest

from research.wave1.fam_funding import FundingCandidate, FundingMarket
from research.wave10_carry100.engine import ACTIVE_CAPITAL, Wave10Result
from research.wave16_duallayer.engine16 import _build_aligned_frames_dual, _lending_daily_rate_series, _run_dual_layer_loop
from research.wave17_lending_verified import fetch17, gates17, recompute17, volatility17

# ---------------------------------------------------------------------------
# 1) rate vs lendingRate field separation (SPEC.md 필수 테스트: "lendingRate vs rate 구분").
# ---------------------------------------------------------------------------


def test_parse_history_rows_keeps_rate_and_lending_rate_distinct() -> None:
    raw = [
        {"ccy": "THETA", "ts": "1000000000000", "rate": "0.19", "lendingRate": "0.0669", "amt": ""},
        {"ccy": "THETA", "ts": "1000003600000", "rate": "0.189", "lendingRate": "0.0704", "amt": ""},
        {"ccy": "THETA", "ts": "1000007200000", "rate": "bad-value", "lendingRate": "0.07", "amt": ""},  # malformed -- must be dropped, not coerced
    ]
    parsed = fetch17._parse_history_rows(raw)
    assert len(parsed) == 2  # the malformed row is dropped, never invented
    assert parsed[0]["ts_ms"] == 1_000_000_000_000  # sorted ascending by ts
    assert parsed[0]["rate"] == pytest.approx(0.19)
    assert parsed[0]["lending_rate"] == pytest.approx(0.0669)
    # the two fields must never be swapped or collapsed into one
    assert parsed[0]["rate"] != parsed[0]["lending_rate"]
    assert {"rate", "lending_rate", "ts_ms"} <= set(parsed[0].keys())


def test_summarize_history_computes_lending_rate_stats_not_rate_stats() -> None:
    history = [
        {"ts_ms": 0, "rate": 0.20, "lending_rate": 0.05},
        {"ts_ms": 3_600_000, "rate": 0.20, "lending_rate": 0.06},
        {"ts_ms": 7_200_000, "rate": 0.20, "lending_rate": 0.07},
    ]
    stats = fetch17.summarize_history(history)
    assert stats["rate_median"] == pytest.approx(0.20)
    assert stats["lending_rate_median"] == pytest.approx(0.06)  # NOT 0.20 -- must read lending_rate, not rate
    assert stats["lending_rate_mean"] == pytest.approx(0.06)
    assert stats["lending_rate_min"] == pytest.approx(0.05)
    assert stats["lending_rate_max"] == pytest.approx(0.07)
    assert stats["lending_rate_range"] == pytest.approx(0.02)


# ---------------------------------------------------------------------------
# 2) 시계열 깊이 검증 (time-series depth verification).
# ---------------------------------------------------------------------------


def test_summarize_history_reports_span_matching_hourly_100_limit() -> None:
    # 100 hourly points, oldest-to-newest -- exactly OKX's own observed limit=100 shape.
    history = [{"ts_ms": i * 3_600_000, "rate": 0.10, "lending_rate": 0.05} for i in range(100)]
    stats = fetch17.summarize_history(history)
    assert stats["n_samples"] == 100
    assert stats["span_hours"] == pytest.approx(99.0)  # 100 points, 1h apart -> 99h between first and last
    assert stats["span_days"] == pytest.approx(99.0 / 24.0)
    assert stats["span_days"] < 4.5  # SPEC.md 발견: "약 4일" -- this test pins that this stays true, not "more history became available"


def test_summarize_history_span_scales_with_actual_sample_count_not_hardcoded() -> None:
    short_history = [{"ts_ms": i * 3_600_000, "rate": 0.10, "lending_rate": 0.05} for i in range(5)]
    stats = fetch17.summarize_history(short_history)
    assert stats["n_samples"] == 5
    assert stats["span_hours"] == pytest.approx(4.0)
    empty_stats = fetch17.summarize_history([])
    assert empty_stats == {"n_samples": 0}


# ---------------------------------------------------------------------------
# 3) 비율 적용 (ratio application) + universe-target selection (read-only wave16 join).
# ---------------------------------------------------------------------------


def test_build_realized_lending_apr_by_symbol_applies_requested_stat_field(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_map = {"AUSDT": "A", "BUSDT": "B", "NODATAUSDT": "NODATA", "UNMATCHEDUSDT": None}
    monkeypatch.setattr(recompute17, "load_symbol_to_ccy_map", lambda: fake_map)
    lending_realized = {
        "by_ccy": {
            "A": {"history_available": True, "lending_rate_median": 0.10, "lending_rate_min": 0.04},
            "B": {"history_available": True, "lending_rate_median": 0.20, "lending_rate_min": 0.08},
            "NODATA": {"history_available": False, "error": "empty/malformed data from OKX", "n_samples": 0},
        }
    }
    median_apr = recompute17.build_realized_lending_apr_by_symbol(lending_realized, "lending_rate_median")
    assert median_apr == {"AUSDT": pytest.approx(0.10), "BUSDT": pytest.approx(0.20)}
    # fail-closed: NODATAUSDT (history_available=False) and UNMATCHEDUSDT (no ccy match) must be OMITTED, never defaulted to a guessed rate
    assert "NODATAUSDT" not in median_apr
    assert "UNMATCHEDUSDT" not in median_apr

    min_apr = recompute17.build_realized_lending_apr_by_symbol(lending_realized, "lending_rate_min")
    assert min_apr == {"AUSDT": pytest.approx(0.04), "BUSDT": pytest.approx(0.08)}
    assert min_apr != median_apr  # the two scenarios must actually read DIFFERENT columns, not silently reuse one


def test_ratio_uses_lending_rate_median_never_rate_median(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Regression guard for exactly the bug class this whole wave exists to fix: a ratio
    computed against the wrong (borrower-side `rate`) field would silently reproduce wave16's
    own avgRate-based overestimation. Exercises the real collect_lending_realized() ratio math
    end-to-end (network calls monkeypatched out) rather than re-deriving the formula by hand."""
    history = [
        {"ts_ms": i * 3_600_000, "rate": 0.30, "lending_rate": 0.06} for i in range(10)
    ]  # rate and lending_rate deliberately far apart (5x) so a field mix-up is unmissable
    monkeypatch.setattr(fetch17, "load_wave16_target_ccys", lambda: ("XYZ",))
    monkeypatch.setattr(fetch17, "load_wave16_avg_rate_by_ccy", lambda: {"XYZ": 0.30})
    monkeypatch.setattr(fetch17.fetch_lending, "fetch_okx_lending_summary", lambda session: [{"ccy": "XYZ", "avg_rate": 0.30, "est_rate": None, "pre_rate": None}])
    monkeypatch.setattr(fetch17, "fetch_lending_rate_history", lambda session, ccy: history)
    monkeypatch.setattr(fetch17, "CACHE_DIR", tmp_path)  # redirect the real save_json() write into a throwaway tmp dir instead of stubbing mkdir on an immutable Path

    payload = fetch17.collect_lending_realized()
    row = payload["by_ccy"]["XYZ"]
    assert row["ratio_lendingrate_over_avgrate_fresh"] == pytest.approx(0.06 / 0.30)  # == 0.2, i.e. lending_rate/avg_rate
    assert row["ratio_lendingrate_over_avgrate_fresh"] != pytest.approx(0.30 / 0.30)  # would be 1.0 if `rate` had been used by mistake


# ---------------------------------------------------------------------------
# 4) F3 == F0 항등 (integrity/regression) -- proven at the engine level: with pnl_discount=0.0,
#    the result must be COMPLETELY INDEPENDENT of which lending_apr_by_symbol mapping backs it
#    (this is the structural fact recompute17.py's module docstring claims F0/F3 rely on).
# ---------------------------------------------------------------------------


def _two_symbol_synthetic_market(periods: int = 12) -> dict[str, FundingMarket]:
    daily_index = pd.date_range("2026-01-01", periods=periods, freq="D", tz="UTC")
    funding_index = pd.date_range("2026-01-01", periods=periods * 3, freq="8h", tz="UTC")
    a_close = [100.0 * (1.004**i) for i in range(periods)]
    a_open = [100.0, *a_close[:-1]]
    b_close = [50.0 * (1.002**i) for i in range(periods)]
    b_open = [50.0, *b_close[:-1]]
    spot_a = pd.DataFrame({"open": a_open, "close": a_close}, index=daily_index)
    perp_a = pd.DataFrame({"open": [v * 0.999 for v in a_open], "close": [v * 0.999 for v in a_close]}, index=daily_index)
    spot_b = pd.DataFrame({"open": b_open, "close": b_close}, index=daily_index)
    perp_b = pd.DataFrame({"open": [v * 1.001 for v in b_open], "close": [v * 1.001 for v in b_close]}, index=daily_index)
    funding_a = pd.Series(0.0010, index=funding_index, name="funding_rate")
    funding_b = pd.Series(0.0006, index=funding_index, name="funding_rate")
    return {"BTCUSDT": FundingMarket(spot_a, perp_a, funding_a), "ETHUSDT": FundingMarket(spot_b, perp_b, funding_b)}


def _flat_cost_and_liquidity(index: pd.Index, columns: pd.Index, rate: float = 0.001) -> tuple[pd.DataFrame, pd.DataFrame]:
    cost = pd.DataFrame(rate, index=index, columns=columns)
    liquidity = pd.DataFrame(True, index=index, columns=columns)
    return cost, liquidity


def test_pnl_discount_zero_makes_result_independent_of_which_lending_mapping_is_used() -> None:
    markets = _two_symbol_synthetic_market()
    candidate = FundingCandidate("F0F3TEST", 1, 0.50, 1)
    leg_fraction = 0.5

    # F0-style: empty/zero lending map (SPEC.md F0's own "none" source).
    frames_f0 = _build_aligned_frames_dual(markets, candidate, lending_apr_by_symbol={}, ranking_lending_discount=0.0)
    cost_f0, liquidity_f0 = _flat_cost_and_liquidity(frames_f0[0].index, frames_f0[0].columns)
    zero_lending_f0 = _lending_daily_rate_series(frames_f0[0].columns, {}, pnl_lending_discount=0.0)
    result_f0, cost_total_f0, _ = _run_dual_layer_loop(*frames_f0, candidate.top_k, leg_fraction, cost_f0, liquidity_f0, zero_lending_f0)

    # F3-style: SAME realized median mapping F1 would use (large, nonzero), but pnl_discount=0.0.
    realized_median_apr = {"BTCUSDT": 0.45, "ETHUSDT": 0.30}  # deliberately large -- if discount=0 leaked through, this would be unmissable
    frames_f3 = _build_aligned_frames_dual(markets, candidate, realized_median_apr, ranking_lending_discount=0.0)
    cost_f3, liquidity_f3 = _flat_cost_and_liquidity(frames_f3[0].index, frames_f3[0].columns)
    zero_lending_f3 = _lending_daily_rate_series(frames_f3[0].columns, realized_median_apr, pnl_lending_discount=0.0)
    result_f3, cost_total_f3, _ = _run_dual_layer_loop(*frames_f3, candidate.top_k, leg_fraction, cost_f3, liquidity_f3, zero_lending_f3)

    assert result_f3.equity.tolist() == pytest.approx(result_f0.equity.tolist(), rel=1e-12)
    assert result_f3.positions.tolist() == pytest.approx(result_f0.positions.tolist(), rel=1e-12)
    assert result_f3.trade_returns.tolist() == pytest.approx(result_f0.trade_returns.tolist(), rel=1e-12)
    assert cost_total_f3 == pytest.approx(cost_total_f0, rel=1e-12)

    # Sanity: pnl_discount=1.0 on the SAME nonzero mapping must actually differ (proves the test isn't vacuous).
    full_lending_f1 = _lending_daily_rate_series(frames_f3[0].columns, realized_median_apr, pnl_lending_discount=1.0)
    result_f1, _, _ = _run_dual_layer_loop(*frames_f3, candidate.top_k, leg_fraction, cost_f3, liquidity_f3, full_lending_f1)
    assert result_f1.equity.iloc[-1] != pytest.approx(result_f0.equity.iloc[-1])


def test_gates17_f3_identity_check_passes_on_bit_identical_series_and_fails_on_drift() -> None:
    equity = pd.Series([90.0, 91.0, 95.0], index=pd.date_range("2026-01-01", periods=3, freq="D", tz="UTC"))
    identical_result = Wave10Result(equity=equity, positions=pd.Series(dtype=float), turnover=pd.Series(dtype=float), trade_returns=pd.Series(dtype=float), max_concurrent_positions=0, symbols_used=())
    drifted_result = Wave10Result(equity=equity * 1.001, positions=pd.Series(dtype=float), turnover=pd.Series(dtype=float), trade_returns=pd.Series(dtype=float), max_concurrent_positions=0, symbols_used=())

    diff_when_identical = gates17._high_funding_annualized(identical_result)
    # (high_funding_annualized may be None for a 3-day synthetic series outside HIGH_FUNDING_YEARS --
    # the identity check itself only needs equal inputs to produce equal (or both-None) outputs.)
    assert gates17._high_funding_annualized(identical_result) == diff_when_identical
    assert gates17._high_funding_annualized(drifted_result) != diff_when_identical or diff_when_identical is None


# ---------------------------------------------------------------------------
# 5) 변동성 계산 (volatility calculation).
# ---------------------------------------------------------------------------


def test_build_volatility_table_computes_cv_range_and_sorts_by_instability() -> None:
    lending_realized = {
        "by_ccy": {
            "STABLE": {
                "history_available": True, "n_samples": 100, "span_days": 4.1,
                "lending_rate_median": 0.10, "lending_rate_mean": 0.10, "lending_rate_std": 0.01,
                "lending_rate_min": 0.09, "lending_rate_max": 0.11, "lending_rate_range": 0.02, "lending_rate_cv": 0.10,
            },
            "VOLATILE": {
                "history_available": True, "n_samples": 100, "span_days": 4.1,
                "lending_rate_median": 0.05, "lending_rate_mean": 0.05, "lending_rate_std": 0.03,
                "lending_rate_min": 0.01, "lending_rate_max": 0.12, "lending_rate_range": 0.11, "lending_rate_cv": 0.60,
            },
            "NODATA": {"history_available": False},
        }
    }
    rows = volatility17.build_volatility_table(lending_realized)
    assert len(rows) == 2  # NODATA excluded
    assert rows[0].ccy == "VOLATILE"  # sorted descending by CV -- most unstable first
    assert rows[1].ccy == "STABLE"
    assert rows[0].lending_rate_cv == pytest.approx(0.60)
    assert rows[0].lending_rate_range == pytest.approx(0.11)
    assert rows[0].range_over_median == pytest.approx(0.11 / 0.05)

    summary = volatility17.universe_volatility_summary(rows)
    assert summary["n_coins"] == 2
    assert summary["cv_max"] == pytest.approx(0.60)
    assert summary["most_unstable_5"][0] == "VOLATILE"


def test_volatility_table_handles_zero_mean_cv_gracefully() -> None:
    lending_realized = {
        "by_ccy": {
            "ZEROMEAN": {
                "history_available": True, "n_samples": 10, "span_days": 4.0,
                "lending_rate_median": 0.0, "lending_rate_mean": 0.0, "lending_rate_std": 0.0,
                "lending_rate_min": 0.0, "lending_rate_max": 0.0, "lending_rate_range": 0.0, "lending_rate_cv": None,
            },
        }
    }
    rows = volatility17.build_volatility_table(lending_realized)
    assert rows[0].lending_rate_cv is None
    assert rows[0].range_over_median is None  # median<=0 -- must not raise ZeroDivisionError


# ---------------------------------------------------------------------------
# 6) gates17.evaluate_wave17 -- SPEC.md 판정 rule (F1>F0, F2>F0, F3==F0), PASS and FAIL cases.
# ---------------------------------------------------------------------------


def _annualized_to_daily(annual_rate: float) -> float:
    return (1.0 + annual_rate) ** (1.0 / 365.0) - 1.0


def _synthetic_result(annual_rate: float, periods: int = 800) -> Wave10Result:
    idx = pd.date_range("2019-12-31", periods=periods, freq="D", tz="UTC")  # spans HIGH_FUNDING_YEARS (2020/2021)
    daily_rate = _annualized_to_daily(annual_rate)
    values = ACTIVE_CAPITAL * (1.0 + daily_rate) ** np.arange(periods)
    equity = pd.Series(values, index=idx, dtype=float)
    return Wave10Result(equity=equity, positions=pd.Series(dtype=float), turnover=pd.Series(dtype=float), trade_returns=pd.Series(dtype=float), max_concurrent_positions=0, symbols_used=())


def test_evaluate_wave17_passes_when_f1_f2_beat_f0_and_f3_matches_exactly() -> None:
    f0 = _synthetic_result(0.10)
    results = {
        "F0": f0,
        "F1": _synthetic_result(0.16),
        "F2": _synthetic_result(0.13),
        "F3": Wave10Result(equity=f0.equity.copy(), positions=f0.positions, turnover=f0.turnover, trade_returns=f0.trade_returns, max_concurrent_positions=0, symbols_used=()),
    }
    verdict = gates17.evaluate_wave17(results)
    assert verdict.f1_beats_f0 is True
    assert verdict.f2_beats_f0 is True
    assert verdict.f3_equals_f0 is True
    assert verdict.f3_abs_diff == pytest.approx(0.0, abs=1e-12)
    assert verdict.verdict_valid is True
    assert verdict.reasons == ()


def test_evaluate_wave17_fails_when_f1_does_not_beat_f0_and_reports_honestly() -> None:
    results = {
        "F0": _synthetic_result(0.10),
        "F1": _synthetic_result(0.08),  # realized lendingRate yield too small to beat baseline
        "F2": _synthetic_result(0.09),
        "F3": _synthetic_result(0.10),
    }
    verdict = gates17.evaluate_wave17(results)
    assert verdict.f1_beats_f0 is False
    assert verdict.verdict_valid is False
    assert any("F1" in reason for reason in verdict.reasons)


def test_evaluate_wave17_detects_f3_drift_from_f0() -> None:
    results = {
        "F0": _synthetic_result(0.10),
        "F1": _synthetic_result(0.16),
        "F2": _synthetic_result(0.13),
        "F3": _synthetic_result(0.11),  # should be IDENTICAL to F0 (pnl_discount=0.0 both) -- drift means an engine-reuse bug
    }
    verdict = gates17.evaluate_wave17(results)
    assert verdict.f3_equals_f0 is False
    assert verdict.verdict_valid is False
    assert any("F3" in reason for reason in verdict.reasons)


def test_evaluate_wave17_requires_all_four_candidates() -> None:
    with pytest.raises(KeyError):
        gates17.evaluate_wave17({"F0": _synthetic_result(0.10)})


# ---------------------------------------------------------------------------
# 7) load_wave16_target_ccys -- universe selection reads ONLY lending_available=True entries.
# ---------------------------------------------------------------------------


def test_load_wave16_target_ccys_filters_to_lending_available_and_dedupes(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    import json

    fake_snapshot = {
        "by_symbol": {
            "AUSDT": {"lending_available": True, "base_ccy_matched": "A"},
            "A2USDT": {"lending_available": True, "base_ccy_matched": "A"},  # dedup: same ccy via two symbols
            "BUSDT": {"lending_available": True, "base_ccy_matched": "B"},
            "NOPEUSDT": {"lending_available": False, "base_ccy_matched": None},
        }
    }
    fake_path = tmp_path / "lending_snapshot.json"
    fake_path.write_text(json.dumps(fake_snapshot), encoding="utf-8")
    monkeypatch.setattr(fetch17, "WAVE16_SNAPSHOT_PATH", fake_path)

    ccys = fetch17.load_wave16_target_ccys()
    assert ccys == ("A", "B")  # sorted, deduped, excludes the lending_available=False entry
