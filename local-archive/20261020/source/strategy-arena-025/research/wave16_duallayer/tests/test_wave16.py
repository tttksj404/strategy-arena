# Wave-16 test suite. Deliberately keeps the fast/deterministic regression tests here on
# SYNTHETIC small markets (same style as research/wave13_liquidity/tests/test_wave13.py's own
# `_two_symbol_synthetic_market` equivalence tests) rather than the real 200-symbol L4 cache --
# the REAL E0-vs-wave13-L4 numeric comparison is instead an integration-level check baked into
# reporting16.py's own `_l4_reproduction_section` (produced by actually running the pipeline, not
# by pytest), so this file stays fast and independent of cache freshness/availability.

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
from research.wave13_liquidity.engine13 import _build_aligned_frames as engine13_build_aligned_frames
from research.wave13_liquidity.engine13 import _run_liquidity_loop as engine13_run_liquidity_loop
from research.wave16_duallayer import fetch_lending, gates16
from research.wave16_duallayer.configs16 import CANDIDATES, CANDIDATE_IDS, L4_CONFIG, funding_only_variant_key, get_candidate
from research.wave16_duallayer.engine16 import DualLayerRunner, _build_aligned_frames_dual, _lending_daily_rate_series, _run_dual_layer_loop, current_snapshot_pick


# ---------------------------------------------------------------------------
# 1) Config registry: frozen 5 candidates, discount table matches SPEC.md exactly.
# ---------------------------------------------------------------------------


def test_candidate_registry_is_frozen_to_five_preregistered_ids() -> None:
    assert CANDIDATE_IDS == ("E0", "E1", "E2", "E3", "E4")
    assert len(CANDIDATES) == 5


def test_discount_table_matches_spec_literally() -> None:
    # SPEC.md 후보 5개 표: (ranking_discount, pnl_discount) per ID.
    expected = {"E0": (0.0, 0.0), "E1": (0.0, 1.0), "E2": (1.0, 1.0), "E3": (0.5, 0.5), "E4": (1.0, 0.0)}
    for candidate_id, (ranking, pnl) in expected.items():
        candidate = get_candidate(candidate_id)
        assert candidate.ranking_lending_discount == pytest.approx(ranking)
        assert candidate.pnl_lending_discount == pytest.approx(pnl)


def test_l4_config_is_inherited_verbatim_not_redefined() -> None:
    # SPEC.md "top200 유니버스(L4 승계)" + "wave-13 실측비용" + "공통 $100/$90/$45/1x/15%/7.5%"
    assert L4_CONFIG.candidate.candidate_id == "L4"
    assert L4_CONFIG.breadth == 200
    assert L4_CONFIG.leg_fraction == pytest.approx(0.50)
    assert L4_CONFIG.candidate.threshold_apr == pytest.approx(0.15)
    assert L4_CONFIG.candidate.top_k == 1


def test_funding_only_variant_key_aliases_e1_to_e0_and_e2_to_e4() -> None:
    # SPEC.md 방법 4's "MC/블록셔플은 펀딩 부분에만 적용" companion -- this is what makes E1's
    # companion literally E0 and E2's companion literally E4, with no per-candidate special case.
    assert funding_only_variant_key(get_candidate("E0")) == (0.0, 0.0)
    assert funding_only_variant_key(get_candidate("E1")) == (get_candidate("E0").ranking_lending_discount, get_candidate("E0").pnl_lending_discount)
    assert funding_only_variant_key(get_candidate("E2")) == (get_candidate("E4").ranking_lending_discount, get_candidate("E4").pnl_lending_discount)
    assert funding_only_variant_key(get_candidate("E4")) == (get_candidate("E4").ranking_lending_discount, get_candidate("E4").pnl_lending_discount)
    # E3's companion is UNIQUE -- not equal to any other candidate's own (ranking, pnl) pair.
    e3_companion = funding_only_variant_key(get_candidate("E3"))
    other_keys = {(c.ranking_lending_discount, c.pnl_lending_discount) for c in CANDIDATES if c.candidate_id != "E3"}
    assert e3_companion not in other_keys


# ---------------------------------------------------------------------------
# 2) Synthetic 2-symbol market (mirrors test_wave13.py's own _two_symbol_synthetic_market).
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
    funding_a = pd.Series(0.0010, index=funding_index, name="funding_rate")  # BTCUSDT: higher funding
    funding_b = pd.Series(0.0006, index=funding_index, name="funding_rate")  # ETHUSDT: lower funding
    return {"BTCUSDT": FundingMarket(spot_a, perp_a, funding_a), "ETHUSDT": FundingMarket(spot_b, perp_b, funding_b)}


def _flat_cost_and_liquidity(index: pd.Index, columns: pd.Index, rate: float = 0.001) -> tuple[pd.DataFrame, pd.DataFrame]:
    cost = pd.DataFrame(rate, index=index, columns=columns)
    liquidity = pd.DataFrame(True, index=index, columns=columns)
    return cost, liquidity


# ---------------------------------------------------------------------------
# 3) Engine-equivalence: zero discounts -> engine16 reproduces engine13 bit-for-bit
#    (proves the generalization is faithful; E0's real-data reproduction of wave13's L4 is then
#    just this same code path pointed at the real cache, done at pipeline-run time).
# ---------------------------------------------------------------------------


def test_engine16_matches_engine13_when_both_discounts_are_zero() -> None:
    markets = _two_symbol_synthetic_market()
    candidate = FundingCandidate("EQTEST16", 1, 0.50, 1)
    leg_fraction = 0.5

    frames13 = engine13_build_aligned_frames(markets, candidate)
    cost13, liquidity13 = _flat_cost_and_liquidity(frames13[0].index, frames13[0].columns)
    result13, cost_total13, eligible13 = engine13_run_liquidity_loop(*frames13, candidate.top_k, leg_fraction, cost13, liquidity13)

    frames16 = _build_aligned_frames_dual(markets, candidate, lending_apr_by_symbol={}, ranking_lending_discount=0.0)
    cost16, liquidity16 = _flat_cost_and_liquidity(frames16[0].index, frames16[0].columns)
    zero_lending = _lending_daily_rate_series(frames16[0].columns, {}, pnl_lending_discount=0.0)
    result16, cost_total16, eligible16 = _run_dual_layer_loop(*frames16, candidate.top_k, leg_fraction, cost16, liquidity16, zero_lending)

    assert result16.equity.tolist() == pytest.approx(result13.equity.tolist(), rel=1e-12)
    assert result16.positions.tolist() == pytest.approx(result13.positions.tolist(), rel=1e-12)
    assert result16.turnover.tolist() == pytest.approx(result13.turnover.tolist(), rel=1e-12)
    assert result16.trade_returns.tolist() == pytest.approx(result13.trade_returns.tolist(), rel=1e-12)
    assert cost_total16 == pytest.approx(cost_total13, rel=1e-12)


# ---------------------------------------------------------------------------
# 4) Lending PnL is purely additive on top of the identical trade selection.
# ---------------------------------------------------------------------------


def test_lending_pnl_discount_adds_expected_daily_compounding_with_same_trades() -> None:
    markets = _two_symbol_synthetic_market()
    candidate = FundingCandidate("PNLTEST", 1, 0.50, 1)  # BTCUSDT always wins top_k=1 (higher funding)
    leg_fraction = 0.5
    lending_apr = {"BTCUSDT": 0.20, "ETHUSDT": 0.0}  # only the traded symbol matters here

    frames = _build_aligned_frames_dual(markets, candidate, lending_apr, ranking_lending_discount=0.0)  # ranking unaffected -- same trades as E0
    cost, liquidity = _flat_cost_and_liquidity(frames[0].index, frames[0].columns)

    zero_lending = _lending_daily_rate_series(frames[0].columns, lending_apr, pnl_lending_discount=0.0)
    result_no_lending, _, _ = _run_dual_layer_loop(*frames, candidate.top_k, leg_fraction, cost, liquidity, zero_lending)

    full_lending = _lending_daily_rate_series(frames[0].columns, lending_apr, pnl_lending_discount=1.0)
    result_with_lending, _, _ = _run_dual_layer_loop(*frames, candidate.top_k, leg_fraction, cost, liquidity, full_lending)

    # Same ranking -> IDENTICAL trade selection (positions/turnover unaffected by lending).
    assert result_with_lending.positions.tolist() == pytest.approx(result_no_lending.positions.tolist(), rel=1e-12)
    assert result_with_lending.turnover.tolist() == pytest.approx(result_no_lending.turnover.tolist(), rel=1e-12)
    # But equity must be HIGHER every day a position is held (BTC lending APR=20% > 0).
    held_days = result_no_lending.positions[result_no_lending.positions > 0.0].index
    assert len(held_days) > 0
    for day in held_days:
        assert result_with_lending.equity.loc[day] > result_no_lending.equity.loc[day]
    # 50% discount must land strictly between 0% and 100% (monotonic in discount -- not claiming
    # exact log-linearity, which the compounding loop does not actually guarantee to machine
    # precision; strict ordering is the property that actually must hold).
    half_lending = _lending_daily_rate_series(frames[0].columns, lending_apr, pnl_lending_discount=0.5)
    result_half, _, _ = _run_dual_layer_loop(*frames, candidate.top_k, leg_fraction, cost, liquidity, half_lending)
    assert result_no_lending.equity.iloc[-1] < result_half.equity.iloc[-1] < result_with_lending.equity.iloc[-1]


# ---------------------------------------------------------------------------
# 5) Ranking discount can flip which symbol clears/wins top_k=1 (the whole point of E2).
# ---------------------------------------------------------------------------


def test_combined_ranking_can_flip_symbol_selection_versus_funding_only() -> None:
    markets = _two_symbol_synthetic_market()
    # BTCUSDT funding APR = 0.0010*3*365 = 109.5%; ETHUSDT = 0.0006*3*365 = 65.7%. A large ETH
    # lending bonus should push combined(ETH) above combined(BTC) even though funding-only never
    # would.
    lending_apr = {"BTCUSDT": 0.0, "ETHUSDT": 5.0}  # 500% -- deliberately huge to force a flip unambiguously
    candidate = FundingCandidate("RANKTEST", 1, 0.50, 1)

    frames_funding_only = _build_aligned_frames_dual(markets, candidate, lending_apr, ranking_lending_discount=0.0)
    score_funding_only = frames_funding_only[5]  # score_frame
    last_row = score_funding_only.dropna(how="all").iloc[-1]
    assert last_row.idxmax() == "BTCUSDT"

    frames_combined = _build_aligned_frames_dual(markets, candidate, lending_apr, ranking_lending_discount=1.0)
    score_combined = frames_combined[5]
    last_row_combined = score_combined.dropna(how="all").iloc[-1]
    assert last_row_combined.idxmax() == "ETHUSDT"


# ---------------------------------------------------------------------------
# 6) DualLayerRunner memoization: E2's funding-only companion key IS E4's own key -- same cache
#    entry, not just equal values.
# ---------------------------------------------------------------------------


def test_runner_memoizes_identically_keyed_variants_as_the_same_object() -> None:
    markets = _two_symbol_synthetic_market()
    candidate = FundingCandidate("MEMOTEST", 1, 0.50, 1)
    config = type(L4_CONFIG)(candidate, 0.5, "fixed", ("BTCUSDT", "ETHUSDT"), None, 12.0, None, None, "test config")
    lending_apr = {"BTCUSDT": 0.10, "ETHUSDT": 0.05}
    runner = DualLayerRunner(config, mapping=None, markets=markets, lending_apr_by_symbol=lending_apr)  # type: ignore[arg-type]

    e2_key = (get_candidate("E2").ranking_lending_discount, get_candidate("E2").pnl_lending_discount)
    e4_key = (get_candidate("E4").ranking_lending_discount, get_candidate("E4").pnl_lending_discount)
    e2_companion_key = funding_only_variant_key(get_candidate("E2"))
    assert e2_companion_key == e4_key

    # monkeypatch cost/liquidity build to avoid needing a real MeasuredCostMapping
    def _fake_cost_and_liquidity(stress_multiplier: float) -> tuple[pd.DataFrame, pd.DataFrame]:
        frames = runner._frames(0.0)
        return _flat_cost_and_liquidity(frames[0].index, frames[0].columns)

    runner._cost_and_liquidity = _fake_cost_and_liquidity  # type: ignore[method-assign]

    companion_run = runner.run_variant(e2_companion_key[0], e2_companion_key[1], stress_multiplier=1.0)
    e4_run = runner.run_variant(e4_key[0], e4_key[1], stress_multiplier=1.0)
    assert companion_run is e4_run  # literally the same cached tuple, not just equal


# ---------------------------------------------------------------------------
# 7) gates16: exchange-separation structural ratio matches wave14 M6/M7's own precedent (55%).
# ---------------------------------------------------------------------------


def test_exchange_separation_matches_wave14_m6_m7_precedent() -> None:
    assert gates16.exchange_separation_remaining_fraction() == pytest.approx(0.55, abs=1e-9)


# ---------------------------------------------------------------------------
# 8) gates16.evaluate_structure_validity: SPEC.md 판정 rule, both a PASS and a FAIL (E4<E0) case.
# ---------------------------------------------------------------------------


def _annualized_to_daily(annual_rate: float) -> float:
    return (1.0 + annual_rate) ** (1.0 / 365.0) - 1.0


def _synthetic_combined_result(annual_rate: float, periods: int = 800) -> Wave10Result:
    idx = pd.date_range("2019-12-31", periods=periods, freq="D", tz="UTC")  # spans 2020 AND 2021 (regime.HIGH_FUNDING_YEARS)
    daily_rate = _annualized_to_daily(annual_rate)
    values = ACTIVE_CAPITAL * (1.0 + daily_rate) ** np.arange(periods)
    equity = pd.Series(values, index=idx, dtype=float)
    return Wave10Result(equity=equity, positions=pd.Series(dtype=float), turnover=pd.Series(dtype=float), trade_returns=pd.Series(dtype=float), max_concurrent_positions=0, symbols_used=())


def test_structure_validity_passes_when_e2_e3_beat_e0_and_e4_ties_e0() -> None:
    results = {
        "E0": _synthetic_combined_result(0.10),
        "E1": _synthetic_combined_result(0.10),
        "E2": _synthetic_combined_result(0.20),
        "E3": _synthetic_combined_result(0.15),
        "E4": _synthetic_combined_result(0.10),  # exact tie -- >= must still pass
    }
    verdict = gates16.evaluate_structure_validity(results)
    assert verdict.e2_beats_e0 is True
    assert verdict.e3_beats_e0 is True
    assert verdict.e4_at_least_e0 is True
    assert verdict.structure_valid is True
    assert verdict.reasons == ()


def test_structure_validity_rejects_when_e4_falls_below_e0() -> None:
    results = {
        "E0": _synthetic_combined_result(0.10),
        "E1": _synthetic_combined_result(0.10),
        "E2": _synthetic_combined_result(0.20),
        "E3": _synthetic_combined_result(0.15),
        "E4": _synthetic_combined_result(0.05),  # SPEC.md: E4 < E0 -> 합산 랭킹 기각
    }
    verdict = gates16.evaluate_structure_validity(results)
    assert verdict.e4_at_least_e0 is False
    assert verdict.structure_valid is False
    assert any("E4" in reason for reason in verdict.reasons)


def test_structure_validity_requires_all_five_candidates_present() -> None:
    with pytest.raises(KeyError):
        gates16.evaluate_structure_validity({"E0": _synthetic_combined_result(0.10)})


# ---------------------------------------------------------------------------
# 9) fetch_lending: outlier exclusion, symbol<->ccy join, no-network helpers.
# ---------------------------------------------------------------------------


def test_split_outliers_excludes_beth_and_extreme_apr() -> None:
    rows = [
        {"ccy": "BTC", "avg_rate": 0.005, "est_rate": 0.005, "pre_rate": 0.005},
        {"ccy": "BETH", "avg_rate": 3.65, "est_rate": 3.65, "pre_rate": 3.65},  # SPEC.md 발견: hardcoded outlier
        {"ccy": "MADE_UP", "avg_rate": 1.50, "est_rate": 1.50, "pre_rate": 1.50},  # general cap (>100% APR)
        {"ccy": "THETA", "avg_rate": 0.2252, "est_rate": 0.189, "pre_rate": 0.189},
    ]
    kept, excluded = fetch_lending.split_outliers(rows)
    kept_ccys = {row["ccy"] for row in kept}
    excluded_ccys = {row["ccy"] for row in excluded}
    assert kept_ccys == {"BTC", "THETA"}
    assert excluded_ccys == {"BETH", "MADE_UP"}


def test_base_ccy_candidates_and_resolve_lending_apr_fallback() -> None:
    assert fetch_lending.base_ccy_candidates("BTCUSDT")[0] == "BTC"
    assert fetch_lending.base_ccy_candidates("1INCHUSDT")[0] == "1INCH"
    candidates = fetch_lending.base_ccy_candidates("1000SHIBUSDT")
    assert "1000SHIB" in candidates and "SHIB" in candidates

    lending_by_ccy = {"BTC": 0.005, "SHIB": 0.12}
    rate, matched = fetch_lending.resolve_lending_apr("BTCUSDT", lending_by_ccy)
    assert rate == pytest.approx(0.005) and matched == "BTC"
    rate2, matched2 = fetch_lending.resolve_lending_apr("1000SHIBUSDT", lending_by_ccy)  # falls back past the exact-strip miss
    assert rate2 == pytest.approx(0.12) and matched2 == "SHIB"
    rate3, matched3 = fetch_lending.resolve_lending_apr("NOPEUSDT", lending_by_ccy)
    assert rate3 is None and matched3 is None


# ---------------------------------------------------------------------------
# 10) current_snapshot_pick (method a) -- pure cross-section, no engine loop.
# ---------------------------------------------------------------------------


def _fake_lending_snapshot() -> dict:
    return {
        "by_symbol": {
            "BTCUSDT": {"lending_apr": 0.005, "lending_available": True, "bitget_funding_apr_current": 0.05},
            "THETAUSDT": {"lending_apr": 0.20, "lending_available": True, "bitget_funding_apr_current": 0.11},
            "NODATAUSDT": {"lending_apr": 0.0, "lending_available": False, "bitget_funding_apr_current": 0.03},
            "NOFUNDUSDT": {"lending_apr": 0.0, "lending_available": False, "bitget_funding_apr_current": None},
        }
    }


def test_current_snapshot_pick_funding_only_stays_dormant_below_threshold() -> None:
    snapshot = _fake_lending_snapshot()
    pick = current_snapshot_pick(ranking_lending_discount=0.0, entry_threshold_apr=0.15, lending_snapshot=snapshot)
    assert pick["universe_n"] == 3  # NOFUNDUSDT excluded (no current funding data)
    assert pick["n_clearing_threshold"] == 0  # every funding_apr_current alone is < 15%
    assert pick["top_pick"] is None


def test_current_snapshot_pick_combined_clears_threshold_via_lending() -> None:
    snapshot = _fake_lending_snapshot()
    pick = current_snapshot_pick(ranking_lending_discount=1.0, entry_threshold_apr=0.15, lending_snapshot=snapshot)
    # THETAUSDT: 0.11 + 0.20 = 0.31 > 0.15; BTCUSDT: 0.05+0.005=0.055 < 0.15; NODATAUSDT: 0.03+0(unavailable)=0.03 < 0.15
    assert pick["n_clearing_threshold"] == 1
    assert pick["top_pick"]["symbol"] == "THETAUSDT"
    assert pick["top_pick"]["ranking_score"] == pytest.approx(0.31)
    # unavailable lending must fail closed to 0.0, never invent a rate
    nodata_row = next(row for row in pick["top_5_by_score"] if row["symbol"] == "NODATAUSDT")
    assert nodata_row["lending_apr"] == pytest.approx(0.0)
    assert nodata_row["lending_available"] is False
