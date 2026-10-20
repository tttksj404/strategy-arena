from __future__ import annotations

from pathlib import Path
import sys

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[3]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

import pandas as pd  # noqa: PANDAS_OK
import pytest

from research.wave1.fam_funding import FundingCandidate, FundingMarket
from research.wave10_carry100.configs import Wave10Config
from research.wave10_carry100.engine import ACTIVE_CAPITAL, cost_rate as wave10_cost_rate
from research.wave10_carry100.engine import run_fixed_fraction_portfolio as wave10_run_fixed_fraction_portfolio
from research.wave12_frontier import costs_tiered
from research.wave12_frontier import universe_frontier as uf
from research.wave12_frontier.configs12 import CONFIG_IDS, CONFIGS
from research.wave12_frontier.engine12 import _build_aligned_frames, _run_frontier_loop
from research.wave12_frontier.gates12 import gross_usdt, leg_usdt
from research.wave12_frontier.reporting12 import _gate_table


# ---------------------------------------------------------------------------
# 1) Config registry integrity: frozen 7 candidates, common-fixed $ figures match SPEC.md.
# ---------------------------------------------------------------------------


def test_config_registry_is_frozen_to_seven_preregistered_ids() -> None:
    assert CONFIG_IDS == ("U0", "U1", "U2", "U3", "U4", "U5", "U6")
    assert len(CONFIGS) == 7


def test_config_breadth_and_history_match_spec_table() -> None:
    expected = {
        "U0": (100, 12.0),
        "U1": (150, 12.0),
        "U2": (200, 12.0),
        "U3": (None, 12.0),
        "U4": (100, 6.0),
        "U5": (200, 6.0),
        "U6": (200, 3.0),
    }
    assert {c.candidate.candidate_id for c in CONFIGS} == set(expected)
    for config in CONFIGS:
        breadth, months = expected[config.candidate.candidate_id]
        assert config.breadth == breadth
        assert config.history_months == pytest.approx(months)


def test_config_leg_and_gross_dollar_figures_match_common_fixed_spec() -> None:
    # SPEC.md "공통 고정": 1쌍 $45/$45 @ $90 active, gross 1.0x, 진입 15%APR/청산 7.5% --
    # identical across all seven configs; only breadth/history vary (tested above).
    for config in CONFIGS:
        assert leg_usdt(config) == pytest.approx(45.0)
        assert gross_usdt(config) == pytest.approx(90.0)
        assert gross_usdt(config) / ACTIVE_CAPITAL == pytest.approx(1.0)
        assert config.candidate.top_k == 1
        assert config.candidate.threshold_apr == pytest.approx(0.15)
        assert config.candidate.threshold_apr / 2.0 == pytest.approx(0.075)  # carry_position's built-in exit = threshold/2


# ---------------------------------------------------------------------------
# 2) Tiered slippage assignment (SPEC.md's literal table).
# ---------------------------------------------------------------------------


def test_slippage_bp_for_rank_matches_spec_table() -> None:
    assert costs_tiered.slippage_bp_for_rank("BTCUSDT", 999) == costs_tiered.MAJOR_SLIPPAGE_BP  # majors override rank entirely
    assert costs_tiered.slippage_bp_for_rank("ETHUSDT", 5) == costs_tiered.MAJOR_SLIPPAGE_BP
    assert costs_tiered.slippage_bp_for_rank("SOLUSDT", 3) == 3.0  # SOL is NOT a hardcoded major under the tiered model (unlike the old flat model)
    assert costs_tiered.slippage_bp_for_rank("XUSDT", 1) == 3.0
    assert costs_tiered.slippage_bp_for_rank("XUSDT", 50) == 3.0
    assert costs_tiered.slippage_bp_for_rank("XUSDT", 51) == 6.0
    assert costs_tiered.slippage_bp_for_rank("XUSDT", 100) == 6.0
    assert costs_tiered.slippage_bp_for_rank("XUSDT", 101) == 10.0
    assert costs_tiered.slippage_bp_for_rank("XUSDT", 200) == 10.0
    assert costs_tiered.slippage_bp_for_rank("XUSDT", 201) == 20.0
    assert costs_tiered.slippage_bp_for_rank("XUSDT", 5000) == 20.0
    assert costs_tiered.slippage_bp_for_rank("XUSDT", None) == 20.0  # fail-closed, not fail-cheap
    assert costs_tiered.slippage_bp_for_rank("XUSDT", float("nan")) == 20.0


def test_bp_frame_from_ranks_matches_scalar_lookup_vectorized() -> None:
    idx = pd.date_range("2026-01-01", periods=3, freq="D", tz="UTC")
    ranks = pd.DataFrame({"BTCUSDT": [1.0, 1.0, 1.0], "XUSDT": [10.0, 60.0, 300.0]}, index=idx)
    bp = costs_tiered.bp_frame_from_ranks(ranks, ("BTCUSDT", "XUSDT"))
    assert bp["BTCUSDT"].tolist() == [1.0, 1.0, 1.0]
    assert bp["XUSDT"].tolist() == [3.0, 6.0, 20.0]
    # a symbol requested but absent from the rank frame entirely (e.g. outside the
    # tier-reference pool) must fail closed to the tail tier, never to 0/NaN.
    bp_missing = costs_tiered.bp_frame_from_ranks(ranks, ("BTCUSDT", "ZZZUSDT"))
    assert bp_missing["ZZZUSDT"].tolist() == [20.0, 20.0, 20.0]


# ---------------------------------------------------------------------------
# 3) Point-in-time rank: no lookahead. A future volume spike must never move an earlier
#    day's rank (this is the exact bug class SPEC.md calls out: "미래 볼륨 사용 금지").
# ---------------------------------------------------------------------------


def test_known_avg_is_previous_day_rolling_mean_not_same_day() -> None:
    idx = pd.date_range("2026-01-01", periods=35, freq="D", tz="UTC")
    volumes = pd.Series([float(i) * 100_000.0 for i in range(1, 36)], index=idx)
    frame = pd.DataFrame({"AAAUSDT": volumes})
    raw_rolling = costs_tiered.rolling_trailing_avg_volume(frame)
    known = costs_tiered.point_in_time_known_avg(frame)
    for i in range(30, 35):
        assert known["AAAUSDT"].iloc[i] == pytest.approx(raw_rolling["AAAUSDT"].iloc[i - 1])
        assert known["AAAUSDT"].iloc[i] != pytest.approx(raw_rolling["AAAUSDT"].iloc[i])


def test_point_in_time_rank_unaffected_by_future_volume_spike() -> None:
    idx = pd.date_range("2026-01-01", periods=80, freq="D", tz="UTC")
    base = pd.DataFrame(
        {"AAAUSDT": [1_000_000.0] * 80, "BBBUSDT": [2_000_000.0] * 80, "CCCUSDT": [500_000.0] * 80},
        index=idx,
    )
    spiked = base.copy()
    spiked.loc[idx[60] :, "AAAUSDT"] = 50_000_000.0  # a huge future spike starting day 60

    ranks_base = costs_tiered.point_in_time_ranks(costs_tiered.point_in_time_known_avg(base))
    ranks_spiked = costs_tiered.point_in_time_ranks(costs_tiered.point_in_time_known_avg(spiked))

    before_spike = idx[:60]
    pd.testing.assert_frame_equal(ranks_base.loc[before_spike], ranks_spiked.loc[before_spike])
    # sanity: the spike setup is meaningful -- AAAUSDT's rank DOES improve once its own
    # trailing window has actually absorbed it, so the "no difference before" result
    # above isn't just because ranks never respond to volume at all.
    assert ranks_spiked.loc[idx[-1], "AAAUSDT"] < ranks_base.loc[idx[-1], "AAAUSDT"]


# ---------------------------------------------------------------------------
# 4) Liquidity floor ($2M point-in-time 30d avg quote_volume).
# ---------------------------------------------------------------------------


def test_liquidity_floor_flags_thin_symbols_and_respects_min_periods() -> None:
    idx = pd.date_range("2026-01-01", periods=40, freq="D", tz="UTC")
    frame = pd.DataFrame(
        {"LIQUSDT": [5_000_000.0] * 40, "THINUSDT": [1_000_000.0] * 40},
        index=idx,
    )
    _, liquidity_ok = costs_tiered.build_cost_and_liquidity_frames(frame, ("LIQUSDT", "THINUSDT"))
    assert not liquidity_ok.iloc[:30].to_numpy().any()  # no valid 30d window yet -- fail closed, not "assume liquid"
    assert bool(liquidity_ok["LIQUSDT"].iloc[35])
    assert not bool(liquidity_ok["THINUSDT"].iloc[35])


# ---------------------------------------------------------------------------
# 5) Cost-rate arithmetic: stress_multiplier scales ONLY slippage, never the maker fee.
# ---------------------------------------------------------------------------


def test_cost_rate_from_bp_doubles_only_slippage_under_stress() -> None:
    maker_component = 2.0 * 0.0002  # research.wave2.funding.W2_MAKER_FEE_RATE, both legs
    base = costs_tiered.cost_rate_from_bp(3.0, stress_multiplier=1.0)
    stressed = costs_tiered.cost_rate_from_bp(3.0, stress_multiplier=2.0)
    assert base == pytest.approx(maker_component + 2.0 * 3.0 * 0.0001)
    assert stressed == pytest.approx(maker_component + 2.0 * 3.0 * 0.0001 * 2.0)
    assert stressed - base == pytest.approx(2.0 * 3.0 * 0.0001)  # exactly one extra slippage unit
    assert (stressed - maker_component) == pytest.approx(2.0 * (base - maker_component))  # slippage exactly doubled
    # the maker-fee floor itself is identical at any stress level
    assert costs_tiered.cost_rate_from_bp(0.0, stress_multiplier=1.0) == pytest.approx(maker_component)
    assert costs_tiered.cost_rate_from_bp(0.0, stress_multiplier=2.0) == pytest.approx(maker_component)


# ---------------------------------------------------------------------------
# 6) History-requirement filter + breadth cap (research.wave12_frontier.universe_frontier).
# ---------------------------------------------------------------------------


def _synthetic_pool() -> dict:
    return {
        "symbols": {
            "AAAUSDT": {"ok": True, "history_start": "2024-01-01T00:00:00+00:00", "reference_volume_30d_usdt": 100.0},  # clears 12mo
            "BBBUSDT": {"ok": True, "history_start": "2026-02-01T00:00:00+00:00", "reference_volume_30d_usdt": 500.0},  # clears 3mo/6mo only
            "CCCUSDT": {"ok": True, "history_start": "2019-01-01T00:00:00+00:00", "reference_volume_30d_usdt": 50.0},  # clears everything, lowest volume
            "DDDUSDT": {"ok": False, "history_start": "2019-01-01T00:00:00+00:00", "reference_volume_30d_usdt": 999.0},  # integrity-failed: excluded regardless of volume
        }
    }


def test_symbols_for_breadth_history_filters_by_history_and_integrity() -> None:
    pool = _synthetic_pool()
    twelve_mo = uf.symbols_for_breadth_history(pool, None, 12.0)
    assert set(twelve_mo) == {"AAAUSDT", "CCCUSDT"}
    assert twelve_mo[0] == "AAAUSDT"  # ranked by reference_volume_30d_usdt descending

    three_mo = uf.symbols_for_breadth_history(pool, None, 3.0)
    assert set(three_mo) == {"AAAUSDT", "BBBUSDT", "CCCUSDT"}  # BBBUSDT now clears the looser floor

    top1 = uf.symbols_for_breadth_history(pool, 1, 12.0)
    assert top1 == ("AAAUSDT",)  # DDDUSDT never appears despite having the highest raw volume -- integrity gates first


def test_breadth_cap_is_monotonic_subset_for_same_history_floor() -> None:
    pool = {
        "symbols": {
            f"S{i}USDT": {"ok": True, "history_start": "2020-01-01T00:00:00+00:00", "reference_volume_30d_usdt": float(1000 - i)}
            for i in range(250)
        }
    }
    top100 = set(uf.symbols_for_breadth_history(pool, 100, 12.0))
    top150 = set(uf.symbols_for_breadth_history(pool, 150, 12.0))
    top200 = set(uf.symbols_for_breadth_history(pool, 200, 12.0))
    unlimited = set(uf.symbols_for_breadth_history(pool, None, 12.0))
    assert top100 <= top150 <= top200 <= unlimited
    assert (len(top100), len(top150), len(top200), len(unlimited)) == (100, 150, 200, 250)


# ---------------------------------------------------------------------------
# 7) Engine equivalence + liquidity-floor integration proof.
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
    funding_a = pd.Series(0.0010, index=funding_index, name="funding_rate")  # ~109.5% annualized -- the higher-score symbol
    funding_b = pd.Series(0.0006, index=funding_index, name="funding_rate")  # ~65.7% annualized
    return {"BTCUSDT": FundingMarket(spot_a, perp_a, funding_a), "ETHUSDT": FundingMarket(spot_b, perp_b, funding_b)}


def test_engine12_matches_wave10_when_cost_is_flat_and_liquidity_always_ok() -> None:
    markets = _two_symbol_synthetic_market()
    candidate = FundingCandidate("EQTEST", 1, 0.50, 1)
    leg_fraction = 0.5

    wave10_result = wave10_run_fixed_fraction_portfolio(markets, Wave10Config(candidate, leg_fraction, "equivalence test"))

    frames = _build_aligned_frames(markets, candidate)
    spot_open_frame = frames[0]
    flat_cost = pd.DataFrame(
        {symbol: [wave10_cost_rate(symbol)] * len(spot_open_frame.index) for symbol in spot_open_frame.columns},
        index=spot_open_frame.index,
    )
    always_liquid = pd.DataFrame(True, index=spot_open_frame.index, columns=spot_open_frame.columns)
    wave12_result, total_cost, eligible_counts = _run_frontier_loop(*frames, candidate.top_k, leg_fraction, flat_cost, always_liquid)

    assert wave12_result.equity.tolist() == pytest.approx(wave10_result.equity.tolist(), rel=1e-12)
    assert wave12_result.positions.tolist() == pytest.approx(wave10_result.positions.tolist(), rel=1e-12)
    assert wave12_result.turnover.tolist() == pytest.approx(wave10_result.turnover.tolist(), rel=1e-12)
    assert wave12_result.trade_returns.tolist() == pytest.approx(wave10_result.trade_returns.tolist(), rel=1e-12)
    assert wave12_result.max_concurrent_positions == wave10_result.max_concurrent_positions
    assert total_cost > 0.0
    assert len(eligible_counts) == len(spot_open_frame.index)
    assert eligible_counts.max() <= 2.0  # never more than the 2 symbols in this fixture


def test_engine12_liquidity_floor_makes_illiquid_symbol_behave_as_if_absent() -> None:
    markets = _two_symbol_synthetic_market()
    candidate = FundingCandidate("LIQTEST", 1, 0.50, 1)

    frames_both = _build_aligned_frames(markets, candidate)
    spot_open_both = frames_both[0]
    flat_cost_both = pd.DataFrame(0.001, index=spot_open_both.index, columns=spot_open_both.columns)
    liquidity_btc_illiquid = pd.DataFrame(True, index=spot_open_both.index, columns=spot_open_both.columns)
    liquidity_btc_illiquid["BTCUSDT"] = False  # BTCUSDT has the higher funding score -- would normally win top_k=1
    restricted, _, restricted_eligible = _run_frontier_loop(*frames_both, candidate.top_k, 0.5, flat_cost_both, liquidity_btc_illiquid)

    eth_only_markets = {"ETHUSDT": markets["ETHUSDT"]}
    frames_eth_only = _build_aligned_frames(eth_only_markets, candidate)
    spot_open_eth = frames_eth_only[0]
    flat_cost_eth = pd.DataFrame(0.001, index=spot_open_eth.index, columns=spot_open_eth.columns)
    liquidity_eth_only = pd.DataFrame(True, index=spot_open_eth.index, columns=spot_open_eth.columns)
    eth_only_result, _, _ = _run_frontier_loop(*frames_eth_only, candidate.top_k, 0.5, flat_cost_eth, liquidity_eth_only)

    assert restricted.equity.tolist() == pytest.approx(eth_only_result.equity.tolist(), rel=1e-12)
    assert restricted_eligible.max() <= 1.0  # BTCUSDT never counted as eligible on any day


# ---------------------------------------------------------------------------
# 8) Promotion display: overall gate FAIL must never render as "promoted" even when the
#    config's raw return number beats U0's bar (regression test for a real bug caught
#    during this wave's own execution: reporting12's registry/gate-table once showed
#    "YES"/승격 for every config whose promo["promoted"] flag was True, without checking
#    gates["overall"] -- SPEC.md's rule is a conjunction of BOTH, not either alone).
# ---------------------------------------------------------------------------


def _minimal_gate_payload(overall: str, promoted: bool, is_baseline: bool = False) -> dict:
    return {
        "gates": {
            "gate_s1": {"status": "PASS", "leverage_multiplier_of_active_capital": 1.0},
            "gate_s2": {"status": "PASS", "p05": 120.0, "ruin_probability": 0.0},
            "gate_s3": {"status": "PASS" if overall == "PASS" else "FAIL", "mdd_p95": 0.05 if overall == "PASS" else 0.15},
            "gate_s4": {"status": "PASS", "leg_usdt_nominal": 45.0},
            "gate_s5": {"status": "PASS", "stress_high_funding_annualized": 0.05},
            "overall": overall,
            "failure_reasons": [] if overall == "PASS" else ["MDD초과"],
            "promotion": {
                "high_funding_mean_annualized_return": 0.20 if promoted else 0.10,
                "high_funding_bar": None if is_baseline else 0.15,
                "high_funding_ok": promoted,
                "is_baseline": is_baseline,
                "promoted": promoted,
            },
        }
    }


def test_gate_table_never_shows_promoted_when_overall_gate_failed() -> None:
    # U1 is the adversarial case: beats U0's return bar (promoted=True) but its own S3
    # gate FAILED -- must render as "no", never "YES".
    payloads = {"U0": _minimal_gate_payload("FAIL", False, is_baseline=True)}
    for candidate_id in CONFIG_IDS:
        if candidate_id == "U0":
            continue
        payloads[candidate_id] = _minimal_gate_payload("FAIL" if candidate_id == "U1" else "PASS", True)
    lines = _gate_table(payloads)
    u1_row = next(line for line in lines if line.startswith("| U1 |"))
    assert " no " in u1_row or u1_row.strip().endswith("|")  # sanity: row exists
    assert "| YES |" not in u1_row, f"U1 has overall=FAIL but was rendered as promoted: {u1_row!r}"
    assert "| no |" in u1_row
