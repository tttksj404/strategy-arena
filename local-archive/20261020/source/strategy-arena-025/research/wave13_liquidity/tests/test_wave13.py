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
from research.wave10_carry100.configs import Wave10Config
from research.wave10_carry100.engine import ACTIVE_CAPITAL, Wave10Result, cost_rate as wave10_cost_rate
from research.wave10_carry100.engine import run_fixed_fraction_portfolio as wave10_run_fixed_fraction_portfolio
from research.wave13_liquidity import collect_spreads
from research.wave13_liquidity import costs_measured as cm
from research.wave13_liquidity import gates13
from research.wave13_liquidity.configs13 import CONFIG_IDS, CONFIGS, get_config
from research.wave13_liquidity.engine13 import _build_aligned_frames, _run_liquidity_loop
from research.wave13_liquidity.gates13 import gross_usdt, leg_usdt


# ---------------------------------------------------------------------------
# 1) Config registry integrity: frozen 5 candidates, common-fixed $ figures match SPEC.md.
# ---------------------------------------------------------------------------


def test_config_registry_is_frozen_to_five_preregistered_ids() -> None:
    assert CONFIG_IDS == ("L1", "L2", "L3", "L4", "L5")
    assert len(CONFIGS) == 5


def test_config_common_fixed_dollar_figures_match_spec() -> None:
    # SPEC.md "공통 고정": 델타중립 2레그, 1x, 1쌍 $45/$45 @ $90 active, 진입 15%APR/청산 7.5% --
    # identical across all five configs; only the universe filter varies.
    for config in CONFIGS:
        assert leg_usdt(config) == pytest.approx(45.0)
        assert gross_usdt(config) == pytest.approx(90.0)
        assert gross_usdt(config) / ACTIVE_CAPITAL == pytest.approx(1.0)
        assert config.candidate.top_k == 1
        assert config.candidate.threshold_apr == pytest.approx(0.15)
        assert config.candidate.threshold_apr / 2.0 == pytest.approx(0.075)  # carry_position's built-in exit = threshold/2


def test_l1_universe_is_the_literal_fixed_pair_not_a_breadth_cap() -> None:
    l1 = get_config("L1")
    assert l1.universe_kind == "fixed"
    assert l1.fixed_symbols == ("BTCUSDT", "ETHUSDT")
    l5 = get_config("L5")
    assert l5.universe_kind == "dynamic"
    assert l5.dynamic_volume_floor_usdt == pytest.approx(20_000_000.0)
    assert l5.dynamic_slippage_cap_bp == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# 2) walk_cost_bp calculation (SPEC.md 작업1) + $45-quote-consumption classification.
# ---------------------------------------------------------------------------


def test_walk_cost_bp_full_fill_at_level_one_equals_half_spread() -> None:
    # $45 order fully absorbed by the best ask alone -> execution price IS ask1, so
    # walk_cost_bp must equal half_spread_bp exactly (both measure "distance from mid to
    # ask1"), matching collect_spreads.py's own docstring claim.
    ask_levels = [(100.5, 10.0)]  # $1005 available at level 1, far more than $45
    mid = 100.0
    walk_cost_bp, insufficient, filled_fraction = collect_spreads.compute_walk_cost_bp(ask_levels, mid, order_size_usdt=45.0)
    half_spread_bp = (100.5 - 99.5) / 2.0 / mid * 1.0e4
    assert walk_cost_bp == pytest.approx(50.0)
    assert walk_cost_bp == pytest.approx(half_spread_bp)
    assert insufficient is False
    assert filled_fraction == pytest.approx(1.0)


def test_walk_cost_bp_walks_multiple_levels_and_flags_insufficient_depth() -> None:
    # Only $30.30 total quoted across 3 levels -- a $45 order cannot be filled from the
    # book alone. The conservative fallback fills the $14.70 shortfall at the WORST quoted
    # level's price (102.0), never assumes a cheaper fill than what was actually observed.
    ask_levels = [(100.0, 0.10), (101.0, 0.10), (102.0, 0.10)]  # $10.00, $10.10, $10.20
    mid = 99.95
    walk_cost_bp, insufficient, filled_fraction = collect_spreads.compute_walk_cost_bp(ask_levels, mid, order_size_usdt=45.0)
    real_filled_usdt = 10.0 + 10.1 + 10.2
    shortfall = 45.0 - real_filled_usdt
    expected_weighted_sum = 100.0 * 10.0 + 101.0 * 10.1 + 102.0 * 10.2 + 102.0 * shortfall
    expected_avg_price = expected_weighted_sum / 45.0
    expected_bp = (expected_avg_price - mid) / mid * 1.0e4
    assert insufficient is True
    assert walk_cost_bp == pytest.approx(expected_bp)
    assert filled_fraction == pytest.approx(real_filled_usdt / 45.0)
    assert walk_cost_bp > (100.0 - mid) / mid * 1.0e4  # strictly worse than a level-1-only half-spread would suggest


def test_walk_cost_bp_empty_book_is_treated_as_undefined_not_zero_cost() -> None:
    walk_cost_bp, insufficient, filled_fraction = collect_spreads.compute_walk_cost_bp([], mid=100.0, order_size_usdt=45.0)
    assert insufficient is True
    assert filled_fraction == 0.0
    assert not np.isfinite(walk_cost_bp) or walk_cost_bp != walk_cost_bp  # NaN, not silently 0.0


def test_build_target_ranks_includes_explicit_anchors_and_clears_minimum() -> None:
    ranks = collect_spreads.build_target_ranks(total=500, minimum=60)
    assert len(ranks) >= 60
    assert len(ranks) == len(set(ranks))  # no duplicates
    assert ranks == sorted(ranks)
    for anchor in (1, 2, 5, 10, 20, 35, 50, 75, 100, 125, 150, 175, 200, 250, 300, 350):
        assert anchor in ranks
    assert all(1 <= rank <= 500 for rank in ranks)


# ---------------------------------------------------------------------------
# 3) Mapping-function monotonicity (SPEC.md 작업1b) -- the fitted function must be
#    monotonic even though the raw measured data is deliberately NOT (mirrors the real
#    AMCUSDT anomaly this wave found: a mid-rank symbol priced worse than several
#    lower-rank/lower-volume neighbors).
# ---------------------------------------------------------------------------


def _synthetic_payload_with_anomaly() -> dict:
    volumes = [1.0e5, 2.0e5, 5.0e5, 1.0e6, 2.0e6, 5.0e6, 1.0e7, 2.0e7, 5.0e7, 1.0e8, 2.0e8, 5.0e8, 1.0e9]
    slippage = [30.0, 22.0, 18.0, 14.0, 10.0, 60.0, 6.0, 4.0, 3.0, 2.0, 1.0, 0.5, 0.05]  # index 5 (5e6) is a deliberate anomaly, worse than every cheaper/thinner neighbor around it
    measurements = [{"usdt_volume_24h": v, "effective_slippage_bp": s} for v, s in zip(volumes, slippage)]
    return {"measurements": measurements, "collected_at_utc": "test-fixture"}


def test_fitted_mapping_is_monotonic_despite_non_monotonic_raw_data() -> None:
    payload = _synthetic_payload_with_anomaly()
    mapping = cm.fit_mapping(payload, n_buckets=6)
    assert len(mapping.anchor_bp) >= 2
    diffs = np.diff(mapping.anchor_bp)
    assert (diffs <= 1e-9).all(), f"fitted anchors are not non-increasing: {mapping.anchor_bp}"
    # the anomaly's raw 60.0bp must be smoothed DOWN by the bucket-median + isotonic fit,
    # not silently reproduced as a mapped value at that volume level.
    anomaly_volume = 5.0e6
    mapped_bp_at_anomaly = cm.slippage_bp_for_volume(anomaly_volume, mapping)
    assert mapped_bp_at_anomaly < 60.0


def test_slippage_bp_for_volume_scalar_matches_frame_vectorized_lookup() -> None:
    payload = _synthetic_payload_with_anomaly()
    mapping = cm.fit_mapping(payload, n_buckets=6)
    idx = pd.date_range("2026-01-01", periods=4, freq="D", tz="UTC")
    known_avg = pd.DataFrame({"AAAUSDT": [1.5e6, 3.0e7, np.nan, -5.0], "BBBUSDT": [9.0e9, 1.0, 4.0e5, np.nan]}, index=idx)
    bp_frame = cm.bp_frame_from_known_avg(known_avg, mapping)
    for column in known_avg.columns:
        for timestamp in known_avg.index:
            expected = cm.slippage_bp_for_volume(known_avg.loc[timestamp, column], mapping)
            assert bp_frame.loc[timestamp, column] == pytest.approx(expected)
    # fail-closed: NaN and non-positive volume both map to the worst (most expensive) anchor
    assert bp_frame.loc[idx[2], "AAAUSDT"] == pytest.approx(mapping.worst_bp)
    assert bp_frame.loc[idx[3], "AAAUSDT"] == pytest.approx(mapping.worst_bp)
    assert bp_frame.loc[idx[3], "BBBUSDT"] == pytest.approx(mapping.worst_bp)


def test_cost_rate_from_bp_stress_multiplier_scales_only_slippage() -> None:
    maker_component = 2.0 * 0.0002  # research.wave2.funding.W2_MAKER_FEE_RATE, both legs
    base = cm.cost_rate_from_bp(4.0, stress_multiplier=1.0)
    stressed = cm.cost_rate_from_bp(4.0, stress_multiplier=3.0)  # SPEC.md S5: x3 (not wave12's x2)
    assert base == pytest.approx(maker_component + 2.0 * 4.0 * 0.0001)
    assert stressed == pytest.approx(maker_component + 2.0 * 4.0 * 0.0001 * 3.0)
    assert (stressed - maker_component) == pytest.approx(3.0 * (base - maker_component))  # slippage exactly tripled
    assert cm.cost_rate_from_bp(0.0, stress_multiplier=3.0) == pytest.approx(maker_component)  # maker floor itself is stress-invariant


# ---------------------------------------------------------------------------
# 4) Point-in-time correctness (no lookahead) for both the plain rolling average and L5's
#    combined dynamic filter -- the volume-domain analogue of test_wave12.py's rank version.
# ---------------------------------------------------------------------------


def test_known_avg_is_previous_day_rolling_mean_not_same_day() -> None:
    idx = pd.date_range("2026-01-01", periods=35, freq="D", tz="UTC")
    volumes = pd.Series([float(i) * 100_000.0 for i in range(1, 36)], index=idx)
    frame = pd.DataFrame({"AAAUSDT": volumes})
    raw_rolling = cm.rolling_trailing_avg_volume(frame)
    known = cm.point_in_time_known_avg(frame)
    for i in range(30, 35):
        assert known["AAAUSDT"].iloc[i] == pytest.approx(raw_rolling["AAAUSDT"].iloc[i - 1])
        assert known["AAAUSDT"].iloc[i] != pytest.approx(raw_rolling["AAAUSDT"].iloc[i])


def test_dynamic_liquidity_mask_has_no_lookahead_from_future_volume_change() -> None:
    idx = pd.date_range("2026-01-01", periods=80, freq="D", tz="UTC")
    base = pd.DataFrame({"AAAUSDT": [25_000_000.0] * 80}, index=idx)  # comfortably clears the $20M floor throughout
    collapsed = base.copy()
    collapsed.loc[idx[60] :, "AAAUSDT"] = 5_000_000.0  # a future volume COLLAPSE starting day 60

    # A flat mapping (bp is always 1.0 regardless of volume) isolates the test to the
    # volume-floor clause alone -- the slippage cap never binds here.
    flat_mapping = cm.MeasuredCostMapping(
        anchor_log_volume=np.array([0.0, 20.0]), anchor_bp=np.array([1.0, 1.0]), bucket_counts=(1, 1), raw_point_count=2, source_collected_at_utc="test"
    )
    mask_base = cm.build_dynamic_liquidity_mask(base, ("AAAUSDT",), flat_mapping, 20_000_000.0, 5.0)
    mask_collapsed = cm.build_dynamic_liquidity_mask(collapsed, ("AAAUSDT",), flat_mapping, 20_000_000.0, 5.0)

    before_change = idx[:60]
    pd.testing.assert_frame_equal(mask_base.loc[before_change], mask_collapsed.loc[before_change])
    # sanity: the collapse eventually DOES flip the mask false once its own trailing 30d
    # window fully absorbs it -- proves the two masks aren't just identically constant.
    assert bool(mask_base.loc[idx[-1], "AAAUSDT"]) is True
    assert bool(mask_collapsed.loc[idx[-1], "AAAUSDT"]) is False


def test_dynamic_liquidity_mask_requires_both_volume_floor_and_slippage_cap() -> None:
    idx = pd.date_range("2026-01-01", periods=5, freq="D", tz="UTC")

    # Scenario (a): the volume floor is the binding constraint (mapped bp at the floor is
    # already under the cap) -- a symbol below the $20M floor must fail even though its
    # mapped bp alone would have passed the 5bp cap.
    mapping_a = cm.MeasuredCostMapping(
        anchor_log_volume=np.array([6.0, 8.0]),  # $1M .. $100M
        anchor_bp=np.array([10.0, 1.0]),
        bucket_counts=(1, 1),
        raw_point_count=2,
        source_collected_at_utc="test",
    )
    frame_a = pd.DataFrame({"BELOW_FLOOR_OK_BP": [15_000_000.0] * 40, "ABOVE_FLOOR_OK_BP": [25_000_000.0] * 40}, index=pd.date_range("2026-01-01", periods=40, freq="D", tz="UTC"))
    mask_a = cm.build_dynamic_liquidity_mask(frame_a, tuple(frame_a.columns), mapping_a, 20_000_000.0, 5.0)
    assert bool(mapping_a.anchor_bp[0]) and cm.slippage_bp_for_volume(15_000_000.0, mapping_a) < 5.0  # bp alone would pass
    assert bool(mask_a.iloc[-1]["BELOW_FLOOR_OK_BP"]) is False  # ...but the $20M volume floor still rejects it
    assert bool(mask_a.iloc[-1]["ABOVE_FLOOR_OK_BP"]) is True

    # Scenario (b): the slippage cap is the binding constraint BEYOND the floor (mapped bp
    # at the floor is still above the cap) -- a symbol that clears $20M must still fail if
    # its measured cost hasn't dropped under 5bp yet.
    mapping_b = cm.MeasuredCostMapping(
        anchor_log_volume=np.array([6.0, 7.301029995663981, 9.0]),  # $1M, $20M, $1B
        anchor_bp=np.array([20.0, 8.0, 1.0]),
        bucket_counts=(1, 1, 1),
        raw_point_count=3,
        source_collected_at_utc="test",
    )
    frame_b = pd.DataFrame({"ABOVE_FLOOR_BAD_BP": [30_000_000.0] * 40, "ABOVE_FLOOR_GOOD_BP": [1_000_000_000.0] * 40}, index=pd.date_range("2026-01-01", periods=40, freq="D", tz="UTC"))
    mask_b = cm.build_dynamic_liquidity_mask(frame_b, tuple(frame_b.columns), mapping_b, 20_000_000.0, 5.0)
    assert cm.slippage_bp_for_volume(30_000_000.0, mapping_b) > 5.0  # clears the $20M floor but bp still too high
    assert bool(mask_b.iloc[-1]["ABOVE_FLOOR_BAD_BP"]) is False
    assert bool(mask_b.iloc[-1]["ABOVE_FLOOR_GOOD_BP"]) is True
    _ = idx  # unused placeholder kept for readability of the two date ranges above


def test_data_availability_mask_fails_closed_before_full_window() -> None:
    idx = pd.date_range("2026-01-01", periods=40, freq="D", tz="UTC")
    frame = pd.DataFrame({"AAAUSDT": [5_000_000.0] * 40}, index=idx)
    mask = cm.build_data_availability_mask(frame, ("AAAUSDT",))
    assert not mask.iloc[:30].to_numpy().any()  # no valid 30d window yet
    assert bool(mask.iloc[35]["AAAUSDT"])


# ---------------------------------------------------------------------------
# 5) Engine equivalence: fed a flat cost frame and an always-liquid mask, engine13 must
#    reproduce research.wave10_carry100.engine.run_fixed_fraction_portfolio bit-for-bit --
#    proof the measured-cost/liquidity model is the only thing engine13 changes versus the
#    shared wave10 loop body (same regression style as test_wave12.py's own equivalence test).
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


def test_engine13_matches_wave10_when_cost_is_flat_and_liquidity_always_ok() -> None:
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
    wave13_result, total_cost, eligible_counts = _run_liquidity_loop(*frames, candidate.top_k, leg_fraction, flat_cost, always_liquid)

    assert wave13_result.equity.tolist() == pytest.approx(wave10_result.equity.tolist(), rel=1e-12)
    assert wave13_result.positions.tolist() == pytest.approx(wave10_result.positions.tolist(), rel=1e-12)
    assert wave13_result.turnover.tolist() == pytest.approx(wave10_result.turnover.tolist(), rel=1e-12)
    assert wave13_result.trade_returns.tolist() == pytest.approx(wave10_result.trade_returns.tolist(), rel=1e-12)
    assert total_cost > 0.0
    assert len(eligible_counts) == len(spot_open_frame.index)


def test_engine13_liquidity_mask_makes_gated_symbol_behave_as_if_absent() -> None:
    markets = _two_symbol_synthetic_market()
    candidate = FundingCandidate("LIQTEST", 1, 0.50, 1)

    frames_both = _build_aligned_frames(markets, candidate)
    spot_open_both = frames_both[0]
    flat_cost_both = pd.DataFrame(0.001, index=spot_open_both.index, columns=spot_open_both.columns)
    liquidity_btc_gated = pd.DataFrame(True, index=spot_open_both.index, columns=spot_open_both.columns)
    liquidity_btc_gated["BTCUSDT"] = False  # BTCUSDT has the higher funding score -- would normally win top_k=1
    restricted, _, restricted_eligible = _run_liquidity_loop(*frames_both, candidate.top_k, 0.5, flat_cost_both, liquidity_btc_gated)

    eth_only_markets = {"ETHUSDT": markets["ETHUSDT"]}
    frames_eth_only = _build_aligned_frames(eth_only_markets, candidate)
    spot_open_eth = frames_eth_only[0]
    flat_cost_eth = pd.DataFrame(0.001, index=spot_open_eth.index, columns=spot_open_eth.columns)
    liquidity_eth_only = pd.DataFrame(True, index=spot_open_eth.index, columns=spot_open_eth.columns)
    eth_only_result, _, _ = _run_liquidity_loop(*frames_eth_only, candidate.top_k, 0.5, flat_cost_eth, liquidity_eth_only)

    assert restricted.equity.tolist() == pytest.approx(eth_only_result.equity.tolist(), rel=1e-12)
    assert restricted_eligible.max() <= 1.0


# ---------------------------------------------------------------------------
# 6) S5 gate: SPEC.md conjoins sign-preservation AND stress-MDD<=15% -- both must
#    independently hold; a config that stays positive-sign but blows through the stress
#    MDD cap must still FAIL (wave12's S5 never checked MDD at all).
# ---------------------------------------------------------------------------


def _synthetic_wave10_result(equity_values: list[float]) -> Wave10Result:
    idx = pd.date_range("2024-01-01", periods=len(equity_values), freq="D", tz="UTC")
    equity = pd.Series(equity_values, index=idx, dtype=float)
    zeros = pd.Series(0.0, index=idx, dtype=float)
    return Wave10Result(equity=equity, positions=zeros, turnover=zeros, trade_returns=pd.Series(dtype=float), max_concurrent_positions=0, symbols_used=())


def test_gate_s5_passes_only_when_sign_and_stress_mdd_both_hold() -> None:
    # Case A: smooth monotonic rise -> ~0% drawdown, comfortably under the 15% stress cap.
    smooth_returns = np.full(300, 0.0005)
    smooth_equity = (90.0 * np.cumprod(1.0 + smooth_returns)).tolist()
    smooth_equity = [90.0, *smooth_equity]
    result_calm = _synthetic_wave10_result(smooth_equity)
    report_calm = gates13.gate_s5_stress(0.15, 0.08, result_calm, seed_offset=1)
    assert report_calm["sign_preserved"] is True
    assert report_calm["stress_mdd_ok"] is True
    assert report_calm["status"] == "PASS"

    # Case B: a single embedded ~50% crash guarantees every block-shuffle permutation
    # (the crash's own 90-day block always appears somewhere in every shuffle) blows past
    # a 15% drawdown bar, even though the path still ends higher than it started.
    block1 = np.full(90, 0.001)
    rise = np.full(40, 0.004)
    crash = np.full(10, -0.07)  # (1-0.07)^10 ~= -51.6% within 10 days
    flat = np.full(40, 0.0005)
    block2 = np.concatenate([rise, crash, flat])
    block3 = np.full(90, 0.001)
    crash_returns = np.concatenate([block1, block2, block3])
    crash_equity = (90.0 * np.cumprod(1.0 + crash_returns)).tolist()
    crash_equity = [90.0, *crash_equity]
    result_crash = _synthetic_wave10_result(crash_equity)
    report_crash = gates13.gate_s5_stress(0.15, 0.05, result_crash, seed_offset=2)
    assert report_crash["sign_preserved"] is True  # stress_high_funding=0.05 passed in directly, positive
    assert report_crash["stress_mdd_ok"] is False  # but the embedded crash must blow the 15% stress-MDD bar
    assert report_crash["status"] == "FAIL"  # AND semantics: sign alone is not enough


def test_gate_s5_fails_on_negative_sign_even_if_mdd_is_fine() -> None:
    smooth_returns = np.full(200, 0.0003)
    equity = (90.0 * np.cumprod(1.0 + smooth_returns)).tolist()
    equity = [90.0, *equity]
    result = _synthetic_wave10_result(equity)
    report = gates13.gate_s5_stress(0.10, -0.02, result, seed_offset=3)  # negative stress return
    assert report["sign_preserved"] is False
    assert report["stress_mdd_ok"] is True
    assert report["status"] == "FAIL"
