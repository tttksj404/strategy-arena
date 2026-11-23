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
from research.wave10_carry100.engine import Wave10Result
from research.wave10_carry100.engine import cost_rate as wave10_cost_rate
from research.wave10_carry100.engine import run_fixed_fraction_portfolio as wave10_run_fixed_fraction_portfolio
from research.wave13_liquidity import costs_measured as wave13_costs_measured
from research.wave14_multivenue import costs_venue as cv
from research.wave14_multivenue import engine14
from research.wave14_multivenue import fetch_venues
from research.wave14_multivenue import gates14
from research.wave14_multivenue import universe_multi as um
from research.wave14_multivenue.configs14 import AUX_BASELINES, CONFIG_IDS, CONFIGS, LEG_USDT, get_config
from research.wave14_multivenue.universe_multi import CrossVenuePair

# ---------------------------------------------------------------------------
# 1) Config registry: frozen 8 candidates, $45 leg at every tier, exactly 1x leverage.
# ---------------------------------------------------------------------------


def test_config_registry_is_frozen_to_eight_preregistered_ids() -> None:
    assert CONFIG_IDS == ("M0", "M1", "M2", "M3", "M4", "M5", "M6", "M7")
    assert len(CONFIGS) == 8


def test_leg_usdt_fixed_and_gross_equals_active_capital_exactly_at_every_tier() -> None:
    # SPEC.md's (total_capital, top_k) pairs were chosen so every config lands at EXACTLY
    # 1x leverage (gross == active_capital) with a constant $45 leg -- configs14.py's own
    # module docstring claim, pinned here numerically for all 8 configs + both internal
    # reference baselines.
    for config in (*CONFIGS, *AUX_BASELINES):
        assert config.leg_fraction * config.active_capital == pytest.approx(LEG_USDT)
        gross = 2.0 * config.candidate.top_k * config.leg_fraction * config.active_capital
        assert gross == pytest.approx(config.active_capital)
        assert config.active_capital == pytest.approx(config.total_capital * 0.90)


def test_m6_m7_are_the_only_cross_venue_spread_structures_and_have_matched_baselines() -> None:
    by_id = {c.candidate_id: c for c in CONFIGS}
    assert by_id["M6"].structure == "cross_venue_spread"
    assert by_id["M7"].structure == "cross_venue_spread"
    for cid in ("M0", "M1", "M2", "M3", "M4", "M5"):
        assert by_id[cid].structure == "carry"
    assert by_id["M0"].baseline_candidate_id is None  # M0/M2 ARE their own tier's baseline
    assert by_id["M2"].baseline_candidate_id is None
    assert by_id["M1"].baseline_candidate_id == "M0"
    assert by_id["M3"].baseline_candidate_id == "M2"
    assert by_id["M6"].baseline_candidate_id == "M2"
    assert get_config(by_id["M4"].baseline_candidate_id).total_capital == pytest.approx(1_000.0)


# ---------------------------------------------------------------------------
# 2) daily_funding_score: exact equivalence to wave1.funding_score on uniform 8h cadence,
#    and correct generalization to a non-uniform (e.g. 4h) cadence.
# ---------------------------------------------------------------------------


def test_daily_funding_score_matches_wave1_funding_score_for_uniform_8h_cadence() -> None:
    from research.wave1.fam_funding import funding_score as wave1_funding_score

    idx = pd.date_range("2024-01-01", periods=40 * 3, freq="8h", tz="UTC")
    rng = np.random.default_rng(7)
    funding = pd.Series(rng.normal(0.0001, 0.0003, len(idx)), index=idx)
    old = wave1_funding_score(funding, 7).resample("1D").last()
    new = engine14.daily_funding_score(funding.resample("1D").sum(), 7)
    aligned = pd.concat([old.rename("old"), new.rename("new")], axis=1).dropna()
    assert len(aligned) > 20
    assert aligned["old"].to_numpy() == pytest.approx(aligned["new"].to_numpy(), abs=1e-9)


def test_daily_funding_score_correctly_annualizes_a_non_8h_cadence() -> None:
    # A 4h-cadence symbol (6 events/day) with a CONSTANT per-event rate: wave1's own
    # event-count-based formula would assume 3 events/day and read this as HALF its true
    # daily funding sum. daily_funding_score operates on the already-daily-resampled sum,
    # so it is cadence-agnostic by construction -- verified against a hand-computed value.
    idx_4h = pd.date_range("2024-01-01", periods=20 * 6, freq="4h", tz="UTC")
    rate_per_event = 0.0005
    funding_4h = pd.Series(rate_per_event, index=idx_4h)
    funding_daily = funding_4h.resample("1D").sum()
    assert funding_daily.iloc[0] == pytest.approx(rate_per_event * 6)  # 6 events/day
    score = engine14.daily_funding_score(funding_daily, 7)
    expected_apr = (rate_per_event * 6 * 7) * (365.0 / 7.0)  # 7 days x 6 events/day x rate, annualized
    valid = score.dropna()
    assert len(valid) > 5
    assert valid.iloc[-1] == pytest.approx(expected_apr)


# ---------------------------------------------------------------------------
# 3) costs_venue: per-venue fee asymmetry, cross-venue combination, mapping fit wiring.
# ---------------------------------------------------------------------------


def test_bybit_pair_cost_rate_is_not_symmetric_and_uses_real_published_fees() -> None:
    # Unlike wave13's cost_rate_from_bp (2.0 * ONE fee, valid because Bitget's own maker fee
    # happens to be identical on both legs), Bybit's spot (0.10%) and linear (0.02%) maker
    # fees genuinely differ -- summing them, not doubling either one, is the whole point of
    # this function existing separately.
    rate = cv.bybit_pair_cost_rate(spot_bp=0.0, linear_bp=0.0, stress_multiplier=1.0)
    assert rate == pytest.approx(cv.BYBIT_SPOT_MAKER_FEE_RATE + cv.BYBIT_LINEAR_MAKER_FEE_RATE)
    assert cv.BYBIT_SPOT_MAKER_FEE_RATE != cv.BYBIT_LINEAR_MAKER_FEE_RATE
    stressed = cv.bybit_pair_cost_rate(spot_bp=4.0, linear_bp=4.0, stress_multiplier=3.0)
    base = cv.bybit_pair_cost_rate(spot_bp=4.0, linear_bp=4.0, stress_multiplier=1.0)
    maker_component = cv.BYBIT_SPOT_MAKER_FEE_RATE + cv.BYBIT_LINEAR_MAKER_FEE_RATE
    assert (stressed - maker_component) == pytest.approx(3.0 * (base - maker_component))  # stress scales slippage only


def test_cross_venue_leg_cost_rate_sums_binance_bitget_proxy_and_bybit_linear() -> None:
    from research.wave2.funding import W2_MAKER_FEE_RATE

    rate = cv.cross_venue_leg_cost_rate(binance_linear_bp=0.0, bybit_linear_bp=0.0, stress_multiplier=1.0)
    assert rate == pytest.approx(W2_MAKER_FEE_RATE + cv.BYBIT_LINEAR_MAKER_FEE_RATE)


def test_fit_bybit_mappings_produces_two_independent_monotonic_mappings() -> None:
    def _synthetic(scale: float) -> list[dict]:
        volumes = [1.0e5, 5.0e5, 2.0e6, 1.0e7, 5.0e7, 2.0e8, 1.0e9]
        slippage = [30.0 * scale, 20.0 * scale, 12.0 * scale, 7.0 * scale, 4.0 * scale, 2.0 * scale, 0.5 * scale]
        return [{"usdt_volume_24h": v, "effective_slippage_bp": s} for v, s in zip(volumes, slippage)]

    payload = {"collected_at_utc": "test", "spot": {"measurements": _synthetic(5.0)}, "linear": {"measurements": _synthetic(1.0)}}
    mappings = cv.fit_bybit_mappings(payload, n_buckets=6)
    assert np.all(np.diff(mappings.spot.anchor_bp) <= 1e-9)
    assert np.all(np.diff(mappings.linear.anchor_bp) <= 1e-9)
    # spot was fit from a scale-5x-pricier synthetic snapshot -- its anchors must land
    # materially above linear's own, at every matching point, proving the two fits are
    # genuinely independent (not accidentally sharing state).
    assert mappings.spot.worst_bp > mappings.linear.worst_bp * 2.0


# ---------------------------------------------------------------------------
# 4) engine14 carry loop: equivalence to wave10 at $90, and exact linear scaling at a
#    different capital tier (both loops are purely multiplicative in `capital`).
# ---------------------------------------------------------------------------


def _two_symbol_synthetic_market(periods: int = 14) -> dict[str, FundingMarket]:
    daily_index = pd.date_range("2024-01-01", periods=periods, freq="D", tz="UTC")
    funding_index = pd.date_range("2024-01-01", periods=periods * 3, freq="8h", tz="UTC")
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


def test_carry_loop_matches_wave10_at_90usd_when_cost_flat_and_liquidity_always_ok() -> None:
    markets = _two_symbol_synthetic_market()
    candidate = FundingCandidate("EQTEST", 1, 0.50, 1)
    leg_fraction = 0.5

    wave10_result = wave10_run_fixed_fraction_portfolio(markets, Wave10Config(candidate, leg_fraction, "equivalence test"))
    frames = engine14._build_aligned_frames_multi(markets, candidate)
    spot_open_frame = frames[0]
    flat_cost = pd.DataFrame({s: [wave10_cost_rate(s)] * len(spot_open_frame.index) for s in spot_open_frame.columns}, index=spot_open_frame.index)
    always_liquid = pd.DataFrame(True, index=spot_open_frame.index, columns=spot_open_frame.columns)

    result, total_cost, eligible = engine14._run_carry_loop(*frames, candidate.top_k, leg_fraction, 90.0, flat_cost, always_liquid)

    assert result.equity.tolist() == pytest.approx(wave10_result.equity.tolist(), rel=1e-12)
    assert result.positions.tolist() == pytest.approx(wave10_result.positions.tolist(), rel=1e-12)
    assert result.turnover.tolist() == pytest.approx(wave10_result.turnover.tolist(), rel=1e-12)
    assert total_cost > 0.0
    assert len(eligible) == len(spot_open_frame.index)


def test_carry_loop_active_capital_scales_the_whole_equity_path_linearly() -> None:
    # capital compounds purely multiplicatively (no fixed-dollar terms independent of
    # capital anywhere in the loop) -- active_capital=$270 must reproduce EXACTLY 3x the
    # $90 path at every single timestep, not just at the end.
    markets = _two_symbol_synthetic_market()
    candidate = FundingCandidate("SCALETEST", 1, 0.50, 1)
    leg_fraction = 0.5
    frames = engine14._build_aligned_frames_multi(markets, candidate)
    spot_open_frame = frames[0]
    flat_cost = pd.DataFrame({s: [wave10_cost_rate(s)] * len(spot_open_frame.index) for s in spot_open_frame.columns}, index=spot_open_frame.index)
    always_liquid = pd.DataFrame(True, index=spot_open_frame.index, columns=spot_open_frame.columns)

    result_90, _, _ = engine14._run_carry_loop(*frames, candidate.top_k, leg_fraction, 90.0, flat_cost, always_liquid)
    result_270, _, _ = engine14._run_carry_loop(*frames, candidate.top_k, leg_fraction, 270.0, flat_cost, always_liquid)

    ratio = (result_270.equity / result_90.equity).to_numpy()
    assert ratio == pytest.approx(np.full(len(ratio), 3.0), rel=1e-9)


# ---------------------------------------------------------------------------
# 5) OVERLAP window slicing: fresh start, warmup preserved, no lookahead.
# ---------------------------------------------------------------------------


def test_window_slicing_gives_a_fresh_capital_start_using_pre_window_warmup() -> None:
    # Funding is high enough to be "active" well before OVERLAP_START -- slicing must NOT
    # force a cold re-warm at the window boundary (that would misrepresent "the same
    # strategy, just re-anchored to a later start date" as "a strategy that takes weeks to
    # even begin trading" every single time it's re-run on a later window).
    periods = 40
    daily_index = pd.date_range("2023-12-01", periods=periods, freq="D", tz="UTC")  # spans across engine14.OVERLAP_START
    funding_index = pd.date_range("2023-12-01", periods=periods * 3, freq="8h", tz="UTC")
    spot = pd.DataFrame({"open": [100.0] * periods, "close": [100.0] * periods}, index=daily_index)
    perp = pd.DataFrame({"open": [99.9] * periods, "close": [99.9] * periods}, index=daily_index)
    funding = pd.Series(0.002, index=funding_index)  # very rich funding throughout -- comfortably active by day 1
    markets = {"BTCUSDT": FundingMarket(spot, perp, funding)}
    candidate = FundingCandidate("WINTEST", 3, 0.10, 1)  # short window so it warms up well before 2024-01-01
    frames = engine14._build_aligned_frames_multi(markets, candidate)
    active_frame = frames[6]
    window_mask = (active_frame.index >= engine14.OVERLAP_START) & (active_frame.index < engine14.OVERLAP_END)
    sliced_active = active_frame.loc[window_mask]
    assert len(sliced_active) > 0
    assert bool(sliced_active.iloc[0]["BTCUSDT"] > 0.0)  # already warm/active on day 1 of the sliced window

    flat_cost = pd.DataFrame(0.0, index=frames[0].index, columns=frames[0].columns)
    always_liquid = pd.DataFrame(True, index=frames[0].index, columns=frames[0].columns)
    sliced_frames = tuple(frame.loc[window_mask] for frame in frames)
    result, _, _ = engine14._run_carry_loop(*sliced_frames, candidate.top_k, 0.5, 90.0, flat_cost.loc[window_mask], always_liquid.loc[window_mask])
    # funding=0.002/event x 3 events/day (8h cadence) = 0.006/day; flat spot/perp prices
    # contribute 0 to the intraday term, so day 1's growth is exactly leg_fraction x
    # daily funding, applied to a FRESH $90 (not the pre-window-warmed-up trajectory).
    assert float(result.equity.iloc[0]) == pytest.approx(90.0 * (1.0 + 0.5 * 0.002 * 3.0), rel=1e-9)


# ---------------------------------------------------------------------------
# 6) M6/M7 cross-venue structure: hysteresis side-lock and PnL sign/magnitude.
# ---------------------------------------------------------------------------


def test_cross_venue_position_locks_side_at_entry_and_uses_threshold_half_exit_band() -> None:
    candidate = FundingCandidate("SIDETEST", 1, 0.10, 1)
    # Binance richer for a while (side should lock +1), then the spread narrows but stays
    # positive (must remain active, same side) until it drops under threshold/2 = 0.05.
    raw_spread = pd.Series([0.20, 0.20, 0.20, 0.08, 0.08, 0.02, 0.02], index=pd.date_range("2024-01-01", periods=7, freq="D", tz="UTC"))
    active, side = engine14.cross_venue_position(raw_spread, candidate)
    # shift(1)'d: day0 always 0 (nothing known yet).
    assert active.iloc[0] == 0.0
    assert list(active.iloc[1:4]) == [1.0, 1.0, 1.0]  # stays active while locked-direction spread > 0.05
    assert list(side.iloc[1:4]) == [1.0, 1.0, 1.0]
    assert active.iloc[-1] == 0.0  # exits once locked-direction spread (0.02) < 0.05
    assert side.iloc[-1] == 0.0


def test_cross_venue_position_picks_the_opposite_side_when_bybit_is_richer() -> None:
    candidate = FundingCandidate("SIDETEST2", 1, 0.10, 1)
    raw_spread = pd.Series([-0.30, -0.30, -0.30], index=pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC"))
    active, side = engine14.cross_venue_position(raw_spread, candidate)
    assert side.iloc[-1] == -1.0  # Bybit richer -> short Bybit / long Binance
    assert active.iloc[-1] == 1.0


def test_run_cross_venue_loop_pnl_matches_hand_derived_formula_binance_richer() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
    flat = pd.DataFrame({"open": [100.0] * 5, "close": [100.0] * 5}, index=idx)  # flat prices -> price term is exactly 0
    binance_funding = pd.Series([0.002] * 5, index=idx)
    bybit_funding = pd.Series([0.0] * 5, index=idx)
    pairs = {"AAA": CrossVenuePair("AAA", flat, binance_funding, flat, bybit_funding)}
    candidate = FundingCandidate("PNLTEST", 1, 0.10, 1)
    frames = engine14._build_cross_venue_frames(pairs, candidate)
    assert frames[7]["AAA"].iloc[1] == 1.0  # side locks +1 (short Binance / long Bybit) once active
    cost = pd.DataFrame(0.0, index=frames[0].index, columns=frames[0].columns)
    liquidity = pd.DataFrame(True, index=frames[0].index, columns=frames[0].columns)
    result, total_cost, _ = engine14._run_cross_venue_loop(*frames, 1, 0.5, 100.0, cost, liquidity)
    # per-active-day growth factor = 1 + leg_fraction * (funding_binance - funding_bybit) = 1 + 0.5*0.002
    day_factor = 1.0 + 0.5 * 0.002
    assert float(result.equity.iloc[1]) == pytest.approx(100.0 * day_factor)
    assert float(result.equity.iloc[2]) == pytest.approx(100.0 * day_factor**2)
    assert total_cost == pytest.approx(0.0)  # zero cost frame in, zero cost out


def test_run_cross_venue_loop_pnl_matches_hand_derived_formula_bybit_richer() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC")
    binance_flat = pd.DataFrame({"open": [100.0] * 4, "close": [100.0] * 4}, index=idx)
    bybit_flat = pd.DataFrame({"open": [50.0] * 4, "close": [50.0] * 4}, index=idx)
    binance_funding = pd.Series([0.0] * 4, index=idx)
    bybit_funding = pd.Series([0.003] * 4, index=idx)
    pairs = {"BBB": CrossVenuePair("BBB", binance_flat, binance_funding, bybit_flat, bybit_funding)}
    candidate = FundingCandidate("PNLTEST2", 1, 0.10, 1)
    frames = engine14._build_cross_venue_frames(pairs, candidate)
    assert frames[7]["BBB"].iloc[1] == -1.0  # Bybit richer -> side locks -1
    cost = pd.DataFrame(0.0, index=frames[0].index, columns=frames[0].columns)
    liquidity = pd.DataFrame(True, index=frames[0].index, columns=frames[0].columns)
    result, _, _ = engine14._run_cross_venue_loop(*frames, 1, 0.5, 100.0, cost, liquidity)
    day_factor = 1.0 + 0.5 * 0.003  # shorting the richer (Bybit) venue still earns the spread
    assert float(result.equity.iloc[1]) == pytest.approx(100.0 * day_factor)
    assert float(result.equity.iloc[2]) == pytest.approx(100.0 * day_factor**2)


# ---------------------------------------------------------------------------
# 7) gates14 S6: structural (not empirical) cross-venue exposure.
# ---------------------------------------------------------------------------


def test_gate_s6_cross_venue_structure_is_a_fixed_55pct_residual_at_every_tier() -> None:
    for candidate_id in ("M6", "M7"):
        config = get_config(candidate_id)
        report = gates14.gate_s6_cross_venue_structure(config)
        assert report["applicable"] is True
        # gross == active_capital (1x) and split exactly 50/50 across venues by
        # construction -> residual = 1 - 0.5*active_capital/total_capital = 1 - 0.5*0.9 = 0.55, always.
        assert report["residual_capital_fraction_if_one_venue_wiped"] == pytest.approx(0.55)
        assert report["status"] == "PASS"


def test_gate_s6_pool_venue_exposure_is_informational_and_never_blocks() -> None:
    m1 = get_config("M1")
    report_no_data = gates14.gate_s6_pool_venue_exposure(m1, None)
    assert report_no_data["applicable"] is True
    assert report_no_data["status"] == "INFO"  # never PASS/FAIL -- see gates14.py's own module docstring
    assert report_no_data["structural_worst_case_residual_if_one_venue_wiped"] == pytest.approx(0.10)

    m0 = get_config("M0")
    report_na = gates14.gate_s6_pool_venue_exposure(m0, None)
    assert report_na["applicable"] is False
    assert report_na["status"] == "N/A"

    share = pd.Series([0.2, 0.4, 0.6], index=pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC"))
    report_with_data = gates14.gate_s6_pool_venue_exposure(m1, share)
    assert report_with_data["empirical_bybit_share"]["available"] is True
    assert report_with_data["empirical_bybit_share"]["max_bybit_share_of_filled_slots"] == pytest.approx(0.6)


def test_gate_s1_leverage_passes_at_exactly_1x_for_every_config() -> None:
    for config in CONFIGS:
        report = gates14.gate_s1_structure(config)
        assert report["leverage_1x_ok"] is True
        assert report["status"] == "PASS"
        assert report["leverage_multiplier_of_active_capital"] == pytest.approx(1.0)


def test_promotion_check_requires_beating_baseline_when_one_is_registered() -> None:
    m1 = get_config("M1")
    baseline_regime = {"high_funding_mean_annualized_return": 0.10}
    worse = gates14.promotion_check({"high_funding_mean_annualized_return": 0.05}, "PASS", m1, baseline_regime)
    assert worse.beats_baseline is False
    assert worse.promoted is False
    better = gates14.promotion_check({"high_funding_mean_annualized_return": 0.15}, "PASS", m1, baseline_regime)
    assert better.beats_baseline is True
    assert better.promoted is True
    m0 = get_config("M0")
    no_baseline = gates14.promotion_check({"high_funding_mean_annualized_return": 0.01}, "PASS", m0, None)
    assert no_baseline.beats_baseline is None
    assert no_baseline.promoted is True  # M0 has nothing to beat -- PASS alone promotes it


# ---------------------------------------------------------------------------
# 8) universe_multi key helpers + fetch_venues pure logic (network-mocked).
# ---------------------------------------------------------------------------


def test_venue_key_helpers_roundtrip() -> None:
    assert um.venue_of_key("BTCUSDT") == "binance"
    assert um.venue_of_key(um.bybit_key("BTCUSDT")) == "bybit"
    assert um.base_symbol(um.bybit_key("BTCUSDT")) == "BTCUSDT"
    assert um.base_symbol("BTCUSDT") == "BTCUSDT"


def test_fetch_venues_reuses_wave13_walk_cost_function_not_reimplemented() -> None:
    from research.wave13_liquidity.collect_spreads import compute_walk_cost_bp as wave13_walk_cost

    assert fetch_venues.compute_walk_cost_bp is wave13_walk_cost


def test_discover_bybit_universe_intersects_l4_and_excludes_low_funding_interval(monkeypatch: pytest.MonkeyPatch) -> None:
    spot_book = {"AAAUSDT": {}, "BBBUSDT": {}, "CCCUSDT": {}}
    linear_book = {
        "AAAUSDT": {"fundingInterval": 480},
        "BBBUSDT": {"fundingInterval": 60},  # excluded -- sub-4h cadence
        "DDDUSDT": {"fundingInterval": 240},  # not in L4 -- excluded by intersection
    }

    def fake_fetch_instruments(category: str) -> dict:
        return spot_book if category == "spot" else linear_book

    monkeypatch.setattr(fetch_venues, "fetch_instruments", fake_fetch_instruments)
    monkeypatch.setattr(fetch_venues, "MINIMUM_UNIVERSE_SYMBOLS", 1)  # tiny synthetic universe; real runs use the module default
    payload = fetch_venues.discover_bybit_universe(("AAAUSDT", "BBBUSDT"))
    assert payload["universe"] == ["AAAUSDT"]
    assert payload["excluded_low_funding_interval"] == ["BBBUSDT"]


def test_measure_orderbook_computes_expected_bp_from_a_mocked_response(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_get_json(path: str, params: dict, base_url: str = "") -> dict:
        assert path == fetch_venues.ORDERBOOK_PATH
        return {"result": {"a": [["100.5", "10.0"]], "b": [["99.5", "10.0"]]}}

    monkeypatch.setattr(fetch_venues, "_get_json", fake_get_json)
    measurement = fetch_venues.measure_orderbook("AAAUSDT", "linear", usdt_volume_24h=1.0e7)
    assert measurement is not None
    expected_half_spread = (100.5 - 99.5) / 2.0 / 100.0 * 1.0e4
    assert measurement["half_spread_bp"] == pytest.approx(expected_half_spread)
    assert measurement["walk_cost_bp"] == pytest.approx(expected_half_spread)  # $45 fully absorbed at level 1
    assert measurement["insufficient_depth"] is False
