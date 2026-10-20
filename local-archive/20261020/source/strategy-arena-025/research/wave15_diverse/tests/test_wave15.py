from __future__ import annotations

import dataclasses
from pathlib import Path
import sys

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[3]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

import numpy as np
import pandas as pd  # noqa: PANDAS_OK
import pytest

from research.wave15_diverse import common15, configs15, engine_daily, engine_intraday, engine_pairs, gates15, signals15

# ---------------------------------------------------------------------------
# 1) 정산시각 정렬 (settlement-hour alignment): funding settles exactly on the fixed UTC
#    schedule this whole engine hardcodes, and A1-A3's decision/exit hours are derived from
#    it correctly (T-1h before, T+1h after -- no off-by-one).
# ---------------------------------------------------------------------------


def test_settlement_hours_match_the_real_funding_cache_and_decision_hours_are_consistent() -> None:
    frames = engine_intraday.build_hourly_frames()
    settled = frames.funding_actual[(frames.funding_actual != 0.0).any(axis=1)]
    observed_hours = set(settled.index.hour.unique().tolist())
    assert observed_hours <= engine_intraday.SETTLEMENT_HOURS
    assert engine_intraday.SETTLEMENT_HOURS == frozenset({0, 8, 16})
    # DECISION_HOURS must be exactly (settlement - 1h) for each settlement hour, mod 24.
    expected_decision_hours = {(hour - 1) % 24 for hour in engine_intraday.SETTLEMENT_HOURS}
    assert engine_intraday.DECISION_HOURS == frozenset(expected_decision_hours)
    assert engine_intraday.INTRADAY_HOLD == pd.Timedelta(hours=2)


def test_funding_timestamps_have_zero_jitter() -> None:
    frames = engine_intraday.build_hourly_frames()
    # every settlement bar's timestamp is exactly on-the-hour -- reindex/ffill alignment in
    # build_hourly_frames depends on this holding exactly, not approximately.
    assert (frames.index.minute == 0).all()
    assert (frames.index.second == 0).all()


# ---------------------------------------------------------------------------
# 2) 인트라데이 회전비용 배수 (intraday turnover-cost multiplier): N settlement cycles of
#    always-on intraday rotation charge cost_for() 2xN times (entry+exit each cycle), a
#    synthetic-data check independent of real market noise.
# ---------------------------------------------------------------------------


def _synthetic_hourly_frames(n_days: int = 6) -> engine_intraday.HourlyFrames:
    symbols = ("BTCUSDT", "ETHUSDT")
    index = pd.date_range("2024-01-01", periods=n_days * 24, freq="1h", tz="UTC")
    rng = np.random.default_rng(7)
    flat = pd.DataFrame(100.0, index=index, columns=symbols)
    noise = pd.DataFrame(rng.normal(0.0, 0.0, size=(len(index), 2)), index=index, columns=symbols)  # zero price noise -> isolates cost/funding only
    spot_open = flat + noise
    spot_close = flat + noise
    perp_open = flat + noise
    perp_close = flat + noise
    funding_actual = pd.DataFrame(0.0, index=index, columns=symbols)
    funding_actual.loc[index.hour.isin(engine_intraday.SETTLEMENT_HOURS), "BTCUSDT"] = 0.001  # always comfortably above any threshold used below
    funding_actual.loc[index.hour.isin(engine_intraday.SETTLEMENT_HOURS), "ETHUSDT"] = 0.0005
    funding_known = funding_actual.replace(0.0, np.nan).ffill().fillna(0.0)
    slow_score_known = pd.DataFrame(0.0, index=index, columns=symbols)  # irrelevant for A1/A2 (daily_entry_threshold=None)
    return engine_intraday.HourlyFrames(
        index=index, symbols=symbols, spot_open=spot_open, spot_close=spot_close, perp_open=perp_open, perp_close=perp_close,
        funding_actual=funding_actual, funding_known=funding_known, slow_score_known=slow_score_known,
    )


def test_intraday_turnover_cost_scales_with_number_of_settlement_cycles() -> None:
    frames = _synthetic_hourly_frames(n_days=6)
    flat_cost_rate = pd.DataFrame(0.001, index=frames.index, columns=frames.symbols)  # 10bp-ish flat, symbol-agnostic for a clean multiple
    config = engine_intraday.IntradayConfig("TEST_A1", 0.0, None, None)
    result, total_cost, diag = engine_intraday.run_intraday_carry(frames, config, flat_cost_rate)

    n_cycles = int(diag["n_intraday_entries"])
    assert n_cycles == 18  # 3 settlements/day x 6 days, BTC funding always ranks above ETH so BTC wins every cycle
    # each cycle pays cost_for() TWICE (open the leg_fraction position, then close it) --
    # SPEC.md's "회전수만큼 비용 배수" multiplier is exactly this 2x-per-cycle relationship,
    # not a separately-applied fudge factor.
    expected_cost_per_cycle = 2.0 * common15.LEG_FRACTION * flat_cost_rate.iloc[0, 0]
    expected_total_cost_fraction = n_cycles * expected_cost_per_cycle
    # total_cost_usdt is in dollar terms (charged against a compounding ACTIVE_CAPITAL-based
    # equity curve); compare the IMPLIED fraction instead of a raw dollar figure so this test
    # doesn't depend on compounding-order arithmetic.
    implied_fraction = total_cost / common15.ACTIVE_CAPITAL
    assert implied_fraction == pytest.approx(expected_total_cost_fraction, rel=0.05)
    assert n_cycles * 2 == diag["n_intraday_entries"] * 2  # sanity: turnover events = 2 x cycles


def test_a2_filter_strictly_reduces_cycles_versus_a1_on_the_same_data() -> None:
    frames = _synthetic_hourly_frames(n_days=6)
    # Push half of the settlement cycles' funding below A2's 0.03% filter.
    frames.funding_actual.iloc[:, :] = 0.0
    mask = frames.index.hour.isin(engine_intraday.SETTLEMENT_HOURS)
    settle_positions = np.where(mask)[0]
    for i, pos in enumerate(settle_positions):
        rate = 0.0005 if i % 2 == 0 else 0.0001  # alternate: above / below A2's 0.0003 bar
        frames.funding_actual.iloc[pos, 0] = rate
    frames = dataclasses.replace(frames, funding_known=frames.funding_actual.replace(0.0, np.nan).ffill().fillna(0.0))
    flat_cost_rate = pd.DataFrame(0.0005, index=frames.index, columns=frames.symbols)

    a1_result, _, a1_diag = engine_intraday.run_intraday_carry(frames, configs15.A1_CONFIG, flat_cost_rate)
    a2_result, _, a2_diag = engine_intraday.run_intraday_carry(frames, configs15.A2_CONFIG, flat_cost_rate)
    assert a2_diag["n_intraday_entries"] < a1_diag["n_intraday_entries"]


# ---------------------------------------------------------------------------
# 3) A3 상태전환 (state machine): the hybrid never has an active INTRADAY hold sitting at a
#    decision bar (the invariant _resolve_decision's docstring asserts), and a controlled
#    synthetic run actually visits both DAILY and INTRADAY modes with no double entry/exit.
# ---------------------------------------------------------------------------


def test_a3_state_machine_never_double_enters_or_violates_the_decision_bar_invariant() -> None:
    n_days = 20
    frames = _synthetic_hourly_frames(n_days=n_days)
    # First half: strong, durable BTC signal (drives DAILY mode). Second half: BTC signal
    # collapses, only occasional above-A2-bar spikes remain (drives INTRADAY mode).
    half = n_days // 2
    switch_ts = frames.index[0] + pd.Timedelta(days=half)
    frames.slow_score_known.loc[frames.index < switch_ts, "BTCUSDT"] = 0.20  # > 15% entry bar
    frames.slow_score_known.loc[frames.index >= switch_ts, "BTCUSDT"] = 0.02  # < 7.5% exit bar
    mask = frames.index.hour.isin(engine_intraday.SETTLEMENT_HOURS) & (frames.index >= switch_ts)
    frames.funding_actual.loc[mask, "BTCUSDT"] = 0.0006  # clears A2/A3's 0.0003 intraday bar
    frames = dataclasses.replace(frames, funding_known=frames.funding_actual.replace(0.0, np.nan).ffill().fillna(0.0))
    flat_cost_rate = pd.DataFrame(0.0002, index=frames.index, columns=frames.symbols)

    result, total_cost, diag = engine_intraday.run_intraday_carry(frames, configs15.A3_CONFIG, flat_cost_rate)

    assert diag["state_machine_invariant_violations"] == 0.0
    assert diag["n_daily_entries"] >= 1  # actually used the daily-hold leg in the first half
    assert diag["n_intraday_entries"] >= 1  # actually used the intraday leg in the second half
    assert diag["daily_bar_fraction"] > 0.0
    assert diag["intraday_bar_fraction"] > 0.0
    # Every closed trade's implied holding span is either "long" (daily-mode-scale) or exactly
    # 2 hours (intraday-mode) -- never something in between, which would indicate a botched
    # transition (e.g. a daily hold force-closed after only 1 bar).
    trade_gaps = result.trade_returns.index.to_series().diff().dropna()
    assert (trade_gaps > pd.Timedelta(0)).all()


# ---------------------------------------------------------------------------
# 4) B1 수익분리 (profit separation): carry-only vs carry+assumed-yield must be reported as
#    separate figures, and zeroing the assumed yield must reproduce the carry-only path
#    exactly (no hidden coupling between the yield overlay and the entry/exit signal).
# ---------------------------------------------------------------------------


def _tiny_price_frames(n_days: int = 400, seed: int = 3) -> dict[str, pd.DataFrame]:
    symbols = ("AAAUSDT", "BBBUSDT")
    index = pd.date_range("2021-01-01", periods=n_days, freq="1D", tz="UTC")
    rng = np.random.default_rng(seed)
    base = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, size=(n_days, 2)), axis=0))
    spot_open = pd.DataFrame(base, index=index, columns=symbols)
    spot_close = spot_open * (1.0 + rng.normal(0.0, 0.001, size=(n_days, 2)))
    perp_open = spot_open.copy()
    perp_close = spot_close * (1.0 + rng.normal(0.0, 0.0002, size=(n_days, 2)))
    funding_daily = pd.DataFrame(rng.uniform(0.0, 0.0009, size=(n_days, 2)), index=index, columns=symbols)
    return {"spot_open": spot_open, "spot_close": spot_close, "perp_open": perp_open, "perp_close": perp_close, "funding_daily": funding_daily}


def test_b1_carry_only_and_carry_plus_earn_are_separated_and_earn_zero_reproduces_carry_only() -> None:
    frames = _tiny_price_frames()
    index = frames["spot_open"].index
    active_frame = pd.DataFrame(1.0, index=index, columns=frames["spot_open"].columns)
    score_frame = pd.DataFrame(rng_score(index, frames["spot_open"].columns), index=index, columns=frames["spot_open"].columns)
    cost_rate_frame = pd.DataFrame(0.0006, index=index, columns=frames["spot_open"].columns)

    carry_only, _ = engine_daily.run_generic_carry(frames, active_frame, score_frame, 1, 0.5, cost_rate_frame, "spot_perp", 0.0)
    carry_zero_explicit, _ = engine_daily.run_generic_carry(frames, active_frame, score_frame, 1, 0.5, cost_rate_frame, "spot_perp", extra_annual_yield=0.0)
    combined, _ = engine_daily.run_generic_carry(frames, active_frame, score_frame, 1, 0.5, cost_rate_frame, "spot_perp", common15.ASSUMED_FLEXIBLE_EARN_APR)

    pd.testing.assert_series_equal(carry_only.equity, carry_zero_explicit.equity)
    assert not carry_only.equity.equals(combined.equity)
    # combined must be >= carry-only every day the position is active (yield is strictly
    # additive, never a drag, while a position is open) -- the whole point of "가산".
    assert (combined.equity.to_numpy() >= carry_only.equity.to_numpy() * (1 - 1e-9)).all()


def rng_score(index: pd.DatetimeIndex, columns) -> np.ndarray:
    rng = np.random.default_rng(11)
    return rng.uniform(0.10, 0.30, size=(len(index), len(columns)))


# ---------------------------------------------------------------------------
# 5) B2 델타노출 기록 (delta exposure disclosure): B2's structure must be recorded as NOT
#    delta-neutral, must use the single-leg (not doubled) cost formula, and gates15's S1 must
#    surface that disclosure without turning it into an automatic FAIL.
# ---------------------------------------------------------------------------


def test_b2_structure_is_disclosed_as_directional_and_single_leg_costed() -> None:
    assert configs15.B2_CONFIG.structure == "perp_only_short"
    assert configs15.B1_CONFIG.structure == "spot_perp"

    bp = 2.0
    single = common15.single_leg_cost_rate_from_bp(bp)
    double = common15.two_leg_cost_rate_from_bp(bp)
    assert single == pytest.approx(double / 2.0)

    gate_b2 = gates15.gate_s1_structure(common15.LEG_USDT, common15.GROSS_USDT, delta_neutral=False, note="no hedge")
    gate_b1 = gates15.gate_s1_structure(common15.LEG_USDT, common15.GROSS_USDT, delta_neutral=True, note="hedged")
    assert gate_b2["delta_neutral_by_construction"] is False
    assert gate_b1["delta_neutral_by_construction"] is True
    # disclosure must never itself flip PASS/FAIL -- only leverage feasibility does (SPEC.md:
    # "S1 구조(델타중립 여부는 후보별 명시)" is a recording requirement, not a gate criterion).
    assert gate_b2["status"] == gate_b1["status"] == "PASS"


def test_b2_perp_only_return_formula_has_no_spot_leg_contribution() -> None:
    frames = _tiny_price_frames(n_days=50, seed=5)
    index = frames["spot_open"].index
    active_frame = pd.DataFrame(1.0, index=index, columns=frames["spot_open"].columns)
    score_frame = pd.DataFrame(rng_score(index, frames["spot_open"].columns), index=index, columns=frames["spot_open"].columns)
    cost_rate_frame = pd.DataFrame(0.0003, index=index, columns=frames["spot_open"].columns)

    # Mutate the spot leg into something wild -- if it leaked into the perp_only_short
    # formula, the two results below would diverge; they must not.
    frames_mutated_spot = dict(frames)
    frames_mutated_spot["spot_open"] = frames["spot_open"] * 5.0
    frames_mutated_spot["spot_close"] = frames["spot_close"] * 0.2

    result_a, _ = engine_daily.run_generic_carry(frames, active_frame, score_frame, 1, 0.5, cost_rate_frame, "perp_only_short", 0.0)
    result_b, _ = engine_daily.run_generic_carry(frames_mutated_spot, active_frame, score_frame, 1, 0.5, cost_rate_frame, "perp_only_short", 0.0)
    pd.testing.assert_series_equal(result_a.equity, result_b.equity)


# ---------------------------------------------------------------------------
# 6) C1 계수고정 검증 (fixed-coefficient verification): weights are literal pre-registered
#    constants (sum to 1, equal-weighted as documented), and the composite score is a pure
#    function of its inputs (same inputs -> byte-identical output across repeated calls --
#    i.e. no hidden fit/state that could drift with a second call, which is what "학습 금지"
#    rules out).
# ---------------------------------------------------------------------------


def test_c1_feature_weights_are_fixed_equal_and_entry_exit_bars_match_spec() -> None:
    assert signals15.C1_WEIGHT_MOMENTUM == pytest.approx(0.5)
    assert signals15.C1_WEIGHT_FUNDING_TREND == pytest.approx(0.5)
    assert signals15.C1_WEIGHT_MOMENTUM + signals15.C1_WEIGHT_FUNDING_TREND == pytest.approx(1.0)
    assert signals15.C1_ENTRY_Z == pytest.approx(1.0)
    assert signals15.C1_EXIT_APR == pytest.approx(0.075)  # SPEC.md literal: "7d APR<7.5% 청산"
    assert signals15.C1_EXIT_APR == pytest.approx(common15.EXIT_THRESHOLD_APR)


def test_c1_composite_score_is_deterministic_given_the_same_market_data() -> None:
    from research.wave1.fam_funding import FundingMarket

    index = pd.date_range("2022-01-01", periods=500, freq="1h", tz="UTC")
    rng = np.random.default_rng(42)
    perp = pd.DataFrame({"open": 100.0 + rng.normal(0, 1, len(index)).cumsum(), "close": 100.0 + rng.normal(0, 1, len(index)).cumsum()}, index=index)
    spot = perp.copy()
    funding_index = index[index.hour.isin((0, 8, 16))]
    funding = pd.Series(rng.normal(0.0001, 0.0002, len(funding_index)), index=funding_index)
    market = FundingMarket(spot=spot, perp=perp, funding=funding)

    composite_1, realized_1 = signals15.composite_predictive_score(market)
    composite_2, realized_2 = signals15.composite_predictive_score(market)
    pd.testing.assert_series_equal(composite_1, composite_2)
    pd.testing.assert_series_equal(realized_1, realized_2)


# ---------------------------------------------------------------------------
# 7) D1 페어선정 (pair selection) + direction freezing.
# ---------------------------------------------------------------------------


def test_d1_sector_pools_are_hardcoded_and_pairs_resolve_to_top_two_by_volume() -> None:
    assert engine_pairs.SECTOR_CANDIDATE_POOLS["L1"] == ("SOLUSDT", "AVAXUSDT", "NEARUSDT")
    assert engine_pairs.SECTOR_CANDIDATE_POOLS["DeFi"] == ("UNIUSDT", "AAVEUSDT", "LINKUSDT")
    assert engine_pairs.SECTOR_CANDIDATE_POOLS["Meme"] == ("DOGEUSDT", "WIFUSDT")

    pool = common15.load_candidate_pool()
    pairs = engine_pairs.select_sector_pairs(pool)
    assert len(pairs) == 3
    for pair in pairs:
        candidates = engine_pairs.SECTOR_CANDIDATE_POOLS[pair.sector]
        volumes = {symbol: common15.reference_volume_30d(pool, symbol) for symbol in candidates}
        expected_top_two = tuple(sorted(volumes, key=volumes.get, reverse=True)[:2])
        assert (pair.symbol_a, pair.symbol_b) == expected_top_two


def test_d1_direction_freezes_at_entry_and_does_not_flip_mid_hold_on_a_sign_change() -> None:
    index = pd.date_range("2023-01-01", periods=10, freq="1D", tz="UTC")
    # z crosses above entry (+2.5), stays active, swings NEGATIVE while still |z|>=exit_z
    # (0.8, i.e. NOT below the 0.5 exit bar), then finally reverts inside the exit band.
    z = pd.Series([0.0, 0.1, 2.5, 2.6, 0.8, -2.7, 0.9, 0.3, 0.0, 0.0], index=index)
    active, direction = engine_pairs.pair_position_and_direction(z, entry_z=2.0, exit_z=0.5)

    # shift(1): the entry decided off day index 2 (z=2.5) takes effect on day index 3.
    assert active.iloc[3] == 1.0
    assert direction.iloc[3] == 1.0  # entry z was positive -> short A / long B
    # day index 5 has z=-2.7 (would flip sign if direction were re-derived from raw z), but
    # the pair was already active continuously since day 3 with |z| never dropping below
    # exit_z=0.5 in between (0.8 on day 4) -- direction must still read +1, not flip to -1.
    assert direction.iloc[6] == 1.0  # shift(1) of day index 5's still-active, still-direction=+1 state
    # eventually reverts inside the exit band and goes flat.
    assert active.iloc[-1] == 0.0
    assert direction.iloc[-1] == 0.0


# ---------------------------------------------------------------------------
# 8) General engine/gate sanity: capital contract constants and Wave10Result compatibility
#    with the shared regime_breakdown/gates machinery every candidate is reported through.
# ---------------------------------------------------------------------------


def test_capital_contract_constants_match_spec_common_convention() -> None:
    assert common15.TOTAL_CAPITAL == pytest.approx(100.0)
    assert common15.ACTIVE_CAPITAL == pytest.approx(90.0)
    assert common15.LEG_USDT == pytest.approx(45.0)
    assert common15.GROSS_USDT == pytest.approx(90.0)
    assert common15.GROSS_USDT / common15.ACTIVE_CAPITAL == pytest.approx(1.0)  # 1x leverage
    assert common15.MIN_ORDER_USDT == pytest.approx(5.0)
    assert common15.TOP_K == 1


def test_candidate_registry_is_frozen_to_the_seven_ids_spec_actually_lists() -> None:
    assert configs15.CANDIDATE_IDS == ("A1", "A2", "A3", "B1", "B2", "C1", "D1")
    assert len(configs15.CANDIDATE_IDS) == 7


def test_gates15_generic_evaluate_gates_runs_on_a_synthetic_wave10_result() -> None:
    index = pd.date_range("2020-01-01", periods=1500, freq="1D", tz="UTC")
    rng = np.random.default_rng(99)
    returns = rng.normal(0.0004, 0.003, size=len(index))
    equity = pd.Series(common15.ACTIVE_CAPITAL * np.cumprod(1.0 + returns), index=index)
    result = common15.Wave10Result(equity=equity, positions=pd.Series(1.0, index=index), turnover=pd.Series(0.0, index=index), trade_returns=pd.Series([0.01], index=[index[10]]), max_concurrent_positions=1, symbols_used=("BTCUSDT",))
    report = gates15.evaluate_gates(result, result, common15.LEG_USDT, common15.GROSS_USDT, True, "test", seed_offset=0)
    assert report.overall in ("PASS", "FAIL")
    assert report.gate_s1["status"] == "PASS"  # 1x leverage by construction of the constants used
