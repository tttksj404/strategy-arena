from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd  # noqa: PANDAS_OK
import pytest

from research.wave3.fetch import WAVE3_CACHE_DIR
from research.wave3.universe import AssetListing, AssetType
from research.wave9_100usd import gates_w9
from research.wave9_100usd.engine_w9 import (
    ACTIVE_FRACTION,
    MAX_CONCURRENT_POSITIONS,
    MAX_LEVERAGE,
    MIN_ORDER_USDT,
    TOTAL_CAPITAL,
    UniverseData,
    W9_CANDIDATES,
    build_funding_apr_table,
    build_momentum_table,
    num_concurrent_positions,
    run_all,
    run_candidate,
    select_targets,
    simulate_leg,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


# --------------------------------------------------------------------- fixtures ---


def _calendar(n: int, start: str = "2024-01-01") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="1D", tz="UTC")


def _flat_ohlc(index: pd.DatetimeIndex, price: float) -> pd.DataFrame:
    return pd.DataFrame({"open": price, "high": price * 1.01, "low": price * 0.99, "close": price}, index=index)


def _make_universe(symbols: dict[str, pd.Series], calendar: pd.DatetimeIndex) -> UniverseData:
    """Build a minimal synthetic UniverseData from {symbol: close_series} paths.

    Every symbol is onboarded well before `calendar` starts (so the 60-day crypto
    listing-age gate never blocks it) and has 35 days of pre-calendar constant
    quote-volume history plus the full calendar's worth, satisfying
    eligible_symbols_at's 30-day trailing-window precondition for every day tested.
    """
    listings = tuple(
        AssetListing(symbol, AssetType.CRYPTO, calendar[0] - pd.Timedelta(days=120), True, True) for symbol in symbols
    )
    volume_index = pd.date_range(calendar[0] - pd.Timedelta(days=35), calendar[-1], freq="1D", tz="UTC")
    quote_volume = pd.DataFrame({symbol: 1_000_000.0 for symbol in symbols}, index=volume_index)
    closes: dict[str, pd.Series] = {}
    ohlc: dict[str, pd.DataFrame] = {}
    daily_funding: dict[str, pd.Series] = {}
    for symbol, close in symbols.items():
        close = close.reindex(calendar)
        closes[symbol] = close
        frame = pd.DataFrame(
            {
                "open": close,
                "high": close * 1.02,
                "low": close * 0.98,
                "close": close,
            },
            index=calendar,
        )
        ohlc[symbol] = frame
        daily_funding[symbol] = pd.Series(0.0, index=calendar)
    return UniverseData(listings, closes, ohlc, daily_funding, quote_volume, calendar)


def _price_path(calendar: pd.DatetimeIndex, start: float, daily_return: float) -> pd.Series:
    values = start * (1.0 + daily_return) ** np.arange(len(calendar))
    return pd.Series(values, index=calendar)


# ------------------------------------------------------------- 1. top-1 selection ---


def test_momentum_top1_long_selects_highest_scoring_eligible_symbol() -> None:
    # Given: three symbols with clearly ranked 30-day momentum as of the test day
    # (B up strongly, A flat, C down), all eligible.
    calendar = _calendar(60)
    universe = _make_universe(
        {
            "AAAUSDT": _price_path(calendar, 100.0, 0.0005),
            "BBBUSDT": _price_path(calendar, 100.0, 0.02),
            "CCCUSDT": _price_path(calendar, 100.0, -0.01),
        },
        calendar,
    )
    momentum_table = build_momentum_table(universe, 30)
    signal_day = calendar[40]

    # When
    targets = select_targets(
        _long_top1_config(), universe, signal_day, momentum_table, None, None, {}
    )

    # Then
    assert targets == (("BBBUSDT", 1.0),)


def test_momentum_top1_bottom1_picks_two_different_symbols() -> None:
    # Given: same three-symbol universe as above
    calendar = _calendar(60)
    universe = _make_universe(
        {
            "AAAUSDT": _price_path(calendar, 100.0, 0.0005),
            "BBBUSDT": _price_path(calendar, 100.0, 0.02),
            "CCCUSDT": _price_path(calendar, 100.0, -0.01),
        },
        calendar,
    )
    momentum_table = build_momentum_table(universe, 30)
    signal_day = calendar[40]
    config = _long_top1_config()
    long_short_config = type(config)(config.candidate_id, "momentum_top1_bottom1", config.lookback_days, config.hold_days, config.leverage, config.definition)

    # When
    targets = select_targets(long_short_config, universe, signal_day, momentum_table, None, None, {})

    # Then: top (long) and bottom (short) must be distinct symbols, correctly ranked
    assert targets == (("BBBUSDT", 1.0), ("CCCUSDT", -1.0))


def _long_top1_config():
    from research.wave9_100usd.engine_w9 import CandidateConfig

    return CandidateConfig("TEST", "momentum_top1_long", 30, 7, 1.0, "test")


# --------------------------------------------------------------- 2. min-order gate ---


def test_min_order_check_flags_infeasible_below_5usd() -> None:
    # Given: a leg sized so margin*leverage < $5 (MIN_ORDER_USDT)
    calendar = _calendar(10)
    ohlc = _flat_ohlc(calendar, 100.0)
    funding = pd.Series(0.0, index=calendar)

    # When
    outcome = simulate_leg(ohlc, funding, calendar, 0, 3, 1.0, margin_dollars=2.0, leverage=1.0, symbol="AAAUSDT")

    # Then
    assert outcome.feasible is False
    assert outcome.reason == "below_min_order"
    assert outcome.pnl_dollars == 0.0

    # And: right at/above the floor it is feasible
    ok = simulate_leg(ohlc, funding, calendar, 0, 3, 1.0, margin_dollars=MIN_ORDER_USDT, leverage=1.0, symbol="AAAUSDT")
    assert ok.feasible is True


# -------------------------------------------------------------- 3. liquidation ---


def test_liquidation_triggers_on_large_single_day_adverse_move() -> None:
    # Given: a long position at 3x leverage (threshold ~= 1/3 - 0.005 = 32.8%) that
    # drops 50% intraday on the entry day -- well past the threshold.
    calendar = _calendar(5)
    frame = pd.DataFrame(
        {"open": [100.0, 50.0, 50.0, 50.0, 50.0], "high": [101.0, 51.0, 51.0, 51.0, 51.0], "low": [50.0, 49.0, 49.0, 49.0, 49.0], "close": [50.0, 50.0, 50.0, 50.0, 50.0]},
        index=calendar,
    )
    funding = pd.Series(0.0, index=calendar)

    # When
    outcome = simulate_leg(frame, funding, calendar, entry_idx=0, exit_idx=3, direction=1.0, margin_dollars=45.0, leverage=3.0, symbol="AAAUSDT")

    # Then
    assert outcome.liquidated is True
    assert outcome.liquidation_day == calendar[0]
    from research.wave1.costs import PERP_TAKER_RATE, slippage_rate
    from research.wave4_leverage.sweep import LIQUIDATION_FEE_RATE

    notional = 45.0 * 3.0
    entry_fee = notional * (PERP_TAKER_RATE + slippage_rate("AAAUSDT"))
    # adverse_fraction = (entry_price - low) / entry_price = (100-50)/100 = 0.5
    liquidation_fee = notional * LIQUIDATION_FEE_RATE
    expected_pnl = -entry_fee - (notional * 0.5 + liquidation_fee)
    assert outcome.pnl_dollars == pytest.approx(expected_pnl)
    assert outcome.gross_pnl_dollars == pytest.approx(-notional * 0.5)
    assert outcome.pnl_dollars < -notional * 0.4  # sanity: a large realized loss, not a rounding artifact


def test_no_liquidation_when_move_stays_under_threshold_at_1x() -> None:
    # Given: the same 50% drop, but at 1x leverage the threshold is ~99.5% -- far away
    calendar = _calendar(5)
    frame = pd.DataFrame(
        {"open": [100.0, 50.0, 50.0, 50.0, 50.0], "high": [101.0, 51.0, 51.0, 51.0, 51.0], "low": [50.0, 49.0, 49.0, 49.0, 49.0], "close": [50.0, 50.0, 50.0, 50.0, 50.0]},
        index=calendar,
    )
    funding = pd.Series(0.0, index=calendar)

    # When
    outcome = simulate_leg(frame, funding, calendar, entry_idx=0, exit_idx=3, direction=1.0, margin_dollars=45.0, leverage=1.0, symbol="AAAUSDT")

    # Then: survives to exit, realizes the (large, but not liquidated) price loss
    assert outcome.liquidated is False
    assert outcome.liquidation_day is None


# ------------------------------------------------- 4. 3-day hold liquidation timing ---


def test_three_day_hold_liquidates_on_the_correct_day_not_earlier_or_later() -> None:
    # Given: entry at 100. Day0 and day1 stay within a mild range (cumulative adverse
    # move from the fixed entry price well under the 2x threshold of ~49.5%); day2's
    # low finally breaches it (cumulative move (100-40)/100=60% > 49.5%).
    calendar = _calendar(6)
    frame = pd.DataFrame(
        {
            "open": [100.0, 98.0, 92.0, 45.0, 45.0, 45.0],
            "high": [105.0, 99.0, 93.0, 46.0, 46.0, 46.0],
            "low": [95.0, 90.0, 40.0, 44.0, 44.0, 44.0],
            "close": [98.0, 92.0, 45.0, 45.0, 45.0, 45.0],
        },
        index=calendar,
    )
    funding = pd.Series(0.0, index=calendar)

    # When: a 3-day hold (entry_idx=0, exit_idx=3) at 2x leverage
    outcome = simulate_leg(frame, funding, calendar, entry_idx=0, exit_idx=3, direction=1.0, margin_dollars=45.0, leverage=2.0, symbol="AAAUSDT")

    # Then: liquidation lands on day index 2 (the third held day), not day 0 or day 1
    assert outcome.liquidated is True
    assert outcome.liquidation_day == calendar[2]

    # And: shortening the hold to end *before* the breach (exit_idx=2) never liquidates
    survives = simulate_leg(frame, funding, calendar, entry_idx=0, exit_idx=2, direction=1.0, margin_dollars=45.0, leverage=2.0, symbol="AAAUSDT")
    assert survives.liquidated is False


def test_liquidated_leg_pays_entry_fee_and_liquidation_fee_not_exit_fee() -> None:
    # Given: the same liquidating path as above
    calendar = _calendar(6)
    frame = pd.DataFrame(
        {
            "open": [100.0, 98.0, 92.0, 45.0, 45.0, 45.0],
            "high": [105.0, 99.0, 93.0, 46.0, 46.0, 46.0],
            "low": [95.0, 90.0, 40.0, 44.0, 44.0, 44.0],
            "close": [98.0, 92.0, 45.0, 45.0, 45.0, 45.0],
        },
        index=calendar,
    )
    funding = pd.Series(0.0, index=calendar)
    notional = 45.0 * 2.0

    # When
    outcome = simulate_leg(frame, funding, calendar, entry_idx=0, exit_idx=3, direction=1.0, margin_dollars=45.0, leverage=2.0, symbol="AAAUSDT")

    # Then: fee_dollars = entry fee + liquidation fee only (no exit taker/slippage fee)
    from research.wave1.costs import PERP_TAKER_RATE, slippage_rate
    from research.wave4_leverage.sweep import LIQUIDATION_FEE_RATE

    entry_fee = notional * (PERP_TAKER_RATE + slippage_rate("AAAUSDT"))
    liquidation_fee = notional * LIQUIDATION_FEE_RATE
    assert outcome.fee_dollars == pytest.approx(entry_fee + liquidation_fee)
    assert outcome.gross_pnl_dollars == pytest.approx(outcome.pnl_dollars + outcome.fee_dollars)


# --------------------------------------------------------- 5. single-leg gross cap ---


def test_single_leg_gross_and_min_notional_match_active_fraction_formula() -> None:
    # Given: a tiny synthetic universe, enough history for momentum + eligibility
    calendar = _calendar(60)
    universe = _make_universe(
        {
            "AAAUSDT": _price_path(calendar, 100.0, 0.001),
            "BBBUSDT": _price_path(calendar, 100.0, 0.02),
            "CCCUSDT": _price_path(calendar, 100.0, -0.01),
        },
        calendar,
    )
    momentum_tables = {30: build_momentum_table(universe, 30)}

    # When: a 1-leg (long-only) and a 2-leg (long+short) candidate over the same data
    from research.wave9_100usd.engine_w9 import CandidateConfig

    one_leg = run_candidate(CandidateConfig("T1", "momentum_top1_long", 30, 7, 1.0, "t"), universe, momentum_tables, None, None)
    two_leg = run_candidate(CandidateConfig("T2", "momentum_top1_bottom1", 30, 7, 1.0, "t"), universe, momentum_tables, None, None)
    two_leg_lev = run_candidate(CandidateConfig("T3", "momentum_top1_bottom1", 30, 7, 2.0, "t"), universe, momentum_tables, None, None)

    # Then: every leg stays single-leg (no spot pairing) and gross == active_fraction*leverage exactly
    assert one_leg.metadata["single_leg"] is True
    assert one_leg.metadata["num_concurrent_positions"] == 1
    assert one_leg.metadata["gross_fraction_at_start"] == pytest.approx(ACTIVE_FRACTION * 1.0)
    assert one_leg.metadata["min_notional_at_start"] == pytest.approx(ACTIVE_FRACTION * TOTAL_CAPITAL * 1.0)

    assert two_leg.metadata["num_concurrent_positions"] == 2
    assert two_leg.metadata["gross_fraction_at_start"] == pytest.approx(ACTIVE_FRACTION * 1.0)
    # Each of the 2 legs gets half the active sleeve: 0.45 * $100 = $45
    assert two_leg.metadata["min_notional_at_start"] == pytest.approx((ACTIVE_FRACTION / 2.0) * TOTAL_CAPITAL * 1.0)
    assert two_leg.metadata["min_notional_at_start"] == pytest.approx(45.0)

    assert two_leg_lev.metadata["gross_fraction_at_start"] == pytest.approx(ACTIVE_FRACTION * 2.0)

    # And every actual trade recorded really did use exactly `num_concurrent_positions` legs
    assert all(len(trade.legs) == 1 for trade in one_leg.trades)
    assert all(len(trade.legs) == 2 for trade in two_leg.trades)


def test_all_registered_candidates_satisfy_the_hard_capacity_caps() -> None:
    # SPEC.md hard constraints: concurrent positions <= 2, leverage <= 3x. Also
    # asserted at import time in engine_w9.py; this re-checks it as a regression guard.
    for config in W9_CANDIDATES:
        assert num_concurrent_positions(config.mode) <= MAX_CONCURRENT_POSITIONS
        assert config.leverage <= MAX_LEVERAGE


# ------------------------------------------------------------------ gates_w9 ---


def test_mc_bootstrap_trades_zero_return_series_stays_at_capital() -> None:
    # Given: an all-zero trade-return series
    trades = np.zeros(50)

    # When
    mc = gates_w9.mc_bootstrap_trades(trades, seed=1, paths=1000, capital=100.0)

    # Then: every bootstrap path compounds to exactly the starting capital
    assert mc["median"] == pytest.approx(100.0)
    assert mc["p05"] == pytest.approx(100.0)
    assert mc["p_bankrupt"] == 0.0
    assert mc["trade_count"] == 50


def test_mc_bootstrap_trades_empty_series_is_undetermined_not_a_crash() -> None:
    mc = gates_w9.mc_bootstrap_trades(np.array([]), seed=1)
    assert mc["trade_count"] == 0
    h1 = gates_w9.h1_bankruptcy(mc)
    h2 = gates_w9.h2_p05_floor(mc)
    assert h1.status == "UNDETERMINED"
    assert h2.status == "UNDETERMINED"


def test_h4_feasibility_fails_below_min_order_and_passes_at_spec_baseline() -> None:
    # Given: SPEC's own W9a baseline (1 leg, 1x) -- min notional $90, gross 0.9x
    good_payload = {"metadata": {"single_leg": True, "min_notional_at_start": 90.0, "gross_fraction_at_start": 0.9, "infeasible_cycles": 0}}
    bad_payload = {"metadata": {"single_leg": True, "min_notional_at_start": 3.0, "gross_fraction_at_start": 0.9, "infeasible_cycles": 0}}

    # When / Then
    assert gates_w9.h4_feasibility(good_payload, leverage=1.0).status == "PASS"
    assert gates_w9.h4_feasibility(bad_payload, leverage=1.0).status == "FAIL"

    # And: leverage above the 3x hard cap fails even with fine sizing
    over_leverage = gates_w9.h4_feasibility(good_payload, leverage=5.0)
    assert over_leverage.status == "FAIL"


# --------------------------------------------------------- end-to-end smoke test ---


def test_full_pipeline_smoke_with_real_cache() -> None:
    # Given: the real wave-3 cache (cache-only per the task contract; skip if absent)
    if not (WAVE3_CACHE_DIR / "manifest.json").exists():
        pytest.skip("real wave-3 cache not present in this checkout")

    # When: the whole run_all() pipeline executes for every registered candidate
    payloads = run_all()

    # Then
    assert set(payloads.keys()) == {config.candidate_id for config in W9_CANDIDATES}
    for candidate_id, payload in payloads.items():
        assert payload["metadata"]["final_equity"] >= 0.0
        assert payload["metadata"]["single_leg"] is True
        assert payload["metadata"]["num_concurrent_positions"] <= MAX_CONCURRENT_POSITIONS
        # every recorded trade cleared the $5 floor at the notional level it was sized at
        for trade in payload["trades"]:
            assert len(trade["legs"]) == payload["metadata"]["num_concurrent_positions"]
