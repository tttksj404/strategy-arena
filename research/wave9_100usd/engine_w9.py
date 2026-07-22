# Wave-9 ($100-native) single-leg perpetual engine. See SPEC.md for the frozen
# pre-registration this module implements.
#
# Scope and reuse (task contract: "research/wave9_100usd/ 밖 파일 수정 금지(임포트만 허용)"):
#   - Universe/eligibility: research.wave3.universe.eligible_symbols_at and
#     research.wave3.engine.load_listings/load_markets are imported as-is (read-only)
#     to reuse the exact listing-age (60d) + trailing-30d-volume top-150 selection
#     already validated for wave-3/wave-7. This module only *reimplements* the small
#     cross-sectional momentum/funding-APR scoring formulas locally (same formulas as
#     research.wave3.engine._momentum / _carry_apr) so wave9 never needs a private
#     (underscore-prefixed) import from another wave's module.
#   - Liquidation contract: research.wave4_leverage.sweep.liquidation_loss (and its
#     MAINTENANCE_RATE=0.5%/LIQUIDATION_FEE_RATE=0.06% constants) is imported directly
#     and used unmodified, per the task instruction to reuse wave-4's liquidation
#     contract. wave-4's SYM/ASYM spot-financed notional/margin helpers are *not*
#     reused because they model a paired spot+perp carry trade; wave-9 is single-leg
#     (no spot leg), so notional = margin_dollars * leverage directly.
#   - Data source: the SPEC's hard-constraint line ("Bitget USDT-M") describes the
#     intended *live* venue's perpetual contract type. The only cached crypto
#     perp/funding OHLC in this repo is Binance USDT-M (research/wave3/cache); Bitget
#     cache coverage is limited to tokenized-stock candles. Per the task's explicit
#     "캐시만 사용" instruction, this backtest uses the Binance-sourced cache and the
#     universe is restricted to AssetType.CRYPTO (tokenized stocks excluded) since the
#     strategy family (momentum/funding-harvest/breakout on crypto majors+alts) is
#     crypto-perpetual-native. This is a data-source clarification, not a rule change.
#
# Execution timing (uniform across all 6 candidates, per task instruction):
#   signal confirmed at close(t) using only data through t -> entry/exit fill at
#   open(t+1) -> no lookahead. hold_days counts full daily bars from entry to exit
#   (exit_idx = entry_idx + hold_days).
#
# Position sizing: capital $100, 10% cash buffer -> active fraction 0.90 of *current*
# equity (compounding, not fixed to the original $100). Each concurrent leg gets
# active_fraction / num_legs as its margin; notional = margin * leverage. Concurrent
# legs <= 2 (only W9c uses 2; all others use 1).
#
# Equity/trade-return series are recorded at *trade boundaries* only (entry/exit
# marks), not daily marks -- SPEC's own MC method is "트레이드 수익 부트스트랩"
# (trade-return bootstrap), so the per-trade return is the primary artifact. Within a
# held trade, liquidation is still checked at daily granularity (see _simulate_leg),
# so the boundary-only equity curve does not affect P&L correctness -- it only means
# the reported equity curve is a step function between cycles rather than a smooth
# daily path. This is disclosed in each result's metadata
# ("equity_curve_granularity").

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Final, Literal

import numpy as np
import pandas as pd  # noqa: PANDAS_OK

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.wave1.costs import PERP_TAKER_RATE, slippage_rate
from research.wave3.engine import AssetMarket, load_listings, load_markets
from research.wave3.fetch import WAVE3_CACHE_DIR
from research.wave3.universe import AssetListing, AssetType, eligible_symbols_at
from research.wave4_leverage.sweep import LIQUIDATION_FEE_RATE, MAINTENANCE_RATE, liquidation_loss


TOTAL_CAPITAL: Final = 100.0
CASH_BUFFER_FRACTION: Final = 0.10
ACTIVE_FRACTION: Final = 1.0 - CASH_BUFFER_FRACTION  # 0.90
MIN_ORDER_USDT: Final = 5.0
MAX_CONCURRENT_POSITIONS: Final = 2
MAX_LEVERAGE: Final = 3.0
VOLUME_LIMIT: Final = 150
OOS_SPLIT: Final = pd.Timestamp("2025-09-30T23:59:59Z")  # matches wave3/wave7 IS/OOS split
SLIPPAGE_MAJOR_BP: Final = 1.0
SLIPPAGE_ALT_BP: Final = 3.0

# W9f-only parameters. SPEC.md states "전일 ATR 대비 종가 돌파" without pinning an exact
# ATR window or breakout multiplier. These two values are fixed *before* any candidate
# is run and never adjusted afterward (pre-registration spirit): ATR(14) is the
# conventional default window; multiplier=1.0 means "the day's close-over-close move
# must exceed a full ATR unit" (a genuine breakout, not any positive move).
ATR_WINDOW: Final = 14
ATR_BREAKOUT_MULTIPLIER: Final = 1.0
W9F_UNIVERSE: Final = ("BTCUSDT", "ETHUSDT", "SOLUSDT")

W9_CANDIDATE_IDS: Final = ("W9a", "W9b", "W9c", "W9d", "W9e", "W9f")

Mode = Literal["momentum_top1_long", "momentum_top1_bottom1", "funding_short", "vol_breakout_long"]


@dataclass(frozen=True, slots=True)
class Wave9Error(Exception):
    message: str

    def __str__(self) -> str:
        return self.message


@dataclass(frozen=True, slots=True)
class CandidateConfig:
    candidate_id: str
    mode: Mode
    lookback_days: int  # momentum lookback, or funding-APR window for funding_short
    hold_days: int
    leverage: float
    definition: str
    funding_threshold_apr: float = 0.30


W9_CANDIDATES: Final[tuple[CandidateConfig, ...]] = (
    CandidateConfig("W9a", "momentum_top1_long", 30, 7, 1.0, "집중 모멘텀 롱온리: W3c 랭킹(30d) top-1, 활성자본 100%, 주간 리밸런스, 1x"),
    CandidateConfig("W9b", "momentum_top1_long", 30, 7, 2.0, "W9a + 2x 레버리지"),
    CandidateConfig("W9c", "momentum_top1_bottom1", 30, 7, 1.0, "top-1 롱 + bottom-1 숏 (각 활성자본 50% = $45), 주간, 1x"),
    CandidateConfig("W9d", "momentum_top1_long", 7, 3, 1.0, "초단기 모멘텀: 7d 수익률 top-1, 3일 보유, 1x"),
    CandidateConfig("W9e", "funding_short", 7, 3, 1.0, "단일레그 펀딩 하베스트: 7d 펀딩 APR>30% 심볼 퍼프 숏 (헤지 없음), 3일 보유, 1x", funding_threshold_apr=0.30),
    CandidateConfig("W9f", "vol_breakout_long", 1, 1, 2.0, "변동성 돌파: BTC/ETH/SOL 중 전일 ATR(14) 대비 종가 돌파 심볼 롱, 1일 보유, 2x"),
)


def num_concurrent_positions(mode: Mode) -> int:
    return 2 if mode == "momentum_top1_bottom1" else 1


for _config in W9_CANDIDATES:
    if num_concurrent_positions(_config.mode) > MAX_CONCURRENT_POSITIONS:
        raise Wave9Error(f"{_config.candidate_id}: concurrent positions exceed the {MAX_CONCURRENT_POSITIONS}-position hard cap")
    if _config.leverage > MAX_LEVERAGE:
        raise Wave9Error(f"{_config.candidate_id}: leverage {_config.leverage}x exceeds the {MAX_LEVERAGE}x hard cap")
del _config


# --------------------------------------------------------------------------- data ---


@dataclass(frozen=True, slots=True)
class UniverseData:
    listings: tuple[AssetListing, ...]
    closes: dict[str, pd.Series]       # each reindexed to `calendar`
    ohlc: dict[str, pd.DataFrame]      # each reindexed to `calendar`; columns open/high/low/close
    daily_funding: dict[str, pd.Series]  # each reindexed to `calendar`, NaN filled to 0.0
    quote_volume: pd.DataFrame          # NOT reindexed/filled (eligible_symbols_at needs true gaps)
    calendar: pd.DatetimeIndex


def load_universe(cache_dir: Path = WAVE3_CACHE_DIR) -> dict[str, AssetMarket]:
    """Crypto-only Binance USDT-M perp+funding markets from the wave-3 cache.

    wave9 trades single-leg perps only; the spot frame that
    research.wave3.engine.load_markets also loads (needed upstream only to confirm
    listing eligibility) is present on each AssetMarket but unused here.
    """
    listings = load_listings(cache_dir)
    crypto_listings = tuple(listing for listing in listings if listing.asset_type is AssetType.CRYPTO)
    if not crypto_listings:
        raise Wave9Error("no crypto listings available in the wave-3 cache")
    return load_markets(crypto_listings, cache_dir)


def _daily_ohlc(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.resample("1D").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()


def _daily_close(frame: pd.DataFrame) -> pd.Series:
    return frame["close"].resample("1D").last().dropna()


def _daily_quote_volume(frame: pd.DataFrame) -> pd.Series:
    return frame["quote_volume"].resample("1D").sum(min_count=1)


def build_universe_data(markets: dict[str, AssetMarket]) -> UniverseData:
    if not markets:
        raise Wave9Error("wave-9 market set is empty")
    raw_closes = {symbol: _daily_close(market.perp) for symbol, market in markets.items()}
    raw_ohlc = {symbol: _daily_ohlc(market.perp) for symbol, market in markets.items()}
    volumes = {symbol: _daily_quote_volume(market.perp) for symbol, market in markets.items()}
    quote_frame = pd.DataFrame(volumes).sort_index()
    all_days = sorted(pd.DatetimeIndex(pd.concat(raw_closes.values()).index).unique())
    if not all_days:
        raise Wave9Error("wave-9 universe has no daily bars")
    calendar = pd.DatetimeIndex(all_days)
    closes = {symbol: series.reindex(calendar) for symbol, series in raw_closes.items()}
    ohlc = {symbol: frame.reindex(calendar) for symbol, frame in raw_ohlc.items()}
    daily_funding = {
        symbol: market.funding.resample("1D").sum(min_count=1).reindex(calendar).fillna(0.0)
        for symbol, market in markets.items()
    }
    listings = tuple(market.listing for market in markets.values())
    return UniverseData(listings, closes, ohlc, daily_funding, quote_frame, calendar)


# ---------------------------------------------------------------------- factors ---


def momentum(close: pd.Series, lookback: int) -> pd.Series:
    """Cross-sectional momentum score, identical formula to
    research.wave3.engine._momentum (reimplemented locally; see module docstring)."""
    return close.pct_change(lookback).replace([np.inf, -np.inf], np.nan)


def funding_apr(daily_funding: pd.Series, window_days: int) -> pd.Series:
    """Trailing funding APR, identical formula to research.wave3.engine._carry_apr
    (reimplemented locally with a configurable window; wave3's is fixed at 7d)."""
    return daily_funding.rolling(window_days, min_periods=window_days).mean() * 365.0


def true_range(ohlc: pd.DataFrame) -> pd.Series:
    prior_close = ohlc["close"].shift(1)
    ranges = pd.concat(
        [ohlc["high"] - ohlc["low"], (ohlc["high"] - prior_close).abs(), (ohlc["low"] - prior_close).abs()],
        axis=1,
    )
    return ranges.max(axis=1)


def atr(ohlc: pd.DataFrame, window: int = ATR_WINDOW) -> pd.Series:
    return true_range(ohlc).rolling(window, min_periods=window).mean()


def build_momentum_table(universe: UniverseData, lookback: int) -> pd.DataFrame:
    return pd.DataFrame({symbol: momentum(close, lookback) for symbol, close in universe.closes.items()}).reindex(universe.calendar)


def build_funding_apr_table(universe: UniverseData, window_days: int) -> pd.DataFrame:
    return pd.DataFrame(
        {symbol: funding_apr(series, window_days) for symbol, series in universe.daily_funding.items()}
    ).reindex(universe.calendar)


def build_breakout_table(universe: UniverseData) -> pd.DataFrame:
    """(close[t]-close[t-1]) / ATR[t-1] for each W9F_UNIVERSE symbol -- "전일 ATR 대비
    종가 돌파". The explicit .shift(1) on the ATR series makes the no-lookahead
    property auditable: row t only ever uses ATR computed from bars through t-1."""
    scores: dict[str, pd.Series] = {}
    for symbol in W9F_UNIVERSE:
        ohlc = universe.ohlc.get(symbol)
        if ohlc is None:
            continue
        atr_prior_day = atr(ohlc).shift(1)
        prior_close = ohlc["close"].shift(1)
        scores[symbol] = (ohlc["close"] - prior_close) / atr_prior_day
    return pd.DataFrame(scores).reindex(universe.calendar)


# --------------------------------------------------------------------- selection ---


EligibilityCache = dict[pd.Timestamp, tuple[str, ...]]


def eligible_at(universe: UniverseData, day: pd.Timestamp, cache: EligibilityCache | None = None) -> tuple[str, ...]:
    """Memoized wrapper around research.wave3.universe.eligible_symbols_at.

    That function is expensive (~150ms/call: an internal per-column pd.to_numeric
    pass over the whole quote-volume frame, unrelated to `day`) and does not depend on
    which candidate is asking -- only on `day`. Callers share one `cache` dict across
    all six wave9 candidates within a single run_all() so every unique day is priced
    exactly once no matter how many candidates land on it (see run_candidate/run_all).
    """
    if cache is not None and day in cache:
        return cache[day]
    result = eligible_symbols_at(universe.listings, universe.quote_volume, day, VOLUME_LIMIT)
    if cache is not None:
        cache[day] = result
    return result


def select_targets(
    config: CandidateConfig,
    universe: UniverseData,
    signal_day: pd.Timestamp,
    momentum_table: pd.DataFrame | None,
    funding_table: pd.DataFrame | None,
    breakout_table: pd.DataFrame | None,
    eligibility_cache: EligibilityCache | None = None,
) -> tuple[tuple[str, float], ...]:
    """Return (symbol, direction) legs chosen using only data through `signal_day`."""
    if config.mode in ("momentum_top1_long", "momentum_top1_bottom1"):
        if momentum_table is None or signal_day not in momentum_table.index:
            return ()
        eligible = eligible_at(universe, signal_day, eligibility_cache)
        if not eligible:
            return ()
        row = momentum_table.loc[signal_day, list(eligible)].dropna()
        if row.empty:
            return ()
        top = str(row.idxmax())
        if config.mode == "momentum_top1_long":
            return ((top, 1.0),)
        remaining = row.drop(index=top)
        if remaining.empty:
            return ()
        bottom = str(remaining.idxmin())
        return ((top, 1.0), (bottom, -1.0))
    if config.mode == "funding_short":
        if funding_table is None or signal_day not in funding_table.index:
            return ()
        # Cheap vectorized pre-filter before paying for the expensive eligibility
        # call: skip straight to "no signal" on any day where *no* symbol at all
        # (eligible or not) clears the APR bar -- the >30% APR threshold is rarely
        # met, so this avoids the ~150ms eligibility cost on the vast majority of days.
        row_all = funding_table.loc[signal_day].dropna()
        if row_all[row_all > config.funding_threshold_apr].empty:
            return ()
        eligible = eligible_at(universe, signal_day, eligibility_cache)
        if not eligible:
            return ()
        row = funding_table.loc[signal_day, list(eligible)].dropna()
        qualifying = row[row > config.funding_threshold_apr]
        if qualifying.empty:
            return ()
        return ((str(qualifying.idxmax()), -1.0),)
    if config.mode == "vol_breakout_long":
        if breakout_table is None or signal_day not in breakout_table.index:
            return ()
        row = breakout_table.loc[signal_day].dropna()
        row = row[row > ATR_BREAKOUT_MULTIPLIER]
        if row.empty:
            return ()
        return ((str(row.idxmax()), 1.0),)
    raise Wave9Error(f"unknown wave9 mode: {config.mode}")


# ------------------------------------------------------------------- simulation ---


@dataclass(frozen=True, slots=True)
class LegOutcome:
    feasible: bool
    reason: str | None
    pnl_dollars: float
    fee_dollars: float  # transaction costs only: entry+exit taker+slippage, or entry fee + liquidation fee
    liquidated: bool
    liquidation_day: pd.Timestamp | None
    notional: float
    margin: float

    @property
    def gross_pnl_dollars(self) -> float:
        """P&L before transaction/liquidation fees (price return + funding, including
        a liquidated leg's adverse-move loss but excluding its liquidation fee).
        Used downstream to separate "no edge" failures from "edge eaten by costs"."""
        return self.pnl_dollars + self.fee_dollars


def simulate_leg(
    ohlc: pd.DataFrame,
    daily_funding: pd.Series,
    calendar: pd.DatetimeIndex,
    entry_idx: int,
    exit_idx: int,
    direction: float,
    margin_dollars: float,
    leverage: float,
    symbol: str,
) -> LegOutcome:
    """Simulate one single-leg perp position from entry_idx's open to exit_idx's open.

    Liquidation is checked once per held day (entry_idx..exit_idx-1). The adverse
    fraction is the *cumulative* move from the fixed entry price to that day's own
    worst point (low for a long, high for a short) -- "바 내 최악 역행" (worst move
    within the bar), with the bar's extreme measured against the position's entry
    price rather than that day's own open. This matters for multi-day holds: real
    isolated-margin liquidation is always relative to the price the position was
    opened at (margin erosion is cumulative), not reset every day, so anchoring each
    day's check to that day's own open would silently under-detect a large *gradual*
    adverse move built up over several days without ever tripping a single day's own
    range on its own. On the entry day itself this reduces exactly to the
    open-to-low/high intrabar range, since entry_price *is* that day's open.
    Liquidation is applied via research.wave4_leverage.sweep.liquidation_loss
    unmodified (maintenance 0.5%, liquidation fee 0.06%). Funding is charged/credited
    every held day regardless of strategy (a real perp position always exchanges
    funding); price P&L and the exit fee are only applied if the leg survives to
    exit_idx without liquidating.

    `fee_dollars` isolates the pure transaction-cost component (entry+exit
    taker+slippage, or entry fee + liquidation fee if liquidated) from the market-risk
    component, so the run/report stage can tell "no edge before costs" apart from
    "edge existed but costs consumed it".
    """
    entry_price = float(ohlc["open"].iloc[entry_idx])
    if not np.isfinite(entry_price) or entry_price <= 0.0:
        return LegOutcome(False, "missing_entry_price", 0.0, 0.0, False, None, 0.0, margin_dollars)
    notional = margin_dollars * leverage
    initial_margin = margin_dollars
    if notional < MIN_ORDER_USDT:
        return LegOutcome(False, "below_min_order", 0.0, 0.0, False, None, notional, initial_margin)
    fee_rate = PERP_TAKER_RATE + slippage_rate(symbol)
    entry_fee = notional * fee_rate
    pnl = -entry_fee  # entry cost
    fee_dollars = entry_fee
    liquidated = False
    liquidation_day: pd.Timestamp | None = None
    last_idx = entry_idx
    for day_idx in range(entry_idx, exit_idx):
        day = calendar[day_idx]
        bar = ohlc.iloc[day_idx]
        rate = float(daily_funding.iloc[day_idx]) if day_idx < len(daily_funding) else 0.0
        if np.isfinite(rate):
            pnl += -notional * rate * direction
        worst_price = bar["low"] if direction > 0.0 else bar["high"]
        if not np.isfinite(worst_price) or worst_price <= 0.0:
            last_idx = day_idx
            continue  # data gap mid-hold: no price/liquidation signal available this day
        adverse_fraction = max(0.0, -direction * (worst_price / entry_price - 1.0))
        loss = liquidation_loss(notional, adverse_fraction, initial_margin)
        if loss is not None:
            liquidated = True
            liquidation_day = day
            liquidation_fee = notional * LIQUIDATION_FEE_RATE
            pnl -= loss
            fee_dollars += liquidation_fee
            last_idx = day_idx
            break
        last_idx = day_idx
    if not liquidated:
        exit_price = float(ohlc["open"].iloc[exit_idx])
        if not np.isfinite(exit_price) or exit_price <= 0.0:
            exit_price = float(ohlc["close"].iloc[last_idx])
        if np.isfinite(exit_price) and exit_price > 0.0:
            price_return = direction * (exit_price / entry_price - 1.0)
            pnl += notional * price_return
        exit_fee = notional * fee_rate
        pnl -= exit_fee  # exit cost
        fee_dollars += exit_fee
    return LegOutcome(True, None, float(pnl), float(fee_dollars), liquidated, liquidation_day, notional, initial_margin)


@dataclass(frozen=True, slots=True)
class TradeRecord:
    entry_day: pd.Timestamp
    exit_day: pd.Timestamp
    legs: tuple[str, ...]
    directions: tuple[float, ...]
    pnl_dollars: float
    gross_pnl_dollars: float  # pnl_dollars + fee_dollars (before transaction/liquidation fees)
    fee_dollars: float
    equity_before: float
    equity_after: float
    liquidated_legs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Wave9Result:
    candidate_id: str
    config: CandidateConfig
    equity: pd.Series           # trade-boundary marks only; starts at $100
    trade_returns: pd.Series    # one point per completed trade, indexed at exit day
    trades: tuple[TradeRecord, ...]
    metadata: dict


def run_candidate(
    config: CandidateConfig,
    universe: UniverseData,
    momentum_tables: dict[int, pd.DataFrame],
    funding_table: pd.DataFrame | None,
    breakout_table: pd.DataFrame | None,
    eligibility_cache: EligibilityCache | None = None,
) -> Wave9Result:
    calendar = universe.calendar
    n = len(calendar)
    if n < 2:
        raise Wave9Error("wave-9 calendar is too short to trade")
    if eligibility_cache is None:
        eligibility_cache = {}
    momentum_table = momentum_tables.get(config.lookback_days)
    equity = TOTAL_CAPITAL
    equity_points: list[tuple[pd.Timestamp, float]] = [(calendar[0], TOTAL_CAPITAL)]
    trade_return_points: list[tuple[pd.Timestamp, float]] = []
    trades: list[TradeRecord] = []
    infeasible_cycles = 0
    liquidation_events = 0
    no_signal_days = 0
    num_legs = num_concurrent_positions(config.mode)

    signal_idx = 0
    while signal_idx < n - 1:
        signal_day = calendar[signal_idx]
        entry_idx = signal_idx + 1
        targets = select_targets(config, universe, signal_day, momentum_table, funding_table, breakout_table, eligibility_cache)
        if not targets:
            no_signal_days += 1
            signal_idx += 1
            continue
        exit_idx = min(entry_idx + config.hold_days, n - 1)
        if exit_idx <= entry_idx:
            break  # out of calendar

        margin_each = (ACTIVE_FRACTION / len(targets)) * equity
        equity_before = equity
        total_pnl = 0.0
        total_gross_pnl = 0.0
        total_fees = 0.0
        liquidated_symbols: list[str] = []
        applied_legs: list[tuple[str, float]] = []
        for symbol, direction in targets:
            ohlc = universe.ohlc.get(symbol)
            daily_funding = universe.daily_funding.get(symbol)
            if ohlc is None or daily_funding is None:
                infeasible_cycles += 1
                continue
            outcome = simulate_leg(ohlc, daily_funding, calendar, entry_idx, exit_idx, direction, margin_each, config.leverage, symbol)
            if not outcome.feasible:
                infeasible_cycles += 1
                continue
            applied_legs.append((symbol, direction))
            total_pnl += outcome.pnl_dollars
            total_gross_pnl += outcome.gross_pnl_dollars
            total_fees += outcome.fee_dollars
            if outcome.liquidated:
                liquidated_symbols.append(symbol)
                liquidation_events += 1

        if not applied_legs:
            signal_idx = exit_idx
            continue

        equity = max(0.0, equity + total_pnl)
        exit_day = calendar[exit_idx]
        equity_points.append((exit_day, equity))
        trade_return = (total_pnl / equity_before) if equity_before > 0.0 else 0.0
        trade_return_points.append((exit_day, trade_return))
        trades.append(
            TradeRecord(
                entry_day=calendar[entry_idx],
                exit_day=exit_day,
                legs=tuple(symbol for symbol, _ in applied_legs),
                directions=tuple(direction for _, direction in applied_legs),
                pnl_dollars=total_pnl,
                gross_pnl_dollars=total_gross_pnl,
                fee_dollars=total_fees,
                equity_before=equity_before,
                equity_after=equity,
                liquidated_legs=tuple(liquidated_symbols),
            )
        )
        signal_idx = exit_idx
        if equity <= 0.0:
            break

    equity_series = pd.Series(dict(equity_points), dtype=float).sort_index()
    equity_series.index.name = "timestamp"
    trade_series = pd.Series(dict(trade_return_points), dtype=float).sort_index()
    trade_series.index.name = "timestamp"

    min_notional_at_start = (ACTIVE_FRACTION / num_legs) * TOTAL_CAPITAL * config.leverage
    gross_fraction_at_start = ACTIVE_FRACTION * config.leverage
    metadata = {
        "mode": config.mode,
        "lookback_days": config.lookback_days,
        "hold_days": config.hold_days,
        "leverage": config.leverage,
        "num_concurrent_positions": num_legs,
        "single_leg": True,
        "min_notional_at_start": min_notional_at_start,
        "gross_fraction_at_start": gross_fraction_at_start,
        "min_order_usdt": MIN_ORDER_USDT,
        "trades_executed": len(trades),
        "no_signal_days": no_signal_days,
        "infeasible_cycles": infeasible_cycles,
        "liquidation_events": liquidation_events,
        "final_equity": float(equity),
        "universe_size_crypto": len(universe.closes),
        "funding_applied": True,
        "atr_window": ATR_WINDOW if config.mode == "vol_breakout_long" else None,
        "atr_breakout_multiplier": ATR_BREAKOUT_MULTIPLIER if config.mode == "vol_breakout_long" else None,
        "funding_threshold_apr": config.funding_threshold_apr if config.mode == "funding_short" else None,
        "cost_model": {
            "taker_fee_rate": PERP_TAKER_RATE,
            "slippage_major_bp": SLIPPAGE_MAJOR_BP,
            "slippage_alt_bp": SLIPPAGE_ALT_BP,
            "maker_assumed": False,
        },
        "liquidation_model": {
            "maintenance_rate": MAINTENANCE_RATE,
            "liquidation_fee_rate": LIQUIDATION_FEE_RATE,
            "source": "research.wave4_leverage.sweep.liquidation_loss (unmodified import)",
        },
        "execution_timing": (
            "signal confirmed at close(t) using data through t only; entry/exit fills "
            "at open(t+1); liquidation checked once per held day using the cumulative "
            "adverse move from the fixed entry price to that day's own low/high "
            "(worst point reached within the bar, measured against entry -- not that "
            "day's own open, so a gradual multi-day adverse move is still caught)"
        ),
        "equity_curve_granularity": "trade_boundary (entry/exit marks only); liquidation itself is still checked daily within each hold",
        "cost_conservatism_note": "every cycle pays a fresh entry+exit fee even if the same symbol re-qualifies next cycle (no position carry-through optimization)",
        "data_source": (
            "research/wave3/cache (Binance USDT-M perp OHLC + funding), crypto-only "
            "(AssetType.CRYPTO; tokenized stocks excluded). SPEC.md's 'Bitget USDT-M' "
            "describes the intended live venue's perpetual contract type, not this "
            "backtest's cached data source -- no Bitget crypto-perp OHLC cache exists in this repo."
        ),
    }
    return Wave9Result(config.candidate_id, config, equity_series, trade_series, tuple(trades), metadata)


# --------------------------------------------------------------------- payloads ---


def _series_payload(series: pd.Series) -> list[dict]:
    return [{"timestamp": pd.Timestamp(idx).isoformat(), "value": float(value)} for idx, value in series.items()]


def series_from_payload(items: list[dict]) -> pd.Series:
    data = {pd.Timestamp(item["timestamp"]): float(item["value"]) for item in items}
    series = pd.Series(data, dtype=float).sort_index()
    series.index.name = "timestamp"
    return series


def _trade_payload(trade: TradeRecord) -> dict:
    return {
        "entry_day": pd.Timestamp(trade.entry_day).isoformat(),
        "exit_day": pd.Timestamp(trade.exit_day).isoformat(),
        "legs": list(trade.legs),
        "directions": list(trade.directions),
        "pnl_dollars": trade.pnl_dollars,
        "gross_pnl_dollars": trade.gross_pnl_dollars,
        "fee_dollars": trade.fee_dollars,
        "equity_before": trade.equity_before,
        "equity_after": trade.equity_after,
        "liquidated_legs": list(trade.liquidated_legs),
    }


def to_result_payload(result: Wave9Result) -> dict:
    return {
        "candidate_id": result.candidate_id,
        "family": "F9",
        "definition": result.config.definition,
        "candidate_config": {
            "mode": result.config.mode,
            "lookback_days": result.config.lookback_days,
            "hold_days": result.config.hold_days,
            "leverage": result.config.leverage,
            "funding_threshold_apr": result.config.funding_threshold_apr,
        },
        "equity": _series_payload(result.equity),
        "trade_returns": _series_payload(result.trade_returns),
        "trades": [_trade_payload(trade) for trade in result.trades],
        "metadata": result.metadata,
    }


def run_all(cache_dir: Path = WAVE3_CACHE_DIR) -> dict[str, dict]:
    """Load the universe once, build the shared factor tables once, run all six
    frozen candidates, and return {candidate_id: json_payload}."""
    markets = load_universe(cache_dir)
    universe = build_universe_data(markets)
    momentum_tables = {30: build_momentum_table(universe, 30), 7: build_momentum_table(universe, 7)}
    funding_table = build_funding_apr_table(universe, 7)
    breakout_table = build_breakout_table(universe)
    eligibility_cache: EligibilityCache = {}  # shared across all 6 candidates -- see eligible_at's docstring
    payloads: dict[str, dict] = {}
    for config in W9_CANDIDATES:
        result = run_candidate(config, universe, momentum_tables, funding_table, breakout_table, eligibility_cache)
        payloads[config.candidate_id] = to_result_payload(result)
    return payloads


__all__ = [
    "ACTIVE_FRACTION",
    "ATR_BREAKOUT_MULTIPLIER",
    "ATR_WINDOW",
    "CASH_BUFFER_FRACTION",
    "MAX_CONCURRENT_POSITIONS",
    "MAX_LEVERAGE",
    "MIN_ORDER_USDT",
    "OOS_SPLIT",
    "TOTAL_CAPITAL",
    "VOLUME_LIMIT",
    "W9F_UNIVERSE",
    "W9_CANDIDATES",
    "W9_CANDIDATE_IDS",
    "CandidateConfig",
    "EligibilityCache",
    "LegOutcome",
    "TradeRecord",
    "UniverseData",
    "Wave9Error",
    "Wave9Result",
    "atr",
    "build_breakout_table",
    "build_funding_apr_table",
    "build_momentum_table",
    "build_universe_data",
    "eligible_at",
    "funding_apr",
    "load_universe",
    "momentum",
    "num_concurrent_positions",
    "run_all",
    "run_candidate",
    "select_targets",
    "series_from_payload",
    "simulate_leg",
    "to_result_payload",
    "true_range",
]
