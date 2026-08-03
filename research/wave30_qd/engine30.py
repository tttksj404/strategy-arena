# Wave-30 leveraged single-instrument engine on 1h bars.
#
# ---------------------------------------------------------------------------------------
# What this engine does differently from wave29_lev10 (the repo's only prior leverage model)
# ---------------------------------------------------------------------------------------
# wave29 took the ALREADY-FIXED trade list of wave20's V1 and asked "what if each of those
# trades had been run at Nx?", declaring a liquidation whenever the trade's measured MAE
# exceeded the band. That answers a counterfactual about a strategy that was never designed
# for leverage, and it necessarily finds ruin: V1's stop is nowhere near 4.5% wide.
#
# Here the stop is INSIDE the liquidation band by construction (genome30 enforces
# stop_pct <= 0.9*liq_band, fail-closed). So MAE exceeding the band is no longer sufficient
# for liquidation -- the stop would have fired first. Liquidation requires the price to GAP
# THROUGH the stop far enough in a single 1h bar to reach the band, which this engine detects
# from the real bar OPEN:
#
#     exit_price = stop_price          if the bar opened on the safe side of the stop
#                = bar open            if the bar opened already past the stop (a real gap)
#
# and only then, if that fill is at or beyond the band, is the margin declared wiped. This is
# strictly more honest than assuming stop fills are always exact, and strictly less punitive
# than wave29's MAE rule -- and both differences are measured, not assumed.
#
# ---------------------------------------------------------------------------------------
# Same-bar stop/target ambiguity is resolved AGAINST the strategy
# ---------------------------------------------------------------------------------------
# When one 1h bar's range contains both the stop and the take-profit, the true intrabar order
# is unknowable from OHLC. This engine always assumes the STOP filled first. That biases
# every reported return DOWNWARD, which is the correct direction for a leverage study.
#
# ---------------------------------------------------------------------------------------
# t -> t+1 execution discipline
# ---------------------------------------------------------------------------------------
# Signals are evaluated on bar i's CLOSE using features that never touch bar i's own high or
# low (dataio30._prior_extreme shifts the breakout channel by one bar) and fill at bar i+1's
# OPEN -- the same convention wave9_100usd.engine_w9 and wave20_convex.engine20 state for
# their own breakout tables.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Final

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

import numpy as np

from research.wave30_qd.dataio30 import MarketCache, OOSLeakageError, SymbolArrays
from research.wave30_qd.genome30 import Genome

TOTAL_CAPITAL: Final = 100.0
MIN_ORDER_USDT: Final = 5.0
SLEEVE_DEAD_THRESHOLD: Final = 0.005  # below half a cent the sleeve is dead, stop trading


@dataclass(frozen=True)
class SizingMode:
    """How much capital each position risks.

    `compounding` (the default, and what wave30/31/32 all used) sizes every position off the
    CURRENT sleeve equity, so results are a growth multiple.

    `fixed_base=True` sizes every position off the STARTING sleeve forever. wave33 needs this
    for two independent reasons:
      1. The question "how many dollars does one entry make" has no answer under compounding --
         trade #900's dollars are not comparable to trade #1's. Fixing the base makes per-trade
         P&L a single common unit, which is the only way to test a "$10 per entry" requirement.
      2. It removes the capacity fiction that inflated wave30 (notional reached $1.9M against a
         cost model fitted at $45). A fixed $100 base keeps every order the same small size.
    Equity is still tracked and still floors at zero: a fixed-size loser can drain the account,
    and when it does, trading stops. So this is not a way to hide ruin -- it exposes it.
    """

    fixed_base: bool = False


COMPOUNDING: Final = SizingMode()


@dataclass(frozen=True)
class ExecutionStress:
    """Optional adverse-execution overlay. Defaults are a strict no-op.

    Added for wave32, whose deployment candidate has a 1.46% stop -- at 3.08x leverage the
    round-trip taker cost alone is 25% of that stop distance, so the honest question is not
    "what if slippage doubles" but "what if the STOP itself fills worse than its trigger".
    `stop_slippage` therefore worsens the fill on MARKET exits only (stop and forced max-hold
    close); a take-profit is a resting limit order and does not suffer adverse slippage, so it
    is deliberately excluded. A worsened stop fill also feeds the liquidation test, because a
    bad fill really can push the realised loss past the maintenance band.

    With cost_multiplier=1.0 and stop_slippage=0.0 every arithmetic operation is a
    multiplication by exactly 1.0, so pre-existing wave30/wave31 results are reproduced
    bit-for-bit (pinned by tests and by verify31.py's trade-for-trade check).
    """

    cost_multiplier: float = 1.0
    stop_slippage: float = 0.0


NO_STRESS: Final = ExecutionStress()


@dataclass(frozen=True)
class Wave30Trade:
    symbol: str
    direction: float
    entry_bar: int
    exit_bar: int
    entry_price: float
    exit_price: float
    leverage: float
    notional_usdt: float
    base_usdt: float  # capital committed to this position (its own compounding base)
    gross_price_return: float  # direction * (exit/entry - 1), before leverage
    funding_fraction: float  # funding PAID as a fraction of notional (negative = received)
    cost_fraction: float  # round-trip fee+slippage as a fraction of notional
    net_return_on_base: float  # what actually multiplied base_usdt, floored at -1.0
    mae: float  # max adverse excursion while open, fraction of entry price
    exit_reason: str
    liquidated: bool


@dataclass(frozen=True)
class Wave30Result:
    genome: Genome
    trades: tuple[Wave30Trade, ...]
    sleeve_equity_daily: np.ndarray  # $-basis, len == len(cache.daily_index)
    total_equity_daily: np.ndarray  # sleeve + stable sleeve, $100 basis
    daily_valid: np.ndarray  # bool: this day is inside the evaluated span
    trade_returns: np.ndarray  # net_return_on_base per trade, in exit order
    n_liquidations: int
    mean_realized_leverage: float
    min_notional_usdt: float
    sleeve_start_usdt: float
    stable_start_usdt: float


# ---------------------------------------------------------------------------------------
# Signal generation (vectorised; all features precomputed in dataio30)
# ---------------------------------------------------------------------------------------


def signal_direction(arrays: SymbolArrays, genome: Genome) -> np.ndarray:
    """+1 / -1 / 0 per bar, decided at that bar's close. Fill happens at the NEXT bar's open."""
    lookback = genome.lookback_bars
    close = arrays.close
    long_ok = np.zeros(len(close), dtype=bool)
    short_ok = np.zeros(len(close), dtype=bool)

    if genome.signal_family in {"breakout", "funding_breakout"}:
        long_ok = close > arrays.prior_high[lookback]
        short_ok = close < arrays.prior_low[lookback]
        if genome.signal_family == "funding_breakout":
            # Only take the side funding pays us for: a short receives funding when the rate
            # is positive, a long when it is negative. The most recent charged rate is the
            # only funding information available at bar i's close.
            recent = _last_known_funding(arrays)
            long_ok &= recent < 0.0
            short_ok &= recent > 0.0
    elif genome.signal_family == "momentum":
        horizon_vol = arrays.vol[lookback] * np.sqrt(lookback)
        threshold = genome.entry_threshold * horizon_vol
        long_ok = arrays.ret[lookback] > threshold
        short_ok = arrays.ret[lookback] < -threshold
    elif genome.signal_family == "reversion":
        z = arrays.zscore[lookback]
        long_ok = z < -genome.entry_threshold
        short_ok = z > genome.entry_threshold

    if not genome.allow_short:
        short_ok = np.zeros_like(short_ok)

    direction = np.zeros(len(close), dtype=np.int8)
    direction[long_ok & ~short_ok] = 1
    direction[short_ok & ~long_ok] = -1
    direction[~arrays.tradable] = 0
    direction[np.isnan(close)] = 0
    return direction


def _last_known_funding(arrays: SymbolArrays) -> np.ndarray:
    """Forward-filled most recent charged funding rate, usable at bar i's close."""
    rates = arrays.funding_at_bar
    known = np.where(rates != 0.0, rates, np.nan)
    index = np.arange(len(known))
    valid = ~np.isnan(known)
    if not valid.any():
        return np.zeros(len(known))
    positions = np.maximum.accumulate(np.where(valid, index, 0))
    return np.where(valid.any(), known[positions], 0.0)


# ---------------------------------------------------------------------------------------
# Per-trade outcome. Independent of equity, so it is computed once per candidate entry and
# reused regardless of how the sequential pass sizes the position.
# ---------------------------------------------------------------------------------------


def _resolve_trade(
    arrays: SymbolArrays,
    entry_bar: int,
    direction: float,
    genome: Genome,
    liq_band: float,
    n_bars: int,
    stress: ExecutionStress = NO_STRESS,
) -> tuple[int, float, str, float, bool] | None:
    """Walk bar entry_bar..entry_bar+max_hold and return
    (exit_bar, exit_price, exit_reason, mae, liquidated), or None if unfillable.

    entry_bar is the bar we FILL on (at its open); the signal fired on entry_bar-1's close.
    """
    if entry_bar >= n_bars or not arrays.tradable[entry_bar]:
        return None
    entry_price = arrays.open[entry_bar]
    if not np.isfinite(entry_price) or entry_price <= 0.0:
        return None

    last_bar = min(n_bars - 1, entry_bar + genome.max_hold_bars)
    high = arrays.high[entry_bar : last_bar + 1]
    low = arrays.low[entry_bar : last_bar + 1]
    open_ = arrays.open[entry_bar : last_bar + 1]
    close = arrays.close[entry_bar : last_bar + 1]
    span = len(close)
    if span == 0:
        return None

    stop_distance = genome.stop_pct
    target_distance = genome.stop_pct * genome.target_r

    if direction > 0:
        if genome.trail_enabled:
            # Trailing stop uses only bars strictly BEFORE the one being tested.
            anchor = np.empty(span)
            anchor[0] = entry_price
            np.maximum.accumulate(high[:-1], out=anchor[1:]) if span > 1 else None
            np.maximum(anchor, entry_price, out=anchor)
            stop_level = anchor * (1.0 - stop_distance)
        else:
            stop_level = entry_price * (1.0 - stop_distance)
        target_level = entry_price * (1.0 + target_distance)
        stop_hits = low <= stop_level
        target_hits = high >= target_level
        adverse = (entry_price - np.minimum.accumulate(low)) / entry_price
    else:
        if genome.trail_enabled:
            anchor = np.empty(span)
            anchor[0] = entry_price
            np.minimum.accumulate(low[:-1], out=anchor[1:]) if span > 1 else None
            np.minimum(anchor, entry_price, out=anchor)
            stop_level = anchor * (1.0 + stop_distance)
        else:
            stop_level = entry_price * (1.0 + stop_distance)
        target_level = entry_price * (1.0 - target_distance)
        stop_hits = high >= stop_level
        target_hits = low <= target_level
        adverse = (np.maximum.accumulate(high) - entry_price) / entry_price

    first_stop = int(np.argmax(stop_hits)) if stop_hits.any() else span
    first_target = int(np.argmax(target_hits)) if target_hits.any() else span

    # Same-bar ambiguity resolved against the strategy: stop wins ties (module docstring).
    if first_stop <= first_target and first_stop < span:
        offset = first_stop
        reason = "stop"
        level = float(stop_level[offset]) if np.ndim(stop_level) else float(stop_level)
        if direction > 0:
            exit_price = level if open_[offset] > level else open_[offset]
        else:
            exit_price = level if open_[offset] < level else open_[offset]
    elif first_target < span:
        offset = first_target
        reason = "target"
        exit_price = target_level
        if direction > 0 and open_[offset] > target_level:
            exit_price = open_[offset]  # gapped through the target in our favour
        elif direction < 0 and open_[offset] < target_level:
            exit_price = open_[offset]
    else:
        offset = span - 1
        reason = "max_hold"
        exit_price = close[offset]

    if reason != "target" and stress.stop_slippage:
        # Market exits fill worse than their trigger under stress; a resting take-profit limit
        # does not (see ExecutionStress docstring).
        exit_price = exit_price * (1.0 - direction * stress.stop_slippage)

    mae = float(max(0.0, adverse[offset]))
    realised = direction * (exit_price / entry_price - 1.0)
    liquidated = bool(realised <= -liq_band)
    return entry_bar + offset, float(exit_price), reason, mae, liquidated


def _funding_paid_fraction(arrays: SymbolArrays, direction: float, entry_bar: int, exit_bar: int) -> float:
    """Funding paid as a fraction of NOTIONAL over the holding window. A long pays when the
    rate is positive; a short receives it. Charged on notional, so leverage amplifies it --
    which at 20x is a first-order cost, not a rounding detail."""
    if exit_bar < entry_bar:
        return 0.0
    return float(direction * arrays.funding_at_bar[entry_bar : exit_bar + 1].sum())


# ---------------------------------------------------------------------------------------
# Sequential pass: concurrency, cooldown, compounding, hourly mark-to-market
# ---------------------------------------------------------------------------------------


def run_genome(
    cache: MarketCache,
    genome: Genome,
    mode: str = "is",
    stress: ExecutionStress = NO_STRESS,
    sizing: SizingMode = COMPOUNDING,
) -> Wave30Result:
    """Simulate `genome`. mode='is' evaluates bars up to OOS_SPLIT only; mode='full' uses the
    whole span. Any mode other than these two raises -- and 'oos'/'full' must never be reached
    from inside the search loop (search30 passes mode='is' unconditionally).

    `stress` defaults to a no-op; wave32 uses it to re-price the chosen candidate under adverse
    execution. It is NOT reachable from the search loop, so no genome can be selected for
    looking good under a stress setting the search itself chose."""
    if mode not in {"is", "full"}:
        raise OOSLeakageError(f"unsupported evaluation mode {mode!r}")
    genome.validate()

    n_bars = cache.n_bars
    horizon = int(cache.is_mask.sum()) if mode == "is" else n_bars
    leverage = genome.leverage
    liq_band = genome.liquidation_band

    sleeve_start = TOTAL_CAPITAL * genome.sleeve_fraction
    stable_start = TOTAL_CAPITAL - sleeve_start

    universe = cache.signal_universe(genome.symbols)

    # Candidate entry bars per symbol, kept as numpy arrays so the sequential pass can JUMP
    # over blocked stretches with searchsorted instead of walking every signal bar. A
    # persistent signal (momentum/reversion fire on most bars) would otherwise make this loop
    # 160k python iterations per genome, which dominated everything else.
    candidate_bars: dict[str, np.ndarray] = {}
    candidate_dirs: dict[str, np.ndarray] = {}
    for arrays in universe:
        signal = signal_direction(arrays, genome)
        bars = np.flatnonzero(signal != 0)
        bars = bars[(bars + 1) < horizon]
        candidate_bars[arrays.symbol] = bars + 1  # the bar we FILL on
        candidate_dirs[arrays.symbol] = signal[bars].astype(float)

    unrealised = np.zeros(n_bars, dtype=float)  # $ of open-position P&L at each bar
    realised_marks: list[tuple[int, float]] = []  # (bar, sleeve equity right after this exit)
    open_positions: list[tuple[int, float, Wave30Trade]] = []  # (exit_bar, ret_on_base, trade)
    trades: list[Wave30Trade] = []
    symbol_busy_until: dict[str, int] = {s: -1 for s in genome.symbols}
    symbol_cooldown_until: dict[str, int] = {s: -1 for s in genome.symbols}
    pointer: dict[str, int] = {s: 0 for s in genome.symbols}
    sleeve_equity = sleeve_start
    min_notional = np.inf
    leverages: list[float] = []
    dead = False

    def settle(up_to_bar: int) -> None:
        nonlocal sleeve_equity, dead, open_positions
        due = [item for item in open_positions if item[0] <= up_to_bar]
        if not due:
            return
        due.sort(key=lambda item: item[0])
        for exit_bar, ret_on_base, trade in due:
            sleeve_equity = max(0.0, sleeve_equity + ret_on_base * trade.base_usdt)
            realised_marks.append((exit_bar, sleeve_equity))
            trades.append(trade)
            if sleeve_equity <= SLEEVE_DEAD_THRESHOLD:
                dead = True
        open_positions = [item for item in open_positions if item[0] > up_to_bar]

    def advance_all_pointers(min_bar: int) -> None:
        for name, bars in candidate_bars.items():
            pointer[name] = max(pointer[name], int(np.searchsorted(bars, min_bar, side="left")))

    def next_candidate() -> tuple[int, str, int, float] | None:
        best: tuple[int, str, int, float] | None = None
        for name, bars in candidate_bars.items():
            floor_bar = max(symbol_busy_until[name], symbol_cooldown_until[name]) + 1
            index = max(pointer[name], int(np.searchsorted(bars, floor_bar, side="left")))
            if index >= len(bars):
                continue
            bar = int(bars[index])
            if best is None or bar < best[0]:
                best = (bar, name, index, float(candidate_dirs[name][index]))
        return best

    while True:
        picked = next_candidate()
        if picked is None:
            break
        entry_bar, symbol, index, direction = picked
        settle(entry_bar - 1)
        if dead:
            break

        if len(open_positions) >= genome.max_concurrent:
            # Every candidate at or before the earliest pending exit is blocked by the
            # concurrency cap, so skip the whole stretch at once rather than one bar at a
            # time. earliest_exit >= entry_bar always holds here (we just settled
            # entry_bar-1), so this strictly advances and cannot loop forever.
            earliest_exit = min(item[0] for item in open_positions)
            settle(earliest_exit)
            advance_all_pointers(earliest_exit + 1)
            if dead:
                break
            continue

        pointer[symbol] = index + 1
        arrays = cache.arrays[symbol]
        resolved = _resolve_trade(arrays, entry_bar, direction, genome, liq_band, horizon, stress)
        if resolved is None:
            continue
        exit_bar, exit_price, reason, mae, liquidated = resolved

        base = (sleeve_start if sizing.fixed_base else sleeve_equity) / genome.max_concurrent
        notional = base * leverage
        if sizing.fixed_base and sleeve_equity <= base:
            # Fixed sizing cannot fund another full-size position out of what is left. This is
            # the fixed-size analogue of ruin and must stop the run, not silently shrink the
            # bet (shrinking would quietly turn it back into compounding).
            dead = True
            break
        if base <= SLEEVE_DEAD_THRESHOLD:
            # A sleeve that can no longer fund even one slot is finished. This must terminate
            # the run rather than `continue`: otherwise a sleeve sitting just above the dead
            # threshold walks every remaining candidate for the rest of the span, which is
            # both meaningless and (measured) 15x the entire engine's runtime.
            dead = True
            break
        min_notional = min(min_notional, notional)

        entry_price = float(arrays.open[entry_bar])
        gross = direction * (exit_price / entry_price - 1.0)
        funding_fraction = _funding_paid_fraction(arrays, direction, entry_bar, exit_bar)
        cost_fraction = 2.0 * arrays.cost_rate * stress.cost_multiplier
        if liquidated:
            net = -1.0
        else:
            net = gross * leverage - cost_fraction * leverage - funding_fraction * leverage
            net = max(net, -1.0)

        trade = Wave30Trade(
            symbol=symbol,
            direction=direction,
            entry_bar=entry_bar,
            exit_bar=exit_bar,
            entry_price=entry_price,
            exit_price=exit_price,
            leverage=leverage,
            notional_usdt=notional,
            base_usdt=base,
            gross_price_return=gross,
            funding_fraction=funding_fraction,
            cost_fraction=cost_fraction,
            net_return_on_base=net,
            mae=mae,
            exit_reason="liquidation" if liquidated else reason,
            liquidated=liquidated,
        )
        open_positions.append((exit_bar, net, trade))
        leverages.append(leverage)
        symbol_busy_until[symbol] = exit_bar
        if net < 0.0:
            symbol_cooldown_until[symbol] = exit_bar + genome.cooldown_bars_after_loss

        # Hourly mark-to-market of this position, for an MDD that includes OPEN-position pain
        # rather than only realised steps (at 20x the intra-trade trough is the whole story).
        path = arrays.close[entry_bar : exit_bar + 1]
        with np.errstate(invalid="ignore"):
            path_return = direction * (path / entry_price - 1.0) * leverage - cost_fraction * leverage
        path_pnl = np.nan_to_num(path_return, nan=0.0) * base
        path_pnl = np.maximum(path_pnl, -base)
        path_pnl[-1] = net * base
        unrealised[entry_bar : exit_bar + 1] += path_pnl

    settle(n_bars - 1)

    # Build the hourly sleeve curve: realised step function + open-position mark-to-market.
    realised_curve = np.full(n_bars, sleeve_start, dtype=float)
    if realised_marks:
        realised_marks.sort(key=lambda item: item[0])
        bars = np.array([item[0] for item in realised_marks])
        values = np.array([item[1] for item in realised_marks])
        # last mark wins on duplicate bars
        realised_curve[:] = sleeve_start
        positions = np.searchsorted(bars, np.arange(n_bars), side="right") - 1
        realised_curve = np.where(positions >= 0, values[np.clip(positions, 0, None)], sleeve_start)
    hourly_sleeve = np.maximum(0.0, realised_curve + unrealised)

    # Collapse to daily (last bar of each day), then add the stable sleeve.
    n_days = len(cache.daily_index)
    sleeve_daily = np.full(n_days, np.nan)
    np.put(sleeve_daily, cache.day_of_bar, hourly_sleeve)  # later bars overwrite earlier ones
    sleeve_daily = _forward_fill(sleeve_daily, sleeve_start)

    stable_daily = stable_start * cache.stable_per_dollar
    total_daily = sleeve_daily + stable_daily

    horizon_day = int(cache.day_of_bar[horizon - 1]) if horizon > 0 else 0
    daily_valid = np.arange(n_days) <= horizon_day

    return Wave30Result(
        genome=genome,
        trades=tuple(trades),
        sleeve_equity_daily=sleeve_daily,
        total_equity_daily=total_daily,
        daily_valid=daily_valid,
        trade_returns=np.array([t.net_return_on_base for t in trades], dtype=float),
        n_liquidations=int(sum(1 for t in trades if t.liquidated)),
        mean_realized_leverage=float(np.mean(leverages)) if leverages else 0.0,
        min_notional_usdt=float(min_notional) if np.isfinite(min_notional) else float("nan"),
        sleeve_start_usdt=sleeve_start,
        stable_start_usdt=stable_start,
    )


def _forward_fill(values: np.ndarray, seed: float) -> np.ndarray:
    out = values.copy()
    if not np.isfinite(out[0]):
        out[0] = seed
    valid = np.isfinite(out)
    index = np.arange(len(out))
    positions = np.maximum.accumulate(np.where(valid, index, 0))
    return out[positions]


def max_drawdown(curve: np.ndarray) -> float:
    if len(curve) == 0:
        return 0.0
    peak = np.maximum.accumulate(curve)
    return float(np.min((curve - peak) / np.maximum(peak, 1e-12)))


def annualized_return(curve: np.ndarray, days: float) -> float:
    if len(curve) < 2 or days <= 0 or curve[0] <= 0:
        return 0.0
    ratio = curve[-1] / curve[0]
    if ratio <= 0:
        return -1.0
    return float(ratio ** (365.0 / days) - 1.0)
