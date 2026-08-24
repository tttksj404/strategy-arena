"""Advanced trading signal features: OI Divergence, VWAP Deviation, Smart Money Concepts.

These features are designed to integrate with the existing predictability_score
system as additional inputs to the scorer and edge estimator.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, pstdev

from quant_binance.data.state import KlineBar, SymbolMarketState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ema(values: list[float], period: int) -> float:
    if not values:
        return 0.0
    alpha = 2.0 / (period + 1.0)
    ema = values[0]
    for v in values[1:]:
        ema = alpha * v + (1.0 - alpha) * ema
    return ema


def _zscore(value: float, series: list[float]) -> float:
    if len(series) < 2:
        return 0.0
    m = mean(series)
    s = pstdev(series)
    if s < 1e-12:
        return 0.0
    return (value - m) / s


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _recent_closed(state: SymbolMarketState, interval: str, limit: int) -> list[KlineBar]:
    bars = [b for b in state.klines.get(interval, []) if b.is_closed]
    return bars[-limit:]


# ---------------------------------------------------------------------------
# 1. OI-Price Divergence
# ---------------------------------------------------------------------------

def compute_oi_divergence(
    oi_samples: list[float],
    price_series: list[float],
    lookback: int = 24,
) -> float:
    """Detect divergence between price breakouts and open interest.

    Returns a score from -1.0 to +1.0:
      Positive = OI confirms price move (healthy breakout)
      Negative = OI diverges from price (likely fake breakout)
    """
    if len(price_series) < lookback + 1 or len(oi_samples) < lookback + 1:
        return 0.0

    recent_prices = price_series[-(lookback + 1):]
    recent_oi = oi_samples[-(lookback + 1):]

    current_price = recent_prices[-1]
    prev_prices = recent_prices[:-1]
    prev_high = max(prev_prices)
    prev_low = min(prev_prices)

    oi_now = recent_oi[-1]
    oi_ema = _ema(recent_oi[:-1], min(lookback, len(recent_oi) - 1))
    oi_series_for_z = recent_oi[:-1]
    oi_delta = oi_now - oi_ema
    oi_delta_z = _zscore(oi_delta, [oi_samples[i] - _ema(oi_samples[max(0, i - lookback):i] or [oi_samples[0]], min(lookback, max(1, i))) for i in range(max(1, len(oi_samples) - lookback), len(oi_samples))])

    new_high = current_price > prev_high
    new_low = current_price < prev_low

    if new_high and oi_delta_z < -0.5:
        return round(max(-0.8, -0.4 - 0.4 * abs(oi_delta_z)), 6)
    elif new_high and oi_delta_z > 1.0:
        return round(min(0.7, 0.3 + 0.2 * oi_delta_z), 6)
    elif new_low and oi_delta_z < -0.5:
        return round(min(0.6, 0.3 + 0.15 * abs(oi_delta_z)), 6)
    elif new_low and oi_delta_z > 1.0:
        return round(max(-0.7, -0.35 - 0.175 * oi_delta_z), 6)
    else:
        return round(_clamp(oi_delta_z * 0.15, -0.3, 0.3), 6)


def oi_divergence_from_state(state: SymbolMarketState, lookback: int = 24) -> float:
    """Convenience: compute OI divergence directly from market state."""
    oi_samples = state.open_interest_samples or [state.open_interest]
    bars_1h = _recent_closed(state, "1h", lookback + 5)
    prices = [b.close_price for b in bars_1h]
    return compute_oi_divergence(oi_samples, prices, lookback)


# ---------------------------------------------------------------------------
# 2. VWAP Deviation Score
# ---------------------------------------------------------------------------

def compute_vwap(bars: list[KlineBar]) -> float:
    """Volume-weighted average price over given bars."""
    cum_pv = 0.0
    cum_v = 0.0
    for b in bars:
        typical_price = (b.high_price + b.low_price + b.close_price) / 3.0
        cum_pv += typical_price * b.quote_volume
        cum_v += b.quote_volume
    if cum_v < 1e-12:
        return bars[-1].close_price if bars else 0.0
    return cum_pv / cum_v


def compute_vwap_deviation(
    bars: list[KlineBar],
    lookback_for_zscore: int = 96,
) -> tuple[float, float]:
    """Compute VWAP and z-score deviation of current price from VWAP.

    Returns (vwap, deviation_zscore).
    deviation_zscore > 0 means price is above VWAP (potential overbought).
    deviation_zscore < 0 means price is below VWAP (potential oversold).
    """
    if len(bars) < 3:
        return 0.0, 0.0

    vwap = compute_vwap(bars)
    current_price = bars[-1].close_price

    deviations = []
    for i in range(max(1, len(bars) - lookback_for_zscore), len(bars)):
        sub_bars = bars[:i + 1]
        sub_vwap = compute_vwap(sub_bars)
        dev = sub_bars[-1].close_price - sub_vwap
        deviations.append(dev)

    if len(deviations) < 2:
        return vwap, 0.0

    current_dev = current_price - vwap
    dev_z = _zscore(current_dev, deviations)
    return round(vwap, 6), round(_clamp(dev_z, -4.0, 4.0), 6)


def vwap_deviation_from_state(state: SymbolMarketState) -> tuple[float, float]:
    """Convenience: compute VWAP deviation from market state using 5m bars."""
    bars_5m = _recent_closed(state, "5m", 288)  # ~24h of 5m bars
    if len(bars_5m) < 12:
        return 0.0, 0.0
    return compute_vwap_deviation(bars_5m, lookback_for_zscore=96)


# ---------------------------------------------------------------------------
# 3. Smart Money Concepts (SMC)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FairValueGap:
    gap_type: str  # "bullish" or "bearish"
    zone_low: float
    zone_high: float
    size_bps: float
    bar_index: int


@dataclass(frozen=True)
class OrderBlock:
    ob_type: str  # "bullish" or "bearish"
    zone_low: float
    zone_high: float
    bar_index: int


@dataclass(frozen=True)
class StructureBreak:
    break_type: str  # "bos_up", "bos_down", "choch_up", "choch_down"
    level: float
    bar_index: int


@dataclass(frozen=True)
class SMCSignals:
    fvg_score: float        # 0-1: presence and quality of FVG retest
    ob_score: float         # 0-1: presence and quality of order block retest
    structure_score: float  # 0-1: BOS/CHoCH confirmation
    composite_score: float  # 0-1: weighted combination


def detect_fair_value_gaps(
    bars: list[KlineBar],
    min_gap_bps: float = 8.0,
    max_gap_bps: float = 80.0,
    max_age: int = 50,
) -> list[FairValueGap]:
    """Detect Fair Value Gaps (3-candle pattern where middle candle creates a gap)."""
    gaps: list[FairValueGap] = []
    if len(bars) < 3:
        return gaps

    for i in range(2, len(bars)):
        mid_price = (bars[i - 1].high_price + bars[i - 1].low_price) / 2.0
        if mid_price < 1e-12:
            continue

        # Bullish FVG: bar[i] low > bar[i-2] high (gap up)
        if bars[i].low_price > bars[i - 2].high_price:
            gap_bps = (bars[i].low_price - bars[i - 2].high_price) / mid_price * 10000.0
            if min_gap_bps <= gap_bps <= max_gap_bps:
                gaps.append(FairValueGap(
                    gap_type="bullish",
                    zone_low=bars[i - 2].high_price,
                    zone_high=bars[i].low_price,
                    size_bps=round(gap_bps, 2),
                    bar_index=i,
                ))

        # Bearish FVG: bar[i] high < bar[i-2] low (gap down)
        if bars[i].high_price < bars[i - 2].low_price:
            gap_bps = (bars[i - 2].low_price - bars[i].high_price) / mid_price * 10000.0
            if min_gap_bps <= gap_bps <= max_gap_bps:
                gaps.append(FairValueGap(
                    gap_type="bearish",
                    zone_low=bars[i].high_price,
                    zone_high=bars[i - 2].low_price,
                    size_bps=round(gap_bps, 2),
                    bar_index=i,
                ))

    # Keep only recent and unfilled gaps
    current_price = bars[-1].close_price
    recent_gaps = []
    for gap in gaps:
        age = len(bars) - 1 - gap.bar_index
        if age > max_age:
            continue
        # Check if gap has been filled
        filled = False
        for j in range(gap.bar_index + 1, len(bars)):
            if gap.gap_type == "bullish" and bars[j].low_price <= gap.zone_low:
                filled = True
                break
            if gap.gap_type == "bearish" and bars[j].high_price >= gap.zone_high:
                filled = True
                break
        if not filled:
            recent_gaps.append(gap)

    return recent_gaps


def detect_order_blocks(
    bars: list[KlineBar],
    atr: float,
    lookback: int = 30,
    displacement_mult: float = 1.5,
) -> list[OrderBlock]:
    """Detect Order Blocks: last opposing candle before a strong displacement move."""
    blocks: list[OrderBlock] = []
    if len(bars) < 3 or atr < 1e-12:
        return blocks

    start = max(0, len(bars) - lookback)
    for i in range(start, len(bars) - 1):
        bar_range = bars[i + 1].close_price - bars[i + 1].open_price
        displacement = abs(bar_range) / atr

        if displacement < displacement_mult:
            continue

        # Bullish OB: bearish candle followed by strong bullish displacement
        if bars[i].close_price < bars[i].open_price and bar_range > 0:
            blocks.append(OrderBlock(
                ob_type="bullish",
                zone_low=bars[i].low_price,
                zone_high=bars[i].high_price,
                bar_index=i,
            ))

        # Bearish OB: bullish candle followed by strong bearish displacement
        if bars[i].close_price > bars[i].open_price and bar_range < 0:
            blocks.append(OrderBlock(
                ob_type="bearish",
                zone_low=bars[i].low_price,
                zone_high=bars[i].high_price,
                bar_index=i,
            ))

    return blocks


def detect_structure_breaks(
    bars: list[KlineBar],
    swing_lookback: int = 5,
) -> list[StructureBreak]:
    """Detect Break of Structure (BOS) and Change of Character (CHoCH)."""
    breaks: list[StructureBreak] = []
    if len(bars) < swing_lookback * 3:
        return breaks

    # Find swing highs and lows
    swing_highs: list[tuple[int, float]] = []
    swing_lows: list[tuple[int, float]] = []
    for i in range(swing_lookback, len(bars) - swing_lookback):
        is_high = all(bars[i].high_price >= bars[j].high_price for j in range(i - swing_lookback, i + swing_lookback + 1) if j != i)
        is_low = all(bars[i].low_price <= bars[j].low_price for j in range(i - swing_lookback, i + swing_lookback + 1) if j != i)
        if is_high:
            swing_highs.append((i, bars[i].high_price))
        if is_low:
            swing_lows.append((i, bars[i].low_price))

    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return breaks

    # Detect BOS: higher high (uptrend) or lower low (downtrend)
    for i in range(1, len(swing_highs)):
        if swing_highs[i][1] > swing_highs[i - 1][1]:
            # Check if structure is bullish (higher lows too)
            relevant_lows = [sl for sl in swing_lows if sl[0] > swing_highs[i - 1][0] and sl[0] < swing_highs[i][0]]
            if relevant_lows:
                breaks.append(StructureBreak("bos_up", swing_highs[i][1], swing_highs[i][0]))

    for i in range(1, len(swing_lows)):
        if swing_lows[i][1] < swing_lows[i - 1][1]:
            relevant_highs = [sh for sh in swing_highs if sh[0] > swing_lows[i - 1][0] and sh[0] < swing_lows[i][0]]
            if relevant_highs:
                breaks.append(StructureBreak("bos_down", swing_lows[i][1], swing_lows[i][0]))

    # Detect CHoCH: trend reversal (bearish-to-bullish or bullish-to-bearish)
    if len(swing_highs) >= 2 and len(swing_lows) >= 2:
        last_two_highs = swing_highs[-2:]
        last_two_lows = swing_lows[-2:]

        # CHoCH up: was making lower highs, now breaks above
        if last_two_highs[0][1] > last_two_highs[1][1]:
            # Previous trend was down, check if price now breaks last swing high
            if bars[-1].high_price > last_two_highs[1][1]:
                breaks.append(StructureBreak("choch_up", last_two_highs[1][1], len(bars) - 1))

        # CHoCH down: was making higher lows, now breaks below
        if last_two_lows[0][1] < last_two_lows[1][1]:
            if bars[-1].low_price < last_two_lows[1][1]:
                breaks.append(StructureBreak("choch_down", last_two_lows[1][1], len(bars) - 1))

    return breaks


def compute_smc_signals(
    bars: list[KlineBar],
    current_price: float,
    atr: float,
    trend_direction: int = 0,
) -> SMCSignals:
    """Compute composite Smart Money Concepts score.

    Returns SMCSignals with individual and composite scores (0-1).
    """
    if len(bars) < 20 or atr < 1e-12:
        return SMCSignals(0.0, 0.0, 0.0, 0.0)

    # --- FVG Score ---
    fvgs = detect_fair_value_gaps(bars)
    fvg_score = 0.0
    for fvg in fvgs:
        in_zone = fvg.zone_low <= current_price <= fvg.zone_high
        near_zone = abs(current_price - (fvg.zone_low + fvg.zone_high) / 2) / atr < 2.0
        if in_zone:
            age = len(bars) - 1 - fvg.bar_index
            recency = max(0.0, 1.0 - age / 50.0)
            if (fvg.gap_type == "bullish" and trend_direction >= 0) or \
               (fvg.gap_type == "bearish" and trend_direction <= 0):
                fvg_score = max(fvg_score, 0.6 + 0.4 * recency)
            else:
                fvg_score = max(fvg_score, 0.3 + 0.2 * recency)
        elif near_zone:
            fvg_score = max(fvg_score, 0.2)

    # --- Order Block Score ---
    obs = detect_order_blocks(bars, atr)
    ob_score = 0.0
    for ob in obs:
        in_zone = ob.zone_low <= current_price <= ob.zone_high
        if in_zone:
            age = len(bars) - 1 - ob.bar_index
            recency = max(0.0, 1.0 - age / 30.0)
            if (ob.ob_type == "bullish" and trend_direction >= 0) or \
               (ob.ob_type == "bearish" and trend_direction <= 0):
                ob_score = max(ob_score, 0.5 + 0.5 * recency)
            else:
                ob_score = max(ob_score, 0.2 + 0.2 * recency)

    # --- Structure Break Score ---
    structure_breaks = detect_structure_breaks(bars)
    structure_score = 0.0
    recent_breaks = [b for b in structure_breaks if (len(bars) - 1 - b.bar_index) < 15]
    for sb in recent_breaks:
        recency = max(0.0, 1.0 - (len(bars) - 1 - sb.bar_index) / 15.0)
        if sb.break_type == "bos_up" and trend_direction >= 0:
            structure_score = max(structure_score, 0.5 + 0.5 * recency)
        elif sb.break_type == "bos_down" and trend_direction <= 0:
            structure_score = max(structure_score, 0.5 + 0.5 * recency)
        elif sb.break_type == "choch_up" and trend_direction >= 0:
            structure_score = max(structure_score, 0.7 + 0.3 * recency)
        elif sb.break_type == "choch_down" and trend_direction <= 0:
            structure_score = max(structure_score, 0.7 + 0.3 * recency)

    # --- Composite ---
    composite = 0.3 * fvg_score + 0.3 * ob_score + 0.4 * structure_score

    return SMCSignals(
        fvg_score=round(fvg_score, 6),
        ob_score=round(ob_score, 6),
        structure_score=round(structure_score, 6),
        composite_score=round(composite, 6),
    )


def smc_signals_from_state(state: SymbolMarketState, trend_direction: int = 0) -> SMCSignals:
    """Convenience: compute SMC signals from market state using 5m bars."""
    bars_5m = _recent_closed(state, "5m", 100)
    if len(bars_5m) < 20:
        return SMCSignals(0.0, 0.0, 0.0, 0.0)

    # Compute ATR from 5m bars
    ranges = []
    for i in range(1, len(bars_5m)):
        tr = max(
            bars_5m[i].high_price - bars_5m[i].low_price,
            abs(bars_5m[i].high_price - bars_5m[i - 1].close_price),
            abs(bars_5m[i].low_price - bars_5m[i - 1].close_price),
        )
        ranges.append(tr)
    atr = mean(ranges[-14:]) if len(ranges) >= 14 else (mean(ranges) if ranges else 0.0)

    return compute_smc_signals(
        bars=bars_5m,
        current_price=state.last_trade_price,
        atr=atr,
        trend_direction=trend_direction,
    )
