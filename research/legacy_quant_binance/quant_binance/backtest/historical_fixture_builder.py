"""Build time-series slices from historical klines for backtesting."""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

from quant_binance.data.rest_seed import _parse_kline
from quant_binance.data.state import SymbolMarketState, TopOfBook
from quant_binance.features.extractor import MarketFeatureExtractor
from quant_binance.features.primitive import FeatureHistoryContext, PrimitiveInputs
from quant_binance.settings import Settings


@dataclass(frozen=True)
class HistoricalTimeSlice:
    decision_time: datetime
    symbol: str
    state: SymbolMarketState
    primitive_inputs: PrimitiveInputs
    history: FeatureHistoryContext
    forward_return_1h_bps: float
    forward_return_4h_bps: float


def _kline_bars_from_raw(symbol: str, interval: str, raw_klines: list[dict[str, Any]]) -> list:
    """Convert raw kline dicts to KlineBar objects."""
    bars = []
    for row in raw_klines:
        try:
            bars.append(_parse_kline(symbol, interval, row))
        except Exception:
            continue
    return bars


def _find_close_at(bars_1h: list, target_time_ms: int, tolerance_ms: int = 7_200_000) -> float | None:
    """Find the close price of a 1h bar nearest to target_time_ms."""
    best = None
    best_diff = tolerance_ms
    for bar in bars_1h:
        bar_ms = int(bar.close_time.timestamp() * 1000)
        diff = abs(bar_ms - target_time_ms)
        if diff < best_diff:
            best = bar.close_price
            best_diff = diff
    return best


def build_historical_slices(
    *,
    symbol: str,
    klines_5m: list[dict[str, Any]],
    klines_1h: list[dict[str, Any]],
    klines_4h: list[dict[str, Any]],
    klines_1m: list[dict[str, Any]] | None = None,
    spot_klines_1h: list[dict[str, Any]] | None = None,
    funding_rates: list[dict[str, Any]] | None = None,
    settings: Settings,
    extractor: MarketFeatureExtractor,
    warmup_bars_1h: int = 120,
    step_hours: int = 1,
) -> list[HistoricalTimeSlice]:
    """Build time slices from raw klines.

    Each slice represents the state visible at a 1h boundary,
    plus forward returns for PnL simulation.
    """
    bars_5m = _kline_bars_from_raw(symbol, "5m", klines_5m)
    bars_1h = _kline_bars_from_raw(symbol, "1h", klines_1h)
    bars_4h = _kline_bars_from_raw(symbol, "4h", klines_4h)
    bars_1m = _kline_bars_from_raw(symbol, "1m", klines_1m or [])

    # Build spot price lookup for basis calculation: {close_time_ms: close_price}
    _spot_bars_1h = _kline_bars_from_raw(symbol, "1h", spot_klines_1h or [])
    spot_price_map: dict[int, float] = {}
    for sb in _spot_bars_1h:
        sb_ms = int(sb.close_time.timestamp() * 1000)
        spot_price_map[sb_ms] = sb.close_price

    # Build funding rate lookup: sorted list of (time_ms, rate)
    _funding_sorted: list[tuple[int, float]] = []
    for fr in (funding_rates or []):
        try:
            _funding_sorted.append((int(fr["funding_time"]), float(fr["funding_rate"])))
        except (KeyError, ValueError, TypeError):
            continue
    _funding_sorted.sort(key=lambda x: x[0])

    if len(bars_1h) < warmup_bars_1h + 10:
        return []

    slices: list[HistoricalTimeSlice] = []

    for i in range(warmup_bars_1h, len(bars_1h) - 4):
        current_bar = bars_1h[i]
        decision_time = current_bar.close_time
        current_price = current_bar.close_price
        if current_price <= 0:
            continue

        # Build visible kline windows (only bars closed at or before decision_time)
        dt_ms = int(decision_time.timestamp() * 1000)
        visible_5m = [b for b in bars_5m if int(b.close_time.timestamp() * 1000) <= dt_ms][-500:]
        visible_1h = bars_1h[: i + 1][-200:]
        visible_4h = [b for b in bars_4h if int(b.close_time.timestamp() * 1000) <= dt_ms][-200:]
        visible_1m = [b for b in bars_1m if int(b.close_time.timestamp() * 1000) <= dt_ms][-500:] if bars_1m else []

        if len(visible_1h) < 21 or len(visible_4h) < 2 or len(visible_5m) < 2:
            continue

        # --- Funding rate: find latest rate at or before decision_time ---
        current_funding = 0.0001  # fallback
        funding_history: list[float] = []
        for ft_ms, fr_val in _funding_sorted:
            if ft_ms <= dt_ms:
                current_funding = fr_val
                funding_history.append(fr_val)
            else:
                break
        # Keep last 200 funding samples for history
        funding_history = funding_history[-200:]

        # --- Basis: futures_close - spot_close ---
        spot_close = spot_price_map.get(dt_ms)
        if spot_close and spot_close > 0 and current_price > 0:
            current_basis_bps = ((current_price / spot_close) - 1.0) * 10000
        else:
            current_basis_bps = 0.0
        # Build basis history from visible 1h bars
        basis_history: list[float] = []
        for bar in visible_1h[-200:]:
            bar_ms = int(bar.close_time.timestamp() * 1000)
            sp = spot_price_map.get(bar_ms)
            if sp and sp > 0 and bar.close_price > 0:
                basis_history.append(((bar.close_price / sp) - 1.0) * 10000)

        # --- OI approximation from volume surge (no historical OI API) ---
        # Use quote volume as OI proxy (higher volume ≈ higher OI in trending markets)
        oi_proxy_samples: list[float] = []
        for bar in visible_1h[-200:]:
            oi_proxy_samples.append(bar.quote_volume if hasattr(bar, 'quote_volume') else 0.0)
        current_oi = oi_proxy_samples[-1] if oi_proxy_samples else 0.0

        # Synthetic top of book (2bps spread)
        spread_half = current_price * 0.0001
        top_of_book = TopOfBook(
            bid_price=current_price - spread_half,
            bid_qty=1.0,
            ask_price=current_price + spread_half,
            ask_qty=1.0,
            updated_at=decision_time,
        )

        state = SymbolMarketState(
            symbol=symbol,
            top_of_book=top_of_book,
            last_trade_price=current_price,
            funding_rate=current_funding,
            open_interest=current_oi,
            basis_bps=current_basis_bps,
            last_update_time=decision_time,
        )
        state.klines["5m"] = list(visible_5m)
        state.klines["1h"] = list(visible_1h)
        state.klines["4h"] = list(visible_4h)
        if visible_1m:
            state.klines["1m"] = list(visible_1m)
        # Populate historical samples for percentile ranking
        for fr_val in funding_history:
            state.funding_rate_samples.append(fr_val)
        for bb_val in basis_history:
            state.basis_bps_samples.append(bb_val)
        for oi_val in oi_proxy_samples:
            state.open_interest_samples.append(oi_val)

        # Build features using existing extractor
        try:
            primitive_inputs = extractor.build_primitive_inputs(state)
            history = extractor.build_history_context(state)
        except Exception:
            continue

        # Forward returns (lookahead — only used for PnL, not visible to strategy)
        next_1h_close = bars_1h[i + 1].close_price if i + 1 < len(bars_1h) else None
        next_4h_close = _find_close_at(bars_1h, dt_ms + 4 * 3_600_000)

        fwd_1h = ((next_1h_close / current_price) - 1) * 10000 if next_1h_close else 0.0
        fwd_4h = ((next_4h_close / current_price) - 1) * 10000 if next_4h_close else 0.0

        slices.append(HistoricalTimeSlice(
            decision_time=decision_time,
            symbol=symbol,
            state=state,
            primitive_inputs=primitive_inputs,
            history=history,
            forward_return_1h_bps=round(fwd_1h, 4),
            forward_return_4h_bps=round(fwd_4h, 4),
        ))

    return slices
