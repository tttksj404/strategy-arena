from __future__ import annotations

from statistics import mean, median, pstdev
from pathlib import Path

from quant_binance.cost_calibration import CostCalibration, load_cost_calibration
from quant_binance.data.state import KlineBar, SymbolMarketState
from quant_binance.features.advanced_signals import (
    oi_divergence_from_state,
    vwap_deviation_from_state,
    smc_signals_from_state,
)
from quant_binance.features.primitive import FeatureHistoryContext, PrimitiveInputs
from quant_binance.models import FeatureVector
from quant_binance.settings import Settings
from quant_binance.strategy.scorer import compute_predictability_score
from quant_binance.strategy.edge import ConditionalEdgeLookup
from quant_binance.strategy.normalize import clamp
from quant_binance.strategy.coin_profiles import get_profile


def _pct_returns(bars: list[KlineBar]) -> list[float]:
    returns: list[float] = []
    for prev, curr in zip(bars, bars[1:]):
        if prev.close_price > 0:
            returns.append((curr.close_price / prev.close_price) - 1.0)
    return returns


def _true_ranges(bars: list[KlineBar]) -> list[float]:
    if not bars:
        return []
    ranges: list[float] = []
    prev_close = bars[0].close_price
    for bar in bars[1:]:
        high_low = bar.high_price - bar.low_price
        high_prev = abs(bar.high_price - prev_close)
        low_prev = abs(bar.low_price - prev_close)
        ranges.append(max(high_low, high_prev, low_prev))
        prev_close = bar.close_price
    return ranges


def _ema(values: list[float], period: int) -> float:
    if not values:
        return 0.0
    alpha = 2.0 / (period + 1.0)
    ema = values[0]
    for value in values[1:]:
        ema = alpha * value + (1.0 - alpha) * ema
    return ema


def _recent_closed_bars(state: SymbolMarketState, interval: str, limit: int) -> list[KlineBar]:
    bars = [bar for bar in state.klines.get(interval, []) if bar.is_closed]
    return bars[-limit:]


def _adx_from_bars(bars_1h: list[KlineBar], period: int = 14) -> float:
    """Compute ADX from 1h bars. Returns 0.0 if insufficient data."""
    if len(bars_1h) < 2 * period + 2:
        return 0.0
    highs = [b.high_price for b in bars_1h]
    lows = [b.low_price for b in bars_1h]
    closes = [b.close_price for b in bars_1h]
    n = len(closes)

    plus_dm: list[float] = []
    minus_dm: list[float] = []
    tr_list: list[float] = []
    for i in range(1, n):
        up = highs[i] - highs[i - 1]
        down = lows[i - 1] - lows[i]
        plus_dm.append(up if up > down and up > 0 else 0.0)
        minus_dm.append(down if down > up and down > 0 else 0.0)
        tr_list.append(max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        ))

    if len(tr_list) < 2 * period:
        return 0.0

    sm_tr = sum(tr_list[:period])
    sm_plus = sum(plus_dm[:period])
    sm_minus = sum(minus_dm[:period])
    for i in range(period, len(tr_list)):
        sm_tr = sm_tr - sm_tr / period + tr_list[i]
        sm_plus = sm_plus - sm_plus / period + plus_dm[i]
        sm_minus = sm_minus - sm_minus / period + minus_dm[i]

    if sm_tr == 0:
        return 0.0
    pdi = 100.0 * sm_plus / sm_tr
    mdi = 100.0 * sm_minus / sm_tr
    denom = pdi + mdi
    dx = 100.0 * abs(pdi - mdi) / denom if denom > 0 else 0.0

    # Smooth DX into ADX (one more period of smoothing)
    dx_values: list[float] = []
    sm_tr2 = sum(tr_list[:period])
    sm_plus2 = sum(plus_dm[:period])
    sm_minus2 = sum(minus_dm[:period])
    for i in range(period, len(tr_list)):
        sm_tr2 = sm_tr2 - sm_tr2 / period + tr_list[i]
        sm_plus2 = sm_plus2 - sm_plus2 / period + plus_dm[i]
        sm_minus2 = sm_minus2 - sm_minus2 / period + minus_dm[i]
        if sm_tr2 > 0:
            p = 100.0 * sm_plus2 / sm_tr2
            m = 100.0 * sm_minus2 / sm_tr2
            d = p + m
            dx_values.append(100.0 * abs(p - m) / d if d > 0 else 0.0)

    if len(dx_values) < period:
        return dx_values[-1] if dx_values else 0.0
    adx_val = mean(dx_values[:period])
    for i in range(period, len(dx_values)):
        adx_val = (adx_val * (period - 1) + dx_values[i]) / period
    return round(adx_val, 2)


def _rsi(closes: list[float], period: int = 14) -> float:
    """RSI on last `period+1` closes. Returns 0-100 or 50.0 if insufficient data."""
    if len(closes) < period + 1:
        return 50.0
    recent = closes[-(period + 1):]
    deltas = [recent[i + 1] - recent[i] for i in range(len(recent) - 1)]
    gains = [d if d > 0 else 0.0 for d in deltas]
    losses = [-d if d < 0 else 0.0 for d in deltas]
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def _pullback_signal(closes_1h: list[float], ema_period: int = 21, rsi_entry: float = 40.0) -> int:
    """Detect pullback entry on 1h bars.

    Long: price > EMA50, EMA(p) > EMA50, RSI crosses up through rsi_entry (oversold recovery)
    Short: price < EMA50, EMA(p) < EMA50, RSI crosses down through (100-rsi_entry)
    Returns +1 (long pullback), -1 (short pullback), 0 (none).
    """
    if len(closes_1h) < 52:
        return 0
    ema_p_now = _ema(closes_1h[-ema_period:], ema_period)
    ema50_now = _ema(closes_1h[-50:], 50)
    rsi_now = _rsi(closes_1h, 14)
    rsi_prev = _rsi(closes_1h[:-1], 14)
    price = closes_1h[-1]

    # Long pullback: uptrend + RSI recovers from oversold
    if price > ema50_now and ema_p_now > ema50_now and rsi_prev < rsi_entry and rsi_now >= rsi_entry and rsi_now < 60:
        return 1
    # Short pullback: downtrend + RSI drops from overbought
    if price < ema50_now and ema_p_now < ema50_now and rsi_prev > (100 - rsi_entry) and rsi_now <= (100 - rsi_entry) and rsi_now > 40:
        return -1
    return 0


def _ema_cross_signal(closes_1h: list[float], fast_period: int = 9, slow_period: int = 21) -> int:
    """Detect EMA cross on 1h bars. Returns +1 (bullish cross), -1 (bearish), 0 (none)."""
    if len(closes_1h) < slow_period + 2:
        return 0
    ema_fast_now = _ema(closes_1h[-fast_period:], fast_period)
    ema_slow_now = _ema(closes_1h[-slow_period:], slow_period)
    ema_fast_prev = _ema(closes_1h[-fast_period - 1:-1], fast_period)
    ema_slow_prev = _ema(closes_1h[-slow_period - 1:-1], slow_period)

    if ema_fast_prev <= ema_slow_prev and ema_fast_now > ema_slow_now:
        return 1
    if ema_fast_prev >= ema_slow_prev and ema_fast_now < ema_slow_now:
        return -1
    return 0


def _intraday_trend_signal(
    *,
    bars_5m: list[KlineBar],
    bars_1m: list[KlineBar],
) -> tuple[int, float]:
    components: list[tuple[int, float]] = []
    closes_5m = [bar.close_price for bar in bars_5m]
    if len(closes_5m) >= 8:
        ema_fast_5m = _ema(closes_5m[-3:], min(3, len(closes_5m[-3:])))
        ema_slow_5m = _ema(closes_5m[-8:], min(8, len(closes_5m[-8:])))
        ret_5m = (closes_5m[-1] / closes_5m[-4] - 1.0) if closes_5m[-4] > 0 else 0.0
        if ema_fast_5m > ema_slow_5m and ret_5m > 0:
            components.append((1, clamp(abs(ret_5m) * 1800.0 + 0.35)))
        elif ema_fast_5m < ema_slow_5m and ret_5m < 0:
            components.append((-1, clamp(abs(ret_5m) * 1800.0 + 0.35)))

    closes_1m = [bar.close_price for bar in bars_1m]
    if len(closes_1m) >= 12:
        ema_fast_1m = _ema(closes_1m[-5:], min(5, len(closes_1m[-5:])))
        ema_slow_1m = _ema(closes_1m[-12:], min(12, len(closes_1m[-12:])))
        ret_1m = (closes_1m[-1] / closes_1m[-6] - 1.0) if closes_1m[-6] > 0 else 0.0
        if ema_fast_1m > ema_slow_1m and ret_1m > 0:
            components.append((1, clamp(abs(ret_1m) * 3200.0 + 0.25)))
        elif ema_fast_1m < ema_slow_1m and ret_1m < 0:
            components.append((-1, clamp(abs(ret_1m) * 3200.0 + 0.25)))

    if not components:
        return 0, 0.0
    signed_strength = sum(direction * strength for direction, strength in components)
    if signed_strength > 0:
        return 1, round(min(abs(signed_strength), 1.0), 6)
    if signed_strength < 0:
        return -1, round(min(abs(signed_strength), 1.0), 6)
    return 0, 0.0


class MarketFeatureExtractor:
    def __init__(
        self,
        settings: Settings,
        edge_lookup: ConditionalEdgeLookup | None = None,
        cost_calibration: CostCalibration | None = None,
    ) -> None:
        self.settings = settings
        self.edge_lookup = edge_lookup
        if cost_calibration is not None:
            self.cost_calibration = cost_calibration
        else:
            calibration_path = Path(__file__).resolve().parents[2] / "quant_runtime" / "artifacts" / "cost_calibration.json"
            self.cost_calibration = load_cost_calibration(calibration_path)

    def build_history_context(self, state: SymbolMarketState) -> FeatureHistoryContext:
        bars_1h = _recent_closed_bars(state, "1h", 120)
        bars_4h = _recent_closed_bars(state, "4h", 120)
        returns_1h = tuple(_pct_returns(bars_1h) or [0.0])
        returns_4h = tuple(_pct_returns(bars_4h) or [0.0])
        quote_volume_5m = tuple(bar.quote_volume for bar in _recent_closed_bars(state, "5m", 120) or [])
        quote_volume_1h = tuple(bar.quote_volume for bar in bars_1h or [])
        realized_vol_1h = tuple(abs(value) for value in returns_1h)
        realized_vol_4h = tuple(abs(value) for value in returns_4h)
        funding_abs = tuple(abs(value) for value in (state.funding_rate_samples or [state.funding_rate]))
        basis_abs = tuple(abs(value) for value in (state.basis_bps_samples or [state.basis_bps]))
        oi_values = state.open_interest_samples or [state.open_interest]
        oi_surge = []
        for index, value in enumerate(oi_values):
            base = mean(oi_values[max(0, index - self.settings.feature_thresholds.oi_ema_hours + 1) : index + 1])
            oi_surge.append(max(value / max(base, 1e-9) - 1.0, 0.0))
        return FeatureHistoryContext(
            returns_1h=returns_1h,
            returns_4h=returns_4h,
            quote_volume_5m=quote_volume_5m or (0.0,),
            quote_volume_1h=quote_volume_1h or (0.0,),
            realized_vol_1h=realized_vol_1h or (0.0,),
            realized_vol_4h=realized_vol_4h or (0.0,),
            funding_abs=funding_abs,
            basis_abs=basis_abs,
            oi_surge=tuple(oi_surge or [0.0]),
        )

    def build_primitive_inputs(self, state: SymbolMarketState) -> PrimitiveInputs:
        bars_1h = _recent_closed_bars(state, "1h", 120)
        bars_4h = _recent_closed_bars(state, "4h", 120)
        bars_5m = _recent_closed_bars(state, "5m", 40)
        bars_1m = _recent_closed_bars(state, "1m", 80)
        if len(bars_1h) < 21:
            raise ValueError("at least 21 closed 1h bars are required for primitive extraction")
        if len(bars_4h) < 2 or len(bars_5m) < 2:
            raise ValueError("insufficient closed bars for primitive extraction")

        returns_1h = _pct_returns(bars_1h)
        returns_4h = _pct_returns(bars_4h)
        closes_1h = [bar.close_price for bar in bars_1h]

        ema_fast = _ema(closes_1h[-20:], min(20, len(closes_1h[-20:])))
        ema_mid = _ema(closes_1h[-50:], min(50, len(closes_1h[-50:])))
        ema_slow = _ema(closes_1h[-100:], min(100, len(closes_1h[-100:])))
        if ema_fast > ema_mid > ema_slow:
            trend_direction = 1
            ema_stack_score = 1.0
        elif ema_fast < ema_mid < ema_slow:
            trend_direction = -1
            ema_stack_score = 1.0
        elif (ema_fast > ema_mid and closes_1h[-1] > ema_slow) or (ema_fast < ema_mid and closes_1h[-1] < ema_slow):
            trend_direction = 1 if closes_1h[-1] > ema_slow else -1
            ema_stack_score = 0.5
        else:
            trend_direction = 0
            ema_stack_score = 0.0

        intraday_bias, intraday_strength = _intraday_trend_signal(
            bars_5m=bars_5m,
            bars_1m=bars_1m,
        )
        if trend_direction == 0 and intraday_bias != 0:
            trend_direction = intraday_bias
            ema_stack_score = max(ema_stack_score, round(0.4 + 0.35 * intraday_strength, 6))
        elif intraday_bias != 0 and intraday_bias == trend_direction:
            ema_stack_score = min(1.0, round(ema_stack_score + 0.25 * intraday_strength, 6))
        elif intraday_bias != 0 and intraday_bias != trend_direction and intraday_strength >= 0.45:
            trend_direction = intraday_bias
            ema_stack_score = max(0.6, round(0.5 + 0.3 * intraday_strength, 6))

        lookback = bars_1h[-21:-1]
        breakout_reference_price = (
            max(bar.high_price for bar in lookback)
            if trend_direction >= 0
            else min(bar.low_price for bar in lookback)
        )
        true_ranges = _true_ranges(bars_1h[-15:])
        atr_14_1h_price = mean(true_ranges[-14:]) if true_ranges else 0.0

        buy_taker_quote = sum(
            trade.price * trade.quantity for trade in state.trades[-100:] if not trade.is_buyer_maker
        )
        sell_taker_quote = sum(
            trade.price * trade.quantity for trade in state.trades[-100:] if trade.is_buyer_maker
        )
        spread_bps = (
            ((state.top_of_book.ask_price - state.top_of_book.bid_price) / state.last_trade_price) * 10000.0
            if state.last_trade_price > 0
            else 0.0
        )
        depth_usd_within_10bps = (
            (state.top_of_book.bid_qty + state.top_of_book.ask_qty) * state.last_trade_price
        )
        order_book_imbalance_std = pstdev(state.order_book_imbalance_samples[-30:]) if len(state.order_book_imbalance_samples) > 1 else 0.0
        realized_vol_1h = pstdev(returns_1h[-20:]) if len(returns_1h) > 1 else 0.0
        realized_vol_4h = pstdev(returns_4h[-20:]) if len(returns_4h) > 1 else 0.0
        median_realized_vol_1h_30d = median(abs(value) for value in returns_1h[-30:]) if returns_1h else 0.0
        open_interest_ema = _ema(state.open_interest_samples[-self.settings.feature_thresholds.oi_ema_hours :], min(self.settings.feature_thresholds.oi_ema_hours, len(state.open_interest_samples[-self.settings.feature_thresholds.oi_ema_hours :]))) if state.open_interest_samples else state.open_interest
        adx_1h = _adx_from_bars(bars_1h)
        _cp = get_profile(state.symbol)
        # Main EMA cross (long direction)
        ema_cross = _ema_cross_signal(closes_1h, fast_period=_cp.ema_fast, slow_period=_cp.ema_slow)
        # Also check short-specific EMA cross if profile has it
        if ema_cross == 0 and _cp.short_ema_fast > 0:
            ema_cross = _ema_cross_signal(closes_1h, fast_period=_cp.short_ema_fast, slow_period=_cp.short_ema_slow)

        gross_expected_edge_bps = 0.0
        if self.edge_lookup is not None:
            score_hint = 80.0 if trend_direction != 0 else 50.0
            lookup_value = self.edge_lookup.expected_edge_bps(
                symbol=state.symbol,
                mode="futures" if trend_direction != 0 else "spot",
                predictability_score=score_hint,
                trend_direction=trend_direction or 1,
            )
            if lookup_value is not None:
                gross_expected_edge_bps = lookup_value

        return PrimitiveInputs(
            ret_1h=returns_1h[-1],
            ret_4h=returns_4h[-1],
            trend_direction=trend_direction,
            ema_stack_score=ema_stack_score,
            breakout_reference_price=breakout_reference_price,
            last_trade_price=state.last_trade_price,
            atr_14_1h_price=atr_14_1h_price,
            quote_volume_5m=bars_5m[-1].quote_volume,
            quote_volume_1h=bars_1h[-1].quote_volume,
            buy_taker_volume=buy_taker_quote,
            sell_taker_volume=sell_taker_quote,
            spread_bps=spread_bps,
            probe_slippage_bps=spread_bps * 1.5,
            depth_usd_within_10bps=depth_usd_within_10bps,
            order_book_imbalance_std=order_book_imbalance_std,
            realized_vol_1h=realized_vol_1h,
            realized_vol_4h=realized_vol_4h,
            median_realized_vol_1h_30d=median_realized_vol_1h_30d,
            funding_rate=state.funding_rate,
            open_interest=state.open_interest,
            open_interest_ema=open_interest_ema,
            basis_bps=state.basis_bps,
            gross_expected_edge_bps=gross_expected_edge_bps,
            intraday_trend_direction=intraday_bias,
            intraday_trend_strength=intraday_strength,
            adx_1h=adx_1h,
            ema_cross_signal=ema_cross,
        )

    def _compute_b3_msb_signal(self, bars_1h: list[KlineBar], state: SymbolMarketState) -> tuple[int, float]:
        """B3 Market Structure Break 시그널 계산.

        Returns:
            (signal, strength): signal은 -1/0/1, strength는 0~1.
        """
        cfg = self.settings.b3_msb
        if not cfg.enabled:
            return 0, 0.0

        sw = cfg.swing_window
        if len(bars_1h) < sw + 15:
            return 0, 0.0

        # 스윙 구간 (현재 봉 제외, 직전 sw개)
        swing_bars = bars_1h[-(sw + 1):-1]
        swing_high = max(bar.high_price for bar in swing_bars)
        swing_low = min(bar.low_price for bar in swing_bars)
        swing_range = swing_high - swing_low

        # ATR 계산 (최근 14개 1h 바)
        recent = bars_1h[-15:]
        true_ranges = []
        for i in range(1, len(recent)):
            tr = max(
                recent[i].high_price - recent[i].low_price,
                abs(recent[i].high_price - recent[i - 1].close_price),
                abs(recent[i].low_price - recent[i - 1].close_price),
            )
            true_ranges.append(tr)
        atr = sum(true_ranges) / len(true_ranges) if true_ranges else 0.0

        if atr <= 0:
            return 0, 0.0

        # 최소 스윙 크기 체크
        if swing_range < cfg.min_swing_size_atr * atr:
            return 0, 0.0

        price = state.last_trade_price
        buf = cfg.breakout_buffer_pct

        # ADX 계산 (간이 - ADX가 이미 features에 없으므로 직접 계산)
        # Hyperopt 결과 adx_min=4이므로 거의 항상 통과, 간이 체크
        if cfg.adx_min > 0 and len(bars_1h) >= 28:
            closes = [b.close_price for b in bars_1h[-28:]]
            highs = [b.high_price for b in bars_1h[-28:]]
            lows = [b.low_price for b in bars_1h[-28:]]
            plus_dm_sum = 0.0
            minus_dm_sum = 0.0
            tr_sum = 0.0
            for i in range(1, min(15, len(closes))):
                up = highs[i] - highs[i - 1]
                dn = lows[i - 1] - lows[i]
                plus_dm_sum += up if (up > dn and up > 0) else 0.0
                minus_dm_sum += dn if (dn > up and dn > 0) else 0.0
                tr_sum += max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
            if tr_sum > 0:
                pdi = 100 * plus_dm_sum / tr_sum
                mdi = 100 * minus_dm_sum / tr_sum
                denom = pdi + mdi
                adx_approx = 100 * abs(pdi - mdi) / denom if denom > 0 else 0.0
            else:
                adx_approx = 0.0
            if adx_approx < cfg.adx_min:
                return 0, 0.0

        # RSI 체크 (간이 RSI14)
        closes_recent = [b.close_price for b in bars_1h[-15:]]
        gains, losses = [], []
        for i in range(1, len(closes_recent)):
            diff = closes_recent[i] - closes_recent[i - 1]
            gains.append(max(diff, 0))
            losses.append(max(-diff, 0))
        avg_gain = sum(gains) / len(gains) if gains else 0.0
        avg_loss = sum(losses) / len(losses) if losses else 1e-9
        rs = avg_gain / max(avg_loss, 1e-9)
        rsi = 100 - (100 / (1 + rs))

        # Volume z-score 체크
        if cfg.vol_z_min > 0 and len(bars_1h) >= 21:
            vols = [b.quote_volume for b in bars_1h[-21:]]
            vol_mean = sum(vols[:-1]) / len(vols[:-1])
            vol_std = (sum((v - vol_mean) ** 2 for v in vols[:-1]) / len(vols[:-1])) ** 0.5
            vol_z = (vols[-1] - vol_mean) / max(vol_std, 1e-9)
            if vol_z < cfg.vol_z_min:
                return 0, 0.0

        # 돌파 시그널
        long_breakout = price > swing_high * (1.0 + buf)
        short_breakout = price < swing_low * (1.0 - buf)

        if long_breakout and rsi < cfg.rsi_upper:
            # 강도: 돌파 거리 / ATR (클수록 강한 돌파)
            breakout_dist = (price - swing_high * (1.0 + buf)) / atr
            strength = min(breakout_dist / 2.0, 1.0)  # 2 ATR 이상이면 1.0
            return 1, max(strength, 0.1)

        if short_breakout and rsi > cfg.rsi_lower:
            breakout_dist = (swing_low * (1.0 - buf) - price) / atr
            strength = min(breakout_dist / 2.0, 1.0)
            return -1, max(strength, 0.1)

        return 0, 0.0

    def enrich_feature_vector(self, *, state: SymbolMarketState, features: FeatureVector) -> FeatureVector:
        bars_1h = _recent_closed_bars(state, "1h", 120)
        if len(bars_1h) < 50:
            return features
        closes_1h = [bar.close_price for bar in bars_1h]
        ema20 = _ema(closes_1h[-20:], min(20, len(closes_1h[-20:])))
        ema50 = _ema(closes_1h[-50:], min(50, len(closes_1h[-50:])))
        lookback = bars_1h[-21:-1]
        local_low = min(bar.low_price for bar in lookback)
        local_high = max(bar.high_price for bar in lookback)
        range_size = max(local_high - local_low, 1e-9)
        fib50 = local_low + 0.5 * range_size
        fib618 = local_low + 0.618 * range_size

        support_hits = 0
        support_hits += 1 if abs(state.last_trade_price - ema20) / state.last_trade_price <= 0.01 else 0
        support_hits += 1 if fib50 <= state.last_trade_price <= fib618 else 0
        support_hits += 1 if abs(state.last_trade_price - local_low) / state.last_trade_price <= 0.012 else 0
        support_alignment = min(support_hits / 3.0, 1.0)

        resistance_hits = 0
        resistance_hits += 1 if abs(state.last_trade_price - ema50) / state.last_trade_price <= 0.01 else 0
        resistance_hits += 1 if abs(state.last_trade_price - local_high) / state.last_trade_price <= 0.012 else 0
        resistance_penalty = min(resistance_hits / 2.0, 1.0)

        pullback = _pullback_signal(closes_1h, ema_period=21, rsi_entry=40.0)
        b3_signal, b3_strength = self._compute_b3_msb_signal(bars_1h, state)

        # --- Advanced signals ---
        oi_div = oi_divergence_from_state(state, lookback=24)
        vwap_price, vwap_dev_z = vwap_deviation_from_state(state)
        smc = smc_signals_from_state(state, trend_direction=features.trend_direction)

        enriched = FeatureVector(
            **{
                **features.as_dict(),
                "support_alignment": round(support_alignment, 6),
                "resistance_penalty": round(resistance_penalty, 6),
                "pullback_signal": pullback,
                "b3_msb_signal": b3_signal,
                "b3_msb_strength": round(b3_strength, 6),
                "oi_divergence_score": oi_div,
                "vwap_price": vwap_price,
                "vwap_deviation_z": vwap_dev_z,
                "smc_fvg_score": smc.fvg_score,
                "smc_ob_score": smc.ob_score,
                "smc_structure_score": smc.structure_score,
                "smc_composite_score": smc.composite_score,
            }
        )
        if self.cost_calibration is not None:
            calibration = self.cost_calibration.for_symbol(state.symbol)
            enriched = FeatureVector(
                **{
                    **enriched.as_dict(),
                    "empirical_fee_bps": calibration.empirical_fee_bps,
                    "empirical_entry_slippage_bps": calibration.empirical_entry_slippage_bps,
                    "empirical_exit_slippage_bps": calibration.empirical_exit_slippage_bps,
                }
            )
        score = compute_predictability_score(enriched, self.settings)
        return FeatureVector(**{**enriched.as_dict(), "predictability_score": score})
