"""No-lookahead signal construction for the predeclared search families."""

from __future__ import annotations

from typing import TypeAlias, assert_never

import numpy as np
import numpy.typing as npt

from alt_search_models import Candidate, Family, FloatMatrix, MarketData

BoolMatrix: TypeAlias = npt.NDArray[np.bool_]


def _cross_sectional_positions(score: FloatMatrix, top_k: int, direction: int, entry_threshold: float = 0.0, eligibility: BoolMatrix | None = None) -> FloatMatrix:
    """Build neutral equal-weight top/bottom positions from row-wise scores."""

    positions = np.zeros_like(score)
    for row in range(score.shape[0]):
        values = score[row]
        if eligibility is None:
            valid_indices = np.arange(len(values))
            if not np.isfinite(values).all():
                continue
        else:
            valid_indices = np.flatnonzero(np.isfinite(values) & eligibility[row])
        if top_k <= 0 or 2 * top_k > len(valid_indices):
            continue
        ordered = valid_indices[np.argsort(values[valid_indices], kind="stable")]
        if entry_threshold > 0.0:
            negative, positive = ordered[values[ordered] <= -entry_threshold], ordered[values[ordered] >= entry_threshold]
            if direction > 0:
                long_indices, short_indices = positive[-top_k:], negative[:top_k]
            else:
                long_indices, short_indices = negative[:top_k], positive[-top_k:]
        else:
            long_indices, short_indices = ordered[-top_k:], ordered[:top_k]
            if direction < 0:
                long_indices, short_indices = short_indices, long_indices
        count = min(len(long_indices), len(short_indices))
        if count == 0:
            continue
        long_indices, short_indices = long_indices[:count], short_indices[:count]
        weight = 0.5 / count
        positions[row, long_indices] = weight
        positions[row, short_indices] = -weight
    return positions


def _time_series_positions(close: FloatMatrix, lookback: int, long_short: bool) -> FloatMatrix:
    """Build positions at t using returns that end no later than t-1."""

    positions = np.zeros_like(close)
    for index in range(lookback + 1, len(close)):
        signals = np.sign(close[index - 1] / close[index - 1 - lookback] - 1.0)
        if long_short:
            active = signals != 0.0
            count = int(active.sum())
            if count:
                positions[index, active] = signals[active] / count
        else:
            active = signals > 0.0
            count = int(active.sum())
            if count:
                positions[index, active] = 1.0 / count
    return positions


def _donchian_positions(data: MarketData, candidate: Candidate) -> FloatMatrix:
    """Build prior-bar Donchian breakout positions."""

    positions = np.zeros_like(data.close_price)
    for index in range(candidate.lookback + 1, len(positions)):
        prior_high = data.high_price[index - candidate.lookback - 1 : index - 1].max(axis=0)
        prior_low = data.low_price[index - candidate.lookback - 1 : index - 1].min(axis=0)
        latest = data.close_price[index - 1]
        long = latest > prior_high
        short = latest < prior_low
        if candidate.direction > 0:
            signs = long.astype(float) - short.astype(float)
        else:
            signs = long.astype(float)
        active = signs != 0.0
        count = int(active.sum())
        if count:
            positions[index, active] = signs[active] / count
    return positions


def _ema(close: FloatMatrix, span: int) -> FloatMatrix:
    """Return a column-wise exponential moving average."""

    output = np.empty_like(close)
    output[0] = close[0]
    alpha = 2.0 / (span + 1.0)
    for index in range(1, len(close)):
        output[index] = alpha * close[index] + (1.0 - alpha) * output[index - 1]
    return output


def _ema_positions(data: MarketData, candidate: Candidate) -> FloatMatrix:
    """Build lagged EMA-cross positions."""

    fast, slow = _ema(data.close_price, candidate.lookback), _ema(data.close_price, candidate.secondary)
    positions = np.zeros_like(data.close_price)
    for index in range(candidate.secondary + 1, len(positions)):
        signs = np.sign(fast[index - 1] - slow[index - 1])
        if candidate.direction < 0:
            signs = np.maximum(signs, 0.0)
        active = signs != 0.0
        count = int(active.sum())
        if count:
            positions[index, active] = signs[active] / count
    return positions


def _bollinger_score(close: FloatMatrix, lookback: int) -> FloatMatrix:
    """Return a lagged z-score of the last close within its prior window."""

    score = np.full_like(close, np.nan)
    for index in range(lookback + 1, len(close)):
        window = close[index - lookback : index]
        std = window.std(axis=0)
        valid = std > 0.0
        score[index, valid] = (close[index - 1, valid] - window[:, valid].mean(axis=0)) / std[valid]
    return score


def _range_score(data: MarketData, lookback: int) -> FloatMatrix:
    """Return lagged return divided by lagged mean intrabar range."""

    score = np.full_like(data.close_price, np.nan)
    ranges = (data.high_price - data.low_price) / data.close_price
    for index in range(lookback + 1, len(score)):
        mean_range = ranges[index - lookback : index].mean(axis=0)
        valid = mean_range > 0.0
        momentum = data.close_price[index - 1] / data.close_price[index - 1 - lookback] - 1.0
        score[index, valid] = momentum[valid] / mean_range[valid]
    return score


def _price_volume_score(data: MarketData, lookback: int) -> FloatMatrix:
    """Return lagged price momentum confirmed by relative quote volume."""

    score = np.full_like(data.close_price, np.nan)
    for index in range(lookback + 1, len(score)):
        price = data.close_price[index - 1] / data.close_price[index - 1 - lookback] - 1.0
        volume_mean = data.quote_volume[index - lookback : index].mean(axis=0)
        valid = volume_mean > 0.0
        ratio = np.zeros_like(volume_mean)
        np.divide(data.quote_volume[index - 1], volume_mean, out=ratio, where=valid)
        score[index, valid] = price[valid] * ratio[valid]
    return score


def _pair_positions(data: MarketData, candidate: Candidate) -> FloatMatrix:
    """Build z-score mean-reversion positions for one fixed altcoin pair."""

    positions = np.zeros_like(data.close_price)
    if candidate.pair is None:
        return positions
    base, quote = (data.symbols.index(symbol) for symbol in candidate.pair)
    spread = np.log(data.close_price[:, base]) - np.log(data.close_price[:, quote])
    for index in range(candidate.lookback + 1, len(positions)):
        window = spread[index - candidate.lookback : index]
        std = float(window.std())
        if std == 0.0:
            continue
        zscore = float((spread[index - 1] - window.mean()) / std)
        if zscore >= candidate.entry_z:
            positions[index, base], positions[index, quote] = -0.5, 0.5
        elif zscore <= -candidate.entry_z:
            positions[index, base], positions[index, quote] = 0.5, -0.5
    return positions


def _hold_between_rebalances(positions: FloatMatrix, rebalance_days: int) -> FloatMatrix:
    """Hold the last known position between scheduled rebalances."""

    held = positions.copy()
    for index in range(1, len(held)):
        if index % rebalance_days:
            held[index] = held[index - 1]
    return held


def _prior_volume_eligibility(data: MarketData, floor: float) -> BoolMatrix:
    """Return a no-lookahead eligibility mask based on prior closed turnover."""

    eligibility = np.zeros(data.quote_volume.shape, dtype=bool)
    eligibility[1:] = data.quote_volume[:-1] >= floor
    return eligibility


def _apply_liquidity_guard(positions: FloatMatrix, eligibility: BoolMatrix, candidate: Candidate) -> FloatMatrix:
    """Remove stale or illiquid exposure without creating a partial market-neutral trade."""

    guarded = positions.copy()
    market_neutral = {Family.CROSS_SECTIONAL, Family.BOLLINGER, Family.RANGE, Family.PRICE_VOLUME, Family.PAIR}
    if candidate.family in market_neutral:
        active = np.abs(guarded) > 0.0
        compromised = np.any(active & ~eligibility, axis=1)
        guarded[compromised] = 0.0
        return guarded
    guarded[~eligibility] = 0.0
    return guarded


def build_positions(data: MarketData, candidate: Candidate) -> FloatMatrix:
    """Build candidate positions known before each applied return."""

    eligibility = _prior_volume_eligibility(data, candidate.minimum_daily_quote_volume) if candidate.minimum_daily_quote_volume > 0.0 else None
    match candidate.family:
        case Family.CROSS_SECTIONAL:
            score = np.full_like(data.close_price, np.nan)
            for index in range(candidate.lookback + 1, len(score)):
                score[index] = data.close_price[index - 1] / data.close_price[index - 1 - candidate.lookback] - 1.0
            positions = _cross_sectional_positions(score, candidate.top_k, candidate.direction, eligibility=eligibility)
        case Family.TIME_SERIES:
            positions = _time_series_positions(data.close_price, candidate.lookback, candidate.direction > 0)
        case Family.DONCHIAN:
            positions = _donchian_positions(data, candidate)
        case Family.EMA:
            positions = _ema_positions(data, candidate)
        case Family.BOLLINGER:
            positions = _cross_sectional_positions(_bollinger_score(data.close_price, candidate.lookback), candidate.top_k, -1, candidate.entry_z, eligibility)
        case Family.RANGE:
            positions = _cross_sectional_positions(_range_score(data, candidate.lookback), candidate.top_k, candidate.direction, eligibility=eligibility)
        case Family.PRICE_VOLUME:
            positions = _cross_sectional_positions(_price_volume_score(data, candidate.lookback), candidate.top_k, candidate.direction, eligibility=eligibility)
        case Family.PAIR:
            positions = _pair_positions(data, candidate)
        case unreachable:
            assert_never(unreachable)
    held = _hold_between_rebalances(positions, candidate.rebalance_days)
    return held if eligibility is None else _apply_liquidity_guard(held, eligibility, candidate)
