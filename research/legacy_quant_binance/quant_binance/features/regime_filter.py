"""Regime detection via Hurst exponent (DFA) + HAR-RV volatility forecast.

References:
- Peng et al 1994 — Detrended Fluctuation Analysis (DFA)
- Di Matteo 2007 — multi-scale Hurst for market inefficiency
- Corsi 2009 — HAR-RV model
- Bollerslev et al 2016 — HAR-RV with jump component
- Bariviera 2017 — Hurst in cryptocurrency markets
- Catania & Sandholdt 2019 — HAR-RV in crypto
"""
from __future__ import annotations

import math
from typing import Sequence


def hurst_dfa(returns: Sequence[float], min_lag: int = 4, max_lag: int = 64) -> float:
    """DFA-based Hurst exponent from RETURNS (not prices).

    Peng et al (1994) Detrended Fluctuation Analysis.
    Works better than R/S for financial time series with trends.

    H > 0.5 → persistent (trending)
    H ≈ 0.5 → random walk
    H < 0.5 → anti-persistent (mean-reverting)
    """
    n = len(returns)
    if n < max_lag * 4:
        return 0.5

    # Cumulative sum of mean-centered returns (profile)
    mean_r = sum(returns) / n
    profile = []
    acc = 0.0
    for r in returns:
        acc += r - mean_r
        profile.append(acc)

    lags = []
    f_values = []

    lag = min_lag
    while lag <= max_lag and lag < n // 4:
        n_segments = n // lag
        if n_segments < 4:
            break
        f2_list = []
        for s in range(n_segments):
            seg = profile[s * lag:(s + 1) * lag]
            # Linear detrend
            x_vals = list(range(lag))
            mean_x = sum(x_vals) / lag
            mean_y = sum(seg) / lag
            num = sum((x_vals[i] - mean_x) * (seg[i] - mean_y) for i in range(lag))
            den = sum((x_vals[i] - mean_x) ** 2 for i in range(lag))
            if den == 0:
                continue
            slope = num / den
            intercept = mean_y - slope * mean_x
            residuals = [seg[i] - (slope * x_vals[i] + intercept) for i in range(lag)]
            f2 = sum(r * r for r in residuals) / lag
            f2_list.append(f2)
        if f2_list:
            f_avg = math.sqrt(sum(f2_list) / len(f2_list))
            if f_avg > 0:
                lags.append(lag)
                f_values.append(f_avg)
        lag = int(lag * 1.3)

    if len(lags) < 4:
        return 0.5

    # Log-log fit
    log_lags = [math.log(l) for l in lags]
    log_f = [math.log(f) for f in f_values]
    n_pts = len(log_lags)
    mean_x = sum(log_lags) / n_pts
    mean_y = sum(log_f) / n_pts
    num = sum((log_lags[i] - mean_x) * (log_f[i] - mean_y) for i in range(n_pts))
    den = sum((log_lags[i] - mean_x) ** 2 for i in range(n_pts))
    if den == 0:
        return 0.5
    hurst = num / den
    return max(0.0, min(1.0, hurst))


def hurst_rs(prices: Sequence[float], min_lag: int = 10, max_lag: int = 100) -> float:
    """R/S analysis (kept for reference, but DFA is preferred)."""
    n = len(prices)
    if n < max_lag + 10:
        return 0.5

    lags = []
    rs_values = []

    for lag in range(min_lag, min(max_lag, n // 2), max(1, (max_lag - min_lag) // 20)):
        rs_chunk = []
        for start in range(0, n - lag, lag):
            segment = list(prices[start:start + lag])
            if len(segment) < lag:
                continue
            mean = sum(segment) / lag
            deviations = [p - mean for p in segment]
            cum_dev = []
            acc = 0.0
            for d in deviations:
                acc += d
                cum_dev.append(acc)
            R = max(cum_dev) - min(cum_dev)
            mean_sq = sum(d * d for d in deviations) / lag
            S = math.sqrt(mean_sq)
            if S > 0:
                rs_chunk.append(R / S)
        if rs_chunk:
            lags.append(lag)
            rs_values.append(sum(rs_chunk) / len(rs_chunk))

    if len(lags) < 4:
        return 0.5

    log_lags = [math.log(l) for l in lags]
    log_rs = [math.log(r) if r > 0 else 0 for r in rs_values]
    n_pts = len(log_lags)
    mean_x = sum(log_lags) / n_pts
    mean_y = sum(log_rs) / n_pts
    num = sum((log_lags[i] - mean_x) * (log_rs[i] - mean_y) for i in range(n_pts))
    den = sum((log_lags[i] - mean_x) ** 2 for i in range(n_pts))
    if den == 0:
        return 0.5
    hurst = num / den
    return max(0.0, min(1.0, hurst))


def realized_variance(returns: Sequence[float]) -> float:
    return sum(r * r for r in returns)


def har_rv_forecast(
    returns_5m: Sequence[float],
    weights: tuple[float, float, float] = (0.35, 0.35, 0.30),
) -> tuple[float, float, float, float]:
    """HAR-RV realized variance forecast."""
    n = len(returns_5m)
    if n < 2016:
        return 0.0, 0.0, 0.0, 0.0

    r_bps = [r * 10000 for r in returns_5m]
    rv_d = realized_variance(r_bps[-288:])
    rv_w = realized_variance(r_bps[-2016:])
    rv_m_samples = r_bps[-8640:] if n >= 8640 else r_bps
    rv_m = realized_variance(rv_m_samples) * (8640 / len(rv_m_samples)) if rv_m_samples else rv_w

    w_d, w_w, w_m = weights
    forecast = w_d * (rv_d / 288) + w_w * (rv_w / 2016) + w_m * (rv_m / 8640)
    return forecast, rv_d, rv_w, rv_m


def classify_regime(
    hurst: float,
    trending_threshold: float = 0.55,
    reverting_threshold: float = 0.45,
) -> str:
    if hurst > trending_threshold:
        return "trending"
    if hurst < reverting_threshold:
        return "reverting"
    return "random"


def vol_regime(
    rv_forecast: float,
    historical_rv_median: float,
) -> str:
    if historical_rv_median <= 0:
        return "normal"
    ratio = rv_forecast / historical_rv_median
    if ratio > 1.5:
        return "high"
    if ratio < 0.5:
        return "low"
    return "normal"


def entry_allowed(
    hurst: float,
    vol_regime_value: str,
    strategy_type: str,
) -> bool:
    price_regime = classify_regime(hurst)
    if strategy_type == "reversal":
        return price_regime == "reverting" and vol_regime_value in ("high", "normal")
    if strategy_type == "trend":
        return price_regime == "trending" and vol_regime_value in ("high", "normal")
    return False
