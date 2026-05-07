"""Backtesting engine with indicator calculations.

All indicators use numpy for performance.
Components are composable: signals, filters, risk management, sizing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ─── Indicator Functions ───

def sma(data: np.ndarray, period: int) -> np.ndarray:
    out = np.full_like(data, np.nan)
    if len(data) < period:
        return out
    cs = np.cumsum(data)
    cs[period:] = cs[period:] - cs[:-period]
    out[period - 1:] = cs[period - 1:] / period
    return out


def ema(data: np.ndarray, period: int) -> np.ndarray:
    out = np.full_like(data, np.nan)
    if len(data) < period:
        return out
    k = 2.0 / (period + 1)
    out[period - 1] = np.mean(data[:period])
    for i in range(period, len(data)):
        out[i] = data[i] * k + out[i - 1] * (1 - k)
    return out


def rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    out = np.full_like(close, np.nan)
    if len(close) < period + 1:
        return out
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = np.mean(gain[:period])
    avg_loss = np.mean(loss[:period])
    if avg_loss == 0:
        out[period] = 100.0
    else:
        out[period] = 100.0 - 100.0 / (1.0 + avg_gain / max(avg_loss, 1e-10))
    for i in range(period, len(delta)):
        avg_gain = (avg_gain * (period - 1) + gain[i]) / period
        avg_loss = (avg_loss * (period - 1) + loss[i]) / period
        if avg_loss == 0:
            out[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            out[i + 1] = 100.0 - 100.0 / (1.0 + rs)
    return out


def macd(close: np.ndarray, fast: int = 12, slow: int = 26, signal_period: int = 9):
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line[~np.isnan(macd_line)], signal_period)
    full_signal = np.full_like(close, np.nan)
    valid_start = np.argmin(np.isnan(macd_line))
    if len(signal_line) > 0:
        offset = valid_start
        full_signal[offset:offset + len(signal_line)] = signal_line
    histogram = macd_line - full_signal
    return macd_line, full_signal, histogram


def bollinger(close: np.ndarray, period: int = 20, std_mult: float = 2.0):
    mid = sma(close, period)
    std = np.full_like(close, np.nan)
    for i in range(period - 1, len(close)):
        std[i] = np.std(close[i - period + 1:i + 1])
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    pct_b = np.where(upper != lower, (close - lower) / (upper - lower), 0.5)
    return upper, mid, lower, pct_b


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    tr = np.maximum(high[1:] - low[1:],
                    np.maximum(np.abs(high[1:] - close[:-1]),
                               np.abs(low[1:] - close[:-1])))
    tr = np.insert(tr, 0, high[0] - low[0])
    out = np.full_like(close, np.nan)
    if len(tr) < period:
        return out
    out[period - 1] = np.mean(tr[:period])
    for i in range(period, len(tr)):
        out[i] = (out[i - 1] * (period - 1) + tr[i]) / period
    return out


def adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    n = len(close)
    out = np.full(n, np.nan)
    if n < period * 2:
        return out
    up_move = high[1:] - high[:-1]
    down_move = low[:-1] - low[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr_vals = atr(high, low, close, period)
    plus_di = np.full(n, np.nan)
    minus_di = np.full(n, np.nan)
    smooth_plus = np.mean(plus_dm[:period])
    smooth_minus = np.mean(minus_dm[:period])
    for i in range(period, n - 1):
        smooth_plus = (smooth_plus * (period - 1) + plus_dm[i]) / period
        smooth_minus = (smooth_minus * (period - 1) + minus_dm[i]) / period
        if not np.isnan(atr_vals[i + 1]) and atr_vals[i + 1] > 0:
            plus_di[i + 1] = 100 * smooth_plus / atr_vals[i + 1]
            minus_di[i + 1] = 100 * smooth_minus / atr_vals[i + 1]
    dx = np.where((plus_di + minus_di) > 0,
                  100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    valid_dx = dx[~np.isnan(dx)]
    if len(valid_dx) < period:
        return out
    adx_val = np.mean(valid_dx[:period])
    start_idx = np.argmin(np.isnan(dx)) + period
    if start_idx < n:
        out[start_idx] = adx_val
        for i in range(start_idx + 1, n):
            if not np.isnan(dx[i]):
                adx_val = (adx_val * (period - 1) + dx[i]) / period
                out[i] = adx_val
    return out


def vwap(high: np.ndarray, low: np.ndarray, close: np.ndarray,
         volume: np.ndarray, period: int = 20) -> np.ndarray:
    tp = (high + low + close) / 3.0
    out = np.full_like(close, np.nan)
    for i in range(period - 1, len(close)):
        s = slice(i - period + 1, i + 1)
        vol_sum = np.sum(volume[s])
        if vol_sum > 0:
            out[i] = np.sum(tp[s] * volume[s]) / vol_sum
        else:
            out[i] = tp[i]
    return out


def vwap_zscore(close: np.ndarray, vwap_vals: np.ndarray, period: int = 20) -> np.ndarray:
    diff = close - vwap_vals
    out = np.full_like(close, np.nan)
    for i in range(period - 1, len(close)):
        s = slice(i - period + 1, i + 1)
        std = np.std(diff[s])
        if std > 0:
            out[i] = diff[i] / std
    return out


def volume_ratio(volume: np.ndarray, period: int = 20) -> np.ndarray:
    vol_ma = sma(volume, period)
    return np.where(vol_ma > 0, volume / vol_ma, 1.0)


def obv(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    direction = np.sign(np.diff(close))
    direction = np.insert(direction, 0, 0)
    return np.cumsum(direction * volume)


def stoch_rsi(close: np.ndarray, rsi_period: int = 14, stoch_period: int = 14,
              k_period: int = 3, d_period: int = 3):
    rsi_vals = rsi(close, rsi_period)
    k = np.full_like(close, np.nan)
    for i in range(stoch_period - 1, len(rsi_vals)):
        if np.isnan(rsi_vals[i]):
            continue
        window = rsi_vals[max(0, i - stoch_period + 1):i + 1]
        window = window[~np.isnan(window)]
        if len(window) == 0:
            continue
        lo, hi = np.min(window), np.max(window)
        k[i] = ((rsi_vals[i] - lo) / (hi - lo) * 100) if hi > lo else 50.0
    d = sma(k[~np.isnan(k)], d_period) if np.sum(~np.isnan(k)) >= d_period else k
    return k, d


# ─── Component Definitions ───

COMPONENTS = {
    "signals": [
        {
            "id": "ema_cross", "name": "EMA 크로스",
            "desc": "두 개의 평균선을 비교합니다. 짧은 기간 평균이 긴 기간 평균 위로 올라가면 '상승 시작'으로 보고 매수하고, 아래로 내려가면 '하락 시작'으로 보고 매도합니다. 가장 기본적인 전략이에요.",
            "params": [
                {"key": "fast", "label": "빠른 EMA", "type": "int", "default": 9, "min": 2, "max": 200},
                {"key": "slow", "label": "느린 EMA", "type": "int", "default": 21, "min": 5, "max": 500},
            ],
        },
        {
            "id": "rsi_threshold", "name": "RSI 과매수/과매도",
            "desc": "RSI는 0~100 사이 값으로 '지금 너무 많이 올랐나/떨어졌나'를 알려줍니다. 30 이하면 '너무 떨어져서 반등할 수 있다' → 매수, 70 이상이면 '너무 올라서 떨어질 수 있다' → 매도합니다.",
            "params": [
                {"key": "period", "label": "기간", "type": "int", "default": 14, "min": 2, "max": 100},
                {"key": "oversold", "label": "과매도", "type": "int", "default": 30, "min": 5, "max": 50},
                {"key": "overbought", "label": "과매수", "type": "int", "default": 70, "min": 50, "max": 95},
            ],
        },
        {
            "id": "macd_cross", "name": "MACD 크로스",
            "desc": "두 평균선의 차이(MACD)와 그 평균(시그널)을 비교합니다. MACD가 시그널 위로 올라가면 '상승 전환' 매수 신호. EMA 크로스보다 느리지만 더 신뢰도가 높습니다.",
            "params": [
                {"key": "fast", "label": "빠른", "type": "int", "default": 12, "min": 2, "max": 50},
                {"key": "slow", "label": "느린", "type": "int", "default": 26, "min": 10, "max": 100},
                {"key": "signal", "label": "시그널", "type": "int", "default": 9, "min": 2, "max": 50},
            ],
        },
        {
            "id": "boll_bounce", "name": "볼린저 밴드 반등",
            "desc": "가격 위아래로 '정상 범위' 밴드를 그립니다. 가격이 아래 밴드까지 떨어지면 '싸다=반등 기대' 매수, 위 밴드까지 올라가면 '비싸다' 매도. 가격이 오르내리는 횡보장에서 효과적이에요.",
            "params": [
                {"key": "period", "label": "기간", "type": "int", "default": 20, "min": 5, "max": 100},
                {"key": "std", "label": "표준편차", "type": "float", "default": 2.0, "min": 0.5, "max": 4.0, "step": 0.1},
            ],
        },
        {
            "id": "boll_breakout", "name": "볼린저 밴드 돌파",
            "desc": "위 밴드(정상 범위)를 뚫고 올라가면 '강한 상승이 시작됐다'고 보고 매수합니다. 위의 '반등'과 반대 개념으로, 한 방향으로 쭉 가는 추세장에서 효과적이에요.",
            "params": [
                {"key": "period", "label": "기간", "type": "int", "default": 20, "min": 5, "max": 100},
                {"key": "std", "label": "표준편차", "type": "float", "default": 2.0, "min": 0.5, "max": 4.0, "step": 0.1},
            ],
        },
        {
            "id": "vwap_mean_reversion", "name": "VWAP 평균회귀",
            "desc": "거래량을 고려한 '진짜 평균가격(VWAP)'에서 가격이 너무 멀어지면 다시 평균으로 돌아올 거라 보고 진입합니다. 시장이 조용하고 방향 없이 오르내릴 때 쓰세요.",
            "params": [
                {"key": "period", "label": "기간", "type": "int", "default": 20, "min": 5, "max": 100},
                {"key": "z_threshold", "label": "Z-Score 임계", "type": "float", "default": 2.0, "min": 0.5, "max": 4.0, "step": 0.1},
            ],
        },
        {
            "id": "price_breakout", "name": "가격 돌파",
            "desc": "최근 N개 봉 중 가장 높은 가격을 뚫고 올라가면 매수, 가장 낮은 가격을 뚫고 내려가면 매도합니다. '신고가 갱신 = 더 오른다'라는 단순하지만 강력한 논리예요.",
            "params": [
                {"key": "lookback", "label": "돌파 기간", "type": "int", "default": 20, "min": 5, "max": 200},
            ],
        },
        {
            "id": "stoch_rsi_cross", "name": "Stochastic RSI",
            "desc": "RSI를 한번 더 가공해서 더 민감하게 만든 지표입니다. 매수/매도 타이밍을 RSI보다 빨리 잡아줍니다. 대신 잘못된 신호도 더 자주 나올 수 있어서 필터와 함께 쓰는 게 좋아요.",
            "params": [
                {"key": "rsi_period", "label": "RSI 기간", "type": "int", "default": 14, "min": 5, "max": 50},
                {"key": "stoch_period", "label": "Stoch 기간", "type": "int", "default": 14, "min": 5, "max": 50},
                {"key": "oversold", "label": "과매도", "type": "int", "default": 20, "min": 5, "max": 50},
                {"key": "overbought", "label": "과매수", "type": "int", "default": 80, "min": 50, "max": 95},
            ],
        },
        {
            "id": "volume_breakout", "name": "거래량 폭증 + 방향",
            "desc": "갑자기 거래량이 평소의 N배 이상 터지면 '큰손이 움직인다'는 뜻입니다. 이때 가격이 올랐으면 매수, 떨어졌으면 매도합니다. 뉴스나 큰 이벤트 때 효과적이에요.",
            "params": [
                {"key": "vol_mult", "label": "거래량 배수", "type": "float", "default": 2.0, "min": 1.2, "max": 5.0, "step": 0.1},
                {"key": "ma_period", "label": "평균 기간", "type": "int", "default": 20, "min": 5, "max": 100},
            ],
        },
        {
            "id": "williams_r", "name": "Williams %R",
            "desc": "0~-100 사이 값으로 과매수/과매도를 판단합니다. RSI보다 더 민감하고 빠른 신호를 줍니다. -80 이하면 '바닥 근처' 매수, -20 이상이면 '천장 근처' 매도. 단타에 특히 좋아요.",
            "params": [
                {"key": "period", "label": "기간", "type": "int", "default": 14, "min": 5, "max": 100},
                {"key": "oversold", "label": "과매도", "type": "int", "default": -80, "min": -95, "max": -50},
                {"key": "overbought", "label": "과매수", "type": "int", "default": -20, "min": -50, "max": -5},
            ],
        },
        {
            "id": "obv_divergence", "name": "OBV 다이버전스",
            "desc": "가격은 오르는데 거래량 흐름(OBV)이 안 따라가면 '가짜 상승'입니다. 반대로 가격은 떨어지는데 OBV가 올라가면 '곧 반등' 신호. 추세 반전을 미리 잡을 수 있어요.",
            "params": [
                {"key": "lookback", "label": "비교 기간", "type": "int", "default": 14, "min": 5, "max": 50},
            ],
        },
        {
            "id": "cci_signal", "name": "CCI (상품채널지수)",
            "desc": "가격이 평균에서 얼마나 벗어났는지 측정합니다. +100 위로 올라가면 매수, -100 아래로 내려가면 매도. 주기적으로 오르내리는 코인에서 잘 맞는 지표예요.",
            "params": [
                {"key": "period", "label": "기간", "type": "int", "default": 20, "min": 5, "max": 100},
                {"key": "threshold", "label": "기준선", "type": "int", "default": 100, "min": 50, "max": 200},
            ],
        },
    ],
    "filters": [
        {
            "id": "trend_filter", "name": "추세 필터",
            "desc": "큰 흐름(장기 평균선)이 올라가고 있을 때만 매수를 허용합니다. '큰 물결을 거스르지 말자'라는 원칙이에요. 가장 기본적이면서 효과 좋은 필터입니다.",
            "params": [
                {"key": "period", "label": "EMA 기간", "type": "int", "default": 50, "min": 10, "max": 500},
            ],
        },
        {
            "id": "adx_filter", "name": "ADX 추세 강도",
            "desc": "지금 시장에 '방향성이 있는지' 측정합니다. ADX 25 이상이면 추세가 있는 상태. 방향 없이 왔다갔다하는 구간에서 헛거래하는 걸 막아줍니다.",
            "params": [
                {"key": "period", "label": "기간", "type": "int", "default": 14, "min": 5, "max": 50},
                {"key": "threshold", "label": "최소 ADX", "type": "int", "default": 25, "min": 10, "max": 60},
            ],
        },
        {
            "id": "volume_filter", "name": "거래량 필터",
            "desc": "거래량이 평소보다 너무 적으면 진입을 막습니다. 사람이 적은 시장에서는 가격이 헛방으로 움직이기 쉽기 때문이에요. 거의 모든 전략에 도움이 됩니다.",
            "params": [
                {"key": "min_ratio", "label": "최소 배수", "type": "float", "default": 0.8, "min": 0.1, "max": 3.0, "step": 0.1},
                {"key": "period", "label": "평균 기간", "type": "int", "default": 20, "min": 5, "max": 100},
            ],
        },
        {
            "id": "volatility_filter", "name": "변동성 필터",
            "desc": "가격 변동폭이 너무 크거나 너무 작은 구간을 피합니다. 너무 조용하면 수익이 안 나고, 너무 난리면 손절이 자주 걸립니다. 적당한 변동성일 때만 진입해요.",
            "params": [
                {"key": "period", "label": "ATR 기간", "type": "int", "default": 14, "min": 5, "max": 50},
                {"key": "min_pct", "label": "최소 %", "type": "float", "default": 0.5, "min": 0.1, "max": 5.0, "step": 0.1},
                {"key": "max_pct", "label": "최대 %", "type": "float", "default": 5.0, "min": 1.0, "max": 20.0, "step": 0.5},
            ],
        },
        {
            "id": "rsi_range_filter", "name": "RSI 범위 필터",
            "desc": "RSI가 극단적인 값(너무 높거나 낮음)일 때 진입을 차단합니다. 이미 많이 오른 상태에서 추격 매수하거나, 폭락 중에 잡는 것을 막아줍니다.",
            "params": [
                {"key": "period", "label": "기간", "type": "int", "default": 14, "min": 5, "max": 50},
                {"key": "min_rsi", "label": "최소", "type": "int", "default": 25, "min": 0, "max": 50},
                {"key": "max_rsi", "label": "최대", "type": "int", "default": 75, "min": 50, "max": 100},
            ],
        },
        {
            "id": "time_filter", "name": "시간대 필터",
            "desc": "하루 중 특정 시간에만 거래를 허용합니다. 예: 아시아 장 시간(0~8 UTC)에만 거래. 1분~15분 같은 단타에서만 의미 있고, 1일봉에서는 효과 없어요.",
            "params": [
                {"key": "start_hour", "label": "시작 (UTC)", "type": "int", "default": 0, "min": 0, "max": 23},
                {"key": "end_hour", "label": "종료 (UTC)", "type": "int", "default": 24, "min": 1, "max": 24},
            ],
        },
        {
            "id": "consecutive_candle", "name": "연속봉 필터",
            "desc": "최근 N개 봉 중 같은 방향 봉이 M개 이상일 때만 진입합니다. 예: 3개 중 2개가 양봉이면 매수 허용. 갈팡질팡하는 구간에서 헛거래를 크게 줄여줘요.",
            "params": [
                {"key": "lookback", "label": "확인 봉수", "type": "int", "default": 3, "min": 2, "max": 10},
                {"key": "min_count", "label": "최소 같은방향", "type": "int", "default": 2, "min": 1, "max": 10},
            ],
        },
        {
            "id": "drawdown_breaker", "name": "낙폭 차단기",
            "desc": "자산이 고점에서 일정% 이상 떨어지면 모든 신규 진입을 차단합니다. 전략이 맞지 않는 장에서 연속 손실을 막아주는 안전장치예요. 5~10% 권장.",
            "params": [
                {"key": "max_dd", "label": "최대 낙폭 %", "type": "float", "default": 5.0, "min": 1.0, "max": 20.0, "step": 0.5},
            ],
        },
    ],
    "risk": [
        {
            "id": "atr_stop", "name": "ATR 손절 (추천)",
            "desc": "현재 시장의 변동폭(ATR)에 맞춰 손절 위치를 자동 조절합니다. 변동이 크면 넓게, 작으면 좁게. 시장 상황에 따라 알아서 적응하므로 가장 추천하는 손절 방식이에요.",
            "params": [
                {"key": "period", "label": "ATR 기간", "type": "int", "default": 14, "min": 5, "max": 50},
                {"key": "multiplier", "label": "ATR 배수", "type": "float", "default": 2.0, "min": 0.5, "max": 5.0, "step": 0.1},
            ],
        },
        {
            "id": "fixed_stop", "name": "고정 손절",
            "desc": "산 가격에서 정해진 %만큼 떨어지면 무조건 손절합니다. 예: 2%로 설정하면 100원에 사서 98원이 되면 자동으로 팝니다. 단순하고 이해하기 쉬워요.",
            "params": [
                {"key": "stop_pct", "label": "손절 %", "type": "float", "default": 2.0, "min": 0.1, "max": 20.0, "step": 0.1},
            ],
        },
        {
            "id": "take_profit", "name": "익절 (수익 확보)",
            "desc": "목표 수익에 도달하면 자동으로 팔아서 이익을 확정합니다. R:R 2.0이면 손절폭의 2배 올랐을 때 익절. 예: 손절 2%면 4% 오르면 수익 확보. 2.0 이상을 권장해요.",
            "params": [
                {"key": "rr_ratio", "label": "R:R 비율", "type": "float", "default": 2.0, "min": 0.5, "max": 10.0, "step": 0.1},
            ],
        },
        {
            "id": "trailing_stop", "name": "트레일링 스탑",
            "desc": "가격이 올라가면 손절선도 같이 따라 올라갑니다. 상승하는 동안은 계속 보유하다가, 일정 %만큼 되돌리면 그때 팝니다. 큰 추세에서 수익을 극대화할 수 있어요.",
            "params": [
                {"key": "activation_pct", "label": "활성화 %", "type": "float", "default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1},
                {"key": "trail_pct", "label": "추적 간격 %", "type": "float", "default": 0.5, "min": 0.1, "max": 5.0, "step": 0.1},
            ],
        },
        {
            "id": "max_hold", "name": "최대 보유 시간",
            "desc": "정해진 봉 수가 지나면 수익이든 손실이든 강제로 청산합니다. 예: 24로 설정하면 1시간봉 기준 24시간 후 자동 종료. 하나의 거래에 너무 오래 묶이는 것을 방지해요.",
            "params": [
                {"key": "bars", "label": "최대 봉수", "type": "int", "default": 24, "min": 1, "max": 500},
            ],
        },
        {
            "id": "breakeven_stop", "name": "손익분기 스탑",
            "desc": "수익이 일정% 이상 나면 손절선을 진입가로 올립니다. 이후 최소한 본전은 보장됩니다. 수동 트레이더들이 가장 많이 쓰는 기법이에요.",
            "params": [
                {"key": "trigger_pct", "label": "활성화 수익%", "type": "float", "default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1},
            ],
        },
    ],
    "sizing": [
        {
            "id": "fixed_risk", "name": "리스크 한도",
            "desc": "한 번 거래에서 최대 얼마를 잃을 수 있는지 정합니다. 1%면 자본 1만원 중 100원까지만 손해볼 수 있도록 투자 금액을 자동 계산합니다. 1~2%가 안전하고, 5% 이상은 매우 공격적이에요.",
            "params": [
                {"key": "risk_pct", "label": "리스크 %", "type": "float", "default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1},
            ],
        },
        {
            "id": "leverage", "name": "레버리지 (배율)",
            "desc": "투자금을 N배로 뻥튀기합니다. 3배면 수익도 3배, 하지만 손실도 3배! 10배 레버리지에 10% 하락하면 전액 청산됩니다. 초보자는 1~2배를 권장해요. 양날의 검이에요.",
            "params": [
                {"key": "leverage", "label": "배수", "type": "float", "default": 1.0, "min": 1.0, "max": 10.0, "step": 0.5},
            ],
        },
        {
            "id": "max_position", "name": "최대 투자 비율",
            "desc": "전체 자본 중 한 번에 최대 몇 %까지 투자할 수 있는지 제한합니다. 20%면 1만원 중 최대 2천원까지만 한 거래에 투입. 나머지는 안전하게 남겨둡니다. 올인 방지용이에요.",
            "params": [
                {"key": "max_pct", "label": "최대 %", "type": "float", "default": 20.0, "min": 5.0, "max": 100.0, "step": 5.0},
            ],
        },
        {
            "id": "direction", "name": "포지션 방향",
            "desc": "롱(매수)만 할지, 숏(매도)만 할지, 양방향 다 할지 선택합니다. 롱은 가격이 오를 때 수익, 숏은 내릴 때 수익입니다. 초보자는 '롱만'부터 시작하는 걸 추천해요.",
            "params": [
                {"key": "direction", "label": "방향", "type": "select", "default": "both",
                 "options": [
                     {"value": "both", "label": "양방향 (롱+숏)"},
                     {"value": "long", "label": "롱만 (매수만)"},
                     {"value": "short", "label": "숏만 (매도만)"},
                 ]},
            ],
        },
    ],
}


# ─── Signal Generation ───

def generate_signals(close: np.ndarray, high: np.ndarray, low: np.ndarray,
                     volume: np.ndarray, timestamps: np.ndarray,
                     components: list[dict]) -> np.ndarray:
    """Generate combined signal array: +1=long, -1=short, 0=none."""
    n = len(close)
    signals = np.zeros(n)
    filter_mask = np.ones(n, dtype=bool)

    signal_components = [c for c in components if c["category"] == "signals"]
    filter_components = [c for c in components if c["category"] == "filters"]

    for comp in filter_components:
        mask = _eval_filter(comp, close, high, low, volume, timestamps)
        filter_mask &= mask

    if not signal_components:
        return signals

    for comp in signal_components:
        sig = _eval_signal(comp, close, high, low, volume)
        signals = np.where(signals == 0, sig, signals)

    signals *= filter_mask

    sizing_components = [c for c in components if c["category"] == "sizing"]
    for comp in sizing_components:
        if comp["id"] == "direction":
            d = comp.get("params", {}).get("direction", "both")
            if d == "long":
                signals = np.where(signals == -1, 0, signals)
            elif d == "short":
                signals = np.where(signals == 1, 0, signals)

    return signals


def _eval_signal(comp: dict, close, high, low, volume) -> np.ndarray:
    n = len(close)
    sig = np.zeros(n)
    p = comp.get("params", {})
    cid = comp["id"]

    if cid == "ema_cross":
        fast_ema = ema(close, p.get("fast", 9))
        slow_ema = ema(close, p.get("slow", 21))
        for i in range(1, n):
            if np.isnan(fast_ema[i]) or np.isnan(slow_ema[i]):
                continue
            if fast_ema[i] > slow_ema[i] and fast_ema[i - 1] <= slow_ema[i - 1]:
                sig[i] = 1
            elif fast_ema[i] < slow_ema[i] and fast_ema[i - 1] >= slow_ema[i - 1]:
                sig[i] = -1

    elif cid == "rsi_threshold":
        rsi_vals = rsi(close, p.get("period", 14))
        oversold = p.get("oversold", 30)
        overbought = p.get("overbought", 70)
        for i in range(1, n):
            if np.isnan(rsi_vals[i]) or np.isnan(rsi_vals[i - 1]):
                continue
            if rsi_vals[i] > oversold and rsi_vals[i - 1] <= oversold:
                sig[i] = 1
            elif rsi_vals[i] < overbought and rsi_vals[i - 1] >= overbought:
                sig[i] = -1

    elif cid == "macd_cross":
        ml, sl, _ = macd(close, p.get("fast", 12), p.get("slow", 26), p.get("signal", 9))
        for i in range(1, n):
            if np.isnan(ml[i]) or np.isnan(sl[i]):
                continue
            if ml[i] > sl[i] and ml[i - 1] <= sl[i - 1]:
                sig[i] = 1
            elif ml[i] < sl[i] and ml[i - 1] >= sl[i - 1]:
                sig[i] = -1

    elif cid == "boll_bounce":
        upper, mid, lower, pct_b = bollinger(close, p.get("period", 20), p.get("std", 2.0))
        for i in range(1, n):
            if np.isnan(lower[i]):
                continue
            if close[i - 1] <= lower[i - 1] and close[i] > lower[i]:
                sig[i] = 1
            elif close[i - 1] >= upper[i - 1] and close[i] < upper[i]:
                sig[i] = -1

    elif cid == "boll_breakout":
        upper, mid, lower, pct_b = bollinger(close, p.get("period", 20), p.get("std", 2.0))
        for i in range(1, n):
            if np.isnan(upper[i]):
                continue
            if close[i] > upper[i] and close[i - 1] <= upper[i - 1]:
                sig[i] = 1
            elif close[i] < lower[i] and close[i - 1] >= lower[i - 1]:
                sig[i] = -1

    elif cid == "vwap_mean_reversion":
        vwap_vals = vwap(high, low, close, volume, p.get("period", 20))
        z = vwap_zscore(close, vwap_vals, p.get("period", 20))
        z_thr = p.get("z_threshold", 2.0)
        for i in range(n):
            if np.isnan(z[i]):
                continue
            if z[i] < -z_thr:
                sig[i] = 1
            elif z[i] > z_thr:
                sig[i] = -1

    elif cid == "price_breakout":
        lb = p.get("lookback", 20)
        for i in range(lb, n):
            hi = np.max(high[i - lb:i])
            lo = np.min(low[i - lb:i])
            if close[i] > hi:
                sig[i] = 1
            elif close[i] < lo:
                sig[i] = -1

    elif cid == "stoch_rsi_cross":
        k, d = stoch_rsi(close, p.get("rsi_period", 14), p.get("stoch_period", 14))
        oversold = p.get("oversold", 20)
        overbought = p.get("overbought", 80)
        for i in range(1, n):
            if np.isnan(k[i]):
                continue
            if k[i] > oversold and k[i - 1] <= oversold:
                sig[i] = 1
            elif k[i] < overbought and k[i - 1] >= overbought:
                sig[i] = -1

    elif cid == "volume_breakout":
        vratio = volume_ratio(volume, p.get("ma_period", 20))
        mult = p.get("vol_mult", 2.0)
        for i in range(1, n):
            if vratio[i] >= mult:
                sig[i] = 1 if close[i] > close[i - 1] else -1

    elif cid == "williams_r":
        period = p.get("period", 14)
        oversold = p.get("oversold", -80)
        overbought = p.get("overbought", -20)
        for i in range(period, n):
            hh = np.max(high[i - period + 1:i + 1])
            ll = np.min(low[i - period + 1:i + 1])
            if hh == ll:
                continue
            wr = (hh - close[i]) / (hh - ll) * -100
            if i > period:
                prev_hh = np.max(high[i - period:i])
                prev_ll = np.min(low[i - period:i])
                prev_wr = (prev_hh - close[i - 1]) / max(prev_hh - prev_ll, 1e-10) * -100
                if wr > oversold and prev_wr <= oversold:
                    sig[i] = 1
                elif wr < overbought and prev_wr >= overbought:
                    sig[i] = -1

    elif cid == "obv_divergence":
        lb = p.get("lookback", 14)
        obv_vals = obv(close, volume)
        for i in range(lb, n):
            price_trend = close[i] - close[i - lb]
            obv_trend = obv_vals[i] - obv_vals[i - lb]
            if price_trend > 0 and obv_trend < 0:
                sig[i] = -1
            elif price_trend < 0 and obv_trend > 0:
                sig[i] = 1

    elif cid == "cci_signal":
        period = p.get("period", 20)
        threshold = p.get("threshold", 100)
        tp_arr = (high + low + close) / 3
        for i in range(period, n):
            window = tp_arr[i - period + 1:i + 1]
            mean_tp = np.mean(window)
            mean_dev = np.mean(np.abs(window - mean_tp))
            cci = (tp_arr[i] - mean_tp) / max(0.015 * mean_dev, 1e-10) if mean_dev > 0 else 0
            if i > period:
                prev_w = tp_arr[i - period:i]
                prev_m = np.mean(prev_w)
                prev_md = np.mean(np.abs(prev_w - prev_m))
                prev_cci = (tp_arr[i - 1] - prev_m) / max(0.015 * prev_md, 1e-10) if prev_md > 0 else 0
                if cci > threshold and prev_cci <= threshold:
                    sig[i] = 1
                elif cci < -threshold and prev_cci >= -threshold:
                    sig[i] = -1

    return sig


def _eval_filter(comp: dict, close, high, low, volume, timestamps) -> np.ndarray:
    n = len(close)
    mask = np.ones(n, dtype=bool)
    p = comp.get("params", {})
    cid = comp["id"]

    if cid == "trend_filter":
        e = ema(close, p.get("period", 50))
        mask = ~np.isnan(e) & (close > e)

    elif cid == "adx_filter":
        adx_vals = adx(high, low, close, p.get("period", 14))
        mask = ~np.isnan(adx_vals) & (adx_vals >= p.get("threshold", 25))

    elif cid == "volume_filter":
        vratio = volume_ratio(volume, p.get("period", 20))
        mask = vratio >= p.get("min_ratio", 0.8)

    elif cid == "volatility_filter":
        atr_vals = atr(high, low, close, p.get("period", 14))
        atr_pct = np.where(close > 0, atr_vals / close * 100, 0)
        mask = (atr_pct >= p.get("min_pct", 0.5)) & (atr_pct <= p.get("max_pct", 5.0))

    elif cid == "rsi_range_filter":
        rsi_vals = rsi(close, p.get("period", 14))
        mask = ~np.isnan(rsi_vals) & (rsi_vals >= p.get("min_rsi", 25)) & (rsi_vals <= p.get("max_rsi", 75))

    elif cid == "time_filter":
        start_h = p.get("start_hour", 0)
        end_h = p.get("end_hour", 24)
        hours = np.array([(t // 3600000) % 24 for t in timestamps])
        if start_h < end_h:
            mask = (hours >= start_h) & (hours < end_h)
        else:
            mask = (hours >= start_h) | (hours < end_h)

    elif cid == "consecutive_candle":
        lb = p.get("lookback", 3)
        min_count = p.get("min_count", 2)
        for i in range(lb, n):
            green = sum(1 for j in range(max(1, i - lb), i) if close[j] > close[j - 1])
            red = lb - green
            mask[i] = green >= min_count or red >= min_count

    elif cid == "drawdown_breaker":
        pass

    return mask


# ─── Backtest Engine ───

@dataclass
class Trade:
    entry_idx: int
    entry_price: float
    side: int  # +1=long, -1=short
    size_usd: float
    stop_price: float
    tp_price: float
    exit_idx: int = 0
    exit_price: float = 0.0
    pnl_pct: float = 0.0
    pnl_usd: float = 0.0
    exit_reason: str = ""
    entry_time: int = 0
    exit_time: int = 0


@dataclass
class BacktestResult:
    trades: list[Trade] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)
    total_return_pct: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0
    avg_pnl_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    profit_factor: float = 0.0
    avg_hold_bars: float = 0.0
    initial_equity: float = 0.0
    final_equity: float = 0.0

    def to_dict(self) -> dict:
        return {
            "total_return_pct": round(self.total_return_pct, 2),
            "win_rate": round(self.win_rate, 2),
            "total_trades": self.total_trades,
            "avg_pnl_pct": round(self.avg_pnl_pct, 3),
            "max_drawdown_pct": round(self.max_drawdown_pct, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 3),
            "profit_factor": round(self.profit_factor, 2),
            "avg_hold_bars": round(self.avg_hold_bars, 1),
            "initial_equity": self.initial_equity,
            "final_equity": round(self.final_equity, 2),
            "equity_curve": [round(e, 2) for e in self.equity_curve[::max(1, len(self.equity_curve) // 500)]],
            "trades": [
                {
                    "entry_idx": t.entry_idx, "exit_idx": t.exit_idx,
                    "side": "Long" if t.side == 1 else "Short",
                    "entry_price": round(t.entry_price, 4),
                    "exit_price": round(t.exit_price, 4),
                    "pnl_pct": round(t.pnl_pct, 3),
                    "pnl_usd": round(t.pnl_usd, 2),
                    "exit_reason": t.exit_reason,
                    "entry_time": t.entry_time,
                    "exit_time": t.exit_time,
                }
                for t in self.trades[-200:]
            ],
        }


INTERVAL_ANNUAL = {
    "1m": 365 * 24 * 60, "5m": 365 * 24 * 12, "15m": 365 * 24 * 4,
    "1h": 365 * 24, "4h": 365 * 6, "1d": 365,
}

def run_backtest(close: np.ndarray, high: np.ndarray, low: np.ndarray,
                 volume: np.ndarray, timestamps: np.ndarray,
                 components: list[dict],
                 initial_equity: float = 10000.0,
                 fee_pct: float = 0.075,
                 interval: str = "1h") -> BacktestResult:
    n = len(close)
    signals = generate_signals(close, high, low, volume, timestamps, components)

    risk_comps = [c for c in components if c["category"] == "risk"]
    sizing_comps = [c for c in components if c["category"] == "sizing"]

    risk_pct = 1.0
    leverage = 1.0
    max_pos_pct = 100.0
    rr_ratio = 2.0
    max_hold = 9999
    use_trailing = False
    trail_activation = 1.0
    trail_pct = 0.5
    use_atr_stop = False
    atr_stop_mult = 2.0
    atr_stop_period = 14
    fixed_stop_pct = 2.0
    use_breakeven = False
    breakeven_trigger = 1.0

    for c in sizing_comps:
        p = c.get("params", {})
        if c["id"] == "fixed_risk":
            risk_pct = p.get("risk_pct", 1.0)
        elif c["id"] == "leverage":
            leverage = p.get("leverage", 1.0)
        elif c["id"] == "max_position":
            max_pos_pct = p.get("max_pct", 100.0)

    for c in risk_comps:
        p = c.get("params", {})
        if c["id"] == "take_profit":
            rr_ratio = p.get("rr_ratio", 2.0)
        elif c["id"] == "max_hold":
            max_hold = p.get("bars", 9999)
        elif c["id"] == "trailing_stop":
            use_trailing = True
            trail_activation = p.get("activation_pct", 1.0)
            trail_pct = p.get("trail_pct", 0.5)
        elif c["id"] == "atr_stop":
            use_atr_stop = True
            atr_stop_mult = p.get("multiplier", 2.0)
            atr_stop_period = p.get("period", 14)
        elif c["id"] == "fixed_stop":
            fixed_stop_pct = p.get("stop_pct", 2.0)
        elif c["id"] == "breakeven_stop":
            use_breakeven = True
            breakeven_trigger = p.get("trigger_pct", 1.0)

    dd_breaker_pct = 999.0
    for c in [cc for cc in components if cc.get("category") == "filters"]:
        if c["id"] == "drawdown_breaker":
            dd_breaker_pct = c.get("params", {}).get("max_dd", 5.0)

    atr_vals = atr(high, low, close, atr_stop_period) if use_atr_stop else None

    equity = initial_equity
    peak_equity = initial_equity
    equity_curve = [equity]
    trades: list[Trade] = []
    position: Trade | None = None

    for i in range(1, n):
        if position is not None:
            pnl_mult = position.side * (close[i] - position.entry_price) / position.entry_price
            current_pnl_pct = pnl_mult * 100 * leverage

            hit_stop = (position.side == 1 and low[i] <= position.stop_price) or \
                       (position.side == -1 and high[i] >= position.stop_price)
            hit_tp = (position.side == 1 and high[i] >= position.tp_price) or \
                     (position.side == -1 and low[i] <= position.tp_price)
            hit_max_hold = (i - position.entry_idx) >= max_hold
            opposite_signal = signals[i] != 0 and signals[i] != position.side

            if use_breakeven and not hit_stop:
                if pnl_mult * 100 >= breakeven_trigger:
                    if position.side == 1 and position.stop_price < position.entry_price:
                        position.stop_price = position.entry_price
                    elif position.side == -1 and position.stop_price > position.entry_price:
                        position.stop_price = position.entry_price

            if use_trailing and not hit_stop:
                if position.side == 1:
                    new_stop = high[i] * (1 - trail_pct / 100)
                    if pnl_mult * 100 >= trail_activation and new_stop > position.stop_price:
                        position.stop_price = new_stop
                        hit_stop = low[i] <= position.stop_price
                else:
                    new_stop = low[i] * (1 + trail_pct / 100)
                    if pnl_mult * 100 >= trail_activation and new_stop < position.stop_price:
                        position.stop_price = new_stop
                        hit_stop = high[i] >= position.stop_price

            if hit_stop or hit_tp or hit_max_hold or opposite_signal:
                if hit_stop:
                    exit_price = position.stop_price
                    reason = "손절"
                elif hit_tp:
                    exit_price = position.tp_price
                    reason = "익절"
                elif hit_max_hold:
                    exit_price = close[i]
                    reason = "시간초과"
                else:
                    exit_price = close[i]
                    reason = "반대 시그널"

                raw_pnl = position.side * (exit_price - position.entry_price) / position.entry_price
                net_pnl_pct = raw_pnl * 100 - fee_pct * 2
                pnl_usd = position.size_usd * net_pnl_pct / 100

                position.exit_idx = i
                position.exit_price = exit_price
                position.pnl_pct = net_pnl_pct * leverage
                position.pnl_usd = pnl_usd
                position.exit_reason = reason
                position.exit_time = int(timestamps[i]) if i < len(timestamps) else 0
                trades.append(position)
                equity += pnl_usd
                position = None

        if equity > peak_equity:
            peak_equity = equity
        current_dd = (peak_equity - equity) / peak_equity * 100 if peak_equity > 0 else 0

        if position is None and signals[i] != 0 and equity > 0 and current_dd < dd_breaker_pct:
            side = int(signals[i])
            if use_atr_stop and atr_vals is not None and not np.isnan(atr_vals[i]):
                stop_dist = atr_vals[i] * atr_stop_mult
            else:
                stop_dist = close[i] * fixed_stop_pct / 100

            stop_price = max(0.0001, close[i] - side * stop_dist)
            tp_dist = stop_dist * rr_ratio
            tp_price = max(0.0001, close[i] + side * tp_dist)

            risk_usd = equity * risk_pct / 100
            stop_pct = abs(stop_dist / close[i])
            if stop_pct > 0:
                size_usd = min(risk_usd / stop_pct, equity * max_pos_pct / 100) * leverage
            else:
                size_usd = equity * risk_pct / 100 * leverage

            position = Trade(
                entry_idx=i, entry_price=close[i], side=side,
                size_usd=size_usd, stop_price=stop_price, tp_price=tp_price,
                entry_time=int(timestamps[i]) if i < len(timestamps) else 0,
            )

        equity_curve.append(equity)

    if position is not None:
        raw_pnl = position.side * (close[-1] - position.entry_price) / position.entry_price
        net_pnl_pct = raw_pnl * 100 - fee_pct * 2
        pnl_usd = position.size_usd * net_pnl_pct / 100
        position.exit_idx = n - 1
        position.exit_price = close[-1]
        position.pnl_pct = net_pnl_pct * leverage
        position.pnl_usd = pnl_usd
        position.exit_reason = "종료"
        position.exit_time = int(timestamps[-1]) if len(timestamps) > 0 else 0
        trades.append(position)
        equity += pnl_usd
        equity_curve.append(equity)

    result = BacktestResult(
        trades=trades,
        equity_curve=equity_curve,
        initial_equity=initial_equity,
        final_equity=equity,
    )

    if trades:
        result.total_trades = len(trades)
        wins = [t for t in trades if t.pnl_pct > 0]
        losses = [t for t in trades if t.pnl_pct <= 0]
        result.win_rate = len(wins) / len(trades) * 100
        result.total_return_pct = (equity - initial_equity) / initial_equity * 100
        result.avg_pnl_pct = np.mean([t.pnl_pct for t in trades])
        result.avg_hold_bars = np.mean([t.exit_idx - t.entry_idx for t in trades])

        gross_profit = sum(t.pnl_usd for t in wins) if wins else 0
        gross_loss = abs(sum(t.pnl_usd for t in losses)) if losses else 1
        result.profit_factor = gross_profit / max(gross_loss, 0.01)

        peak = initial_equity
        max_dd = 0
        for e in equity_curve:
            if e > peak:
                peak = e
            dd = (peak - e) / peak * 100
            if dd > max_dd:
                max_dd = dd
        result.max_drawdown_pct = max_dd

        returns = np.diff(equity_curve) / np.maximum(np.array(equity_curve[:-1]), 1e-10)
        returns = returns[np.isfinite(returns)]
        if len(returns) > 1 and np.std(returns) > 0:
            ann_factor = math.sqrt(INTERVAL_ANNUAL.get(interval, 365 * 24))
            result.sharpe_ratio = np.mean(returns) / np.std(returns) * ann_factor
        else:
            result.sharpe_ratio = 0

    return result


# ─── Indicator data for charts ───

def compute_indicators(close, high, low, volume, components):
    """Compute indicator values for chart overlays."""
    indicators = {}

    for comp in components:
        p = comp.get("params", {})
        cid = comp["id"]

        if cid == "ema_cross":
            indicators[f"EMA({p.get('fast', 9)})"] = ema(close, p.get("fast", 9)).tolist()
            indicators[f"EMA({p.get('slow', 21)})"] = ema(close, p.get("slow", 21)).tolist()
        elif cid == "rsi_threshold":
            indicators["RSI"] = rsi(close, p.get("period", 14)).tolist()
        elif cid == "macd_cross":
            ml, sl, hist = macd(close, p.get("fast", 12), p.get("slow", 26), p.get("signal", 9))
            indicators["MACD"] = ml.tolist()
            indicators["MACD Signal"] = sl.tolist()
        elif cid in ("boll_bounce", "boll_breakout"):
            upper, mid, lower, _ = bollinger(close, p.get("period", 20), p.get("std", 2.0))
            indicators["BB Upper"] = upper.tolist()
            indicators["BB Mid"] = mid.tolist()
            indicators["BB Lower"] = lower.tolist()
        elif cid == "vwap_mean_reversion":
            v = vwap(high, low, close, volume, p.get("period", 20))
            indicators["VWAP"] = v.tolist()
        elif cid == "trend_filter":
            indicators[f"EMA({p.get('period', 50)})"] = ema(close, p.get("period", 50)).tolist()
        elif cid == "adx_filter":
            indicators["ADX"] = adx(high, low, close, p.get("period", 14)).tolist()
        elif cid == "atr_stop":
            indicators["ATR"] = atr(high, low, close, p.get("period", 14)).tolist()

    return {k: [None if (isinstance(v, float) and math.isnan(v)) else v for v in vals]
            for k, vals in indicators.items()}


# ─── Recommendations ───

def recommend_components(close, high, low, volume, timestamps):
    """Suggest components based on current market state."""
    recs = []

    rsi_now = rsi(close, 14)
    adx_now = adx(high, low, close, 14)
    vratio = volume_ratio(volume, 20)
    _, _, _, pct_b = bollinger(close, 20, 2.0)

    last_rsi = rsi_now[-1] if not np.isnan(rsi_now[-1]) else 50
    last_adx = adx_now[-1] if not np.isnan(adx_now[-1]) else 20
    last_vratio = vratio[-1] if not np.isnan(vratio[-1]) else 1.0
    last_pctb = pct_b[-1] if not np.isnan(pct_b[-1]) else 0.5

    trending = last_adx > 25
    ranging = last_adx < 20
    high_volume = last_vratio > 1.5
    oversold = last_rsi < 35
    overbought = last_rsi > 65

    if trending and high_volume:
        recs.append({
            "name": "추세 추종 조합",
            "reason": f"ADX={last_adx:.0f}(강한 추세) + 거래량 {last_vratio:.1f}배(활발)",
            "components": [
                {"id": "ema_cross", "category": "signals", "params": {"fast": 9, "slow": 21}},
                {"id": "adx_filter", "category": "filters", "params": {"period": 14, "threshold": 22}},
                {"id": "volume_filter", "category": "filters", "params": {"min_ratio": 1.0, "period": 20}},
                {"id": "atr_stop", "category": "risk", "params": {"period": 14, "multiplier": 2.0}},
                {"id": "take_profit", "category": "risk", "params": {"rr_ratio": 2.5}},
                {"id": "fixed_risk", "category": "sizing", "params": {"risk_pct": 1.0}},
            ],
        })
    elif trending:
        recs.append({
            "name": "MACD 추세 조합",
            "reason": f"ADX={last_adx:.0f}(추세 확인) — 크로스 시그널로 진입",
            "components": [
                {"id": "macd_cross", "category": "signals", "params": {"fast": 12, "slow": 26, "signal": 9}},
                {"id": "trend_filter", "category": "filters", "params": {"period": 50}},
                {"id": "atr_stop", "category": "risk", "params": {"period": 14, "multiplier": 1.5}},
                {"id": "take_profit", "category": "risk", "params": {"rr_ratio": 2.0}},
                {"id": "fixed_risk", "category": "sizing", "params": {"risk_pct": 1.0}},
            ],
        })

    if ranging:
        recs.append({
            "name": "횡보장 역추세 조합",
            "reason": f"ADX={last_adx:.0f}(횡보) — 볼린저 밴드 반등 전략",
            "components": [
                {"id": "boll_bounce", "category": "signals", "params": {"period": 20, "std": 2.0}},
                {"id": "rsi_range_filter", "category": "filters", "params": {"period": 14, "min_rsi": 20, "max_rsi": 80}},
                {"id": "fixed_stop", "category": "risk", "params": {"stop_pct": 1.5}},
                {"id": "take_profit", "category": "risk", "params": {"rr_ratio": 1.5}},
                {"id": "fixed_risk", "category": "sizing", "params": {"risk_pct": 0.5}},
            ],
        })

    if ranging and last_vratio < 1.0:
        recs.append({
            "name": "VWAP 평균회귀 조합",
            "reason": f"ADX={last_adx:.0f}(횡보) + 거래량 {last_vratio:.1f}배(조용) — VWAP 이탈 복귀",
            "components": [
                {"id": "vwap_mean_reversion", "category": "signals", "params": {"period": 20, "z_threshold": 2.0}},
                {"id": "volume_filter", "category": "filters", "params": {"min_ratio": 0.5, "period": 20}},
                {"id": "fixed_stop", "category": "risk", "params": {"stop_pct": 1.0}},
                {"id": "take_profit", "category": "risk", "params": {"rr_ratio": 1.5}},
                {"id": "fixed_risk", "category": "sizing", "params": {"risk_pct": 0.5}},
            ],
        })

    if oversold:
        recs.append({
            "name": "과매도 반등 조합",
            "reason": f"RSI={last_rsi:.0f}(과매도 구간) — 반등 시 진입",
            "components": [
                {"id": "rsi_threshold", "category": "signals", "params": {"period": 14, "oversold": 30, "overbought": 70}},
                {"id": "volume_filter", "category": "filters", "params": {"min_ratio": 0.8, "period": 20}},
                {"id": "atr_stop", "category": "risk", "params": {"period": 14, "multiplier": 2.0}},
                {"id": "take_profit", "category": "risk", "params": {"rr_ratio": 2.0}},
                {"id": "fixed_risk", "category": "sizing", "params": {"risk_pct": 1.0}},
            ],
        })

    if high_volume:
        recs.append({
            "name": "거래량 돌파 조합",
            "reason": f"거래량 {last_vratio:.1f}배(폭증) — 방향성 있는 돌파",
            "components": [
                {"id": "volume_breakout", "category": "signals", "params": {"vol_mult": 2.0, "ma_period": 20}},
                {"id": "trend_filter", "category": "filters", "params": {"period": 50}},
                {"id": "atr_stop", "category": "risk", "params": {"period": 14, "multiplier": 1.5}},
                {"id": "take_profit", "category": "risk", "params": {"rr_ratio": 3.0}},
                {"id": "trailing_stop", "category": "risk", "params": {"activation_pct": 1.5, "trail_pct": 0.8}},
                {"id": "fixed_risk", "category": "sizing", "params": {"risk_pct": 1.0}},
            ],
        })

    if not recs:
        recs.append({
            "name": "기본 EMA 크로스 조합",
            "reason": "현재 시장 상태에 맞는 특별한 추천이 없어 기본 전략을 추천합니다",
            "components": [
                {"id": "ema_cross", "category": "signals", "params": {"fast": 9, "slow": 21}},
                {"id": "atr_stop", "category": "risk", "params": {"period": 14, "multiplier": 2.0}},
                {"id": "take_profit", "category": "risk", "params": {"rr_ratio": 2.0}},
                {"id": "fixed_risk", "category": "sizing", "params": {"risk_pct": 1.0}},
            ],
        })

    return recs


# ─── Live Signal Evaluation ───

def evaluate_live_signal(close: np.ndarray, high: np.ndarray, low: np.ndarray,
                         volume: np.ndarray, timestamps: np.ndarray,
                         components: list[dict]) -> dict:
    """Evaluate strategy components against current data and return signal status."""
    n = len(close)
    if n < 2:
        return {"signal": "neutral", "signal_text": "데이터 부족", "checks": []}

    signal_components = [c for c in components if c["category"] == "signals"]
    filter_components = [c for c in components if c["category"] == "filters"]
    sizing_components = [c for c in components if c["category"] == "sizing"]

    checks = []
    final_signal = 0

    for comp in signal_components:
        sig = _eval_signal(comp, close, high, low, volume)
        last_sig = int(sig[-1]) if not np.isnan(sig[-1]) else 0
        recent_sig = 0
        for j in range(min(3, n), 0, -1):
            if sig[-j] != 0:
                recent_sig = int(sig[-j])
                break

        comp_name = _get_component_name(comp["id"])
        detail = _get_signal_detail(comp, close, high, low, volume)

        if last_sig == 1:
            checks.append({"name": comp_name, "type": "signal", "status": "buy", "icon": "BUY", "detail": detail})
            if final_signal == 0:
                final_signal = 1
        elif last_sig == -1:
            checks.append({"name": comp_name, "type": "signal", "status": "sell", "icon": "SELL", "detail": detail})
            if final_signal == 0:
                final_signal = -1
        elif recent_sig != 0:
            label = "매수 대기" if recent_sig == 1 else "매도 대기"
            checks.append({"name": comp_name, "type": "signal", "status": "recent", "icon": label, "detail": detail})
        else:
            checks.append({"name": comp_name, "type": "signal", "status": "neutral", "icon": "대기", "detail": detail})

    for comp in filter_components:
        if comp["id"] == "drawdown_breaker":
            checks.append({"name": "낙폭 차단기", "type": "filter", "status": "pass", "icon": "PASS",
                           "detail": "실시간에서는 낙폭 차단 미적용"})
            continue

        mask = _eval_filter(comp, close, high, low, volume, timestamps)
        passed = bool(mask[-1])
        comp_name = _get_component_name(comp["id"])
        detail = _get_filter_detail(comp, close, high, low, volume, timestamps)

        checks.append({
            "name": comp_name, "type": "filter",
            "status": "pass" if passed else "fail",
            "icon": "PASS" if passed else "FAIL",
            "detail": detail,
        })
        if not passed:
            final_signal = 0

    direction = "both"
    for comp in sizing_components:
        if comp["id"] == "direction":
            direction = comp.get("params", {}).get("direction", "both")

    if direction == "long" and final_signal == -1:
        final_signal = 0
    elif direction == "short" and final_signal == 1:
        final_signal = 0

    if final_signal == 1:
        signal_text = "매수 신호"
        signal_status = "buy"
    elif final_signal == -1:
        signal_text = "매도 신호"
        signal_status = "sell"
    else:
        all_signals_neutral = all(c["status"] in ("neutral", "recent") for c in checks if c["type"] == "signal")
        any_filter_fail = any(c["status"] == "fail" for c in checks if c["type"] == "filter")
        if any_filter_fail:
            signal_text = "필터 미충족"
            signal_status = "filtered"
        elif all_signals_neutral:
            signal_text = "신호 대기중"
            signal_status = "neutral"
        else:
            signal_text = "조건 불충분"
            signal_status = "neutral"

    return {
        "signal": signal_status,
        "signal_text": signal_text,
        "direction": final_signal,
        "checks": checks,
        "price": round(float(close[-1]), 4),
        "timestamp": int(timestamps[-1]) if len(timestamps) > 0 else 0,
    }


def _get_component_name(cid: str) -> str:
    name_map = {
        "ema_cross": "EMA 크로스", "rsi_threshold": "RSI 과매수/과매도",
        "macd_cross": "MACD 크로스", "boll_bounce": "볼린저 반등",
        "boll_breakout": "볼린저 돌파", "vwap_mean_reversion": "VWAP 평균회귀",
        "price_breakout": "가격 돌파", "stoch_rsi_cross": "Stoch RSI",
        "volume_breakout": "거래량 돌파", "williams_r": "Williams %R",
        "obv_divergence": "OBV 다이버전스", "cci_signal": "CCI",
        "trend_filter": "추세 필터", "adx_filter": "ADX 필터",
        "volume_filter": "거래량 필터", "volatility_filter": "변동성 필터",
        "rsi_range_filter": "RSI 범위", "time_filter": "시간대 필터",
        "consecutive_candle": "연속봉", "drawdown_breaker": "낙폭 차단기",
    }
    return name_map.get(cid, cid)


def _get_signal_detail(comp: dict, close, high, low, volume) -> str:
    p = comp.get("params", {})
    cid = comp["id"]

    if cid == "ema_cross":
        fast_ema = ema(close, p.get("fast", 9))
        slow_ema = ema(close, p.get("slow", 21))
        if not np.isnan(fast_ema[-1]) and not np.isnan(slow_ema[-1]):
            diff = (fast_ema[-1] - slow_ema[-1]) / close[-1] * 100
            return f"EMA{p.get('fast',9)}={fast_ema[-1]:.1f} vs EMA{p.get('slow',21)}={slow_ema[-1]:.1f} (차이: {diff:+.3f}%)"
    elif cid == "rsi_threshold":
        rsi_vals = rsi(close, p.get("period", 14))
        if not np.isnan(rsi_vals[-1]):
            return f"RSI={rsi_vals[-1]:.1f} (과매도:{p.get('oversold',30)} / 과매수:{p.get('overbought',70)})"
    elif cid == "macd_cross":
        ml, sl, hist = macd(close, p.get("fast", 12), p.get("slow", 26), p.get("signal", 9))
        if not np.isnan(ml[-1]) and not np.isnan(sl[-1]):
            return f"MACD={ml[-1]:.2f} Signal={sl[-1]:.2f} Hist={ml[-1]-sl[-1]:.2f}"
    elif cid == "boll_bounce" or cid == "boll_breakout":
        upper, mid, lower, pct_b = bollinger(close, p.get("period", 20), p.get("std", 2.0))
        if not np.isnan(pct_b[-1]):
            return f"%B={pct_b[-1]:.3f} (하단:0 중앙:0.5 상단:1.0)"
    elif cid == "volume_breakout":
        vratio = volume_ratio(volume, p.get("ma_period", 20))
        return f"거래량비={vratio[-1]:.2f}x (기준:{p.get('vol_mult',2.0)}x)"
    elif cid == "williams_r":
        period = p.get("period", 14)
        if len(close) > period:
            hh = np.max(high[-period:])
            ll = np.min(low[-period:])
            wr = (hh - close[-1]) / max(hh - ll, 1e-10) * -100
            return f"W%R={wr:.1f} (과매도:{p.get('oversold',-80)} / 과매수:{p.get('overbought',-20)})"
    elif cid == "stoch_rsi_cross":
        k, d = stoch_rsi(close, p.get("rsi_period", 14), p.get("stoch_period", 14))
        if not np.isnan(k[-1]):
            return f"StochRSI K={k[-1]:.1f} (과매도:{p.get('oversold',20)} / 과매수:{p.get('overbought',80)})"
    elif cid == "cci_signal":
        tp_arr = (high + low + close) / 3
        period = p.get("period", 20)
        if len(tp_arr) >= period:
            window = tp_arr[-period:]
            mean_tp = np.mean(window)
            mean_dev = np.mean(np.abs(window - mean_tp))
            cci = (tp_arr[-1] - mean_tp) / max(0.015 * mean_dev, 1e-10) if mean_dev > 0 else 0
            return f"CCI={cci:.1f} (기준:±{p.get('threshold',100)})"
    elif cid == "obv_divergence":
        lb = p.get("lookback", 14)
        obv_vals = obv(close, volume)
        price_trend = close[-1] - close[-min(lb, len(close))]
        obv_trend = obv_vals[-1] - obv_vals[-min(lb, len(obv_vals))]
        return f"가격추세={'↑' if price_trend > 0 else '↓'} OBV추세={'↑' if obv_trend > 0 else '↓'}"
    elif cid == "vwap_mean_reversion":
        vwap_vals = vwap(high, low, close, volume, p.get("period", 20))
        z = vwap_zscore(close, vwap_vals, p.get("period", 20))
        if not np.isnan(z[-1]):
            return f"Z-Score={z[-1]:.2f} (기준:±{p.get('z_threshold',2.0)})"
    elif cid == "price_breakout":
        lb = p.get("lookback", 20)
        if len(close) > lb:
            hi = np.max(high[-lb-1:-1])
            lo = np.min(low[-lb-1:-1])
            return f"현재={close[-1]:.1f} 고점={hi:.1f} 저점={lo:.1f}"

    return ""


def _get_filter_detail(comp: dict, close, high, low, volume, timestamps) -> str:
    p = comp.get("params", {})
    cid = comp["id"]

    if cid == "trend_filter":
        e = ema(close, p.get("period", 50))
        if not np.isnan(e[-1]):
            above = close[-1] > e[-1]
            return f"가격={close[-1]:.1f} {'>' if above else '<'} EMA{p.get('period',50)}={e[-1]:.1f}"
    elif cid == "adx_filter":
        adx_vals = adx(high, low, close, p.get("period", 14))
        if not np.isnan(adx_vals[-1]):
            return f"ADX={adx_vals[-1]:.1f} (기준:{p.get('threshold',25)})"
    elif cid == "volume_filter":
        vratio = volume_ratio(volume, p.get("period", 20))
        return f"거래량비={vratio[-1]:.2f}x (최소:{p.get('min_ratio',0.8)}x)"
    elif cid == "volatility_filter":
        atr_vals = atr(high, low, close, p.get("period", 14))
        if not np.isnan(atr_vals[-1]) and close[-1] > 0:
            atr_pct = atr_vals[-1] / close[-1] * 100
            return f"ATR%={atr_pct:.3f} (범위:{p.get('min_pct',0.5)}~{p.get('max_pct',5.0)}%)"
    elif cid == "rsi_range_filter":
        rsi_vals = rsi(close, p.get("period", 14))
        if not np.isnan(rsi_vals[-1]):
            return f"RSI={rsi_vals[-1]:.1f} (범위:{p.get('min_rsi',25)}~{p.get('max_rsi',75)})"
    elif cid == "time_filter":
        hour = (timestamps[-1] // 3600000) % 24 if len(timestamps) > 0 else 0
        return f"현재 UTC {hour}시 (허용:{p.get('start_hour',0)}~{p.get('end_hour',24)}시)"
    elif cid == "consecutive_candle":
        lb = p.get("lookback", 3)
        if len(close) > lb:
            green = sum(1 for j in range(-lb, 0) if close[j] > close[j - 1])
            return f"양봉 {green}/{lb}개 (최소:{p.get('min_count',2)}개)"

    return ""
