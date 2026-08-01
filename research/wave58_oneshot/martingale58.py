"""wave58b — 단발 승부에서 구조적 엣지를 만들 수 있는가.

사용자 반박("예전에 도박수로 수익 났었다")에 따른 재검정. 세 가지를 측정한다:

1. 추세 필터(200일선/50일선/모멘텀30)를 단발 승부에 적용하면 베타를 제거할 수 있는가
2. 청산 전에 손절해 '청산 벌금'(가격 함의 90% vs 실제 97.5%)을 피할 수 있는가
3. 그렇게 얻은 개선이 연도 표본에서 유의한가

핵심 결과: 7개 손절 위치 전부에서 실측 승률이 손익분기 승률에 0.1%p 이내로 붙는다.
배리어 도달은 정확히 마팅게일이며 구조에서 공짜 엣지는 나오지 않는다.

    $V research/wave58_oneshot/martingale58.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.wave30_qd.dataio30 import SYMBOLS, build_market_cache  # noqa: E402

TAKER = 0.0006
ROUND_TRIP = 2 * TAKER
CAPITAL = 100.0
STEP = 6
MAX_HOLD = 24 * 60
GAIN = 30.0
LEVERAGE = 5.0
LIQUIDATION_LOSS = 97.5   # 청산 시 실제 손실(유지증거금 $2.5 잔존)
STOP_LIQUIDATES = 0.90    # 이 비율 이상은 청산으로 취급


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(values).rolling(window, min_periods=window).mean().to_numpy()


def build_trends(cache) -> dict:
    trends = {}
    for symbol in SYMBOLS:
        close = cache.arrays[symbol].close
        lag = 30 * 24
        momentum = np.concatenate(
            [np.full(lag, np.nan), close[lag:] / close[:-lag] - 1.0]
        )
        trends[symbol] = {
            "ma200": rolling_mean(close, 200 * 24),
            "ma50": rolling_mean(close, 50 * 24),
            "mom30": momentum,
        }
    return trends


def direction(cache, trends, rule: str, symbol: str, i: int) -> int:
    """+1 롱, -1 숏, 0 진입 보류."""
    if rule == "long":
        return 1
    signal = trends[symbol][rule][i]
    if not np.isfinite(signal):
        return 0
    if rule == "mom30":
        return 1 if signal > 0 else -1
    return 1 if cache.arrays[symbol].close[i] > signal else -1


def simulate(cache, trends, mask, rule: str, stop_frac: float) -> dict:
    """배리어 승률과 기대값을 실측한다."""
    target_pct = GAIN / (CAPITAL * LEVERAGE)
    stop_pct = stop_frac / LEVERAGE
    wins = losses = 0
    for symbol in SYMBOLS:
        arr = cache.arrays[symbol]
        high, low, close = arr.high, arr.low, arr.close
        n = len(close)
        for entry in range(0, n - 1, STEP):
            if not mask[entry]:
                continue
            price = close[entry]
            if not np.isfinite(price) or price <= 0:
                continue
            side = direction(cache, trends, rule, symbol, entry)
            if side == 0:
                continue
            if side > 0:
                take, kill = price * (1 + target_pct), price * (1 - stop_pct)
            else:
                take, kill = price * (1 - target_pct), price * (1 + stop_pct)
            for bar in range(entry + 1, min(entry + MAX_HOLD, n)):
                if side > 0:
                    if low[bar] <= kill:
                        losses += 1
                        break
                    if high[bar] >= take:
                        wins += 1
                        break
                else:
                    if high[bar] >= kill:
                        losses += 1
                        break
                    if low[bar] <= take:
                        wins += 1
                        break
    resolved = wins + losses
    win_prob = wins / resolved if resolved else float("nan")
    fee = CAPITAL * LEVERAGE * ROUND_TRIP
    if stop_frac >= STOP_LIQUIDATES:
        loss = LIQUIDATION_LOSS
    else:
        loss = CAPITAL * stop_frac + fee
    payoff = GAIN - fee
    return {
        "win_prob": win_prob,
        "expected_value": win_prob * payoff - (1 - win_prob) * loss,
        "breakeven": loss / (payoff + loss),
        "martingale": stop_frac / (GAIN / CAPITAL + stop_frac),
        "resolved": resolved,
    }


def main() -> None:
    cache = build_market_cache()
    index = pd.DatetimeIndex(cache.index)
    trends = build_trends(cache)
    btc = cache.arrays["BTCUSDT"].close
    rules = ("long", "ma200", "ma50", "mom30")

    print(f"=== 목표 ${GAIN:.0f} · {LEVERAGE:.0f}x · 추세 필터별 연도 EV ===\n")
    header = " | ".join(f"{r:>14}" for r in rules)
    print(f"{'연도':>6} {'BTC':>8} | {header}")

    per_rule: dict[str, list] = {r: [] for r in rules}
    for year in range(2020, 2027):
        mask = np.asarray(index.year == year)
        if mask.sum() < 500:
            continue
        pos = np.flatnonzero(mask)
        btc_ret = btc[pos[-1]] / btc[pos[0]] - 1.0
        cells = []
        for rule in rules:
            res = simulate(cache, trends, mask, rule, STOP_LIQUIDATES)
            per_rule[rule].append((btc_ret, res["expected_value"]))
            cells.append(f"{res['win_prob']:5.1%} ${res['expected_value']:+7.2f}")
        joined = " | ".join(f"{c:>14}" for c in cells)
        print(f"{year:6d} {btc_ret:+7.1%} | {joined}")

    print(f"\n=== 레짐 일관성 ===")
    print(f"{'규칙':>10} {'상승연EV':>10} {'하락연EV':>10} {'전체':>9} {'BTC상관':>9}")
    for rule in rules:
        data = per_rule[rule]
        rets = np.array([d[0] for d in data])
        evs = np.array([d[1] for d in data])
        print(f"{rule:>10} ${evs[rets > 0.2].mean():+9.2f} ${evs[rets < 0].mean():+9.2f} "
              f"${evs.mean():+8.2f} {np.corrcoef(rets, evs)[0, 1]:+8.3f}")

    print(f"\n=== 손절 위치를 바꿔 '청산 벌금'을 피할 수 있는가 (200일선) ===")
    print(f"{'손절':>6} {'무편향p':>8} {'분기p':>7} {'실측p':>7} {'초과':>7} {'EV':>9} {'표본':>7}")
    everything = np.ones(len(index), dtype=bool)
    sweep = []
    for stop_frac in (0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90):
        res = simulate(cache, trends, everything, "ma200", stop_frac)
        excess = res["win_prob"] - res["breakeven"]
        sweep.append((stop_frac, res["expected_value"]))
        print(f"{stop_frac:5.0%} {res['martingale']:8.1%} {res['breakeven']:7.1%} "
              f"{res['win_prob']:7.1%} {excess:+6.1f}p "
              f"${res['expected_value']:+8.2f} {res['resolved']:7d}")
    print("\n  => 초과가 모든 설정에서 -0.0p면 배리어 도달은 정확히 마팅게일이다")

    best_stop = max(sweep, key=lambda x: x[1])[0]
    print(f"\n=== 최적 손절 {best_stop:.0%}의 연도별 유의성 ===")
    evs = []
    for year in range(2020, 2027):
        mask = np.asarray(index.year == year)
        if mask.sum() < 500:
            continue
        res = simulate(cache, trends, mask, "ma200", best_stop)
        evs.append(res["expected_value"])
    evs = np.array(evs)
    t_stat = evs.mean() / (evs.std(ddof=1) / np.sqrt(len(evs)))
    print(f"  전체 ${evs.mean():+.2f} · 양수 {(evs > 0).sum()}/{len(evs)}년 · t = {t_stat:+.2f}")
    print(f"  (7년 표본에서 유의하려면 |t| > 2.4)")


if __name__ == "__main__":
    main()
