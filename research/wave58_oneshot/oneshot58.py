"""wave58 — 단발 도박수의 기대값 측정.

$100 전액을 레버리지 L로 한 번에 걸고, 목표 도달 시 익절 / 청산선 도달 시 전액 손실.
무편향 확률(추세 없는 랜덤워크의 배리어 도달 확률)과 실측 승률을 비교해
초과 승률이 진짜 엣지인지 방향 베타인지 판정한다.

핵심 결과: 숏의 초과 승률은 BTC 연수익과 상관 -0.970 — 방향 베타이며 엣지가 아니다.

    $V research/wave58_oneshot/oneshot58.py
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
STEP = 6           # 진입 간격(봉)
MAX_HOLD = 24 * 60  # 최대 보유(봉) = 60일 @1H


def fair_probability(target: float, stop: float) -> float:
    """추세 없는 랜덤워크가 손절보다 목표에 먼저 닿을 확률."""
    return stop / (target + stop)


def measure_barrier(
    cache,
    target: float,
    stop: float,
    mask: np.ndarray,
    direction: int,
    max_hold: int = MAX_HOLD,
) -> tuple[float, int]:
    """배리어 도달 승률을 실측한다. direction=+1 롱, -1 숏."""
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
            if direction > 0:
                take, kill = price * (1 + target), price * (1 - stop)
            else:
                take, kill = price * (1 - target), price * (1 + stop)
            for bar in range(entry + 1, min(entry + max_hold, n)):
                if direction > 0:
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
    return (wins / resolved if resolved else float("nan")), resolved


def expected_value(win_prob: float, gain: float, leverage: float) -> float:
    """이기면 목표에서 비용을 뺀 금액, 지면 자본 전액 손실."""
    cost = CAPITAL * leverage * ROUND_TRIP
    return win_prob * (gain - cost) - (1 - win_prob) * CAPITAL


def main() -> None:
    cache = build_market_cache()
    index = pd.DatetimeIndex(cache.index)
    btc = cache.arrays["BTCUSDT"].close

    leverage, gain = 5.0, 50.0
    stop = 0.9 / leverage
    target = gain / (CAPITAL * leverage)
    fair = fair_probability(target, stop)

    print(f"=== {leverage:.0f}x · 목표 ${gain:.0f}(가격 {target:.0%}) "
          f"· 청산 {stop:.0%} · 무편향 {fair:.1%} ===\n")
    print(f"{'연도':>6} {'BTC수익':>9} {'롱승률':>8} {'숏승률':>8} {'롱EV':>9} {'숏EV':>9}")

    rows = []
    for year in range(2019, 2027):
        mask = np.asarray(index.year == year)
        if mask.sum() < 500:
            continue
        pos = np.flatnonzero(mask)
        btc_return = btc[pos[-1]] / btc[pos[0]] - 1.0
        p_long, _ = measure_barrier(cache, target, stop, mask, +1)
        p_short, _ = measure_barrier(cache, target, stop, mask, -1)
        ev_long = expected_value(p_long, gain, leverage)
        ev_short = expected_value(p_short, gain, leverage)
        rows.append((year, btc_return, p_long, p_short, ev_long, ev_short))
        print(f"{year:6d} {btc_return:+8.1%} {p_long:7.1%} {p_short:7.1%} "
              f"${ev_long:+8.2f} ${ev_short:+8.2f}")

    btc_returns = np.array([r[1] for r in rows])
    shorts = np.array([r[3] for r in rows])
    longs = np.array([r[2] for r in rows])

    print("\n=== 판정: 승률이 BTC 방향에 종속되는가 ===")
    print(f"  BTC연수익 vs 숏승률 상관: {np.corrcoef(btc_returns, shorts)[0, 1]:+.3f}")
    print(f"  BTC연수익 vs 롱승률 상관: {np.corrcoef(btc_returns, longs)[0, 1]:+.3f}")

    up = [r for r in rows if r[1] > 0.20]
    down = [r for r in rows if r[1] < 0]
    up_ev = float(np.mean([r[5] for r in up]))
    down_ev = float(np.mean([r[5] for r in down]))
    print(f"\n  상승연도({len(up)}개, BTC>+20%): 숏승률 {np.mean([r[3] for r in up]):.1%} "
          f"· 숏EV ${up_ev:+.2f}")
    print(f"  하락연도({len(down)}개, BTC<0):    숏승률 {np.mean([r[3] for r in down]):.1%} "
          f"· 숏EV ${down_ev:+.2f}")
    verdict = "방향 베타, 예측력 없음" if up_ev < 0 else "진짜 엣지"
    print(f"\n  => 숏 기대값이 상승연도에 뒤집힌다면 방향 베타: {verdict}")

    print("\n=== 엣지 없이(방향 예측 없이) 공정 도박의 기대값 ===")
    print(f"{'구조':>18} {'승률':>7} {'이기면':>9} {'기대값':>9}")
    for lev, gn in ((5.0, 30.0), (5.0, 50.0), (10.0, 50.0), (20.0, 100.0)):
        st = 0.9 / lev
        tg = gn / (CAPITAL * lev)
        if tg >= st * 0.95:
            continue
        fp = fair_probability(tg, st)
        cost = CAPITAL * lev * ROUND_TRIP
        print(f"{lev:14.0f}x ${gn:3.0f} {fp:6.1%} ${gn - cost:+8.2f} "
              f"${expected_value(fp, gn, lev):+8.2f}")


if __name__ == "__main__":
    main()
