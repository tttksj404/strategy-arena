"""wave59b — 현재 배치와 같은 국면에서 과거에 무슨 일이 있었는가.

sweep59 결과: 현재 단기(ma50)는 강세, 장기(ma200)는 약세 = '하락장 반등' 배치.
이 사분면 조건에서 과거 롱/숏 성과를 재면 "과거를 보고 판단"의 직접적 답이 된다.

사분면:
  bull_bull  ma50 강세 & ma200 강세  = 확립된 상승장
  bull_bear  ma50 강세 & ma200 약세  = 하락장 반등  <- 2026-07 현재
  bear_bull  ma50 약세 & ma200 강세  = 상승장 눌림
  bear_bear  ma50 약세 & ma200 약세  = 확립된 하락장

    $V research/wave59_regime/quadrant59.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.wave30_qd.dataio30 import SYMBOLS, build_market_cache  # noqa: E402

CACHE = Path(__file__).resolve().parent / "cache"
CAPITAL = 100.0
ROUND_TRIP = 2 * 0.0006
WIN, LOSS = 1, -1
DAY = 24

QUADRANTS = {
    "bull_bull": (+1, +1, "확립된 상승장"),
    "bull_bear": (+1, -1, "하락장 반등  <- 현재"),
    "bear_bull": (-1, +1, "상승장 눌림"),
    "bear_bear": (-1, -1, "확립된 하락장"),
}


def state(cache):
    out = {}
    for symbol in SYMBOLS:
        close = pd.Series(cache.arrays[symbol].close)
        ma50 = close.rolling(50 * DAY, min_periods=50 * DAY).mean()
        ma200 = close.rolling(200 * DAY, min_periods=200 * DAY).mean()
        out[symbol] = (
            np.sign(close - ma50).fillna(0).to_numpy(),
            np.sign(close - ma200).fillna(0).to_numpy(),
        )
    return out


def measure(data, states, quad, side, gain, lev, stop_frac, years=None):
    fee = CAPITAL * lev * ROUND_TRIP
    payoff = gain - fee
    loss = 97.5 if stop_frac >= 0.90 else CAPITAL * stop_frac + fee
    want50, want200, _ = QUADRANTS[quad]
    wins = losses = 0
    for symbol in SYMBOLS:
        entries = data[f"{symbol}_entries"]
        outcomes = data[f"{symbol}_long" if side > 0 else f"{symbol}_short"]
        s50, s200 = states[symbol]
        mask = (s50[entries] == want50) & (s200[entries] == want200)
        if years is not None:
            mask &= np.isin(data["year"][entries], years)
        wins += int((outcomes[mask] == WIN).sum())
        losses += int((outcomes[mask] == LOSS).sum())
    n = wins + losses
    if n < 100:
        return None
    p = wins / n
    return {"p": p, "ev": p * payoff - (1 - p) * loss, "n": n,
            "breakeven": loss / (payoff + loss)}


def main() -> None:
    cache = build_market_cache()
    states = state(cache)
    index = pd.DatetimeIndex(cache.index)
    name, gain, lev, stop_frac = "g30_l5_s90", 30.0, 5.0, 0.90
    data = np.load(CACHE / f"outcomes_{name}.npz")
    fee = CAPITAL * lev * ROUND_TRIP
    be = 97.5 / (gain - fee + 97.5)

    print(f"=== 목표 ${gain:.0f} · {lev:.0f}x · 청산손절 · 손익분기 승률 {be:.1%} ===\n")
    print(f"{'사분면':>11} {'설명':>20} {'롱승률':>7} {'롱EV':>8} "
          f"{'숏승률':>7} {'숏EV':>8} {'표본':>7}")
    for quad, (_, _, desc) in QUADRANTS.items():
        lo = measure(data, states, quad, +1, gain, lev, stop_frac)
        sh = measure(data, states, quad, -1, gain, lev, stop_frac)
        if lo is None:
            print(f"{quad:>11} {desc:>20}  표본 부족")
            continue
        print(f"{quad:>11} {desc:>20} {lo['p']:6.1%} ${lo['ev']:+7.2f} "
              f"{sh['p']:6.1%} ${sh['ev']:+7.2f} {lo['n']:7d}")

    print(f"\n=== 'bull_bear'(현재 배치)를 연도별로 — 표본 우연인지 확인 ===")
    print(f"{'연도':>6} {'롱승률':>7} {'롱EV':>8} {'숏승률':>7} {'숏EV':>8} {'표본':>7}")
    rows = []
    for y in range(2020, 2027):
        lo = measure(data, states, "bull_bear", +1, gain, lev, stop_frac, [y])
        sh = measure(data, states, "bull_bear", -1, gain, lev, stop_frac, [y])
        if lo is None:
            continue
        rows.append((y, lo["ev"], sh["ev"]))
        print(f"{y:6d} {lo['p']:6.1%} ${lo['ev']:+7.2f} "
              f"{sh['p']:6.1%} ${sh['ev']:+7.2f} {lo['n']:7d}")
    if len(rows) >= 3:
        for label, col in (("롱", 1), ("숏", 2)):
            v = np.array([r[col] for r in rows])
            t = v.mean() / (v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else np.nan
            print(f"  {label}: 평균 ${v.mean():+.2f} · 양수 {(v > 0).sum()}/{len(v)}년 "
                  f"· t={t:+.2f}")

    print(f"\n=== 현재 상태 확인 ({index[-1]:%Y-%m-%d}) ===")
    for symbol in SYMBOLS:
        s50, s200 = states[symbol]
        f = lambda v: "강세" if v > 0 else ("약세" if v < 0 else "-")
        print(f"  {symbol:>9}: ma50 {f(s50[-1])} · ma200 {f(s200[-1])}")
    votes = [(states[s][0][-1], states[s][1][-1]) for s in SYMBOLS]
    quad_now = ("bull" if sum(v[0] for v in votes) > 0 else "bear") + "_" + \
               ("bull" if sum(v[1] for v in votes) > 0 else "bear")
    print(f"  => 현재 사분면: {quad_now} ({QUADRANTS[quad_now][2]})")


if __name__ == "__main__":
    main()



# ---------------------------------------------------------------------------
# wave59c — 사분면 결과에서 나온 가설의 정식 검정
#
#   관찰: ma50과 ma200이 **일치**하는 두 사분면에서 롱 EV가 양수다
#         (bull_bull +$2.60 · bear_bear +$3.85)
#         갈리는 두 사분면에서는 죽는다 (bull_bear 롱 -$8.23/숏 -$2.21)
#
#   가설: "두 이평이 일치할 때만 롱, 갈리면 진입 안 함"이 엣지다
#   검정: 연도별 EV의 t통계량 + 상승연/하락연 일관성
# ---------------------------------------------------------------------------


def measure_agreement(data, states, agree: bool, side: int,
                      gain, lev, stop_frac, years=None):
    """agree=True면 ma50·ma200이 같은 방향인 구간만."""
    fee = CAPITAL * lev * ROUND_TRIP
    payoff = gain - fee
    loss = 97.5 if stop_frac >= 0.90 else CAPITAL * stop_frac + fee
    wins = losses = 0
    for symbol in SYMBOLS:
        entries = data[f"{symbol}_entries"]
        outcomes = data[f"{symbol}_long" if side > 0 else f"{symbol}_short"]
        s50, s200 = states[symbol]
        a, b = s50[entries], s200[entries]
        valid = (a != 0) & (b != 0)
        mask = valid & ((a == b) if agree else (a != b))
        if years is not None:
            mask &= np.isin(data["year"][entries], years)
        wins += int((outcomes[mask] == WIN).sum())
        losses += int((outcomes[mask] == LOSS).sum())
    n = wins + losses
    if n < 100:
        return None
    p = wins / n
    return {"p": p, "ev": p * payoff - (1 - p) * loss, "n": n}


def test_agreement_hypothesis() -> None:
    cache = build_market_cache()
    states = state(cache)
    index = pd.DatetimeIndex(cache.index)
    btc = cache.arrays["BTCUSDT"].close
    name, gain, lev, stop_frac = "g30_l5_s90", 30.0, 5.0, 0.90
    data = np.load(CACHE / f"outcomes_{name}.npz")

    print("\n" + "=" * 72)
    print("=== wave59c 가설 검정: '두 이평 일치 시 롱' ===\n")
    for agree in (True, False):
        label = "일치(ma50=ma200)" if agree else "불일치(갈림)"
        lo = measure_agreement(data, states, agree, +1, gain, lev, stop_frac)
        sh = measure_agreement(data, states, agree, -1, gain, lev, stop_frac)
        print(f"{label:>20}: 롱 {lo['p']:5.1%} ${lo['ev']:+7.2f} | "
              f"숏 {sh['p']:5.1%} ${sh['ev']:+7.2f} | 표본 {lo['n']}")

    print(f"\n--- '일치 시 롱'을 연도별로 (미래정보 없음) ---")
    print(f"{'연도':>6} {'BTC':>8} {'승률':>7} {'EV':>8} {'표본':>7}")
    rows = []
    for y in range(2020, 2027):
        pos = np.flatnonzero(index.year == y)
        br = btc[pos[-1]] / btc[pos[0]] - 1.0
        r = measure_agreement(data, states, True, +1, gain, lev, stop_frac, [y])
        if r is None:
            continue
        rows.append((br, r["ev"]))
        print(f"{y:6d} {br:+7.1%} {r['p']:6.1%} ${r['ev']:+7.2f} {r['n']:7d}")
    rets = np.array([r[0] for r in rows])
    evs = np.array([r[1] for r in rows])
    t = evs.mean() / (evs.std(ddof=1) / np.sqrt(len(evs)))
    up, dn = evs[rets > 0.2], evs[rets < 0]
    print(f"\n  전체 ${evs.mean():+.2f} · 양수 {(evs > 0).sum()}/{len(evs)}년 · t={t:+.2f}")
    print(f"  상승연 ${up.mean():+.2f} · 하락연 ${dn.mean():+.2f}")
    verdict = (up.mean() > 0 and dn.mean() > 0 and abs(t) > 2.4)
    print(f"  판정: {'엣지 확정' if verdict else '기준 미달 — 표본 우연과 구별 안 됨'}")
    print(f"        (필요: 상승연>0 · 하락연>0 · |t|>2.4)")


if __name__ == "__main__":
    test_agreement_hypothesis()
