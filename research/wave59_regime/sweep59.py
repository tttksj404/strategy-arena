"""wave59 — 레짐 규칙 대량 탐색 + 현재 시점 판정.

precompute59가 저장한 배리어 결과에 레짐 필터를 불리언으로 씌워 수백 개 규칙을 훑는다.
모든 신호는 **과거 데이터만** 사용한다(rolling, min_periods 강제) — 미래 정보 없음.

판정 기준(둘 다 통과해야 엣지):
  1. 상승연·하락연 EV가 모두 양수 (레짐 종속이 아님)
  2. 연도별 EV의 t통계량이 유의 (표본 우연이 아님)

마지막에 **각 규칙이 현재 무엇을 말하는지** 출력한다.

    $V research/wave59_regime/sweep59.py
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
TAKER = 0.0006
ROUND_TRIP = 2 * TAKER
WIN, LOSS, OPEN = 1, -1, 0
DAY = 24  # 1시간봉


def payoff_and_loss(gain: float, lev: float, stop_frac: float) -> tuple[float, float]:
    fee = CAPITAL * lev * ROUND_TRIP
    loss = 97.5 if stop_frac >= 0.90 else CAPITAL * stop_frac + fee
    return gain - fee, loss


def build_signals(cache) -> dict:
    """과거만 쓰는 레짐 신호. +1 강세, -1 약세, 0 판정불가."""
    signals: dict[str, dict[str, np.ndarray]] = {}
    for symbol in SYMBOLS:
        close = pd.Series(cache.arrays[symbol].close)
        s: dict[str, np.ndarray] = {}
        for w in (20, 50, 100, 200, 300):
            ma = close.rolling(w * DAY, min_periods=w * DAY).mean()
            s[f"ma{w}"] = np.sign(close - ma).fillna(0).to_numpy()
        ma50 = close.rolling(50 * DAY, min_periods=50 * DAY).mean()
        ma200 = close.rolling(200 * DAY, min_periods=200 * DAY).mean()
        s["cross50_200"] = np.sign(ma50 - ma200).fillna(0).to_numpy()
        for d in (7, 14, 30, 90, 180):
            s[f"mom{d}"] = np.sign(
                close.pct_change(d * DAY)
            ).fillna(0).to_numpy()
        high365 = close.rolling(365 * DAY, min_periods=200 * DAY).max()
        dd = close / high365 - 1.0
        for thr in (0.10, 0.20, 0.30):
            # 고점 대비 -thr 이내면 강세, 더 깊으면 약세
            v = np.where(dd.isna(), 0.0, np.where(dd > -thr, 1.0, -1.0))
            s[f"dd{int(thr * 100)}"] = v
        vol = close.pct_change().rolling(30 * DAY, min_periods=30 * DAY).std()
        volmed = vol.rolling(365 * DAY, min_periods=200 * DAY).median()
        # 저변동이 강세
        s["lowvol"] = np.where(
            vol.isna() | volmed.isna(), 0.0, np.where(vol < volmed, 1.0, -1.0)
        )
        signals[symbol] = s
    return signals


def evaluate(data, signals, rule: str, mode: str, gain, lev, stop_frac,
             years=None) -> dict:
    """mode: both(강세롱/약세숏) · long_only · short_only"""
    payoff, loss = payoff_and_loss(gain, lev, stop_frac)
    wins = losses = 0
    for symbol in SYMBOLS:
        entries = data[f"{symbol}_entries"]
        longs = data[f"{symbol}_long"]
        shorts = data[f"{symbol}_short"]
        sig = signals[symbol][rule][entries]
        yr = data["year"][entries]
        keep = np.ones(len(entries), dtype=bool) if years is None else np.isin(yr, years)
        bull = keep & (sig > 0)
        bear = keep & (sig < 0)
        if mode in ("both", "long_only"):
            wins += int((longs[bull] == WIN).sum())
            losses += int((longs[bull] == LOSS).sum())
        if mode in ("both", "short_only"):
            wins += int((shorts[bear] == WIN).sum())
            losses += int((shorts[bear] == LOSS).sum())
    resolved = wins + losses
    if resolved < 200:
        return {"win_prob": float("nan"), "expected_value": float("nan"),
                "resolved": resolved}
    p = wins / resolved
    return {"win_prob": p, "expected_value": p * payoff - (1 - p) * loss,
            "resolved": resolved}


def main() -> None:
    cache = build_market_cache()
    signals = build_signals(cache)
    btc = cache.arrays["BTCUSDT"].close
    index = pd.DatetimeIndex(cache.index)

    years = list(range(2020, 2027))
    btc_ret = {}
    for y in years:
        pos = np.flatnonzero(index.year == y)
        btc_ret[y] = btc[pos[-1]] / btc[pos[0]] - 1.0

    config = ("g30_l5_s90", 30.0, 5.0, 0.90)
    name, gain, lev, stop_frac = config
    data = np.load(CACHE / f"outcomes_{name}.npz")
    rules = list(signals[SYMBOLS[0]].keys())
    modes = ("both", "long_only", "short_only")

    print(f"=== 목표 ${gain:.0f} · {lev:.0f}x · 손절 {stop_frac:.0%} "
          f"(이기면 ${payoff_and_loss(gain, lev, stop_frac)[0]:+.2f} / "
          f"지면 ${-payoff_and_loss(gain, lev, stop_frac)[1]:+.2f}) ===")
    be = payoff_and_loss(gain, lev, stop_frac)[1] / (
        sum(payoff_and_loss(gain, lev, stop_frac)))
    print(f"손익분기 승률 {be:.1%} · 규칙 {len(rules)}개 × 모드 {len(modes)}개 "
          f"= {len(rules) * len(modes)}개 조합\n")

    results = []
    for rule in rules:
        for mode in modes:
            per_year = []
            for y in years:
                r = evaluate(data, signals, rule, mode, gain, lev, stop_frac, [y])
                if np.isfinite(r["expected_value"]):
                    per_year.append((btc_ret[y], r["expected_value"]))
            if len(per_year) < 5:
                continue
            rets = np.array([x[0] for x in per_year])
            evs = np.array([x[1] for x in per_year])
            up = evs[rets > 0.2]
            dn = evs[rets < 0]
            if len(up) == 0 or len(dn) == 0:
                continue
            t = evs.mean() / (evs.std(ddof=1) / np.sqrt(len(evs)))
            full = evaluate(data, signals, rule, mode, gain, lev, stop_frac)
            results.append({
                "rule": rule, "mode": mode, "full_ev": full["expected_value"],
                "win": full["win_prob"], "n": full["resolved"],
                "up": up.mean(), "dn": dn.mean(), "t": t,
                "pos_years": int((evs > 0).sum()), "years": len(evs),
            })

    results.sort(key=lambda r: -min(r["up"], r["dn"]))
    print("=== 상위 15개: 상승연·하락연 EV 중 나쁜 쪽이 가장 좋은 순 ===")
    print(f"{'규칙':>12} {'모드':>11} {'승률':>7} {'전체EV':>8} "
          f"{'상승연':>8} {'하락연':>8} {'t':>6} {'양수년':>7} {'표본':>7}")
    for r in results[:15]:
        print(f"{r['rule']:>12} {r['mode']:>11} {r['win']:6.1%} "
              f"${r['full_ev']:+7.2f} ${r['up']:+7.2f} ${r['dn']:+7.2f} "
              f"{r['t']:+5.2f} {r['pos_years']}/{r['years']:<5d} {r['n']:7d}")

    passed = [r for r in results if r["up"] > 0 and r["dn"] > 0 and abs(r["t"]) > 2.4]
    print(f"\n=== 판정 기준 통과(상승연>0 · 하락연>0 · |t|>2.4): "
          f"{len(passed)}개 / {len(results)}개 ===")
    for r in passed:
        print(f"  통과: {r['rule']} {r['mode']} 전체EV ${r['full_ev']:+.2f} t={r['t']:+.2f}")
    if not passed:
        best = max(results, key=lambda r: min(r["up"], r["dn"]))
        print(f"  없음. 최선은 {best['rule']}/{best['mode']}: "
              f"상승연 ${best['up']:+.2f} 하락연 ${best['dn']:+.2f} t={best['t']:+.2f}")

    print(f"\n=== 현재 시점({index[-1]:%Y-%m-%d}) 각 신호의 판정 ===")
    print(f"{'규칙':>12} " + " ".join(f"{s.replace('USDT', ''):>8}" for s in SYMBOLS)
          + "   합의")
    for rule in rules:
        vals = [signals[s][rule][-1] for s in SYMBOLS]
        labels = ["강세" if v > 0 else ("약세" if v < 0 else "-") for v in vals]
        total = sum(vals)
        verdict = "롱" if total > 0 else ("숏" if total < 0 else "중립")
        print(f"{rule:>12} " + " ".join(f"{x:>8}" for x in labels) + f"   {verdict}")

    all_now = [signals[s][r][-1] for s in SYMBOLS for r in rules]
    bull_n = sum(1 for v in all_now if v > 0)
    bear_n = sum(1 for v in all_now if v < 0)
    print(f"\n  전체 신호 집계: 강세 {bull_n} · 약세 {bear_n} "
          f"(총 {len(all_now)}) → {'롱' if bull_n > bear_n else '숏'} 우세")


if __name__ == "__main__":
    main()
