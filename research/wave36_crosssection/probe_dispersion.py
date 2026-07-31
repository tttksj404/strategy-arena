#!/usr/bin/env python3
# Wave-36 opportunity probe: is there a CROSS-SECTIONAL funding edge, before building anything?
#
# ---------------------------------------------------------------------------------------
# Why this direction, and why it is genuinely new here
# ---------------------------------------------------------------------------------------
# Every wave from 30 to 35 searched the same space: a rule that decides WHICH WAY PRICE WILL GO.
# Six search algorithms, quality-diversity archives, genetic programming and reinforcement learning
# all failed out-of-sample on it, and the RL agent's learned answer was literally "do not trade".
# Meanwhile the one family that ever passed every gate -- funding carry -- does not predict price at
# all; it collects a payment that exists for structural reasons.
#
# So the untried move is not another optimiser. It is to drop directional prediction entirely and
# harvest the CROSS-SECTIONAL dispersion of funding: short the perpetuals paying the most, long the
# ones paying the least (or receiving), in equal notional so the book is roughly market-neutral.
# Both legs then collect funding in our favour, and price direction cancels.
#
# This was IMPOSSIBLE before wave35. With three symbols there is no cross-section to rank. The
# 20-symbol universe collected in wave35 is what unlocks it -- the data expansion enabled a
# structurally different strategy class, not just a wider version of the old one.
#
# ---------------------------------------------------------------------------------------
# What this probe decides
# ---------------------------------------------------------------------------------------
# Nothing is built until the arithmetic clears, the same discipline that killed 1m timeframes in
# wave35 before a single backtest ran. The question is whether the funding spread between the
# top-k and bottom-k symbols exceeds what it costs to hold and rotate that book. If the median
# spread is smaller than the cost of maintaining it, the idea is dead and no search can revive it.

from __future__ import annotations

from pathlib import Path
import sys
from typing import Final

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

import numpy as np
import pandas as pd  # noqa: PANDAS_OK

from research.wave35_universe.dataio35 import build_wide_cache

FUNDING_PER_DAY: Final = 3  # 8h stamps
LOOKBACKS_DAYS: Final = (1, 3, 7, 14, 21, 30, 45, 60)
TOP_K: Final = (1, 2, 3, 5)
# Net APR improved monotonically with lookback in the first pass (1d -38.8% -> 14d +6.8%) because
# turnover cost, not gross spread, is the binding term: a longer averaging window changes the ranking
# less often. The ladder is extended to find where that trade-off turns, and a hold-band variant is
# added because the cheapest way to cut turnover is to not rotate on marginal rank changes at all.
HOLD_BANDS: Final = (0.0, 0.25, 0.50)  # keep an existing position unless it falls out of the top/bottom (k + band*n)


def funding_panel(cache) -> pd.DataFrame:
    """Realised funding rate per symbol at each 8h stamp (rows = stamps, cols = symbols)."""
    frames = {}
    for symbol in cache.symbols:
        series = pd.Series(cache.arrays[symbol].funding_at_bar, index=cache.index)
        frames[symbol] = series[series != 0.0]
    panel = pd.DataFrame(frames)
    return panel.dropna(how="all")


def main() -> int:
    cache, symbols = build_wide_cache()
    panel = funding_panel(cache)
    cost_rates = {s: cache.arrays[s].cost_rate for s in symbols}
    mean_cost = float(np.mean(list(cost_rates.values())))

    print(f"횡단면 펀딩 패널: {panel.shape[0]:,} 스탬프 x {panel.shape[1]}종목 "
          f"({panel.index[0].date()} ~ {panel.index[-1].date()})")
    print(f"평균 편도비용 {mean_cost*1e4:.3f}bp — 2레그 왕복 = {4*mean_cost*1e4:.1f}bp\n")

    apr = panel * FUNDING_PER_DAY * 365
    print("=== (A) 횡단면 펀딩 분산 자체 (스탬프별 종목간 APR) ===")
    spread_all = (apr.max(axis=1) - apr.min(axis=1)).dropna()
    print(f"  최고−최저 APR: 중앙 {spread_all.median():7.2%} · p25 {spread_all.quantile(.25):7.2%} "
          f"· p75 {spread_all.quantile(.75):7.2%}")
    print(f"  전 종목 APR 중앙값 {apr.stack().median():+7.2%} | 음수 펀딩 비율 {(panel.stack()<0).mean():6.2%}")

    print("\n=== (B) 신호(과거 N일 평균 펀딩) → 다음 스탬프 실현 스프레드 ===")
    print("    롱=하위k(가장 적게 내는 쪽) / 숏=상위k(가장 많이 내는 쪽), 달러 중립")
    print(f"{'룩백':>5} {'k':>3} {'밴드':>5} {'실현 APR(중앙)':>18} {'평균':>10} {'양수비율':>9} "
          f"{'회전율':>12} {'회전비용APR':>12} {'순APR':>12}")
    print("-" * 110)
    best = None
    for lookback_days in LOOKBACKS_DAYS:
        window = lookback_days * FUNDING_PER_DAY
        signal = panel.rolling(window, min_periods=window).mean().shift(1)  # 직전 스탬프까지만 사용
        for k in TOP_K:
          for band in HOLD_BANDS:
            realised, turnovers = [], []
            held_long: list[str] = []
            held_short: list[str] = []
            previous_book: set[str] = set()  # per-combination, never shared across the grid
            for stamp in panel.index:
                row = signal.loc[stamp].dropna()
                if len(row) < 2 * k + 2:
                    continue
                ordered = row.sort_values()
                keep_width = int(round(k + band * len(ordered)))
                long_pool = set(ordered.index[:keep_width])
                short_pool = set(ordered.index[-keep_width:])
                # Hysteresis: an existing leg is kept while it stays inside the wider pool; only
                # then are free slots filled from the strict top/bottom k.
                held_long = [s for s in held_long if s in long_pool]
                held_short = [s for s in held_short if s in short_pool]
                for symbol in ordered.index:
                    if len(held_long) >= k:
                        break
                    if symbol not in held_long and symbol not in held_short:
                        held_long.append(symbol)
                for symbol in reversed(ordered.index):
                    if len(held_short) >= k:
                        break
                    if symbol not in held_short and symbol not in held_long:
                        held_short.append(symbol)
                actual = panel.loc[stamp]
                received = float(actual[held_short].mean() - actual[held_long].mean())
                realised.append(received)
                book = set(held_long) | set(held_short)
                turnovers.append(len(book - previous_book) / max(len(book), 1))
                previous_book = book
            if len(realised) < 100:
                continue
            series = pd.Series(realised)
            realised_apr = series * FUNDING_PER_DAY * 365
            turnover = float(np.mean(turnovers))
            # each replaced position costs a round trip on 1 leg; annualised over stamps
            cost_apr = turnover * 2 * mean_cost * FUNDING_PER_DAY * 365
            net = float(realised_apr.median() - cost_apr)
            if net > 0 or band == 0.0:
                print(f"{lookback_days:4d}d {k:3d} {band:5.2f} {realised_apr.median():18.2%} "
                      f"{realised_apr.mean():9.2%} {(series>0).mean():8.2%} {turnover:12.2%} "
                      f"{cost_apr:12.2%} {net:12.2%}")
            if best is None or net > best[0]:
                best = (net, lookback_days, k, float(realised_apr.median()), turnover, cost_apr, band)

    print()
    if best and best[0] > 0:
        net, lookback_days, k, gross, turnover, cost_apr, band = best
        print(f"[결과] 최선 조합: 룩백 {lookback_days}일 · k={k} · 유지밴드 {band:.2f}")
        print(f"  총 펀딩 수취 {gross:.2%} APR − 회전비용 {cost_apr:.2%} APR = **순 {net:.2%} APR**")
        print(f"  회전율 {turnover:.1%}/스탬프 (하루 {turnover*FUNDING_PER_DAY:.2f}회 교체)")
        print("  → 비용을 넘는다. 구조적 엣지가 존재하므로 wave36 본실험 진행 정당.")
    else:
        print("[결과] 어떤 조합도 비용을 넘지 못한다 → 이 아이디어는 산술 단계에서 기각.")
    print("\n주의: 이 프로브는 펀딩 수취만 센다. 가격 손익(스프레드 변동)·슬리피지 확대·")
    print("      최소주문·레버리지 제약은 본실험에서 엔진으로 계산해야 한다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
