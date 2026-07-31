#!/usr/bin/env python3
# Wave-38 step 1: measure the premise BEFORE building anything.
#
# wave18 named the biggest untapped resource precisely: L4 holds a position on only 59% of days and
# sits idle the other 41%, and even on active days it deploys just leg_fraction=0.50 of capital into
# top_k=1 symbol. I5 filled the idle days with USDT lending and gained +0.90%p (9.37% -> 10.27%).
#
# There is a different, untested lever on the SAME verified family. I3 already tried lowering the
# quality bar (threshold 15% -> 8% APR across the full universe) and LOST: CAGR 8.72%, below the
# 9.37% baseline, because low-funding carry does not cover its own turnover cost. But top_k > 1 is
# not that intervention. It keeps the 15% bar exactly where it is and simply takes MORE of the
# opportunities that already qualify. Those are different in kind: one degrades the signal, the other
# harvests more of the same signal.
#
# Whether that lever has anything to pull on is an empirical question with a cheap answer, and this
# probe answers it before any engine is written. The decisive statistic is the per-day COUNT of
# symbols already clearing 15% APR:
#   - If the median count is 1, top_k>1 is worthless and wave38 should stop here.
#   - If several symbols routinely clear 15%, top_k=1 is leaving measured, qualifying carry on the
#     table and the lever is real.
#
# This probe deliberately does NOT backtest. It only counts opportunities, so it cannot be confused
# for a performance claim and cannot consume a holdout.

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Final

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

import numpy as np
import pandas as pd

from research.wave1.fam_funding import funding_score

REPO_ROOT: Final = Path(__file__).resolve().parents[2]
CACHE: Final = REPO_ROOT / "research" / "wave3" / "cache"
RESULTS_DIR: Final = Path(__file__).resolve().parent / "results"

L4_THRESHOLD: Final = 0.15  # L4/I5 entry bar, unchanged
L4_EXIT: Final = 0.075  # carry_position's own threshold/2 exit
I3_THRESHOLD: Final = 0.08  # the bar I3 lowered to, and lost with
WINDOW_DAYS: Final = 7  # shared by every wave18 layer
MIN_HISTORY_DAYS: Final = 365


def load_funding_apr_panel() -> pd.DataFrame:
    """Per-symbol 7d funding APR, daily, for every symbol with spot+perp+funding all present.

    Carry needs all three legs to exist, so the same triple-file requirement fam_funding.load_markets
    applies is applied here; counting an opportunity on a symbol with no spot leg would overstate the
    lever. The APR convention is funding_score's own (rolling mean of the 8h rate x 3 x 365), so a
    count produced here means the same thing L4's own threshold means.
    """
    columns: dict[str, pd.Series] = {}
    for funding_path in sorted(CACHE.glob("binance_funding_*.csv.gz")):
        symbol = funding_path.name[len("binance_funding_") : -len(".csv.gz")]
        if not (CACHE / f"binance_spot_{symbol}_1d.csv.gz").exists():
            continue
        if not (CACHE / f"binance_fapi_{symbol}_1d.csv.gz").exists():
            continue
        frame = pd.read_csv(funding_path)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, format="ISO8601")
        series = frame.set_index("timestamp")["funding_rate"].astype(float)
        series = series[~series.index.duplicated(keep="first")].sort_index()
        if len(series) < WINDOW_DAYS * 3 + MIN_HISTORY_DAYS:
            continue
        apr = funding_score(series, WINDOW_DAYS).resample("1D").last()
        columns[symbol] = apr
    panel = pd.DataFrame(columns).sort_index()
    return panel


def main() -> int:
    print("=== wave38 step1: top_k>1 이 당길 것이 있는가 (백테스트 아님, 기회 개수만) ===\n")
    panel = load_funding_apr_panel()
    print(f"현물+선물+펀딩 3종 모두 존재하고 이력 충분한 종목: {panel.shape[1]}개")
    print(f"기간: {panel.index[0].date()} ~ {panel.index[-1].date()} ({len(panel)}일)\n")

    # Count only days where the universe itself is meaningfully populated, so an early-history day
    # with 2 listed symbols cannot masquerade as "no opportunity".
    listed = panel.notna().sum(axis=1)
    usable = panel.loc[listed >= 10]
    print(f"종목 10개 이상 상장된 날: {len(usable)}일 (이 구간만 집계)\n")

    above_15 = (usable > L4_THRESHOLD).sum(axis=1)
    above_08 = (usable > I3_THRESHOLD).sum(axis=1)
    best = usable.max(axis=1)

    print("=== 하루에 15% APR 기준을 넘는 종목이 몇 개인가 ===")
    print(f"  중앙값 {above_15.median():.0f}개 · 평균 {above_15.mean():.2f}개 · 최대 {above_15.max():.0f}개")
    quantiles = [0.10, 0.25, 0.50, 0.75, 0.90]
    labels = " · ".join(f"p{int(q*100)} {above_15.quantile(q):.0f}" for q in quantiles)
    print(f"  분위: {labels}")
    print(f"  0개인 날 {float((above_15 == 0).mean()):.1%} · 1개 이상 {float((above_15 >= 1).mean()):.1%} "
          f"· 2개 이상 {float((above_15 >= 2).mean()):.1%} · 3개 이상 {float((above_15 >= 3).mean()):.1%} "
          f"· 5개 이상 {float((above_15 >= 5).mean()):.1%}")

    print("\n=== 참고: 8% 기준 (I3가 낮춰서 실패한 그 기준) ===")
    print(f"  중앙값 {above_08.median():.0f}개 · 1개 이상인 날 {float((above_08 >= 1).mean()):.1%}")

    print("\n=== 그날 최고 펀딩 APR (기회의 품질) ===")
    print(f"  중앙 {best.median():.2%} · p25 {best.quantile(0.25):.2%} · p75 {best.quantile(0.75):.2%}")
    print(f"  15% 미만인 날 {float((best < L4_THRESHOLD).mean()):.1%} (= L4가 대기하는 근본 이유)")

    # top_k=1 harvests only the single best qualifying symbol. What does the campaign leave behind?
    # Sum the APR of qualifying symbols ranked 2..k to size the unharvested carry directly.
    print("\n=== top_k=1 이 남기고 있는 것 (2위 이하 자격 종목들의 펀딩 APR) ===")
    for k in (2, 3, 5):
        captured, left = [], []
        for _, row in usable.iterrows():
            qualifying = row[row > L4_THRESHOLD].dropna().sort_values(ascending=False)
            if len(qualifying) == 0:
                continue
            captured.append(float(qualifying.iloc[0]))
            left.append(float(qualifying.iloc[1:k].sum()))
        if captured:
            print(f"  top_k={k}: 1위 평균 {np.mean(captured):.2%} · 2~{k}위 합 평균 {np.mean(left):.2%} "
                  f"(1위 대비 {np.mean(left)/np.mean(captured):.0%} 추가)")

    verdict_multi = float((above_15 >= 2).mean())
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "wave": "wave38_breadth_probe",
        "question": "does top_k>1 have anything to pull on at the unchanged 15% APR bar?",
        "symbols": int(panel.shape[1]),
        "days_counted": int(len(usable)),
        "threshold_apr": L4_THRESHOLD,
        "count_above_threshold": {
            "median": float(above_15.median()),
            "mean": float(above_15.mean()),
            "max": float(above_15.max()),
            "share_zero": float((above_15 == 0).mean()),
            "share_ge_1": float((above_15 >= 1).mean()),
            "share_ge_2": verdict_multi,
            "share_ge_3": float((above_15 >= 3).mean()),
            "share_ge_5": float((above_15 >= 5).mean()),
        },
        "best_apr": {
            "median": float(best.median()),
            "share_below_threshold": float((best < L4_THRESHOLD).mean()),
        },
        "lever_is_real": bool(verdict_multi > 0.20),
    }
    (RESULTS_DIR / "opportunity_probe.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n=== 판정 ===")
    if verdict_multi > 0.20:
        print(f"  자격 종목이 2개 이상인 날이 {verdict_multi:.1%} — top_k=1 은 측정 가능한 캐리를 남기고 있다.")
        print("  => 레버가 실재한다. wave38 본 검정으로 진행한다.")
    else:
        print(f"  자격 종목이 2개 이상인 날이 {verdict_multi:.1%}뿐 — top_k>1 이 당길 것이 거의 없다.")
        print("  => 레버가 없다. wave38 은 여기서 중단하고 이 사실만 기록한다.")
    print("\nresults/opportunity_probe.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
