#!/usr/bin/env python3
# Wave-37 step 1: is the Binance funding series wave36 depends on actually representative of the
# venue whose prices it is paired with?
#
# wave36 earned 43.3% of its P&L from funding, and that funding came from the local Binance cache
# while prices came from Bitget (Binance klines are region-blocked here, and Bitget's funding history
# only reaches back ~3 months). That mismatch was disclosed but never MEASURED. It has to be, because
# if the two venues' funding rates diverge materially then wave36's headline is an artefact of pairing
# one venue's payments with another venue's prices.
#
# Bitget's ~3-month funding window is short, but it is enough: 270-540 settled stamps per symbol
# across 20 symbols is thousands of paired observations. What matters is not a long history but
# whether the two series agree on LEVEL and SIGN, because the strategy ranks symbols by funding and
# collects the difference.

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

REPO_ROOT: Final = Path(__file__).resolve().parents[2]
BITGET_CACHE: Final = REPO_ROOT / "research" / "wave35_universe" / "cache"
BINANCE_CACHE: Final = REPO_ROOT / "research" / "wave3" / "cache"
FUNDING_PER_DAY: Final = 3


def load_bitget(symbol: str) -> pd.Series | None:
    path = BITGET_CACHE / f"bitget_{symbol}_funding.csv.gz"
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, format="ISO8601")
    series = frame.set_index("timestamp")["funding_rate"].astype(float)
    return series[~series.index.duplicated(keep="first")].sort_index()


def load_binance(symbol: str) -> pd.Series | None:
    path = BINANCE_CACHE / f"binance_funding_{symbol}.csv.gz"
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, format="ISO8601")
    series = frame.set_index("timestamp")["funding_rate"].astype(float)
    return series[~series.index.duplicated(keep="first")].sort_index()


def main() -> int:
    from research.wave35_universe.dataio35 import build_wide_cache

    _cache, symbols = build_wide_cache()
    print(f"wave36이 사용한 20종목에 대해 Bitget vs Binance 펀딩 정합성 검정\n")
    print(f"{'심볼':>12} {'겹침':>6} {'Bitget APR':>11} {'Binance APR':>12} {'차이(APR)':>10} "
          f"{'부호일치':>9} {'상관':>8}")
    print("-" * 82)

    rows = []
    for symbol in symbols:
        bitget = load_bitget(symbol)
        binance = load_binance(symbol)
        if bitget is None or binance is None:
            print(f"{symbol:>12} 데이터 없음")
            continue
        # align on the 8h stamp (both venues settle at 00/08/16 UTC; floor removes ms jitter)
        bitget_aligned = bitget.copy()
        bitget_aligned.index = bitget_aligned.index.floor("1h")
        binance_aligned = binance.copy()
        binance_aligned.index = binance_aligned.index.floor("1h")
        joined = pd.DataFrame({"bg": bitget_aligned, "bn": binance_aligned}).dropna()
        if len(joined) < 30:
            print(f"{symbol:>12} 겹침 {len(joined)}건 — 표본 부족")
            continue
        bg_apr = float(joined["bg"].mean() * FUNDING_PER_DAY * 365)
        bn_apr = float(joined["bn"].mean() * FUNDING_PER_DAY * 365)
        sign_agree = float((np.sign(joined["bg"]) == np.sign(joined["bn"])).mean())
        correlation = float(np.corrcoef(joined["bg"], joined["bn"])[0, 1])
        rows.append(
            {
                "symbol": symbol,
                "n": len(joined),
                "bitget_apr": bg_apr,
                "binance_apr": bn_apr,
                "gap_apr": bg_apr - bn_apr,
                "sign_agreement": sign_agree,
                "correlation": correlation,
            }
        )
        print(f"{symbol:>12} {len(joined):6d} {bg_apr:10.2%} {bn_apr:11.2%} {bg_apr-bn_apr:+9.2%} "
              f"{sign_agree:8.1%} {correlation:8.4f}")

    if not rows:
        print("검정 불가")
        return 1

    frame = pd.DataFrame(rows)
    print("\n=== 종합 ===")
    print(f"  종목 {len(frame)}개 · 총 겹침 {int(frame['n'].sum()):,} 스탬프")
    print(f"  APR 차이:      중앙 {frame['gap_apr'].median():+.2%} · 절대 중앙 {frame['gap_apr'].abs().median():.2%} "
          f"· 최대 절대 {frame['gap_apr'].abs().max():.2%}")
    print(f"  부호일치:      중앙 {frame['sign_agreement'].median():.1%} · 최저 {frame['sign_agreement'].min():.1%}")
    print(f"  상관:          중앙 {frame['correlation'].median():.4f} · 최저 {frame['correlation'].min():.4f}")

    # What the strategy actually needs: does the RANKING agree? That is what selects the book.
    bitget_panel = {}
    binance_panel = {}
    for symbol in symbols:
        bg, bn = load_bitget(symbol), load_binance(symbol)
        if bg is None or bn is None:
            continue
        bg.index = bg.index.floor("1h")
        bn.index = bn.index.floor("1h")
        bitget_panel[symbol] = bg
        binance_panel[symbol] = bn
    bg_frame = pd.DataFrame(bitget_panel).dropna(how="all")
    bn_frame = pd.DataFrame(binance_panel).reindex(bg_frame.index)
    common = bg_frame.dropna(thresh=10).index.intersection(bn_frame.dropna(thresh=10).index)
    spearman = []
    top_overlap = []
    for stamp in common:
        a = bg_frame.loc[stamp].dropna()
        b = bn_frame.loc[stamp].dropna()
        shared = a.index.intersection(b.index)
        if len(shared) < 8:
            continue
        spearman.append(float(pd.Series(a[shared]).corr(pd.Series(b[shared]), method="spearman")))
        k = 2
        top_a = set(a[shared].nlargest(k).index) | set(a[shared].nsmallest(k).index)
        top_b = set(b[shared].nlargest(k).index) | set(b[shared].nsmallest(k).index)
        top_overlap.append(len(top_a & top_b) / len(top_a))
    if spearman:
        print(f"\n  === 전략이 실제로 쓰는 것: 횡단면 순위 일치 ({len(spearman):,} 스탬프) ===")
        print(f"  스피어만 순위상관: 중앙 {np.median(spearman):.4f} · p10 {np.percentile(spearman,10):.4f}")
        print(f"  상·하위 k=2 종목 집합 일치율: 중앙 {np.median(top_overlap):.1%} · 평균 {np.mean(top_overlap):.1%}")

    ok = (
        frame["gap_apr"].abs().median() < 0.05
        and frame["sign_agreement"].median() > 0.85
        and (not spearman or np.median(spearman) > 0.7)
    )
    print(f"\n판정: {'PASS — Binance 펀딩을 대리로 쓰는 것이 방어 가능' if ok else 'FAIL — 불일치가 크다, wave36 결과를 재해석해야 함'}")
    print("  기준: |APR 차이| 중앙 < 5%p ∧ 부호일치 중앙 > 85% ∧ 횡단면 순위상관 중앙 > 0.7")
    print("\n한계: Bitget 펀딩 이력이 약 3개월이라 이 검정은 최근 구간만 본다. 2022~2025 구간의")
    print("      일치도는 측정 불가이며, 이는 wave36 결과에 남아 있는 미해결 위험이다.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
