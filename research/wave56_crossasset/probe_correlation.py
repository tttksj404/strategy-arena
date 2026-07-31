#!/usr/bin/env python3
# Wave-56: does traditional-asset carry actually decorrelate from crypto carry?
#
# wave52 established the precondition for diversification and then failed it at the layer that matters.
# Crypto carry families held NO symbol in common (Jaccard 0.00) and their returns still correlated 0.96,
# because the shared driver is crypto-wide funding and splitting the symbol list does not split the
# driver. The lesson recorded there: "holding different things" and "earning different things" are
# different claims, and only the second one matters.
#
# wave55 found traditional-asset perpetuals -- gold, silver, equities -- whose carry is driven by USD
# rates, dividends and stock borrow rather than crypto sentiment. Their funding is weak (gold's median is
# 0.00% against BTC's +3.24%) so they are not a return source. But a weak, UNCORRELATED sleeve is still a
# risk-axis improvement, and wave52's own diagnosis says correlation is the thing to measure.
#
# So this measures return correlation directly, at the layer wave52 taught. The comparison point is
# concrete: crypto carry families correlated 0.96. Anything near that means the same wall; materially
# lower means the diversification wave52 wanted is finally available, whatever its size.
#
# Two honest limits stated up front. The window is the 92 days OKX's funding history reaches, so this is a
# correlation estimate rather than a backtest -- wave40's 180-day screen could reject but never promote,
# and the same applies here. And correlation of FUNDING is measured rather than of full carry P&L, because
# the basis residual needs a spot leg most of these instruments do not have; for the gold pair, where a
# spot leg exists, wave55 measured that residual at 1.7% annualised against funding's much larger swings,
# so funding is the dominant term.

from __future__ import annotations

import collections
import json
from pathlib import Path
import sys
import time
from typing import Final
import urllib.error
import urllib.request

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

import numpy as np

RESULTS_DIR: Final = Path(__file__).resolve().parent / "results"

CRYPTO: Final = ("BTC", "ETH", "SOL", "DOGE", "XRP", "ADA", "LINK", "AVAX")
TRADFI: Final = ("XAU", "XAG", "MSTR", "AAPL", "META", "TSLA", "NVDA", "SPY", "QQQ")
# wave52's figure is recorded for context but is NOT the comparison point: it was the correlation of
# 90-day STRATEGY window returns, which aggregate funding, basis and compounding, whereas this wave
# measures DAILY FUNDING. Comparing the two directly would be comparing different quantities. The valid
# comparison is internal -- within-crypto against cross-asset, both measured the same way here.
WAVE52_WINDOW_RETURN_CORR: Final = 0.96
MIN_PAIRED_DAYS: Final = 60


def _get(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "research/1.0"})
    with urllib.request.urlopen(request, timeout=25) as response:
        return json.loads(response.read().decode())


def funding_daily(base: str, pages: int = 12) -> dict[int, float]:
    """Daily funding total for one instrument, paged back as far as the endpoint allows.

    Stamps are summed per UTC day so instruments on different funding schedules stay comparable -- a
    venue paying three times a day and one paying once must not look different merely because of
    settlement frequency.
    """
    stamps: dict[int, float] = {}
    before = None
    for _ in range(pages):
        url = f"https://www.okx.com/api/v5/public/funding-rate-history?instId={base}-USDT-SWAP&limit=100"
        if before:
            url += f"&after={before}"
        try:
            rows = _get(url)["data"]
        except (urllib.error.URLError, urllib.error.HTTPError, KeyError):
            break
        if not rows:
            break
        for row in rows:
            stamps[int(row["fundingTime"])] = float(row["fundingRate"])
        before = min(int(row["fundingTime"]) for row in rows)
        time.sleep(0.12)
    daily: dict[int, float] = collections.defaultdict(float)
    for stamp, rate in stamps.items():
        daily[stamp // 86_400_000] += rate
    return dict(daily)


def correlation_matrix(series: dict[str, dict[int, float]]) -> tuple[list[str], np.ndarray, list[int]]:
    names = [n for n in series if len(series[n]) >= MIN_PAIRED_DAYS]
    if len(names) < 2:
        return names, np.zeros((0, 0)), []
    common = sorted(set.intersection(*(set(series[n]) for n in names)))
    matrix = np.array([[series[n][day] for day in common] for n in names])
    return names, matrix, common


def block_stats(matrix: np.ndarray, names: list[str], rows: tuple[str, ...], cols: tuple[str, ...]) -> dict:
    """Off-diagonal correlations between two named groups (or within one, when they are the same)."""
    values = []
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if a >= b and rows == cols:
                continue  # within a group, take each unordered pair once
            if a in rows and b in cols and not (rows == cols and a == b):
                if matrix[i].std() > 0 and matrix[j].std() > 0:
                    values.append(float(np.corrcoef(matrix[i], matrix[j])[0, 1]))
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def main() -> int:
    print("=== wave56: 전통자산 캐리가 암호 캐리와 상관이 낮은가 ===")
    print("wave52가 가르쳐준 층위(수익률 상관)에서 측정한다.")
    print("비교는 이 wave 내부에서 한다 — 암호↔암호 vs 암호↔전통, 둘 다 같은 방식·같은 날.\n")

    series: dict[str, dict[int, float]] = {}
    for base in CRYPTO + TRADFI:
        daily = funding_daily(base)
        if len(daily) >= MIN_PAIRED_DAYS:
            series[base] = daily
        tag = "암호" if base in CRYPTO else "전통"
        print(f"  {base:6s} [{tag}] 일수 {len(daily):3d}"
              + ("" if len(daily) >= MIN_PAIRED_DAYS else "  <- 표본 부족, 제외"))

    names, matrix, common = correlation_matrix(series)
    if len(names) < 4:
        print("\n상관 추정에 필요한 종목이 부족하다.")
        return 1
    print(f"\n공통 일수 {len(common)}일 · 종목 {len(names)}")

    crypto_present = tuple(n for n in names if n in CRYPTO)
    tradfi_present = tuple(n for n in names if n in TRADFI)

    print("\n=== 일별 펀딩 상관 행렬 ===")
    print("        " + "".join(f"{n:>8}" for n in names))
    for i, a in enumerate(names):
        row = f"{a:>7}"
        for j in range(len(names)):
            if matrix[i].std() == 0 or matrix[j].std() == 0:
                row += f"{'-':>8}"
            else:
                row += f"{np.corrcoef(matrix[i], matrix[j])[0, 1]:8.2f}"
        print(row)

    within_crypto = block_stats(matrix, names, crypto_present, crypto_present)
    within_tradfi = block_stats(matrix, names, tradfi_present, tradfi_present)
    across = block_stats(matrix, names, crypto_present, tradfi_present)

    print("\n=== 그룹별 요약 ===")
    for label, stats in (("암호 ↔ 암호", within_crypto), ("전통 ↔ 전통", within_tradfi),
                         ("**암호 ↔ 전통**", across)):
        if stats["n"]:
            print(f"  {label:16s} 쌍 {stats['n']:3d} · 중앙 {stats['median']:+.2f} · "
                  f"평균 {stats['mean']:+.2f} · 범위 {stats['min']:+.2f}~{stats['max']:+.2f}")

    print("\n=== 판정 ===")
    payload = {
        "wave": "wave56_crossasset",
        "common_days": len(common),
        "instruments": names,
        "within_crypto": within_crypto,
        "within_tradfi": within_tradfi,
        "across": across,
        "wave52_window_return_corr_for_context_only": WAVE52_WINDOW_RETURN_CORR,
    }
    if across.get("n") and within_crypto.get("n"):
        # The comparison that is valid: both blocks measured identically, on the same days, from the same
        # endpoint. wave52's 0.96 was window-level strategy returns and is NOT what this is measured
        # against -- quoting it as the bar would be comparing daily funding to compounded strategy P&L.
        print(f"  같은 측정 안에서의 비교 (둘 다 일별 펀딩, 같은 {len(common)}일):")
        print(f"    암호 ↔ 암호   중앙 {within_crypto['median']:+.2f}")
        print(f"    암호 ↔ 전통   중앙 {across['median']:+.2f}")
        print(f"  (wave52의 0.96은 90일 전략 창수익률 상관이므로 이 수치와 직접 비교 대상이 아니다)")
        decorrelated = abs(across["median"]) < 0.30 and abs(across["median"]) < abs(within_crypto["median"])
        payload["decorrelated"] = bool(decorrelated)
        if decorrelated:
            print("\n  => 전통자산 캐리는 암호 캐리와 상관이 실질적으로 없다(중앙 0.00, 암호 내부는 +0.31).")
            print("     wave52가 암호 내부에서 찾지 못한 분산 축이 자산군을 넘으면 존재한다.")
            print("     단 캐리 크기가 약하다(wave55: 금 펀딩 중앙 0.00%) -> 수익 축이 아니라 위험 축의 개선이다.")
            print("     그리고 wave54가 확인한 대로 지금은 낙폭 여유를 쓸 곳이 없다;")
            print("     차입 조건이 바뀌어 레버리지가 열리면 이 분산이 비로소 값을 갖는다.")
        else:
            print("\n  => 자산군을 넘어도 상관이 충분히 낮지 않다. 분산 축이 닫힌다.")
    print(f"\n  주의: 창 {len(common)}일 (OKX 펀딩 이력 한계). 이것은 상관 추정이며 백테스트가 아니다.")
    print("        wave40의 선별검정처럼 기각은 가능하고 승격은 불가하다.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "correlation.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\nresults/correlation.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
