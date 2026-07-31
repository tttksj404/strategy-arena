#!/usr/bin/env python3
# Wave-40 part 2: screen the leading-indicator axis (open interest, long/short account ratio).
#
# wave19 named this axis and stopped at stage 1: its regime-rotation design needed a signal that
# predicted funding spikes with lead time, and no such signal was found in the data it had. The
# suggestion since has been that live-only series -- open interest, liquidations, order flow -- might
# carry the lead time that price and funding history do not.
#
# The first thing to establish is how much history exists, because a strategy cannot be backtested on a
# window shorter than its own selection needs. OKX's rubik endpoints return 180 days. That is one regime
# and roughly two 90-day windows: a walk-forward would get one reselection, and an IS/OOS split would be
# a coin flip dressed as evidence. So this is explicitly a SCREEN, not a backtest, and its only job is to
# answer whether a lead-lag relationship large enough to be worth forward-collecting for exists at all.
#
# A screen can only reject, never promote. If nothing shows up in 180 days the axis stays closed for now
# and forward collection is the only route; if something does show up, that justifies building the
# collector and waiting, not deploying.

from __future__ import annotations

import json
from pathlib import Path
import statistics
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
CURRENCIES: Final = (
    "BTC", "ETH", "SOL", "DOGE", "ADA", "XRP", "LINK", "AVAX",
    "NEAR", "ARB", "OP", "SUI", "APT", "LTC", "BCH", "FIL",
)
MIN_PAIRED_DAYS: Final = 60
# A |Spearman| below this is not worth building a collector for even if it were significant: wave14
# required 0.7 for a venue-agreement claim, and a predictive edge needs to survive costs, so 0.20 is
# already a generous floor for "worth waiting six months to test properly".
INTERESTING_RHO: Final = 0.20


def _get(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "research/1.0"})
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.loads(response.read().decode())


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Rank correlation without scipy (not a dependency of this repo)."""
    if len(x) < 3:
        return float("nan")
    def rank(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="stable")
        ranks = np.empty(len(values), dtype=float)
        ranks[order] = np.arange(len(values), dtype=float)
        return ranks
    rx, ry = rank(x), rank(y)
    rx -= rx.mean()
    ry -= ry.mean()
    denominator = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / denominator) if denominator > 0 else float("nan")


def fetch_series(currency: str) -> dict | None:
    """Daily open interest, long/short account ratio and swap close for one currency."""
    try:
        oi = _get(
            f"https://www.okx.com/api/v5/rubik/stat/contracts/open-interest-volume?ccy={currency}&period=1D"
        )["data"]
        lsr = _get(
            f"https://www.okx.com/api/v5/rubik/stat/contracts/long-short-account-ratio?ccy={currency}&period=1D"
        )["data"]
        candles = _get(
            f"https://www.okx.com/api/v5/market/history-candles?instId={currency}-USDT-SWAP&bar=1D&limit=300"
        )["data"]
    except (urllib.error.URLError, urllib.error.HTTPError, KeyError):
        return None
    if not oi or not lsr or not candles:
        return None

    oi_map = {int(row[0]) // 86_400_000: float(row[1]) for row in oi}
    lsr_map = {int(row[0]) // 86_400_000: float(row[1]) for row in lsr}
    close_map = {int(row[0]) // 86_400_000: float(row[4]) for row in candles}
    days = sorted(set(oi_map) & set(lsr_map) & set(close_map))
    if len(days) < MIN_PAIRED_DAYS:
        return None
    return {
        "days": days,
        "oi": np.array([oi_map[d] for d in days]),
        "lsr": np.array([lsr_map[d] for d in days]),
        "close": np.array([close_map[d] for d in days]),
    }


def main() -> int:
    print("=== wave40-2: 선행지표(OI·롱숏비율) 선별 검정 — 백테스트 아님 ===")
    print("OKX rubik 소급 깊이 180일 = 단일 레짐, 90일창 2개분. 백테스트하면 과최적화가 보장된다.")
    print("이 검정은 '전진 수집할 가치가 있는 크기의 선행관계가 존재하는가'만 묻는다.\n")

    rows = []
    for currency in CURRENCIES:
        series = fetch_series(currency)
        if series is None:
            print(f"  {currency:6s} 데이터 부족/실패")
            continue
        close = series["close"]
        oi = series["oi"]
        lsr = series["lsr"]

        forward_return = close[1:] / close[:-1] - 1.0  # return realised AFTER the indicator is known
        oi_change = np.diff(oi) / np.where(oi[:-1] == 0, np.nan, oi[:-1])
        lsr_level = lsr[:-1]
        lsr_change = np.diff(lsr)

        # Align so every predictor is strictly earlier than the return it is tested against.
        n = min(len(forward_return), len(oi_change), len(lsr_level), len(lsr_change)) - 1
        if n < MIN_PAIRED_DAYS:
            print(f"  {currency:6s} 정렬 후 {n}일뿐")
            continue
        target = forward_return[1 : n + 1]
        rho_oi = _spearman(oi_change[:n], target)
        rho_lsr_level = _spearman(lsr_level[:n], target)
        rho_lsr_change = _spearman(lsr_change[:n], target)
        rows.append(
            {
                "currency": currency,
                "days": int(n),
                "rho_oi_change_vs_next_return": rho_oi,
                "rho_lsr_level_vs_next_return": rho_lsr_level,
                "rho_lsr_change_vs_next_return": rho_lsr_change,
            }
        )
        print(
            f"  {currency:6s} n={n:3d} | OI변화->익일수익 {rho_oi:+.3f} | "
            f"롱숏수준 {rho_lsr_level:+.3f} | 롱숏변화 {rho_lsr_change:+.3f}"
        )
        time.sleep(0.15)

    if not rows:
        print("\n데이터를 확보하지 못했다 — 판정 보류.")
        return 1

    print(f"\n=== 통합 ({len(rows)}종목) ===")
    summary = {}
    for key, label in (
        ("rho_oi_change_vs_next_return", "OI변화 -> 익일수익"),
        ("rho_lsr_level_vs_next_return", "롱숏비율 수준 -> 익일수익"),
        ("rho_lsr_change_vs_next_return", "롱숏비율 변화 -> 익일수익"),
    ):
        values = [r[key] for r in rows if np.isfinite(r[key])]
        median = statistics.median(values)
        mean = statistics.fmean(values)
        share_interesting = sum(1 for v in values if abs(v) >= INTERESTING_RHO) / len(values)
        # If the sign is not even consistent across symbols there is no shared mechanism to exploit.
        sign_consistency = max(
            sum(1 for v in values if v > 0), sum(1 for v in values if v < 0)
        ) / len(values)
        summary[key] = {
            "median": median,
            "mean": mean,
            "share_abs_ge_threshold": share_interesting,
            "sign_consistency": sign_consistency,
        }
        print(
            f"  {label:24s} 중앙 {median:+.3f} · 평균 {mean:+.3f} · "
            f"|rho|>={INTERESTING_RHO} 비율 {share_interesting:.0%} · 부호일치 {sign_consistency:.0%}"
        )

    best = max(abs(summary[k]["median"]) for k in summary)
    worth_collecting = best >= INTERESTING_RHO

    print("\n=== 판정 ===")
    print(f"  최대 |중앙 rho| = {best:.3f} (문턱 {INTERESTING_RHO})")
    if worth_collecting:
        print("  => 전진 수집을 정당화할 크기의 선행관계가 보인다. 수집기를 만들고 6개월 기다려 검정한다.")
        print("     지금 배포할 근거는 아니다 — 180일은 여전히 백테스트 불가 길이다.")
    else:
        print("  => 180일 창에서 익일 수익을 설명하는 선행관계가 문턱에 못 미친다.")
        print("     wave19가 예측신호에서 멈춘 결론이 OI·롱숏비율로도 재현된다.")
        print("     이 축은 '데이터가 없어서 미측정'이 아니라 '측정했고 신호가 약함'으로 바뀐다.")
    print("  주의: n이 작아 통계력이 낮다. 이 검정은 기각만 할 수 있고 승격은 할 수 없다.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "leading_screen.json").write_text(
        json.dumps(
            {
                "axis": "leading indicators (open interest, long/short ratio)",
                "note": "screen only -- 180 day history is too short to backtest",
                "history_days_available": 180,
                "interesting_threshold": INTERESTING_RHO,
                "per_currency": rows,
                "summary": summary,
                "worth_forward_collecting": worth_collecting,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("\nresults/leading_screen.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
