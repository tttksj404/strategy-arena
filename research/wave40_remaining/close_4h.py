#!/usr/bin/env python3
# Wave-40 part 1: settle the 4H timeframe axis.
#
# The 4H axis has been open since wave35 measured cost/volatility ratios by timeframe (1m 2.33x, 1H
# 0.20x, 1D 0.03x) and left 4H unmeasured between 1H and 1D. The temptation is to collect 4H bars for
# dozens of symbols and re-run everything. That would cost hours, so the first question is whether the
# answer is already determined by measurements in hand -- and for the only surviving strategy family it
# is, by arithmetic rather than by preference.
#
# What finer granularity can buy is specific and narrow: funding is paid only at discrete stamps (three
# per day, 8h apart), so in principle a book could hold only around each stamp, collect 100% of the
# funding, and carry price exposure for a fraction of the time. That is a real edge FOR A BOOK WITH
# PRICE EXPOSURE. The surviving family is delta-neutral, and wave38/39 measured its price term directly:
# basis contributed +$8.39 to +$19.29 over the full period while funding contributed +$125 to +$143.
# There is no price exposure to shed. Meanwhile stamp-timing multiplies turnover, and turnover cost is
# already a third of funding income.
#
# So the honest test is not "does 4H work" in the abstract but "does the turnover that stamp-timing
# requires cost more than the exposure it saves". This computes both sides from measured quantities.

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Final

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

RESULTS_DIR: Final = Path(__file__).resolve().parent / "results"
WAVE38_RESULTS: Final = Path(__file__).resolve().parents[1] / "wave38_breadth" / "results" / "final.json"
WAVE39_RESULTS: Final = Path(__file__).resolve().parents[1] / "wave39_margin" / "results" / "final.json"

FUNDING_STAMPS_PER_DAY: Final = 3  # Binance/Bitget/OKX 8h schedule
# wave35's measured cost-to-typical-range ratios, for the record
COST_RATIO_BY_TIMEFRAME: Final = {"1m": 2.33, "5m": 0.79, "15m": 0.49, "30m": 0.34, "1H": 0.20, "1D": 0.03}


def main() -> int:
    print("=== wave40-1: 4H 시간프레임 축 판정 ===")
    print("wave35 실측 비용/전형변동폭 비율:", ", ".join(f"{k} {v}x" for k, v in COST_RATIO_BY_TIMEFRAME.items()))
    print("  4H는 1H(0.20x)와 1D(0.03x) 사이 — 1D보다 반드시 불리하다(비용이 변동폭 대비 크다).\n")

    base = json.loads(WAVE38_RESULTS.read_text(encoding="utf-8"))["base"]
    port = json.loads(WAVE39_RESULTS.read_text(encoding="utf-8"))["base_portfolio_margin"]

    print("=== 세밀한 시간프레임이 줄여줄 '가격 노출'이 실제로 존재하는가 ===")
    for label, run in (("wave38 격리마진", base), ("wave39 통합마진", port)):
        funding, basis, cost = run["funding_usd"], run["basis_usd"], run["cost_usd"]
        print(f"  {label}: 펀딩 ${funding:+.2f} · 베이시스 ${basis:+.2f} · 비용 ${cost:+.2f}")
    print("  => 베이시스는 작고 **양수**다. 델타중립이므로 줄일 손실이 없다.")
    print("     4H가 제공하는 유일한 새 능력(스탬프 근처만 보유)은 여기서 아무것도 절약하지 못한다.\n")

    print("=== 스탬프 타이밍의 비용 (측정값으로 계산) ===")
    run = port
    active_days = run["days_active"]
    current_entries = run["entries"]
    current_cost = abs(run["cost_usd"])
    funding_income = run["funding_usd"]

    # Holding only around each stamp means entering and exiting three times per active day instead of
    # holding through. Turnover scales with the number of round trips, and the measured cost per round
    # trip is the only rate needed -- it comes from this run's own realised cost, not an assumption.
    stamp_round_trips = active_days * FUNDING_STAMPS_PER_DAY
    cost_per_round_trip = current_cost / current_entries if current_entries else 0.0
    projected_cost = stamp_round_trips * cost_per_round_trip

    print(f"  현재: 진입 {current_entries}회 · 비용 ${current_cost:.2f} -> 진입당 ${cost_per_round_trip:.4f}")
    print(f"  스탬프 타이밍: 활성 {active_days}일 x {FUNDING_STAMPS_PER_DAY}스탬프 = {stamp_round_trips:,}회 왕복")
    print(f"  예상 비용 ${projected_cost:,.2f} (현재의 {projected_cost/current_cost:.1f}배)")
    print(f"  펀딩 수입은 ${funding_income:+.2f} 로 불변 (스탬프에 있으면 전액 수취)")
    net = funding_income - projected_cost + run["basis_usd"]
    print(f"  순손익 ${net:+,.2f}  <- {'양수' if net > 0 else '음수: 비용이 펀딩을 초과'}")

    print("\n=== 판정 ===")
    verdict_closed = net <= 0
    if verdict_closed:
        print("  4H(및 그 이하) 축은 **닫힌다**. 이유는 두 겹이다:")
        print("   1) 델타중립 캐리에는 세밀화로 줄일 가격 노출이 없다(베이시스가 이미 작고 양수).")
        print("   2) 세밀화가 만드는 회전비용이 펀딩 수입을 초과한다.")
        print("  세밀한 시간프레임이 도움될 계열은 '가격 노출이 큰 방향성 전략'이지만,")
        print("  그 계열은 wave19·21~24·30~34에서 이미 기각됐다. 열린 조합이 남아있지 않다.")
    else:
        print("  산술로 닫히지 않는다 -> 4H 실데이터 수집이 정당화된다.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "close_4h.json").write_text(
        json.dumps(
            {
                "axis": "4H timeframe",
                "cost_ratio_by_timeframe": COST_RATIO_BY_TIMEFRAME,
                "basis_usd_wave38": base["basis_usd"],
                "basis_usd_wave39": port["basis_usd"],
                "funding_usd_wave39": funding_income,
                "current_entries": current_entries,
                "current_cost_usd": current_cost,
                "cost_per_round_trip": cost_per_round_trip,
                "stamp_timing_round_trips": stamp_round_trips,
                "stamp_timing_projected_cost": projected_cost,
                "stamp_timing_net": net,
                "closed": verdict_closed,
                "reason": (
                    "delta-neutral carry has no price exposure to shed, and stamp timing's turnover cost "
                    "exceeds the funding it would capture"
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("\nresults/close_4h.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
