#!/usr/bin/env python3
# Wave-44: evidence for the listing-alignment artifact, and sensitivity of the fix.
#
# This script exists so the correction rests on reproducible measurement rather than on the narrative in
# DATA_CORRECTION.md. It establishes four things in order:
#   1. The artifact exists and is identifiable (WIFUSDT's spot leg lists 46 days after its perp).
#   2. It is systemic, not one symbol (share of extreme-basis days falling in the first days of listing).
#   3. The campaign's own data verification could not have caught it (it only checked END alignment).
#   4. The chosen basis sanity limit is not load-bearing (results across 1%, 2%, 5%).
#
# Point 4 matters most. MIN_LISTING_AGE_DAYS follows from the measured distribution, but
# BASIS_SANITY_LIMIT is a judgement, and a judgement that changes the conclusion is a problem. Reporting
# the sensitivity is the only honest way to hold that number.

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

from research.wave38_breadth import dataio38
from research.wave38_breadth.dataio38 import CACHE, build_panel
from research.wave38_breadth.engine38 import ACTIVE_CAPITAL, CarryConfig, simulate

RESULTS_DIR: Final = Path(__file__).resolve().parent / "results"
VERIFICATION_JSON: Final = (
    Path(__file__).resolve().parents[1] / "wave12_frontier" / "cache" / "spot_verification.json"
)


def listing_lag_table() -> pd.DataFrame:
    rows = []
    for spot_path in sorted(CACHE.glob("binance_spot_*_1d.csv.gz")):
        symbol = spot_path.name[len("binance_spot_") : -len("_1d.csv.gz")]
        perp_path = CACHE / f"binance_fapi_{symbol}_1d.csv.gz"
        if not perp_path.exists():
            continue
        try:
            spot_start = pd.read_csv(spot_path)["timestamp"].iloc[0][:10]
            perp_start = pd.read_csv(perp_path)["timestamp"].iloc[0][:10]
        except (KeyError, IndexError, ValueError):
            continue
        rows.append(
            {
                "symbol": symbol,
                "spot_start": spot_start,
                "perp_start": perp_start,
                "lag_days": (pd.Timestamp(spot_start) - pd.Timestamp(perp_start)).days,
            }
        )
    return pd.DataFrame(rows)


def yearly(result, days) -> dict[int, float]:
    index = pd.DatetimeIndex(days[1 : 1 + len(result.equity)])
    equity = result.equity
    previous = ACTIVE_CAPITAL
    out: dict[int, float] = {}
    for year in sorted(set(index.year)):
        mask = index.year == year
        if mask.sum() < 2:
            continue
        out[int(year)] = float(equity[mask][-1] / previous - 1.0)
        previous = equity[mask][-1]
    return out


def main() -> int:
    payload: dict = {"wave": "wave44_correction"}

    print("=== 1. 인공물의 존재: WIFUSDT ===")
    for prefix, label in (("fapi", "퍼프"), ("spot", "현물")):
        frame = pd.read_csv(CACHE / f"binance_{prefix}_WIFUSDT_1d.csv.gz")
        print(f"  {label} 첫 행 {frame['timestamp'].iloc[0][:10]} · {len(frame)}행")
    spot = pd.read_csv(CACHE / "binance_spot_WIFUSDT_1d.csv.gz")
    perp = pd.read_csv(CACHE / "binance_fapi_WIFUSDT_1d.csv.gz")
    first_day = spot["timestamp"].iloc[0][:10]
    perp_row = perp[perp["timestamp"].str.startswith(first_day)]
    spot_move = spot["close"].iloc[0] / spot["open"].iloc[0] - 1.0
    perp_move = float(perp_row["close"].iloc[0] / perp_row["open"].iloc[0] - 1.0) if len(perp_row) else float("nan")
    print(f"  현물 상장 첫날 {first_day}: 현물 {spot_move:+.2%} · 퍼프 {perp_move:+.2%} "
          f"-> 계상되던 베이시스 {spot_move - perp_move:+.2%}")
    payload["wifusdt"] = {"spot_first_day": first_day, "spot_move": spot_move, "perp_move": perp_move,
                          "booked_basis": spot_move - perp_move}

    print("\n=== 2. 체계적 규모: 현물이 퍼프보다 늦게 상장한 종목 ===")
    lags = listing_lag_table()
    late = lags[lags.lag_days > 0]
    print(f"  전체 {len(lags)}종목 중 현물이 늦음 {len(late)}종목 ({len(late)/len(lags):.1%})")
    print(f"  30일 이상 늦음 {int((lags.lag_days >= 30).sum())}종목 · 90일 이상 {int((lags.lag_days >= 90).sum())}종목")
    print(f"  최대 {lags.lag_days.max()}일 ({lags.loc[lags.lag_days.idxmax(), 'symbol']})")
    payload["listing_lag"] = {
        "symbols": int(len(lags)),
        "spot_later": int(len(late)),
        "ge_30d": int((lags.lag_days >= 30).sum()),
        "ge_90d": int((lags.lag_days >= 90).sum()),
        "max_lag_days": int(lags.lag_days.max()),
    }

    print("\n=== 3. 왜 자체 검증을 통과했나 ===")
    if VERIFICATION_JSON.exists():
        verification = json.loads(VERIFICATION_JSON.read_text(encoding="utf-8"))
        fields = sorted({key for entry in verification.values() for key in entry})
        print(f"  spot_verification.json 검증 필드: {fields}")
        print("  => 전부 종료일(end/gap) 관련. 시작일 정렬은 미검증 -> WIFUSDT 유형은 통과한다.")
        payload["verification_fields"] = fields
    else:
        print("  spot_verification.json 없음")

    print("\n=== 4. 정정의 민감도: 베이시스 상한을 바꿔도 결론이 유지되는가 ===")
    original_limit = dataio38.BASIS_SANITY_LIMIT
    original_age = dataio38.MIN_LISTING_AGE_DAYS
    table = []
    try:
        for age, limit in ((0, 10.0), (0, original_limit), (original_age, 10.0), (original_age, 0.01),
                           (original_age, 0.02), (original_age, 0.05)):
            dataio38.MIN_LISTING_AGE_DAYS = age
            dataio38.BASIS_SANITY_LIMIT = limit
            build_panel.cache_clear()
            panel = build_panel()
            n_days = len(panel.days)
            result = simulate(panel, CarryConfig(1, 0.50, 0.50), 1, n_days)
            years = yearly(result, panel.days)
            recent = [v for y, v in years.items() if y >= 2023]
            row = {
                "min_listing_age_days": age,
                "basis_limit": limit,
                "annualised": result.annualised(n_days - 1),
                "recent_mean": float(np.mean(recent)) if recent else float("nan"),
                "basis_usd": result.basis_usd,
                "funding_usd": result.funding_usd,
            }
            table.append(row)
            tag = "정정 전" if age == 0 and limit == 10.0 else ("채택" if (age, limit) == (original_age, original_limit) else "")
            print(f"  연령{age:3d}일 상한{limit:>5} | 연 {row['annualised']:+7.2%} | 2023+ {row['recent_mean']:+6.2%} "
                  f"| 베이시스 ${row['basis_usd']:+7.2f} | 펀딩 ${row['funding_usd']:+7.2f}  {tag}")
    finally:
        dataio38.MIN_LISTING_AGE_DAYS = original_age
        dataio38.BASIS_SANITY_LIMIT = original_limit
        build_panel.cache_clear()

    payload["sensitivity"] = table
    baseline = next(r for r in table if r["min_listing_age_days"] == 0 and r["basis_limit"] == 10.0)
    corrected = [r for r in table if r["min_listing_age_days"] == original_age and r["basis_limit"] in (0.01, 0.02, 0.05)]
    spread = max(r["recent_mean"] for r in corrected) - min(r["recent_mean"] for r in corrected)
    print(f"\n  정정 전 2023+ {baseline['recent_mean']:+.2%}")
    print(f"  상한 1%~5% 사이 2023+ 변동폭 {spread:.2%}p -> "
          f"{'상한 선택이 결론을 좌우하지 않는다' if spread < 0.02 else '상한 선택이 결론에 영향을 준다(주의)'}")
    payload["limit_sensitivity_spread"] = spread

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "correction.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\nresults/correction.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
