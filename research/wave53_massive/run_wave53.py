#!/usr/bin/env python3
# Wave-53: millions of configurations, and the only question worth spending them on.
#
# fast53 reproduces engine38 to 5e-14 at roughly 1,800x the speed, which makes a genuinely large sweep
# possible for the first time in this campaign. The tempting use is to search harder for a better
# configuration. wave50 already showed where that leads: raising the search budget 5x made results WORSE
# and widened the seed spread, because more search finds configurations that fit the past better.
#
# So the compute is spent on the question a small search cannot answer at all: IS THE INCUMBENT
# DISTINGUISHABLE FROM SEARCH ITSELF? White's Reality Check answers it by building the distribution of the
# best result obtainable from an UNINFORMATIVE signal under the same search intensity. If exhaustively
# searching a randomised signal routinely finds something as good as the real one, then the real one is a
# search artefact and the whole carry premise collapses. If it does not, the incumbent has an edge that
# survives the multiplicity of everything tried across 52 waves.
#
# Constructing the null correctly is the crux. The SIGNAL is permuted across symbols within each day, so
# the strategy gates and ranks on a value with no relationship to the symbol it belongs to, while the
# REALISED funding, basis, prices and costs stay exactly as they were -- what a position earns must remain
# that symbol's actual cash flow. Permuting the earnings instead would test nothing about selection. The
# marginal distribution of the signal is preserved by construction, so the null strategy trades as often
# and at the same thresholds as the real one.

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
import sys
import time
from typing import Final

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

import numpy as np

from research.wave38_breadth.dataio38 import build_panel, with_threshold
from research.wave53_massive.fast53 import ACTIVE_CAPITAL, build_daily_series, evaluate_grid

RESULTS_DIR: Final = Path(__file__).resolve().parent / "results"

THRESHOLDS: Final = (0.10, 0.15, 0.25, 0.35, 0.50)
TOP_KS: Final = (1, 2, 3, 5, 8, 12, 20)
LEG_VALUES: Final = np.linspace(0.05, 1.00, 40)
CAP_VALUES: Final = np.linspace(0.10, 1.00, 40)
MIN_LEG_FLOOR: Final = 5.0
I5_CORRECTED: Final = 0.0828
WAVE42_FULL: Final = 0.1331

# The null uses a reduced but still exhaustive sweep: enough to give chance a fair shot at the same kind of
# search the real strategy got, while keeping each replication inside a few seconds.
NULL_THRESHOLDS: Final = (0.15, 0.25, 0.35)
NULL_TOP_KS: Final = (1, 2, 3, 5)


def permute_signal(panel, rng: np.random.Generator):
    """Panel with the ranking/gating signal shuffled across symbols within each day.

    Only raw_apr is permuted; `active` is then rebuilt from it by with_threshold so the hysteresis sees the
    permuted series, and ranking_apr is set to its shifted form to match dataio38's own convention. Prices,
    realised funding, costs and tradability are untouched, so the null earns exactly what the real strategy
    would earn from whichever symbols its meaningless signal happens to pick.
    """
    permuted = panel.raw_apr.copy()
    for day in range(permuted.shape[0]):
        row = permuted[day]
        finite = np.flatnonzero(np.isfinite(row))
        if len(finite) > 1:
            row[finite] = rng.permutation(row[finite])
    shifted = np.full_like(permuted, np.nan)
    shifted[1:] = permuted[:-1]
    return dataclasses.replace(panel, raw_apr=permuted, ranking_apr=shifted)


def sweep(panel, thresholds, top_ks, leg_values, cap_values, start_day: int = 1) -> dict:
    """Exhaustive sweep over [start_day, end). Returns the best annualised return and evaluation count.

    The daily series are always built over the WHOLE panel even when only a later slice is evaluated, so
    the hysteresis state entering `start_day` is the state a live book would actually have carried in.
    Rebuilding the series from start_day would hand the evaluation a flat book on its first day, which is a
    different and easier problem.
    """
    n_days = len(panel.days)
    years = (n_days - start_day) / 365.0
    best = {"annualised": -np.inf}
    evaluated = 0
    for threshold in thresholds:
        variant = with_threshold(panel, threshold)
        for top_k in top_ks:
            series = build_daily_series(variant, top_k)
            grid = evaluate_grid(series, leg_values, cap_values, start_day, n_days)
            final = grid["final"]
            feasible = grid["min_leg"] >= MIN_LEG_FLOOR
            annualised = np.where(
                (final > 0) & feasible, (np.maximum(final, 1e-9) / ACTIVE_CAPITAL) ** (1.0 / years) - 1.0, -np.inf
            )
            evaluated += annualised.size
            flat = int(np.argmax(annualised))
            if annualised.flat[flat] > best["annualised"]:
                index = np.unravel_index(flat, annualised.shape)
                best = {
                    "annualised": float(annualised[index]),
                    "final": float(final[index]),
                    "mdd": float(grid["mdd"][index]),
                    "min_leg": float(grid["min_leg"][index]),
                    "threshold": threshold,
                    "top_k": top_k,
                    "leg": float(grid["leg"][index]),
                    "cap": float(grid["cap"][index]),
                }
    best["evaluated"] = evaluated
    return best


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="wave53: massive sweep and empirical null")
    parser.add_argument("--stage", choices=("real", "null", "judge"), required=True)
    parser.add_argument("--replications", type=int, default=20)
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument("--from-year", type=int,
                        help="evaluate only from this calendar year onward; the full-period test is "
                             "dominated by 2021, so significance in the recent regime is a separate claim")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    started = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "final.json"
    payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"null_best": []}
    panel = build_panel()

    # A separate results file per period, so the full-period verdict is never overwritten by the
    # recent-regime one and both stay auditable side by side.
    start_day = 1
    period = "full"
    if args.from_year:
        period = f"from{args.from_year}"
        matching = [i for i, day in enumerate(panel.days) if day.year >= args.from_year]
        if not matching:
            print(f"{args.from_year}년 이후 데이터가 없다")
            return 1
        start_day = matching[0]
        path = RESULTS_DIR / f"final_{period}.json"
        payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"null_best": []}
        print(f"[구간 제한] {panel.days[start_day].date()} ~ {panel.days[-1].date()} "
              f"({len(panel.days)-start_day}일, {(len(panel.days)-start_day)/365.0:.2f}년)\n")

    if args.stage == "real":
        total = len(THRESHOLDS) * len(TOP_KS) * len(LEG_VALUES) * len(CAP_VALUES)
        print(f"=== wave53 실제 신호 전수 탐색 ===")
        print(f"임계 {len(THRESHOLDS)} x top_k {len(TOP_KS)} x leg {len(LEG_VALUES)} x cap {len(CAP_VALUES)} "
              f"= {total:,}조합")
        best = sweep(panel, THRESHOLDS, TOP_KS, LEG_VALUES, CAP_VALUES, start_day)
        payload["real_full"] = best
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        print(f"\n  최고 연환산 {best['annualised']:+.2%} (${best['final']:,.2f}) · MDD {best['mdd']:.2%}")
        print(f"  구성: 임계 {best['threshold']:.0%} · k {best['top_k']} · leg {best['leg']:.3f} · cap {best['cap']:.3f}")
        print(f"  최소레그 ${best['min_leg']:.2f} · 평가 {best['evaluated']:,}조합 · {time.time()-started:.0f}s")

        print(f"\n=== 귀무 탐색과 동일 조건의 실제 신호 (비교 기준) ===")
        reduced = sweep(panel, NULL_THRESHOLDS, NULL_TOP_KS, LEG_VALUES, CAP_VALUES, start_day)
        payload["real_reduced"] = reduced
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        print(f"  최고 연환산 {reduced['annualised']:+.2%} · 평가 {reduced['evaluated']:,}조합")
        print("\n다음: --stage null --replications 20")
        return 0

    if args.stage == "null":
        if "real_reduced" not in payload:
            print("먼저 --stage real")
            return 1
        print(f"=== 경험적 귀무분포 (신호를 일자별로 종목간 순열) ===")
        print(f"실제(동일 조건) 최고 {payload['real_reduced']['annualised']:+.2%}")
        print(f"복제 {args.replications}회 · 회당 {len(NULL_THRESHOLDS)*len(NULL_TOP_KS)*len(LEG_VALUES)*len(CAP_VALUES):,}조합 전수\n")
        for index in range(args.replications):
            rng = np.random.default_rng(20260731 + args.seed_base + index)
            shuffled = permute_signal(panel, rng)
            best = sweep(shuffled, NULL_THRESHOLDS, NULL_TOP_KS, LEG_VALUES, CAP_VALUES, start_day)
            payload["null_best"].append(best["annualised"])
            path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
            print(f"  복제 {args.seed_base+index:3d}: 최고 {best['annualised']:+.2%} "
                  f"(k{best['top_k']} 임계{best['threshold']:.0%})", flush=True)
        print(f"\n  누적 복제 {len(payload['null_best'])}회 · {time.time()-started:.0f}s")
        print("다음: --stage judge (복제 20회 이상 권장)")
        return 0

    # judge
    if "real_reduced" not in payload or len(payload["null_best"]) < 5:
        print("real 단계와 null 복제 5회 이상이 필요하다")
        return 1
    real = payload["real_reduced"]["annualised"]
    real_full = payload["real_full"]["annualised"]
    null = np.array(payload["null_best"], dtype=float)
    exceed = int((null >= real).sum())
    p_value = (exceed + 1) / (len(null) + 1)  # 보수적(+1) 추정

    print(f"=== wave53 판정 ===")
    print(f"  전수 탐색(전체 격자) 최고 {real_full:+.2%} · {payload['real_full']['evaluated']:,}조합")
    print(f"  귀무와 동일 조건 실제 최고 {real:+.2%}")
    print(f"\n  귀무분포 ({len(null)}복제, 각 복제가 동일 전수 탐색):")
    print(f"    중앙 {np.median(null):+.2%} · 평균 {null.mean():+.2%} · 표준편차 {null.std():.2%}")
    print(f"    최대 {null.max():+.2%} · p90 {np.percentile(null,90):+.2%} · p95 {np.percentile(null,95):+.2%}")
    print(f"\n  실제 >= 귀무 최고인 복제 수 {exceed}/{len(null)} -> **p = {p_value:.3f}**")
    if p_value <= 0.05:
        print("  => 실제 신호가 귀무분포를 유의하게 넘는다. 캐리 edge는 탐색 산물이 아니다.")
    else:
        print("  => 실제 신호가 귀무분포와 구별되지 않는다. 무의미한 신호를 같은 강도로 탐색해도")
        print("     비슷한 성과가 나온다는 뜻이며, 캐리 결과는 데이터 스누핑으로 설명된다.")
    payload["p_value"] = p_value
    payload["null_summary"] = {"n": len(null), "median": float(np.median(null)),
                               "max": float(null.max()), "p95": float(np.percentile(null, 95))}
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\nresults/final.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
