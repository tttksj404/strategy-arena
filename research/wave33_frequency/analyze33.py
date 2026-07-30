# Wave-33 frontier: what per-entry dollars are actually reachable at each entry frequency, and
# what capital would be required for the average entry to net $10.
#
# The judged candidate is only one point (SPEC.md's median-seed rule deliberately refuses to
# report the luckiest seed as the result). The frontier below is the actual deliverable: it says
# where the request breaks and by how much.
#
# IS only, except the judged candidate's already-recorded OOS numbers. Archive genomes are never
# re-run past OOS_SPLIT.

from __future__ import annotations

import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

import numpy as np

from research.wave30_qd.fitness30 import LEVERAGE_EDGES
from research.wave33_frequency.fitness33 import FREQUENCY_EDGES, TARGET_PER_ENTRY_USDT
from research.wave33_frequency.run_wave33 import RESULTS_DIR, SEEDS


def load_archive() -> list[dict]:
    rows: list[dict] = []
    for seed in SEEDS:
        payload = json.loads((RESULTS_DIR / f"seed_{seed}.json").read_text(encoding="utf-8"))
        for item in payload["archive"] + payload["pareto_front"]:
            rows.append({**item, "seed": seed})
    unique = {json.dumps(item["genome"], sort_keys=True): item for item in rows}
    return list(unique.values())


def band(edges, index, cap=None) -> str:
    high = edges[index + 1] if cap is None else min(edges[index + 1], cap)
    return f"{edges[index]:g}-{'inf' if high == np.inf else f'{high:g}'}"


def capital_for_ev(mean_usdt: float, base_usdt: float = 100.0) -> float:
    if mean_usdt <= 0:
        return float("inf")
    return TARGET_PER_ENTRY_USDT / mean_usdt * base_usdt


def main() -> int:
    archive = load_archive()
    final = json.loads((RESULTS_DIR / "final.json").read_text(encoding="utf-8"))
    feasible = [item for item in archive if not item["infeasible_reasons"]]

    print("=" * 116)
    print("1. 요구 조건 충족 여부 — 하루 1회 이상 진입 ∧ 계좌 생존")
    print("=" * 116)
    print(f"   탐색한 유니크 유전자: {len(archive)}")
    print(f"   요구 조건(≥1회/활동일 ∧ 전기간 생존) 충족: {len(feasible)} ({len(feasible)/len(archive):.1%})")
    if feasible:
        freqs = np.array([f["entry_profile"]["trades_per_active_day"] for f in feasible])
        print(f"   충족 개체의 진입 빈도: 최소 {freqs.min():.2f} · 중앙 {np.median(freqs):.2f} · 최대 {freqs.max():.2f} 회/활동일")
        print(f"   → 빈도 요구 자체는 달성 가능하다.")

    print()
    print("=" * 116)
    print("2. 진입당 달러 — 요구는 +$10.00 (고정 $100 base)")
    print("=" * 116)
    if feasible:
        med = np.array([f["entry_profile"]["median_usdt"] for f in feasible])
        mean = np.array([f["entry_profile"]["mean_usdt"] for f in feasible])
        print(f"{'':22} {'최소':>10} {'중앙':>10} {'최대':>10}   목표 {TARGET_PER_ENTRY_USDT:+.2f}")
        print(f"{'진입당 중앙값($)':22} {med.min():+10.3f} {np.median(med):+10.3f} {med.max():+10.3f}")
        print(f"{'진입당 평균($)':22} {mean.min():+10.3f} {np.median(mean):+10.3f} {mean.max():+10.3f}")
        n_ge = int((med >= TARGET_PER_ENTRY_USDT).sum())
        print(f"\n   진입당 중앙값이 +$10 이상인 개체: **{n_ge} / {len(feasible)}**")
        print(f"   달성된 최대 진입당 중앙값: **${med.max():+.3f}** (목표의 {med.max()/TARGET_PER_ENTRY_USDT:.1%})")
        best = max(feasible, key=lambda f: f["entry_profile"]["median_usdt"])
        p = best["entry_profile"]
        g = best["genome"]
        print(f"\n   최고 개체 (seed {best['seed']}, IS 전용·OOS 미개봉):")
        print(f"     {g['signal_family']} lb{g['lookback_bars']} stop{g['stop_pct']*100:.2f}% R{g['target_r']:.2f} "
              f"lev{best['mean_leverage']:.2f}x hold{g['max_hold_bars']} conc{g['max_concurrent']} "
              f"sym{len(g['symbols'])} sleeve{g['sleeve_fraction']*100:.0f}%")
        print(f"     진입당: 중앙값 ${p['median_usdt']:+.3f} · 평균 ${p['mean_usdt']:+.3f} · "
              f"p95 ${p['p95_usdt']:+.2f} · 최고 ${p['best_usdt']:+.2f} · 최악 ${p['worst_usdt']:+.2f}")
        print(f"     빈도 {p['trades_per_active_day']:.2f}회/활동일 ({p['n_trades']}거래, {p['active_days']:.0f}일) · "
              f"승률 {p['win_share']:.1%}")
        print(f"     P(≥+$10) {p['share_ge_target']:.2%} · P(≤−$10) {p['share_le_negative_target']:.2%}")
        print(f"     계좌 $100 → ${p['account_final_usdt']:,.2f} · MDD {p['account_mdd']:.1%} · "
              f"누적 ${p['total_usdt']:+,.2f}")

    print()
    print("=" * 116)
    print("3. 빈도별 프론티어 — 각 진입빈도에서 달성 가능한 최고 진입당 달러 (IS)")
    print("=" * 116)
    print(f"{'회/활동일':>12} {'개체':>6} {'최고 중앙값$':>13} {'그 개체 평균$':>14} {'P(≥+$10)':>10} "
          f"{'lev':>6} {'거래':>7} {'계좌최종$':>11} {'MDD':>7} {'$10 EV 필요자본':>18}")
    print("-" * 116)
    by_freq: dict[int, list[dict]] = {}
    for item in archive:
        by_freq.setdefault(item["descriptor"][0], []).append(item)
    for index in range(len(FREQUENCY_EDGES) - 1):
        rows = by_freq.get(index, [])
        if not rows:
            continue
        best = max(rows, key=lambda r: r["entry_profile"]["median_usdt"])
        p = best["entry_profile"]
        need = capital_for_ev(p["mean_usdt"])
        need_text = f"${need:,.0f}" if np.isfinite(need) else "도달 불가"
        print(f"{band(FREQUENCY_EDGES, index):>12} {len(rows):6d} {p['median_usdt']:+12.3f} {p['mean_usdt']:+13.3f} "
              f"{p['share_ge_target']:9.2%} {best['mean_leverage']:6.2f} {p['n_trades']:7d} "
              f"{p['account_final_usdt']:10,.0f} {p['account_mdd']:6.1%} {need_text:>18}")

    print()
    print("=" * 116)
    print("4. 레버리지별 — 진입당 달러를 레버리지로 키울 수 있는가 (요구 조건 충족 개체만)")
    print("=" * 116)
    print(f"{'lev band':>10} {'개체':>6} {'최고 중앙값$':>13} {'평균$':>10} {'P(≥+$10)':>10} {'P(≤−$10)':>10} {'MDD':>8}")
    print("-" * 116)
    by_lev: dict[int, list[dict]] = {}
    for item in feasible:
        by_lev.setdefault(item["descriptor"][1], []).append(item)
    for index in range(len(LEVERAGE_EDGES) - 1):
        rows = by_lev.get(index, [])
        if not rows:
            continue
        best = max(rows, key=lambda r: r["entry_profile"]["median_usdt"])
        p = best["entry_profile"]
        print(f"{band(LEVERAGE_EDGES, index, 20.0)+'x':>10} {len(rows):6d} {p['median_usdt']:+12.3f} "
              f"{p['mean_usdt']:+9.3f} {p['share_ge_target']:9.2%} {p['share_le_negative_target']:9.2%} {p['account_mdd']:7.1%}")

    print()
    print("=" * 116)
    print("5. 진입당 기대값 $10을 만들려면 자본이 얼마여야 하는가")
    print("=" * 116)
    print("   진입당 손익은 포지션 base에 정비례하므로 $100에서 평균 $m이면 $10을 위해 필요한 자본 = 10/m × $100.")
    print("   (기대값이 0 이하면 자본을 키워도 손실만 비례해서 커진다 — 도달 불가.)")
    print()
    positives = [f for f in feasible if f["entry_profile"]["mean_usdt"] > 0]
    print(f"   요구 조건 충족 개체 중 진입당 기대값이 양수인 것: {len(positives)} / {len(feasible)}")
    if positives:
        positives.sort(key=lambda f: -f["entry_profile"]["mean_usdt"])
        print()
        print(f"{'#':>2} {'평균$/진입':>11} {'중앙값$':>10} {'회/일':>7} {'거래':>7} {'lev':>6} "
              f"{'$10 EV 필요자본':>18} {'그때 하루 기대':>15} {'계좌최종$':>11} {'MDD':>7}")
        for rank, item in enumerate(positives[:10], 1):
            p = item["entry_profile"]
            need = capital_for_ev(p["mean_usdt"])
            daily = TARGET_PER_ENTRY_USDT * p["trades_per_active_day"]
            print(f"{rank:2d} {p['mean_usdt']:+10.4f} {p['median_usdt']:+9.3f} {p['trades_per_active_day']:6.2f} "
                  f"{p['n_trades']:7d} {item['mean_leverage']:6.2f} ${need:16,.0f} ${daily:14,.2f} "
                  f"{p['account_final_usdt']:10,.0f} {p['account_mdd']:6.1%}")
        best = positives[0]
        need = capital_for_ev(best["entry_profile"]["mean_usdt"])
        print()
        print(f"   → 가장 좋은 개체 기준: 진입당 기대값 $10에 **약 ${need:,.0f}** 필요 "
              f"(현재 $100의 {need/100:,.0f}배)")

    print()
    print("=" * 116)
    print("6. 판정 후보 (SPEC 중앙값-시드 규칙) 와 OOS")
    print("=" * 116)
    ip = final["candidate"]["entry_profile"]
    oos = final["entry_profiles"]["oos"]
    print(f"   IS  진입당 중앙값 ${ip['median_usdt']:+.4f} · 평균 ${ip['mean_usdt']:+.4f} · "
          f"{ip['trades_per_active_day']:.2f}회/활동일 · {ip['n_trades']}거래")
    print(f"   OOS 진입당 중앙값 ${oos['median_usdt']:+.4f} · 평균 ${oos['mean_usdt']:+.4f} · {oos['n_trades']}거래")
    print(f"   게이트: " + " ".join(f"{k.split('_')[0]}={v['status']}" for k, v in final["gates"].items()))
    print(f"   종합: {final['overall']} (FAIL: {final['failure_reasons']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
