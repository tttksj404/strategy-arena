# Wave-31 post-hoc analysis. The gate table says the candidate failed on ruin probability; the
# useful output is the FRONTIER -- for each holding window and for each risk band, what return
# was reachable and what it cost. Plus one question the gates cannot answer on their own: does a
# LOWER-risk archive entry exist that would have passed the ruin gate, and what does it give up?
#
# Everything here is IS-only except the candidate's own already-unsealed OOS profile (recorded
# once in results/final.json). Archive entries are re-run with mode='is' so no additional OOS
# unsealing happens.

from __future__ import annotations

import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

import numpy as np

from research.wave30_qd.dataio30 import build_market_cache
from research.wave30_qd.engine30 import TOTAL_CAPITAL, run_genome
from research.wave30_qd.fitness30 import LEVERAGE_EDGES, bootstrap_wipe_probability
from research.wave30_qd.gates30 import _daily_returns
from research.wave30_qd.run_wave30 import _genome_from_dict
from research.wave31_sprint.fitness31 import FITNESS_WINDOW, HALVING_EDGES, WINDOWS, sprint_profile
from research.wave31_sprint.gates31 import (
    Q3_MAX_PROB_HALVING,
    Q3_MAX_WIPE_PROBABILITY,
    Q4_MAX_RUIN_PROBABILITY,
    Q4_RUIN_FLOOR_USDT,
)
from research.wave31_sprint.run_wave31 import RESULTS_DIR, SEEDS

MC_PATHS = 10_000


def is_ruin_probability(curve: np.ndarray, seed: int = 31_777) -> float:
    """Q4's statistic computed on the IS curve only, so archive entries can be screened without
    unsealing OOS."""
    rng = np.random.default_rng(seed)
    daily = _daily_returns(curve)
    if len(daily) < 3:
        return 1.0
    draws = rng.integers(0, len(daily), size=(MC_PATHS, len(daily)))
    finals = (TOTAL_CAPITAL * np.cumprod(1.0 + daily[draws], axis=1))[:, -1]
    return float((finals < Q4_RUIN_FLOOR_USDT).mean())


def band_label(edges, index, cap=None) -> str:
    high = edges[index + 1] if cap is None else min(edges[index + 1], cap)
    return f"{edges[index]:g}-{high:g}"


def main() -> int:
    cache = build_market_cache()
    final = json.loads((RESULTS_DIR / "final.json").read_text(encoding="utf-8"))

    print("=" * 112)
    print("1. SPRINT FRONTIER -- judged candidate vs I5 baseline, by holding window (full span)")
    print("=" * 112)
    print(f"{'window':>8} | {'candidate p50':>13} {'p95 (best case)':>16} {'p05':>9} {'pos%':>6} "
          f"{'P(-50%)':>8} | {'I5 p50':>8} {'I5 p95':>8}")
    print("-" * 112)
    cand = final["candidate_sprint_profiles"]["full"]["windows"]
    base = final["baseline_sprint_profiles"]["full"]["windows"]
    for w in WINDOWS:
        c, b = cand[str(w)], base[str(w)]
        print(f"{w:6d}d | {c['p50']*100:12.2f}% {c['p95']*100:15.2f}% {c['p05']*100:8.2f}% "
              f"{c['positive_share']*100:5.1f}% {c['prob_loss_over_50']:8.4f} | "
              f"{b['p50']*100:7.2f}% {b['p95']*100:7.2f}%")
    print()
    fp = final["candidate_sprint_profiles"]["full"]
    bp = final["baseline_sprint_profiles"]["full"]
    print(f"   시간까지의 속도  candidate: 2x {fp['days_to_2x']}일 · 5x {fp['days_to_5x']}일 · 10x {fp['days_to_10x']}일")
    print(f"                    I5:        2x {bp['days_to_2x']} · 5x {bp['days_to_5x']} · 10x {bp['days_to_10x']}")

    print()
    print("=" * 112)
    print("2. IS vs OOS of the judged candidate (the one sealed unsealing)")
    print("=" * 112)
    for span in ("is", "oos"):
        c = final["candidate_sprint_profiles"][span]["windows"][str(FITNESS_WINDOW)]
        b = final["baseline_sprint_profiles"][span]["windows"][str(FITNESS_WINDOW)]
        print(f"   {span.upper():4s} 30일창 중앙값 {c['p50']*100:+7.2f}% (I5 {b['p50']*100:+5.2f}%) | "
              f"p95 {c['p95']*100:+8.2f}% | 양수비율 {c['positive_share']*100:5.1f}% | "
              f"P(-50%) {c['prob_loss_over_50']:.4f} | 창수 {c['n_windows']}")

    archive: list[dict] = []
    for seed in SEEDS:
        payload = json.loads((RESULTS_DIR / f"seed_{seed}.json").read_text(encoding="utf-8"))
        for item in payload["archive"] + payload["pareto_front"]:
            archive.append(item)
    unique = {json.dumps(item["genome"], sort_keys=True): item for item in archive}
    archive = list(unique.values())

    print()
    print("=" * 112)
    print(f"3. ARCHIVE BY 30-DAY HALVING RISK -- 'how much median 30d return can I buy at each risk level'")
    print(f"   (union of 5 seeds, {len(archive)} unique genomes, IS only)")
    print("=" * 112)
    print(f"{'P(30d -50%) band':>18} {'genomes':>8} {'best p50-30d':>13} {'그 후보 p95':>12} "
          f"{'lev':>6} {'sleeve%':>8} {'wipeP':>7} {'trades':>7} {'family':>10}")
    print("-" * 112)
    by_band: dict[int, list[dict]] = {}
    for item in archive:
        by_band.setdefault(item["descriptor"][1], []).append(item)
    for band in range(len(HALVING_EDGES) - 1):
        rows = by_band.get(band, [])
        if not rows:
            continue
        best = max(rows, key=lambda r: r["fitness"])
        focus = best["sprint"]["windows"][str(FITNESS_WINDOW)]
        print(f"{band_label(HALVING_EDGES, band, 1.0):>18} {len(rows):8d} {best['fitness']*100:12.2f}% "
              f"{focus['p95']*100:11.2f}% {best['mean_leverage']:6.2f} {best['genome']['sleeve_fraction']*100:7.0f}% "
              f"{best['wipe_probability']:7.3f} {best['n_trades']:7d} {best['genome']['signal_family']:>10}")

    print()
    print("=" * 112)
    print("4. ARCHIVE BY LEVERAGE BAND (sprint objective)")
    print("=" * 112)
    print(f"{'lev band':>10} {'genomes':>8} {'best p50-30d':>13} {'P(-50%)':>9} {'wipeP':>7} {'sleeveMDD':>10}")
    print("-" * 112)
    by_lev: dict[int, list[dict]] = {}
    for item in archive:
        by_lev.setdefault(item["descriptor"][0], []).append(item)
    for band in range(len(LEVERAGE_EDGES) - 1):
        rows = by_lev.get(band, [])
        if not rows:
            continue
        best = max(rows, key=lambda r: r["fitness"])
        print(f"{band_label(LEVERAGE_EDGES, band, 20.0)+'x':>10} {len(rows):8d} {best['fitness']*100:12.2f}% "
              f"{best['prob_halving_30d']:9.4f} {best['wipe_probability']:7.3f} {best['sleeve_mdd']*100:9.1f}%")

    print()
    print("=" * 112)
    print("5. DOES A LOWER-RISK ARCHIVE ENTRY EXIST THAT WOULD PASS THE RUIN GATE?")
    print(f"   screen: IS P(30d -50%) < {Q3_MAX_PROB_HALVING} AND wipe < {Q3_MAX_WIPE_PROBABILITY} "
          f"AND trades >= 100, then re-run on IS and compute Q4's ruin probability (< {Q4_MAX_RUIN_PROBABILITY})")
    print("=" * 112)
    shortlist = [
        item for item in archive
        if item["prob_halving_30d"] < Q3_MAX_PROB_HALVING
        and item["wipe_probability"] < Q3_MAX_WIPE_PROBABILITY
        and item["n_trades"] >= 100
    ]
    shortlist.sort(key=lambda r: -r["fitness"])
    print(f"   shortlist before ruin test: {len(shortlist)} genomes; testing top 60")
    rng_seed = 31_777
    survivors: list[dict] = []
    for item in shortlist[:60]:
        genome = _genome_from_dict(item["genome"])
        result = run_genome(cache, genome, mode="is")
        curve = result.total_equity_daily[result.daily_valid]
        ruin = is_ruin_probability(curve, rng_seed)
        item["_is_ruin"] = ruin
        item["_is_mdd"] = float(abs(np.min((curve - np.maximum.accumulate(curve)) / np.maximum.accumulate(curve))))
        item["_is_final"] = float(curve[-1])
        if ruin < Q4_MAX_RUIN_PROBABILITY:
            survivors.append(item)
    print(f"   passing the ruin gate on IS: {len(survivors)} of {min(60, len(shortlist))} tested")
    print()
    if survivors:
        print(f"{'#':>2} {'p50-30d':>8} {'p95-30d':>9} {'IS ruin':>8} {'IS MDD':>8} {'IS final$':>11} "
              f"{'lev':>6} {'slv%':>5} {'P(-50%)':>8} {'wipeP':>7} {'tr':>5} {'family':>10} {'stop%':>6} {'R':>5} {'hold':>5}")
        for i, item in enumerate(sorted(survivors, key=lambda r: -r["fitness"])[:10], 1):
            g = item["genome"]
            focus = item["sprint"]["windows"][str(FITNESS_WINDOW)]
            print(f"{i:2d} {item['fitness']*100:7.2f}% {focus['p95']*100:8.2f}% {item['_is_ruin']:8.4f} "
                  f"{item['_is_mdd']*100:7.1f}% {item['_is_final']:10,.0f} {item['mean_leverage']:6.2f} "
                  f"{g['sleeve_fraction']*100:4.0f}% {item['prob_halving_30d']:8.4f} {item['wipe_probability']:7.3f} "
                  f"{item['n_trades']:5d} {g['signal_family']:>10} {g['stop_pct']*100:6.2f} {g['target_r']:5.2f} {g['max_hold_bars']:5d}")
        print()
        best = max(survivors, key=lambda r: r["fitness"])
        judged = final["candidate"]
        print(f"   대가: 판정후보 p50-30d {judged['fitness']*100:.2f}% (파산 22.52%) "
              f"→ 최저위험 대안 {best['fitness']*100:.2f}% (IS 파산 {best['_is_ruin']*100:.2f}%)")
        print(f"   즉 파산확률을 {22.52:.1f}% → {best['_is_ruin']*100:.2f}% 로 낮추는 비용 = 30일 중앙수익 "
              f"{judged['fitness']*100:.2f}% → {best['fitness']*100:.2f}%")
    else:
        print("   NONE. 이 탐색공간에서 파산확률 5% 미만이면서 단기수익이 I5를 넘는 개체는 없다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
