# Wave-30 post-hoc analysis. Three questions the gate table alone does not answer:
#
#   1. What does the ILLUMINATED MAP actually say? The archive, not the winner, is this wave's
#      deliverable: for each leverage band, what return was reachable and at what risk.
#   2. Is the median-seed candidate's IS figure ($100 -> $230k) even a real number? The cost
#      model was fitted to $45 order-book walks; if the sleeve compounds into six figures the
#      backtest is quoting $45 slippage on a six-figure order.
#   3. Where did the return come from -- an edge, or a handful of compounding wins?
#
# Nothing here touches OOS beyond the single unsealing already recorded in results/final.json.

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
from research.wave30_qd.engine30 import run_genome
from research.wave30_qd.fitness30 import LEVERAGE_EDGES, MDD_EDGES, baseline_reference
from research.wave30_qd.run_wave30 import RESULTS_DIR, SEEDS, _genome_from_dict

# IS-only plausibility screen. These are the two gate thresholds that can be checked WITHOUT
# unsealing OOS (P4's drawdown ceiling and P3's wipe ceiling), applied to the IS run.
IS_MDD_CEILING = 0.35
IS_WIPE_CEILING = 0.05


def load_union_archive() -> list[dict]:
    rows: list[dict] = []
    for seed in SEEDS:
        payload = json.loads((RESULTS_DIR / f"seed_{seed}.json").read_text(encoding="utf-8"))
        for item in payload["archive"]:
            rows.append({**item, "seed": seed})
    return rows


def leverage_band_label(index: int) -> str:
    return f"{LEVERAGE_EDGES[index]:.0f}-{min(LEVERAGE_EDGES[index + 1], 20.0):.0f}x"


def main() -> int:
    cache = build_market_cache()
    baseline = baseline_reference(cache)
    archive = load_union_archive()
    final = json.loads((RESULTS_DIR / "final.json").read_text(encoding="utf-8"))

    print("=" * 108)
    print("1. ILLUMINATED MAP -- best IS result reachable in each leverage band (union of 5 seeds)")
    print(f"   I5 baseline IS walk-forward fitness = {baseline['is_fitness']:.4f} "
          f"(IS CAGR {baseline['is_cagr']*100:.2f}%, IS MDD {baseline['is_mdd']*100:.2f}%)")
    print("=" * 108)
    header = (f"{'lev band':>9} {'cells':>6} {'genomes':>8} {'best fit':>9} {'IS CAGR':>10} "
              f"{'sleeveMDD':>10} {'wipeP':>7} {'trades':>7} {'sleeve%':>8} {'family':>17}")
    print(header)
    print("-" * 108)
    by_band: dict[int, list[dict]] = {}
    for row in archive:
        by_band.setdefault(row["descriptor"][0], []).append(row)
    for band in range(len(LEVERAGE_EDGES) - 1):
        rows = by_band.get(band, [])
        if not rows:
            print(f"{leverage_band_label(band):>9} {'0':>6} {'0':>8}   (empty)")
            continue
        best = max(rows, key=lambda item: item["fitness"])
        cells = len({tuple(item["descriptor"]) for item in rows})
        print(f"{leverage_band_label(band):>9} {cells:6d} {len(rows):8d} {best['fitness']:9.4f} "
              f"{best['is_total_cagr']*100:9.2f}% {best['sleeve_mdd']*100:9.1f}% {best['wipe_probability']:7.3f} "
              f"{best['n_trades']:7d} {best['genome']['sleeve_fraction']*100:7.0f}% "
              f"{best['genome']['signal_family']:>17}")

    print()
    print("=" * 108)
    print("2. IS-ONLY PLAUSIBILITY SCREEN -- how many archive entries clear the two risk ceilings")
    print(f"   that can be tested without unsealing OOS: sleeve MDD <= {IS_MDD_CEILING:.0%} AND "
          f"wipe prob < {IS_WIPE_CEILING:.0%} AND IS fitness > I5's {baseline['is_fitness']:.4f}")
    print("=" * 108)
    print(f"{'lev band':>9} {'entries':>8} {'MDD ok':>8} {'wipe ok':>8} {'beats I5':>9} {'ALL THREE':>11} {'best surviving fit':>19}")
    print("-" * 108)
    survivor_total = 0
    survivors_by_band: dict[int, list[dict]] = {}
    for band in range(len(LEVERAGE_EDGES) - 1):
        rows = by_band.get(band, [])
        if not rows:
            continue
        mdd_ok = [r for r in rows if r["sleeve_mdd"] <= IS_MDD_CEILING]
        wipe_ok = [r for r in rows if r["wipe_probability"] < IS_WIPE_CEILING]
        beats = [r for r in rows if r["fitness"] > baseline["is_fitness"]]
        survivors = [
            r for r in rows
            if r["sleeve_mdd"] <= IS_MDD_CEILING
            and r["wipe_probability"] < IS_WIPE_CEILING
            and r["fitness"] > baseline["is_fitness"]
        ]
        survivors_by_band[band] = survivors
        survivor_total += len(survivors)
        best_text = f"{max(s['fitness'] for s in survivors):.4f}" if survivors else "-"
        print(f"{leverage_band_label(band):>9} {len(rows):8d} {len(mdd_ok):8d} {len(wipe_ok):8d} "
              f"{len(beats):9d} {len(survivors):11d} {best_text:>19}")
    print("-" * 108)
    print(f"   total entries clearing all three IS screens: {survivor_total} of {len(archive)}")

    print()
    print("=" * 108)
    print("3. CAPACITY REALITY CHECK on the judged candidate")
    print("=" * 108)
    genome = _genome_from_dict(final["candidate"]["genome"])
    result = run_genome(cache, genome, mode="full")
    notionals = np.array([t.notional_usdt for t in result.trades])
    returns = np.array([t.net_return_on_base for t in result.trades])
    print(f"   trades {len(result.trades)} | wins {int((returns>0).sum())} "
          f"({(returns>0).mean()*100:.1f}%) | liquidations {result.n_liquidations}")
    print(f"   notional per trade: min ${notionals.min():,.0f}  median ${np.median(notionals):,.0f}  "
          f"max ${notionals.max():,.0f}")
    print(f"   the measured slippage mapping was fitted to $45 Bitget order-book walks and is")
    print(f"   applied UNCHANGED at every one of these sizes -- the largest is "
          f"{notionals.max()/45:,.0f}x the fitted order size.")
    order = np.argsort(-returns)
    top5 = returns[order[:5]]
    contribution = np.log1p(np.clip(returns, -0.999999, None))
    total_log = contribution.sum()
    top5_log = contribution[order[:5]].sum()
    print(f"   top 5 trade returns: {', '.join(f'{x*100:+.1f}%' for x in top5)}")
    print(f"   those 5 trades supply {top5_log/total_log*100:.1f}% of the total log-growth "
          f"({len(result.trades)} trades in {final['final_evaluation']['full']['days']/365:.1f} years)")
    print(f"   worst 5: {', '.join(f'{x*100:+.1f}%' for x in returns[order[-5:]])}")

    print()
    print("=" * 108)
    print("4. WHAT THE PARETO FRONTS CHOSE (union across seeds, IS only)")
    print("=" * 108)
    pareto: list[dict] = []
    for seed in SEEDS:
        payload = json.loads((RESULTS_DIR / f"seed_{seed}.json").read_text(encoding="utf-8"))
        pareto.extend(payload["pareto_front"])
    families: dict[str, int] = {}
    for item in pareto:
        families[item["genome"]["signal_family"]] = families.get(item["genome"]["signal_family"], 0) + 1
    print(f"   front members: {len(pareto)} | family mix: "
          + ", ".join(f"{k} {v}" for k, v in sorted(families.items(), key=lambda kv: -kv[1])))
    low_risk = [p for p in pareto if p["sleeve_mdd"] <= IS_MDD_CEILING and p["wipe_probability"] < IS_WIPE_CEILING]
    print(f"   front members clearing both IS risk ceilings: {len(low_risk)}")
    if low_risk:
        best = max(low_risk, key=lambda item: item["fitness"])
        print(f"   best such member: fitness {best['fitness']:.4f} | IS CAGR {best['is_total_cagr']*100:.2f}% "
              f"| lev {best['mean_leverage']:.2f}x | sleeve {best['genome']['sleeve_fraction']*100:.0f}% "
              f"| MDD {best['sleeve_mdd']*100:.1f}% | {best['genome']['signal_family']}")
    print()
    print(f"   leverage distribution of all archive entries (union, {len(archive)} genomes):")
    levs = np.array([r["mean_leverage"] for r in archive])
    for q in (5, 25, 50, 75, 95, 100):
        print(f"      p{q:<3d} {np.percentile(levs, q):6.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
