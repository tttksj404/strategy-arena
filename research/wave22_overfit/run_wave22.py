#!/usr/bin/env python
"""Wave-22 (G1 overfitting-vs-edge audit) pipeline CLI. `--stage evaluate` builds the market
cache once (cache-only, no network -- same fitness.build_market_cache() every prior wave uses)
and runs all 6 validations against the frozen G1/I5 genomes (research/wave22_overfit/genomes.py),
writing one results/*.json per validation plus results/verdict.json. `--stage report` re-writes
report/wave22_report.md + REGISTRY.md from whatever is already in results/ (no recompute).
`--stage all` (default) does both. Mirrors research/wave21_ga/run_wave21.py's own stage
convention.
"""

from __future__ import annotations

import argparse
from enum import StrEnum
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Final, assert_never

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

from research.wave21_ga import fitness
from research.wave22_overfit import attribution, dsr, regime, rolling, sensitivity, shuffle_control, verdict
from research.wave22_overfit.evaluate import MetricsCache, full_equity
from research.wave22_overfit.genomes import G1_GENOME, I5_GENOME
from research.wave22_overfit.reporting22 import write_wave22_report

BASE_DIR: Final = Path(__file__).resolve().parent
RESULTS_DIR: Final = BASE_DIR / "results"
REPORT_DIR: Final = BASE_DIR / "report"
REGISTRY_PATH: Final = BASE_DIR / "REGISTRY.md"


class Stage(StrEnum):
    EVALUATE = "evaluate"
    REPORT = "report"
    ALL = "all"


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (bool, int, str)) or value is None:
        return value
    return str(value)  # last-resort: e.g. numpy scalar types that slip through


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _stage_evaluate() -> None:
    started_total = time.time()
    print("evaluate: building market cache (cache-only)...", flush=True)
    t0 = time.time()
    cache = fitness.build_market_cache()
    print(f"evaluate: cache built in {time.time() - t0:.1f}s, n_days={len(cache.index)}, n_symbols={len(cache.symbols)}, range={cache.index[0]}..{cache.index[-1]}", flush=True)

    metrics_cache = MetricsCache()

    t0 = time.time()
    g1_equity = full_equity(G1_GENOME, cache)
    i5_equity = full_equity(I5_GENOME, cache)
    print(f"evaluate: G1/I5 full equity curves computed in {time.time() - t0:.2f}s", flush=True)

    t0 = time.time()
    sensitivity_result = sensitivity.run(cache, G1_GENOME, metrics_cache)
    print(f"evaluate: [1/6] sensitivity done in {time.time() - t0:.2f}s (cache size={len(metrics_cache)}, hits={metrics_cache.hits}, misses={metrics_cache.misses})", flush=True)
    _save_json(RESULTS_DIR / "sensitivity.json", sensitivity_result)

    t0 = time.time()
    rolling_result = rolling.run(g1_equity, i5_equity)
    print(f"evaluate: [2/6] rolling done in {time.time() - t0:.2f}s", flush=True)
    _save_json(RESULTS_DIR / "rolling.json", rolling_result)

    t0 = time.time()
    regime_result = regime.run(g1_equity, i5_equity)
    print(f"evaluate: [3/6] regime done in {time.time() - t0:.2f}s", flush=True)
    _save_json(RESULTS_DIR / "regime.json", regime_result)

    t0 = time.time()
    dsr_result = dsr.run(g1_equity, cache)
    print(f"evaluate: [4/6] dsr done in {time.time() - t0:.2f}s", flush=True)
    _save_json(RESULTS_DIR / "dsr.json", dsr_result)

    t0 = time.time()
    attribution_result = attribution.run(cache, metrics_cache)
    print(f"evaluate: [5/6] attribution done in {time.time() - t0:.2f}s (cache size={len(metrics_cache)}, hits={metrics_cache.hits}, misses={metrics_cache.misses})", flush=True)
    _save_json(RESULTS_DIR / "attribution.json", attribution_result)

    t0 = time.time()
    shuffle_result = shuffle_control.run(cache, metrics_cache)
    print(f"evaluate: [6/6] shuffle_control done in {time.time() - t0:.2f}s (draws attempted={shuffle_result['methodology']['n_draws_attempted']})", flush=True)
    _save_json(RESULTS_DIR / "shuffle_control.json", shuffle_result)

    verdict_result = verdict.combine(sensitivity_result, rolling_result, regime_result, dsr_result, attribution_result, shuffle_result)
    _save_json(RESULTS_DIR / "verdict.json", verdict_result)

    print(f"evaluate: total genome evaluations = {len(metrics_cache)} unique ({metrics_cache.hits} cache hits, {metrics_cache.misses} fresh backtests)", flush=True)
    print(f"evaluate: OVERALL VERDICT = {verdict_result['overall']}", flush=True)
    print(f"evaluate: done in {time.time() - started_total:.1f}s total", flush=True)


def _stage_report() -> None:
    write_wave22_report(RESULTS_DIR, REPORT_DIR, REGISTRY_PATH)
    print(f"report: wrote {REPORT_DIR / 'wave22_report.md'} and {REGISTRY_PATH}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Wave-22 (G1 overfitting-vs-edge audit) pipeline")
    parser.add_argument("--stage", type=Stage, choices=tuple(Stage), default=Stage.ALL)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        match args.stage:
            case Stage.EVALUATE:
                _stage_evaluate()
            case Stage.REPORT:
                _stage_report()
            case Stage.ALL:
                _stage_evaluate()
                _stage_report()
            case unreachable:
                assert_never(unreachable)
    except (FileNotFoundError, RuntimeError, ValueError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
