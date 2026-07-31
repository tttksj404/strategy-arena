#!/usr/bin/env python3
# Wave-36 grid runner. Deterministic full enumeration of the 9,720 frozen combinations, sharded by
# lookback so each invocation finishes in ~12 minutes and is resumable.
#
# A full grid rather than a stochastic optimiser is a deliberate methodological choice: the space is
# small enough to enumerate, so there is no "did the optimiser overfit" ambiguity and the trial count
# is exactly 9,720 rather than an argument. wave34 already established that optimiser choice matters
# when the space is too large to enumerate; here it is not.
#
#   python run_wave36.py grid 7          # one lookback shard
#   python run_wave36.py status
#   python run_wave36.py judge           # selection + THE single OOS unsealing

from __future__ import annotations

import json
from pathlib import Path
import sys
import time
from typing import Any, Final

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

import numpy as np
import pandas as pd  # noqa: PANDAS_OK

from research.wave35_universe.dataio35 import build_wide_cache
from research.wave36_crosssection.engine36 import (
    CAPACITY_LIMIT_USDT,
    LEV_CAP,
    MIN_LEG_USDT,
    Config36,
    Result36,
    StampPanel,
    build_stamp_panel,
    run_config,
)

LOOKBACKS: Final = (3, 7, 14, 21, 30, 45)
K_VALUES: Final = (1, 2, 3, 5)
HOLD_BANDS: Final = (0.00, 0.25, 0.50)
LEVERAGES: Final = (1.0, 2.0, 3.0, 5.0, 8.0)
REBALANCE: Final = (1, 3, 9)
MIN_DISPERSION: Final = (0.00, 0.10, 0.20)
SLEEVE_FRACTIONS: Final = (0.25, 0.50, 1.00)
WF_FOLDS: Final = 4
TOTAL_COMBINATIONS: Final = (
    len(LOOKBACKS) * len(K_VALUES) * len(HOLD_BANDS) * len(LEVERAGES)
    * len(REBALANCE) * len(MIN_DISPERSION) * len(SLEEVE_FRACTIONS)
)
RESULTS_DIR: Final = Path(__file__).resolve().parent / "results"


def _annualised(curve: np.ndarray, stamps: pd.DatetimeIndex, start: int, end: int) -> float:
    if end <= start or curve[start] <= 0:
        return 0.0
    years = (stamps[end] - stamps[start]).days / 365.25
    if years <= 0:
        return 0.0
    ratio = curve[end] / curve[start]
    return float(ratio ** (1.0 / years) - 1.0) if ratio > 0 else -1.0


def _mdd(curve: np.ndarray) -> float:
    peak = np.maximum.accumulate(curve)
    return float(abs(np.min((curve - peak) / np.maximum(peak, 1e-12))))


def walk_forward_fitness(result: Result36, panel: StampPanel) -> tuple[float, list[float]]:
    """Median minus stdev of per-fold annualised return over IS (wave21 convention).

    Penalising dispersion across folds is what stops a configuration that worked in one regime and
    nowhere else from winning the grid.
    """
    n_is = int(panel.is_mask.sum())
    edges = np.linspace(0, n_is - 1, WF_FOLDS + 1).astype(int)
    folds = []
    for i in range(WF_FOLDS):
        start, end = int(edges[i]), int(edges[i + 1])
        if end > start:
            folds.append(_annualised(result.total_equity, panel.stamps, start, end))
    if not folds:
        return -1.0, []
    return float(np.median(folds) - np.std(folds, ddof=0)), folds


def evaluate(panel: StampPanel, config: Config36) -> dict[str, Any]:
    result = run_config(panel, config, is_only=True)
    n_is = int(panel.is_mask.sum())
    fitness, folds = walk_forward_fitness(result, panel)
    feasible_reasons: list[str] = []
    if result.margin_call:
        feasible_reasons.append("margin call")
    if np.isfinite(result.min_leg_notional) and result.min_leg_notional < MIN_LEG_USDT:
        feasible_reasons.append(f"leg notional ${result.min_leg_notional:.2f} < ${MIN_LEG_USDT}")
    if not np.isfinite(result.min_leg_notional):
        feasible_reasons.append("never opened a book")
    if result.max_leg_notional > CAPACITY_LIMIT_USDT:
        feasible_reasons.append("leg notional over capacity limit")
    return {
        "config": config.to_dict(),
        "fitness": fitness if not feasible_reasons else fitness - 100.0,
        "raw_fitness": fitness,
        "fold_returns": folds,
        "is_annualised": _annualised(result.total_equity, panel.stamps, 0, n_is - 1),
        "is_final_usdt": float(result.total_equity[n_is - 1]),
        "is_mdd": _mdd(result.total_equity[:n_is]),
        "funding_usdt": result.funding_collected_usdt,
        "price_pnl_usdt": result.price_pnl_usdt,
        "cost_usdt": result.cost_paid_usdt,
        "rotations": result.n_rotations,
        "margin_call": result.margin_call,
        "min_leg_notional": result.min_leg_notional,
        "max_leg_notional": result.max_leg_notional,
        "infeasible_reasons": feasible_reasons,
    }


def shard_path(lookback: int) -> Path:
    return RESULTS_DIR / f"grid_lookback_{lookback}.json"


def run_shard(lookback: int) -> dict[str, Any]:
    cache, _symbols = build_wide_cache()
    panel = build_stamp_panel(cache)
    started = time.time()
    rows: list[dict[str, Any]] = []
    for k in K_VALUES:
        for band in HOLD_BANDS:
            for leverage in LEVERAGES:
                for rebalance in REBALANCE:
                    for dispersion in MIN_DISPERSION:
                        for sleeve in SLEEVE_FRACTIONS:
                            config = Config36(
                                lookback_days=lookback,
                                k=k,
                                hold_band=band,
                                leverage=leverage,
                                rebalance_every_stamps=rebalance,
                                min_dispersion_apr=dispersion,
                                sleeve_fraction=sleeve,
                            )
                            rows.append(evaluate(panel, config))
    payload = {
        "lookback_days": lookback,
        "combinations": len(rows),
        "runtime_seconds": time.time() - started,
        "rows": rows,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    shard_path(lookback).write_text(json.dumps(payload), encoding="utf-8")
    feasible = [r for r in rows if not r["infeasible_reasons"]]
    best = max(feasible, key=lambda r: r["fitness"]) if feasible else None
    print(
        f"lookback {lookback:3d}d: {len(rows)} 조합 | 제약충족 {len(feasible)} | "
        f"최고 적합도 {best['fitness']:+.4f} (IS 연 {best['is_annualised']:+.2%})"
        if best
        else f"lookback {lookback:3d}d: 제약충족 조합 없음",
        flush=True,
    )
    print(f"  {payload['runtime_seconds']/60:.1f}분", flush=True)
    return payload


def load_all() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for lookback in LOOKBACKS:
        path = shard_path(lookback)
        if path.exists():
            rows.extend(json.loads(path.read_text(encoding="utf-8"))["rows"])
    return rows


def status() -> None:
    done = [lb for lb in LOOKBACKS if shard_path(lb).exists()]
    rows = load_all()
    print(f"완료 샤드 {len(done)}/{len(LOOKBACKS)} {done} | 누적 조합 {len(rows)}/{TOTAL_COMBINATIONS}")


def main(argv: list[str]) -> int:
    command = argv[1] if len(argv) > 1 else "status"
    if command == "status":
        status()
        return 0
    if command == "grid":
        targets = [int(v) for v in argv[2:] if v.isdigit()] or list(LOOKBACKS)
        for lookback in targets:
            if shard_path(lookback).exists():
                print(f"lookback {lookback}d: 이미 완료 — 건너뜀", flush=True)
                continue
            run_shard(lookback)
        return 0
    if command == "judge":
        from research.wave36_crosssection.judge36 import judge

        judge()
        return 0
    print(f"unknown command {command!r}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
