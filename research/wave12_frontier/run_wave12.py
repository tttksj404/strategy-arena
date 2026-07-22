#!/usr/bin/env python
"""Wave-12 (universe-expansion-frontier, U0-U6) pipeline CLI.

Mirrors research/wave11_yield/run_wave11.py's --stage fetch|run|gates|report|all
convention. `fetch` is network-bound (research/wave12_frontier/universe_frontier.py);
`run`/`gates`/`report` are cache-only. See research/wave12_frontier/SPEC.md for the
pre-registered contract.
"""

from __future__ import annotations

import argparse
from enum import StrEnum
import json
import math
from pathlib import Path
import sys
from typing import Any, Final, assert_never

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

import numpy as np
import pandas as pd  # noqa: PANDAS_OK

from research.wave1.gate_reporting import _series
from research.wave10_carry100.engine import ACTIVE_CAPITAL, MIN_ORDER_USDT, RESERVE_FRACTION, TOTAL_CAPITAL, Wave10Result
from research.wave10_carry100.regime import regime_breakdown
from research.wave12_frontier import engine12
from research.wave12_frontier import gates12
from research.wave12_frontier import universe_frontier as uf
from research.wave12_frontier.configs12 import CONFIG_IDS, CONFIGS, get_config
from research.wave12_frontier.reporting12 import write_wave12_report

BASE_DIR: Final = Path(__file__).resolve().parent
RESULTS_DIR: Final = BASE_DIR / "results"
REPORT_DIR: Final = BASE_DIR / "report"
CACHE_DIR: Final = BASE_DIR / "cache"
REGISTRY_PATH: Final = BASE_DIR / "REGISTRY.md"


class Stage(StrEnum):
    FETCH = "fetch"
    RUN = "run"
    GATES = "gates"
    REPORT = "report"
    ALL = "all"


class Wave12Error(Exception):
    pass


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _series_payload(series: pd.Series) -> list[dict[str, Any]]:
    return [{"timestamp": str(timestamp), "value": float(value)} for timestamp, value in series.items()]


def _eligible_stats(series: pd.Series) -> dict[str, float]:
    if series.empty:
        return {"median": 0.0, "mean": 0.0, "p25": 0.0, "p75": 0.0, "min": 0.0, "max": 0.0}
    return {
        "median": float(series.median()),
        "mean": float(series.mean()),
        "p25": float(series.quantile(0.25)),
        "p75": float(series.quantile(0.75)),
        "min": float(series.min()),
        "max": float(series.max()),
    }


def _median_reference_liquidity(symbols: tuple[str, ...]) -> float | None:
    pool = uf.load_candidate_pool()
    values = [
        float(pool["symbols"][symbol]["reference_volume_30d_usdt"])
        for symbol in symbols
        if symbol in pool["symbols"] and pool["symbols"][symbol].get("ok") is True
    ]
    return float(np.median(values)) if values else None


def _base_payload(
    candidate_id: str,
    result: Wave10Result,
    total_cost_usdt: float,
    eligible: pd.Series,
    stress_result: Wave10Result,
    stress_total_cost_usdt: float,
    stress_eligible: pd.Series,
) -> dict[str, Any]:
    config = get_config(candidate_id)
    candidate = config.candidate
    leg = gates12.leg_usdt(config)
    gross = gates12.gross_usdt(config)
    n_trades = len(result.trade_returns)
    avg_cost_per_trade = (total_cost_usdt / n_trades) if n_trades else 0.0
    return {
        "candidate_id": candidate_id,
        "family": "wave12_frontier",
        "breadth": config.breadth,
        "history_months": config.history_months,
        "definition": config.note,
        "config": {
            "window_days": candidate.window_days,
            "threshold_apr": candidate.threshold_apr,
            "top_k_pairs": candidate.top_k,
            "leg_fraction_of_active_capital": config.leg_fraction,
        },
        "capital_contract": {
            "total_capital_usdt": TOTAL_CAPITAL,
            "reserve_fraction": RESERVE_FRACTION,
            "active_capital_usdt": ACTIVE_CAPITAL,
            "min_order_usdt": MIN_ORDER_USDT,
            "leg_usdt_nominal": leg,
            "gross_usdt_nominal": gross,
        },
        "cost_model": "tiered_slippage_point_in_time(1bp_major/3bp_rank1-50/6bp_rank51-100/10bp_rank101-200/20bp_rank201plus)+maker_0.02pct_per_leg+liquidity_floor_2M_30d_avg",
        "equity": _series_payload(result.equity),
        "positions": _series_payload(result.positions),
        "turnover": _series_payload(result.turnover),
        "trade_returns": _series_payload(result.trade_returns),
        "stress_equity": _series_payload(stress_result.equity),
        "stress_positions": _series_payload(stress_result.positions),
        "stress_turnover": _series_payload(stress_result.turnover),
        "stress_trade_returns": _series_payload(stress_result.trade_returns),
        "metadata": {
            "symbols_used": list(result.symbols_used),
            "universe_size_static": len(result.symbols_used),
            "eligible_count_stats": _eligible_stats(eligible),
            "median_reference_liquidity_usdt": _median_reference_liquidity(result.symbols_used),
            "max_concurrent_positions": result.max_concurrent_positions,
            "n_trades": n_trades,
            "total_cost_usdt": total_cost_usdt,
            "avg_cost_per_trade_usdt": avg_cost_per_trade,
            "stress_total_cost_usdt": stress_total_cost_usdt,
            "stress_n_trades": len(stress_result.trade_returns),
            "stress_eligible_count_stats": _eligible_stats(stress_eligible),
            "utilization": gates12.utilization(result),
            "annualized_round_trips": gates12.annualized_round_trips(result),
            "source_engine": (
                "research.wave12_frontier.engine12.run_candidate (copies "
                "research.wave10_carry100.engine.run_fixed_fraction_portfolio's per-timestamp "
                "bookkeeping unmodified; only the tiered point-in-time cost/liquidity model differs)"
            ),
        },
    }


def _stage_fetch(force: bool) -> None:
    uf.run_fetch_stage(force)


def _stage_run(only: str | None) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for config in CONFIGS:
        candidate_id = config.candidate.candidate_id
        if only is not None and candidate_id != only:
            continue
        print(f"run: {candidate_id} starting (breadth={config.breadth}, history_months={config.history_months})...")
        result, total_cost, eligible = engine12.run_candidate(config, stress_multiplier=engine12.DEFAULT_STRESS_MULTIPLIER)
        stress_result, stress_total_cost, stress_eligible = engine12.run_candidate(config, stress_multiplier=engine12.STRESS_MULTIPLIER)
        payload = _base_payload(candidate_id, result, total_cost, eligible, stress_result, stress_total_cost, stress_eligible)
        _save_json(RESULTS_DIR / f"{candidate_id}.json", payload)
        final_equity = float(result.equity.iloc[-1]) if len(result.equity) else float("nan")
        print(
            f"run: {candidate_id} done (universe={len(result.symbols_used)}, trades={len(result.trade_returns)}, "
            f"final_active_equity=${final_equity:.2f}, total_cost=${total_cost:.2f}, "
            f"eligible_median={eligible.median() if len(eligible) else 0:.0f})"
        )


def _result_from_payload(payload: dict[str, Any], prefix: str) -> Wave10Result:
    metadata = payload["metadata"]
    if prefix:
        return Wave10Result(
            equity=_series(payload[f"{prefix}equity"]),
            positions=_series(payload[f"{prefix}positions"]),
            turnover=_series(payload[f"{prefix}turnover"]),
            trade_returns=_series(payload[f"{prefix}trade_returns"]),
            max_concurrent_positions=0,
            symbols_used=(),
        )
    return Wave10Result(
        equity=_series(payload["equity"]),
        positions=_series(payload["positions"]),
        turnover=_series(payload["turnover"]),
        trade_returns=_series(payload["trade_returns"]),
        max_concurrent_positions=int(metadata["max_concurrent_positions"]),
        symbols_used=tuple(metadata["symbols_used"]),
    )


def _evaluate_and_save(candidate_id: str, seed_offset: int, high_funding_bar: float | None) -> gates12.GateReport:
    path = RESULTS_DIR / f"{candidate_id}.json"
    payload = _load_json(path)
    result = _result_from_payload(payload, "")
    stress_result = _result_from_payload(payload, "stress_")
    config = get_config(candidate_id)
    report = gates12.evaluate_gates(config, result, stress_result, seed_offset, high_funding_bar)
    payload["gates"] = gates12.gate_report_payload(report)
    payload["regime_breakdown"] = regime_breakdown(result)
    payload["stress_regime_breakdown"] = regime_breakdown(stress_result)
    payload["reference_metrics"] = {
        "dsr": gates12.deflated_sharpe_reference(result),
        "total_trials_disclosed": gates12.DSR_CUMULATIVE_TRIALS,
    }
    _save_json(path, payload)
    return report


def _stage_gates(only: str | None) -> None:
    # U0 always evaluated first (no dependency on `only`) -- every other config's
    # promotion check needs U0's own tiered-cost high-funding return as its bar
    # (SPEC.md: "고펀딩기 연환산 > U0"), so it must exist before anything else is judged.
    u0_report = _evaluate_and_save("U0", 0, None)
    u0_payload = _load_json(RESULTS_DIR / "U0.json")
    u0_high_funding_bar = u0_payload["regime_breakdown"].get("high_funding_mean_annualized_return")
    print(f"gates: U0 -> {u0_report.overall} (baseline; high_funding_annualized={u0_high_funding_bar})")

    for seed_offset, candidate_id in enumerate(CONFIG_IDS):
        if candidate_id == "U0":
            continue
        if only is not None and candidate_id != only:
            continue
        report = _evaluate_and_save(candidate_id, seed_offset, u0_high_funding_bar)
        # report.promotion.promoted means only "beat U0's return number" -- SPEC.md's
        # actual promotion rule is that AND report.overall == "PASS" together (see
        # reporting12.py's actually_promoted for the same conjunction, applied to what
        # gets written to the report/registry). Logged explicitly as both fields here so
        # this console line can't be misread the way an earlier version of this log was.
        actually_promoted = report.overall == "PASS" and report.promotion.promoted
        print(
            f"gates: {candidate_id} -> {report.overall} (beat_u0_return={report.promotion.promoted}, "
            f"actually_promoted={actually_promoted}, reasons={list(report.failure_reasons)})"
        )


def _stage_report() -> None:
    write_wave12_report(RESULTS_DIR, REPORT_DIR, REGISTRY_PATH, CACHE_DIR)
    print(f"report: wrote {REPORT_DIR / 'wave12_report.md'} and {REGISTRY_PATH}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Wave-12 universe-expansion-frontier pipeline")
    parser.add_argument("--stage", required=True, type=Stage, choices=tuple(Stage))
    parser.add_argument("--only", choices=CONFIG_IDS)
    parser.add_argument("--force", action="store_true", help="fetch stage only: refetch even if cached")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        match args.stage:
            case Stage.FETCH:
                _stage_fetch(args.force)
            case Stage.RUN:
                _stage_run(args.only)
            case Stage.GATES:
                _stage_gates(args.only)
            case Stage.REPORT:
                _stage_report()
            case Stage.ALL:
                _stage_fetch(args.force)
                _stage_run(args.only)
                _stage_gates(args.only)
                _stage_report()
            case unreachable:
                assert_never(unreachable)
    except (FileNotFoundError, Wave12Error, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
