#!/usr/bin/env python
"""Wave-13 (measured-spread cost recalibration + liquidity-constrained carry, L1-L5)
pipeline CLI. Mirrors research/wave12_frontier/run_wave12.py's --stage X convention, with
"fetch" renamed to "collect" (this wave collects Bitget SPREAD measurements, not new
OHLCV/funding history -- see research/wave13_liquidity/universe_liquidity.py's module
docstring for why the backtest data itself is borrowed read-only from
research/wave12_frontier/cache/). `collect` is network-bound (Bitget REST);
`run`/`gates`/`report` are cache-only. See research/wave13_liquidity/SPEC.md for the
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
from research.wave13_liquidity import collect_spreads
from research.wave13_liquidity import costs_measured
from research.wave13_liquidity import engine13
from research.wave13_liquidity import gates13
from research.wave13_liquidity import universe_liquidity as ul
from research.wave13_liquidity.configs13 import CONFIG_IDS, CONFIGS, get_config
from research.wave13_liquidity.reporting13 import write_wave13_report

BASE_DIR: Final = Path(__file__).resolve().parent
RESULTS_DIR: Final = BASE_DIR / "results"
REPORT_DIR: Final = BASE_DIR / "report"
CACHE_DIR: Final = BASE_DIR / "cache"
REGISTRY_PATH: Final = BASE_DIR / "REGISTRY.md"


class Stage(StrEnum):
    COLLECT = "collect"
    RUN = "run"
    GATES = "gates"
    REPORT = "report"
    ALL = "all"


class Wave13Error(Exception):
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
    pool = ul.load_candidate_pool()
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
    leg = gates13.leg_usdt(config)
    gross = gates13.gross_usdt(config)
    n_trades = len(result.trade_returns)
    avg_cost_per_trade = (total_cost_usdt / n_trades) if n_trades else 0.0
    return {
        "candidate_id": candidate_id,
        "family": "wave13_liquidity",
        "universe_kind": config.universe_kind,
        "breadth": config.breadth,
        "fixed_symbols": list(config.fixed_symbols) if config.fixed_symbols else None,
        "history_months": config.history_months,
        "dynamic_volume_floor_usdt": config.dynamic_volume_floor_usdt,
        "dynamic_slippage_cap_bp": config.dynamic_slippage_cap_bp,
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
        "cost_model": "bitget_measured_volume_mapping(costs_measured.py, isotonic-fit over cache/measured_spreads.json)+maker_0.02pct_per_leg",
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
            "utilization": gates13.utilization(result),
            "annualized_round_trips": gates13.annualized_round_trips(result),
            "source_engine": (
                "research.wave13_liquidity.engine13.run_candidate (copies "
                "research.wave12_frontier.engine12's per-timestamp bookkeeping unmodified; "
                "only the measured-cost/liquidity model and universe-membership rule differ)"
            ),
        },
    }


def _stage_collect(minimum_symbols: int) -> None:
    collect_spreads.collect_measured_spreads(minimum_symbols)


def _stage_run(only: str | None) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    mapping = costs_measured.fit_mapping()
    print(
        f"run: measured-cost mapping fitted from {mapping.raw_point_count} Bitget points "
        f"-> {len(mapping.anchor_bp)} isotonic buckets (worst={mapping.worst_bp:.3f}bp, best={mapping.best_bp:.4f}bp)"
    )
    for config in CONFIGS:
        candidate_id = config.candidate.candidate_id
        if only is not None and candidate_id != only:
            continue
        print(f"run: {candidate_id} starting (universe_kind={config.universe_kind}, breadth={config.breadth})...")
        result, total_cost, eligible = engine13.run_candidate(config, mapping, stress_multiplier=engine13.DEFAULT_STRESS_MULTIPLIER)
        stress_result, stress_total_cost, stress_eligible = engine13.run_candidate(config, mapping, stress_multiplier=engine13.STRESS_MULTIPLIER)
        payload = _base_payload(candidate_id, result, total_cost, eligible, stress_result, stress_total_cost, stress_eligible)
        payload["cost_mapping"] = {
            "anchor_log_volume": mapping.anchor_log_volume.tolist(),
            "anchor_bp": mapping.anchor_bp.tolist(),
            "bucket_counts": list(mapping.bucket_counts),
            "raw_point_count": mapping.raw_point_count,
            "source_collected_at_utc": mapping.source_collected_at_utc,
        }
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


def _evaluate_and_save(candidate_id: str, seed_offset: int) -> gates13.GateReport:
    path = RESULTS_DIR / f"{candidate_id}.json"
    payload = _load_json(path)
    result = _result_from_payload(payload, "")
    stress_result = _result_from_payload(payload, "stress_")
    config = get_config(candidate_id)
    report = gates13.evaluate_gates(config, result, stress_result, seed_offset)
    payload["gates"] = gates13.gate_report_payload(report)
    payload["regime_breakdown"] = regime_breakdown(result)
    payload["stress_regime_breakdown"] = regime_breakdown(stress_result)
    payload["reference_metrics"] = {
        "dsr": gates13.deflated_sharpe_reference(result),
        "total_trials_disclosed": gates13.DSR_CUMULATIVE_TRIALS,
    }
    _save_json(path, payload)
    return report


def _stage_gates(only: str | None) -> None:
    for seed_offset, candidate_id in enumerate(CONFIG_IDS):
        if only is not None and candidate_id != only:
            continue
        report = _evaluate_and_save(candidate_id, seed_offset)
        print(
            f"gates: {candidate_id} -> {report.overall} "
            f"(high_funding_annualized={report.promotion.high_funding_mean_annualized_return}, "
            f"promoted={report.promotion.promoted}, reasons={list(report.failure_reasons)})"
        )


def _stage_report() -> None:
    write_wave13_report(RESULTS_DIR, REPORT_DIR, REGISTRY_PATH, CACHE_DIR)
    print(f"report: wrote {REPORT_DIR / 'wave13_report.md'} and {REGISTRY_PATH}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Wave-13 measured-spread liquidity-constrained-carry pipeline")
    parser.add_argument("--stage", required=True, type=Stage, choices=tuple(Stage))
    parser.add_argument("--only", choices=CONFIG_IDS)
    parser.add_argument("--minimum-symbols", type=int, default=collect_spreads.MINIMUM_SYMBOLS, help="collect stage only: minimum Bitget symbols to measure")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        match args.stage:
            case Stage.COLLECT:
                _stage_collect(args.minimum_symbols)
            case Stage.RUN:
                _stage_run(args.only)
            case Stage.GATES:
                _stage_gates(args.only)
            case Stage.REPORT:
                _stage_report()
            case Stage.ALL:
                _stage_collect(args.minimum_symbols)
                _stage_run(args.only)
                _stage_gates(args.only)
                _stage_report()
            case unreachable:
                assert_never(unreachable)
    except (FileNotFoundError, Wave13Error, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
