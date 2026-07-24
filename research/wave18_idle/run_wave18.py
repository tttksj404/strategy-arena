#!/usr/bin/env python
"""Wave-18 (idle-capital-deployment overlay, I0-I5) pipeline CLI. Mirrors
research/wave13_liquidity/run_wave13.py's / research/wave17_lending_verified/run_wave17.py's
--stage convention: `fetch` is network-bound (OKX lending-rate-history for USDT --
fetch18.py); `run`/`gates`/`report` are cache-only. See research/wave18_idle/SPEC.md for the
pre-registered, frozen contract this pipeline implements.
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

import pandas as pd  # noqa: PANDAS_OK

from research.wave1.gate_reporting import _series
from research.wave10_carry100.engine import ACTIVE_CAPITAL, MIN_ORDER_USDT, RESERVE_FRACTION, TOTAL_CAPITAL
from research.wave10_carry100.regime import regime_breakdown as compute_regime_breakdown
from research.wave13_liquidity import costs_measured
from research.wave18_idle import engine18, fetch18, gates18
from research.wave18_idle.configs18 import CONFIG_IDS, CONFIGS, L4_CONFIG, LEG_FRACTION, TOP_K, get_config
from research.wave18_idle.reporting18 import write_wave18_report

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


class Wave18Error(Exception):
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


def _layer_series_payload(series: pd.Series) -> list[dict[str, Any]]:
    return [{"timestamp": str(timestamp), "layer": str(value)} for timestamp, value in series.items()]


def _layer_series(records: list[dict[str, Any]]) -> pd.Series:
    if not records:
        return pd.Series(dtype=object)
    idx = pd.DatetimeIndex([pd.Timestamp(item["timestamp"]) for item in records])
    values = [str(item["layer"]) for item in records]
    return pd.Series(values, index=idx, dtype=object).sort_index()


def _layer_used_from_positions(positions: pd.Series) -> pd.Series:
    """I0 has no native layer_used (engine18.run_i0_reference == engine13.run_candidate,
    returning a bare Wave10Result) -- synthesized here so I0's saved JSON has the same shape
    as I1-I5's (needed by gates18's S6 check AND reporting18's layer-breakdown table)."""
    return positions.apply(lambda value: engine18.LAYER_L4 if abs(float(value)) > 0.0 else engine18.LAYER_CASH)


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


def _capital_contract() -> dict[str, Any]:
    return {
        "total_capital_usdt": TOTAL_CAPITAL,
        "reserve_fraction": RESERVE_FRACTION,
        "active_capital_usdt": ACTIVE_CAPITAL,
        "min_order_usdt": MIN_ORDER_USDT,
        "leg_usdt_nominal": gates18.leg_usdt(),
        "gross_usdt_nominal": gates18.gross_usdt(),
    }


def _payload(
    candidate_id: str,
    definition: str,
    extra_config: dict[str, Any],
    equity: pd.Series,
    positions: pd.Series,
    turnover: pd.Series,
    trade_returns: pd.Series,
    layer_used: pd.Series,
    total_cost_usdt: float,
    eligible: pd.Series,
    symbols_used: tuple[str, ...],
    max_concurrent: int,
    stress_equity: pd.Series,
    stress_positions: pd.Series,
    stress_turnover: pd.Series,
    stress_trade_returns: pd.Series,
    stress_layer_used: pd.Series,
    stress_total_cost_usdt: float,
) -> dict[str, Any]:
    n_trades = len(trade_returns)
    avg_cost_per_trade = (total_cost_usdt / n_trades) if n_trades else 0.0
    return {
        "candidate_id": candidate_id,
        "family": "wave18_idle",
        "definition": definition,
        "config": {
            "window_days": L4_CONFIG.candidate.window_days,
            "l4_threshold_apr": L4_CONFIG.candidate.threshold_apr,
            "top_k_pairs": TOP_K,
            "leg_fraction_of_active_capital": LEG_FRACTION,
            **extra_config,
        },
        "capital_contract": _capital_contract(),
        "cost_model": "bitget_measured_volume_mapping(wave13_liquidity.costs_measured, unmodified)+maker_0.02pct_per_leg -- L4 승계, wave18이 재도출하지 않음",
        "equity": _series_payload(equity),
        "positions": _series_payload(positions),
        "turnover": _series_payload(turnover),
        "trade_returns": _series_payload(trade_returns),
        "layer_used": _layer_series_payload(layer_used),
        "stress_equity": _series_payload(stress_equity),
        "stress_positions": _series_payload(stress_positions),
        "stress_turnover": _series_payload(stress_turnover),
        "stress_trade_returns": _series_payload(stress_trade_returns),
        "stress_layer_used": _layer_series_payload(stress_layer_used),
        "metadata": {
            "symbols_used": list(symbols_used),
            "universe_size_static": len(symbols_used),
            "eligible_l4_count_stats": _eligible_stats(eligible),
            "max_concurrent_positions": max_concurrent,
            "n_trades": n_trades,
            "total_cost_usdt": total_cost_usdt,
            "avg_cost_per_trade_usdt": avg_cost_per_trade,
            "stress_total_cost_usdt": stress_total_cost_usdt,
            "stress_n_trades": len(stress_trade_returns),
            "utilization": gates18.utilization(positions),
            "annualized_round_trips": gates18.annualized_round_trips(equity, trade_returns),
            "layer_breakdown": gates18.layer_breakdown(layer_used),
            "source_engine": (
                "research.wave18_idle.engine18.run_i0_reference (== research.wave13_liquidity.engine13.run_candidate, unmodified)"
                if candidate_id == "I0"
                else "research.wave18_idle.engine18.run_idle_candidate"
            ),
        },
    }


def _stage_fetch() -> None:
    fetch18.collect_usdt_lending()
    fetch18.collect_margin_borrow_rates()


def _stage_run(only: str | None) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    mapping = costs_measured.fit_mapping()
    print(
        f"run: measured-cost mapping fitted from {mapping.raw_point_count} Bitget points "
        f"-> {len(mapping.anchor_bp)} isotonic buckets (worst={mapping.worst_bp:.3f}bp, best={mapping.best_bp:.4f}bp)"
    )

    lending_apr: float | None = None
    try:
        lending_apr = fetch18.load_usdt_lending_apr(conservative=True)
        print(f"run: USDT lending APR (conservative/min, snapshot) = {lending_apr:.6f}")
    except RuntimeError as error:
        print(f"run: USDT lending APR unavailable ({error}) -- I1/I5 will raise if requested")

    for idle_config in CONFIGS:
        candidate_id = idle_config.candidate_id
        if only is not None and candidate_id != only:
            continue
        print(f"run: {candidate_id} starting...")
        if candidate_id == "I0":
            result, total_cost, eligible = engine18.run_i0_reference(mapping, engine18.DEFAULT_STRESS_MULTIPLIER)
            stress_result, stress_total_cost, _stress_eligible = engine18.run_i0_reference(mapping, engine18.STRESS_MULTIPLIER)
            layer_used = _layer_used_from_positions(result.positions)
            stress_layer_used = _layer_used_from_positions(stress_result.positions)
            extra_config: dict[str, Any] = {"universe_kind": L4_CONFIG.universe_kind, "breadth": L4_CONFIG.breadth}
        else:
            result, total_cost, eligible = engine18.run_idle_candidate(idle_config, mapping, lending_apr, engine18.DEFAULT_STRESS_MULTIPLIER)
            stress_result, stress_total_cost, _stress_eligible = engine18.run_idle_candidate(
                idle_config, mapping, lending_apr, engine18.STRESS_MULTIPLIER
            )
            layer_used = result.layer_used
            stress_layer_used = stress_result.layer_used
            extra_config = {
                "uses_carry_overlay": idle_config.uses_carry_overlay,
                "overlay_symbols": list(idle_config.overlay_symbols) if idle_config.overlay_symbols else None,
                "uses_reverse_overlay": idle_config.uses_reverse_overlay,
                "uses_lending_fallback": idle_config.uses_lending_fallback,
                "lending_apr_used": lending_apr if idle_config.uses_lending_fallback else None,
                "lending_apr_source": (
                    "wave18_idle/cache/usdt_lending.json: lending_rate_min (보수적 하한, ~4일 스냅샷 상수, 시계열 미검증)"
                    if idle_config.uses_lending_fallback
                    else None
                ),
            }

        payload = _payload(
            candidate_id,
            idle_config.note,
            extra_config,
            result.equity,
            result.positions,
            result.turnover,
            result.trade_returns,
            layer_used,
            total_cost,
            eligible,
            result.symbols_used,
            result.max_concurrent_positions,
            stress_result.equity,
            stress_result.positions,
            stress_result.turnover,
            stress_result.trade_returns,
            stress_layer_used,
            stress_total_cost,
        )
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
            f"run: {candidate_id} done (final_active_equity=${final_equity:.2f}, trades={len(result.trade_returns)}, "
            f"utilization={gates18.utilization(result.positions):.3f})"
        )


def _result_series(payload: dict[str, Any], prefix: str) -> tuple[pd.Series, pd.Series, pd.Series]:
    equity = _series(payload[f"{prefix}equity"])
    positions = _series(payload[f"{prefix}positions"])
    layer_used = _layer_series(payload[f"{prefix}layer_used"])
    return equity, positions, layer_used


def _evaluate_and_save(
    candidate_id: str,
    i0_positions: pd.Series,
    i0_full_period: float | None,
    i0_high_funding: float | None,
    seed_offset: int,
) -> gates18.GateReport:
    path = RESULTS_DIR / f"{candidate_id}.json"
    payload = _load_json(path)
    equity, positions, layer_used = _result_series(payload, "")
    stress_equity, _stress_positions, _stress_layer_used = _result_series(payload, "stress_")
    idle_config = get_config(candidate_id)
    report = gates18.evaluate_gates(
        idle_config, equity, positions, stress_equity, layer_used, i0_positions, i0_full_period, i0_high_funding, seed_offset
    )
    payload["gates"] = gates18.gate_report_payload(report)
    payload["regime_breakdown"] = compute_regime_breakdown(gates18._EquityOnly(equity))
    payload["stress_regime_breakdown"] = compute_regime_breakdown(gates18._EquityOnly(stress_equity))
    payload["full_period_annualized"] = gates18.full_period_annualized(equity)
    payload["reference_metrics"] = {
        "dsr": gates18.deflated_sharpe_reference(equity),
        "total_trials_disclosed": gates18.DSR_CUMULATIVE_TRIALS,
    }
    _save_json(path, payload)
    return report


def _stage_gates(only: str | None) -> None:
    i0_payload = _load_json(RESULTS_DIR / "I0.json")
    i0_equity, i0_positions, _i0_layer_used = _result_series(i0_payload, "")
    i0_full_period = gates18.full_period_annualized(i0_equity)
    i0_regime = compute_regime_breakdown(gates18._EquityOnly(i0_equity))
    i0_high_funding = i0_regime.get("high_funding_mean_annualized_return")
    print(f"gates: I0 reference -- full_period_cagr={i0_full_period}, high_funding_annualized={i0_high_funding}")

    for seed_offset, candidate_id in enumerate(CONFIG_IDS):
        if only is not None and candidate_id != only:
            continue
        report = _evaluate_and_save(candidate_id, i0_positions, i0_full_period, i0_high_funding, seed_offset)
        print(
            f"gates: {candidate_id} -> {report.overall} (promoted={report.promotion.promoted}, "
            f"full_period_cagr={report.promotion.full_period_annualized}, reasons={list(report.failure_reasons)})"
        )


def _stage_report() -> None:
    write_wave18_report(RESULTS_DIR, REPORT_DIR, REGISTRY_PATH, CACHE_DIR)
    print(f"report: wrote {REPORT_DIR / 'wave18_report.md'} and {REGISTRY_PATH}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Wave-18 idle-capital-deployment overlay pipeline")
    parser.add_argument("--stage", required=True, type=Stage, choices=tuple(Stage))
    parser.add_argument("--only", choices=CONFIG_IDS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        match args.stage:
            case Stage.FETCH:
                _stage_fetch()
            case Stage.RUN:
                _stage_run(args.only)
            case Stage.GATES:
                _stage_gates(args.only)
            case Stage.REPORT:
                _stage_report()
            case Stage.ALL:
                _stage_fetch()
                _stage_run(args.only)
                _stage_gates(args.only)
                _stage_report()
            case unreachable:
                assert_never(unreachable)
    except (FileNotFoundError, Wave18Error, RuntimeError, ValueError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
