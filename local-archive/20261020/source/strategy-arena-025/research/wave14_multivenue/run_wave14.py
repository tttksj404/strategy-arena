#!/usr/bin/env python
"""Wave-14 (multi-venue x concurrent-position x capital-tier carry, M0-M7) pipeline CLI.
Mirrors research/wave13_liquidity/run_wave13.py's --stage X convention. `fetch` is
network-bound (Bybit REST + a Hyperliquid feasibility probe); `run`/`gates`/`report` are
cache-only. See research/wave14_multivenue/SPEC.md for the pre-registered contract.
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
from research.wave10_carry100.engine import Wave10Result
from research.wave10_carry100.regime import regime_breakdown
from research.wave13_liquidity import costs_measured
from research.wave14_multivenue import costs_venue
from research.wave14_multivenue import engine14
from research.wave14_multivenue import fetch_venues
from research.wave14_multivenue import gates14
from research.wave14_multivenue import universe_multi as um
from research.wave14_multivenue.configs14 import AUX_BASELINES, CONFIG_IDS, CONFIGS, Wave14Config, get_config
from research.wave14_multivenue.reporting14 import write_wave14_report

BASE_DIR: Final = Path(__file__).resolve().parent
RESULTS_DIR: Final = BASE_DIR / "results"
REPORT_DIR: Final = BASE_DIR / "report"
CACHE_DIR: Final = BASE_DIR / "cache"
REGISTRY_PATH: Final = BASE_DIR / "REGISTRY.md"

ALL_RUNNABLE: Final[tuple[Wave14Config, ...]] = (*CONFIGS, *AUX_BASELINES)
ALL_RUNNABLE_IDS: Final[tuple[str, ...]] = tuple(config.candidate_id for config in ALL_RUNNABLE)


class Stage(StrEnum):
    FETCH = "fetch"
    RUN = "run"
    GATES = "gates"
    REPORT = "report"
    ALL = "all"


class Wave14Error(Exception):
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
    return [{"timestamp": str(timestamp), "value": float(value) if value == value else None} for timestamp, value in series.items()]


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


# ---------------------------------------------------------------------------
# Stage: fetch (network).
# ---------------------------------------------------------------------------


def _stage_fetch(force: bool, max_workers: int) -> None:
    l4_symbols = um.load_l4_symbols()
    fetch_venues.run_fetch_stage(l4_symbols, force=force, max_workers=max_workers)


# ---------------------------------------------------------------------------
# Stage: run (cache-only backtest).
# ---------------------------------------------------------------------------


def _needs_bybit(config: Wave14Config) -> bool:
    return config.include_bybit or config.structure == "cross_venue_spread"


def _base_payload(
    config: Wave14Config,
    result: Wave10Result,
    total_cost_usdt: float,
    eligible: pd.Series,
    bybit_share: pd.Series | None,
    stress_result: Wave10Result,
    stress_total_cost_usdt: float,
    stress_eligible: pd.Series,
) -> dict[str, Any]:
    n_trades = len(result.trade_returns)
    avg_cost_per_trade = (total_cost_usdt / n_trades) if n_trades else 0.0
    if config.structure == "cross_venue_spread":
        # M6/M7: every symbol in result.symbols_used has BOTH a Binance leg and a Bybit
        # leg simultaneously (paired structure, no ":BYBIT"-suffixed union keys the way
        # M0-M5's pool has) -- "binance_pool_size"/"bybit_pool_size" would misleadingly
        # read as a venue SPLIT if computed via venue_of_key the M0-M5 way, so both are
        # reported as the full paired-symbol count instead.
        binance_symbols = result.symbols_used
        bybit_symbols = result.symbols_used
    else:
        binance_symbols = tuple(s for s in result.symbols_used if um.venue_of_key(s) == "binance")
        bybit_symbols = tuple(s for s in result.symbols_used if um.venue_of_key(s) == "bybit")
    return {
        "candidate_id": config.candidate_id,
        "family": "wave14_multivenue" if config.structure == "carry" else "wave14_multivenue_cross_venue_spread",
        "is_frozen_candidate": config.candidate_id in CONFIG_IDS,
        "structure": config.structure,
        "total_capital_usdt": config.total_capital,
        "top_k_pairs": config.candidate.top_k,
        "include_bybit": config.include_bybit,
        "baseline_candidate_id": config.baseline_candidate_id,
        "definition": config.note,
        "config": {
            "window_days": config.candidate.window_days,
            "threshold_apr": config.candidate.threshold_apr,
            "top_k_pairs": config.candidate.top_k,
            "leg_fraction_of_active_capital": config.leg_fraction,
        },
        "capital_contract": {
            "total_capital_usdt": config.total_capital,
            "reserve_fraction": 0.10,
            "active_capital_usdt": config.active_capital,
            "leg_usdt_nominal": gates14.leg_usdt(config),
            "gross_usdt_nominal": gates14.gross_usdt(config),
        },
        "cost_model": (
            "binance_leg=bitget_measured(wave13.costs_measured)+maker_0.02pct; "
            "bybit_leg=bybit_measured(costs_venue.py, spot_maker_0.10pct/linear_maker_0.02pct)"
            if _needs_bybit(config)
            else "binance_leg_only=bitget_measured(wave13.costs_measured)+maker_0.02pct"
        ),
        "equity": _series_payload(result.equity),
        "positions": _series_payload(result.positions),
        "turnover": _series_payload(result.turnover),
        "trade_returns": _series_payload(result.trade_returns),
        "stress_equity": _series_payload(stress_result.equity),
        "stress_positions": _series_payload(stress_result.positions),
        "stress_turnover": _series_payload(stress_result.turnover),
        "stress_trade_returns": _series_payload(stress_result.trade_returns),
        "daily_bybit_share_of_filled_slots": _series_payload(bybit_share) if bybit_share is not None else None,
        "metadata": {
            "symbols_used": list(result.symbols_used),
            "universe_size_static": len(result.symbols_used),
            "binance_pool_size": len(binance_symbols),
            "bybit_pool_size": len(bybit_symbols),
            "eligible_count_stats": _eligible_stats(eligible),
            "max_concurrent_positions": result.max_concurrent_positions,
            "n_trades": n_trades,
            "total_cost_usdt": total_cost_usdt,
            "avg_cost_per_trade_usdt": avg_cost_per_trade,
            "stress_total_cost_usdt": stress_total_cost_usdt,
            "stress_n_trades": len(stress_result.trade_returns),
            "stress_eligible_count_stats": _eligible_stats(stress_eligible),
            "utilization": gates14.utilization_wrap(result),
            "annualized_round_trips": gates14.annualized_round_trips_wrap(result),
            "window_start": str(engine14.OVERLAP_START),
            "window_end_exclusive": str(engine14.OVERLAP_END),
            "source_engine": (
                "research.wave14_multivenue.engine14.run_carry_candidate"
                if config.structure == "carry"
                else "research.wave14_multivenue.engine14.run_cross_venue_candidate"
            ),
        },
    }


def _stage_run(only: str | None) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    binance_mapping = costs_measured.fit_mapping()
    print(
        f"run: Binance-leg mapping (Bitget-measured, wave13) fitted from {binance_mapping.raw_point_count} points "
        f"-> {len(binance_mapping.anchor_bp)} buckets (worst={binance_mapping.worst_bp:.3f}bp, best={binance_mapping.best_bp:.4f}bp)"
    )
    spread_path = CACHE_DIR / "bybit_spreads.json"
    if not spread_path.exists():
        raise Wave14Error(f"{spread_path} missing -- run `--stage fetch` first")
    bybit_mappings = costs_venue.fit_bybit_mappings(_load_json(spread_path))
    print(
        f"run: Bybit-leg mappings fitted -- spot(worst={bybit_mappings.spot.worst_bp:.3f}bp,best={bybit_mappings.spot.best_bp:.4f}bp), "
        f"linear(worst={bybit_mappings.linear.worst_bp:.3f}bp,best={bybit_mappings.linear.best_bp:.4f}bp)"
    )

    for config in ALL_RUNNABLE:
        if only is not None and config.candidate_id != only:
            continue
        print(f"run: {config.candidate_id} starting (total_capital=${config.total_capital:.0f}, top_k={config.candidate.top_k}, include_bybit={config.include_bybit}, structure={config.structure})...")
        bybit_arg_base = bybit_mappings if config.include_bybit else None
        if config.structure == "carry":
            result, total_cost, eligible, bybit_share = engine14.run_carry_candidate(config, binance_mapping, bybit_arg_base, engine14.DEFAULT_STRESS_MULTIPLIER)
            stress_result, stress_total_cost, stress_eligible, _ = engine14.run_carry_candidate(config, binance_mapping, bybit_arg_base, engine14.STRESS_MULTIPLIER)
        elif config.structure == "cross_venue_spread":
            result, total_cost, eligible = engine14.run_cross_venue_candidate(config, binance_mapping, bybit_mappings.linear, engine14.DEFAULT_STRESS_MULTIPLIER)
            stress_result, stress_total_cost, stress_eligible = engine14.run_cross_venue_candidate(config, binance_mapping, bybit_mappings.linear, engine14.STRESS_MULTIPLIER)
            bybit_share = None
        else:
            raise Wave14Error(f"unknown structure: {config.structure}")
        payload = _base_payload(config, result, total_cost, eligible, bybit_share, stress_result, stress_total_cost, stress_eligible)
        _save_json(RESULTS_DIR / f"{config.candidate_id}.json", payload)
        final_equity = float(result.equity.iloc[-1]) if len(result.equity) else float("nan")
        print(
            f"run: {config.candidate_id} done (universe={len(result.symbols_used)}, trades={len(result.trade_returns)}, "
            f"final_equity=${final_equity:.2f}, total_cost=${total_cost:.2f}, days={len(result.equity)})"
        )


# ---------------------------------------------------------------------------
# Stage: gates.
# ---------------------------------------------------------------------------


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


def _bybit_share_from_payload(payload: dict[str, Any]) -> pd.Series | None:
    raw = payload.get("daily_bybit_share_of_filled_slots")
    if not raw:
        return None
    records = [item for item in raw if isinstance(item, dict) and item.get("value") is not None]
    if not records:
        return None
    index = pd.DatetimeIndex([pd.Timestamp(item["timestamp"]) for item in records])
    return pd.Series([float(item["value"]) for item in records], index=index, dtype=float)


def _baseline_regime(candidate_id: str | None) -> dict[str, Any] | None:
    if candidate_id is None:
        return None
    path = RESULTS_DIR / f"{candidate_id}.json"
    if not path.exists():
        return None
    payload = _load_json(path)
    result = _result_from_payload(payload, "")
    return regime_breakdown(result)


def _evaluate_and_save(candidate_id: str, seed_offset: int) -> gates14.GateReport:
    path = RESULTS_DIR / f"{candidate_id}.json"
    payload = _load_json(path)
    result = _result_from_payload(payload, "")
    stress_result = _result_from_payload(payload, "stress_")
    config = get_config(candidate_id)
    bybit_share = _bybit_share_from_payload(payload)
    baseline_regime = _baseline_regime(config.baseline_candidate_id)
    report = gates14.evaluate_gates(config, result, stress_result, seed_offset, bybit_share, baseline_regime)
    payload["gates"] = gates14.gate_report_payload(report)
    payload["regime_breakdown"] = regime_breakdown(result)
    payload["stress_regime_breakdown"] = regime_breakdown(stress_result)
    payload["reference_metrics"] = {
        "dsr": gates14.deflated_sharpe_reference(result),
        "total_trials_disclosed": gates14.DSR_CUMULATIVE_TRIALS,
    }
    _save_json(path, payload)
    return report


def _stage_gates(only: str | None) -> None:
    for seed_offset, candidate_id in enumerate(ALL_RUNNABLE_IDS):
        if only is not None and candidate_id != only:
            continue
        report = _evaluate_and_save(candidate_id, seed_offset)
        print(
            f"gates: {candidate_id} -> {report.overall} "
            f"(high_funding_annualized={report.promotion.high_funding_mean_annualized_return}, "
            f"beats_baseline={report.promotion.beats_baseline}, promoted={report.promotion.promoted}, "
            f"reasons={list(report.failure_reasons)})"
        )


def _stage_report() -> None:
    write_wave14_report(RESULTS_DIR, REPORT_DIR, REGISTRY_PATH, CACHE_DIR)
    print(f"report: wrote {REPORT_DIR / 'wave14_report.md'} and {REGISTRY_PATH}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Wave-14 multi-venue x concurrent-position x capital-tier carry pipeline")
    parser.add_argument("--stage", required=True, type=Stage, choices=tuple(Stage))
    parser.add_argument("--only", choices=ALL_RUNNABLE_IDS)
    parser.add_argument("--force", action="store_true", help="fetch stage only: refetch even if cache files already exist")
    parser.add_argument("--max-workers", type=int, default=fetch_venues.DEFAULT_MAX_WORKERS, help="fetch stage only: concurrent worker threads")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        match args.stage:
            case Stage.FETCH:
                _stage_fetch(args.force, args.max_workers)
            case Stage.RUN:
                _stage_run(args.only)
            case Stage.GATES:
                _stage_gates(args.only)
            case Stage.REPORT:
                _stage_report()
            case Stage.ALL:
                _stage_fetch(args.force, args.max_workers)
                _stage_run(args.only)
                _stage_gates(args.only)
                _stage_report()
            case unreachable:
                assert_never(unreachable)
    except (FileNotFoundError, Wave14Error, RuntimeError, ValueError, um.UniverseError, fetch_venues.VenueFetchError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
