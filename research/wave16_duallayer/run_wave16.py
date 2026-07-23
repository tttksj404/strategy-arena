#!/usr/bin/env python
"""Wave-16 (dual-layer carry: funding + spot lending, E0-E4) pipeline CLI. Mirrors
research/wave13_liquidity/run_wave13.py's --stage X convention: `fetch` is network-bound (OKX +
Bitget current snapshot -- research/wave16_duallayer/fetch_lending.py); `run`/`gates`/`report`
are cache-only. See research/wave16_duallayer/SPEC.md for the pre-registered contract.
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
from research.wave16_duallayer import engine16
from research.wave16_duallayer import fetch_lending
from research.wave13_liquidity.gates13 import gate_report_payload
from research.wave16_duallayer import gates16
from research.wave16_duallayer.configs16 import CANDIDATE_IDS, L4_CONFIG, get_candidate
from research.wave16_duallayer.reporting16 import write_wave16_report

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


class Wave16Error(Exception):
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


def _variant_payload(prefix: str, variant: engine16.VariantRun) -> dict[str, Any]:
    return {
        f"{prefix}equity": _series_payload(variant.result.equity),
        f"{prefix}positions": _series_payload(variant.result.positions),
        f"{prefix}turnover": _series_payload(variant.result.turnover),
        f"{prefix}trade_returns": _series_payload(variant.result.trade_returns),
        f"{prefix}total_cost_usdt": variant.total_cost_usdt,
        f"{prefix}n_trades": len(variant.result.trade_returns),
        f"{prefix}stress_equity": _series_payload(variant.stress_result.equity),
        f"{prefix}stress_positions": _series_payload(variant.stress_result.positions),
        f"{prefix}stress_turnover": _series_payload(variant.stress_result.turnover),
        f"{prefix}stress_trade_returns": _series_payload(variant.stress_result.trade_returns),
        f"{prefix}stress_total_cost_usdt": variant.stress_total_cost_usdt,
        f"{prefix}stress_n_trades": len(variant.stress_result.trade_returns),
    }


def _result_from_payload(payload: dict[str, Any], key: str, symbols_used: tuple[str, ...], max_concurrent: int) -> Wave10Result:
    return Wave10Result(
        equity=_series(payload[f"{key}equity"]),
        positions=_series(payload[f"{key}positions"]),
        turnover=_series(payload[f"{key}turnover"]),
        trade_returns=_series(payload[f"{key}trade_returns"]),
        max_concurrent_positions=max_concurrent,
        symbols_used=symbols_used,
    )


def _base_payload(candidate_id: str, result: engine16.DualLayerResult, lending_snapshot: dict[str, Any]) -> dict[str, Any]:
    candidate = get_candidate(candidate_id)
    fam_candidate = L4_CONFIG.candidate
    snapshot_pick = engine16.current_snapshot_pick(candidate.ranking_lending_discount, fam_candidate.threshold_apr, lending_snapshot)
    n_with_lending_in_universe = sum(
        1 for symbol in result.symbols_used if lending_snapshot["by_symbol"].get(symbol, {}).get("lending_available")
    )
    payload: dict[str, Any] = {
        "candidate_id": candidate_id,
        "family": "wave16_duallayer",
        "definition": candidate.note,
        "ranking_lending_discount": candidate.ranking_lending_discount,
        "pnl_lending_discount": candidate.pnl_lending_discount,
        "config": {
            "window_days": fam_candidate.window_days,
            "threshold_apr": fam_candidate.threshold_apr,
            "top_k_pairs": fam_candidate.top_k,
            "leg_fraction_of_active_capital": L4_CONFIG.leg_fraction,
            "universe_kind": L4_CONFIG.universe_kind,
            "breadth": L4_CONFIG.breadth,
            "history_months": L4_CONFIG.history_months,
        },
        "capital_contract": {
            "total_capital_usdt": engine16.TOTAL_CAPITAL,
            "reserve_fraction": engine16.RESERVE_FRACTION,
            "active_capital_usdt": engine16.ACTIVE_CAPITAL,
            "min_order_usdt": engine16.MIN_ORDER_USDT,
        },
        "cost_model": "bitget_measured_volume_mapping(wave13_liquidity.costs_measured, unmodified)+maker_0.02pct_per_leg -- L4 승계, wave16이 재도출하지 않음",
        "metadata": {
            "symbols_used": list(result.symbols_used),
            "universe_size_static": len(result.symbols_used),
            "n_symbols_with_lending_data": n_with_lending_in_universe,
            "n_symbols_with_lending_data_pct": (n_with_lending_in_universe / len(result.symbols_used) * 100.0) if result.symbols_used else 0.0,
            "lending_snapshot_collected_at_utc": lending_snapshot.get("collected_at_utc"),
            "max_concurrent_positions_combined": result.combined.result.max_concurrent_positions,
            "max_concurrent_positions_funding_only": result.funding_only.result.max_concurrent_positions,
        },
        "current_snapshot_pick": snapshot_pick,
    }
    payload.update(_variant_payload("combined_", result.combined))
    payload.update(_variant_payload("funding_only_", result.funding_only))
    return payload


def _stage_fetch() -> None:
    fetch_lending.collect_lending_snapshot()


def _stage_run(only: str | None) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    lending_snapshot = fetch_lending.load_lending_snapshot()
    runner = engine16.build_runner(lending_snapshot)
    print(f"run: universe {len(runner.symbols())} symbols (inherited from wave13 L4), lending snapshot collected_at_utc={lending_snapshot.get('collected_at_utc')}")
    for candidate_id in CANDIDATE_IDS:
        if only is not None and candidate_id != only:
            continue
        print(f"run: {candidate_id} starting...")
        result = engine16.run_candidate(candidate_id, runner)
        payload = _base_payload(candidate_id, result, lending_snapshot)
        _save_json(RESULTS_DIR / f"{candidate_id}.json", payload)
        combined_final = float(result.combined.result.equity.iloc[-1]) if len(result.combined.result.equity) else float("nan")
        funding_only_final = float(result.funding_only.result.equity.iloc[-1]) if len(result.funding_only.result.equity) else float("nan")
        print(
            f"run: {candidate_id} done (combined_final=${combined_final:.2f}, funding_only_final=${funding_only_final:.2f}, "
            f"combined_trades={len(result.combined.result.trade_returns)}, funding_only_trades={len(result.funding_only.result.trade_returns)})"
        )


def _evaluate_and_save(candidate_id: str, seed_offset: int) -> None:
    path = RESULTS_DIR / f"{candidate_id}.json"
    payload = _load_json(path)
    symbols_used = tuple(payload["metadata"]["symbols_used"])
    max_combined = int(payload["metadata"]["max_concurrent_positions_combined"])
    max_funding_only = int(payload["metadata"]["max_concurrent_positions_funding_only"])

    combined_result = _result_from_payload(payload, "combined_", symbols_used, max_combined)
    funding_only_result = _result_from_payload(payload, "funding_only_", symbols_used, max_funding_only)
    funding_only_stress_result = _result_from_payload(payload, "funding_only_stress_", symbols_used, max_funding_only)

    gate_report = gates16.evaluate_funding_only_gates(funding_only_result, funding_only_stress_result, seed_offset)
    payload["gates_funding_only"] = gate_report_payload(gate_report)
    payload["regime_breakdown_combined"] = regime_breakdown(combined_result)
    payload["regime_breakdown_funding_only"] = regime_breakdown(funding_only_result)
    payload["reference_metrics"] = {
        "dsr_funding_only": gates16.deflated_sharpe_reference(funding_only_result),
        "total_trials_disclosed": gates16.DSR_CUMULATIVE_TRIALS,
        "utilization_funding_only": gates16.utilization(funding_only_result),
    }
    _save_json(path, payload)
    print(
        f"gates: {candidate_id} -> funding_only={gate_report.overall} "
        f"(high_funding_annualized_combined={payload['regime_breakdown_combined'].get('high_funding_mean_annualized_return')}, "
        f"reasons={list(gate_report.failure_reasons)})"
    )


def _stage_gates(only: str | None) -> None:
    for seed_offset, candidate_id in enumerate(CANDIDATE_IDS):
        if only is not None and candidate_id != only:
            continue
        _evaluate_and_save(candidate_id, seed_offset)


def _stage_report() -> None:
    write_wave16_report(RESULTS_DIR, REPORT_DIR, REGISTRY_PATH, CACHE_DIR)
    print(f"report: wrote {REPORT_DIR / 'wave16_report.md'} and {REGISTRY_PATH}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Wave-16 dual-layer-carry (funding + spot lending) pipeline")
    parser.add_argument("--stage", required=True, type=Stage, choices=tuple(Stage))
    parser.add_argument("--only", choices=CANDIDATE_IDS)
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
    except (FileNotFoundError, Wave16Error, RuntimeError, ValueError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
