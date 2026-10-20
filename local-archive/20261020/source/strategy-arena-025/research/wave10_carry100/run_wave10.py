#!/usr/bin/env python
"""Wave-10 ($100 fixed-fraction carry) pipeline CLI.

Mirrors the wave2/wave9 --stage run|gates|report convention. Cache-only (research/wave1/cache);
no network calls. See research/wave10_carry100/SPEC.md for the pre-registered contract.
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

from research.wave1.common import CACHE_DIR as WAVE1_CACHE_DIR
from research.wave1.gate_reporting import _series
from research.wave10_carry100 import gates as gates_mod
from research.wave10_carry100 import regime as regime_mod
from research.wave10_carry100.configs import CONFIG_IDS, CONFIGS, get_config
from research.wave10_carry100.engine import (
    ACTIVE_CAPITAL,
    MIN_ORDER_USDT,
    RESERVE_FRACTION,
    TOTAL_CAPITAL,
    Wave10Result,
    run_fixed_fraction_portfolio,
    verify_cache_and_load_symbols,
)
from research.wave10_carry100.reporting import write_wave10_report
from research.wave1.fam_funding import load_markets


BASE_DIR: Final = Path(__file__).resolve().parent
RESULTS_DIR: Final = BASE_DIR / "results"
REPORT_DIR: Final = BASE_DIR / "report"
REGISTRY_PATH: Final = BASE_DIR / "REGISTRY.md"


class Stage(StrEnum):
    RUN = "run"
    GATES = "gates"
    REPORT = "report"
    ALL = "all"


class Wave10Error(Exception):
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


def _series_payload(series) -> list[dict[str, Any]]:  # noqa: ANN001 - pd.Series, kept untyped to avoid a pandas import here
    return [{"timestamp": str(timestamp), "value": float(value)} for timestamp, value in series.items()]


def _base_payload(candidate_id: str, result: Wave10Result) -> dict[str, Any]:
    config = get_config(candidate_id)
    leg = config.leg_fraction * ACTIVE_CAPITAL
    gross = 2.0 * config.candidate.top_k * config.leg_fraction * ACTIVE_CAPITAL
    return {
        "candidate_id": candidate_id,
        "family": "wave10_carry100",
        "definition": config.note,
        "config": {
            "window_days": config.candidate.window_days,
            "threshold_apr": config.candidate.threshold_apr,
            "top_k_pairs": config.candidate.top_k,
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
        "cost_model": "maker_0.02pct_per_leg_plus_slippage(1bp_major_3bp_alt)_both_legs",
        "equity": _series_payload(result.equity),
        "positions": _series_payload(result.positions),
        "turnover": _series_payload(result.turnover),
        "trade_returns": _series_payload(result.trade_returns),
        "metadata": {
            "symbols_used": list(result.symbols_used),
            "universe_size": len(result.symbols_used),
            "max_concurrent_positions": result.max_concurrent_positions,
            "n_trades": int(len(result.trade_returns)),
            "source_engine": "research.wave10_carry100.engine.run_fixed_fraction_portfolio "
            "(imports research.wave1.fam_funding signal/universe primitives; only leg sizing "
            "and top_k pair count differ from research.wave2.funding W2c)",
        },
    }


def _stage_run(only: str | None) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    symbols = verify_cache_and_load_symbols()
    markets = load_markets(WAVE1_CACHE_DIR, symbols)
    for config in CONFIGS:
        candidate_id = config.candidate.candidate_id
        if only is not None and candidate_id != only:
            continue
        result = run_fixed_fraction_portfolio(markets, config)
        payload = _base_payload(candidate_id, result)
        _save_json(RESULTS_DIR / f"{candidate_id}.json", payload)
        final_equity = float(result.equity.iloc[-1]) if len(result.equity) else float("nan")
        print(f"run: {candidate_id} done (trades={len(result.trade_returns)}, final_active_equity=${final_equity:.2f})")


def _stage_gates(only: str | None) -> None:
    for seed_offset, candidate_id in enumerate(CONFIG_IDS):
        if only is not None and candidate_id != only:
            continue
        path = RESULTS_DIR / f"{candidate_id}.json"
        payload = _load_json(path)
        result = Wave10Result(
            equity=_series(payload["equity"]),
            positions=_series(payload["positions"]),
            turnover=_series(payload["turnover"]),
            trade_returns=_series(payload["trade_returns"]),
            max_concurrent_positions=int(payload["metadata"]["max_concurrent_positions"]),
            symbols_used=tuple(payload["metadata"]["symbols_used"]),
        )
        config = get_config(candidate_id)
        report = gates_mod.evaluate_gates(config, result, seed_offset)
        payload["gates"] = {
            "gate_a": report.gate_a,
            "gate_b": report.gate_b,
            "gate_c": report.gate_c,
            "gate_d": report.gate_d,
            "gate_e": report.gate_e,
            "overall": report.overall,
            "failure_reasons": list(report.failure_reasons),
        }
        payload["regime_breakdown"] = regime_mod.regime_breakdown(result)
        _save_json(path, payload)
        print(f"gates: {candidate_id} -> {report.overall} (reasons={list(report.failure_reasons)})")


def _stage_report() -> None:
    write_wave10_report(RESULTS_DIR, REPORT_DIR, REGISTRY_PATH)
    print(f"report: wrote {REPORT_DIR / 'wave10_report.md'} and {REGISTRY_PATH}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Wave-10 $100 fixed-fraction carry pipeline")
    parser.add_argument("--stage", required=True, type=Stage, choices=tuple(Stage))
    parser.add_argument("--only", choices=CONFIG_IDS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        match args.stage:
            case Stage.RUN:
                _stage_run(args.only)
            case Stage.GATES:
                _stage_gates(args.only)
            case Stage.REPORT:
                _stage_report()
            case Stage.ALL:
                _stage_run(args.only)
                _stage_gates(args.only)
                _stage_report()
            case unreachable:
                assert_never(unreachable)
    except (FileNotFoundError, Wave10Error, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
