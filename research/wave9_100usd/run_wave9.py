#!/usr/bin/env python
"""Wave-9 ($100-native single-leg perp) pipeline CLI.

Mirrors the wave3/wave7 --stage run|gates|report convention, minus a fetch stage:
wave9 is cache-only (research/wave3/cache; no network calls), per the task contract.
"""

from __future__ import annotations

import argparse
from enum import StrEnum
import json
import math
from pathlib import Path
import sys
from typing import Final, assert_never

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

from research.wave9_100usd import gates_w9
from research.wave9_100usd.engine_w9 import W9_CANDIDATE_IDS, Wave9Error, run_all
from research.wave9_100usd.reporting_w9 import write_wave9_report


BASE_DIR: Final = Path(__file__).resolve().parent
RESULTS_DIR: Final = BASE_DIR / "results"
REPORT_DIR: Final = BASE_DIR / "report"
REGISTRY_PATH: Final = BASE_DIR / "REGISTRY.md"

CANDIDATE_LEVERAGE: Final = {"W9a": 1.0, "W9b": 2.0, "W9c": 1.0, "W9d": 1.0, "W9e": 1.0, "W9f": 2.0}
CANDIDATE_HOLD_DAYS: Final = {"W9a": 7, "W9b": 7, "W9c": 7, "W9d": 3, "W9e": 3, "W9f": 1}


class Stage(StrEnum):
    RUN = "run"
    GATES = "gates"
    REPORT = "report"


def _json_safe(value):
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _stage_run(only: str | None) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payloads = run_all()
    for candidate_id, payload in payloads.items():
        if only is not None and candidate_id != only:
            continue
        _save_json(RESULTS_DIR / f"{candidate_id}.json", payload)
        print(f"run: completed {candidate_id} (trades={payload['metadata']['trades_executed']}, final_equity=${payload['metadata']['final_equity']:.2f})")


def _stage_gates(only: str | None) -> None:
    for seed_index, candidate_id in enumerate(W9_CANDIDATE_IDS):
        if only is not None and candidate_id != only:
            continue
        path = RESULTS_DIR / f"{candidate_id}.json"
        payload = _load_json(path)
        validation = gates_w9.evaluate_payload(payload, CANDIDATE_LEVERAGE[candidate_id], CANDIDATE_HOLD_DAYS[candidate_id], seed_index)
        payload["validation"] = validation
        _save_json(path, payload)
        print(f"gates: {candidate_id} -> {validation['overall']['status']} ({validation['overall']['passed']}/{validation['overall']['total']}, hard={validation['overall']['hard_gates_status']})")


def _stage_report() -> None:
    write_wave9_report(RESULTS_DIR, REPORT_DIR, REGISTRY_PATH)
    print(f"report: wrote {REPORT_DIR / 'wave9_report.md'} and {REGISTRY_PATH}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Wave-9 $100-native single-leg perp pipeline")
    parser.add_argument("--stage", required=True, type=Stage, choices=tuple(Stage))
    parser.add_argument("--only", choices=W9_CANDIDATE_IDS)
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
            case unreachable:
                assert_never(unreachable)
    except (FileNotFoundError, Wave9Error) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
