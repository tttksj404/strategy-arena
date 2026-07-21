#!/usr/bin/env python
"""Wave-7 preregistered pipeline CLI (carry+momentum combination).

Mirrors the wave2/wave3 three/four-stage CLI convention (--stage run|gates|report),
minus a fetch stage: wave7 consumes only data that is already cached on disk (W2c's
and W3c's own result JSON, plus the wave-1 funding/price CSV caches), so there is
nothing new to fetch.
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

from research.wave7 import deepval_w7, engine_w7
from research.wave7.engine_w7 import W7_CANDIDATE_IDS, Wave7Error
from research.wave7.reporting_w7 import write_wave7_report


BASE_DIR: Final = Path(__file__).resolve().parent
REPO_ROOT: Final = BASE_DIR.parents[1]
RESULTS_DIR: Final = BASE_DIR / "results"
REPORT_DIR: Final = BASE_DIR / "report"
REGISTRY_PATH: Final = BASE_DIR / "REGISTRY.md"

W2C_PATH: Final = REPO_ROOT / "research" / "wave2" / "results" / "W2c.json"
W3C_PATH: Final = REPO_ROOT / "research" / "wave3" / "results" / "W3c.json"
WAVE1_CACHE_DIR: Final = REPO_ROOT / "research" / "wave1" / "cache"


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
    payloads = engine_w7.run_all(W2C_PATH, W3C_PATH, WAVE1_CACHE_DIR)
    for candidate_id, payload in payloads.items():
        if only is not None and candidate_id != only:
            continue
        _save_json(RESULTS_DIR / f"{candidate_id}.json", payload)
        print(f"run: completed {candidate_id}")


def _stage_gates(only: str | None) -> None:
    carry_returns, _momentum_returns = engine_w7.load_component_returns(W2C_PATH, W3C_PATH)
    carry_alone_equity = engine_w7.equity_from_returns(carry_returns)
    carry_alone_metrics = deepval_w7.standard_metrics(carry_alone_equity)
    carry_alone_oos_return = deepval_w7.oos_dormant_return(carry_alone_equity)
    for seed_index, candidate_id in enumerate(W7_CANDIDATE_IDS):
        if only is not None and candidate_id != only:
            continue
        path = RESULTS_DIR / f"{candidate_id}.json"
        payload = _load_json(path)
        equity = engine_w7.series_from_payload(payload["equity"])
        combined_returns = engine_w7.series_from_payload(payload["trade_returns"])
        deep_result = deepval_w7.evaluate_candidate(
            candidate_id,
            equity,
            combined_returns,
            carry_returns,
            carry_alone_metrics,
            carry_alone_oos_return,
            seed_index,
        )
        payload["deep_validation"] = deep_result
        payload["gates"] = deep_result["gates"]
        _save_json(path, payload)
        print(f"gates: {candidate_id} -> {deep_result['overall']['status']} ({deep_result['overall']['passed_gates']}/{deep_result['overall']['total_gates']})")


def _stage_report() -> None:
    write_wave7_report(RESULTS_DIR, REPORT_DIR, REGISTRY_PATH)
    print(f"report: wrote {REPORT_DIR / 'wave7_report.md'} and {REGISTRY_PATH}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Wave-7 preregistered carry+momentum combination pipeline")
    parser.add_argument("--stage", required=True, type=Stage, choices=tuple(Stage))
    parser.add_argument("--only", choices=W7_CANDIDATE_IDS)
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
    except (FileNotFoundError, Wave7Error, deepval_w7.Wave7ValidationError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
