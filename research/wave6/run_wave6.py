# Command-line orchestration for Wave-6 fetch, run, gate, and report stages.

from __future__ import annotations

import argparse
from enum import StrEnum
from pathlib import Path
import sys
from typing import assert_never

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

from research.wave1.common import save_json, strategy_payload
from research.wave6.engine_w6 import RESULTS_DIR
from research.wave6.fetch_w6 import ensure_output_dirs, stage_fetch
from research.wave6.gates_w6 import run_gates
from research.wave6.reporting_w6 import write_reports
from research.wave6.strategies_w6 import run_candidates


class Stage(StrEnum):
    FETCH = "fetch"
    RUN = "run"
    GATES = "gates"
    REPORT = "report"


def _stage_run() -> None:
    ensure_output_dirs()
    candidates = run_candidates()
    for result in candidates:
        save_json(RESULTS_DIR / f"{result.candidate_id}.json", strategy_payload(result))
    print(f"run: wrote {len(candidates)} candidate results")


def _stage_gates() -> None:
    summary = run_gates()
    print(f"gates: standard={summary['standard_verdicts']}; survivors={summary['survivors']}")


def _stage_report() -> None:
    path = write_reports()
    print(f"report: {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Wave-6 preregistered research pipeline")
    parser.add_argument("--stage", required=True, type=Stage, choices=tuple(Stage))
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    match args.stage:
        case Stage.FETCH:
            stage_fetch(args.force)
        case Stage.RUN:
            _stage_run()
        case Stage.GATES:
            _stage_gates()
        case Stage.REPORT:
            _stage_report()
        case unreachable:
            assert_never(unreachable)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
