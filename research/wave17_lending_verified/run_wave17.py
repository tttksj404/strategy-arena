#!/usr/bin/env python
"""Wave-17 (lending-rate-history realized-yield verification, F0-F3 + F_min) pipeline CLI.
Mirrors research/wave16_duallayer/run_wave16.py's --stage convention: `fetch` is network-bound
(OKX lending-rate-history + a fresh lending-rate-summary -- fetch17.py); `run`/`report` are
cache-only (recompute17.py / reporting17.py). See research/wave17_lending_verified/SPEC.md for
the pre-registered contract.
"""

from __future__ import annotations

import argparse
from enum import StrEnum
from pathlib import Path
import sys
from typing import Final, assert_never

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

from research.wave17_lending_verified import fetch17, recompute17
from research.wave17_lending_verified.reporting17 import write_wave17_report

BASE_DIR: Final = Path(__file__).resolve().parent
RESULTS_DIR: Final = BASE_DIR / "results"
REPORT_DIR: Final = BASE_DIR / "report"
CACHE_DIR: Final = BASE_DIR / "cache"
REGISTRY_PATH: Final = BASE_DIR / "REGISTRY.md"


class Stage(StrEnum):
    FETCH = "fetch"
    RUN = "run"
    REPORT = "report"
    ALL = "all"


def _stage_fetch() -> None:
    fetch17.collect_lending_realized()


def _stage_run() -> None:
    recompute17.run_and_save()


def _stage_report() -> None:
    write_wave17_report(RESULTS_DIR, REPORT_DIR, REGISTRY_PATH, CACHE_DIR)
    print(f"report: wrote {REPORT_DIR / 'wave17_report.md'} and {REGISTRY_PATH}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Wave-17 lending-realized (lendingRate vs avgRate) verification pipeline")
    parser.add_argument("--stage", required=True, type=Stage, choices=tuple(Stage))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        match args.stage:
            case Stage.FETCH:
                _stage_fetch()
            case Stage.RUN:
                _stage_run()
            case Stage.REPORT:
                _stage_report()
            case Stage.ALL:
                _stage_fetch()
                _stage_run()
                _stage_report()
            case unreachable:
                assert_never(unreachable)
    except (FileNotFoundError, RuntimeError, ValueError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
