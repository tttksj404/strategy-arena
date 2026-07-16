#!/usr/bin/env python
"""Wave-3 preregistered pipeline CLI."""

from __future__ import annotations

import argparse
from enum import StrEnum
from pathlib import Path
import sys
from typing import Final, assert_never

import pandas as pd  # noqa: PANDAS_OK

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

from research.wave1.common import CACHE_DIR as WAVE1_CACHE_DIR
from research.wave1.common import PipelineError, ensure_output_dirs, load_frame, save_json, strategy_payload
from research.wave2.gates import evaluate_result_file_wave2
from research.wave3.engine import W3_CANDIDATE_IDS, W3_CANDIDATES, load_listings, load_markets, run_candidate
from research.wave3.fetch import WAVE3_CACHE_DIR, run_fetch, verify_manifest
from research.wave3.reporting import write_wave3_reports


BASE_DIR: Final = Path(__file__).resolve().parent
RESULTS_DIR: Final = BASE_DIR / "results"
REPORT_DIR: Final = BASE_DIR / "report"
REGISTRY_PATH: Final = BASE_DIR / "REGISTRY.md"


class Stage(StrEnum):
    FETCH = "fetch"
    RUN = "run"
    GATES = "gates"
    REPORT = "report"


def _require_fetch_manifest() -> None:
    if not (WAVE3_CACHE_DIR / "manifest.json").exists():
        raise PipelineError("wave-3 fetch stage has not produced cache/manifest.json")
    verify_manifest()


def _stage_run(only: str | None) -> None:
    ensure_output_dirs()
    _require_fetch_manifest()
    listings = load_listings()
    markets = load_markets(listings)
    candidates = tuple(candidate for candidate in W3_CANDIDATES if only is None or candidate.candidate_id == only)
    if not candidates:
        raise PipelineError(f"unknown candidate: {only}")
    for candidate in candidates:
        result = run_candidate(markets, candidate)
        save_json(RESULTS_DIR / f"{candidate.candidate_id}.json", strategy_payload(result))
        print(f"run: completed {candidate.candidate_id}")


def _stage_gates(only: str | None) -> None:
    btc_path = WAVE1_CACHE_DIR / "binance_fapi_BTCUSDT_1d.csv.gz"
    btc_returns = load_frame(btc_path)["close"].pct_change() if btc_path.exists() else pd.Series(dtype=float)
    selected = tuple(candidate_id for candidate_id in W3_CANDIDATE_IDS if only is None or candidate_id == only)
    for candidate_id in selected:
        path = RESULTS_DIR / f"{candidate_id}.json"
        if not path.exists():
            raise FileNotFoundError(path)
        evaluate_result_file_wave2(path, btc_returns)


def build_parser() -> argparse.ArgumentParser:
    """Build the four-stage CLI shared by wave-1 and wave-2."""
    parser = argparse.ArgumentParser(description="Wave-3 preregistered research pipeline")
    parser.add_argument("--stage", required=True, type=Stage, choices=tuple(Stage))
    parser.add_argument("--only", choices=W3_CANDIDATE_IDS)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Dispatch one pipeline stage."""
    args = build_parser().parse_args(argv)
    try:
        match args.stage:
            case Stage.FETCH:
                run_fetch(args.force)
            case Stage.RUN:
                _stage_run(args.only)
            case Stage.GATES:
                _stage_gates(args.only)
            case Stage.REPORT:
                write_wave3_reports(RESULTS_DIR, REPORT_DIR, REGISTRY_PATH)
            case unreachable:
                assert_never(unreachable)
    except (FileNotFoundError, PipelineError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
