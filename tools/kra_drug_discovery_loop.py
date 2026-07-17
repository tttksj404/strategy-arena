#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final


ROOT: Final = Path(__file__).resolve().parents[1]
SEARCH: Final = ROOT / "tools" / "kra_drug_discovery_search.py"
REPORT: Final = ROOT / "runs" / "kra_drug_discovery_results.json"
LOCK: Final = ROOT / "runs" / "kra_drug_discovery_loop.lock"
STOP: Final = ROOT / "runs" / "kra_drug_discovery.stop"
MINIMUM_CANDIDATES: Final = 20_000
MINIMUM_GENERATIONS: Final = 5


@dataclass(frozen=True, slots=True)
class SearchPolicy:
    candidates: int = MINIMUM_CANDIDATES
    generations: int = MINIMUM_GENERATIONS
    beam_width: int = 64
    maximum_market_weight: float = 0.15

    def __post_init__(self) -> None:
        if self.candidates < MINIMUM_CANDIDATES:
            raise ValueError("every cycle must assay at least 20,000 candidates")
        if self.generations < MINIMUM_GENERATIONS:
            raise ValueError("every cycle must run at least five generations")
        if self.beam_width < 1:
            raise ValueError("beam width must be positive")
        if not 0.0 <= self.maximum_market_weight <= 0.15:
            raise ValueError("market weight must stay between 0% and 15%")


def should_continue(
    report: dict, completed_cycles: int, max_cycles: int
) -> bool:  # noqa: DICT_OK — JSON process boundary
    if bool(report.get("historical_market_parity_pass", False)):
        return False
    return max_cycles == 0 or completed_cycles < max_cycles


def _acquire_lock(path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        try:
            pid = int(path.read_text(encoding="utf-8"))
            os.kill(pid, 0)
        except (OSError, ValueError):
            path.unlink(missing_ok=True)
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        else:
            raise RuntimeError(f"search loop already active: {path}") from error
    os.write(descriptor, str(os.getpid()).encode())
    return descriptor


def _run_cycle(policy: SearchPolicy, report_path: Path) -> dict:  # noqa: DICT_OK
    command = [
        sys.executable,
        str(SEARCH),
        "--report",
        str(report_path),
        "--candidates",
        str(policy.candidates),
        "--generations",
        str(policy.generations),
        "--beam-width",
        str(policy.beam_width),
        "--maximum-market-weight",
        str(policy.maximum_market_weight),
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    return json.loads(report_path.read_text(encoding="utf-8"))


def run_loop(args: argparse.Namespace) -> int:
    policy = SearchPolicy(
        candidates=args.candidates,
        generations=args.generations,
        beam_width=args.beam_width,
        maximum_market_weight=args.maximum_market_weight,
    )
    descriptor = _acquire_lock(args.lock)
    completed = 0
    try:
        while not args.stop_file.exists():
            report = _run_cycle(policy, args.report)
            completed += 1
            if not should_continue(report, completed, args.max_cycles):
                return 0
            if args.interval_seconds > 0:
                time.sleep(args.interval_seconds)
        return 0
    finally:
        os.close(descriptor)
        args.lock.unlink(missing_ok=True)


def main() -> int:
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--lock", type=Path, default=LOCK)
    parser.add_argument("--stop-file", type=Path, default=STOP)
    parser.add_argument("--candidates", type=int, default=MINIMUM_CANDIDATES)
    parser.add_argument("--generations", type=int, default=MINIMUM_GENERATIONS)
    parser.add_argument("--beam-width", type=int, default=64)
    parser.add_argument("--maximum-market-weight", type=float, default=0.15)
    parser.add_argument("--interval-seconds", type=int, default=900)
    parser.add_argument("--max-cycles", type=int, default=0)
    return run_loop(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
