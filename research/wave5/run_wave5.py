from __future__ import annotations

import argparse
from enum import StrEnum
import hashlib
from pathlib import Path
import re
import sys
from typing import assert_never

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

from research.wave1.common import PipelineError, load_json, save_json, strategy_payload
from research.wave5.gates import run_gates
from research.wave5.reporting import write_reports
from research.wave5.strategies import run_candidates


BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR.parent / "wave1" / "cache"
MANIFEST_PATH = BASE_DIR.parent / "wave2" / "cache_manifest.json"
RESULTS_DIR = BASE_DIR / "results"
REPORT_DIR = BASE_DIR / "report"
REGISTRY_PATH = BASE_DIR / "REGISTRY.md"


class Stage(StrEnum):
    RUN = "run"
    GATES = "gates"
    REPORT = "report"


def _require_cache() -> None:
    manifest = load_json(MANIFEST_PATH)
    if not isinstance(manifest, dict) or manifest.get("network_calls") is not False or manifest.get("source_integrity") is not True:
        raise PipelineError("wave-1 cache manifest is not a verified cache-only source")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise PipelineError("verified wave-1 cache manifest has no file hashes")
    cache_root = CACHE_DIR.resolve()
    for item in files:
        if not isinstance(item, dict):
            raise PipelineError("verified wave-1 cache manifest contains an invalid file entry")
        name = item.get("file")
        expected_bytes = item.get("bytes")
        expected_hash = item.get("sha256")
        relative = Path(name) if isinstance(name, str) else Path("")
        path = (cache_root / relative).resolve()
        if (
            not isinstance(name, str)
            or relative.is_absolute()
            or relative.name != name
            or path.parent != cache_root
            or not isinstance(expected_bytes, int)
            or isinstance(expected_bytes, bool)
            or expected_bytes < 0
            or not isinstance(expected_hash, str)
            or re.fullmatch(r"[0-9a-fA-F]{64}", expected_hash) is None
        ):
            raise PipelineError("verified wave-1 cache manifest contains an unsafe or invalid file entry")
        if not path.is_file():
            raise PipelineError(f"verified wave-1 cache has missing file: {name}")
        if path.stat().st_size != expected_bytes:
            raise PipelineError(f"verified wave-1 cache size mismatch: {name}")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1_048_576), b""):
                digest.update(chunk)
        if digest.hexdigest().lower() != expected_hash.lower():
            raise PipelineError(f"verified wave-1 cache hash mismatch: {name}")


def _stage_run() -> None:
    _require_cache()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    candidates = run_candidates(CACHE_DIR)
    for result in candidates:
        save_json(RESULTS_DIR / f"{result.candidate_id}.json", strategy_payload(result))
    print(f"run: wrote {len(candidates)} cached candidate results")


def _stage_gates() -> None:
    _require_cache()
    selected, rows, combination = run_gates(RESULTS_DIR, CACHE_DIR)
    print(f"gates: selected={selected}; candidates={len(rows)}; W5g={combination['verdict']}")


def _stage_report() -> None:
    _require_cache()
    payload = load_json(RESULTS_DIR / "W5g.json")
    if not isinstance(payload, dict) or not isinstance(payload.get("combination_gates"), dict):
        raise PipelineError("W5g gates must run before report")
    combination = payload["combination_gates"]
    selected = str(payload.get("metadata", {}).get("selected_candidate", "unknown")) if isinstance(payload.get("metadata"), dict) else "unknown"
    write_reports(RESULTS_DIR, REPORT_DIR, REGISTRY_PATH, selected, combination)
    print(f"report: {REPORT_DIR / 'wave5_report.md'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Wave-5 cache-only research pipeline")
    parser.add_argument("--stage", required=True, type=Stage, choices=tuple(Stage))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    match args.stage:
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
