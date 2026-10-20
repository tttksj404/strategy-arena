from __future__ import annotations

import argparse
from enum import StrEnum
from datetime import timedelta
import hashlib
from pathlib import Path
import sys
from typing import Final, assert_never

import pandas as pd  # noqa: PANDAS_OK

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

from research.wave1.common import CACHE_DIR as WAVE1_CACHE_DIR
from research.wave1.common import JsonValue, PipelineError, ensure_output_dirs, integrity_report, load_frame, load_json, report_payload, save_json, strategy_payload, validate_symbol
from research.wave2.funding import W2_FUNDING_CANDIDATES, W2_FUNDING_IDS, run_funding_variant
from research.wave2.gates import evaluate_result_file_wave2
from research.wave2.reporting import write_wave2_reports
from research.wave2.session import W2G_SYMBOLS, run_w2g


BASE_DIR: Final = Path(__file__).resolve().parent
RESULTS_DIR: Final = BASE_DIR / "results"
REPORT_DIR: Final = BASE_DIR / "report"
REGISTRY_PATH: Final = BASE_DIR / "REGISTRY.md"
ALL_IDS: Final = (*W2_FUNDING_IDS, "W2g")


class Stage(StrEnum):
    FETCH = "fetch"
    RUN = "run"
    GATES = "gates"
    REPORT = "report"


def _load_symbols() -> tuple[str, ...]:
    payload = load_json(WAVE1_CACHE_DIR / "universe.json")
    if not isinstance(payload, dict) or not isinstance(payload.get("symbols"), list):
        raise PipelineError("wave-1 cache universe.json is invalid")
    return tuple(str(symbol) for symbol in payload["symbols"])


def _required_cache_files(symbols: tuple[str, ...]) -> set[Path]:
    required = {WAVE1_CACHE_DIR / "universe.json"}
    for symbol in symbols:
        validate_symbol(symbol)
        required.update(
            {
                WAVE1_CACHE_DIR / f"binance_spot_{symbol}_1d.csv.gz",
                WAVE1_CACHE_DIR / f"binance_fapi_{symbol}_1d.csv.gz",
                WAVE1_CACHE_DIR / f"binance_funding_{symbol}.csv.gz",
            }
        )
    for symbol in W2G_SYMBOLS:
        required.update(
            {
                WAVE1_CACHE_DIR / f"bitget_{symbol}USDT_1H.csv.gz",
                WAVE1_CACHE_DIR / f"bitget_funding_{symbol}USDT.csv.gz",
            }
        )
    return required


def _expected_interval(path: Path) -> timedelta:
    if "_1H" in path.name:
        return timedelta(hours=1)
    if "funding" in path.name:
        return timedelta(hours=8)
    return timedelta(days=1)


def _cache_record(path: Path) -> dict[str, JsonValue]:
    if path.suffix == ".json":
        payload = load_json(path)
        symbols = payload.get("symbols") if isinstance(payload, dict) else None
        source_integrity = isinstance(symbols, list) and all(isinstance(symbol, str) and validate_symbol(symbol) for symbol in symbols)
        return {
            "file": path.name,
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "rows": len(symbols) if isinstance(symbols, list) else 0,
            "start": None,
            "end": None,
            "integrity": {"valid": source_integrity},
            "source_integrity": source_integrity,
        }
    frame = load_frame(path)
    integrity = integrity_report(frame, _expected_interval(path))
    return {
        "file": path.name,
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "rows": len(frame),
        "start": pd.Timestamp(frame.index.min()).isoformat() if len(frame) else None,
        "end": pd.Timestamp(frame.index.max()).isoformat() if len(frame) else None,
        "integrity": report_payload(integrity),
        "source_integrity": integrity.valid and len(frame) > 0,
    }


def _require_manifest() -> None:
    manifest = load_json(BASE_DIR / "cache_manifest.json")
    if not isinstance(manifest, dict) or manifest.get("network_calls") is not False or manifest.get("source_integrity") is not True:
        raise PipelineError("wave-1 cache manifest is missing or failed integrity verification")


def _stage_fetch(force: bool) -> None:
    ensure_output_dirs()
    symbols = _load_symbols()
    required = _required_cache_files(symbols)
    missing = sorted(str(path.relative_to(WAVE1_CACHE_DIR)) for path in required if not path.exists())
    if missing:
        raise PipelineError(f"wave-1 cache incomplete; orchestrator must collect: {', '.join(missing[:8])}")
    records = [_cache_record(path) for path in sorted(required)]
    source_integrity = all(record["source_integrity"] is True for record in records)
    if not source_integrity:
        raise PipelineError("wave-1 cache failed structural integrity verification")
    save_json(
        BASE_DIR / "cache_manifest.json",
        {
            "source": "research/wave1/cache",
            "network_calls": False,
            "force_ignored": force,
            "required_files": len(required),
            "source_integrity": source_integrity,
            "files": records,
        },
    )
    print(f"fetch: verified wave-1 cache only ({len(required)} files)")


def _stage_run(only: str | None) -> None:
    ensure_output_dirs()
    _require_manifest()
    symbols = _load_symbols()
    candidates = tuple(candidate for candidate in W2_FUNDING_CANDIDATES if only is None or candidate.candidate_id == only)
    if only == "W2g":
        result = run_w2g(WAVE1_CACHE_DIR)
        save_json(RESULTS_DIR / "W2g.json", strategy_payload(result))
        return
    if not candidates:
        raise PipelineError(f"unknown candidate: {only}")
    for candidate in candidates:
        result = run_funding_variant(WAVE1_CACHE_DIR, symbols, candidate)
        save_json(RESULTS_DIR / f"{candidate.candidate_id}.json", strategy_payload(result))


def _stage_gates(only: str | None) -> None:
    btc_path = WAVE1_CACHE_DIR / "binance_fapi_BTCUSDT_1d.csv.gz"
    btc_returns = load_frame(btc_path)["close"].pct_change() if btc_path.exists() else pd.Series(dtype=float)
    selected = tuple(candidate_id for candidate_id in ALL_IDS if only is None or candidate_id == only)
    for candidate_id in selected:
        path = RESULTS_DIR / f"{candidate_id}.json"
        if not path.exists():
            raise FileNotFoundError(path)
        evaluate_result_file_wave2(path, btc_returns)


def _stage_report() -> None:
    write_wave2_reports(RESULTS_DIR, REPORT_DIR, REGISTRY_PATH, ALL_IDS)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Wave-2 preregistered research pipeline")
    parser.add_argument("--stage", required=True, type=Stage, choices=tuple(Stage))
    parser.add_argument("--only", choices=ALL_IDS)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    match args.stage:
        case Stage.FETCH:
            _stage_fetch(args.force)
        case Stage.RUN:
            _stage_run(args.only)
        case Stage.GATES:
            _stage_gates(args.only)
        case Stage.REPORT:
            _stage_report()
        case unreachable:
            assert_never(unreachable)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
