from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import sys
from typing import Final, TypeAlias

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.validation.deep_stats import (
    BootstrapResult,
    BlockBootstrapResult,
    DeepValidationError,
    DsrResult,
    FundingComparison,
    LeaveOneOutRow,
    TimedValue,
    block_bootstrap,
    compare_funding,
    deflated_sharpe,
    leave_one_year_out,
    trade_bootstrap,
)


CANDIDATES: Final = ("W2c", "F1e", "F1f", "W3c", "W3d")
CARRY_THRESHOLDS: Final = {"F1e": 0.08, "W2c": 0.15}
NATIVE_REQUIRED_DAYS: Final = 133
JsonValue: TypeAlias = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]


def _timestamp(raw: str | int | float) -> datetime:
    if isinstance(raw, (int, float)):
        return datetime.fromtimestamp(raw / 1000.0, tz=timezone.utc)
    text = raw.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(text)
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


def _series(payload: dict[str, JsonValue], key: str) -> tuple[TimedValue, ...]:
    raw = payload.get(key)
    if not isinstance(raw, list):
        raise DeepValidationError(f"result field {key!r} is not a list")
    values: list[TimedValue] = []
    for item in raw:
        if not isinstance(item, dict) or not isinstance(item.get("timestamp"), (str, int, float)) or not isinstance(item.get("value"), (int, float)):
            continue
        values.append(TimedValue(_timestamp(item["timestamp"]), float(item["value"])))
    if not values:
        raise DeepValidationError(f"result field {key!r} is empty")
    return tuple(sorted(values, key=lambda item: item.timestamp))


def _load_result(path: Path) -> tuple[tuple[TimedValue, ...], tuple[TimedValue, ...], dict[str, JsonValue]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise DeepValidationError(f"result is not an object: {path}")
    metadata = payload.get("metadata")
    return _series(payload, "equity"), _series(payload, "trade_returns"), metadata if isinstance(metadata, dict) else {}


def _funding_file(path: Path) -> tuple[TimedValue, ...]:
    values: list[TimedValue] = []
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            raw_timestamp = row.get("timestamp") or row.get("")
            raw_value = row.get("funding_rate")
            if raw_timestamp is None or raw_value is None:
                continue
            try:
                values.append(TimedValue(_timestamp(raw_timestamp), float(raw_value)))
            except (TypeError, ValueError, OverflowError):
                continue
    return tuple(sorted(values, key=lambda item: item.timestamp))


def _load_funding(cache_dir: Path) -> tuple[dict[str, tuple[TimedValue, ...]], dict[str, tuple[TimedValue, ...]]]:
    bitget: dict[str, tuple[TimedValue, ...]] = {}
    binance: dict[str, tuple[TimedValue, ...]] = {}
    for path in cache_dir.glob("bitget_funding_*.csv.gz"):
        bitget[path.name.removeprefix("bitget_funding_").removesuffix(".csv.gz")] = _funding_file(path)
    for path in cache_dir.glob("binance_funding_*.csv.gz"):
        binance[path.name.removeprefix("binance_funding_").removesuffix(".csv.gz")] = _funding_file(path)
    return bitget, binance


def _cache_integrity(root: Path, cache_dir: Path) -> dict[str, JsonValue]:
    manifest_path = root / "research" / "wave2" / "cache_manifest.json"
    if not manifest_path.is_file():
        return {"status": "UNDETERMINED", "reason": "wave2 cache manifest is missing"}
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return {"status": "FAIL", "reason": f"manifest unreadable: {error}"}
    if not isinstance(payload, dict) or payload.get("network_calls") is not False or payload.get("source_integrity") is not True:
        return {"status": "FAIL", "reason": "manifest does not certify cache-only source integrity"}
    raw_files = payload.get("files")
    if not isinstance(raw_files, list):
        return {"status": "FAIL", "reason": "manifest files is not a list"}
    records: dict[str, dict[str, JsonValue]] = {}
    for raw_file in raw_files:
        if isinstance(raw_file, dict) and isinstance(raw_file.get("file"), str):
            records[raw_file["file"]] = raw_file
    checked = 0
    mismatches: list[str] = []
    for path in sorted(cache_dir.glob("*_funding_*.csv.gz")):
        record = records.get(path.name)
        if record is None:
            mismatches.append(f"missing manifest entry: {path.name}")
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        if record.get("bytes") != path.stat().st_size or record.get("sha256") != digest.hexdigest():
            mismatches.append(f"hash/size mismatch: {path.name}")
        else:
            checked += 1
    return {
        "status": "PASS" if not mismatches else "FAIL",
        "manifest": str(manifest_path.relative_to(root)),
        "checked_files": checked,
        "mismatches": mismatches,
    }


JsonInput: TypeAlias = JsonValue | BootstrapResult | BlockBootstrapResult | DsrResult | FundingComparison | LeaveOneOutRow | tuple[LeaveOneOutRow, ...] | tuple[JsonValue, ...]


def _dataclass_json(value: JsonInput) -> JsonValue:
    if hasattr(value, "__dataclass_fields__"):
        return {key: _dataclass_json(item) for key, item in asdict(value).items()}
    if isinstance(value, tuple):
        return [_dataclass_json(item) for item in value]
    if isinstance(value, list):
        return [_dataclass_json(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _dataclass_json(item) for key, item in value.items()}
    if isinstance(value, float) and (value != value or abs(value) == float("inf")):
        return None
    return value if isinstance(value, (str, int, float, bool)) or value is None else str(value)


def _status(value: bool) -> str:
    return "PASS" if value else "FAIL"


def _candidate_result(
    candidate: str,
    result_dir: Path,
    cache_dir: Path,
    funding: tuple[dict[str, tuple[TimedValue, ...]], dict[str, tuple[TimedValue, ...]]],
    cache_integrity: dict[str, JsonValue],
) -> dict[str, JsonValue]:
    equity, trades, metadata = _load_result(result_dir / f"{candidate}.json")
    bootstrap = trade_bootstrap(tuple(item.value for item in trades), 20_260_715 + CANDIDATES.index(candidate) * 101)
    loo = leave_one_year_out(equity)
    dsr = deflated_sharpe(equity, trials=28)
    blocks = block_bootstrap(trades, 20_260_715 + CANDIDATES.index(candidate) * 103)
    bitget, binance = funding
    native: FundingComparison | None = None
    if candidate in CARRY_THRESHOLDS:
        native = compare_funding(bitget, binance, CARRY_THRESHOLDS[candidate])
    native_payload = _dataclass_json(native) if native is not None else {"status": "N/A", "reason": "momentum candidate; native carry check not applicable"}
    if isinstance(native_payload, dict):
        native_payload["required_coverage_days"] = NATIVE_REQUIRED_DAYS
        native_payload["coverage_sufficient"] = native is not None and native.coverage_days >= NATIVE_REQUIRED_DAYS
    # F1f shares the wave-1 F1 engine with F1e, so its trade_returns carry the same closed-trade semantics.
    semantic = "closed_trade_net_simple_return" if candidate in {"F1e", "F1f", "W2c"} else "daily_active_net_simple_return"
    mc_gate = bootstrap.unit.p05 > 300.0 and bootstrap.unit.ruin_probability < 0.05
    mc_status = _status(mc_gate) if semantic == "closed_trade_net_simple_return" else "UNDETERMINED"
    native_status = native is None or (cache_integrity.get("status") == "PASS" and native.sign_agreement > 0.80 and native.coverage_days >= NATIVE_REQUIRED_DAYS)
    loo_status, block_status = len(loo) >= 2, blocks.block_count >= 2
    overall_status = mc_status == "PASS" and dsr.score > 0.0 and native_status and loo_status and block_status
    return {
        "candidate_id": candidate,
        "family": metadata.get("intended_factor", "unknown"),
        "source": {"result": str((result_dir / f"{candidate}.json").relative_to(result_dir.parent.parent.parent)), "cache": str(cache_dir.relative_to(result_dir.parent.parent.parent))},
        "period": {"equity_points": len(equity), "trade_count": len(trades), "start": equity[0].timestamp.isoformat(), "end": equity[-1].timestamp.isoformat()},
        "input_contract": {"trade_return_semantic": semantic, "trade_bootstrap_eligible": semantic == "closed_trade_net_simple_return"},
        "cache_integrity": cache_integrity,
        "bootstrap_mc": _dataclass_json(bootstrap),
        "leave_one_year_out": _dataclass_json(loo),
        "deflated_sharpe": _dataclass_json(dsr),
        "bitget_native": native_payload,
        "regime_block_bootstrap": _dataclass_json(blocks),
        "criteria": {
            "mc": {"status": mc_status, "p05_gt_300": mc_gate, "ruin_lt_5pct": bootstrap.unit.ruin_probability < 0.05, "input_semantic": semantic},
            "dsr": {"status": _status(dsr.score > 0.0), "score_gt_0": dsr.score > 0.0},
            "bitget_sign": None if native is None else {"status": _status(native_status), "sign_gt_80pct": native.sign_agreement > 0.80, "coverage_sufficient": native.coverage_days >= NATIVE_REQUIRED_DAYS, "cache_integrity": cache_integrity.get("status")},
            "loo": {"status": _status(loo_status), "years": len(loo)},
            "regime_blocks": {"status": _status(block_status), "blocks": blocks.block_count},
        },
        "overall": {"status": _status(overall_status), "required_checks": ["mc", "dsr", "bitget_sign" if native is not None else "not_applicable", "loo", "regime_blocks"]},
    }

def run(root: Path) -> tuple[dict[str, JsonValue], ...]:
    result_dir = root / "research" / "wave1" / "results"
    cache_dir = root / "research" / "wave1" / "cache"
    output_dir = root / "research" / "validation" / "results"
    funding = _load_funding(cache_dir)
    cache_integrity = _cache_integrity(root, cache_dir)
    results: list[dict[str, JsonValue]] = []
    for candidate in CANDIDATES:
        source_dir = result_dir if candidate in ("F1e", "F1f") else root / "research" / ("wave2" if candidate == "W2c" else "wave3") / "results"
        result = _candidate_result(candidate, source_dir, cache_dir, funding, cache_integrity)
        results.append(result)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / f"deep_{candidate}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    from research.validation.deep_report import write_report
    write_report(results, root / "research" / "validation" / "DEEP_REPORT.md")
    return tuple(results)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run cache-only deep watchlist validation")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    run(args.root.resolve())
    print(f"wrote {len(CANDIDATES)} candidate JSON files and research/validation/DEEP_REPORT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

