#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import engine


def key_for_record(record):
    return (
        str(record.get("date", "")),
        str(record.get("meet", "")),
        str(record.get("race_no", "")).zfill(2),
        str(record.get("board_hash", "")),
    )


def load_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_key_tokens(path):
    key_path = Path(path)
    if not key_path.exists():
        return set()
    with key_path.open(encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def year_of(record):
    year = str(record.get("stnd_yr") or "")
    if year:
        return year
    return str(record.get("date") or "")[:4]


def is_compatible_signal_downgrade(stored_tier, recomputed_tier):
    return (
        stored_tier == "market_trifecta_50_candidate"
        and recomputed_tier == "market_trifecta_watch_low_sample"
    )


def snapshot_training_class(record):
    phase = str(record.get("snapshot_phase") or "")
    source = str(record.get("source") or "")
    has_actual = bool(record.get("actual_order"))
    captured_at = str(record.get("captured_at") or "")
    if phase == "pre_result_market_snapshot" and source != "archive_import":
        return "live_like_pre_result"
    if phase == "post_result_archive_join" or source == "archive_import":
        return "post_result_archive_join"
    if has_actual and not captured_at:
        return "post_result_archive_join"
    if has_actual:
        return "actual_joined_unknown_phase"
    return "unsettled_or_unknown"


def timing_phase(record):
    timing = record.get("market_timing")
    if not isinstance(timing, dict):
        return ""
    return str(timing.get("phase") or "")


def is_v2_snapshot(record):
    return str(record.get("schema") or "") == "kcycle_trifecta_snapshot_v2"


def signal_deployability(signal):
    if not isinstance(signal, dict):
        return "none"
    if signal.get("deployable") is True or signal.get("robust_status") == "passed_oos":
        return "deployable"
    if signal.get("robust_status") == "failed_small_n":
        return "watch_only_failed_small_n"
    return "watch_only"


def audit(records, key_tokens):
    seen = set()
    duplicate_keys = 0
    hash_mismatch = 0
    board_count_mismatch = 0
    stored_signal_mismatch = 0
    stored_signal_compatible_downgrades = 0
    selected_by_year = {}
    hits_by_year = {}
    actual_by_year = {}
    signal_count = 0
    actual_count = 0
    training_classes = {}
    signal_deployability_counts = {}
    actual_joined_live_like_risk = 0
    missing_timing_for_live_like_v2 = 0
    post_start_marked_pre_result = 0
    timing_quality = {
        "live_like_with_timing": 0,
        "live_like_missing_timing": 0,
        "pre_start_snapshots": 0,
        "post_start_snapshots": 0,
    }
    for record in records:
        key = key_for_record(record)
        token = engine._snapshot_key_token(key)
        if token in seen:
            duplicate_keys += 1
        seen.add(token)
        board = {str(k): float(v) for k, v in (record.get("board") or {}).items()}
        board_hash = engine._trifecta_board_hash(board)
        if board_hash != str(record.get("board_hash")):
            hash_mismatch += 1
        if len(board) != int(record.get("board_count") or -1) or len(board) != 210:
            board_count_mismatch += 1
        recomputed_signal = engine._market_trifecta_signal(board)
        stored_signal = record.get("signal")
        training_class = snapshot_training_class(record)
        training_classes[training_class] = training_classes.get(training_class, 0) + 1
        phase = timing_phase(record)
        if training_class == "live_like_pre_result":
            if phase:
                timing_quality["live_like_with_timing"] += 1
            else:
                timing_quality["live_like_missing_timing"] += 1
                if is_v2_snapshot(record):
                    missing_timing_for_live_like_v2 += 1
        if phase == "post_start":
            timing_quality["post_start_snapshots"] += 1
        elif phase in {"early", "mid", "late", "unknown"}:
            timing_quality["pre_start_snapshots"] += 1
        if phase == "post_start" and str(record.get("snapshot_phase") or "") == "pre_result_market_snapshot":
            post_start_marked_pre_result += 1
        signal_class = signal_deployability(stored_signal)
        signal_deployability_counts[signal_class] = signal_deployability_counts.get(signal_class, 0) + 1
        if record.get("actual_order") and training_class == "live_like_pre_result":
            actual_joined_live_like_risk += 1
        stored_tier = stored_signal.get("tier") if isinstance(stored_signal, dict) else None
        recomputed_tier = recomputed_signal.get("tier") if recomputed_signal else None
        if is_compatible_signal_downgrade(stored_tier, recomputed_tier):
            stored_signal_compatible_downgrades += 1
        elif stored_tier != recomputed_tier:
            stored_signal_mismatch += 1
        if recomputed_signal:
            signal_count += 1
        actual = record.get("actual_order")
        if actual:
            actual_count += 1
            year = year_of(record)
            actual_by_year[year] = actual_by_year.get(year, 0) + 1
            if recomputed_signal:
                selected_by_year[year] = selected_by_year.get(year, 0) + 1
                best = "-".join(str(x) for x in recomputed_signal["order"])
                if best == actual:
                    hits_by_year[year] = hits_by_year.get(year, 0) + 1
    by_year = {}
    for year in sorted(actual_by_year):
        selected = selected_by_year.get(year, 0)
        hits = hits_by_year.get(year, 0)
        by_year[year] = {
            "actual_n": actual_by_year[year],
            "selected_n": selected,
            "hits": hits,
            "exact": hits / selected if selected else None,
        }
    missing_index_tokens = sorted(seen - key_tokens)
    extra_index_tokens = sorted(key_tokens - seen)
    critical_failures = {
        "duplicate_keys": duplicate_keys,
        "hash_mismatch": hash_mismatch,
        "board_count_mismatch": board_count_mismatch,
        "stored_signal_mismatch": stored_signal_mismatch,
        "missing_index_tokens": len(missing_index_tokens),
        "extra_index_tokens": len(extra_index_tokens),
        "actual_joined_live_like_risk": actual_joined_live_like_risk,
        "missing_timing_for_live_like_v2": missing_timing_for_live_like_v2,
        "post_start_marked_pre_result": post_start_marked_pre_result,
    }
    return {
        "ok": all(value == 0 for value in critical_failures.values()) and actual_count > 0,
        "records": len(records),
        "unique_keys": len(seen),
        "key_index_tokens": len(key_tokens),
        "actual_count": actual_count,
        "signal_count": signal_count,
        "training_classes": training_classes,
        "timing_quality": timing_quality,
        "signal_deployability": signal_deployability_counts,
        "stored_signal_compatible_downgrades": stored_signal_compatible_downgrades,
        "critical_failures": critical_failures,
        "rule_metrics_by_year": by_year,
        "rule_selected_total": sum(selected_by_year.values()),
        "rule_hits_total": sum(hits_by_year.values()),
        "rule_exact_total": (
            sum(hits_by_year.values()) / sum(selected_by_year.values())
            if sum(selected_by_year.values()) else None
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshots", default=str(ROOT / "data" / "kcycle_trifecta_snapshots.jsonl"))
    parser.add_argument("--keys", default="")
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    key_path = args.keys or f"{args.snapshots}.keys"
    payload = audit(load_jsonl(args.snapshots), load_key_tokens(key_path))
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    if not payload["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
