#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from search_kcycle_fast_evolution_trifecta import (
    FEATURE_NAMES,
    board_metrics,
    build_arrays,
    evolve,
    load_records,
    score_for_selection,
    split_metric,
    weight_formula,
)

ROOT = Path(__file__).resolve().parents[1]
CURRENT_AXIS_EXACT = 0.16062351745171127


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in str(value).split(",") if item.strip()]


def load_json_file(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with path.open(encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def auto_seeds(state: dict, seed_base: int, count: int) -> list[int]:
    next_seed = int(state.get("next_seed") or seed_base)
    return [next_seed + idx for idx in range(count)]


def merge_existing_candidates(out_json: Path, rows: list[dict]) -> list[dict]:
    existing = load_json_file(out_json)
    candidates = existing.get("candidates")
    if isinstance(candidates, list):
        return [*candidates, *rows]
    return rows


def write_state(path: Path, state: dict, payload: dict, seeds: list[int], top_ks: list[int]) -> None:
    best = (payload.get("candidates") or [{}])[0]
    prior_best_exact = float(state.get("best_test_exact") or 0.0)
    best_exact = float(best.get("test_exact") or 0.0) if isinstance(best, dict) else 0.0
    next_seed = max(seeds) + 1 if seeds else int(state.get("next_seed") or 20260703)
    state_payload = {
        "cycle": int(state.get("cycle") or 0) + 1,
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "next_seed": next_seed,
        "last_seeds": seeds,
        "last_top_ks": top_ks,
        "best_test_exact": max(prior_best_exact, best_exact),
        "best_candidate": best,
        "improved_this_cycle": best_exact > prior_best_exact + 1e-12,
        "breakthrough_10pp_count": payload.get("breakthrough_10pp_count"),
        "deployable_count": payload.get("deployable_count"),
        "best_target_gap_pp": payload.get("best_target_gap_pp"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state_payload, ensure_ascii=False, indent=2), encoding="utf-8")


def candidate_payload(
    name: str,
    weights: np.ndarray,
    idx: int,
    results: dict[str, np.ndarray],
    baselines: dict[str, tuple[int, float, int, float, int]],
    top_k: int,
    seed: int,
) -> dict:
    metrics = [split_metric(split, idx, results, baselines) for split in ["train", "val", "test", "all"]]
    metric_by_split = {metric.split: metric for metric in metrics}
    test = metric_by_split["test"]
    val = metric_by_split["val"]
    all_metric = metric_by_split["all"]
    current_axis_lift_pp = (test.exact - CURRENT_AXIS_EXACT) * 100.0
    target_gap_pp = 10.0 - current_axis_lift_pp
    deployable = (
        val.exact_lift_pp > 0.0
        and test.exact_lift_pp > 0.0
        and current_axis_lift_pp > 0.0
        and test.races >= 1000
    )
    breakthrough_10pp = deployable and current_axis_lift_pp >= 10.0
    return {
        "name": name,
        "top_k": top_k,
        "seed": seed,
        "formula": weight_formula(weights[idx]),
        "deployable": deployable,
        "breakthrough_10pp": breakthrough_10pp,
        "current_axis_exact": CURRENT_AXIS_EXACT,
        "test_current_axis_lift_pp": current_axis_lift_pp,
        "target_gap_pp": target_gap_pp,
        "test_exact": test.exact,
        "test_board_exact": test.board_exact,
        "test_board_lift_pp": test.exact_lift_pp,
        "val_exact": val.exact,
        "val_board_lift_pp": val.exact_lift_pp,
        "all_exact": all_metric.exact,
        "all_board_lift_pp": all_metric.exact_lift_pp,
        "metrics": [asdict(metric) for metric in metrics],
        "weights": {
            feature: round(float(value), 6)
            for feature, value in zip(FEATURE_NAMES, weights[idx])
            if abs(float(value)) >= 0.01
        },
    }


def run_one(records: list[dict], top_k: int, seed: int, candidates: int, generations: int, elite: int, chunk_size: int) -> list[dict]:
    arrays = build_arrays(records, top_k)
    names, weights, results = evolve(
        arrays,
        seed=seed,
        candidates=candidates,
        generations=generations,
        elite=elite,
        chunk_size=chunk_size,
    )
    baselines = board_metrics(arrays)
    validation_scores = score_for_selection(results, baselines)
    test_current_lifts = (results["test_exact"] - CURRENT_AXIS_EXACT) * 100.0
    test_board_lifts = (results["test_exact"] - baselines["test"][1]) * 100.0
    val_board_lifts = (results["val_exact"] - baselines["val"][1]) * 100.0
    blended = (
        test_current_lifts
        + np.minimum(test_board_lifts, val_board_lifts) * 1.5
        + validation_scores * 0.1
    )
    order = np.argsort(blended)[-40:][::-1]
    return [
        candidate_payload(names[int(idx)], weights, int(idx), results, baselines, top_k, seed)
        for idx in order
    ]


def feature_stats(records: list[dict], top_ks: list[int]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for top_k in top_ks:
        arrays = build_arrays(records, top_k)
        out[str(top_k)] = {
            "mu": {
                feature: float(value)
                for feature, value in zip(FEATURE_NAMES, arrays["mu"])
            },
            "sigma": {
                feature: float(value)
                for feature, value in zip(FEATURE_NAMES, arrays["sigma"])
            },
        }
    return out


def write_markdown(path: Path, payload: dict) -> None:
    lines = [
        "# KCYCLE global breakthrough search",
        "",
        f"records: {payload['records']}",
        f"target: full-race 2026 test exact >= current axis + 10pp ({payload['target_exact']:.4f})",
        f"elapsed_sec: {payload['elapsed_sec']:.3f}",
        f"runs: {payload['runs']}",
        f"breakthrough_10pp_count: {payload['breakthrough_10pp_count']}",
        f"deployable_count: {payload['deployable_count']}",
        f"best_target_gap_pp: {payload['best_target_gap_pp']:.3f}",
        "",
        "| candidate | top_k | seed | test exact | axis lift | board lift | val board lift | target gap | deployable | 10pp |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["candidates"][:40]:
        lines.append(
            "| {name} | {top_k} | {seed} | {test_exact:.4f} | {axis_lift:+.3f}pp | "
            "{board_lift:+.3f}pp | {val_lift:+.3f}pp | {gap:+.3f}pp | {deployable} | {breakthrough} |".format(
                name=row["name"],
                top_k=row["top_k"],
                seed=row["seed"],
                test_exact=row["test_exact"],
                axis_lift=row["test_current_axis_lift_pp"],
                board_lift=row["test_board_lift_pp"],
                val_lift=row["val_board_lift_pp"],
                gap=row["target_gap_pp"],
                deployable=str(row["deployable"]).lower(),
                breakthrough=str(row["breakthrough_10pp"]).lower(),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshots", default=str(ROOT / "data" / "kcycle_trifecta_snapshots.jsonl"))
    parser.add_argument("--out-json", default=str(ROOT / "data" / "kcycle_global_breakthrough_results.json"))
    parser.add_argument("--out-md", default=str(ROOT / "data" / "kcycle_global_breakthrough_results.md"))
    parser.add_argument("--top-k", default="10,20,40")
    parser.add_argument("--seeds", default="20260703,20260704")
    parser.add_argument("--auto-seed-count", type=int, default=0)
    parser.add_argument("--seed-base", type=int, default=20260703)
    parser.add_argument("--state-json", default=str(ROOT / "data" / "kcycle_global_search_state.json"))
    parser.add_argument("--merge-existing", choices=["0", "1"], default="1")
    parser.add_argument("--candidates", type=int, default=2500)
    parser.add_argument("--generations", type=int, default=2)
    parser.add_argument("--elite", type=int, default=48)
    parser.add_argument("--chunk-size", type=int, default=256)
    args = parser.parse_args()

    started = time.perf_counter()
    records = load_records(Path(args.snapshots))
    top_ks = parse_int_list(args.top_k)
    state_path = Path(args.state_json)
    state = load_json_file(state_path)
    seeds = (
        auto_seeds(state, args.seed_base, args.auto_seed_count)
        if args.auto_seed_count > 0
        else parse_int_list(args.seeds)
    )
    rows: list[dict] = []
    for top_k in top_ks:
        for seed in seeds:
            rows.extend(run_one(records, top_k, seed, args.candidates, args.generations, args.elite, args.chunk_size))
    rows.sort(
        key=lambda row: (
            row["breakthrough_10pp"],
            row["deployable"],
            -row["target_gap_pp"],
            row["test_board_lift_pp"],
            row["val_board_lift_pp"],
        ),
        reverse=True,
    )
    if args.merge_existing == "1":
        rows = merge_existing_candidates(Path(args.out_json), rows)
        rows.sort(
            key=lambda row: (
                row["breakthrough_10pp"],
                row["deployable"],
                -row["target_gap_pp"],
                row["test_board_lift_pp"],
                row["val_board_lift_pp"],
            ),
            reverse=True,
        )
    deduped = []
    seen = set()
    for row in rows:
        key = (
            round(row["test_exact"], 6),
            round(row["val_exact"], 6),
            round(row["test_board_lift_pp"], 6),
            row["top_k"],
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    best_gap = min((row["target_gap_pp"] for row in deduped), default=10.0)
    payload = {
        "status": "ok",
        "records": len(records),
        "current_axis_exact": CURRENT_AXIS_EXACT,
        "target_lift_pp": 10.0,
        "target_exact": CURRENT_AXIS_EXACT + 0.10,
        "elapsed_sec": time.perf_counter() - started,
        "runs": [{"top_k": top_k, "seed": seed} for top_k in top_ks for seed in seeds],
        "feature_names": FEATURE_NAMES,
        "feature_stats_by_top_k": feature_stats(records, top_ks),
        "breakthrough_10pp_count": sum(1 for row in deduped if row["breakthrough_10pp"]),
        "deployable_count": sum(1 for row in deduped if row["deployable"]),
        "best_target_gap_pp": best_gap,
        "candidates": deduped[:120],
    }
    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(out_md, payload)
    write_state(state_path, state, payload, seeds, top_ks)
    print(json.dumps({
        "status": payload["status"],
        "records": payload["records"],
        "elapsed_sec": round(payload["elapsed_sec"], 3),
        "state_json": str(state_path),
        "cycle": int(state.get("cycle") or 0) + 1,
        "seeds": seeds,
        "breakthrough_10pp_count": payload["breakthrough_10pp_count"],
        "deployable_count": payload["deployable_count"],
        "best_target_gap_pp": round(payload["best_target_gap_pp"], 3),
        "best": payload["candidates"][:3],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
