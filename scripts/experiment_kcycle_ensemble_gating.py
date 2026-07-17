#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPTS))

import engine  # noqa: E402
from search_kcycle_fast_evolution_trifecta import (  # noqa: E402
    FEATURE_NAMES,
    board_metrics,
    build_arrays,
    combo_parts,
    load_records,
    split_mask,
)

OUT_JSON = ROOT / "data" / "kcycle_ensemble_gating_results.json"
OUT_MD = ROOT / "data" / "kcycle_ensemble_gating_results.md"
PROGRESS = ROOT / "runs" / "prediction_uplift_progress.md"
TOP_KS = (10, 20, 40)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def append_progress(text: str) -> None:
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    with PROGRESS.open("a", encoding="utf-8") as handle:
        handle.write(f"- {utc_now()} Phase 0: {text}\n")


def combo_code(combo: str) -> int:
    a, b, c = combo_parts(combo)
    return a * 100 + b * 10 + c


def candidate_vector(row: dict) -> np.ndarray:
    weights = row.get("weights") if isinstance(row.get("weights"), dict) else {}
    return np.asarray([float(weights.get(name, 0.0)) for name in FEATURE_NAMES], dtype=np.float32)


def load_selected_candidates(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = [row for row in payload.get("candidates", []) if row.get("deployable")]
    rows.sort(key=lambda row: float(row.get("val_exact") or 0.0), reverse=True)
    return rows[:20]


def ensemble_eval(records: list[dict], selected: list[dict], top_k: int) -> dict:
    arrays = build_arrays(records, top_k)
    weights = np.vstack([candidate_vector(row) for row in selected]).astype(np.float32)
    scores = np.tensordot(arrays["x"], weights.T, axes=([2], [0]))
    rank_order = np.argsort(-scores, axis=1)
    ranks = np.empty_like(rank_order, dtype=np.float32)
    race_idx = np.arange(rank_order.shape[0])[:, None, None]
    cand_idx = np.arange(rank_order.shape[2])[None, None, :]
    ranks[race_idx, rank_order, cand_idx] = np.arange(top_k, dtype=np.float32)[None, :, None]
    avg_rank = ranks.mean(axis=2)
    best_idx = avg_rank.argmin(axis=1)
    pred_codes = arrays["codes"][np.arange(len(best_idx)), best_idx]
    pred_firsts = arrays["firsts"][np.arange(len(best_idx)), best_idx]
    baseline = board_metrics(arrays)
    splits = {}
    for split in ("train", "val", "test", "all"):
        mask = split_mask(arrays["years"], split)
        n = int(mask.sum())
        exact_hits = int((pred_codes[mask] == arrays["actual_codes"][mask]).sum())
        top1_hits = int((pred_firsts[mask] == arrays["actual_firsts"][mask]).sum())
        exact = exact_hits / n if n else 0.0
        top1 = top1_hits / n if n else 0.0
        base = baseline[split]
        splits[split] = {
            "n": n,
            "exact": exact,
            "exact_hits": exact_hits,
            "board_exact": base[1],
            "board_exact_lift_pp": (exact - base[1]) * 100.0,
            "top1": top1,
            "top1_hits": top1_hits,
            "board_top1": base[3],
            "top1_lift_pp": (top1 - base[3]) * 100.0,
        }
    return {"top_k": top_k, "splits": splits}


def best_board_combo(record: dict) -> tuple[str, float, float] | None:
    board = {
        str(combo): float(odds)
        for combo, odds in (record.get("board") or {}).items()
        if float(odds) > 0.0
    }
    if len(board) != 210:
        return None
    ranked = sorted(board.items(), key=lambda item: (item[1], item[0]))
    return ranked[0][0], ranked[0][1], ranked[1][1] / ranked[0][1]


def signal_strength(best_odds: float, gap12: float) -> float:
    return math.log(max(gap12, 1.000001)) * (3.0 / max(best_odds, 0.01))


def gating_rows(records: list[dict]) -> list[dict]:
    rows = []
    for record in records:
        best = best_board_combo(record)
        actual = str(record.get("actual_order") or "")
        if best is None or not actual:
            continue
        combo, best_odds, gap12 = best
        actual_first = combo_parts(actual)[0]
        pred_first = combo_parts(combo)[0]
        rows.append(
            {
                "split": "train"
                if str(record.get("date") or record.get("stnd_yr"))[:4] in {"2018", "2019", "2020", "2021", "2022", "2023"}
                else "val"
                if str(record.get("date") or record.get("stnd_yr"))[:4] in {"2024", "2025"}
                else "test"
                if str(record.get("date") or record.get("stnd_yr"))[:4] == "2026"
                else "other",
                "combo": combo,
                "actual": actual,
                "exact_hit": combo == actual,
                "top1_hit": pred_first == actual_first,
                "best_odds": best_odds,
                "gap12": gap12,
                "strong_pull": best_odds <= 3.0 and gap12 >= 1.2,
                "signal_strength": signal_strength(best_odds, gap12),
            }
        )
    return rows


def metric(items: list[dict]) -> dict:
    n = len(items)
    if n == 0:
        return {"n": 0, "coverage": 0.0, "exact": None, "board_exact": None, "top1": None}
    return {
        "n": n,
        "coverage": 0.0,
        "exact": sum(bool(row["exact_hit"]) for row in items) / n,
        "board_exact": sum(bool(row["exact_hit"]) for row in items) / n,
        "top1": sum(bool(row["top1_hit"]) for row in items) / n,
    }


def precision_coverage(rows: list[dict]) -> list[dict]:
    strong = [row for row in rows if row["strong_pull"]]
    strong.sort(key=lambda row: float(row["signal_strength"]), reverse=True)
    tiers = [("all", rows), ("strong_pull_all", strong)]
    if strong:
        tiers.append(("strong_pull_top50pct", strong[: max(1, math.ceil(len(strong) * 0.50))]))
        tiers.append(("strong_pull_top16pct", strong[: max(1, math.ceil(len(strong) * 0.16))]))
    out = []
    for label, items in tiers:
        row = metric(items)
        row["tier"] = label
        row["coverage"] = len(items) / len(rows) if rows else 0.0
        out.append(row)
    return out


def write_markdown(payload: dict) -> None:
    lines = [
        "# KCYCLE ensemble + strong-pull gating",
        "",
        f"generated_at: {payload['generated_at']}",
        f"records: {payload['records']}",
        f"available_deployable_candidates: {payload['available_deployable_candidates']}",
        "selection: train/val only; selected top 20 by val exact",
        "",
        "## Rank-average ensemble",
        "| top_k | val exact | val board lift | test exact | test board lift | test top1 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["ensemble"]:
        val = row["splits"]["val"]
        test = row["splits"]["test"]
        lines.append(
            f"| {row['top_k']} | {val['exact']:.4f} | {val['board_exact_lift_pp']:+.3f}pp | "
            f"{test['exact']:.4f} | {test['board_exact_lift_pp']:+.3f}pp | {test['top1']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Strong-pull precision-coverage tradeoff",
            "This table is not a lift claim; it reports precision vs coverage for the board favorite under stronger market concentration.",
            "| tier | n | coverage | exact | board_exact | top1 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["gating"]:
        exact = "n/a" if row["exact"] is None else f"{row['exact']:.4f}"
        board = "n/a" if row["board_exact"] is None else f"{row['board_exact']:.4f}"
        top1 = "n/a" if row["top1"] is None else f"{row['top1']:.4f}"
        lines.append(f"| {row['tier']} | {row['n']} | {row['coverage']:.4f} | {exact} | {board} | {top1} |")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    records = load_records(ROOT / "data" / "kcycle_trifecta_snapshots.jsonl")
    selected = load_selected_candidates(ROOT / "data" / "kcycle_global_breakthrough_results.json")
    ensemble = [ensemble_eval(records, selected, top_k) for top_k in TOP_KS]
    gates = precision_coverage(gating_rows(records))
    payload = {
        "generated_at": utc_now(),
        "selection_rule": "deployable candidates sorted by val_exact desc; top 20 selected before test evaluation",
        "records": len(records),
        "available_deployable_candidates": len(json.loads((ROOT / "data" / "kcycle_global_breakthrough_results.json").read_text(encoding="utf-8")).get("candidates", [])),
        "selected_candidates": [
            {"name": row["name"], "top_k": row["top_k"], "val_exact": row["val_exact"]}
            for row in selected
        ],
        "ensemble": ensemble,
        "gating": gates,
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(payload)
    best = max(ensemble, key=lambda row: row["splits"]["val"]["board_exact_lift_pp"])
    append_progress(
        f"rank-average best top_k={best['top_k']} val_lift={best['splits']['val']['board_exact_lift_pp']:+.3f}pp "
        f"test_lift={best['splits']['test']['board_exact_lift_pp']:+.3f}pp"
    )
    print(json.dumps({"out_json": str(OUT_JSON), "out_md": str(OUT_MD)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
