#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPTS))

import engine  # noqa: E402
from search_kcycle_fast_evolution_trifecta import (  # noqa: E402
    FEATURE_NAMES,
    build_arrays,
    combo_code,
    combo_parts,
    load_records,
    split_mask,
)

OUT_JSON = ROOT / "data" / "kcycle_model_market_blend_results.json"
OUT_MD = ROOT / "data" / "kcycle_model_market_blend_results.md"
PROGRESS = ROOT / "runs" / "prediction_uplift_progress.md"
TOP_KS = (10, 20, 40)
BETAS = tuple(round(x, 2) for x in np.arange(0.50, 1.00, 0.05))
NUMERIC_FIELDS = (
    "win_rate",
    "high_rate",
    "high_3_rate",
    "gear_rate",
    "rec_200m_scr",
    "racer_age",
    "tot_tms_avg_scr",
    "area_tms3_avg_scr",
    "win_tot_tcnt",
    "period_no",
    "race_len",
    "mrk_win_cnt",
    "pre_win_cnt",
    "pas_win_cnt",
    "brk_win_cnt",
)


@dataclass(frozen=True, slots=True)
class RaceModel:
    key: tuple[str, str, str]
    year: str
    actual_code: int
    actual_first: int
    strengths: dict[int, float]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def append_progress(text: str) -> None:
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    with PROGRESS.open("a", encoding="utf-8") as handle:
        handle.write(f"- {utc_now()} Phase 2: {text}\n")


def race_key(record: dict) -> tuple[str, str, str]:
    date = "".join(ch for ch in str(record.get("date") or "") if ch.isdigit())[:8]
    meet = str(record.get("meet") or "").strip()
    race_no = str(record.get("race_no") or "").strip().lstrip("0") or "0"
    return date, meet, race_no


def parse_float(value: object) -> float:
    text = str(value or "").replace(",", "").replace('"', ".").strip()
    match = next(iter(__import__("re").finditer(r"-?\d+(?:\.\d+)?", text)), None)
    return float(match.group(0)) if match else float("nan")


def grade_score(value: object) -> float:
    text = str(value or "").strip().upper()
    if text.startswith("SS"):
        return 5.0
    if text.startswith("S") or "특선" in text:
        return 4.0
    if text.startswith("A") or "우수" in text:
        return 3.0
    if text.startswith("B") or "선발" in text:
        return 2.0
    return 0.0


def load_entries(path: Path) -> dict[tuple[str, str, str], list[dict]]:
    out = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (
                str(row.get("date") or ""),
                str(row.get("meet") or "").strip(),
                str(row.get("race_no") or "").strip().lstrip("0") or "0",
            )
            entrants = row.get("entrants")
            if isinstance(entrants, list) and len(entrants) >= 5:
                out[key] = entrants
    return out


def engine_scores(entrants: list[dict]) -> dict[int, float]:
    if not bool(int(__import__("os").environ.get("KCYCLE_BLEND_INCLUDE_JOBLIB", "0"))):
        return {}
    model, error = engine.load_model()
    if error or model is None:
        return {}
    rows, error = engine.score_keirin_with_model(entrants, model)
    if error or rows is None:
        return {}
    return {int(row["bno"]): float(row["pwin"]) for row in rows}


def race_features(entrants: list[dict]) -> tuple[list[int], np.ndarray]:
    bnos = [int(str(row.get("back_no") or "0").strip() or "0") for row in entrants]
    raw_rows = []
    model_score = engine_scores(entrants)
    for row, bno in zip(entrants, bnos):
        values = [parse_float(row.get(field)) for field in NUMERIC_FIELDS]
        values.append(grade_score(row.get("racer_grd_cur_cd") or row.get("racer_grd_cd")))
        values.append(float(model_score.get(bno, float("nan"))))
        raw_rows.append(values)
    raw = np.asarray(raw_rows, dtype=float)
    med = np.nanmedian(raw, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    raw = np.where(np.isfinite(raw), raw, med)
    std = raw.std(axis=0)
    std = np.where(std < 1e-9, 1.0, std)
    z = (raw - raw.mean(axis=0)) / std
    return bnos, z.astype(np.float64)


def fit_strengths(records: list[dict], entries: dict[tuple[str, str, str], list[dict]]) -> dict[tuple[str, str, str], RaceModel]:
    x_rows = []
    y_rows = []
    race_vectors = {}
    for record in records:
        key = race_key(record)
        entrants = entries.get(key)
        if not entrants:
            continue
        bnos, features = race_features(entrants)
        actual_first = combo_parts(str(record["actual_order"]))[0]
        race_vectors[key] = (bnos, features)
        if str(record.get("date") or "")[:4] in {"2018", "2019", "2020", "2021", "2022", "2023"}:
            x_rows.extend(features.tolist())
            y_rows.extend([1 if bno == actual_first else 0 for bno in bnos])
    if not x_rows:
        return {}
    clf = LogisticRegression(max_iter=1000, C=0.5, class_weight="balanced", random_state=20260712)
    clf.fit(np.asarray(x_rows), np.asarray(y_rows))
    out = {}
    for record in records:
        key = race_key(record)
        if key not in race_vectors:
            continue
        bnos, features = race_vectors[key]
        logits = clf.decision_function(features)
        strengths = {bno: float(math.exp(max(min(score, 20.0), -20.0))) for bno, score in zip(bnos, logits)}
        out[key] = RaceModel(
            key=key,
            year=str(record.get("date") or "")[:4],
            actual_code=combo_code(str(record["actual_order"])),
            actual_first=combo_parts(str(record["actual_order"]))[0],
            strengths=strengths,
        )
    return out


def pl_prob(parts: tuple[int, int, int], strengths: dict[int, float]) -> float:
    a, b, c = parts
    total = sum(strengths.values())
    if total <= 0 or a not in strengths or b not in strengths or c not in strengths:
        return 1e-15
    pa = strengths[a] / total
    pb = strengths[b] / max(total - strengths[a], 1e-15)
    pc = strengths[c] / max(total - strengths[a] - strengths[b], 1e-15)
    return max(pa * pb * pc, 1e-15)


def market_probs(board: dict) -> dict[str, float]:
    inv = {str(combo): 1.0 / float(odds) for combo, odds in board.items() if float(odds) > 0}
    total = sum(inv.values())
    return {combo: value / total for combo, value in inv.items()} if total > 0 else {}


def split_name(year: str) -> str:
    if year in {"2018", "2019", "2020", "2021", "2022", "2023"}:
        return "train"
    if year in {"2024", "2025"}:
        return "val"
    if year == "2026":
        return "test"
    return "other"


def blend_pick(record: dict, race_model: RaceModel | None, beta: float, top_k: int) -> tuple[int, int]:
    board = {str(k): float(v) for k, v in (record.get("board") or {}).items() if float(v) > 0}
    ranked = sorted(board.items(), key=lambda item: (item[1], item[0]))[:top_k]
    if race_model is None:
        combo = ranked[0][0]
        return combo_code(combo), combo_parts(combo)[0]
    mkt = market_probs(board)
    best_combo = ranked[0][0]
    best_score = -1e300
    for combo, _odds in ranked:
        parts = combo_parts(combo)
        score = beta * math.log(max(mkt.get(combo, 0.0), 1e-15)) + (1.0 - beta) * math.log(pl_prob(parts, race_model.strengths))
        if score > best_score or (score == best_score and combo < best_combo):
            best_score = score
            best_combo = combo
    return combo_code(best_combo), combo_parts(best_combo)[0]


def evaluate_blend(records: list[dict], models: dict[tuple[str, str, str], RaceModel], beta: float, top_k: int, scope: str, splits: tuple[str, ...]) -> dict[str, dict]:
    split_rows = {split: {"n": 0, "exact_hits": 0, "top1_hits": 0, "board_hits": 0, "board_top1_hits": 0} for split in splits}
    for record in records:
        key = race_key(record)
        race_model = models.get(key)
        if scope == "joined_subset" and race_model is None:
            continue
        split = split_name(str(record.get("date") or "")[:4])
        if split not in split_rows:
            continue
        pred_code, pred_first = blend_pick(record, race_model, beta, top_k)
        actual_code = combo_code(str(record["actual_order"]))
        actual_first = combo_parts(str(record["actual_order"]))[0]
        board_combo = sorted((record.get("board") or {}).items(), key=lambda item: (float(item[1]), item[0]))[0][0]
        row = split_rows[split]
        row["n"] += 1
        row["exact_hits"] += int(pred_code == actual_code)
        row["top1_hits"] += int(pred_first == actual_first)
        row["board_hits"] += int(combo_code(board_combo) == actual_code)
        row["board_top1_hits"] += int(combo_parts(board_combo)[0] == actual_first)
    out = {}
    for split, row in split_rows.items():
        n = row["n"]
        exact = row["exact_hits"] / n if n else 0.0
        top1 = row["top1_hits"] / n if n else 0.0
        board_exact = row["board_hits"] / n if n else 0.0
        board_top1 = row["board_top1_hits"] / n if n else 0.0
        out[split] = {
            "n": n,
            "exact": exact,
            "exact_hits": row["exact_hits"],
            "board_exact": board_exact,
            "board_exact_lift_pp": (exact - board_exact) * 100.0,
            "top1": top1,
            "top1_hits": row["top1_hits"],
            "board_top1": board_top1,
            "top1_lift_pp": (top1 - board_top1) * 100.0,
        }
    return out


def baseline_eval(records: list[dict], top_k: int, weights: np.ndarray | None, scope_keys: set[tuple[str, str, str]] | None) -> dict[str, dict]:
    selected = [record for record in records if scope_keys is None or race_key(record) in scope_keys]
    arrays = build_arrays(selected, top_k)
    if weights is None:
        pred_codes = arrays["codes"][:, 0]
        pred_firsts = arrays["firsts"][:, 0]
    else:
        scores = np.tensordot(arrays["x"], weights, axes=([2], [0]))
        idx = scores.argmax(axis=1)
        pred_codes = arrays["codes"][np.arange(len(idx)), idx]
        pred_firsts = arrays["firsts"][np.arange(len(idx)), idx]
    out = {}
    for split in ("train", "val", "test"):
        mask = split_mask(arrays["years"], split)
        n = int(mask.sum())
        exact_hits = int((pred_codes[mask] == arrays["actual_codes"][mask]).sum())
        top1_hits = int((pred_firsts[mask] == arrays["actual_firsts"][mask]).sum())
        out[split] = {
            "n": n,
            "exact": exact_hits / n if n else 0.0,
            "exact_hits": exact_hits,
            "top1": top1_hits / n if n else 0.0,
            "top1_hits": top1_hits,
        }
    return out


def breakthrough_weights() -> np.ndarray:
    payload = json.loads((ROOT / "data" / "kcycle_global_breakthrough_results.json").read_text(encoding="utf-8"))
    row = next(item for item in payload["candidates"] if item["name"] == "gen2_mut_436")
    weights = row.get("weights") or {}
    return np.asarray([float(weights.get(name, 0.0)) for name in FEATURE_NAMES], dtype=np.float32)


def select_beta(records: list[dict], models: dict[tuple[str, str, str], RaceModel], top_k: int) -> tuple[float, list[dict]]:
    grid = []
    for beta in BETAS:
        metrics = evaluate_blend(records, models, beta, top_k, "joined_subset", ("train", "val"))
        grid.append({"beta": beta, "metrics": metrics})
    grid.sort(key=lambda row: (row["metrics"]["val"]["board_exact_lift_pp"], row["metrics"]["val"]["exact"], -row["beta"]), reverse=True)
    return float(grid[0]["beta"]), grid


def write_markdown(payload: dict) -> None:
    lines = [
        "# KCYCLE model-market blend",
        "",
        f"generated_at: {payload['generated_at']}",
        f"records: {payload['records']}",
        f"joined_records: {payload['joined_records']}",
        f"join_coverage: {payload['join_coverage']:.4f}",
        "selection: beta selected on val only; test reported once for selected beta per top_k",
        "",
        "## Selected beta results",
        "| scope | top_k | beta | val lift | test lift | val exact | test exact |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["blend_results"]:
        val = row["metrics"]["val"]
        test = row["metrics"]["test"]
        lines.append(
            f"| {row['scope']} | {row['top_k']} | {row['beta']:.2f} | {val['board_exact_lift_pp']:+.3f}pp | "
            f"{test['board_exact_lift_pp']:+.3f}pp | {val['exact']:.4f} | {test['exact']:.4f} |"
        )
    lines.extend(["", "## Baselines", "| scope | model | top_k | val exact | test exact |", "|---|---|---:|---:|---:|"])
    for row in payload["baselines"]:
        lines.append(f"| {row['scope']} | {row['model']} | {row['top_k']} | {row['metrics']['val']['exact']:.4f} | {row['metrics']['test']['exact']:.4f} |")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    records = load_records(ROOT / "data" / "kcycle_trifecta_snapshots.jsonl")
    entries = load_entries(ROOT / "data" / "kcycle_entries.jsonl")
    models = fit_strengths(records, entries)
    joined_keys = set(models)
    gen_weights = breakthrough_weights()
    blend_results = []
    beta_grids = {}
    for top_k in TOP_KS:
        beta, grid = select_beta(records, models, top_k)
        beta_grids[str(top_k)] = grid
        for scope in ("joined_subset", "all_with_market_fallback"):
            blend_results.append(
                {
                    "scope": scope,
                    "top_k": top_k,
                    "beta": beta,
                    "metrics": evaluate_blend(records, models, beta, top_k, scope, ("train", "val", "test")),
                }
            )
    baselines = []
    for scope, keys in (("joined_subset", joined_keys), ("all_records", None)):
        for top_k in TOP_KS:
            baselines.append({"scope": scope, "model": "current_axis", "top_k": top_k, "metrics": baseline_eval(records, top_k, None, keys)})
            baselines.append({"scope": scope, "model": "gen2_mut_436", "top_k": top_k, "metrics": baseline_eval(records, top_k, gen_weights, keys)})
    include_joblib = bool(int(os.environ.get("KCYCLE_BLEND_INCLUDE_JOBLIB", "0")))
    payload = {
        "generated_at": utc_now(),
        "records": len(records),
        "joined_records": len(joined_keys),
        "join_coverage": len(joined_keys) / len(records) if records else 0.0,
        "feature_policy": (
            "runner features are within-race z-scores; static/models/keirin_model_final.joblib pwin included"
            if include_joblib
            else "runner features are within-race z-scores; static/models/keirin_model_final.joblib pwin disabled for bounded runtime"
        ),
        "beta_grid_train_val_only": beta_grids,
        "blend_results": blend_results,
        "baselines": baselines,
        "stretch_conditional_logit": {"status": "not_run", "reason": "bounded campaign run used first-pass logistic runner strength only"},
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(payload)
    best = max(
        (row for row in blend_results if row["scope"] == "joined_subset"),
        key=lambda row: row["metrics"]["val"]["board_exact_lift_pp"],
    )
    append_progress(
        f"joined top_k={best['top_k']} beta={best['beta']:.2f} "
        f"val_lift={best['metrics']['val']['board_exact_lift_pp']:+.3f}pp "
        f"test_lift={best['metrics']['test']['board_exact_lift_pp']:+.3f}pp"
    )
    print(json.dumps({"out_json": str(OUT_JSON), "out_md": str(OUT_MD)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
