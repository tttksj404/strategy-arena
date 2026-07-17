#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

import numpy as np
from sklearn.isotonic import IsotonicRegression

ROOT: Final = Path(__file__).resolve().parents[1]
SCRIPTS: Final = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from kcycle_eval_common import assert_purchase_monotonic  # noqa: E402
from search_kcycle_fast_evolution_trifecta import (  # noqa: E402
    FEATURE_NAMES,
    board_metrics,
    build_arrays,
    combo_code,
    combo_parts,
    load_records,
    split_mask,
)

DB_DEFAULT: Final = Path("/Users/tttksj/keirin/data/keirin.db")
BASE_DEFAULT: Final = ROOT / "data" / "kcycle_trifecta_snapshots.jsonl"
EXP_DEFAULT: Final = ROOT / "data" / "kcycle_trifecta_snapshots_expansion.jsonl"
OUT_JSON: Final = ROOT / "data" / "kcycle_round5a_results.json"
OUT_MD: Final = ROOT / "data" / "kcycle_round5a_results.md"
MODEL_OUT: Final = ROOT / "static" / "models" / "kcycle_trifecta_ensemble_v2_candidate.json"
PROGRESS: Final = ROOT / "runs" / "prediction_uplift_progress.md"
MEET_REV: Final = {"광명": "001", "창원": "002", "부산": "003"}
MEET_NAME: Final = {"001": "광명", "002": "창원", "003": "부산"}
TOP_KS: Final = (10, 20, 40)


@dataclass(frozen=True, slots=True)
class RankedScores:
    pred_codes: np.ndarray
    pred_firsts: np.ndarray
    ranked_codes: np.ndarray


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def append_progress(text: str) -> None:
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    with PROGRESS.open("a", encoding="utf-8") as handle:
        handle.write(f"- {utc_now()} Round5A eval: {text}\n")


def race_key(record: dict) -> tuple[str, str, str]:
    date = re.sub(r"\D", "", str(record.get("date") or ""))[:8]
    meet = str(record.get("meet") or "").strip()
    race_no = str(record.get("race_no") or "").strip().lstrip("0") or "0"
    return date, meet, race_no


def kcycle_key(record: dict) -> tuple[str, str, str, str, str] | None:
    kcycle = record.get("kcycle") if isinstance(record.get("kcycle"), dict) else {}
    year = str(record.get("stnd_yr") or kcycle.get("year") or str(record.get("date") or "")[:4])
    meet = str(kcycle.get("meet") or MEET_REV.get(str(record.get("meet") or "").strip(), "")).strip()
    tms = str(kcycle.get("tms") or "").strip()
    day = str(kcycle.get("day") or "").strip()
    rno = str(kcycle.get("rno") or record.get("race_no") or "").strip().zfill(2)
    if not year or not meet or not tms or not day or not rno:
        return None
    return (year, meet, tms, day, rno)


def load_combined_records(paths: list[Path]) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    seen_race: set[tuple[str, str, str]] = set()
    seen_board: set[tuple[str, str, str, str]] = set()
    input_counts: dict[str, int] = {}
    duplicates = 0
    for path in paths:
        if not path.exists():
            input_counts[str(path)] = 0
            continue
        loaded = load_records(path)
        input_counts[str(path)] = len(loaded)
        for row in loaded:
            key = race_key(row)
            board_key = (*key, str(row.get("board_hash") or ""))
            if key in seen_race or board_key in seen_board:
                duplicates += 1
                continue
            seen_race.add(key)
            seen_board.add(board_key)
            rows.append(row)
    return rows, {"input_counts": input_counts, "dedupe_dropped": duplicates}


def load_db_coverage(path: Path) -> Counter[tuple[str, str]]:
    out: Counter[tuple[str, str]] = Counter()
    with sqlite3.connect(path) as con:
        rows = con.execute(
            "SELECT stnd_yr, meet_nm, COUNT(*) FROM race_result "
            "WHERE stnd_yr BETWEEN '2018' AND '2026' GROUP BY stnd_yr, meet_nm"
        ).fetchall()
    for year, meet, count in rows:
        out[(str(year), str(meet).strip())] += int(count)
    return out


def record_coverage(records: list[dict]) -> Counter[tuple[str, str]]:
    out: Counter[tuple[str, str]] = Counter()
    for row in records:
        out[(str(row.get("stnd_yr") or str(row.get("date") or "")[:4]), str(row.get("meet") or "").strip())] += 1
    return out


def coverage_table(db_path: Path, base_records: list[dict], exp_records: list[dict], combined: list[dict]) -> list[dict]:
    db_counts = load_db_coverage(db_path)
    base_counts = record_coverage(base_records)
    exp_counts = record_coverage(exp_records)
    combined_counts = record_coverage(combined)
    keys = sorted(set(db_counts) | set(base_counts) | set(exp_counts) | set(combined_counts))
    return [
        {
            "year": year,
            "meet": meet,
            "db_races": db_counts[(year, meet)],
            "base_snapshots": base_counts[(year, meet)],
            "expansion_snapshots": exp_counts[(year, meet)],
            "combined_snapshots": combined_counts[(year, meet)],
            "remaining_gap": max(db_counts[(year, meet)] - combined_counts[(year, meet)], 0),
        }
        for year, meet in keys
    ]


def candidate_vector(row: dict) -> np.ndarray:
    weights = row.get("weights") if isinstance(row.get("weights"), dict) else {}
    return np.asarray([float(weights.get(name, 0.0)) for name in FEATURE_NAMES], dtype=np.float32)


def load_candidate_pool(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = [row for row in payload.get("candidates", []) if row.get("deployable")]
    return rows


def pick_top_val_candidates(records: list[dict], pool: list[dict], top_k: int) -> tuple[list[dict], np.ndarray, dict]:
    arrays = build_arrays(records, top_k)
    weights = np.vstack([candidate_vector(row) for row in pool]).astype(np.float32)
    scores = np.tensordot(arrays["x"], weights.T, axes=([2], [0]))
    best_idx = scores.argmax(axis=1)
    pred_codes = np.take_along_axis(arrays["codes"][:, :, None], best_idx[:, None, :], axis=1)[:, 0, :]
    mask = split_mask(arrays["years"], "val")
    val_exact = (pred_codes[mask] == arrays["actual_codes"][mask, None]).mean(axis=0)
    order = np.argsort(-val_exact)[:20]
    selected = [{**pool[int(idx)], "round5a_val_exact": float(val_exact[int(idx)]), "source_top_k": top_k} for idx in order]
    return selected, weights[order], arrays


def rank_average_eval(arrays: dict[str, np.ndarray], weights: np.ndarray) -> dict:
    scores = np.tensordot(arrays["x"], weights.T, axes=([2], [0]))
    rank_order = np.argsort(-scores, axis=1)
    ranks = np.empty_like(rank_order, dtype=np.float32)
    race_idx = np.arange(rank_order.shape[0])[:, None, None]
    cand_idx = np.arange(rank_order.shape[2])[None, None, :]
    ranks[race_idx, rank_order, cand_idx] = np.arange(rank_order.shape[1], dtype=np.float32)[None, :, None]
    best_idx = ranks.mean(axis=2).argmin(axis=1)
    pred_codes = arrays["codes"][np.arange(len(best_idx)), best_idx]
    pred_firsts = arrays["firsts"][np.arange(len(best_idx)), best_idx]
    base = board_metrics(arrays)
    splits = {}
    for split in ("train", "val", "test", "all"):
        mask = split_mask(arrays["years"], split)
        n = int(mask.sum())
        exact_hits = int((pred_codes[mask] == arrays["actual_codes"][mask]).sum())
        top1_hits = int((pred_firsts[mask] == arrays["actual_firsts"][mask]).sum())
        exact = exact_hits / n if n else 0.0
        top1 = top1_hits / n if n else 0.0
        splits[split] = {
            "n": n,
            "exact": exact,
            "exact_hits": exact_hits,
            "board_exact": base[split][1],
            "board_exact_lift_pp": (exact - base[split][1]) * 100.0,
            "top1": top1,
            "top1_hits": top1_hits,
            "board_top1": base[split][3],
            "top1_lift_pp": (top1 - base[split][3]) * 100.0,
        }
    return splits


def best_board(record: dict) -> tuple[str, float, float, float] | None:
    board = {str(combo): float(odds) for combo, odds in (record.get("board") or {}).items() if float(odds) > 0}
    if len(board) != 210:
        return None
    ranked = sorted(board.items(), key=lambda item: (item[1], item[0]))
    q_raw = {combo: 1.0 / odds for combo, odds in board.items()}
    q_sum = sum(q_raw.values())
    best_prob = q_raw[ranked[0][0]] / q_sum if q_sum else 0.0
    return ranked[0][0], ranked[0][1], ranked[1][1] / ranked[0][1], best_prob


def signal_strength(best_odds: float, gap12: float) -> float:
    return math.log(max(gap12, 1.000001)) * (3.0 / max(best_odds, 0.01))


def tier_rows(records: list[dict]) -> list[dict]:
    rows = []
    for record in records:
        best = best_board(record)
        actual = str(record.get("actual_order") or "")
        if best is None or not actual:
            continue
        combo, best_odds, gap12, best_prob = best
        actual_first = combo_parts(actual)[0]
        pred_first = combo_parts(combo)[0]
        rows.append({
            "split": split_of(record),
            "meet": str(record.get("meet") or "").strip(),
            "exact_hit": combo == actual,
            "top1_hit": combo_parts(combo)[0] == actual_first,
            "best_odds": best_odds,
            "gap12": gap12,
            "best_prob": best_prob,
            "signal_strength": signal_strength(best_odds, gap12),
        })
    return rows


def split_of(record: dict) -> str:
    year = str(record.get("stnd_yr") or str(record.get("date") or "")[:4])
    if year in {"2018", "2019", "2020", "2021", "2022", "2023"}:
        return "train"
    if year in {"2024", "2025"}:
        return "val"
    if year == "2026":
        return "test"
    return "other"


def summarize_hits(items: list[dict], total: int) -> dict:
    n = len(items)
    return {
        "n": n,
        "coverage": n / total if total else 0.0,
        "exact": sum(bool(row["exact_hit"]) for row in items) / n if n else None,
        "top1": sum(bool(row["top1_hit"]) for row in items) / n if n else None,
    }


def tier_precision(rows: list[dict]) -> dict:
    out: dict[str, dict[str, list[dict]]] = {"overall": {}, "by_meet": {}}
    for meet in ["__all__", "광명", "창원", "부산"]:
        pool = [row for row in rows if meet == "__all__" or row["meet"] == meet]
        val = [row for row in pool if row["split"] == "val"]
        strong_val = [row for row in val if row["best_odds"] <= 3.0 and row["gap12"] >= 1.2]
        thresholds = {
            "T0_base": None,
            "T1_strong": None,
            "T2_top50": float(np.quantile([r["signal_strength"] for r in strong_val], 0.50)) if strong_val else math.inf,
            "T3_top16": float(np.quantile([r["signal_strength"] for r in strong_val], 0.84)) if strong_val else math.inf,
        }
        summaries = []
        for split in ("val", "test", "all"):
            split_rows = [row for row in pool if row["split"] == split or split == "all"]
            total = len(split_rows)
            for tier, threshold in thresholds.items():
                if tier == "T0_base":
                    items = split_rows
                elif tier == "T1_strong":
                    items = [row for row in split_rows if row["best_odds"] <= 3.0 and row["gap12"] >= 1.2]
                else:
                    items = [
                        row for row in split_rows
                        if row["best_odds"] <= 3.0 and row["gap12"] >= 1.2 and row["signal_strength"] >= float(threshold)
                    ]
                summaries.append({"split": split, "tier": tier, "threshold": threshold, **summarize_hits(items, total)})
        if meet == "__all__":
            out["overall"] = {"thresholds_from_val": thresholds, "rows": summaries}
        else:
            out["by_meet"][meet] = {"thresholds_from_val": thresholds, "rows": summaries}
    return out


def q_for_board(record: dict) -> dict[str, float]:
    board = {str(combo): float(odds) for combo, odds in (record.get("board") or {}).items() if float(odds) > 0}
    inv = {combo: 1.0 / odds for combo, odds in board.items()}
    total = sum(inv.values())
    return {combo: value / total for combo, value in inv.items()} if total else {}


def ranked_eval(records: list[dict], ranked: list[list[str]]) -> dict:
    metrics = {}
    for split in ("train", "val", "test", "all"):
        idxs = [idx for idx, row in enumerate(records) if split == "all" or split_of(row) == split]
        n = len(idxs)
        exact_hits = top1_hits = 0
        purchase_hits = {10: 0, 20: 0, 40: 0}
        board_exact = board_top1 = 0
        for idx in idxs:
            actual = str(records[idx].get("actual_order") or "")
            if not actual:
                continue
            pred = ranked[idx][0]
            if pred == actual:
                exact_hits += 1
            if combo_parts(pred)[0] == combo_parts(actual)[0]:
                top1_hits += 1
            board = best_board(records[idx])
            if board and board[0] == actual:
                board_exact += 1
            if board and combo_parts(board[0])[0] == combo_parts(actual)[0]:
                board_top1 += 1
            for k in purchase_hits:
                if actual in ranked[idx][:k]:
                    purchase_hits[k] += 1
        metrics[split] = {
            "n": n,
            "exact": exact_hits / n if n else 0.0,
            "exact_hits": exact_hits,
            "top1": top1_hits / n if n else 0.0,
            "top1_hits": top1_hits,
            "board_exact": board_exact / n if n else 0.0,
            "board_top1": board_top1 / n if n else 0.0,
            "purchase_exact": {str(k): purchase_hits[k] / n if n else 0.0 for k in purchase_hits},
        }
    return metrics


def calibration_a(records: list[dict]) -> dict:
    train_x: list[float] = []
    train_y: list[int] = []
    for row in records:
        if split_of(row) != "train":
            continue
        q = q_for_board(row)
        actual = str(row.get("actual_order") or "")
        for combo, prob in q.items():
            train_x.append(prob)
            train_y.append(1 if combo == actual else 0)
    model = IsotonicRegression(y_min=0.0, out_of_bounds="clip", increasing=True)
    model.fit(np.asarray(train_x), np.asarray(train_y))
    ranked = []
    for row in records:
        q = q_for_board(row)
        probs = model.predict(np.asarray(list(q.values())))
        pairs = sorted(zip(q.keys(), probs, strict=True), key=lambda item: (-float(item[1]), item[0]))
        ranked.append([combo for combo, _prob in pairs])
    return {"method": "board_combo_isotonic", "train_pairs": len(train_x), "metrics": ranked_eval(records, ranked)}


def parse_float(value: object) -> float:
    text = str(value or "").replace(",", "").strip()
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)", text)
    return float(match.group(1)) if match else float("nan")


def load_pool1(path: Path) -> dict[tuple[str, str, str], float]:
    out: dict[tuple[str, str, str], float] = {}
    with sqlite3.connect(path) as con:
        rows = con.execute(
            "SELECT r.stnd_yr, r.race_ymd, r.meet_nm, r.race_no, p.pool1_val "
            "FROM race_result r LEFT JOIN payoff p "
            "ON r.stnd_yr=p.stnd_yr "
            "AND substr('0000' || replace(r.race_ymd, '/', ''), -4) = substr('0000' || replace(p.race_ymd, '/', ''), -4) "
            "AND r.week_tcnt=p.week_tcnt "
            "AND r.day_tcnt=p.day_tcnt AND r.race_no=p.race_no "
            "WHERE r.stnd_yr BETWEEN '2018' AND '2026'"
        ).fetchall()
    for year, ymd, meet, race_no, value in rows:
        date = str(ymd)
        if len(re.sub(r"\D", "", date)) < 8:
            date = f"{year}{re.sub(r'\\D', '', date)[-4:].zfill(4)}"
        odds = parse_float(value)
        if math.isfinite(odds) and odds > 0:
            out[(re.sub(r"\D", "", date)[:8], str(meet).strip(), str(race_no).strip().lstrip("0") or "0")] = 1.0 / odds
    return out


def first_masses(q: dict[str, float]) -> dict[int, float]:
    masses = {idx: 0.0 for idx in range(1, 8)}
    for combo, prob in q.items():
        masses[combo_parts(combo)[0]] += prob
    return masses


def calibration_b(records: list[dict], db_path: Path) -> dict:
    pool1 = load_pool1(db_path)
    train_x: list[float] = []
    train_y: list[float] = []
    for row in records:
        if split_of(row) != "train":
            continue
        q = q_for_board(row)
        actual = str(row.get("actual_order") or "")
        payout_prob = pool1.get(race_key(row))
        if not q or not actual or payout_prob is None:
            continue
        masses = first_masses(q)
        actual_first = combo_parts(actual)[0]
        if actual_first not in masses:
            continue
        train_x.append(masses[actual_first])
        train_y.append(payout_prob)
    model = IsotonicRegression(y_min=0.0, out_of_bounds="clip", increasing=True)
    if not train_x:
        board_ranked = [
            [combo for combo, _odds in sorted((row.get("board") or {}).items(), key=lambda item: (float(item[1]), str(item[0])))]
            for row in records
        ]
        return {
            "method": "pool1_winner_sample_anchor",
            "train_pairs": 0,
            "limitation": "pool1 join produced no train samples; fell back to uncalibrated board ranking.",
            "metrics": ranked_eval(records, board_ranked),
        }
    model.fit(np.asarray(train_x), np.asarray(train_y))
    ranked = []
    for row in records:
        q = q_for_board(row)
        masses = first_masses(q)
        corrected = {first: float(model.predict([mass])[0]) for first, mass in masses.items()}
        scored = []
        for combo, prob in q.items():
            first = combo_parts(combo)[0]
            factor = corrected[first] / max(masses[first], 1e-12)
            scored.append((combo, prob * factor))
        scored.sort(key=lambda item: (-item[1], item[0]))
        ranked.append([combo for combo, _score in scored])
    return {
        "method": "pool1_winner_sample_anchor",
        "train_pairs": len(train_x),
        "limitation": "pool1 is observed only for winning rider payout in this DB join, so the calibration is a winner-sample anchor and is selection-biased.",
        "metrics": ranked_eval(records, ranked),
    }


def metrics_for_monotonic(records: list[dict]) -> dict[str, dict[str, dict[str, dict[str, float | int]]]]:
    arrays = build_arrays(records, 40)
    codes = arrays["codes"]
    actual = arrays["actual_codes"]
    years = arrays["years"]
    result = {"board_rank": {}}
    for k in TOP_KS:
        result["board_rank"][str(k)] = {}
        purchase = (codes[:, :k] == actual[:, None]).any(axis=1)
        for split in ("train", "val", "test", "all"):
            mask = split_mask(years, split)
            n = int(mask.sum())
            hits = int(purchase[mask].sum())
            top_hits = int((codes[mask, 0] == actual[mask]).sum())
            result["board_rank"][str(k)][split] = {
                "n": n,
                "selection_exact": top_hits / n if n else 0.0,
                "selection_hits": top_hits,
                "purchase_exact": hits / n if n else 0.0,
                "purchase_hits": hits,
                "top1": 0.0,
                "top1_hits": 0,
            }
    return result


def write_model_if_improved(payload: dict, best: dict, selected: list[dict], arrays: dict[str, np.ndarray]) -> bool:
    v1_val = max(
        row["splits"]["val"]["exact"]
        for row in payload["v1_comparison"]["v1_test_metrics"].get("ensemble_by_top_k", [])
        if "splits" in row and "val" in row["splits"]
    )
    if best["splits"]["val"]["exact"] <= v1_val:
        return False
    top_k = int(best["top_k"])
    model = {
        "schema": "kcycle_trifecta_ensemble_v2_candidate",
        "generated_at": payload["generated_at"],
        "audit_status": "Fable5 감사 대기",
        "corpus": payload["corpus"],
        "selection": payload["ensemble_v2"]["selection_rule"],
        "feature_names": FEATURE_NAMES,
        "feature_stats_by_top_k": {
            str(top_k): {
                "mu": {name: float(value) for name, value in zip(FEATURE_NAMES, arrays["mu"], strict=True)},
                "sigma": {name: float(value) for name, value in zip(FEATURE_NAMES, arrays["sigma"], strict=True)},
            }
        },
        "formulas": selected,
        "validation_metrics": best,
    }
    MODEL_OUT.write_text(json.dumps(model, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return True


def write_markdown(payload: dict) -> None:
    lines = [
        "# KCYCLE Round 5A",
        "",
        f"generated_at: {payload['generated_at']}",
        f"records: {payload['corpus']['combined_records']} (base {payload['corpus']['base_records']} + expansion {payload['corpus']['expansion_records']})",
        f"network_status: {payload['expansion']['network_status']}",
        "",
        "## Ensemble v2",
        "| top_k | val exact | val lift | test exact | test lift |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in payload["ensemble_v2"]["by_top_k"]:
        val = row["splits"]["val"]
        test = row["splits"]["test"]
        lines.append(f"| {row['top_k']} | {val['exact']:.4f} | {val['board_exact_lift_pp']:+.3f}pp | {test['exact']:.4f} | {test['board_exact_lift_pp']:+.3f}pp |")
    lines.extend(["", "## Calibration", "| method | val exact | val board | test exact | test board |", "|---|---:|---:|---:|---:|"])
    for key in ("calibration_a", "calibration_b"):
        row = payload[key]
        val = row["metrics"]["val"]
        test = row["metrics"]["test"]
        lines.append(f"| {row['method']} | {val['exact']:.4f} | {val['board_exact']:.4f} | {test['exact']:.4f} | {test['board_exact']:.4f} |")
    lines.extend(["", "## Coverage", "| year | meet | db | base | exp | combined | gap |", "|---:|---|---:|---:|---:|---:|---:|"])
    for row in payload["coverage"]:
        lines.append(f"| {row['year']} | {row['meet']} | {row['db_races']} | {row['base_snapshots']} | {row['expansion_snapshots']} | {row['combined_snapshots']} | {row['remaining_gap']} |")
    lines.extend(["", "## Notes", "- 2999 이상 또는 210개 미만 보드는 유효 삼쌍 보드가 아니므로 확장 코퍼스에서 제외한다.", "- pool1 anchor는 승자 표본 기반이라 선택편향이 있다."])
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DB_DEFAULT)
    parser.add_argument("--base", type=Path, default=BASE_DEFAULT)
    parser.add_argument("--expansion", type=Path, default=EXP_DEFAULT)
    parser.add_argument("--candidate-pool", type=Path, default=ROOT / "data" / "kcycle_global_breakthrough_results.json")
    parser.add_argument("--v1-model", type=Path, default=ROOT / "static" / "models" / "kcycle_trifecta_ensemble_v1.json")
    args = parser.parse_args()

    base_records = load_records(args.base)
    exp_records = load_records(args.expansion) if args.expansion.exists() else []
    records, combine_meta = load_combined_records([args.base, args.expansion])
    assert_purchase_monotonic(metrics_for_monotonic(records))
    pool = load_candidate_pool(args.candidate_pool)
    ensemble_rows = []
    selected_by_best: list[dict] = []
    arrays_by_best: dict[str, np.ndarray] | None = None
    for top_k in TOP_KS:
        selected, weights, arrays = pick_top_val_candidates(records, pool, top_k)
        splits = rank_average_eval(arrays, weights)
        ensemble_rows.append({"top_k": top_k, "selected_candidates": selected, "splits": splits})
    best = max(ensemble_rows, key=lambda row: row["splits"]["val"]["exact"])
    selected_by_best, _weights, arrays_by_best = pick_top_val_candidates(records, pool, int(best["top_k"]))

    expansion_status = {
        "network_status": "DNS blocked in current execution surface; collector is resumable when www.kcycle.or.kr resolves.",
        "combine_meta": combine_meta,
    }
    v1_payload = json.loads(args.v1_model.read_text(encoding="utf-8"))
    payload = {
        "generated_at": utc_now(),
        "corpus": {
            "base_records": len(base_records),
            "expansion_records": len(exp_records),
            "combined_records": len(records),
            "train_years": sorted(["2018", "2019", "2020", "2021", "2022", "2023"]),
            "val_years": ["2024", "2025"],
            "test_years": ["2026"],
        },
        "coverage": coverage_table(args.db, base_records, exp_records, records),
        "expansion": expansion_status,
        "ensemble_v2": {
            "selection_rule": "deployable candidate pool re-ranked by expanded val exact; top 20 selected before test reporting",
            "candidate_pool": str(args.candidate_pool),
            "by_top_k": ensemble_rows,
            "best_top_k": best["top_k"],
        },
        "v1_comparison": {"v1_test_metrics": v1_payload.get("test_metrics", {})},
        "tier_precision": tier_precision(tier_rows(records)),
        "calibration_a": calibration_a(records),
        "calibration_b": calibration_b(records, args.db),
        "anomaly_rules": {
            "odds_ge_2999": "exclude from expansion/evaluation because Kcycle archive uses 2999-like sentinel odds for non-real board cells.",
            "board_count_not_210": "exclude; seven-rider trifecta must contain exactly 210 ordered non-repeating combinations.",
            "duplicate_race": "keep first base snapshot; expansion never overwrites base.",
        },
    }
    model_written = write_model_if_improved(payload, best, selected_by_best, arrays_by_best)
    payload["ensemble_v2"]["model_written"] = model_written
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(payload)
    append_progress(f"done records={len(records)} exp={len(exp_records)} best_top_k={best['top_k']} model_written={model_written}")
    print(json.dumps({"out_json": str(OUT_JSON), "out_md": str(OUT_MD), "model_written": model_written}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
