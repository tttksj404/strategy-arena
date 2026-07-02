#!/usr/bin/env python3
import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


TRAIN_YEARS = {"2018", "2019", "2020", "2021", "2022", "2023"}
HOLDOUT_YEARS = ["2024", "2025", "2026"]
FEATURES_HIGH = [
    "gap12",
    "gap15",
    "gap110",
    "top_prob",
    "top3_mass",
    "top5_mass",
    "top10_mass",
    "entropy_inv",
    "first_mass_best",
    "second_mass_best",
    "third_mass_best",
    "pair12_mass_best",
    "first_gap",
    "pair_gap",
    "top3_same_first",
    "top5_same_first",
    "top3_same_pair",
]
FEATURES_LOW = ["best_odds"]
METHODS = [
    "board_min",
    "pair_mass",
    "first_pair_chain",
    "energy_pair",
    "energy_first_pair",
    "energy_position",
]


@dataclass(frozen=True, slots=True)
class Candidate:
    status: str
    method: str
    rule: str
    train_n: int
    train_hit: float
    n_2024: int
    hit_2024: float
    n_2025: int
    hit_2025: float
    n_2026: int
    hit_2026: float
    holdout_n: int
    holdout_hit: float
    holdout_worst: float


def safe_log(value):
    return math.log(max(float(value), 1e-12))


def argmax_combo(board, scores):
    return min(scores, key=lambda combo: (-scores[combo], board[combo], combo))


def load_records(path):
    records = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("actual_order") and int(row.get("board_count") or 0) == 210:
                records.append(row)
    return records


def frame_from_records(records):
    rows = []
    for record in records:
        board = {str(k): float(v) for k, v in (record.get("board") or {}).items() if float(v) > 0}
        if len(board) != 210:
            continue
        ranked = sorted(board.items(), key=lambda kv: (kv[1], kv[0]))
        q_raw = {combo: 1.0 / odds for combo, odds in board.items()}
        q_sum = sum(q_raw.values())
        q = {combo: value / q_sum for combo, value in q_raw.items()}
        first_mass = {i: 0.0 for i in range(1, 8)}
        second_mass = {i: 0.0 for i in range(1, 8)}
        third_mass = {i: 0.0 for i in range(1, 8)}
        pair_mass = {}
        for combo, prob in q.items():
            a, b, c = map(int, combo.split("-"))
            first_mass[a] += prob
            second_mass[b] += prob
            third_mass[c] += prob
            pair_mass[(a, b)] = pair_mass.get((a, b), 0.0) + prob
        probs = np.array(list(q.values()), dtype=float)
        entropy = float(-(probs * np.log(np.maximum(probs, 1e-12))).sum() / math.log(len(probs)))
        best, best_odds = ranked[0]
        a, b, c = map(int, best.split("-"))
        top3 = [combo for combo, _ in ranked[:3]]
        top5 = [combo for combo, _ in ranked[:5]]
        top10 = [combo for combo, _ in ranked[:10]]
        first_vals = sorted(first_mass.values(), reverse=True)
        pair_vals = sorted(pair_mass.values(), reverse=True)
        base_scores = {combo: safe_log(q[combo]) for combo in board}
        best_pair = max(pair_mass, key=lambda pair: (pair_mass[pair], -pair[0], -pair[1]))
        pair_candidates = [combo for combo in board if combo.startswith(f"{best_pair[0]}-{best_pair[1]}-")]
        chain_first = max(first_mass, key=lambda x: (first_mass[x], -x))
        chain_pairs = {pair: mass for pair, mass in pair_mass.items() if pair[0] == chain_first}
        chain_pair = max(chain_pairs, key=lambda pair: (chain_pairs[pair], -pair[1]))
        chain_candidates = [combo for combo in board if combo.startswith(f"{chain_pair[0]}-{chain_pair[1]}-")]
        pred = {
            "board_min": best,
            "pair_mass": min(pair_candidates, key=lambda combo: (board[combo], combo)),
            "first_pair_chain": min(chain_candidates, key=lambda combo: (board[combo], combo)),
            "energy_pair": argmax_combo(board, {
                combo: base_scores[combo] + 0.8 * safe_log(pair_mass[tuple(map(int, combo.split("-")[:2]))])
                for combo in board
            }),
            "energy_first_pair": argmax_combo(board, {
                combo: base_scores[combo]
                + 0.5 * safe_log(first_mass[int(combo[0])])
                + 0.8 * safe_log(pair_mass[tuple(map(int, combo.split("-")[:2]))])
                for combo in board
            }),
            "energy_position": argmax_combo(board, {
                combo: base_scores[combo]
                + 0.35 * safe_log(first_mass[int(combo[0])])
                + 0.25 * safe_log(second_mass[int(combo[2])])
                + 0.25 * safe_log(third_mass[int(combo[4])])
                for combo in board
            }),
        }
        row = {
            "year": str(record.get("stnd_yr") or str(record.get("date"))[:4]),
            "actual_order": record["actual_order"],
            "best_odds": best_odds,
            "gap12": ranked[1][1] / best_odds,
            "gap15": ranked[4][1] / best_odds,
            "gap110": ranked[9][1] / best_odds,
            "top_prob": q[best],
            "top3_mass": sum(q[x] for x in top3),
            "top5_mass": sum(q[x] for x in top5),
            "top10_mass": sum(q[x] for x in top10),
            "entropy_inv": 1.0 - entropy,
            "first_mass_best": first_mass[a],
            "second_mass_best": second_mass[b],
            "third_mass_best": third_mass[c],
            "pair12_mass_best": pair_mass[(a, b)],
            "first_gap": first_vals[0] / max(first_vals[1], 1e-12),
            "pair_gap": pair_vals[0] / max(pair_vals[1], 1e-12),
            "top3_same_first": float(len({x.split("-")[0] for x in top3}) == 1),
            "top5_same_first": float(len({x.split("-")[0] for x in top5}) == 1),
            "top3_same_pair": float(len({"-".join(x.split("-")[:2]) for x in top3}) == 1),
        }
        for method, combo in pred.items():
            row[f"hit_{method}"] = combo == record["actual_order"]
        rows.append(row)
    return pd.DataFrame(rows)


def predicates(df, train_mask):
    out = [("all", np.ones(len(df), dtype=bool))]
    seen = {"all"}

    def add(name, mask):
        if name not in seen:
            seen.add(name)
            out.append((name, mask))

    for col in FEATURES_HIGH:
        values = df.loc[train_mask, col].dropna().to_numpy(float)
        arr = df[col].to_numpy(float)
        for qv in [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.975, 0.99]:
            cut = float(np.quantile(values, qv))
            add(f"{col}>={cut:.6g}", arr >= cut)
    for col in FEATURES_LOW:
        values = df.loc[train_mask, col].dropna().to_numpy(float)
        arr = df[col].to_numpy(float)
        for qv in [0.01, 0.025, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]:
            cut = float(np.quantile(values, qv))
            add(f"{col}<={cut:.6g}", arr <= cut)
    singles = out[1:]
    for i, (left_name, left_mask) in enumerate(singles):
        for right_name, right_mask in singles[i + 1:]:
            add(f"{left_name}&{right_name}", left_mask & right_mask)
    return out


def eval_candidate(hits, train_base, holdout_base, holdout_year_masks, method, rule, mask):
    train_mask = train_base & mask
    train_n = int(train_mask.sum())
    if train_n < 50:
        return None
    by_year = {}
    for year in HOLDOUT_YEARS:
        year_mask = holdout_year_masks[year] & mask
        n = int(year_mask.sum())
        if n < 5:
            return None
        by_year[year] = (n, float(hits[year_mask].mean()))
    holdout_mask = holdout_base & mask
    holdout_n = int(holdout_mask.sum())
    holdout_hit = float(hits[holdout_mask].mean()) if holdout_n else 0.0
    worst = min(hit for _, hit in by_year.values())
    status = "PASS_HOLDOUT_50" if worst >= 0.5 else "FAIL"
    return Candidate(
        status,
        method,
        rule,
        train_n,
        float(hits[train_mask].mean()),
        by_year["2024"][0],
        by_year["2024"][1],
        by_year["2025"][0],
        by_year["2025"][1],
        by_year["2026"][0],
        by_year["2026"][1],
        holdout_n,
        holdout_hit,
        worst,
    )


def run(args):
    df = frame_from_records(load_records(args.snapshots))
    train_mask = df["year"].isin(TRAIN_YEARS)
    preds = predicates(df, train_mask)
    years = df["year"].to_numpy(str)
    train_base = np.isin(years, list(TRAIN_YEARS))
    holdout_base = np.isin(years, HOLDOUT_YEARS)
    holdout_year_masks = {year: years == year for year in HOLDOUT_YEARS}
    method_hits = {method: df[f"hit_{method}"].to_numpy(bool) for method in METHODS}
    rows = []
    for method in METHODS:
        hits = method_hits[method]
        for rule, mask in preds:
            row = eval_candidate(hits, train_base, holdout_base, holdout_year_masks, method, rule, mask)
            if row is not None:
                rows.append(row)
    rows.sort(
        key=lambda r: (
            r.status == "PASS_HOLDOUT_50",
            min(r.n_2024, r.n_2025, r.n_2026),
            r.holdout_worst,
            r.holdout_hit,
        ),
        reverse=True,
    )
    baseline = {}
    for method in METHODS:
        hits = method_hits[method]
        baseline[method] = {
            "all": float(hits.mean()),
            "holdout": float(hits[df["year"].isin(HOLDOUT_YEARS).to_numpy()].mean()),
        }
    payload = {
        "records": int(len(df)),
        "train_years": sorted(TRAIN_YEARS),
        "holdout_years": HOLDOUT_YEARS,
        "predicate_count": len(preds),
        "evaluated_candidates": len(rows),
        "pass_count": sum(1 for row in rows if row.status == "PASS_HOLDOUT_50"),
        "pass_count_by_min_year_n": {
            str(n): sum(
                1
                for row in rows
                if row.status == "PASS_HOLDOUT_50" and min(row.n_2024, row.n_2025, row.n_2026) >= n
            )
            for n in [5, 10, 20, 30, 50, 100]
        },
        "baseline": baseline,
        "rows": [asdict(row) for row in rows[: args.limit]],
    }
    Path(args.out_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# KCYCLE trifecta rule search",
        "",
        f"records: {payload['records']} train: {payload['train_years']} holdout: {payload['holdout_years']}",
        f"predicate_count: {payload['predicate_count']} evaluated: {payload['evaluated_candidates']} pass_count: {payload['pass_count']}",
        f"pass_count_by_min_year_n: {payload['pass_count_by_min_year_n']}",
        "",
        "| status | method | rule | train_n | train_hit | n24 | hit24 | n25 | hit25 | n26 | hit26 | holdout_n | holdout_hit | worst |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows[:80]:
        lines.append(
            f"| {row.status} | {row.method} | {row.rule} | {row.train_n} | {row.train_hit:.4f} | "
            f"{row.n_2024} | {row.hit_2024:.4f} | {row.n_2025} | {row.hit_2025:.4f} | "
            f"{row.n_2026} | {row.hit_2026:.4f} | {row.holdout_n} | {row.holdout_hit:.4f} | {row.holdout_worst:.4f} |"
        )
    Path(args.out_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ["records", "predicate_count", "evaluated_candidates", "pass_count"]}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshots", default=str(ROOT / "data" / "kcycle_trifecta_snapshots.jsonl"))
    parser.add_argument("--out-json", default=str(ROOT / "data" / "kcycle_trifecta_rule_search_results.json"))
    parser.add_argument("--out-md", default=str(ROOT / "docs" / "kcycle_trifecta_rule_search_results.md"))
    parser.add_argument("--limit", type=int, default=300)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
