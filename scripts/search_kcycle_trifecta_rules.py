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
    "xdom_annealing_escape",
    "xdom_drug_scaffold_hop",
    "xdom_drug_multi_objective_funnel",
    "xdom_clonal_selection_amplify",
    "xdom_ecology_predator_prey",
    "xdom_bandit_explore_exploit",
    "xdom_bayesian_surrogate_focus",
    "xdom_bradley_terry_position",
    "xdom_harville_order_flow",
    "xdom_information_bottleneck",
    "xdom_particle_filter_resample",
]
XDOM_METHODS = tuple(method for method in METHODS if method.startswith("xdom_"))
MIN_TRAIN_N = 50
MIN_HOLDOUT_YEAR_N = 5
PROMOTION_MIN_HOLDOUT_YEAR_N = 10
PROMOTION_MIN_WORST_HIT = 0.5


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
    min_holdout_year_n: int


def safe_log(value):
    return math.log(max(float(value), 1e-12))


def argmax_combo(board, scores):
    return min(scores, key=lambda combo: (-scores[combo], board[combo], combo))


def combo_parts(combo):
    return tuple(map(int, combo.split("-")))


def combo_pair(combo):
    a, b, _ = combo_parts(combo)
    return a, b


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
        rank_score = {combo: -math.log(index + 1.0) for index, (combo, _) in enumerate(ranked)}
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
                combo: base_scores[combo] + 0.8 * safe_log(pair_mass[combo_pair(combo)])
                for combo in board
            }),
            "energy_first_pair": argmax_combo(board, {
                combo: base_scores[combo]
                + 0.5 * safe_log(first_mass[int(combo[0])])
                + 0.8 * safe_log(pair_mass[combo_pair(combo)])
                for combo in board
            }),
            "energy_position": argmax_combo(board, {
                combo: base_scores[combo]
                + 0.35 * safe_log(first_mass[int(combo[0])])
                + 0.25 * safe_log(second_mass[int(combo[2])])
                + 0.25 * safe_log(third_mass[int(combo[4])])
                for combo in board
            }),
            "xdom_annealing_escape": argmax_combo(board, {
                combo: (
                    0.72 * base_scores[combo]
                    + 0.62 * safe_log(pair_mass[combo_pair(combo)])
                    + 0.28 * safe_log(first_mass[int(combo[0])])
                    + 0.08 * rank_score[combo]
                )
                for combo in board
            }),
            "xdom_drug_scaffold_hop": argmax_combo(board, {
                combo: (
                    0.62 * base_scores[combo]
                    + 0.92 * safe_log(pair_mass[combo_pair(combo)])
                    + 0.42 * safe_log(third_mass[int(combo[4])])
                )
                for combo in board
                if combo.startswith(f"{best_pair[0]}-{best_pair[1]}-")
            }),
            "xdom_drug_multi_objective_funnel": argmax_combo(board, {
                combo: (
                    0.58 * base_scores[combo]
                    + 0.38 * safe_log(first_mass[int(combo[0])])
                    + 0.58 * safe_log(pair_mass[combo_pair(combo)])
                    + 0.24 * safe_log(second_mass[int(combo[2])])
                    + 0.24 * safe_log(third_mass[int(combo[4])])
                    + 0.10 * rank_score[combo]
                )
                for combo in board
            }),
            "xdom_clonal_selection_amplify": argmax_combo(board, {
                combo: (
                    0.48 * base_scores[combo]
                    + 0.88 * safe_log(first_mass[int(combo[0])])
                    + 1.08 * safe_log(pair_mass[combo_pair(combo)])
                    + 0.30 * safe_log(second_mass[int(combo[2])])
                    + 0.30 * safe_log(third_mass[int(combo[4])])
                )
                for combo in board
            }),
            "xdom_ecology_predator_prey": argmax_combo(board, {
                combo: (
                    0.68 * base_scores[combo]
                    + 0.74 * safe_log(first_mass[int(combo[0])])
                    + 0.50 * safe_log(pair_mass[combo_pair(combo)])
                    + 0.22 * safe_log(third_mass[int(combo[4])])
                )
                for combo in top10
            }),
            "xdom_bandit_explore_exploit": argmax_combo(board, {
                combo: (
                    0.70 * base_scores[combo]
                    + 0.78 * safe_log(pair_mass[combo_pair(combo)])
                    + 0.34 * safe_log(first_mass[int(combo[0])])
                    - 0.05 * safe_log(q[combo])
                    + 0.06 * rank_score[combo]
                )
                for combo in board
            }),
            "xdom_bayesian_surrogate_focus": argmax_combo(board, {
                combo: (
                    0.52 * base_scores[combo]
                    + 0.95 * safe_log(pair_mass[combo_pair(combo)])
                    + 0.42 * safe_log(first_mass[int(combo[0])])
                    + 0.34 * safe_log(third_mass[int(combo[4])])
                )
                for combo in board
            }),
            "xdom_bradley_terry_position": argmax_combo(board, {
                combo: (
                    0.44 * base_scores[combo]
                    + 0.92 * safe_log(first_mass[int(combo[0])])
                    + 0.66 * safe_log(second_mass[int(combo[2])])
                    + 0.48 * safe_log(third_mass[int(combo[4])])
                )
                for combo in board
            }),
            "xdom_harville_order_flow": argmax_combo(board, {
                combo: (
                    0.50 * base_scores[combo]
                    + 1.10 * safe_log(pair_mass[combo_pair(combo)])
                    + 0.60 * safe_log(first_mass[int(combo[0])])
                    + 0.38 * safe_log(third_mass[int(combo[4])])
                    + 0.05 * rank_score[combo]
                )
                for combo in board
            }),
            "xdom_information_bottleneck": argmax_combo(board, {
                combo: (
                    0.46 * base_scores[combo]
                    + 1.05 * safe_log(pair_mass[combo_pair(combo)])
                    + 0.85 * safe_log(first_mass[int(combo[0])])
                    + 0.15 * rank_score[combo]
                )
                for combo in top10
            }),
            "xdom_particle_filter_resample": argmax_combo(board, {
                combo: (
                    0.36 * base_scores[combo]
                    + 0.80 * safe_log(pair_mass[combo_pair(combo)])
                    + 0.45 * safe_log(second_mass[int(combo[2])])
                    + 0.45 * safe_log(third_mass[int(combo[4])])
                    + 0.20 * rank_score[combo]
                )
                for combo in top5
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
            row[f"pred_{method}"] = combo
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
    for name, mask in xdom_predicates(df, train_mask):
        add(name, mask)
    return out


def train_quantile(df, train_mask, col, qv):
    values = df.loc[train_mask, col].dropna().to_numpy(float)
    return float(np.quantile(values, qv))


def xdom_predicates(df, train_mask):
    cuts = {
        "gap12_80": train_quantile(df, train_mask, "gap12", 0.80),
        "gap12_90": train_quantile(df, train_mask, "gap12", 0.90),
        "gap15_80": train_quantile(df, train_mask, "gap15", 0.80),
        "gap110_90": train_quantile(df, train_mask, "gap110", 0.90),
        "top5_80": train_quantile(df, train_mask, "top5_mass", 0.80),
        "entropy_80": train_quantile(df, train_mask, "entropy_inv", 0.80),
        "first_80": train_quantile(df, train_mask, "first_mass_best", 0.80),
        "pair_80": train_quantile(df, train_mask, "pair12_mass_best", 0.80),
        "first_gap_80": train_quantile(df, train_mask, "first_gap", 0.80),
        "pair_gap_80": train_quantile(df, train_mask, "pair_gap", 0.80),
        "best_odds_20": train_quantile(df, train_mask, "best_odds", 0.20),
        "second_80": train_quantile(df, train_mask, "second_mass_best", 0.80),
        "third_80": train_quantile(df, train_mask, "third_mass_best", 0.80),
        "top10_80": train_quantile(df, train_mask, "top10_mass", 0.80),
    }
    arr = {col: df[col].to_numpy(float) for col in set(FEATURES_HIGH + FEATURES_LOW)}
    return [
        (
            "xdom_annealing_escape:gap12_high&entropy_high",
            (arr["gap12"] >= cuts["gap12_80"]) & (arr["entropy_inv"] >= cuts["entropy_80"]),
        ),
        (
            "xdom_drug_scaffold_hop:pair_mass&pair_gap",
            (arr["pair12_mass_best"] >= cuts["pair_80"]) & (arr["pair_gap"] >= cuts["pair_gap_80"]),
        ),
        (
            "xdom_drug_multi_objective_funnel:top5&low_odds",
            (arr["top5_mass"] >= cuts["top5_80"]) & (arr["best_odds"] <= cuts["best_odds_20"]),
        ),
        (
            "xdom_clonal_selection_amplify:first_mass&same_first",
            (arr["first_mass_best"] >= cuts["first_80"]) & (arr["top5_same_first"] >= 1.0),
        ),
        (
            "xdom_ecology_predator_prey:same_first&gap15",
            (arr["top3_same_first"] >= 1.0) & (arr["gap15"] >= cuts["gap15_80"]),
        ),
        (
            "xdom_bandit_explore_exploit:pair_gap&not_extreme_gap12",
            (arr["pair_gap"] >= cuts["pair_gap_80"]) & (arr["gap12"] < cuts["gap12_90"]),
        ),
        (
            "xdom_bayesian_surrogate_focus:first_gap&gap110",
            (arr["first_gap"] >= cuts["first_gap_80"]) & (arr["gap110"] >= cuts["gap110_90"]),
        ),
        (
            "xdom_bradley_terry_position:position_mass",
            (arr["first_mass_best"] >= cuts["first_80"])
            & (arr["second_mass_best"] >= cuts["second_80"])
            & (arr["third_mass_best"] >= cuts["third_80"]),
        ),
        (
            "xdom_harville_order_flow:pair_position",
            (arr["pair12_mass_best"] >= cuts["pair_80"]) & (arr["third_mass_best"] >= cuts["third_80"]),
        ),
        (
            "xdom_information_bottleneck:compressed_top10",
            (arr["top10_mass"] >= cuts["top10_80"]) & (arr["entropy_inv"] >= cuts["entropy_80"]),
        ),
        (
            "xdom_particle_filter_resample:top5_pair_gap",
            (arr["top5_mass"] >= cuts["top5_80"]) & (arr["pair_gap"] >= cuts["pair_gap_80"]),
        ),
    ]


def eval_candidate(hits, train_base, holdout_base, holdout_year_masks, method, rule, mask):
    train_mask = train_base & mask
    train_n = int(train_mask.sum())
    if train_n < MIN_TRAIN_N:
        return None
    by_year = {}
    for year in HOLDOUT_YEARS:
        year_mask = holdout_year_masks[year] & mask
        n = int(year_mask.sum())
        if n < MIN_HOLDOUT_YEAR_N:
            return None
        by_year[year] = (n, float(hits[year_mask].mean()))
    holdout_mask = holdout_base & mask
    holdout_n = int(holdout_mask.sum())
    holdout_hit = float(hits[holdout_mask].mean()) if holdout_n else 0.0
    worst = min(hit for _, hit in by_year.values())
    min_year_n = min(n for n, _ in by_year.values())
    if worst >= PROMOTION_MIN_WORST_HIT and min_year_n >= PROMOTION_MIN_HOLDOUT_YEAR_N:
        status = "PROMOTE_ROBUST_50"
    elif worst >= PROMOTION_MIN_WORST_HIT:
        status = "WATCH_LOW_SAMPLE_50"
    else:
        status = "FAIL"
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
        min_year_n,
    )


def candidate_metric_signature(row):
    return (
        row.rule,
        row.train_n,
        round(row.train_hit, 12),
        row.n_2024,
        round(row.hit_2024, 12),
        row.n_2025,
        round(row.hit_2025, 12),
        row.n_2026,
        round(row.hit_2026, 12),
        row.holdout_n,
        round(row.holdout_hit, 12),
        round(row.holdout_worst, 12),
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
            r.status == "PROMOTE_ROBUST_50",
            r.status == "WATCH_LOW_SAMPLE_50",
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
    board_min_pred = df["pred_board_min"].to_numpy(str)
    holdout_selector = df["year"].isin(HOLDOUT_YEARS).to_numpy()
    xdom_diversity = {}
    for method in XDOM_METHODS:
        method_pred = df[f"pred_{method}"].to_numpy(str)
        differs = method_pred != board_min_pred
        holdout_differs = holdout_selector & differs
        hits = method_hits[method]
        xdom_diversity[method] = {
            "holdout_diff_rate": float(differs[holdout_selector].mean()),
            "holdout_diff_n": int(holdout_differs.sum()),
            "holdout_hit_when_diff": (
                float(hits[holdout_differs].mean()) if int(holdout_differs.sum()) else 0.0
            ),
        }
    fifty_rows = [row for row in rows if row.status in {"PROMOTE_ROBUST_50", "WATCH_LOW_SAMPLE_50"}]
    promote_rows = [row for row in rows if row.status == "PROMOTE_ROBUST_50"]
    xdom_fifty_rows = [
        row
        for row in fifty_rows
        if row.method.startswith("xdom_") or row.rule.startswith("xdom_")
    ]
    xdom_promote_rows = [
        row
        for row in promote_rows
        if row.method.startswith("xdom_") or row.rule.startswith("xdom_")
    ]
    deduped_fifty_count = len({candidate_metric_signature(row) for row in fifty_rows})
    deduped_xdom_fifty_count = len({candidate_metric_signature(row) for row in xdom_fifty_rows})
    risk_flags = {
        "no_robust_promotion": len(promote_rows) == 0,
        "low_sample_watch_only": len(fifty_rows) > 0 and len(promote_rows) == 0,
        "xdom_duplicate_inflation": len(xdom_fifty_rows) > deduped_xdom_fifty_count,
        "requires_more_outcome_linked_snapshots": len(promote_rows) == 0,
    }
    payload = {
        "records": int(len(df)),
        "train_years": sorted(TRAIN_YEARS),
        "holdout_years": HOLDOUT_YEARS,
        "promotion_rules": {
            "min_train_n": MIN_TRAIN_N,
            "min_eval_holdout_year_n": MIN_HOLDOUT_YEAR_N,
            "min_promote_holdout_year_n": PROMOTION_MIN_HOLDOUT_YEAR_N,
            "min_worst_year_hit": PROMOTION_MIN_WORST_HIT,
        },
        "xdom_methods": list(XDOM_METHODS),
        "predicate_count": len(preds),
        "xdom_predicate_count": sum(1 for name, _ in preds if name.startswith("xdom_")),
        "evaluated_candidates": len(rows),
        "fifty_watch_or_promote_count": len(fifty_rows),
        "promotion_count": len(promote_rows),
        "xdom_fifty_watch_or_promote_count": len(xdom_fifty_rows),
        "xdom_promotion_count": len(xdom_promote_rows),
        "deduped_fifty_watch_or_promote_count": deduped_fifty_count,
        "deduped_xdom_fifty_watch_or_promote_count": deduped_xdom_fifty_count,
        "pass_count_by_min_year_n": {
            str(n): sum(
                1
                for row in rows
                if row.status in {"PROMOTE_ROBUST_50", "WATCH_LOW_SAMPLE_50"}
                and row.min_holdout_year_n >= n
            )
            for n in [5, 10, 20, 30, 50, 100]
        },
        "baseline": baseline,
        "xdom_diversity": xdom_diversity,
        "risk_flags": risk_flags,
        "rows": [asdict(row) for row in rows[: args.limit]],
    }
    Path(args.out_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# KCYCLE trifecta rule search",
        "",
        f"records: {payload['records']} train: {payload['train_years']} holdout: {payload['holdout_years']}",
        f"promotion_rules: {payload['promotion_rules']}",
        f"predicate_count: {payload['predicate_count']} evaluated: {payload['evaluated_candidates']}",
        f"xdom_methods: {payload['xdom_methods']}",
        (
            "fifty_watch_or_promote_count: "
            f"{payload['fifty_watch_or_promote_count']} "
            f"promotion_count: {payload['promotion_count']}"
        ),
        (
            "xdom_fifty_watch_or_promote_count: "
            f"{payload['xdom_fifty_watch_or_promote_count']} "
            f"xdom_promotion_count: {payload['xdom_promotion_count']}"
        ),
        (
            "deduped_fifty_watch_or_promote_count: "
            f"{payload['deduped_fifty_watch_or_promote_count']} "
            "deduped_xdom_fifty_watch_or_promote_count: "
            f"{payload['deduped_xdom_fifty_watch_or_promote_count']}"
        ),
        f"pass_count_by_min_year_n: {payload['pass_count_by_min_year_n']}",
        f"risk_flags: {payload['risk_flags']}",
        "",
        "Interpretation: xdom recipes are evaluated in the KCYCLE trifecta harness, "
        "but current 50%+ holdout passes remain low-sample strong-favorite slices. "
        "No candidate is promoted unless every holdout year has n >= 10.",
        "",
        "## XDOM diversity versus board_min",
        "",
        "| method | holdout_diff_rate | holdout_diff_n | holdout_hit_when_diff |",
        "|---|---:|---:|---:|",
    ]
    for method, diversity in payload["xdom_diversity"].items():
        lines.append(
            f"| {method} | {diversity['holdout_diff_rate']:.4f} | "
            f"{diversity['holdout_diff_n']} | {diversity['holdout_hit_when_diff']:.4f} |"
        )
    lines.extend([
        "",
        "| status | method | rule | train_n | train_hit | n24 | hit24 | n25 | hit25 | n26 | hit26 | holdout_n | holdout_hit | worst | min_year_n |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in rows[:80]:
        lines.append(
            f"| {row.status} | {row.method} | {row.rule} | {row.train_n} | {row.train_hit:.4f} | "
            f"{row.n_2024} | {row.hit_2024:.4f} | {row.n_2025} | {row.hit_2025:.4f} | "
            f"{row.n_2026} | {row.hit_2026:.4f} | {row.holdout_n} | {row.holdout_hit:.4f} | "
            f"{row.holdout_worst:.4f} | {row.min_holdout_year_n} |"
        )
    Path(args.out_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(
        {
            k: payload[k]
            for k in [
                "records",
                "predicate_count",
                "xdom_predicate_count",
                "evaluated_candidates",
                "fifty_watch_or_promote_count",
                "promotion_count",
                "xdom_fifty_watch_or_promote_count",
                "xdom_promotion_count",
                "risk_flags",
            ]
        },
        ensure_ascii=False,
    ))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshots", default=str(ROOT / "data" / "kcycle_trifecta_snapshots.jsonl"))
    parser.add_argument("--out-json", default=str(ROOT / "data" / "kcycle_trifecta_rule_search_results.json"))
    parser.add_argument("--out-md", default=str(ROOT / "docs" / "kcycle_trifecta_rule_search_results.md"))
    parser.add_argument("--limit", type=int, default=300)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
