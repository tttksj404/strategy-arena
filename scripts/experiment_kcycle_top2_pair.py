#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SNAPSHOTS = ROOT / "data" / "kcycle_trifecta_snapshots.jsonl"
DEFAULT_JSON = ROOT / "data" / "kcycle_top2_pair_results.json"
DEFAULT_MD = ROOT / "docs" / "kcycle_top2_pair_results.md"
TRAIN_YEARS = {"2018", "2019", "2020", "2021", "2022", "2023"}
HOLDOUT_YEARS = ["2024", "2025", "2026"]
HYBRID_ALT_METHODS = [
    "slot_mass_top2",
    "first_second_chain",
    "ordered_pair_mass",
    "unordered_pair_mass",
]


def json_dumps(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _combo_parts(combo):
    return [int(part) for part in str(combo).split("-")]


def _actual_top3(record):
    raw = record.get("actual_order")
    if isinstance(raw, str):
        return [int(part) for part in raw.replace(" ", "").split("-") if part]
    return [int(part) for part in (raw or [])]


def _norm_board(board):
    return {
        str(combo): float(odds)
        for combo, odds in (board or {}).items()
        if float(odds) > 0 and len(set(_combo_parts(combo))) == 3
    }


def load_records(path):
    records = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("actual_order") and int(row.get("board_count") or 0) == 210:
                board = _norm_board(row.get("board"))
                if len(board) == 210:
                    row["board"] = board
                    records.append(row)
    return records


def _rank_two(scores):
    return [
        runner
        for runner, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:2]
    ]


def _features(board):
    board = _norm_board(board)
    q_raw = {combo: 1.0 / odds for combo, odds in board.items()}
    q_sum = sum(q_raw.values()) or 1.0
    q = {combo: value / q_sum for combo, value in q_raw.items()}

    first_mass = {runner: 0.0 for runner in range(1, 8)}
    second_mass = {runner: 0.0 for runner in range(1, 8)}
    top2_mass = {runner: 0.0 for runner in range(1, 8)}
    ordered_pair = defaultdict(float)
    unordered_pair = defaultdict(float)
    for combo, prob in q.items():
        a, b, _ = _combo_parts(combo)
        first_mass[a] += prob
        second_mass[b] += prob
        top2_mass[a] += prob
        top2_mass[b] += prob
        ordered_pair[(a, b)] += prob
        unordered_pair[tuple(sorted((a, b)))] += prob

    probs = list(q.values())
    entropy = -sum(p * math.log(max(p, 1e-12)) for p in probs) / math.log(len(probs))
    slot_rank = sorted(top2_mass.items(), key=lambda kv: (-kv[1], kv[0]))
    first_rank = sorted(first_mass.items(), key=lambda kv: (-kv[1], kv[0]))
    ordered_vals = sorted(ordered_pair.values(), reverse=True)
    unordered_vals = sorted(unordered_pair.values(), reverse=True)
    best_combo = min(board, key=lambda combo: (board[combo], combo))
    best_ordered = max(ordered_pair, key=lambda pair: (ordered_pair[pair], -pair[0], -pair[1]))
    best_unordered = max(
        unordered_pair,
        key=lambda pair: (unordered_pair[pair], -(first_mass[pair[0]] + first_mass[pair[1]]), -pair[0], -pair[1]),
    )
    unordered_ordered = sorted(
        best_unordered,
        key=lambda runner: (-(first_mass[runner] + 0.15 * second_mass[runner]), runner),
    )
    chain_first = first_rank[0][0]
    chain_second_scores = {
        b: mass for (a, b), mass in ordered_pair.items() if a == chain_first
    }
    if chain_second_scores:
        chain_second = max(chain_second_scores, key=lambda b: (chain_second_scores[b], -b))
    else:
        chain_second = next(runner for runner, _ in slot_rank if runner != chain_first)
    slot_gap23 = slot_rank[1][1] - slot_rank[2][1]
    return {
        "best_combo": best_combo,
        "first_mass": first_mass,
        "second_mass": second_mass,
        "top2_mass": top2_mass,
        "ordered_pair": ordered_pair,
        "unordered_pair": unordered_pair,
        "slot_rank": slot_rank,
        "entropy": entropy,
        "slot_gap23": slot_gap23,
        "slot_gap12": slot_rank[0][1] - slot_rank[1][1],
        "slot_top1": slot_rank[0][1],
        "slot_top2": slot_rank[1][1],
        "slot_top3": slot_rank[2][1],
        "best_odds": board[best_combo],
        "ordered_pair_gap12": ordered_vals[0] - ordered_vals[1],
        "unordered_pair_gap12": unordered_vals[0] - unordered_vals[1],
        "slot_mass_top2": [runner for runner, _ in slot_rank[:2]],
        "first_second_chain": [chain_first, chain_second],
        "ordered_pair_mass": list(best_ordered),
        "unordered_pair_mass": unordered_ordered,
    }


def method_predictions(board):
    feats = _features(board)
    best_a, best_b, _ = _combo_parts(feats["best_combo"])
    predictions = {
        "best_combo_top2": [best_a, best_b],
        "slot_mass_top2": feats["slot_mass_top2"],
        "first_second_chain": feats["first_second_chain"],
        "ordered_pair_mass": feats["ordered_pair_mass"],
        "unordered_pair_mass": feats["unordered_pair_mass"],
    }
    for threshold in (0.0025, 0.005, 0.01, 0.02, 0.04, 0.08):
        key = f"router_pair_if_gap23_le_{threshold:g}"
        predictions[key] = (
            feats["unordered_pair_mass"]
            if feats["slot_gap23"] <= threshold
            else feats["slot_mass_top2"]
        )
    for threshold in (0.82, 0.86, 0.90, 0.94):
        key = f"router_pair_if_entropy_ge_{threshold:g}"
        predictions[key] = (
            feats["unordered_pair_mass"]
            if feats["entropy"] >= threshold
            else feats["slot_mass_top2"]
        )
    return predictions


def _empty_stats():
    return {
        "n": 0,
        "top1_hits": 0,
        "top2_slot_hits_sum": 0.0,
        "unordered_pair_hits": 0,
        "ordered_pair_hits": 0,
    }


def _add_hit(stats, pred, actual):
    top2_hit, unordered_hit, ordered_hit, top1_hit = _hit_tuple(pred, actual)
    stats["n"] += 1
    stats["top1_hits"] += top1_hit
    stats["top2_slot_hits_sum"] += top2_hit
    stats["unordered_pair_hits"] += unordered_hit
    stats["ordered_pair_hits"] += ordered_hit


def _hit_tuple(pred, actual):
    actual_top2 = actual[:2]
    return (
        len(set(pred[:2]) & set(actual_top2)) / 2.0,
        int(set(pred[:2]) == set(actual_top2)),
        int(pred[:2] == actual_top2),
        int(pred[0] == actual_top2[0]),
    )


def _rates(stats):
    n = stats["n"] or 1
    return {
        "n": stats["n"],
        "top1_hit_rate": stats["top1_hits"] / n,
        "top2_slot_hits": stats["top2_slot_hits_sum"] / n,
        "unordered_pair_hits": stats["unordered_pair_hits"] / n,
        "ordered_pair_hits": stats["ordered_pair_hits"] / n,
    }


def analyze_records(records):
    by_method = defaultdict(_empty_stats)
    by_method_year = defaultdict(lambda: defaultdict(_empty_stats))
    kept = 0
    for record in records:
        actual = _actual_top3(record)
        if len(actual) < 2:
            continue
        year = str(record.get("stnd_yr") or str(record.get("date", ""))[:4])
        predictions = method_predictions(record.get("board"))
        kept += 1
        for method, pred in predictions.items():
            _add_hit(by_method[method], pred, actual)
            _add_hit(by_method_year[method][year], pred, actual)

    baseline = _rates(by_method["best_combo_top2"])
    methods = []
    for method, stats in sorted(by_method.items()):
        rates = _rates(stats)
        yearly = {year: _rates(year_stats) for year, year_stats in sorted(by_method_year[method].items())}
        holdout_stats = _empty_stats()
        train_stats = _empty_stats()
        for year, year_stats in by_method_year[method].items():
            target = train_stats if year in TRAIN_YEARS else holdout_stats
            for key, value in year_stats.items():
                target[key] += value
        train = _rates(train_stats)
        holdout = _rates(holdout_stats)
        worst_holdout_lift = None
        for year in HOLDOUT_YEARS:
            if yearly.get(year, {}).get("n", 0) == 0:
                continue
            base_year = _rates(by_method_year["best_combo_top2"][year])
            lift = yearly[year]["top2_slot_hits"] - base_year["top2_slot_hits"]
            worst_holdout_lift = lift if worst_holdout_lift is None else min(worst_holdout_lift, lift)
        rates.update({
            "method": method,
            "train": train,
            "holdout": holdout,
            "years": yearly,
            "overall_top2_lift_vs_best_combo": rates["top2_slot_hits"] - baseline["top2_slot_hits"],
            "holdout_top2_lift_vs_best_combo": holdout["top2_slot_hits"] - _rates(by_method_year["best_combo_top2"]["2024"])["top2_slot_hits"]
            if False else None,
            "worst_holdout_year_top2_lift": worst_holdout_lift,
        })
        base_holdout = _merge_year_stats(by_method_year["best_combo_top2"], HOLDOUT_YEARS)
        rates["holdout_top2_lift_vs_best_combo"] = (
            holdout["top2_slot_hits"] - _rates(base_holdout)["top2_slot_hits"]
        )
        methods.append(rates)

    methods.sort(
        key=lambda row: (
            row["holdout_top2_lift_vs_best_combo"],
            row["holdout"]["unordered_pair_hits"],
            row["holdout"]["ordered_pair_hits"],
        ),
        reverse=True,
    )
    best = methods[0] if methods else None
    deployable = [
        row for row in methods
        if row["holdout"]["n"] >= 100
        and row["holdout_top2_lift_vs_best_combo"] > 0
        and (row["worst_holdout_year_top2_lift"] is None or row["worst_holdout_year_top2_lift"] >= 0)
    ]
    hybrid_candidates = search_hybrid_candidates(records)
    return {
        "summary": {
            "race_count": kept,
            "train_years": sorted(TRAIN_YEARS),
            "holdout_years": HOLDOUT_YEARS,
            "baseline_method": "best_combo_top2",
            "deployable_candidate_count": len(deployable),
            "hybrid_candidate_count": len(hybrid_candidates),
        },
        "best_candidate": deployable[0] if deployable else None,
        "best_hybrid_candidate": hybrid_candidates[0] if hybrid_candidates else None,
        "top_observed_candidate": best,
        "methods": methods,
        "hybrid_candidates": hybrid_candidates[:50],
    }


def _merge_year_stats(year_stats_map, years):
    merged = _empty_stats()
    for year in years:
        stats = year_stats_map.get(year)
        if not stats:
            continue
        for key, value in stats.items():
            merged[key] += value
    return merged


def _record_rows(records):
    rows = []
    for record in records:
        actual = _actual_top3(record)
        if len(actual) < 2:
            continue
        board = _norm_board(record.get("board"))
        if len(board) != 210:
            continue
        feats = _features(board)
        predictions = method_predictions(board)
        rows.append({
            "year": str(record.get("stnd_yr") or str(record.get("date", ""))[:4]),
            "race_no": int(record.get("race_no") or 0),
            "actual": actual,
            "predictions": predictions,
            "features": {
                "entropy": feats["entropy"],
                "slot_gap12": feats["slot_gap12"],
                "slot_gap23": feats["slot_gap23"],
                "slot_top1": feats["slot_top1"],
                "slot_top2": feats["slot_top2"],
                "slot_top3": feats["slot_top3"],
                "best_odds": feats["best_odds"],
                "ordered_pair_gap12": feats["ordered_pair_gap12"],
                "unordered_pair_gap12": feats["unordered_pair_gap12"],
                "race_no": float(record.get("race_no") or 0),
            },
        })
    return rows


def _candidate_predicates(rows):
    train_rows = [row for row in rows if row["year"] in TRAIN_YEARS]
    if not train_rows:
        train_rows = rows
    predicates = [("all", lambda row: True)]
    feature_names = sorted(train_rows[0]["features"]) if train_rows else []
    for feature in feature_names:
        values = sorted(row["features"][feature] for row in train_rows)
        if not values:
            continue
        quantiles = sorted({
            values[int((len(values) - 1) * q)]
            for q in (0.1, 0.2, 0.33, 0.5, 0.67, 0.8, 0.9)
        })
        for threshold in quantiles:
            predicates.append((
                f"{feature}_le_{threshold:.6g}",
                lambda row, feature=feature, threshold=threshold: row["features"][feature] <= threshold,
            ))
            predicates.append((
                f"{feature}_ge_{threshold:.6g}",
                lambda row, feature=feature, threshold=threshold: row["features"][feature] >= threshold,
            ))
    return predicates


def _hybrid_stats(rows, alt_method, predicate):
    all_stats = _empty_stats()
    train_stats = _empty_stats()
    holdout_stats = _empty_stats()
    year_stats = defaultdict(_empty_stats)
    base_holdout = _empty_stats()
    base_train = _empty_stats()
    base_year_stats = defaultdict(_empty_stats)
    switched = 0
    for row in rows:
        use_alt = predicate(row)
        pred = row["predictions"][alt_method] if use_alt else row["predictions"]["best_combo_top2"]
        base = row["predictions"]["best_combo_top2"]
        if use_alt:
            switched += 1
        _add_hit(all_stats, pred, row["actual"])
        _add_hit(year_stats[row["year"]], pred, row["actual"])
        _add_hit(base_year_stats[row["year"]], base, row["actual"])
        if row["year"] in TRAIN_YEARS:
            _add_hit(train_stats, pred, row["actual"])
            _add_hit(base_train, base, row["actual"])
        else:
            _add_hit(holdout_stats, pred, row["actual"])
            _add_hit(base_holdout, base, row["actual"])
    return all_stats, train_stats, holdout_stats, year_stats, base_train, base_holdout, base_year_stats, switched


def search_hybrid_candidates(records, min_train_lift=0.001):
    rows = _record_rows(records)
    if not rows:
        return []
    candidates = []
    for alt_method in HYBRID_ALT_METHODS:
        for predicate_name, predicate in _candidate_predicates(rows):
            all_stats, train_stats, holdout_stats, year_stats, base_train, base_holdout, base_year_stats, switched = _hybrid_stats(
                rows, alt_method, predicate
            )
            train = _rates(train_stats)
            holdout = _rates(holdout_stats)
            base_train_rates = _rates(base_train)
            base_holdout_rates = _rates(base_holdout)
            train_lift = train["top2_slot_hits"] - base_train_rates["top2_slot_hits"]
            holdout_lift = holdout["top2_slot_hits"] - base_holdout_rates["top2_slot_hits"]
            worst = None
            yearly = {}
            for year in HOLDOUT_YEARS:
                if year_stats.get(year, {}).get("n", 0) == 0:
                    continue
                year_rates = _rates(year_stats[year])
                base_year_rates = _rates(base_year_stats[year])
                yearly[year] = year_rates
                lift = year_rates["top2_slot_hits"] - base_year_rates["top2_slot_hits"]
                worst = lift if worst is None else min(worst, lift)
            if train_lift < min_train_lift or holdout_lift <= 0:
                continue
            if worst is not None and worst < 0:
                continue
            candidates.append({
                "method": f"hybrid_{alt_method}__{predicate_name}",
                "alt_method": alt_method,
                "predicate": predicate_name,
                "n": all_stats["n"],
                "switched_n": switched,
                "train": train,
                "holdout": holdout,
                "years": yearly,
                "train_top2_lift_vs_best_combo": train_lift,
                "holdout_top2_lift_vs_best_combo": holdout_lift,
                "worst_holdout_year_top2_lift": worst,
                "holdout_unordered_lift_vs_best_combo": holdout["unordered_pair_hits"] - base_holdout_rates["unordered_pair_hits"],
                "holdout_ordered_lift_vs_best_combo": holdout["ordered_pair_hits"] - base_holdout_rates["ordered_pair_hits"],
            })
    candidates.sort(
        key=lambda row: (
            row["holdout_top2_lift_vs_best_combo"],
            row["holdout_unordered_lift_vs_best_combo"],
            row["holdout_ordered_lift_vs_best_combo"],
            row["train_top2_lift_vs_best_combo"],
        ),
        reverse=True,
    )
    return candidates


def _write_markdown(result, path):
    summary = result["summary"]
    lines = [
        "# KCYCLE Top2 Pair Experiment",
        "",
        f"- races: {summary['race_count']}",
        f"- baseline: {summary['baseline_method']}",
        f"- deployable candidates: {summary['deployable_candidate_count']}",
        "",
        "## Top Methods",
        "",
        "| method | holdout n | top2 slot | unordered pair | ordered pair | holdout lift pp | worst year lift pp |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["methods"][:12]:
        worst = row["worst_holdout_year_top2_lift"]
        lines.append(
            "| {method} | {n} | {slot:.4f} | {unordered:.4f} | {ordered:.4f} | {lift:.3f} | {worst} |".format(
                method=row["method"],
                n=row["holdout"]["n"],
                slot=row["holdout"]["top2_slot_hits"],
                unordered=row["holdout"]["unordered_pair_hits"],
                ordered=row["holdout"]["ordered_pair_hits"],
                lift=row["holdout_top2_lift_vs_best_combo"] * 100,
                worst="" if worst is None else f"{worst * 100:.3f}",
            )
        )
    lines.append("")
    if result["best_candidate"]:
        lines.append("## Decision")
        lines.append("")
        lines.append(f"Deployable candidate: `{result['best_candidate']['method']}`")
    else:
        lines.append("## Decision")
        lines.append("")
        lines.append("No deployable candidate passed the non-negative holdout-year gate.")
    lines.extend([
        "",
        "## Hybrid Candidates",
        "",
        "| method | switched | holdout top2 | unordered lift pp | ordered lift pp | top2 lift pp | worst year lift pp |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in result.get("hybrid_candidates", [])[:12]:
        worst = row["worst_holdout_year_top2_lift"]
        lines.append(
            "| {method} | {switched} | {slot:.4f} | {unordered:.3f} | {ordered:.3f} | {lift:.3f} | {worst} |".format(
                method=row["method"],
                switched=row["switched_n"],
                slot=row["holdout"]["top2_slot_hits"],
                unordered=row["holdout_unordered_lift_vs_best_combo"] * 100,
                ordered=row["holdout_ordered_lift_vs_best_combo"] * 100,
                lift=row["holdout_top2_lift_vs_best_combo"] * 100,
                worst="" if worst is None else f"{worst * 100:.3f}",
            )
        )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(snapshot_path=DEFAULT_SNAPSHOTS, out_json=DEFAULT_JSON, out_md=DEFAULT_MD):
    records = load_records(snapshot_path)
    result = analyze_records(records)
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json_dumps(result) + "\n", encoding="utf-8")
    _write_markdown(result, out_md)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshots", default=str(DEFAULT_SNAPSHOTS))
    parser.add_argument("--out-json", default=str(DEFAULT_JSON))
    parser.add_argument("--out-md", default=str(DEFAULT_MD))
    args = parser.parse_args()
    result = run(args.snapshots, args.out_json, args.out_md)
    print(json_dumps({
        "summary": result["summary"],
        "best_candidate": result["best_candidate"],
        "top_observed_candidate": result["top_observed_candidate"],
    }))


if __name__ == "__main__":
    main()
