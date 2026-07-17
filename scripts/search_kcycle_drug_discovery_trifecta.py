#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


TRAIN_YEARS = {"2018", "2019", "2020", "2021", "2022", "2023"}
VAL_YEARS = {"2024", "2025"}
TEST_YEARS = {"2026"}


FEATURE_NAMES = [
    "rank_norm",
    "log_q",
    "log_odds",
    "odds_ratio_best",
    "first_mass",
    "second_mass",
    "third_mass",
    "pair_mass",
    "unordered_trio_mass",
    "pair_share",
    "third_share",
    "first_gap",
    "pair_gap",
    "gap12",
    "gap15",
    "gap110",
    "entropy_inv",
    "top3_same_first",
    "top5_same_first",
    "top3_same_pair",
]


@dataclass(frozen=True, slots=True)
class Metric:
    name: str
    split: str
    races: int
    exact: float
    exact_hits: int
    top1: float
    top1_hits: int
    board_exact: float
    board_top1: float
    exact_lift_pp: float
    top1_lift_pp: float
    rule: str


def safe_log(value: float) -> float:
    return math.log(max(float(value), 1e-12))


def year_of(record: dict) -> str:
    year = str(record.get("stnd_yr") or "")
    return year or str(record.get("date") or "")[:4]


def combo_parts(combo: str) -> tuple[int, int, int]:
    return tuple(int(x) for x in str(combo).split("-"))  # type: ignore[return-value]


def load_records(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            board = row.get("board") or {}
            if row.get("actual_order") and int(row.get("board_count") or 0) == 210 and len(board) == 210:
                records.append(row)
    return records


def race_features(record: dict, top_k: int) -> tuple[list[dict], dict]:
    board = {str(k): float(v) for k, v in (record.get("board") or {}).items() if float(v) > 0}
    ranked = sorted(board.items(), key=lambda kv: (kv[1], kv[0]))
    q_raw = {combo: 1.0 / odds for combo, odds in board.items()}
    q_sum = sum(q_raw.values())
    q = {combo: value / q_sum for combo, value in q_raw.items()}
    probs = np.array(list(q.values()), dtype=float)
    entropy_inv = 1.0 - float(-(probs * np.log(np.maximum(probs, 1e-12))).sum() / math.log(len(probs)))
    first_mass = {i: 0.0 for i in range(1, 8)}
    second_mass = {i: 0.0 for i in range(1, 8)}
    third_mass = {i: 0.0 for i in range(1, 8)}
    pair_mass: dict[tuple[int, int], float] = {}
    trio_mass: dict[tuple[int, int, int], float] = {}
    for combo, prob in q.items():
        a, b, c = combo_parts(combo)
        first_mass[a] += prob
        second_mass[b] += prob
        third_mass[c] += prob
        pair_mass[(a, b)] = pair_mass.get((a, b), 0.0) + prob
        trio_key = tuple(sorted((a, b, c)))
        trio_mass[trio_key] = trio_mass.get(trio_key, 0.0) + prob
    first_vals = sorted(first_mass.values(), reverse=True)
    pair_vals = sorted(pair_mass.values(), reverse=True)
    best_odds = ranked[0][1]
    top3 = [combo for combo, _ in ranked[:3]]
    top5 = [combo for combo, _ in ranked[:5]]
    race = {
        "key": f"{record.get('date')}|{record.get('meet')}|{str(record.get('race_no')).zfill(2)}",
        "year": year_of(record),
        "actual": str(record.get("actual_order")),
        "board_pick": ranked[0][0],
        "gap12": ranked[1][1] / best_odds,
        "gap15": ranked[4][1] / best_odds,
        "gap110": ranked[9][1] / best_odds,
        "entropy_inv": entropy_inv,
        "first_gap": first_vals[0] / max(first_vals[1], 1e-12),
        "pair_gap": pair_vals[0] / max(pair_vals[1], 1e-12),
        "top3_same_first": float(len({x.split("-")[0] for x in top3}) == 1),
        "top5_same_first": float(len({x.split("-")[0] for x in top5}) == 1),
        "top3_same_pair": float(len({"-".join(x.split("-")[:2]) for x in top3}) == 1),
    }
    rows = []
    for rank, (combo, odds) in enumerate(ranked[:top_k], start=1):
        a, b, c = combo_parts(combo)
        pair = pair_mass[(a, b)]
        third = third_mass[c]
        rows.append({
            "race_key": race["key"],
            "year": race["year"],
            "combo": combo,
            "rank": rank,
            "actual": race["actual"],
            "board_pick": race["board_pick"],
            "features": [
                rank / 210.0,
                safe_log(q[combo]),
                safe_log(odds),
                odds / best_odds,
                first_mass[a],
                second_mass[b],
                third,
                pair,
                trio_mass[tuple(sorted((a, b, c)))],
                q[combo] / max(pair, 1e-12),
                q[combo] / max(third, 1e-12),
                race["first_gap"],
                race["pair_gap"],
                race["gap12"],
                race["gap15"],
                race["gap110"],
                race["entropy_inv"],
                race["top3_same_first"],
                race["top5_same_first"],
                race["top3_same_pair"],
            ],
            "hit": float(combo == race["actual"]),
        })
    return rows, race


def build_dataset(records: list[dict], top_k: int) -> tuple[list[dict], dict[str, dict]]:
    rows: list[dict] = []
    races: dict[str, dict] = {}
    for record in records:
        race_rows, race = race_features(record, top_k)
        rows.extend(race_rows)
        races[race["key"]] = race
    return rows, races


def split_mask_year(year: str, split: str) -> bool:
    if split == "train":
        return year in TRAIN_YEARS
    if split == "val":
        return year in VAL_YEARS
    if split == "test":
        return year in TEST_YEARS
    return year in TRAIN_YEARS | VAL_YEARS | TEST_YEARS


def metric_for_picks(name: str, split: str, picks: dict[str, str], races: dict[str, dict], rule: str) -> Metric:
    keys = [key for key, race in races.items() if split_mask_year(race["year"], split)]
    exact_hits = 0
    top1_hits = 0
    board_exact_hits = 0
    board_top1_hits = 0
    for key in keys:
        race = races[key]
        actual = str(race["actual"])
        pick = picks.get(key, race["board_pick"])
        exact_hits += int(pick == actual)
        top1_hits += int(pick.split("-")[0] == actual.split("-")[0])
        board_exact_hits += int(race["board_pick"] == actual)
        board_top1_hits += int(race["board_pick"].split("-")[0] == actual.split("-")[0])
    n = len(keys)
    exact = exact_hits / n if n else 0.0
    top1 = top1_hits / n if n else 0.0
    board_exact = board_exact_hits / n if n else 0.0
    board_top1 = board_top1_hits / n if n else 0.0
    return Metric(
        name=name,
        split=split,
        races=n,
        exact=exact,
        exact_hits=exact_hits,
        top1=top1,
        top1_hits=top1_hits,
        board_exact=board_exact,
        board_top1=board_top1,
        exact_lift_pp=(exact - board_exact) * 100.0,
        top1_lift_pp=(top1 - board_top1) * 100.0,
        rule=rule,
    )


def choose_by_scores(rows: list[dict], scores: np.ndarray) -> dict[str, str]:
    best: dict[str, tuple[float, str]] = {}
    for row, score in zip(rows, scores):
        key = row["race_key"]
        combo = row["combo"]
        prior = best.get(key)
        if prior is None or (float(score), combo) > prior:
            best[key] = (float(score), combo)
    return {key: combo for key, (_, combo) in best.items()}


def choose_guarded_by_scores(rows: list[dict], scores: np.ndarray, races: dict[str, dict], min_margin: float, max_rank: int) -> dict[str, str]:
    by_race: dict[str, list[tuple[dict, float]]] = {}
    for row, score in zip(rows, scores):
        by_race.setdefault(row["race_key"], []).append((row, float(score)))
    picks = {}
    for key, race in races.items():
        candidates = by_race.get(key) or []
        if not candidates:
            picks[key] = race["board_pick"]
            continue
        best_row, best_score = max(candidates, key=lambda item: (item[1], -int(item[0]["rank"]), item[0]["combo"]))
        board_score = next((score for row, score in candidates if row["combo"] == race["board_pick"]), None)
        if board_score is None:
            board_score = max(score for row, score in candidates if int(row["rank"]) == 1)
        if (
            best_row["combo"] != race["board_pick"]
            and int(best_row["rank"]) <= max_rank
            and best_score - float(board_score) >= min_margin
        ):
            picks[key] = best_row["combo"]
        else:
            picks[key] = race["board_pick"]
    return picks


def best_guarded_metrics(name: str, rows: list[dict], scores: np.ndarray, races: dict[str, dict], rule: str) -> list[Metric]:
    candidates = []
    for max_rank in [2, 3, 5, 10, 20, 40, 80]:
        margins = []
        by_race: dict[str, list[tuple[dict, float]]] = {}
        for row, score in zip(rows, scores):
            if not split_mask_year(row["year"], "val"):
                continue
            by_race.setdefault(row["race_key"], []).append((row, float(score)))
        for key, race_rows in by_race.items():
            race = races[key]
            best_row, best_score = max(race_rows, key=lambda item: (item[1], -int(item[0]["rank"]), item[0]["combo"]))
            board_score = next((score for row, score in race_rows if row["combo"] == race["board_pick"]), None)
            if board_score is not None and best_row["combo"] != race["board_pick"] and int(best_row["rank"]) <= max_rank:
                margins.append(best_score - float(board_score))
        thresholds = [0.0]
        if margins:
            thresholds.extend(float(x) for x in np.quantile(np.array(margins), [0.25, 0.5, 0.7, 0.8, 0.9, 0.95]))
        for threshold in sorted(set(round(x, 12) for x in thresholds)):
            picks = choose_guarded_by_scores(rows, scores, races, threshold, max_rank)
            val = metric_for_picks(
                f"{name}_guard_r{max_rank}_m{threshold:.6g}",
                "val",
                picks,
                races,
                f"{rule}; switch only if score_margin>={threshold:.6g} and rank<={max_rank}",
            )
            candidates.append((val, picks))
    if not candidates:
        return []
    val, picks = max(candidates, key=lambda item: (item[0].exact_lift_pp, item[0].top1_lift_pp, item[0].exact))
    return [
        metric_for_picks(val.name, split, picks, races, val.rule)
        for split in ["val", "test", "all"]
    ]


def run_surrogate(rows: list[dict], races: dict[str, dict], top_k: int) -> list[Metric]:
    x = np.array([row["features"] for row in rows], dtype=float)
    y = np.array([row["hit"] for row in rows], dtype=int)
    years = np.array([row["year"] for row in rows], dtype=str)
    train = np.isin(years, sorted(TRAIN_YEARS))
    if int(y[train].sum()) < 20:
        return []
    weights = np.where(y[train] == 1, 45.0, 1.0)
    model = HistGradientBoostingClassifier(
        max_depth=3,
        learning_rate=0.05,
        max_iter=220,
        l2_regularization=2.0,
        min_samples_leaf=80,
        random_state=20260703 + top_k,
    )
    model.fit(x[train], y[train], sample_weight=weights)
    scores = model.predict_proba(x)[:, 1]
    picks = choose_by_scores(rows, scores)
    metrics = [
        metric_for_picks(f"drug_qsar_hgb_top{top_k}", split, picks, races, "HGB surrogate on top-K trifecta candidates")
        for split in ["val", "test", "all"]
    ]
    metrics.extend(best_guarded_metrics(f"drug_qsar_hgb_top{top_k}", rows, scores, races, "HGB applicability-domain guard"))
    return metrics


def run_weight_grid(rows: list[dict], races: dict[str, dict], top_k: int) -> list[Metric]:
    x = np.array([row["features"] for row in rows], dtype=float)
    years = np.array([row["year"] for row in rows], dtype=str)
    train = np.isin(years, sorted(TRAIN_YEARS))
    mean = x[train].mean(axis=0)
    std = np.maximum(x[train].std(axis=0), 1e-9)
    z = (x - mean) / std
    feature_index = {name: i for i, name in enumerate(FEATURE_NAMES)}
    specs = []
    for w_pair in [0.0, 0.35, 0.7, 1.05]:
        for w_pos in [0.0, 0.25, 0.5, 0.75]:
            for w_gap in [0.0, 0.2, 0.4]:
                specs.append({
                    "name": f"drug_multiobjective_top{top_k}_p{w_pair}_s{w_pos}_g{w_gap}",
                    "weights": {
                        "log_q": 1.0,
                        "pair_mass": w_pair,
                        "first_mass": w_pos,
                        "second_mass": w_pos * 0.45,
                        "third_mass": w_pos * 0.35,
                        "gap12": w_gap,
                        "entropy_inv": w_gap * 0.5,
                    },
                })
    best_spec = None
    best_val = None
    best_picks = None
    for spec in specs:
        score = np.zeros(len(rows), dtype=float)
        for name, weight in spec["weights"].items():
            score += float(weight) * z[:, feature_index[name]]
        picks = choose_by_scores(rows, score)
        val = metric_for_picks(spec["name"], "val", picks, races, "standardized multi-objective weighted score")
        if best_val is None or (val.exact, val.top1, val.exact_lift_pp) > (best_val.exact, best_val.top1, best_val.exact_lift_pp):
            best_val = val
            best_spec = spec
            best_picks = picks
    if not best_spec or not best_picks:
        return []
    metrics = [
        metric_for_picks(best_spec["name"], split, best_picks, races, json.dumps(best_spec["weights"], sort_keys=True))
        for split in ["val", "test", "all"]
    ]
    score = np.zeros(len(rows), dtype=float)
    for name, weight in best_spec["weights"].items():
        score += float(weight) * z[:, feature_index[name]]
    metrics.extend(best_guarded_metrics(best_spec["name"], rows, score, races, "multi-objective applicability-domain guard"))
    return metrics


def run(args: argparse.Namespace) -> dict:
    started = time.time()
    records = load_records(Path(args.snapshots))
    all_metrics: list[Metric] = []
    for top_k in args.top_k:
        rows, races = build_dataset(records, top_k)
        board_picks = {key: race["board_pick"] for key, race in races.items()}
        all_metrics.extend(
            metric_for_picks(f"board_min_top{top_k}", split, board_picks, races, "lowest trifecta odds")
            for split in ["val", "test", "all"]
        )
        all_metrics.extend(run_surrogate(rows, races, top_k))
        all_metrics.extend(run_weight_grid(rows, races, top_k))
    by_name: dict[str, dict[str, Metric]] = {}
    for metric in all_metrics:
        by_name.setdefault(metric.name, {})[metric.split] = metric
    candidates = []
    for name, splits in by_name.items():
        if name.startswith("board_min"):
            continue
        val = splits.get("val")
        test = splits.get("test")
        if not val or not test:
            continue
        candidates.append({
            "name": name,
            "val": asdict(val),
            "test": asdict(test),
            "deployable": val.exact_lift_pp > 0 and test.exact_lift_pp > 0 and test.races >= 100,
            "strict_deployable": (
                val.exact_lift_pp >= 1.0
                and test.exact_lift_pp >= 1.0
                and val.top1_lift_pp >= 0.0
                and test.top1_lift_pp >= 0.0
                and test.races >= 100
            ),
        })
    candidates.sort(
        key=lambda item: (
            item["strict_deployable"],
            item["deployable"],
            item["val"]["exact_lift_pp"],
            item["val"]["top1_lift_pp"],
            item["test"]["exact_lift_pp"],
            item["test"]["top1_lift_pp"],
        ),
        reverse=True,
    )
    payload = {
        "records": len(records),
        "train_years": sorted(TRAIN_YEARS),
        "val_years": sorted(VAL_YEARS),
        "test_years": sorted(TEST_YEARS),
        "top_k": args.top_k,
        "feature_names": FEATURE_NAMES,
        "metrics": [asdict(metric) for metric in all_metrics],
        "candidates": candidates,
        "deployable_count": sum(1 for item in candidates if item["deployable"]),
        "strict_deployable_count": sum(1 for item in candidates if item["strict_deployable"]),
        "elapsed_sec": round(time.time() - started, 3),
    }
    Path(args.out_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# KCYCLE drug-discovery trifecta search",
        "",
        f"records: {payload['records']}",
        f"train: {payload['train_years']} val: {payload['val_years']} test: {payload['test_years']}",
        f"top_k: {payload['top_k']}",
        f"deployable_count: {payload['deployable_count']} strict_deployable_count: {payload['strict_deployable_count']}",
        "",
        "| name | split | races | exact | board_exact | lift_pp | top1 | board_top1 | top1_lift_pp |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for metric in sorted(all_metrics, key=lambda m: (m.name, m.split)):
        lines.append(
            f"| {metric.name} | {metric.split} | {metric.races} | {metric.exact:.4f} | "
            f"{metric.board_exact:.4f} | {metric.exact_lift_pp:.2f} | {metric.top1:.4f} | "
            f"{metric.board_top1:.4f} | {metric.top1_lift_pp:.2f} |"
        )
    Path(args.out_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshots", default=str(ROOT / "data" / "kcycle_trifecta_snapshots.jsonl"))
    parser.add_argument("--out-json", default=str(ROOT / "data" / "kcycle_drug_discovery_trifecta_results.json"))
    parser.add_argument("--out-md", default=str(ROOT / "data" / "kcycle_drug_discovery_trifecta_results.md"))
    parser.add_argument("--top-k", nargs="+", type=int, default=[10, 20, 40])
    payload = run(parser.parse_args())
    print(json.dumps({
        "records": payload["records"],
        "deployable_count": payload["deployable_count"],
        "strict_deployable_count": payload["strict_deployable_count"],
        "best": payload["candidates"][:3],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
