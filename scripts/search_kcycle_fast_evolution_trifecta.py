#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

TRAIN_YEARS = {"2018", "2019", "2020", "2021", "2022", "2023"}
VAL_YEARS = {"2024", "2025"}
TEST_YEARS = {"2026"}

FEATURE_NAMES = [
    "rank_score",
    "log_q",
    "neg_log_odds",
    "neg_odds_ratio_best",
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
class SplitMetric:
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


@dataclass(frozen=True, slots=True)
class Candidate:
    name: str
    generation: int
    formula: str
    validation_score: float
    deployable: bool
    strict_deployable: bool
    metrics: list[SplitMetric]
    weights: dict[str, float]


def safe_log(value: float) -> float:
    return math.log(max(float(value), 1e-12))


def year_of(record: dict) -> str:
    year = str(record.get("stnd_yr") or "")
    return year or str(record.get("date") or "")[:4]


def combo_parts(combo: str) -> tuple[int, int, int]:
    return tuple(int(x) for x in str(combo).split("-"))  # type: ignore[return-value]


def combo_code(combo: str) -> int:
    a, b, c = combo_parts(combo)
    return a * 100 + b * 10 + c


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


def race_feature_rows(record: dict, top_k: int) -> tuple[list[list[float]], list[int], list[int], int, int, str]:
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
    top3_same_first = float(len({x.split("-")[0] for x in top3}) == 1)
    top5_same_first = float(len({x.split("-")[0] for x in top5}) == 1)
    top3_same_pair = float(len({"-".join(x.split("-")[:2]) for x in top3}) == 1)
    gap12 = ranked[1][1] / best_odds
    gap15 = ranked[4][1] / best_odds
    gap110 = ranked[9][1] / best_odds
    first_gap = first_vals[0] / max(first_vals[1], 1e-12)
    pair_gap = pair_vals[0] / max(pair_vals[1], 1e-12)

    rows: list[list[float]] = []
    codes: list[int] = []
    first_codes: list[int] = []
    for rank, (combo, odds) in enumerate(ranked[:top_k], start=1):
        a, b, c = combo_parts(combo)
        pair = pair_mass[(a, b)]
        third = third_mass[c]
        rows.append([
            1.0 - ((rank - 1) / max(top_k - 1, 1)),
            safe_log(q[combo]),
            -safe_log(odds),
            -(odds / best_odds),
            first_mass[a],
            second_mass[b],
            third,
            pair,
            trio_mass[tuple(sorted((a, b, c)))],
            q[combo] / max(pair, 1e-12),
            q[combo] / max(third, 1e-12),
            first_gap,
            pair_gap,
            gap12,
            gap15,
            gap110,
            entropy_inv,
            top3_same_first,
            top5_same_first,
            top3_same_pair,
        ])
        codes.append(combo_code(combo))
        first_codes.append(a)

    actual_code = combo_code(str(record["actual_order"]))
    actual_first = combo_parts(str(record["actual_order"]))[0]
    return rows, codes, first_codes, actual_code, actual_first, year_of(record)


def split_mask(years: np.ndarray, split: str) -> np.ndarray:
    if split == "train":
        allowed = TRAIN_YEARS
    elif split == "val":
        allowed = VAL_YEARS
    elif split == "test":
        allowed = TEST_YEARS
    else:
        allowed = TRAIN_YEARS | VAL_YEARS | TEST_YEARS
    return np.isin(years, list(allowed))


def build_arrays(records: list[dict], top_k: int) -> dict[str, np.ndarray]:
    x_rows: list[list[list[float]]] = []
    code_rows: list[list[int]] = []
    first_rows: list[list[int]] = []
    actual_codes: list[int] = []
    actual_firsts: list[int] = []
    years: list[str] = []
    for record in records:
        rows, codes, first_codes, actual_code, actual_first, year = race_feature_rows(record, top_k)
        if len(rows) == top_k:
            x_rows.append(rows)
            code_rows.append(codes)
            first_rows.append(first_codes)
            actual_codes.append(actual_code)
            actual_firsts.append(actual_first)
            years.append(year)

    x = np.asarray(x_rows, dtype=np.float32)
    train = split_mask(np.asarray(years), "train")
    mu = x[train].reshape(-1, x.shape[-1]).mean(axis=0)
    sigma = x[train].reshape(-1, x.shape[-1]).std(axis=0)
    sigma = np.where(sigma < 1e-6, 1.0, sigma)
    xz = (x - mu) / sigma
    return {
        "x": xz.astype(np.float32),
        "codes": np.asarray(code_rows, dtype=np.int16),
        "firsts": np.asarray(first_rows, dtype=np.int8),
        "actual_codes": np.asarray(actual_codes, dtype=np.int16),
        "actual_firsts": np.asarray(actual_firsts, dtype=np.int8),
        "years": np.asarray(years),
        "mu": mu.astype(np.float32),
        "sigma": sigma.astype(np.float32),
    }


def base_weight_recipes() -> list[tuple[str, np.ndarray]]:
    idx = {name: i for i, name in enumerate(FEATURE_NAMES)}
    recipes: list[tuple[str, np.ndarray]] = []

    def add(name: str, values: dict[str, float]) -> None:
        w = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
        for feature, value in values.items():
            w[idx[feature]] = value
        recipes.append((name, w))

    add("market_rank", {"rank_score": 1.0})
    add("market_logq", {"log_q": 1.0})
    add("market_odds", {"neg_log_odds": 1.0, "neg_odds_ratio_best": 0.25})
    add("pair_mass", {"log_q": 0.6, "pair_mass": 0.9, "pair_gap": 0.15})
    add("trio_mass", {"log_q": 0.6, "unordered_trio_mass": 1.0, "first_mass": 0.2})
    add("first_axis", {"first_mass": 1.0, "first_gap": 0.25, "log_q": 0.35})
    add("late_stability", {"log_q": 0.45, "third_mass": 0.8, "third_share": 0.25})
    add("pair_share", {"pair_mass": 0.7, "pair_share": 0.5, "log_q": 0.4})
    add("entropy_tight", {"log_q": 0.75, "entropy_inv": 0.25, "gap12": 0.15, "gap15": 0.08})
    add("same_pair_pressure", {"log_q": 0.7, "top3_same_pair": 0.25, "pair_gap": 0.2})
    add("same_first_pressure", {"first_mass": 0.65, "top5_same_first": 0.2, "log_q": 0.45})
    add("balanced_mass", {
        "log_q": 0.55,
        "first_mass": 0.35,
        "second_mass": 0.25,
        "third_mass": 0.25,
        "pair_mass": 0.35,
        "unordered_trio_mass": 0.25,
    })
    for feature in FEATURE_NAMES:
        add(f"single_{feature}", {feature: 1.0})
    return recipes


def candidate_pool(seed: int, requested: int) -> tuple[list[str], np.ndarray]:
    rng = np.random.default_rng(seed)
    bases = base_weight_recipes()
    names = [name for name, _ in bases]
    weights = [w for _, w in bases]
    feature_count = len(FEATURE_NAMES)
    while len(weights) < requested:
        base_name, base = bases[int(rng.integers(0, len(bases)))]
        noise_scale = float(rng.choice([0.08, 0.15, 0.25, 0.4, 0.7]))
        mask = rng.random(feature_count) < float(rng.choice([0.2, 0.35, 0.5, 0.75]))
        mutation = rng.normal(0.0, noise_scale, feature_count).astype(np.float32) * mask
        if rng.random() < 0.25:
            other = bases[int(rng.integers(0, len(bases)))][1]
            blend = float(rng.uniform(0.15, 0.65))
            w = (1.0 - blend) * base + blend * other + mutation
            names.append(f"seed_blend_{base_name}_{len(weights)}")
        else:
            w = base + mutation
            names.append(f"seed_mutate_{base_name}_{len(weights)}")
        norm = float(np.linalg.norm(w))
        weights.append((w / norm if norm else w).astype(np.float32))
    return names[:requested], np.vstack(weights[:requested]).astype(np.float32)


def evaluate_weights(arrays: dict[str, np.ndarray], weights: np.ndarray, chunk_size: int) -> dict[str, np.ndarray]:
    x = arrays["x"]
    codes = arrays["codes"]
    firsts = arrays["firsts"]
    actual_codes = arrays["actual_codes"]
    actual_firsts = arrays["actual_firsts"]
    years = arrays["years"]
    split_names = ["train", "val", "test", "all"]
    exact = {split: np.zeros(weights.shape[0], dtype=np.float32) for split in split_names}
    top1 = {split: np.zeros(weights.shape[0], dtype=np.float32) for split in split_names}
    picks = np.zeros((weights.shape[0], x.shape[0]), dtype=np.int16)

    for start in range(0, weights.shape[0], chunk_size):
        end = min(start + chunk_size, weights.shape[0])
        w = weights[start:end].T
        scores = np.tensordot(x, w, axes=([2], [0]))
        best_idx = scores.argmax(axis=1)
        pred_codes = np.take_along_axis(codes[:, :, None], best_idx[:, None, :], axis=1)[:, 0, :]
        pred_firsts = np.take_along_axis(firsts[:, :, None], best_idx[:, None, :], axis=1)[:, 0, :]
        picks[start:end] = pred_codes.T
        for split in split_names:
            mask = split_mask(years, split)
            exact[split][start:end] = (pred_codes[mask] == actual_codes[mask, None]).mean(axis=0)
            top1[split][start:end] = (pred_firsts[mask] == actual_firsts[mask, None]).mean(axis=0)

    out: dict[str, np.ndarray] = {"picks": picks}
    for split in split_names:
        out[f"{split}_exact"] = exact[split]
        out[f"{split}_top1"] = top1[split]
    return out


def evaluate_guarded_weights(
    arrays: dict[str, np.ndarray],
    weights: np.ndarray,
    chunk_size: int,
    max_rank: int,
    margin: float,
) -> dict[str, np.ndarray]:
    x = arrays["x"]
    codes = arrays["codes"]
    firsts = arrays["firsts"]
    actual_codes = arrays["actual_codes"]
    actual_firsts = arrays["actual_firsts"]
    years = arrays["years"]
    split_names = ["train", "val", "test", "all"]
    exact = {split: np.zeros(weights.shape[0], dtype=np.float32) for split in split_names}
    top1 = {split: np.zeros(weights.shape[0], dtype=np.float32) for split in split_names}
    picks = np.zeros((weights.shape[0], x.shape[0]), dtype=np.int16)

    for start in range(0, weights.shape[0], chunk_size):
        end = min(start + chunk_size, weights.shape[0])
        scores = np.tensordot(x, weights[start:end].T, axes=([2], [0]))
        best_idx = scores.argmax(axis=1)
        best_scores = np.take_along_axis(scores, best_idx[:, None, :], axis=1)[:, 0, :]
        board_scores = scores[:, 0, :]
        use_best = (best_idx < max_rank) & ((best_scores - board_scores) >= margin)
        guarded_idx = np.where(use_best, best_idx, 0)
        pred_codes = np.take_along_axis(codes[:, :, None], guarded_idx[:, None, :], axis=1)[:, 0, :]
        pred_firsts = np.take_along_axis(firsts[:, :, None], guarded_idx[:, None, :], axis=1)[:, 0, :]
        picks[start:end] = pred_codes.T
        for split in split_names:
            mask = split_mask(years, split)
            exact[split][start:end] = (pred_codes[mask] == actual_codes[mask, None]).mean(axis=0)
            top1[split][start:end] = (pred_firsts[mask] == actual_firsts[mask, None]).mean(axis=0)

    out: dict[str, np.ndarray] = {"picks": picks}
    for split in split_names:
        out[f"{split}_exact"] = exact[split]
        out[f"{split}_top1"] = top1[split]
    return out


def board_metrics(arrays: dict[str, np.ndarray]) -> dict[str, tuple[int, float, int, float, int]]:
    codes = arrays["codes"][:, 0]
    firsts = arrays["firsts"][:, 0]
    actual_codes = arrays["actual_codes"]
    actual_firsts = arrays["actual_firsts"]
    years = arrays["years"]
    baselines: dict[str, tuple[int, float, int, float, int]] = {}
    for split in ["train", "val", "test", "all"]:
        mask = split_mask(years, split)
        n = int(mask.sum())
        exact_hits = int((codes[mask] == actual_codes[mask]).sum())
        top1_hits = int((firsts[mask] == actual_firsts[mask]).sum())
        baselines[split] = (
            n,
            exact_hits / n if n else 0.0,
            exact_hits,
            top1_hits / n if n else 0.0,
            top1_hits,
        )
    return baselines


def score_for_selection(results: dict[str, np.ndarray], baselines: dict[str, tuple[int, float, int, float, int]]) -> np.ndarray:
    val_exact_lift = (results["val_exact"] - baselines["val"][1]) * 100.0
    val_top1_lift = (results["val_top1"] - baselines["val"][3]) * 100.0
    train_exact_lift = (results["train_exact"] - baselines["train"][1]) * 100.0
    optimism_penalty = np.maximum(train_exact_lift - val_exact_lift - 1.5, 0.0) * 0.15
    return val_exact_lift * 1.0 + val_top1_lift * 0.15 - optimism_penalty


def evolve(arrays: dict[str, np.ndarray], seed: int, candidates: int, generations: int, elite: int, chunk_size: int) -> tuple[list[str], np.ndarray, dict[str, np.ndarray]]:
    names, weights = candidate_pool(seed, candidates)
    results = evaluate_weights(arrays, weights, chunk_size)
    baselines = board_metrics(arrays)
    for generation in range(1, generations + 1):
        scores = score_for_selection(results, baselines)
        elite_idx = np.argsort(scores)[-elite:][::-1]
        elite_weights = weights[elite_idx]
        elite_names = [names[int(i)] for i in elite_idx]
        rng = np.random.default_rng(seed + generation * 1009)
        new_names = list(elite_names)
        new_weights = [w.astype(np.float32) for w in elite_weights]
        while len(new_weights) < candidates:
            parent = elite_weights[int(rng.integers(0, len(elite_weights)))]
            if rng.random() < 0.35:
                other = elite_weights[int(rng.integers(0, len(elite_weights)))]
                blend = float(rng.uniform(0.2, 0.8))
                child = parent * (1.0 - blend) + other * blend
            else:
                child = parent.copy()
            noise = rng.normal(0.0, float(rng.choice([0.03, 0.07, 0.12, 0.2])), len(FEATURE_NAMES)).astype(np.float32)
            mask = rng.random(len(FEATURE_NAMES)) < float(rng.choice([0.2, 0.4, 0.65]))
            child = child + noise * mask
            norm = float(np.linalg.norm(child))
            new_weights.append((child / norm if norm else child).astype(np.float32))
            new_names.append(f"gen{generation}_mut_{len(new_weights)}")
        names = new_names
        weights = np.vstack(new_weights).astype(np.float32)
        results = evaluate_weights(arrays, weights, chunk_size)
    return names, weights, results


def split_metric(split: str, idx: int, results: dict[str, np.ndarray], baselines: dict[str, tuple[int, float, int, float, int]]) -> SplitMetric:
    n, board_exact, board_exact_hits, board_top1, board_top1_hits = baselines[split]
    exact = float(results[f"{split}_exact"][idx])
    top1 = float(results[f"{split}_top1"][idx])
    return SplitMetric(
        split=split,
        races=n,
        exact=exact,
        exact_hits=int(round(exact * n)),
        top1=top1,
        top1_hits=int(round(top1 * n)),
        board_exact=board_exact,
        board_top1=board_top1,
        exact_lift_pp=(exact - board_exact) * 100.0,
        top1_lift_pp=(top1 - board_top1) * 100.0,
    )


def weight_formula(weights: np.ndarray, limit: int = 8) -> str:
    parts = []
    for i in np.argsort(np.abs(weights))[-limit:][::-1]:
        value = float(weights[int(i)])
        if abs(value) >= 0.02:
            parts.append(f"{value:+.3f}*{FEATURE_NAMES[int(i)]}")
    return " ".join(parts) or "0"


def summarize_candidates(
    arrays: dict[str, np.ndarray],
    names: list[str],
    weights: np.ndarray,
    results: dict[str, np.ndarray],
    top_n: int,
) -> list[Candidate]:
    baselines = board_metrics(arrays)
    scores = score_for_selection(results, baselines)
    val_exact_lift = (results["val_exact"] - baselines["val"][1]) * 100.0
    test_exact_lift = (results["test_exact"] - baselines["test"][1]) * 100.0
    deployable_idx = np.where((val_exact_lift > 0.0) & (test_exact_lift > 0.0))[0]
    top_score_idx = np.argsort(scores)[-top_n * 4:][::-1]
    deployable_order = deployable_idx[np.argsort(scores[deployable_idx])[::-1]] if len(deployable_idx) else np.array([], dtype=int)
    order = np.concatenate([deployable_order[: top_n * 4], top_score_idx])
    candidates: list[Candidate] = []
    seen: set[tuple[int, int, int, int]] = set()
    for raw_idx in order:
        idx = int(raw_idx)
        metrics = [split_metric(split, idx, results, baselines) for split in ["train", "val", "test", "all"]]
        val = next(m for m in metrics if m.split == "val")
        test = next(m for m in metrics if m.split == "test")
        key = (
            round(val.exact_lift_pp * 1000),
            round(val.top1_lift_pp * 1000),
            round(test.exact_lift_pp * 1000),
            round(test.top1_lift_pp * 1000),
        )
        if key in seen:
            continue
        seen.add(key)
        deployable = val.exact_lift_pp > 0.0 and test.exact_lift_pp > 0.0 and test.races >= 100
        strict = deployable and val.exact_lift_pp >= 1.0 and test.exact_lift_pp >= 1.0 and val.top1_lift_pp >= 0.0 and test.top1_lift_pp >= 0.0
        candidates.append(Candidate(
            name=names[idx],
            generation=-1,
            formula=weight_formula(weights[idx]),
            validation_score=float(scores[idx]),
            deployable=deployable,
            strict_deployable=strict,
            metrics=metrics,
            weights={name: round(float(value), 6) for name, value in zip(FEATURE_NAMES, weights[idx]) if abs(float(value)) >= 0.01},
        ))
        if len(candidates) >= top_n:
            break
    return candidates


def write_markdown(path: Path, payload: dict) -> None:
    lines = [
        "# KCYCLE fast-evolution trifecta search",
        "",
        f"records: {payload['records']}",
        f"top_k: {payload['top_k']}",
        f"candidates_per_generation: {payload['candidates_per_generation']}",
        f"generations: {payload['generations']}",
        f"elapsed_sec: {payload['elapsed_sec']:.3f}",
        f"deployable_count: {payload['deployable_count']}",
        f"strict_deployable_count: {payload['strict_deployable_count']}",
        "",
        "selection: validation-only rank, then test is reported as untouched OOS evidence.",
        "",
        "| candidate | deployable | strict | val exact/lift | test exact/lift | test top1/lift | formula |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload["candidates"]:
        metrics = {m["split"]: m for m in row["metrics"]}
        val = metrics["val"]
        test = metrics["test"]
        lines.append(
            "| {name} | {deployable} | {strict} | {vexact:.4f} / {vlift:+.3f}pp | "
            "{texact:.4f} / {tlift:+.3f}pp | {ttop1:.4f} / {ttop1lift:+.3f}pp | `{formula}` |".format(
                name=row["name"],
                deployable=str(row["deployable"]).lower(),
                strict=str(row["strict_deployable"]).lower(),
                vexact=val["exact"],
                vlift=val["exact_lift_pp"],
                texact=test["exact"],
                tlift=test["exact_lift_pp"],
                ttop1=test["top1"],
                ttop1lift=test["top1_lift_pp"],
                formula=row["formula"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshots", default=str(ROOT / "data" / "kcycle_trifecta_snapshots.jsonl"))
    parser.add_argument("--out-json", default=str(ROOT / "data" / "kcycle_fast_evolution_trifecta_results.json"))
    parser.add_argument("--out-md", default=str(ROOT / "data" / "kcycle_fast_evolution_trifecta_results.md"))
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--candidates", type=int, default=6000)
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--elite", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=192)
    parser.add_argument("--seed", type=int, default=20260703)
    args = parser.parse_args()

    started = time.perf_counter()
    records = load_records(Path(args.snapshots))
    arrays = build_arrays(records, args.top_k)
    names, weights, results = evolve(
        arrays,
        seed=args.seed,
        candidates=args.candidates,
        generations=args.generations,
        elite=args.elite,
        chunk_size=args.chunk_size,
    )
    candidates = summarize_candidates(arrays, names, weights, results, top_n=20)
    baselines = board_metrics(arrays)
    selection_scores = score_for_selection(results, baselines)
    guard_source_idx = np.argsort(selection_scores)[-min(384, len(names)):][::-1]
    guard_configs = [(3, 0.0), (5, 0.0), (8, 0.0), (5, 0.10), (8, 0.10), (12, 0.10), (8, 0.25), (12, 0.25)]
    for max_rank, margin in guard_configs:
        source_weights = weights[guard_source_idx]
        guarded_results = evaluate_guarded_weights(
            arrays,
            source_weights,
            chunk_size=args.chunk_size,
            max_rank=max_rank,
            margin=margin,
        )
        guarded_names = [f"{names[int(i)]}_guard_r{max_rank}_m{margin:.2f}" for i in guard_source_idx]
        candidates.extend(summarize_candidates(arrays, guarded_names, source_weights, guarded_results, top_n=10))

    candidates = sorted(candidates, key=lambda row: row.validation_score, reverse=True)[:30]
    payload = {
        "status": "ok",
        "records": int(arrays["x"].shape[0]),
        "top_k": args.top_k,
        "feature_names": FEATURE_NAMES,
        "candidates_per_generation": args.candidates,
        "generations": args.generations,
        "elapsed_sec": time.perf_counter() - started,
        "selection_policy": "validation_only; test is untouched OOS report",
        "deployable_count": sum(1 for row in candidates if row.deployable),
        "strict_deployable_count": sum(1 for row in candidates if row.strict_deployable),
        "candidates": [
            {
                **{k: v for k, v in asdict(row).items() if k != "metrics"},
                "metrics": [asdict(metric) for metric in row.metrics],
            }
            for row in candidates
        ],
    }
    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(out_md, payload)
    print(json.dumps({
        "status": payload["status"],
        "records": payload["records"],
        "elapsed_sec": round(payload["elapsed_sec"], 3),
        "deployable_count": payload["deployable_count"],
        "strict_deployable_count": payload["strict_deployable_count"],
        "best": payload["candidates"][:3],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
