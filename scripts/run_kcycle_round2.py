#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression

ROOT: Final = Path(__file__).resolve().parents[1]
SCRIPTS: Final = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from kcycle_eval_common import (  # noqa: E402
    CURRENT_AXIS_EXACT,
    FEATURE_NAMES,
    ROOT as COMMON_ROOT,
    as_pp,
    assert_purchase_monotonic,
    assert_reproduction,
    combo_parts,
    grade_score,
    load_breakthrough_weights,
    load_entries,
    load_snapshot_records,
    market_rank_scores,
    parse_float,
    race_key,
    score_metrics,
    softmax,
    split_mask,
    weight_rank_scores,
)

OUT_JSON: Final = COMMON_ROOT / "data" / "kcycle_round2_results.json"
OUT_MD: Final = COMMON_ROOT / "data" / "kcycle_round2_results.md"
PROGRESS: Final = COMMON_ROOT / "runs" / "prediction_uplift_progress.md"
TOP_KS: Final = (10, 20, 40)
UNIVERSE_K: Final = 40
RIDGE_ALPHAS: Final = (0.01, 0.1, 1.0, 10.0, 50.0)
PL_ALPHAS: Final = (0.01, 0.1, 1.0, 10.0)
BETAS: Final = tuple(round(float(x), 2) for x in np.arange(0.50, 1.00, 0.05))


@dataclass(frozen=True, slots=True)
class ComboFit:
    name: str
    alpha: float
    feature_names: list[str]
    coefficients: np.ndarray
    scores: np.ndarray
    metrics: dict[str, dict[str, dict[str, float | int]]]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def append_progress(line: str) -> None:
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    with PROGRESS.open("a", encoding="utf-8") as handle:
        handle.write(f"- {utc_now()} Round2: {line}\n")


def entrant_index(entrants: list[dict]) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for row in entrants:
        bno = int(str(row.get("back_no") or "0").strip() or "0")
        if bno:
            out[bno] = row
    return out


def rank_map(values: dict[int, float], reverse: bool) -> dict[int, float]:
    finite = {k: v for k, v in values.items() if math.isfinite(v)}
    ordered = sorted(finite, key=lambda key: (finite[key], key), reverse=reverse)
    return {bno: float(rank) for rank, bno in enumerate(ordered, start=1)}


def line_feature_tensor(
    records: list[dict],
    arrays: dict[str, np.ndarray],
    entries: dict[tuple[str, str, str], list[dict]],
) -> tuple[np.ndarray, list[str], int]:
    names = [
        "same_training_pairs",
        "grade_gap_12",
        "grade_span_trio",
        "gear_span_trio",
        "age_span_trio",
        "rec200_rank_sum",
        "rec200_best_rank",
        "win_rank_sum",
        "high_rank_sum",
        "first_second_win_rank_gap",
        "first_second_high_rank_gap",
    ]
    rows = np.zeros((*arrays["codes"].shape, len(names)), dtype=np.float64)
    joined = 0
    for race_idx, record in enumerate(records):
        entrants = entries.get(race_key(record))
        if not entrants:
            continue
        joined += 1
        by_bno = entrant_index(entrants)
        train_places = {bno: str(row.get("trng_plc_nm") or "") for bno, row in by_bno.items()}
        grades = {bno: grade_score(row.get("racer_grd_cur_cd") or row.get("racer_grd_cd")) for bno, row in by_bno.items()}
        gears = {bno: parse_float(row.get("gear_rate")) for bno, row in by_bno.items()}
        ages = {bno: parse_float(row.get("racer_age")) for bno, row in by_bno.items()}
        rec200 = {bno: parse_float(row.get("rec_200m_scr")) for bno, row in by_bno.items()}
        win = {bno: parse_float(row.get("win_rate")) for bno, row in by_bno.items()}
        high = {bno: parse_float(row.get("high_rate")) for bno, row in by_bno.items()}
        rec_rank = rank_map(rec200, reverse=False)
        win_rank = rank_map(win, reverse=True)
        high_rank = rank_map(high, reverse=True)
        for combo_idx, code in enumerate(arrays["codes"][race_idx]):
            a, b, c = combo_parts(f"{int(code) // 100}-{(int(code) // 10) % 10}-{int(code) % 10}")
            trio = (a, b, c)
            place_pairs = sum(
                1
                for left, right in ((a, b), (a, c), (b, c))
                if train_places.get(left) and train_places.get(left) == train_places.get(right)
            )
            grade_vals = [grades.get(x, 0.0) for x in trio]
            gear_vals = [gears.get(x, float("nan")) for x in trio]
            age_vals = [ages.get(x, float("nan")) for x in trio]
            clean_gear = [x for x in gear_vals if math.isfinite(x)]
            clean_age = [x for x in age_vals if math.isfinite(x)]
            rows[race_idx, combo_idx] = np.asarray(
                [
                    float(place_pairs),
                    abs(grades.get(a, 0.0) - grades.get(b, 0.0)),
                    max(grade_vals) - min(grade_vals),
                    max(clean_gear) - min(clean_gear) if clean_gear else 0.0,
                    max(clean_age) - min(clean_age) if clean_age else 0.0,
                    sum(rec_rank.get(x, 4.0) for x in trio),
                    min(rec_rank.get(x, 4.0) for x in trio),
                    sum(win_rank.get(x, 4.0) for x in trio),
                    sum(high_rank.get(x, 4.0) for x in trio),
                    abs(win_rank.get(a, 4.0) - win_rank.get(b, 4.0)),
                    abs(high_rank.get(a, 4.0) - high_rank.get(b, 4.0)),
                ],
                dtype=np.float64,
            )
    train = split_mask(arrays["years"], "train")
    flat = rows[train].reshape(-1, rows.shape[-1])
    mu = flat.mean(axis=0)
    sigma = flat.std(axis=0)
    sigma = np.where(sigma < 1e-9, 1.0, sigma)
    return ((rows - mu) / sigma).astype(np.float64), names, joined


def ridge_fit_scores(arrays: dict[str, np.ndarray], features: np.ndarray, names: list[str], label: str) -> ComboFit:
    train = split_mask(arrays["years"], "train")
    val = split_mask(arrays["years"], "val")
    y = (arrays["codes"] == arrays["actual_codes"][:, None]).astype(np.float64)
    x_train = features[train].reshape(-1, features.shape[-1])
    y_train = y[train].reshape(-1)
    weights = np.where(y_train > 0.5, 39.0, 1.0)
    x_aug = np.column_stack([np.ones(x_train.shape[0]), x_train])
    xtw = x_aug.T * weights
    best: ComboFit | None = None
    for alpha in RIDGE_ALPHAS:
        reg = np.eye(x_aug.shape[1], dtype=np.float64) * alpha
        reg[0, 0] = 0.0
        coef = np.linalg.solve(xtw @ x_aug + reg, xtw @ y_train)
        scores = (np.column_stack([np.ones(features.reshape(-1, features.shape[-1]).shape[0]), features.reshape(-1, features.shape[-1])]) @ coef).reshape(features.shape[:2])
        metrics = score_metrics(arrays, scores, TOP_KS, ("train", "val"))
        row = ComboFit(label, alpha, ["intercept", *names], coef, scores, metrics)
        if best is None:
            best = row
            continue
        current = float(row.metrics["20"]["val"]["purchase_exact"])
        prior = float(best.metrics["20"]["val"]["purchase_exact"])
        if (current, float(row.metrics["20"]["val"]["selection_exact"]), -alpha) > (
            prior,
            float(best.metrics["20"]["val"]["selection_exact"]),
            -best.alpha,
        ):
            best = row
    if best is None:
        raise RuntimeError("ridge fit produced no candidate")
    if int(val.sum()) == 0:
        raise RuntimeError("validation split is empty")
    return ComboFit(best.name, best.alpha, best.feature_names, best.coefficients, best.scores, score_metrics(arrays, best.scores, TOP_KS))


def market_prob_scores(records: list[dict], arrays: dict[str, np.ndarray]) -> np.ndarray:
    scores = np.zeros(arrays["codes"].shape, dtype=np.float64)
    for row_idx, record in enumerate(records):
        board = {str(k): float(v) for k, v in (record.get("board") or {}).items() if float(v) > 0}
        inv = {combo: 1.0 / odds for combo, odds in board.items()}
        total = sum(inv.values())
        for col_idx, code in enumerate(arrays["codes"][row_idx]):
            combo = f"{int(code) // 100}-{(int(code) // 10) % 10}-{int(code) % 10}"
            scores[row_idx, col_idx] = math.log(max(inv.get(combo, 0.0) / total if total else 0.0, 1e-15))
    return scores


def blend_scores(market_scores: np.ndarray, model_scores: np.ndarray, beta: float) -> np.ndarray:
    return beta * market_scores + (1.0 - beta) * model_scores


def select_blend(
    records: list[dict],
    arrays: dict[str, np.ndarray],
    market_scores: np.ndarray,
    model_scores: np.ndarray,
) -> dict:
    grid = []
    for beta in BETAS:
        scores = blend_scores(market_scores, model_scores, beta)
        metrics = score_metrics(arrays, scores, TOP_KS, ("train", "val"))
        grid.append({"beta": beta, "metrics": metrics})
    grid.sort(
        key=lambda row: (
            float(row["metrics"]["20"]["val"]["purchase_exact"]),
            float(row["metrics"]["20"]["val"]["selection_exact"]),
            -float(row["beta"]),
        ),
        reverse=True,
    )
    selected = grid[0]
    selected_scores = blend_scores(market_scores, model_scores, float(selected["beta"]))
    selected_metrics = score_metrics(arrays, selected_scores, TOP_KS)
    return {"selected_beta": selected["beta"], "selected_metrics": selected_metrics, "grid": grid, "records": len(records)}


RUNNER_FEATURES: Final = [
    "log_first_mass",
    "grade_score",
    "grade_ss",
    "grade_s",
    "grade_a",
    "grade_b",
    "win_rate_z",
    "high_rate_z",
    "gear_z",
    "rec200_z",
    "age_z",
    "same_training_teammates",
]


def first_masses(record: dict) -> dict[int, float]:
    inv = {combo: 1.0 / float(odds) for combo, odds in (record.get("board") or {}).items() if float(odds) > 0}
    total = sum(inv.values())
    masses = {bno: 0.0 for bno in range(1, 8)}
    if total <= 0.0:
        return masses
    for combo, value in inv.items():
        masses[combo_parts(combo)[0]] += value / total
    return masses


def runner_feature_matrix(record: dict, entrants: list[dict]) -> tuple[list[int], np.ndarray]:
    by_bno = entrant_index(entrants)
    bnos = sorted(by_bno)
    masses = first_masses(record)
    numeric_fields = ("win_rate", "high_rate", "gear_rate", "rec_200m_scr", "racer_age")
    raw = np.asarray(
        [[parse_float(by_bno[bno].get(field)) for field in numeric_fields] for bno in bnos],
        dtype=np.float64,
    )
    med = np.nanmedian(raw, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    raw = np.where(np.isfinite(raw), raw, med)
    std = raw.std(axis=0)
    std = np.where(std < 1e-9, 1.0, std)
    z = (raw - raw.mean(axis=0)) / std
    places = {bno: str(by_bno[bno].get("trng_plc_nm") or "") for bno in bnos}
    place_counts = {bno: sum(1 for other in bnos if other != bno and places.get(other) == places.get(bno) and places.get(bno)) for bno in bnos}
    rows = []
    for idx, bno in enumerate(bnos):
        grade = grade_score(by_bno[bno].get("racer_grd_cur_cd") or by_bno[bno].get("racer_grd_cd"))
        rows.append(
            [
                math.log(max(masses.get(bno, 0.0), 1e-12)),
                grade,
                float(grade >= 5.0),
                float(grade == 4.0),
                float(grade == 3.0),
                float(grade == 2.0),
                z[idx, 0],
                z[idx, 1],
                z[idx, 2],
                -z[idx, 3],
                -z[idx, 4],
                float(place_counts[bno]),
            ]
        )
    return bnos, np.asarray(rows, dtype=np.float64)


def joined_runner_data(records: list[dict], entries: dict[tuple[str, str, str], list[dict]]) -> list[tuple[int, dict, list[int], np.ndarray]]:
    rows = []
    for idx, record in enumerate(records):
        entrants = entries.get(race_key(record))
        if not entrants:
            continue
        bnos, features = runner_feature_matrix(record, entrants)
        if len(bnos) >= 5:
            rows.append((idx, record, bnos, features))
    return rows


def first_winner_logistic_scores(
    records: list[dict],
    arrays: dict[str, np.ndarray],
    entries: dict[tuple[str, str, str], list[dict]],
) -> np.ndarray:
    runner_rows = joined_runner_data(records, entries)
    x_rows = []
    y_rows = []
    for _, record, bnos, features in runner_rows:
        year = str(record.get("stnd_yr") or str(record.get("date") or "")[:4])
        if not split_mask(np.asarray([year]), "train")[0]:
            continue
        actual_first = combo_parts(str(record["actual_order"]))[0]
        x_rows.extend(features.tolist())
        y_rows.extend([1 if bno == actual_first else 0 for bno in bnos])
    if not x_rows:
        return market_prob_scores(records, arrays)
    clf = LogisticRegression(max_iter=1000, C=0.5, class_weight="balanced", random_state=20260712)
    clf.fit(np.asarray(x_rows, dtype=np.float64), np.asarray(y_rows, dtype=np.int8))
    scores = market_prob_scores(records, arrays)
    by_record = {idx: (bnos, features) for idx, _, bnos, features in runner_rows}
    for row_idx, (bnos, features) in by_record.items():
        logits = clf.decision_function(features)
        exp_u = {bno: math.exp(min(max(float(value), -30.0), 30.0)) for bno, value in zip(bnos, logits)}
        total_exp = sum(exp_u.values())
        for col_idx, code in enumerate(arrays["codes"][row_idx]):
            a, b, c = combo_parts(f"{int(code) // 100}-{(int(code) // 10) % 10}-{int(code) % 10}")
            if a not in exp_u or b not in exp_u or c not in exp_u:
                continue
            scores[row_idx, col_idx] = (
                math.log(exp_u[a] / max(total_exp, 1e-15))
                + math.log(exp_u[b] / max(total_exp - exp_u[a], 1e-15))
                + math.log(exp_u[c] / max(total_exp - exp_u[a] - exp_u[b], 1e-15))
            )
    return scores


def pl_loss(beta: np.ndarray, data: list[tuple[int, dict, list[int], np.ndarray]], alpha: float) -> tuple[float, np.ndarray]:
    loss = 0.5 * alpha * float(beta @ beta)
    grad = alpha * beta
    for _, record, bnos, features in data:
        actual = combo_parts(str(record["actual_order"]))
        utilities = features @ beta
        remaining = list(range(len(bnos)))
        for picked in actual:
            if picked not in bnos:
                continue
            picked_idx = bnos.index(picked)
            local_pos = remaining.index(picked_idx)
            probs = softmax(utilities[remaining])
            loss -= float(utilities[picked_idx] - math.log(np.exp(np.clip(utilities[remaining] - np.max(utilities[remaining]), -50.0, 50.0)).sum()) - float(np.max(utilities[remaining])))
            grad += probs @ features[remaining] - features[picked_idx]
            remaining.pop(local_pos)
    return loss, grad


def pl_combo_scores(records: list[dict], arrays: dict[str, np.ndarray], runner_rows: list[tuple[int, dict, list[int], np.ndarray]], beta: np.ndarray) -> np.ndarray:
    scores = market_prob_scores(records, arrays)
    by_record = {idx: (bnos, features) for idx, _, bnos, features in runner_rows}
    for row_idx, (bnos, features) in by_record.items():
        utility = {bno: float(value) for bno, value in zip(bnos, features @ beta)}
        total = math.log(sum(math.exp(min(max(value, -30.0), 30.0)) for value in utility.values()))
        exp_u = {bno: math.exp(min(max(value, -30.0), 30.0)) for bno, value in utility.items()}
        total_exp = sum(exp_u.values())
        for col_idx, code in enumerate(arrays["codes"][row_idx]):
            a, b, c = combo_parts(f"{int(code) // 100}-{(int(code) // 10) % 10}-{int(code) % 10}")
            if a not in exp_u or b not in exp_u or c not in exp_u:
                continue
            logp = (
                math.log(exp_u[a] / max(total_exp, 1e-15))
                + math.log(exp_u[b] / max(total_exp - exp_u[a], 1e-15))
                + math.log(exp_u[c] / max(total_exp - exp_u[a] - exp_u[b], 1e-15))
            )
            scores[row_idx, col_idx] = logp + 1e-12 * total
    return scores


def fit_conditional_logit(records: list[dict], arrays: dict[str, np.ndarray], entries: dict[tuple[str, str, str], list[dict]]) -> dict:
    runner_rows = joined_runner_data(records, entries)
    train_rows = [row for row in runner_rows if split_mask(np.asarray([str(row[1].get("stnd_yr") or str(row[1].get("date") or "")[:4])]), "train")[0]]
    best: dict | None = None
    for alpha in PL_ALPHAS:
        initial = np.zeros(len(RUNNER_FEATURES), dtype=np.float64)
        result = minimize(
            lambda vec: pl_loss(vec, train_rows, alpha),
            initial,
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": 80, "ftol": 1e-7},
        )
        beta = np.asarray(result.x, dtype=np.float64)
        scores = pl_combo_scores(records, arrays, runner_rows, beta)
        metrics = score_metrics(arrays, scores, TOP_KS, ("train", "val"))
        row = {"alpha": alpha, "success": bool(result.success), "coefficients": beta, "metrics": metrics, "scores": scores}
        if best is None or (
            float(metrics["20"]["val"]["purchase_exact"]),
            float(metrics["20"]["val"]["selection_exact"]),
            -alpha,
        ) > (
            float(best["metrics"]["20"]["val"]["purchase_exact"]),
            float(best["metrics"]["20"]["val"]["selection_exact"]),
            -float(best["alpha"]),
        ):
            best = row
    if best is None:
        raise RuntimeError("conditional logit fit produced no candidate")
    return {
        "alpha": best["alpha"],
        "success": best["success"],
        "metrics": score_metrics(arrays, best["scores"], TOP_KS),
        "coefficients": {
            name: float(value)
            for name, value in sorted(zip(RUNNER_FEATURES, best["coefficients"]), key=lambda item: abs(float(item[1])), reverse=True)
        },
        "scores": best["scores"],
        "joined_records": len(runner_rows),
    }


def top_coefficients(fit: ComboFit, limit: int = 10) -> list[dict[str, float | str]]:
    rows = [
        {"feature": name, "coefficient": float(value)}
        for name, value in zip(fit.feature_names, fit.coefficients)
        if name != "intercept"
    ]
    return sorted(rows, key=lambda row: abs(float(row["coefficient"])), reverse=True)[:limit]


def lift_row(metrics: dict[str, dict[str, dict[str, float | int]]], baseline: dict[str, dict[str, dict[str, float | int]]]) -> dict[str, float]:
    return {
        "val_purchase_lift_pp": as_pp(float(metrics["20"]["val"]["purchase_exact"]) - float(baseline["20"]["val"]["purchase_exact"])),
        "test_purchase_lift_pp": as_pp(float(metrics["20"]["test"]["purchase_exact"]) - float(baseline["20"]["test"]["purchase_exact"])),
        "val_selection_lift_pp": as_pp(float(metrics["20"]["val"]["selection_exact"]) - float(baseline["20"]["val"]["selection_exact"])),
        "test_selection_lift_pp": as_pp(float(metrics["20"]["test"]["selection_exact"]) - float(baseline["20"]["test"]["selection_exact"])),
    }


def baseline_table(
    market_metrics: dict[str, dict[str, dict[str, float | int]]],
    gen2_metrics: dict[str, dict[str, dict[str, float | int]]],
) -> list[dict]:
    ensemble_payload = json.loads((COMMON_ROOT / "data" / "kcycle_ensemble_gating_results.json").read_text(encoding="utf-8"))
    ensemble = {str(row["top_k"]): row["splits"] for row in ensemble_payload["ensemble"]}
    rows = []
    for top_k in TOP_KS:
        key = str(top_k)
        rows.append({"model": "current_axis_reference", "top_k": top_k, "test_selection_exact": CURRENT_AXIS_EXACT, "test_purchase_exact": None})
        rows.append(
            {
                "model": "market_rank",
                "top_k": top_k,
                "test_selection_exact": market_metrics[key]["test"]["selection_exact"],
                "test_purchase_exact": market_metrics[key]["test"]["purchase_exact"],
            }
        )
        rows.append(
            {
                "model": "gen2_mut_436",
                "top_k": top_k,
                "test_selection_exact": gen2_metrics[key]["test"]["selection_exact"],
                "test_purchase_exact": gen2_metrics[key]["test"]["purchase_exact"],
            }
        )
        rows.append(
            {
                "model": "round1_ensemble",
                "top_k": top_k,
                "test_selection_exact": ensemble[key]["test"]["exact"],
                "test_purchase_exact": None,
            }
        )
    return rows


def write_markdown(payload: dict) -> None:
    lines = [
        "# KCYCLE prediction uplift Round 2",
        "",
        f"generated_at: {payload['generated_at']}",
        f"records: {payload['records']}",
        f"joined_entries: {payload['joined_entries']}",
        "selection: train/val only; test is reported once after val selection.",
        "metric note: selection_exact is one selected trio; purchase_exact is actual trio inside bought top_k set.",
        "",
        "## Gate",
        f"- current_axis reference test exact: {payload['gate']['current_axis_test_exact']:.4f}",
        f"- gen2_mut_436 top20 selection test exact: {payload['gate']['gen2_mut_436_top20_test_exact']:.4f}",
        "- purchase monotonic: pass",
        "",
        "## Baselines",
        "| model | top_k | test selection exact | test purchase exact |",
        "|---|---:|---:|---:|",
    ]
    for row in payload["baselines"]:
        purchase = "n/a" if row["test_purchase_exact"] is None else f"{row['test_purchase_exact']:.4f}"
        lines.append(f"| {row['model']} | {row['top_k']} | {row['test_selection_exact']:.4f} | {purchase} |")
    lines.extend(
        [
            "",
            "## Val/Test Lift vs Market Rank at top_k=20",
            "| experiment | selected | val purchase lift | test purchase lift | val selection lift | test selection lift |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for name in ("blend", "line_features", "conditional_logit"):
        row = payload[name]
        lift = row["lift_vs_market_top20"]
        selected = row.get("selected")
        lines.append(
            f"| {name} | {selected} | {lift['val_purchase_lift_pp']:+.3f}pp | "
            f"{lift['test_purchase_lift_pp']:+.3f}pp | {lift['val_selection_lift_pp']:+.3f}pp | "
            f"{lift['test_selection_lift_pp']:+.3f}pp |"
        )
    lines.extend(["", "## Line Feature Coefficients", "| feature | coefficient |", "|---|---:|"])
    for row in payload["line_features"]["top_coefficients"]:
        lines.append(f"| {row['feature']} | {row['coefficient']:+.6f} |")
    lines.extend(["", "## Conditional Logit Coefficients", "| feature | coefficient |", "|---|---:|"])
    for feature, value in list(payload["conditional_logit"]["coefficients"].items())[:10]:
        lines.append(f"| {feature} | {value:+.6f} |")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    records = load_snapshot_records()
    entries = load_entries()
    gate = assert_reproduction(records)
    arrays, market_scores = market_rank_scores(records, UNIVERSE_K)
    market_metrics = score_metrics(arrays, market_scores, TOP_KS)
    _, gen2_scores = weight_rank_scores(records, load_breakthrough_weights(), UNIVERSE_K)
    gen2_metrics = score_metrics(arrays, gen2_scores, TOP_KS)

    line_x, line_names, joined_entries = line_feature_tensor(records, arrays, entries)
    market_x = arrays["x"].astype(np.float64)
    market_fit = ridge_fit_scores(arrays, market_x, list(FEATURE_NAMES), "market_only_ridge")
    line_fit = ridge_fit_scores(arrays, np.concatenate([market_x, line_x], axis=2), [*FEATURE_NAMES, *line_names], "market_plus_line_ridge")

    blend_model_scores = first_winner_logistic_scores(records, arrays, entries)
    log_market_scores = market_prob_scores(records, arrays)
    blend = select_blend(records, arrays, log_market_scores, blend_model_scores)
    cond = fit_conditional_logit(records, arrays, entries)

    metrics_for_monotonic = {
        "market_rank": market_metrics,
        "gen2_mut_436": gen2_metrics,
        "market_only_ridge": market_fit.metrics,
        "market_plus_line_ridge": line_fit.metrics,
        "blend": blend["selected_metrics"],
        "conditional_logit": cond["metrics"],
    }
    assert_purchase_monotonic(metrics_for_monotonic)

    payload = {
        "generated_at": utc_now(),
        "records": len(records),
        "joined_entries": joined_entries,
        "universe_k": UNIVERSE_K,
        "gate": gate,
        "baselines": baseline_table(market_metrics, gen2_metrics),
        "market_rank_metrics": market_metrics,
        "gen2_mut_436_metrics": gen2_metrics,
        "blend": {
            "selected": f"beta={blend['selected_beta']:.2f}",
            "metrics": blend["selected_metrics"],
            "lift_vs_market_top20": lift_row(blend["selected_metrics"], market_metrics),
        },
        "line_features": {
            "selected": f"alpha={line_fit.alpha:g}",
            "market_only_selected": f"alpha={market_fit.alpha:g}",
            "metrics": line_fit.metrics,
            "market_only_metrics": market_fit.metrics,
            "lift_vs_market_only_top20": lift_row(line_fit.metrics, market_fit.metrics),
            "lift_vs_market_top20": lift_row(line_fit.metrics, market_metrics),
            "top_coefficients": top_coefficients(line_fit),
        },
        "conditional_logit": {
            "selected": f"alpha={cond['alpha']:g}",
            "joined_records": cond["joined_records"],
            "optimizer_success": cond["success"],
            "metrics": cond["metrics"],
            "lift_vs_market_top20": lift_row(cond["metrics"], market_metrics),
            "coefficients": cond["coefficients"],
        },
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(payload)
    append_progress(
        "round2 "
        f"blend_test_purchase_lift={payload['blend']['lift_vs_market_top20']['test_purchase_lift_pp']:+.3f}pp "
        f"line_vs_market_only_test_purchase_lift={payload['line_features']['lift_vs_market_only_top20']['test_purchase_lift_pp']:+.3f}pp "
        f"cond_logit_test_purchase_lift={payload['conditional_logit']['lift_vs_market_top20']['test_purchase_lift_pp']:+.3f}pp "
        f"out={OUT_JSON}"
    )
    print(json.dumps({"out_json": str(OUT_JSON), "out_md": str(OUT_MD)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
