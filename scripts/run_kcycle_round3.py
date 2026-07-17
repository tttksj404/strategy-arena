#!/usr/bin/env python3
# allow: SIZE_OK - self-contained experimental harness with fixed audit output.
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Final

import numpy as np
from scipy.optimize import minimize

ROOT: Final = Path(__file__).resolve().parents[1]
SCRIPTS: Final = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from kcycle_eval_common import (  # noqa: E402
    CURRENT_AXIS_EXACT,
    FEATURE_NAMES,
    as_pp,
    assert_purchase_monotonic,
    assert_reproduction,
    combo_parts,
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

OUT_JSON: Final = ROOT / "data" / "kcycle_round3_results.json"
OUT_MD: Final = ROOT / "data" / "kcycle_round3_results.md"
PROGRESS: Final = ROOT / "runs" / "prediction_uplift_progress.md"
TOP_KS: Final = (10, 20, 40)
UNIVERSE_K: Final = 40
RIDGE_ALPHAS: Final = (0.01, 0.1, 1.0, 10.0, 50.0)
PL_ALPHAS: Final = (0.01, 0.1, 1.0, 10.0)
FORM_FIELDS: Final = (
    "top3_rate_last5",
    "top3_rate_last10",
    "days_since_last",
    "gear_delta",
    "rec200_delta",
    "grade_change",
    "streak",
    "meet_top3_rate",
)


@dataclass(slots=True)
class RacerState:
    dates: list[date] = field(default_factory=list)
    top3: list[int] = field(default_factory=list)
    gears: list[float] = field(default_factory=list)
    rec200: list[float] = field(default_factory=list)
    meet_top3: dict[str, list[int]] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RidgeFit:
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
        handle.write(f"- {utc_now()} Round3: {line}\n")


def race_date(text: str) -> date:
    clean = "".join(ch for ch in text if ch.isdigit())[:8]
    return date(int(clean[:4]), int(clean[4:6]), int(clean[6:8]))


def entrant_index(entrants: list[dict]) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for row in entrants:
        bno = int(str(row.get("back_no") or "0").strip() or "0")
        if bno:
            out[bno] = row
    return out


def first_masses(record: dict) -> dict[int, float]:
    inv = {combo: 1.0 / float(odds) for combo, odds in (record.get("board") or {}).items() if float(odds) > 0.0}
    total = sum(inv.values())
    masses = {bno: 0.0 for bno in range(1, 8)}
    if total <= 0.0:
        return masses
    for combo, value in inv.items():
        masses[combo_parts(combo)[0]] += value / total
    return masses


def grade_value(value: object) -> float:
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


def form_from_state(state: RacerState, meet: str, current: dict, current_date: date) -> dict[str, float]:
    history = len(state.top3)
    last5 = state.top3[-5:]
    last10 = state.top3[-10:]
    meet_history = state.meet_top3.get(meet, [])
    previous_date = state.dates[-1] if state.dates else None
    previous_gear = state.gears[-1] if state.gears and math.isfinite(state.gears[-1]) else float("nan")
    previous_rec200 = state.rec200[-1] if state.rec200 and math.isfinite(state.rec200[-1]) else float("nan")
    current_gear = parse_float(current.get("gear_rate"))
    current_rec200 = parse_float(current.get("rec_200m_scr"))
    streak = 0
    if state.top3:
        last = state.top3[-1]
        run = 0
        for value in reversed(state.top3):
            if value != last:
                break
            run += 1
        streak = run if last else -run
    return {
        "top3_rate_last5": sum(last5) / len(last5) if last5 else 0.0,
        "top3_rate_last10": sum(last10) / len(last10) if last10 else 0.0,
        "days_since_last": float(min((current_date - previous_date).days, 365)) if previous_date else 365.0,
        "gear_delta": current_gear - previous_gear if math.isfinite(current_gear) and math.isfinite(previous_gear) else 0.0,
        "rec200_delta": current_rec200 - previous_rec200 if math.isfinite(current_rec200) and math.isfinite(previous_rec200) else 0.0,
        "grade_change": grade_value(current.get("racer_grd_cur_cd")) - grade_value(current.get("racer_grd_bef_cd")),
        "streak": float(streak if history else 0),
        "meet_top3_rate": sum(meet_history) / len(meet_history) if meet_history else 0.0,
    }


def rolling_forms(records: list[dict], entries: dict[tuple[str, str, str], list[dict]]) -> dict[tuple[str, str, str], dict[int, dict[str, float]]]:
    states: dict[str, RacerState] = {}
    forms: dict[tuple[str, str, str], dict[int, dict[str, float]]] = {}
    ordered = sorted(records, key=lambda row: (str(row.get("date") or ""), str(row.get("meet") or ""), int(str(row.get("race_no") or "0").lstrip("0") or "0")))
    idx = 0
    while idx < len(ordered):
        current_day = str(ordered[idx].get("date") or "")
        day_records = []
        while idx < len(ordered) and str(ordered[idx].get("date") or "") == current_day:
            day_records.append(ordered[idx])
            idx += 1
        updates = []
        for record in day_records:
            key = race_key(record)
            entrants = entries.get(key)
            if not entrants:
                continue
            meet = key[1]
            rdate = race_date(key[0])
            by_bno = entrant_index(entrants)
            actual_top3 = set(combo_parts(str(record.get("actual_order") or "")))
            race_forms: dict[int, dict[str, float]] = {}
            for bno, entrant in by_bno.items():
                name = str(entrant.get("racer_nm") or "").strip()
                state = states.setdefault(name, RacerState())
                race_forms[bno] = form_from_state(state, meet, entrant, rdate)
                updates.append((name, meet, rdate, parse_float(entrant.get("gear_rate")), parse_float(entrant.get("rec_200m_scr")), 1 if bno in actual_top3 else 0))
            forms[key] = race_forms
        for name, meet, rdate, gear, rec200, top3 in updates:
            state = states.setdefault(name, RacerState())
            state.dates.append(rdate)
            state.top3.append(top3)
            state.gears.append(gear)
            state.rec200.append(rec200)
            state.meet_top3.setdefault(meet, []).append(top3)
    return forms


def assert_no_form_leakage(records: list[dict], entries: dict[tuple[str, str, str], list[dict]], forms: dict[tuple[str, str, str], dict[int, dict[str, float]]]) -> list[dict[str, str]]:
    samples = []
    for record in sorted(records, key=lambda row: (str(row.get("date") or ""), str(row.get("meet") or ""), str(row.get("race_no") or ""))):
        race_forms = forms.get(race_key(record), {})
        entrants = entries.get(race_key(record)) or []
        for bno, entrant in entrant_index(entrants).items():
            name = str(entrant.get("racer_nm") or "").strip()
            if len(samples) < 3 and race_forms.get(bno, {}).get("days_since_last", 365.0) < 365.0:
                samples.append({"date": str(record.get("date")), "meet": str(record.get("meet")), "race_no": str(record.get("race_no")), "racer_nm": name, "back_no": str(bno)})
        if len(samples) >= 3:
            break
    for sample in samples:
        cutoff = str(sample["date"])
        truncated = [row for row in records if str(row.get("date") or "") <= cutoff]
        recalculated = rolling_forms(truncated, entries)
        key = (sample["date"], sample["meet"], sample["race_no"].lstrip("0") or "0")
        bno = int(sample["back_no"])
        for field_name in FORM_FIELDS:
            before = forms[key][bno][field_name]
            after = recalculated[key][bno][field_name]
            if abs(before - after) > 1e-12:
                raise AssertionError(f"rolling leakage assertion failed: {sample} {field_name} {before} != {after}")
    return samples


def form_vector(form: dict[str, float]) -> list[float]:
    return [float(form[name]) for name in FORM_FIELDS]


def standardize_train(values: np.ndarray, years: np.ndarray) -> np.ndarray:
    train = split_mask(years, "train")
    flat = values[train].reshape(-1, values.shape[-1])
    mu = flat.mean(axis=0)
    sigma = flat.std(axis=0)
    sigma = np.where(sigma < 1e-9, 1.0, sigma)
    return ((values - mu) / sigma).astype(np.float64)


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


def runner_rows(records: list[dict], entries: dict[tuple[str, str, str], list[dict]], forms: dict[tuple[str, str, str], dict[int, dict[str, float]]]) -> list[tuple[int, dict, list[int], np.ndarray]]:
    raw_rows = []
    for idx, record in enumerate(records):
        entrants = entries.get(race_key(record))
        race_forms = forms.get(race_key(record))
        if not entrants or not race_forms:
            continue
        bnos = sorted(entrant_index(entrants))
        masses = first_masses(record)
        matrix = [[math.log(max(masses.get(bno, 0.0), 1e-12)), *form_vector(race_forms[bno])] for bno in bnos]
        raw_rows.append((idx, record, bnos, np.asarray(matrix, dtype=np.float64)))
    if not raw_rows:
        return []
    train_blocks = [
        matrix[:, 1:]
        for _, record, _, matrix in raw_rows
        if split_mask(np.asarray([str(record.get("stnd_yr") or str(record.get("date") or "")[:4])]), "train")[0]
    ]
    train_flat = np.concatenate(train_blocks, axis=0)
    mu = train_flat.mean(axis=0)
    sigma = train_flat.std(axis=0)
    sigma = np.where(sigma < 1e-9, 1.0, sigma)
    return [(idx, record, bnos, np.column_stack([matrix[:, 0], (matrix[:, 1:] - mu) / sigma])) for idx, record, bnos, matrix in raw_rows]


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
            probs = softmax(utilities[remaining])
            local = remaining.index(picked_idx)
            local_values = utilities[remaining]
            max_value = float(np.max(local_values))
            loss -= float(utilities[picked_idx] - math.log(np.exp(np.clip(local_values - max_value, -50.0, 50.0)).sum()) - max_value)
            grad += probs @ features[remaining] - features[picked_idx]
            remaining.pop(local)
    return loss, grad


def pl_scores(records: list[dict], arrays: dict[str, np.ndarray], rows: list[tuple[int, dict, list[int], np.ndarray]], beta: np.ndarray) -> np.ndarray:
    scores = market_prob_scores(records, arrays)
    for row_idx, _, bnos, features in rows:
        raw = features @ beta
        exp_u = {bno: math.exp(min(max(float(value), -30.0), 30.0)) for bno, value in zip(bnos, raw)}
        total_exp = sum(exp_u.values())
        for col_idx, code in enumerate(arrays["codes"][row_idx]):
            a, b, c = combo_parts(f"{int(code) // 100}-{(int(code) // 10) % 10}-{int(code) % 10}")
            if a in exp_u and b in exp_u and c in exp_u:
                scores[row_idx, col_idx] = (
                    math.log(exp_u[a] / max(total_exp, 1e-15))
                    + math.log(exp_u[b] / max(total_exp - exp_u[a], 1e-15))
                    + math.log(exp_u[c] / max(total_exp - exp_u[a] - exp_u[b], 1e-15))
                )
    return scores


def fit_runner_form(records: list[dict], arrays: dict[str, np.ndarray], rows: list[tuple[int, dict, list[int], np.ndarray]]) -> dict:
    train_rows = [row for row in rows if split_mask(np.asarray([str(row[1].get("stnd_yr") or str(row[1].get("date") or "")[:4])]), "train")[0]]
    best = None
    for alpha in PL_ALPHAS:
        result = minimize(lambda vec: pl_loss(vec, train_rows, alpha), np.zeros(1 + len(FORM_FIELDS), dtype=np.float64), jac=True, method="L-BFGS-B", options={"maxiter": 80, "ftol": 1e-7})
        scores = pl_scores(records, arrays, rows, np.asarray(result.x, dtype=np.float64))
        metrics = score_metrics(arrays, scores, TOP_KS, ("train", "val"))
        row = {"alpha": alpha, "success": bool(result.success), "beta": np.asarray(result.x, dtype=np.float64), "scores": scores, "metrics": metrics}
        if best is None or (float(metrics["20"]["val"]["purchase_exact"]), float(metrics["20"]["val"]["selection_exact"]), -alpha) > (float(best["metrics"]["20"]["val"]["purchase_exact"]), float(best["metrics"]["20"]["val"]["selection_exact"]), -float(best["alpha"])):
            best = row
    if best is None:
        raise RuntimeError("runner form fit produced no candidate")
    names = ["log_first_mass", *FORM_FIELDS]
    return {
        "alpha": best["alpha"],
        "success": best["success"],
        "metrics": score_metrics(arrays, best["scores"], TOP_KS),
        "coefficients": {name: float(value) for name, value in sorted(zip(names, best["beta"]), key=lambda item: abs(float(item[1])), reverse=True)},
        "scores": best["scores"],
    }


def trio_form_features(records: list[dict], arrays: dict[str, np.ndarray], forms: dict[tuple[str, str, str], dict[int, dict[str, float]]]) -> tuple[np.ndarray, list[str], int]:
    names = []
    for field_name in FORM_FIELDS:
        names.extend([f"{field_name}_sum", f"{field_name}_min"])
    names.append("gear_up_count")
    rows = np.zeros((*arrays["codes"].shape, len(names)), dtype=np.float64)
    joined = 0
    for race_idx, record in enumerate(records):
        race_forms = forms.get(race_key(record))
        if not race_forms:
            continue
        joined += 1
        for combo_idx, code in enumerate(arrays["codes"][race_idx]):
            trio = combo_parts(f"{int(code) // 100}-{(int(code) // 10) % 10}-{int(code) % 10}")
            values = []
            for field_name in FORM_FIELDS:
                trio_values = [race_forms.get(bno, {}).get(field_name, 0.0) for bno in trio]
                values.extend([sum(trio_values), min(trio_values)])
            values.append(sum(1 for bno in trio if race_forms.get(bno, {}).get("gear_delta", 0.0) > 1e-9))
            rows[race_idx, combo_idx] = np.asarray(values, dtype=np.float64)
    return standardize_train(rows, arrays["years"]), names, joined


def ridge_fit_scores(arrays: dict[str, np.ndarray], features: np.ndarray, names: list[str]) -> RidgeFit:
    train = split_mask(arrays["years"], "train")
    y = (arrays["codes"] == arrays["actual_codes"][:, None]).astype(np.float64)
    x_train = features[train].reshape(-1, features.shape[-1])
    y_train = y[train].reshape(-1)
    weights = np.where(y_train > 0.5, 39.0, 1.0)
    x_aug = np.column_stack([np.ones(x_train.shape[0]), x_train])
    xtw = x_aug.T * weights
    flat_all = features.reshape(-1, features.shape[-1])
    best = None
    for alpha in RIDGE_ALPHAS:
        reg = np.eye(x_aug.shape[1], dtype=np.float64) * alpha
        reg[0, 0] = 0.0
        coef = np.linalg.solve(xtw @ x_aug + reg, xtw @ y_train)
        scores = (np.column_stack([np.ones(flat_all.shape[0]), flat_all]) @ coef).reshape(features.shape[:2])
        metrics = score_metrics(arrays, scores, TOP_KS, ("train", "val"))
        row = RidgeFit(alpha, ["intercept", *names], coef, scores, metrics)
        if best is None or (float(metrics["20"]["val"]["purchase_exact"]), float(metrics["20"]["val"]["selection_exact"]), -alpha) > (float(best.metrics["20"]["val"]["purchase_exact"]), float(best.metrics["20"]["val"]["selection_exact"]), -best.alpha):
            best = row
    if best is None:
        raise RuntimeError("ridge fit produced no candidate")
    return RidgeFit(best.alpha, best.feature_names, best.coefficients, best.scores, score_metrics(arrays, best.scores, TOP_KS))


def top_coefficients(names: list[str], coefs: np.ndarray, form_only: bool) -> list[dict[str, float | str]]:
    rows = []
    for name, value in zip(names, coefs):
        if name == "intercept":
            continue
        if form_only and not any(field_name in name for field_name in FORM_FIELDS) and name != "gear_up_count":
            continue
        rows.append({"feature": name, "coefficient": float(value)})
    return sorted(rows, key=lambda row: abs(float(row["coefficient"])), reverse=True)[:10]


def lift_row(metrics: dict[str, dict[str, dict[str, float | int]]], baseline: dict[str, dict[str, dict[str, float | int]]]) -> dict[str, float]:
    return {
        "val_purchase_lift_pp": as_pp(float(metrics["20"]["val"]["purchase_exact"]) - float(baseline["20"]["val"]["purchase_exact"])),
        "test_purchase_lift_pp": as_pp(float(metrics["20"]["test"]["purchase_exact"]) - float(baseline["20"]["test"]["purchase_exact"])),
        "val_selection_lift_pp": as_pp(float(metrics["20"]["val"]["selection_exact"]) - float(baseline["20"]["val"]["selection_exact"])),
        "test_selection_lift_pp": as_pp(float(metrics["20"]["test"]["selection_exact"]) - float(baseline["20"]["test"]["selection_exact"])),
    }


def subgroup_metrics(arrays: dict[str, np.ndarray], scores: np.ndarray, forms: dict[tuple[str, str, str], dict[int, dict[str, float]]], records: list[dict], market_metrics: dict[str, dict[str, dict[str, float | int]]]) -> dict[str, dict[str, float | int]]:
    order = np.argsort(-scores[:, :20], axis=1)
    ranked = np.take_along_axis(arrays["codes"][:, :20], order, axis=1)
    out = {}
    for label, predicate in {
        "days_since_last_gt30": lambda race_forms: any(row["days_since_last"] > 30.0 and row["days_since_last"] < 365.0 for row in race_forms),
        "gear_up_included": lambda race_forms: any(row["gear_delta"] > 1e-9 for row in race_forms),
    }.items():
        mask = []
        for record in records:
            race_forms = forms.get(race_key(record), {})
            mask.append(bool(predicate(list(race_forms.values()))))
        arr_mask = np.asarray(mask, dtype=bool) & split_mask(arrays["years"], "test")
        n = int(arr_mask.sum())
        hits = int((ranked[:, :20][arr_mask] == arrays["actual_codes"][arr_mask, None]).any(axis=1).sum())
        market_ranked = arrays["codes"][:, :20]
        market_hits = int((market_ranked[arr_mask] == arrays["actual_codes"][arr_mask, None]).any(axis=1).sum())
        out[label] = {
            "test_n": n,
            "track_b_purchase_exact": hits / n if n else 0.0,
            "market_purchase_exact": market_hits / n if n else 0.0,
            "lift_pp": as_pp((hits / n if n else 0.0) - (market_hits / n if n else 0.0)),
            "overall_track_b_lift_pp": lift_row(score_metrics(arrays, scores, TOP_KS), market_metrics)["test_purchase_lift_pp"],
        }
    return out


def baseline_table(market_metrics: dict[str, dict[str, dict[str, float | int]]], gen2_metrics: dict[str, dict[str, dict[str, float | int]]]) -> list[dict]:
    ensemble_payload = json.loads((ROOT / "data" / "kcycle_ensemble_gating_results.json").read_text(encoding="utf-8"))
    ensemble = {str(row["top_k"]): row["splits"] for row in ensemble_payload["ensemble"]}
    rows = []
    for top_k in TOP_KS:
        key = str(top_k)
        rows.append({"model": "current_axis_reference", "top_k": top_k, "test_selection_exact": CURRENT_AXIS_EXACT, "test_purchase_exact": None})
        rows.append({"model": "market_rank", "top_k": top_k, "test_selection_exact": market_metrics[key]["test"]["selection_exact"], "test_purchase_exact": market_metrics[key]["test"]["purchase_exact"]})
        rows.append({"model": "gen2_mut_436", "top_k": top_k, "test_selection_exact": gen2_metrics[key]["test"]["selection_exact"], "test_purchase_exact": gen2_metrics[key]["test"]["purchase_exact"]})
        rows.append({"model": "round1_ensemble", "top_k": top_k, "test_selection_exact": ensemble[key]["test"]["exact"], "test_purchase_exact": None})
    return rows


def ensemble_bonus(records: list[dict], track_scores: np.ndarray) -> dict[str, float | int]:
    from search_kcycle_fast_evolution_trifecta import build_arrays

    payload = json.loads((ROOT / "data" / "kcycle_global_breakthrough_results.json").read_text(encoding="utf-8"))
    selected = [row for row in payload["candidates"] if row.get("deployable")]
    selected.sort(key=lambda row: float(row.get("val_exact") or 0.0), reverse=True)
    selected = selected[:20]
    arrays = build_arrays(records, 20)
    weights = np.vstack([np.asarray([float((row.get("weights") or {}).get(name, 0.0)) for name in FEATURE_NAMES], dtype=np.float64) for row in selected])
    candidate_scores = np.tensordot(arrays["x"], weights.T, axes=([2], [0]))
    all_scores = np.concatenate([candidate_scores, track_scores[:, :20, None]], axis=2)
    rank_order = np.argsort(-all_scores, axis=1)
    ranks = np.empty_like(rank_order, dtype=np.float64)
    race_idx = np.arange(rank_order.shape[0])[:, None, None]
    model_idx = np.arange(rank_order.shape[2])[None, None, :]
    ranks[race_idx, rank_order, model_idx] = np.arange(20, dtype=np.float64)[None, :, None]
    pred_idx = ranks.mean(axis=2).argmin(axis=1)
    pred = arrays["codes"][np.arange(len(pred_idx)), pred_idx]
    out = {}
    base = json.loads((ROOT / "data" / "kcycle_ensemble_gating_results.json").read_text(encoding="utf-8"))
    base20 = next(row for row in base["ensemble"] if int(row["top_k"]) == 20)
    for split in ("val", "test"):
        mask = split_mask(arrays["years"], split)
        exact = float((pred[mask] == arrays["actual_codes"][mask]).sum() / int(mask.sum()))
        out[f"{split}_selection_exact"] = exact
        out[f"{split}_delta_pp"] = as_pp(exact - float(base20["splits"][split]["exact"]))
    out["top_k"] = 20
    return out


def write_markdown(payload: dict) -> None:
    lines = [
        "# KCYCLE prediction uplift Round 3",
        "",
        f"generated_at: {payload['generated_at']}",
        f"records: {payload['records']}",
        f"joined_entries: {payload['joined_entries']}",
        "selection: train/val only; test is reported once after val selection.",
        "leakage gate: passed by deleting future races for sampled racers and matching features.",
        "",
        "## Baselines",
        "| model | top_k | test selection exact | test purchase exact |",
        "|---|---:|---:|---:|",
    ]
    for row in payload["baselines"]:
        purchase = "n/a" if row["test_purchase_exact"] is None else f"{row['test_purchase_exact']:.4f}"
        lines.append(f"| {row['model']} | {row['top_k']} | {row['test_selection_exact']:.4f} | {purchase} |")
    lines.extend(["", "## Val/Test Lift vs Market Rank at top_k=20", "| experiment | selected | val purchase lift | test purchase lift | val selection lift | test selection lift |", "|---|---|---:|---:|---:|---:|"])
    for name in ("runner_form_conditional_logit", "trio_form_ridge"):
        row = payload[name]
        lift = row["lift_vs_market_top20"]
        lines.append(f"| {name} | {row['selected']} | {lift['val_purchase_lift_pp']:+.3f}pp | {lift['test_purchase_lift_pp']:+.3f}pp | {lift['val_selection_lift_pp']:+.3f}pp | {lift['test_selection_lift_pp']:+.3f}pp |")
    lines.extend(["", "## Form Coefficients", "| track | feature | coefficient |", "|---|---|---:|"])
    for feature, value in list(payload["runner_form_conditional_logit"]["coefficients"].items())[:10]:
        lines.append(f"| runner | {feature} | {value:+.6f} |")
    for row in payload["trio_form_ridge"]["top_form_coefficients"]:
        lines.append(f"| trio | {row['feature']} | {row['coefficient']:+.6f} |")
    lines.extend(["", "## Subgroups", "| subgroup | test n | track_b purchase | market purchase | lift |", "|---|---:|---:|---:|---:|"])
    for name, row in payload["subgroups"].items():
        lines.append(f"| {name} | {row['test_n']} | {row['track_b_purchase_exact']:.4f} | {row['market_purchase_exact']:.4f} | {row['lift_pp']:+.3f}pp |")
    bonus = payload["ensemble_bonus"]
    lines.extend(["", "## Ensemble Bonus", f"- top_k=20 add_trio_form_rank_member val_delta={bonus['val_delta_pp']:+.3f}pp test_delta={bonus['test_delta_pp']:+.3f}pp"])
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    records = load_snapshot_records()
    entries = load_entries()
    gate = assert_reproduction(records)
    arrays, market_scores = market_rank_scores(records, UNIVERSE_K)
    market_metrics = score_metrics(arrays, market_scores, TOP_KS)
    _, gen2_scores = weight_rank_scores(records, load_breakthrough_weights(), UNIVERSE_K)
    gen2_metrics = score_metrics(arrays, gen2_scores, TOP_KS)
    forms = rolling_forms(records, entries)
    leakage_samples = assert_no_form_leakage(records, entries, forms)
    rows = runner_rows(records, entries, forms)
    runner_fit = fit_runner_form(records, arrays, rows)
    trio_x, trio_names, joined_entries = trio_form_features(records, arrays, forms)
    market_x = arrays["x"].astype(np.float64)
    trio_fit = ridge_fit_scores(arrays, np.concatenate([market_x, trio_x], axis=2), [*FEATURE_NAMES, *trio_names])
    metrics_for_monotonic = {
        "market_rank": market_metrics,
        "gen2_mut_436": gen2_metrics,
        "runner_form_conditional_logit": runner_fit["metrics"],
        "trio_form_ridge": trio_fit.metrics,
    }
    assert_purchase_monotonic(metrics_for_monotonic)
    payload = {
        "generated_at": utc_now(),
        "records": len(records),
        "joined_entries": joined_entries,
        "universe_k": UNIVERSE_K,
        "gate": {**gate, "leakage_samples": leakage_samples},
        "baselines": baseline_table(market_metrics, gen2_metrics),
        "market_rank_metrics": market_metrics,
        "gen2_mut_436_metrics": gen2_metrics,
        "runner_form_conditional_logit": {
            "selected": f"alpha={float(runner_fit['alpha']):g}",
            "joined_records": len(rows),
            "optimizer_success": runner_fit["success"],
            "metrics": runner_fit["metrics"],
            "lift_vs_market_top20": lift_row(runner_fit["metrics"], market_metrics),
            "coefficients": runner_fit["coefficients"],
        },
        "trio_form_ridge": {
            "selected": f"alpha={trio_fit.alpha:g}",
            "metrics": trio_fit.metrics,
            "lift_vs_market_top20": lift_row(trio_fit.metrics, market_metrics),
            "top_form_coefficients": top_coefficients(trio_fit.feature_names, trio_fit.coefficients, form_only=True),
        },
        "subgroups": subgroup_metrics(arrays, trio_fit.scores, forms, records, market_metrics),
        "ensemble_bonus": ensemble_bonus(records, trio_fit.scores),
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(payload)
    append_progress(
        f"runner_test_purchase_lift={payload['runner_form_conditional_logit']['lift_vs_market_top20']['test_purchase_lift_pp']:+.3f}pp "
        f"trio_test_purchase_lift={payload['trio_form_ridge']['lift_vs_market_top20']['test_purchase_lift_pp']:+.3f}pp "
        f"bonus_test_delta={payload['ensemble_bonus']['test_delta_pp']:+.3f}pp out={OUT_JSON}"
    )
    print(json.dumps({"out_json": str(OUT_JSON), "out_md": str(OUT_MD)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
