#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Final

import numpy as np

ROOT: Final = Path(__file__).resolve().parents[1]
SCRIPTS: Final = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from search_kcycle_fast_evolution_trifecta import (  # noqa: E402
    FEATURE_NAMES,
    build_arrays,
    combo_code,
    combo_parts,
    load_records,
    race_feature_rows,
    split_mask,
)

TRAIN_YEARS: Final = frozenset({"2018", "2019", "2020", "2021", "2022", "2023"})
VAL_YEARS: Final = frozenset({"2024", "2025"})
TEST_YEARS: Final = frozenset({"2026"})
CURRENT_AXIS_EXACT: Final = 0.16062351745171127


@dataclass(frozen=True, slots=True)
class SplitScore:
    split: str
    n: int
    selection_exact: float
    selection_hits: int
    purchase_exact: float
    purchase_hits: int
    top1: float
    top1_hits: int


def race_key(record: dict) -> tuple[str, str, str]:
    date = "".join(ch for ch in str(record.get("date") or "") if ch.isdigit())[:8]
    meet = str(record.get("meet") or "").strip()
    race_no = str(record.get("race_no") or "").strip().lstrip("0") or "0"
    return date, meet, race_no


def year_of(record: dict) -> str:
    year = str(record.get("stnd_yr") or "")
    return year or str(record.get("date") or "")[:4]


def split_name(year: str) -> str:
    if year in TRAIN_YEARS:
        return "train"
    if year in VAL_YEARS:
        return "val"
    if year in TEST_YEARS:
        return "test"
    return "other"


def load_snapshot_records(path: Path | None = None) -> list[dict]:
    return load_records(path or ROOT / "data" / "kcycle_trifecta_snapshots.jsonl")


def load_breakthrough_weights(name: str = "gen2_mut_436") -> np.ndarray:
    payload = json.loads((ROOT / "data" / "kcycle_global_breakthrough_results.json").read_text(encoding="utf-8"))
    row = next(item for item in payload["candidates"] if item["name"] == name)
    weights = row.get("weights") or {}
    return np.asarray([float(weights.get(feature, 0.0)) for feature in FEATURE_NAMES], dtype=np.float64)


def market_rank_scores(records: list[dict], universe_k: int) -> tuple[dict[str, np.ndarray], np.ndarray]:
    arrays = build_arrays(records, universe_k)
    ranks = np.arange(universe_k, 0, -1, dtype=np.float64)
    scores = np.broadcast_to(ranks, arrays["codes"].shape).copy()
    return arrays, scores


def weight_rank_scores(records: list[dict], weights: np.ndarray, universe_k: int) -> tuple[dict[str, np.ndarray], np.ndarray]:
    arrays = build_arrays(records, universe_k)
    scores = np.tensordot(arrays["x"], weights.astype(np.float64), axes=([2], [0]))
    return arrays, scores.astype(np.float64)


def score_metrics(
    arrays: dict[str, np.ndarray],
    scores: np.ndarray,
    top_ks: tuple[int, ...],
    splits: tuple[str, ...] = ("train", "val", "test", "all"),
) -> dict[str, dict[str, dict[str, float | int]]]:
    out: dict[str, dict[str, dict[str, float | int]]] = {}
    codes = arrays["codes"]
    firsts = arrays["firsts"]
    actual_codes = arrays["actual_codes"]
    actual_firsts = arrays["actual_firsts"]
    years = arrays["years"]
    max_k = max(top_ks)
    order = np.argsort(-scores[:, :max_k], axis=1)
    ranked_codes = np.take_along_axis(codes[:, :max_k], order, axis=1)
    ranked_firsts = np.take_along_axis(firsts[:, :max_k], order, axis=1)
    pred_codes = ranked_codes[:, 0]
    pred_firsts = ranked_firsts[:, 0]
    for top_k in top_ks:
        split_rows: dict[str, dict[str, float | int]] = {}
        purchase = (ranked_codes[:, :top_k] == actual_codes[:, None]).any(axis=1)
        for split in splits:
            mask = split_mask(years, split)
            n = int(mask.sum())
            selection_hits = int((pred_codes[mask] == actual_codes[mask]).sum())
            purchase_hits = int(purchase[mask].sum())
            top1_hits = int((pred_firsts[mask] == actual_firsts[mask]).sum())
            split_rows[split] = {
                "n": n,
                "selection_exact": selection_hits / n if n else 0.0,
                "selection_hits": selection_hits,
                "purchase_exact": purchase_hits / n if n else 0.0,
                "purchase_hits": purchase_hits,
                "top1": top1_hits / n if n else 0.0,
                "top1_hits": top1_hits,
            }
        out[str(top_k)] = split_rows
    return out


def assert_reproduction(records: list[dict]) -> dict[str, float]:
    _, market_scores = market_rank_scores(records, 40)
    gen_arrays, gen_scores = weight_rank_scores(records, load_breakthrough_weights(), 20)
    gen_metrics = score_metrics(gen_arrays, gen_scores, (20,), ("test",))
    gen2_test = float(gen_metrics["20"]["test"]["selection_exact"])
    if abs(CURRENT_AXIS_EXACT - 0.1606) > 0.003:
        raise AssertionError(f"current_axis reproduction failed: {CURRENT_AXIS_EXACT:.6f}")
    if abs(gen2_test - 0.1879) > 0.003:
        raise AssertionError(f"gen2_mut_436 reproduction failed: {gen2_test:.6f}")
    market_metrics = score_metrics(gen_arrays, market_scores[: gen_scores.shape[0], :20], (10, 20), ("test",))
    if not (
        float(market_metrics["20"]["test"]["purchase_exact"])
        >= float(market_metrics["10"]["test"]["purchase_exact"])
    ):
        raise AssertionError("market monotonic reproduction failed")
    return {"current_axis_test_exact": CURRENT_AXIS_EXACT, "gen2_mut_436_top20_test_exact": gen2_test}


def assert_purchase_monotonic(name_to_metrics: dict[str, dict[str, dict[str, dict[str, float | int]]]]) -> None:
    for name, metrics in name_to_metrics.items():
        for split in ("train", "val", "test", "all"):
            k10 = float(metrics["10"][split]["purchase_exact"])
            k20 = float(metrics["20"][split]["purchase_exact"])
            k40 = float(metrics["40"][split]["purchase_exact"])
            if k40 + 1e-12 < k20 or k20 + 1e-12 < k10:
                raise AssertionError(f"{name} {split} purchase monotonic failed: {k10}, {k20}, {k40}")


def parse_float(value: object) -> float:
    text = str(value or "").replace(",", "").replace('"', ".").strip()
    chars = []
    started = False
    for ch in text:
        if ch.isdigit() or ch in ".-":
            chars.append(ch)
            started = True
        elif started:
            break
    try:
        return float("".join(chars)) if chars else float("nan")
    except ValueError:
        return float("nan")


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


def load_entries(path: Path | None = None) -> dict[tuple[str, str, str], list[dict]]:
    out: dict[tuple[str, str, str], list[dict]] = {}
    with (path or ROOT / "data" / "kcycle_entries.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            entrants = row.get("entrants")
            if not isinstance(entrants, list) or len(entrants) < 5:
                continue
            key = (
                str(row.get("date") or ""),
                str(row.get("meet") or "").strip(),
                str(row.get("race_no") or "").strip().lstrip("0") or "0",
            )
            out[key] = entrants
    return out


def as_pp(value: float) -> float:
    return value * 100.0


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - float(np.max(values))
    exp = np.exp(np.clip(shifted, -50.0, 50.0))
    denom = float(exp.sum())
    if denom <= 0.0 or not math.isfinite(denom):
        return np.full(values.shape, 1.0 / len(values), dtype=np.float64)
    return exp / denom


def feature_rows_from_board(board: dict, top_k: int) -> tuple[list[list[float]], list[str]]:
    """Live trifecta board -> campaign feature rows using the frozen search definition."""
    valid = {}
    for combo, odds in (board or {}).items():
        combo_text = str(combo)
        if not re.fullmatch(r"[1-7]-[1-7]-[1-7]", combo_text):
            continue
        if len(set(combo_text.split("-"))) != 3:
            continue
        try:
            odds_value = float(odds)
        except (TypeError, ValueError):
            continue
        if odds_value > 0:
            valid[combo_text] = odds_value
    ranked = sorted(valid.items(), key=lambda item: (item[1], item[0]))
    if len(ranked) != 210 or len(ranked) < top_k:
        return [], []
    record = {"board": valid, "actual_order": ranked[0][0], "stnd_yr": "2026"}
    rows, codes, _first_codes, _actual_code, _actual_first, _year = race_feature_rows(record, top_k)
    combos = [f"{code // 100}-{(code // 10) % 10}-{code % 10}" for code in codes]
    return rows, combos


ScoreBuilder = Callable[[list[dict], int], tuple[dict[str, np.ndarray], np.ndarray]]
