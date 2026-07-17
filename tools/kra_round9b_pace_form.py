#!/usr/bin/env python3
"""Round 9b: time-safe horse pace-form assay for KRA pre-odds ranking."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd  # noqa: PANDAS_OK - existing KRA scikit-learn contract

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kra_model_evaluation import as_dict, metrics, race_normalize  # noqa: E402
from kra_training_features import build_features, load_rows  # noqa: E402
from tools.kra_dual_phase_experiment import CONFIGS, DEFAULT_DB, _fit, _predict  # noqa: E402


TRAIN_FROM: Final = "20190101"
TRAIN_UNTIL: Final = "20250601"
VAL_FROM: Final = "20250601"
VAL_UNTIL: Final = "20260601"
FRESH_FROM: Final = "20260622"
FRESH_UNTIL: Final = "20260712"
REPORT_JSON: Final = ROOT / "data" / "kra_round9b_results.json"
REPORT_MD: Final = ROOT / "data" / "kra_round9b_results.md"
PROGRESS_MD: Final = ROOT / "runs" / "kra_corpus_progress.md"
RAW_COLUMNS: Final = (
    "bu_1fGTime",
    "bu_2fGTime",
    "bu_3fGTime",
    "je_1cTime",
    "seS1fAccTime",
    "buS1fTime",
    "jeS1fTime",
)
EXTRA_RAW_COLUMNS: Final = (
    "bu_1fGTime",
    "bu_2fGTime",
    "bu_3fGTime",
    "je_1cTime",
)
PACE_BASE_FEATURES: Final = (
    "pace_recent3_fast_z_mean",
    "pace_recent5_fast_z_mean",
    "pace_recent3_best_final_time",
    "pace_recent5_best_final_time",
    "pace_recent3_fast_z_trend",
    "pace_recent5_fast_z_trend",
    "pace_recent3_avg_ord",
    "pace_recent3_top3_rate",
    "pace_days_since_last",
    "pace_wg_budam_delta",
    "pace_rating_delta",
    "pace_distance_band_top3_rate",
    "pace_prior_start_count",
    "pace_is_first_start",
)
PACE_FEATURES: Final = (
    *PACE_BASE_FEATURES,
    *(f"{column}_rel" for column in PACE_BASE_FEATURES if column != "pace_is_first_start"),
)
BASELINE_CONFIG: Final = CONFIGS[0]
R9_EXPANDED_FRESH_TOP1: Final = 0.31343283582089554


@dataclass(frozen=True, slots=True)
class SplitFrame:
    """Named chronological split used for fixed train/val/fresh evaluation."""

    name: str
    frame: pd.DataFrame


def _load_round9b_rows(db_path: Path) -> pd.DataFrame:
    base = load_rows(db_path)
    raw_select = ", ".join((*("meet", "rcDate", "rcNo", "chulNo"), *EXTRA_RAW_COLUMNS))
    with sqlite3.connect(db_path) as connection:
        raw = pd.read_sql_query(f"SELECT {raw_select} FROM race_result", connection)
    for column in EXTRA_RAW_COLUMNS:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
    raw["chulNo"] = pd.to_numeric(raw["chulNo"], errors="coerce")
    return base.merge(raw, on=["meet", "rcDate", "rcNo", "chulNo"], how="left")


def _track_bucket(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.extract(r"^\s*([^(\s]+)", expand=False).fillna("NA")


def _valid_time(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").where(lambda value: value.between(8.0, 30.0))


def _finish_time(frame: pd.DataFrame) -> pd.Series:
    meet = frame["meet"].astype(str)
    finish = pd.Series(np.nan, index=frame.index, dtype=float)
    finish = finish.mask(meet.eq("서울"), _valid_time(frame["seS1fAccTime"]))
    finish = finish.mask(meet.eq("부경"), _valid_time(frame["bu_1fGTime"]))
    finish = finish.mask(meet.eq("제주"), _valid_time(frame["je_1cTime"]))
    finish = finish.fillna(_valid_time(frame["buS1fTime"]))
    finish = finish.fillna(_valid_time(frame["jeS1fTime"]))
    return finish


def _condition_fast_z(frame: pd.DataFrame, finish_time: pd.Series) -> pd.Series:
    condition = (
        frame["meet"].astype(str)
        + "|"
        + pd.to_numeric(frame["rcDist"], errors="coerce").round(-2).fillna(-1).astype(int).astype(str)
        + "|"
        + _track_bucket(frame["track"])
    )
    source = pd.DataFrame({
        "condition": condition,
        "rcDate": frame["rcDate"].astype(str),
        "finish_time": finish_time,
    }).dropna(subset=["finish_time"])
    daily = (
        source.assign(square=source["finish_time"] ** 2)
        .groupby(["condition", "rcDate"], sort=True, dropna=False)
        .agg(total=("finish_time", "sum"), square_total=("square", "sum"), count=("finish_time", "count"))
        .reset_index()
        .sort_values(["condition", "rcDate"])
    )
    grouped = daily.groupby("condition", dropna=False)
    prior_count = grouped["count"].cumsum() - daily["count"]
    prior_total = grouped["total"].cumsum() - daily["total"]
    prior_square = grouped["square_total"].cumsum() - daily["square_total"]
    prior_mean = prior_total / prior_count.replace(0, np.nan)
    variance = (prior_square / prior_count.replace(0, np.nan)) - (prior_mean**2)
    daily["prior_count"] = prior_count
    daily["prior_mean"] = prior_mean
    daily["prior_std"] = np.sqrt(variance.clip(lower=0.0))
    z_lookup = daily.set_index(["condition", "rcDate"])[["prior_count", "prior_mean", "prior_std"]]
    keys = list(zip(condition, frame["rcDate"].astype(str)))
    stats = z_lookup.reindex(pd.MultiIndex.from_tuples(keys, names=["condition", "rcDate"]))
    std = stats["prior_std"].to_numpy(dtype=float)
    count = stats["prior_count"].to_numpy(dtype=float)
    mean = stats["prior_mean"].to_numpy(dtype=float)
    z = (mean - finish_time.to_numpy(dtype=float)) / std
    z[(count < 20.0) | ~np.isfinite(z)] = np.nan
    return pd.Series(z, index=frame.index, dtype=float)


def _shifted_rolling(
    grouped: pd.core.groupby.SeriesGroupBy,
    window: int,
    method: str,
) -> pd.Series:
    def calculate(values: pd.Series) -> pd.Series:
        shifted = values.shift(1)
        rolling = shifted.rolling(window, min_periods=1)
        if method == "mean":
            return rolling.mean()
        if method == "min":
            return rolling.min()
        raise AssertionError(method)

    return grouped.transform(calculate)


def _shifted_trend(grouped: pd.core.groupby.SeriesGroupBy, window: int) -> pd.Series:
    return grouped.transform(lambda values: values.shift(1) - values.shift(window))


def _distance_band_top3_rate(frame: pd.DataFrame) -> pd.Series:
    values = pd.Series(np.nan, index=frame.index, dtype=float)
    ordered = frame.sort_values(["hrNo", "rcDate", "rcNo", "chulNo"], kind="stable")
    for _, group in ordered.groupby("hrNo", sort=False, dropna=False):
        distances: list[float] = []
        top3: list[float] = []
        for index, row in group.iterrows():
            current_distance = float(row["rcDist"]) if pd.notna(row["rcDist"]) else np.nan
            if np.isfinite(current_distance) and distances:
                distance_array = np.asarray(distances, dtype=float)
                top3_array = np.asarray(top3, dtype=float)
                mask = np.abs(distance_array - current_distance) <= 200.0
                if mask.any():
                    values.loc[index] = float(top3_array[mask].mean())
            distances.append(current_distance)
            top3.append(float(row["ord"] <= 3))
    return values


def add_pace_form_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.sort_values(["hrNo", "rcDate", "rcNo", "chulNo"], kind="stable").copy()
    result["_pace_finish_time"] = _finish_time(result)
    result["_pace_fast_z"] = _condition_fast_z(result, result["_pace_finish_time"])
    horse_group = result.groupby("hrNo", sort=False, dropna=False)
    result["pace_recent3_fast_z_mean"] = _shifted_rolling(horse_group["_pace_fast_z"], 3, "mean")
    result["pace_recent5_fast_z_mean"] = _shifted_rolling(horse_group["_pace_fast_z"], 5, "mean")
    result["pace_recent3_best_final_time"] = _shifted_rolling(horse_group["_pace_finish_time"], 3, "min")
    result["pace_recent5_best_final_time"] = _shifted_rolling(horse_group["_pace_finish_time"], 5, "min")
    result["pace_recent3_fast_z_trend"] = _shifted_trend(horse_group["_pace_fast_z"], 3)
    result["pace_recent5_fast_z_trend"] = _shifted_trend(horse_group["_pace_fast_z"], 5)
    result["pace_recent3_avg_ord"] = _shifted_rolling(horse_group["ord"], 3, "mean")
    result["pace_recent3_top3_rate"] = _shifted_rolling((result["ord"] <= 3).astype(float).groupby(result["hrNo"], sort=False), 3, "mean")
    race_date = pd.to_datetime(result["rcDate"], format="%Y%m%d")
    result["pace_days_since_last"] = (race_date - horse_group["rcDate"].transform(lambda values: pd.to_datetime(values, format="%Y%m%d").shift(1))).dt.days
    result["pace_wg_budam_delta"] = result["wgBudam"] - horse_group["wgBudam"].shift(1)
    result["pace_rating_delta"] = result["rating"] - horse_group["rating"].shift(1)
    result["pace_distance_band_top3_rate"] = _distance_band_top3_rate(result)
    result["pace_prior_start_count"] = horse_group.cumcount().astype(float)
    result["pace_is_first_start"] = (result["pace_prior_start_count"] == 0.0).astype(float)
    result = result.sort_index(kind="stable")
    for column in PACE_BASE_FEATURES:
        if column == "pace_is_first_start":
            continue
        result[f"{column}_rel"] = result[column] - result.groupby("rk")[column].transform("mean")
    return result.drop(columns=["_pace_finish_time", "_pace_fast_z"])


def _split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = frame[(frame["rcDate"] >= TRAIN_FROM) & (frame["rcDate"] < TRAIN_UNTIL)].copy()
    val = frame[(frame["rcDate"] >= VAL_FROM) & (frame["rcDate"] < VAL_UNTIL)].copy()
    fresh = frame[(frame["rcDate"] >= FRESH_FROM) & (frame["rcDate"] < FRESH_UNTIL)].copy()
    return train, val, fresh


def _evaluate(
    train: pd.DataFrame,
    splits: tuple[SplitFrame, ...],
    columns: list[str],
) -> dict[str, dict[str, float | int]]:
    model, median = _fit(train, columns, "win", BASELINE_CONFIG)
    rows: dict[str, dict[str, float | int]] = {}
    for split in splits:
        probability = race_normalize(split.frame, _predict(model, median, split.frame, columns))
        rows[split.name] = as_dict(metrics(split.frame, probability))
    return rows


def _leakage_assertion(
    base_frame: pd.DataFrame,
    full_featured: pd.DataFrame,
) -> list[dict[str, str]]:
    candidates = full_featured[
        (full_featured["rcDate"] >= VAL_FROM)
        & (full_featured["pace_prior_start_count"] >= 3)
        & full_featured["pace_recent3_fast_z_mean"].notna()
    ].drop_duplicates("hrNo").head(3)
    if len(candidates) != 3:
        raise AssertionError("not enough horses for leakage assertion")
    checks = []
    compare_columns = list(PACE_FEATURES)
    key_columns = ["meet", "rcDate", "rcNo", "chulNo", "hrNo"]
    for _, row in candidates.iterrows():
        cutoff = str(row["rcDate"])
        truncated = base_frame[base_frame["rcDate"] <= cutoff].copy()
        recomputed = add_pace_form_features(truncated)
        mask = pd.Series(True, index=recomputed.index)
        for column in key_columns:
            mask &= recomputed[column].astype(str).eq(str(row[column]))
        matched = recomputed.loc[mask]
        if len(matched) != 1:
            raise AssertionError(f"sample row not found after truncation: {row[key_columns].to_dict()}")
        left = full_featured.loc[row.name, compare_columns].astype(float).to_numpy()
        right = matched.iloc[0][compare_columns].astype(float).to_numpy()
        if not np.allclose(left, right, equal_nan=True):
            raise AssertionError(f"future-row deletion changed pace features: hrNo={row['hrNo']} rcDate={cutoff}")
        checks.append({
            "hrNo": str(row["hrNo"]),
            "rcDate": cutoff,
            "rk": str(row["rk"]),
            "status": "pass",
        })
    return checks


def _write_markdown(report: dict, path: Path) -> None:
    baseline_fresh = report["metrics"]["baseline_v4_pre_odds"]["fresh"]
    candidate_fresh = report["metrics"]["pace_form_candidate"]["fresh"]
    baseline_val = report["metrics"]["baseline_v4_pre_odds"]["val"]
    candidate_val = report["metrics"]["pace_form_candidate"]["val"]
    promotion = report["promotion"]
    lines = [
        "# KRA Round 9b Results",
        "",
        "- status: enabled=false, production_replace=false",
        "- leakage_assertion: pass (3 horses, future rows deleted and recomputed identical)",
        f"- train: {TRAIN_FROM}..20250531",
        f"- val: {VAL_FROM}..20260531",
        f"- fresh: {FRESH_FROM}..20260711",
        f"- fresh_races: {candidate_fresh['races']}",
        "",
        "## Metrics",
        f"- baseline_v4 val top1 {baseline_val['top1']:.2%}, top3 {baseline_val['top3']:.2%}, logloss {baseline_val['race_logloss']:.6f}",
        f"- pace_form val top1 {candidate_val['top1']:.2%}, top3 {candidate_val['top3']:.2%}, logloss {candidate_val['race_logloss']:.6f}",
        f"- baseline_v4 fresh top1 {baseline_fresh['top1']:.2%}, top3 {baseline_fresh['top3']:.2%}, logloss {baseline_fresh['race_logloss']:.6f}",
        f"- pace_form fresh top1 {candidate_fresh['top1']:.2%}, top3 {candidate_fresh['top3']:.2%}, logloss {candidate_fresh['race_logloss']:.6f}",
        f"- R9 expanded_2019_202505 fresh top1 {R9_EXPANDED_FRESH_TOP1:.2%}",
        "",
        "## Verdict",
        f"- fresh_top1_gate_33pct: {promotion['fresh_top1_gate_33pct']}",
        f"- val_noninferior: {promotion['val_noninferior']}",
        f"- qualifies_candidate: {promotion['qualifies_candidate']}",
        f"- conclusion: {promotion['conclusion']}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(db_path: Path) -> dict:
    base_source = _load_round9b_rows(db_path)
    base_frame, baseline_columns = build_features(base_source)
    featured = add_pace_form_features(base_frame)
    leakage_checks = _leakage_assertion(base_frame, featured)
    train, val, fresh = _split(featured)
    baseline_metrics = _evaluate(
        train,
        (SplitFrame("val", val), SplitFrame("fresh", fresh)),
        baseline_columns,
    )
    pace_columns = list(dict.fromkeys([*baseline_columns, *PACE_FEATURES]))
    candidate_metrics = _evaluate(
        train,
        (SplitFrame("val", val), SplitFrame("fresh", fresh)),
        pace_columns,
    )
    fresh_top1_gate = candidate_metrics["fresh"]["top1"] >= 0.33
    val_noninferior = (
        candidate_metrics["val"]["top1"] >= baseline_metrics["val"]["top1"]
        and candidate_metrics["val"]["top3"] >= baseline_metrics["val"]["top3"]
        and candidate_metrics["val"]["race_logloss"] <= baseline_metrics["val"]["race_logloss"]
    )
    qualifies = bool(fresh_top1_gate and val_noninferior)
    conclusion = (
        "candidate_report"
        if qualifies
        else "missed_gate; KRA fundamental pace-form lever exhausted under this protocol"
    )
    return {
        "round": "KRA Round 9b",
        "created_at": datetime.now().astimezone().isoformat(),
        "enabled": False,
        "production_replace": False,
        "odds_columns_excluded": True,
        "baseline_config": BASELINE_CONFIG[0],
        "windows": {
            "train": [TRAIN_FROM, "20250531"],
            "val": [VAL_FROM, "20260531"],
            "fresh": [FRESH_FROM, "20260711"],
        },
        "rows": {
            "train": int(train.shape[0]),
            "val": int(val.shape[0]),
            "fresh": int(fresh.shape[0]),
        },
        "races": {
            "train": int(train["rk"].nunique()),
            "val": int(val["rk"].nunique()),
            "fresh": int(fresh["rk"].nunique()),
        },
        "pace_raw_columns": list(RAW_COLUMNS),
        "pace_feature_columns": list(PACE_FEATURES),
        "leakage_gate": {
            "status": "pass",
            "rule": "each pace feature uses only same-horse rows with rcDate strictly less than the target rcDate",
            "future_deletion_assertions": leakage_checks,
        },
        "metrics": {
            "baseline_v4_pre_odds": baseline_metrics,
            "pace_form_candidate": candidate_metrics,
            "r9_reference": {
                "expanded_2019_202505_fresh_top1": R9_EXPANDED_FRESH_TOP1,
            },
        },
        "promotion": {
            "fresh_threshold": 0.33,
            "fresh_top1_gate_33pct": bool(fresh_top1_gate),
            "val_noninferior": bool(val_noninferior),
            "qualifies_candidate": qualifies,
            "conclusion": conclusion,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--json-report", type=Path, default=REPORT_JSON)
    parser.add_argument("--md-report", type=Path, default=REPORT_MD)
    args = parser.parse_args()
    report = run(args.db)
    args.json_report.parent.mkdir(parents=True, exist_ok=True)
    args.json_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_markdown(report, args.md_report)
    PROGRESS_MD.parent.mkdir(parents=True, exist_ok=True)
    with PROGRESS_MD.open("a", encoding="utf-8") as handle:
        handle.write(
            f"- {datetime.now().astimezone().isoformat()} round9b pace-form complete "
            f"fresh_top1={report['metrics']['pace_form_candidate']['fresh']['top1']:.4f} "
            f"val_noninferior={report['promotion']['val_noninferior']} "
            f"qualifies={report['promotion']['qualifies_candidate']}\n"
        )
    print(json.dumps({
        "json": str(args.json_report),
        "md": str(args.md_report),
        "qualifies": report["promotion"]["qualifies_candidate"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
