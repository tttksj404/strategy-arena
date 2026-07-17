#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from search_kcycle_trifecta_rules import METHODS, frame_from_records, load_records  # noqa: E402


TRAIN_YEARS = {"2018", "2019", "2020", "2021", "2022", "2023"}
VAL_YEARS = {"2024", "2025"}
TEST_YEARS = {"2026"}
GRADE_ORDER = ("선발", "우수", "특선")


@dataclass(frozen=True, slots=True)
class GradeMethodMetric:
    grade: str
    method: str
    split: str
    races: int
    exact: float
    top1: float


@dataclass(frozen=True, slots=True)
class RoutedMetric:
    name: str
    split: str
    races: int
    exact: float
    top1: float
    board_exact: float
    board_top1: float
    exact_lift_pp: float
    top1_lift_pp: float
    route: dict[str, str]


def grade_proxy_from_race_no(race_no: object) -> str:
    try:
        number = int(str(race_no).strip().lstrip("0") or "0")
    except ValueError:
        number = 0
    if number <= 5:
        return "선발"
    if number <= 10:
        return "우수"
    return "특선"


def split_mask(df: pd.DataFrame, split: str) -> pd.Series:
    years = df["year"].astype(str)
    match split:
        case "train":
            return years.isin(TRAIN_YEARS)
        case "val":
            return years.isin(VAL_YEARS)
        case "test":
            return years.isin(TEST_YEARS)
        case "all":
            return years.isin(TRAIN_YEARS | VAL_YEARS | TEST_YEARS)
        case _:
            raise ValueError(f"unknown split: {split}")


def combo_top1(combo: object) -> str:
    return str(combo or "").split("-", maxsplit=1)[0]


def attach_grade_proxy(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["race_no_int"] = [int(str(value).strip().lstrip("0") or "0") for value in out["race_no"]]
    out["grade_proxy"] = [grade_proxy_from_race_no(value) for value in out["race_no_int"]]
    return out


def method_metric(df: pd.DataFrame, method: str, split: str, grade: str | None = None) -> GradeMethodMetric:
    mask = split_mask(df, split)
    if grade is not None:
        mask = mask & (df["grade_proxy"] == grade)
    sub = df.loc[mask]
    if sub.empty:
        return GradeMethodMetric(grade or "all", method, split, 0, 0.0, 0.0)
    pred = sub[f"pred_{method}"].astype(str)
    actual = sub["actual_order"].astype(str)
    exact = float((pred == actual).mean())
    top1 = float((pred.map(combo_top1) == actual.map(combo_top1)).mean())
    return GradeMethodMetric(grade or "all", method, split, int(len(sub)), exact, top1)


def best_method_by_grade(df: pd.DataFrame, split: str, objective: str = "exact") -> dict[str, str]:
    route: dict[str, str] = {}
    for grade in GRADE_ORDER:
        metrics = [method_metric(df, method, split, grade) for method in METHODS]
        if objective == "top1":
            metrics.sort(key=lambda item: (item.top1, item.exact, -METHODS.index(item.method)), reverse=True)
        else:
            metrics.sort(key=lambda item: (item.exact, item.top1, -METHODS.index(item.method)), reverse=True)
        route[grade] = metrics[0].method
    return route


def routed_predictions(df: pd.DataFrame, route: dict[str, str]) -> pd.Series:
    values: list[str] = []
    for _, row in df.iterrows():
        method = route.get(str(row["grade_proxy"]), "board_min")
        values.append(str(row[f"pred_{method}"]))
    return pd.Series(values, index=df.index)


def routed_metric(df: pd.DataFrame, name: str, route: dict[str, str], split: str) -> RoutedMetric:
    mask = split_mask(df, split)
    sub = df.loc[mask]
    if sub.empty:
        return RoutedMetric(name, split, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, dict(route))
    pred = routed_predictions(sub, route)
    board = sub["pred_board_min"].astype(str)
    actual = sub["actual_order"].astype(str)
    exact = float((pred == actual).mean())
    top1 = float((pred.map(combo_top1) == actual.map(combo_top1)).mean())
    board_exact = float((board == actual).mean())
    board_top1 = float((board.map(combo_top1) == actual.map(combo_top1)).mean())
    return RoutedMetric(
        name=name,
        split=split,
        races=int(len(sub)),
        exact=exact,
        top1=top1,
        board_exact=board_exact,
        board_top1=board_top1,
        exact_lift_pp=(exact - board_exact) * 100.0,
        top1_lift_pp=(top1 - board_top1) * 100.0,
        route=dict(route),
    )


def grade_metrics(df: pd.DataFrame) -> list[GradeMethodMetric]:
    rows: list[GradeMethodMetric] = []
    for split in ("train", "val", "test"):
        for grade in GRADE_ORDER:
            for method in METHODS:
                rows.append(method_metric(df, method, split, grade))
    return rows


def top_grade_methods(metrics: list[GradeMethodMetric], limit: int = 5) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for split in ("train", "val", "test"):
        for grade in GRADE_ORDER:
            subset = [item for item in metrics if item.split == split and item.grade == grade]
            subset.sort(key=lambda item: (item.exact, item.top1, item.races), reverse=True)
            out[f"{split}:{grade}"] = [asdict(item) for item in subset[:limit]]
    return out


def build_payload(snapshot_path: Path) -> dict:
    records = load_records(snapshot_path)
    df = attach_grade_proxy(frame_from_records(records))
    metrics = grade_metrics(df)
    train_route = best_method_by_grade(df, "train", "exact")
    train_top1_route = best_method_by_grade(df, "train", "top1")
    val_route = best_method_by_grade(df, "val", "exact")
    val_top1_route = best_method_by_grade(df, "val", "top1")
    static_routes = [
        {"name": f"single_{method}", "route": {grade: method for grade in GRADE_ORDER}}
        for method in METHODS
    ]
    routed = [
        {"name": "grade_route_train_best", "route": train_route},
        {"name": "grade_route_train_top1_best", "route": train_top1_route},
        {"name": "grade_route_val_oracle", "route": val_route},
        {"name": "grade_route_val_top1_oracle", "route": val_top1_route},
        *static_routes,
    ]
    routed_metrics = [
        asdict(routed_metric(df, item["name"], item["route"], split))
        for item in routed
        for split in ("train", "val", "test", "all")
    ]
    test_rows = [row for row in routed_metrics if row["split"] == "test"]
    test_rows.sort(key=lambda row: (row["exact_lift_pp"], row["top1_lift_pp"], row["exact"]), reverse=True)
    train_top1_metrics = [
        row for row in routed_metrics
        if row["name"] == "grade_route_train_top1_best"
    ]
    train_top1_by_split = {str(row["split"]): row for row in train_top1_metrics}
    deployable = [
        row for row in test_rows
        if row["name"] == "grade_route_train_best"
        and row["exact_lift_pp"] > 0
        and row["top1_lift_pp"] >= 0
    ]
    top1_candidate = bool(
        train_top1_by_split.get("train", {}).get("top1_lift_pp", 0.0) > 0.0
        and train_top1_by_split.get("val", {}).get("top1_lift_pp", 0.0) > 0.0
        and train_top1_by_split.get("test", {}).get("top1_lift_pp", 0.0) > 0.0
    )
    return {
        "records": int(len(df)),
        "grade_counts": {str(key): int(value) for key, value in df["grade_proxy"].value_counts().sort_index().items()},
        "train_route": train_route,
        "train_top1_route": train_top1_route,
        "val_oracle_route": val_route,
        "val_top1_oracle_route": val_top1_route,
        "selected_policy": "grade_route_train_best" if deployable else "baseline",
        "selected_top1_policy": "grade_route_train_top1_best" if top1_candidate else "baseline",
        "deployable": bool(deployable),
        "top1_candidate": top1_candidate,
        "top_grade_methods": top_grade_methods(metrics),
        "routed_metrics": routed_metrics,
        "test_leaderboard": test_rows[:20],
    }


def write_markdown(path: Path, payload: dict) -> None:
    lines = [
        "# KCYCLE grade routing experiment",
        "",
        f"records: {payload['records']}",
        f"grade_counts: {json.dumps(payload['grade_counts'], ensure_ascii=False)}",
        f"train_route: {json.dumps(payload['train_route'], ensure_ascii=False)}",
        f"train_top1_route: {json.dumps(payload['train_top1_route'], ensure_ascii=False)}",
        f"selected_policy: {payload['selected_policy']}",
        f"selected_top1_policy: {payload['selected_top1_policy']}",
        f"deployable: {payload['deployable']}",
        f"top1_candidate: {payload['top1_candidate']}",
        "",
        "## Test leaderboard",
        "",
        "| name | exact | top1 | board_exact | board_top1 | exact_lift_pp | top1_lift_pp | route |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload["test_leaderboard"][:12]:
        row_payload = {**row, "route_text": json.dumps(row["route"], ensure_ascii=False)}
        lines.append(
            "| {name} | {exact:.4f} | {top1:.4f} | {board_exact:.4f} | {board_top1:.4f} | "
            "{exact_lift_pp:+.3f} | {top1_lift_pp:+.3f} | {route_text} |".format(**row_payload)
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshots", default=str(ROOT / "data" / "kcycle_trifecta_snapshots.jsonl"))
    parser.add_argument("--out-json", default=str(ROOT / "data" / "kcycle_grade_routing_results.json"))
    parser.add_argument("--out-md", default=str(ROOT / "docs" / "kcycle_grade_routing_results.md"))
    args = parser.parse_args()

    payload = build_payload(Path(args.snapshots))
    Path(args.out_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(Path(args.out_md), payload)
    print(json.dumps({
        "records": payload["records"],
        "train_route": payload["train_route"],
        "train_top1_route": payload["train_top1_route"],
        "selected_policy": payload["selected_policy"],
        "selected_top1_policy": payload["selected_top1_policy"],
        "deployable": payload["deployable"],
        "top1_candidate": payload["top1_candidate"],
        "best_test": payload["test_leaderboard"][0] if payload["test_leaderboard"] else None,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
