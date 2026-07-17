#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN_YEARS = {"2018", "2019", "2020", "2021", "2022", "2023"}
VAL_YEARS = {"2024", "2025"}
TEST_YEARS = {"2026"}


@dataclass(frozen=True, slots=True)
class Row:
    year: str
    actual: str
    board_best: str
    mass_best: str
    bt_best: str
    best_odds: float
    gap12: float
    gap15: float
    first_mass_best: float
    second_mass_best: float
    third_mass_best: float
    top3_same_first: bool
    top5_same_first: bool
    top3_same_pair: bool
    actual_rank: int
    actual_odds: float


@dataclass(frozen=True, slots=True)
class Metric:
    name: str
    split: str
    n: int
    exact: float
    top1: float
    board_exact: float
    board_top1: float
    exact_lift_pp: float
    top1_lift_pp: float
    coverage: float
    rule: str


def parts(combo: str) -> tuple[int, int, int]:
    a, b, c = str(combo).split("-")
    return int(a), int(b), int(c)


def combo_from_parts(values: tuple[int, int, int]) -> str:
    return "-".join(str(x) for x in values)


def year_of(record: dict) -> str:
    return str(record.get("stnd_yr") or str(record.get("date") or "")[:4])


def load_rows(path: Path) -> list[Row]:
    rows: list[Row] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            board = {
                str(combo): float(odds)
                for combo, odds in (record.get("board") or {}).items()
                if float(odds) > 0
            }
            actual = str(record.get("actual_order") or "")
            if len(board) != 210 or actual not in board:
                continue
            ranked = sorted(board.items(), key=lambda item: (item[1], item[0]))
            q_raw = {combo: 1.0 / odds for combo, odds in board.items()}
            q_sum = sum(q_raw.values())
            if q_sum <= 0:
                continue
            q = {combo: value / q_sum for combo, value in q_raw.items()}
            first_mass = {i: 0.0 for i in range(1, 8)}
            second_mass = {i: 0.0 for i in range(1, 8)}
            third_mass = {i: 0.0 for i in range(1, 8)}
            for combo, prob in q.items():
                a, b, c = parts(combo)
                first_mass[a] += prob
                second_mass[b] += prob
                third_mass[c] += prob
            best_combo, best_odds = ranked[0]
            best_a, best_b, best_c = parts(best_combo)
            mass_best = combo_from_parts((
                max(first_mass, key=lambda x: (first_mass[x], -x)),
                max(second_mass, key=lambda x: (second_mass[x], -x)),
                max(third_mass, key=lambda x: (third_mass[x], -x)),
            ))

            def bt_score(combo: str) -> float:
                a, b, c = parts(combo)
                return (
                    0.44 * math.log(max(q[combo], 1e-12))
                    + 0.92 * math.log(max(first_mass[a], 1e-12))
                    + 0.66 * math.log(max(second_mass[b], 1e-12))
                    + 0.48 * math.log(max(third_mass[c], 1e-12))
                )

            bt_best = max(board, key=lambda combo: (bt_score(combo), -board[combo]))
            rank_lookup = {combo: index for index, (combo, _) in enumerate(ranked, start=1)}
            top3 = [combo for combo, _ in ranked[:3]]
            top5 = [combo for combo, _ in ranked[:5]]
            rows.append(Row(
                year=year_of(record),
                actual=actual,
                board_best=best_combo,
                mass_best=mass_best,
                bt_best=bt_best,
                best_odds=best_odds,
                gap12=ranked[1][1] / best_odds,
                gap15=ranked[4][1] / best_odds,
                first_mass_best=first_mass[best_a],
                second_mass_best=second_mass[best_b],
                third_mass_best=third_mass[best_c],
                top3_same_first=len({combo.split("-")[0] for combo in top3}) == 1,
                top5_same_first=len({combo.split("-")[0] for combo in top5}) == 1,
                top3_same_pair=len({"-".join(combo.split("-")[:2]) for combo in top3}) == 1,
                actual_rank=rank_lookup[actual],
                actual_odds=board[actual],
            ))
    return rows


def split_rows(rows: list[Row], split: str) -> list[Row]:
    years = TRAIN_YEARS if split == "train" else VAL_YEARS if split == "val" else TEST_YEARS if split == "test" else TRAIN_YEARS | VAL_YEARS | TEST_YEARS
    return [row for row in rows if row.year in years]


def pick(row: Row, rule: dict) -> str | None:
    name = rule["name"]
    if name == "board":
        return row.board_best
    if name == "mass":
        return row.mass_best
    if name == "bt":
        return row.bt_best
    if name == "bt_guard":
        params = rule["params"]
        if row.second_mass_best >= params["second_mass_min"] and row.third_mass_best >= params["third_mass_min"]:
            return row.bt_best
        return row.board_best
    if name == "bt_selective":
        params = rule["params"]
        if (
            row.second_mass_best >= params["second_mass_min"]
            and row.third_mass_best >= params["third_mass_min"]
            and row.best_odds <= params["best_odds_max"]
        ):
            return row.bt_best
        return None
    if name == "board_selective":
        params = rule["params"]
        if row.best_odds <= params["best_odds_max"] and row.gap12 >= params["gap12_min"]:
            return row.board_best
        return None
    raise ValueError(name)


def metric(rows: list[Row], split: str, rule: dict, total_n: int | None = None) -> Metric:
    total_n = total_n or len(rows)
    selected: list[tuple[Row, str]] = []
    for row in rows:
        selected_combo = pick(row, rule)
        if selected_combo:
            selected.append((row, selected_combo))
    n = len(selected)
    if not n:
        exact = top1 = board_exact = board_top1 = 0.0
    else:
        exact = sum(combo == row.actual for row, combo in selected) / n
        top1 = sum(parts(combo)[0] == parts(row.actual)[0] for row, combo in selected) / n
        board_exact = sum(row.board_best == row.actual for row, _ in selected) / n
        board_top1 = sum(parts(row.board_best)[0] == parts(row.actual)[0] for row, _ in selected) / n
    return Metric(
        name=rule["label"],
        split=split,
        n=n,
        exact=exact,
        top1=top1,
        board_exact=board_exact,
        board_top1=board_top1,
        exact_lift_pp=(exact - board_exact) * 100.0,
        top1_lift_pp=(top1 - board_top1) * 100.0,
        coverage=n / total_n if total_n else 0.0,
        rule=rule["rule"],
    )


def candidate_rules(rows: list[Row]) -> list[dict]:
    train = split_rows(rows, "train")
    second_values = sorted(row.second_mass_best for row in train)
    third_values = sorted(row.third_mass_best for row in train)

    def q(values: list[float], p: float) -> float:
        return values[min(len(values) - 1, max(0, int(len(values) * p)))]

    rules = [
        {"label": "board_best", "name": "board", "rule": "마감 삼쌍 최저배당 순서"},
        {"label": "position_mass", "name": "mass", "rule": "1/2/3착 위치별 암시질량 최댓값 조합"},
        {"label": "bt_position_rerank", "name": "bt", "rule": "마감배당 q + 위치질량 Bradley-Terry식 재랭킹"},
    ]
    for sp in [0.55, 0.65, 0.75, 0.85]:
        for tp in [0.55, 0.65, 0.75, 0.85]:
            s = q(second_values, sp)
            t = q(third_values, tp)
            rules.append({
                "label": f"bt_guard_s{sp:.2f}_t{tp:.2f}",
                "name": "bt_guard",
                "params": {"second_mass_min": s, "third_mass_min": t},
                "rule": f"2착질량>={s:.4f}, 3착질량>={t:.4f}이면 BT 재랭킹, 아니면 최저배당",
            })
            rules.append({
                "label": f"bt_select_s{sp:.2f}_t{tp:.2f}_o12",
                "name": "bt_selective",
                "params": {"second_mass_min": s, "third_mass_min": t, "best_odds_max": 12.0},
                "rule": f"2착질량>={s:.4f}, 3착질량>={t:.4f}, 최저배당<=12일 때만 BT",
            })
    for odds in [3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0]:
        for gap in [1.05, 1.2, 1.5, 2.0]:
            rules.append({
                "label": f"board_select_o{odds:.0f}_g{gap:.2f}",
                "name": "board_selective",
                "params": {"best_odds_max": odds, "gap12_min": gap},
                "rule": f"최저배당<={odds:.1f}, 1-2위 배당격차>={gap:.2f}일 때만 마감 최저배당",
            })
    return rules


def choose_on_val(rows: list[Row], rules: list[dict]) -> tuple[dict, list[Metric], list[Metric]]:
    train = split_rows(rows, "train")
    val = split_rows(rows, "val")
    train_metrics = [metric(train, "train", rule) for rule in rules]
    val_metrics = [metric(val, "val", rule) for rule in rules]
    board_val = next(item for item in val_metrics if item.name == "board_best")
    eligible = [
        (rule, val_item)
        for rule, val_item in zip(rules, val_metrics)
        if val_item.n >= 250 and val_item.exact >= board_val.exact and val_item.top1 >= board_val.top1 - 0.005
    ]
    if not eligible:
        return rules[0], train_metrics, val_metrics
    return max(eligible, key=lambda item: (item[1].exact, item[1].top1, item[1].coverage))[0], train_metrics, val_metrics


def bucket_summary(rows: list[Row]) -> dict:
    buckets = [(0, 10), (10, 20), (20, 40), (40, 80), (80, 160), (160, 999999)]
    counts = {f"{lo}-{hi}": 0 for lo, hi in buckets}
    for row in rows:
        for lo, hi in buckets:
            if lo < row.actual_odds <= hi:
                counts[f"{lo}-{hi}"] += 1
                break
    total = len(rows)
    return {
        label: {"count": count, "share": count / total if total else 0.0}
        for label, count in counts.items()
    }


def run(args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    rows = load_rows(Path(args.snapshots))
    rules = candidate_rules(rows)
    chosen, train_metrics, val_metrics = choose_on_val(rows, rules)
    test = split_rows(rows, "test")
    all_metrics = {
        "train": [asdict(item) for item in train_metrics],
        "val": [asdict(item) for item in val_metrics],
        "test": [asdict(metric(test, "test", rule)) for rule in rules],
    }
    chosen_metrics = {
        split: asdict(metric(split_rows(rows, split), split, chosen))
        for split in ["train", "val", "test", "all"]
    }
    board_metrics = {
        split: asdict(metric(split_rows(rows, split), split, rules[0]))
        for split in ["train", "val", "test", "all"]
    }
    payload = {
        "status": "ok",
        "records": len(rows),
        "elapsed_sec": time.perf_counter() - started,
        "chosen": chosen,
        "chosen_metrics": chosen_metrics,
        "board_metrics": board_metrics,
        "actual_odds_buckets": bucket_summary(rows),
        "metrics": all_metrics,
        "interpretation": {
            "leakage_warning": "archive_import rows are final/confirmed boards; treat as late-market proxy, not verified live-before-close snapshots.",
            "deploy_rule": "Only use live trifecta boards fetched before result settlement; settled result payout boards must stay learning/validation only.",
        },
    }
    return payload


def write_markdown(path: Path, payload: dict) -> None:
    rows = []
    for split in ["val", "test"]:
        for item in payload["metrics"][split]:
            rows.append((split, item))
    rows.sort(key=lambda item: (item[0], -item[1]["exact"], -item[1]["top1"], -item[1]["coverage"]))
    lines = [
        "# KCYCLE late market pull experiment",
        "",
        f"records: {payload['records']}",
        f"elapsed_sec: {payload['elapsed_sec']:.2f}",
        "",
        "주의: archive_import는 확정/마감 배당판 성격이므로 live-before-close 검증과 구분한다.",
        "",
        "## Chosen",
        "",
        f"chosen: {payload['chosen']['label']}",
        "",
        "| split | n | coverage | exact | top1 | board_exact | board_top1 | exact_lift_pp | top1_lift_pp |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ["train", "val", "test", "all"]:
        item = payload["chosen_metrics"][split]
        lines.append(
            f"| {split} | {item['n']} | {item['coverage']:.3f} | {item['exact']:.4f} | {item['top1']:.4f} | "
            f"{item['board_exact']:.4f} | {item['board_top1']:.4f} | {item['exact_lift_pp']:+.3f} | {item['top1_lift_pp']:+.3f} |"
        )
    lines.extend([
        "",
        "## Top Validation/Test Rules",
        "",
        "| split | name | n | coverage | exact | top1 | exact_lift_pp | top1_lift_pp | rule |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ])
    for split, item in rows[:80]:
        lines.append(
            f"| {split} | {item['name']} | {item['n']} | {item['coverage']:.3f} | {item['exact']:.4f} | {item['top1']:.4f} | "
            f"{item['exact_lift_pp']:+.3f} | {item['top1_lift_pp']:+.3f} | {item['rule']} |"
        )
    lines.extend(["", "## Actual Odds Buckets", ""])
    for label, item in payload["actual_odds_buckets"].items():
        lines.append(f"- {label}: {item['count']} ({item['share']:.3f})")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshots", default=str(ROOT / "data" / "kcycle_trifecta_snapshots.jsonl"))
    parser.add_argument("--out-json", default=str(ROOT / "data" / "kcycle_late_market_pull_results.json"))
    parser.add_argument("--out-md", default=str(ROOT / "docs" / "kcycle_late_market_pull_results.md"))
    args = parser.parse_args()
    payload = run(args)
    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(out_md, payload)
    print(json.dumps({
        "status": payload["status"],
        "records": payload["records"],
        "elapsed_sec": round(payload["elapsed_sec"], 3),
        "chosen": payload["chosen"]["label"],
        "chosen_test": payload["chosen_metrics"]["test"],
        "board_test": payload["board_metrics"]["test"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
