#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd  # noqa: PANDAS_OK - existing KRA analysis stack uses pandas

ROOT: Final = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEFAULT_SOURCE_DB: Final = Path("/Users/tttksj/kra/data/kra.db")
DEFAULT_CORPUS: Final = ROOT / "data" / "kra_market_corpus.jsonl"
DEFAULT_RESULTS_JSON: Final = ROOT / "data" / "kra_market_corpus_results.json"
DEFAULT_RESULTS_MD: Final = ROOT / "data" / "kra_market_corpus_results.md"
DEFAULT_PROGRESS: Final = ROOT / "runs" / "kra_corpus_progress.md"
START_DATE: Final = "20240101"
END_DATE: Final = "20260731"
POOL_WIN: Final = "단승식"
POOL_PLACE: Final = "연승식"
POOL_QUINELLA: Final = "복승식"
POOL_QUINELLA_PLACE: Final = "복연승식"
POOL_EXACTA: Final = "쌍승식"
POOL_TRIO: Final = "삼복승식"
POOL_TRIFECTA: Final = "삼쌍승식"
POOLS: Final = (
    POOL_WIN,
    POOL_PLACE,
    POOL_QUINELLA,
    POOL_QUINELLA_PLACE,
    POOL_EXACTA,
    POOL_TRIO,
    POOL_TRIFECTA,
)


@dataclass(frozen=True, slots=True)
class RaceMarket:
    key: str
    meet: str
    rc_date: str
    rc_no: int
    field_size: int
    winner: int
    top2: tuple[int, int] | None
    top3: tuple[int, int, int] | None
    market_top1: int
    market_top3: tuple[int, ...]
    favorite_odds: float
    second_odds: float
    ratio12: float
    favorite_probability: float
    model_top1: int | None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def append_progress(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"- {utc_now()} {message}\n")


def load_race_frame(db_path: Path) -> pd.DataFrame:
    columns = (
        "raw_json,meet,rcDate,rcNo,chulNo,ord,winOdds,plcOdds,hrName,"
        "wgBudam,wgHr,age,sex,rating,rcDist,jkNo,trNo,owNo,budam"
    )
    with sqlite3.connect(db_path) as connection:
        frame = pd.read_sql_query(
            f"""
            SELECT {columns}
            FROM race_result
            WHERE rcDate >= ? AND rcDate <= ?
            """,
            connection,
            params=(START_DATE, END_DATE),
        )
    for column in ("rcDate", "rcNo", "chulNo", "ord", "winOdds", "plcOdds"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame[
        frame["rcDate"].notna()
        & frame["rcNo"].notna()
        & frame["chulNo"].notna()
        & frame["ord"].between(1, 30)
        & (frame["winOdds"] > 0)
    ].copy()
    frame["rcDate"] = frame["rcDate"].astype(int).astype(str)
    frame["rcNo"] = frame["rcNo"].astype(int)
    frame["chulNo"] = frame["chulNo"].astype(int)
    frame["ord"] = frame["ord"].astype(int)
    frame["race_key"] = (
        frame["meet"].astype(str)
        + "|"
        + frame["rcDate"].astype(str)
        + "|"
        + frame["rcNo"].astype(str)
    )
    frame["field_size"] = frame.groupby("race_key")["race_key"].transform("size")
    return frame[frame["field_size"] >= 2].copy()


def race_rows(frame: pd.DataFrame) -> list[RaceMarket]:
    out: list[RaceMarket] = []
    for key, group in frame.groupby("race_key", sort=True):
        ordered_finish = group.sort_values(["ord", "chulNo"], kind="stable")
        finish = tuple(int(value) for value in ordered_finish["chulNo"].tolist())
        if not finish:
            continue
        market = group.sort_values(["winOdds", "chulNo"], kind="stable")
        market_order = tuple(int(value) for value in market["chulNo"].tolist())
        odds = [float(value) for value in market["winOdds"].tolist()]
        if len(market_order) < 2 or odds[0] <= 0:
            continue
        inverse = np.asarray([1.0 / value if value > 0 else 0.0 for value in odds], dtype=float)
        total_inverse = float(inverse.sum())
        top3 = finish[:3] if len(finish) >= 3 else None
        out.append(
            RaceMarket(
                key=str(key),
                meet=str(group["meet"].iloc[0]),
                rc_date=str(group["rcDate"].iloc[0]),
                rc_no=int(group["rcNo"].iloc[0]),
                field_size=int(group["field_size"].iloc[0]),
                winner=int(finish[0]),
                top2=(int(finish[0]), int(finish[1])) if len(finish) >= 2 else None,
                top3=(int(top3[0]), int(top3[1]), int(top3[2])) if top3 is not None else None,
                market_top1=int(market_order[0]),
                market_top3=market_order[:3],
                favorite_odds=float(odds[0]),
                second_odds=float(odds[1]),
                ratio12=float(odds[1] / odds[0]),
                favorite_probability=float(inverse[0] / total_inverse) if total_inverse > 0 else 0.0,
                model_top1=None,
            )
        )
    return out


def field_bucket(field_size: int) -> str:
    if field_size <= 7:
        return "field_le_7"
    if field_size <= 10:
        return "field_8_10"
    return "field_11_plus"


def pull_tiers(races: list[RaceMarket]) -> dict[str, list[RaceMarket]]:
    return {
        "all": races,
        "very_strong_pull": [r for r in races if r.favorite_odds <= 1.8 and r.ratio12 >= 1.50],
        "strong_pull": [r for r in races if r.favorite_odds <= 2.5 and r.ratio12 >= 1.30],
        "price_short": [r for r in races if r.favorite_odds <= 2.0],
        "gap_wide": [r for r in races if r.ratio12 >= 1.50],
        "weak_or_open": [r for r in races if r.favorite_odds > 4.0 or r.ratio12 < 1.10],
    }


def race_metric(items: list[RaceMarket], total: int) -> dict[str, float | int | None]:
    if not items:
        return {"races": 0, "coverage": 0.0, "top1": None, "top3": None}
    return {
        "races": len(items),
        "coverage": len(items) / total if total else 0.0,
        "top1": sum(r.market_top1 == r.winner for r in items) / len(items),
        "top3": sum(r.winner in r.market_top3 for r in items) / len(items),
    }


def tier_tables(races: list[RaceMarket]) -> dict[str, dict[str, float | int | None]]:
    tiers = pull_tiers(races)
    return {name: race_metric(items, len(races)) for name, items in tiers.items()}


def field_tier_tables(races: list[RaceMarket]) -> dict[str, dict[str, dict[str, float | int | None]]]:
    out = {}
    for bucket in ("field_le_7", "field_8_10", "field_11_plus"):
        subset = [race for race in races if field_bucket(race.field_size) == bucket]
        out[bucket] = tier_tables(subset)
    return out


def calibration_curve(races: list[RaceMarket]) -> list[dict[str, float | int | None]]:
    bins = (0.0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.55, 1.01)
    out = []
    for low, high in zip(bins, bins[1:]):
        rows = [r for r in races if low <= r.favorite_probability < high]
        out.append(
            {
                "probability_bin": f"[{low:.2f},{high:.2f})",
                "races": len(rows),
                "mean_market_probability": (
                    sum(r.favorite_probability for r in rows) / len(rows) if rows else None
                ),
                "observed_top1": (
                    sum(r.market_top1 == r.winner for r in rows) / len(rows) if rows else None
                ),
            }
        )
    return out


def combo_key(pool: str, row: sqlite3.Row) -> str:
    first = int(row["chulNo"] or 0)
    second = int(row["chulNo2"] or 0)
    third = int(row["chulNo3"] or 0)
    if pool in {POOL_WIN, POOL_PLACE}:
        return str(first)
    if pool in {POOL_QUINELLA, POOL_QUINELLA_PLACE, POOL_EXACTA}:
        return f"{first}-{second}"
    return f"{first}-{second}-{third}"


def actual_combo(pool: str, race: RaceMarket) -> str | None:
    if pool == POOL_WIN:
        return str(race.winner)
    if pool == POOL_PLACE:
        return None
    if pool == POOL_QUINELLA:
        if race.top2 is None:
            return None
        return "-".join(str(x) for x in sorted(race.top2))
    if pool == POOL_QUINELLA_PLACE:
        return None
    if pool == POOL_EXACTA:
        if race.top2 is None:
            return None
        return f"{race.top2[0]}-{race.top2[1]}"
    if pool == POOL_TRIO:
        if race.top3 is None:
            return None
        return "-".join(str(x) for x in sorted(race.top3))
    if pool == POOL_TRIFECTA:
        if race.top3 is None:
            return None
        return f"{race.top3[0]}-{race.top3[1]}-{race.top3[2]}"
    return None


def normalized_board_probability(board: dict[str, float], combo: str) -> float:
    inverse = {key: 1.0 / value for key, value in board.items() if value > 0.0}
    total = sum(inverse.values())
    return inverse.get(combo, 0.0) / total if total > 0 else 0.0


def dividend_board(connection: sqlite3.Connection, race: RaceMarket) -> dict[str, dict[str, float]]:
    cursor = connection.execute(
        """
        SELECT pool, chulNo, chulNo2, chulNo3, odds
        FROM dividend
        WHERE meet_code = ? AND rcDate = ? AND rcNo = ?
        ORDER BY pool, odds, chulNo, chulNo2, chulNo3
        """,
        (race.meet, int(race.rc_date), race.rc_no),
    )
    boards: dict[str, dict[str, float]] = defaultdict(dict)
    for raw in cursor.fetchall():
        odds = float(raw["odds"] or 0.0)
        if odds <= 0:
            continue
        pool = str(raw["pool"])
        boards[pool][combo_key(pool, raw)] = odds
    return dict(boards)


def write_corpus(
    path: Path,
    db_path: Path,
    frame: pd.DataFrame,
    races: list[RaceMarket],
    progress: Path,
) -> dict[str, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    race_groups = {key: group for key, group in frame.groupby("race_key", sort=False)}
    stats = {"races": 0, "entries": 0, "races_with_dividend": 0}
    with sqlite3.connect(db_path) as connection, path.open("w", encoding="utf-8") as handle:
        connection.row_factory = sqlite3.Row
        for index, race in enumerate(races, start=1):
            group = race_groups[race.key].sort_values(["chulNo"], kind="stable")
            finish_order = [
                int(value)
                for value in group.sort_values(["ord", "chulNo"], kind="stable")["chulNo"].tolist()
            ]
            entries = [
                {
                    "chulNo": int(row.chulNo),
                    "ord": int(row.ord),
                    "winOdds": float(row.winOdds),
                    "plcOdds": float(row.plcOdds) if not math.isnan(float(row.plcOdds)) else None,
                    "hrName": str(row.hrName or ""),
                }
                for row in group.itertuples(index=False)
            ]
            dividends = dividend_board(connection, race)
            record = {
                "race_key": race.key,
                "meet": race.meet,
                "rcDate": race.rc_date,
                "rcNo": race.rc_no,
                "field_size": race.field_size,
                "finish_order": finish_order,
                "entries": entries,
                "dividends": dividends,
            }
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            stats["races"] += 1
            stats["entries"] += len(entries)
            stats["races_with_dividend"] += int(bool(dividends))
            if index % 1000 == 0:
                append_progress(progress, f"corpus write checkpoint races={index}")
    return stats


def exotic_rows(db_path: Path, races: list[RaceMarket]) -> list[dict[str, float | int | str | bool]]:
    rows = []
    race_lookup = {race.key: race for race in races}
    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        for race in races:
            boards = dividend_board(connection, race)
            for pool in POOLS:
                board = boards.get(pool)
                actual = actual_combo(pool, race)
                if not board or actual is None:
                    continue
                ranked = sorted(board.items(), key=lambda item: (item[1], item[0]))
                if len(ranked) < 2:
                    continue
                favorite, favorite_odds = ranked[0]
                second_odds = ranked[1][1]
                top3 = {combo for combo, _ in ranked[:3]}
                rows.append(
                    {
                        "race_key": race.key,
                        "pool": pool,
                        "hit_top1": favorite == actual,
                        "hit_top3": actual in top3,
                        "favorite_odds": float(favorite_odds),
                        "ratio12": float(second_odds / favorite_odds) if favorite_odds > 0 else 0.0,
                        "favorite_probability": normalized_board_probability(board, favorite),
                        "field_bucket": field_bucket(race.field_size),
                    }
                )
    return rows


def exotic_metric(items: list[dict[str, float | int | str | bool]], total: int) -> dict[str, float | int | None]:
    if not items:
        return {"races": 0, "coverage": 0.0, "top1": None, "top3": None}
    return {
        "races": len(items),
        "coverage": len(items) / total if total else 0.0,
        "top1": sum(bool(row["hit_top1"]) for row in items) / len(items),
        "top3": sum(bool(row["hit_top3"]) for row in items) / len(items),
    }


def exotic_tiers(rows: list[dict[str, float | int | str | bool]]) -> dict[str, dict[str, float | int | None]]:
    tiers = {
        "all": rows,
        "very_strong_pull": [
            row for row in rows if float(row["favorite_odds"]) <= 6.0 and float(row["ratio12"]) >= 1.50
        ],
        "strong_pull": [
            row for row in rows if float(row["favorite_odds"]) <= 12.0 and float(row["ratio12"]) >= 1.30
        ],
        "gap_wide": [row for row in rows if float(row["ratio12"]) >= 1.50],
        "weak_or_open": [
            row for row in rows if float(row["favorite_odds"]) > 40.0 or float(row["ratio12"]) < 1.10
        ],
    }
    return {name: exotic_metric(items, len(rows)) for name, items in tiers.items()}


def exotic_summary(rows: list[dict[str, float | int | str | bool]]) -> dict[str, dict]:
    out = {}
    for pool in sorted({str(row["pool"]) for row in rows}):
        pool_rows = [row for row in rows if row["pool"] == pool]
        out[pool] = {
            "tiers": exotic_tiers(pool_rows),
            "by_field_size": {
                bucket: exotic_tiers([row for row in pool_rows if row["field_bucket"] == bucket])
                for bucket in ("field_le_7", "field_8_10", "field_11_plus")
            },
            "calibration": exotic_calibration(pool_rows),
        }
    return out


def exotic_calibration(rows: list[dict[str, float | int | str | bool]]) -> list[dict[str, float | int | None | str]]:
    bins = (0.0, 0.001, 0.003, 0.006, 0.010, 0.020, 0.040, 0.080, 1.01)
    out = []
    for low, high in zip(bins, bins[1:]):
        part = [row for row in rows if low <= float(row["favorite_probability"]) < high]
        out.append(
            {
                "probability_bin": f"[{low:.3f},{high:.3f})",
                "races": len(part),
                "mean_market_probability": (
                    sum(float(row["favorite_probability"]) for row in part) / len(part) if part else None
                ),
                "observed_top1": (
                    sum(bool(row["hit_top1"]) for row in part) / len(part) if part else None
                ),
            }
        )
    return out


def attach_model_picks(db_path: Path, races: list[RaceMarket], progress: Path) -> tuple[list[RaceMarket], dict]:
    import joblib  # noqa: PLC0415
    from kra_model_evaluation import race_normalize  # noqa: PLC0415
    from kra_training_features import build_features, load_rows  # noqa: PLC0415

    append_progress(progress, "model batch scoring start")
    model = joblib.load(ROOT / "static" / "models" / "kra_model.joblib")
    feature_frame, _ = build_features(load_rows(db_path))
    feature_frame = feature_frame[
        (feature_frame["rcDate"] >= START_DATE) & (feature_frame["rcDate"] <= END_DATE)
    ].copy()
    feature_frame["race_key"] = (
        feature_frame["meet"].astype(str)
        + "|"
        + feature_frame["rcDate"].astype(str)
        + "|"
        + feature_frame["rcNo"].astype(str)
    )
    columns = list(model["cols"])
    median = pd.Series(model["med"], dtype=float)
    values = feature_frame.reindex(columns=columns, fill_value=0)
    for column in columns:
        if column in median.index and pd.notna(median[column]):
            values[column] = values[column].fillna(float(median[column]))
    values = values.apply(pd.to_numeric, errors="coerce").fillna(0)
    probability = race_normalize(feature_frame, model["win"].predict_proba(values)[:, 1])
    scored = feature_frame[["race_key", "chulNo", "win"]].copy()
    scored["probability"] = probability
    scored["chulNo"] = pd.to_numeric(scored["chulNo"], errors="coerce").astype(int)
    top_by_key = {
        str(row.race_key): int(row.chulNo)
        for row in scored.sort_values(["race_key", "probability", "chulNo"], ascending=[True, False, True])
        .groupby("race_key", sort=False)
        .head(1)
        .itertuples(index=False)
    }
    updated = [
        RaceMarket(
            key=race.key,
            meet=race.meet,
            rc_date=race.rc_date,
            rc_no=race.rc_no,
            field_size=race.field_size,
            winner=race.winner,
            top2=race.top2,
            top3=race.top3,
            market_top1=race.market_top1,
            market_top3=race.market_top3,
            favorite_odds=race.favorite_odds,
            second_odds=race.second_odds,
            ratio12=race.ratio12,
            favorite_probability=race.favorite_probability,
            model_top1=top_by_key.get(race.key),
        )
        for race in races
    ]
    errors = sum(race.model_top1 is None for race in updated)
    append_progress(progress, f"model batch scoring complete scored={len(updated) - errors} errors={errors}")
    return updated, {"model_scored_races": len(updated) - errors, "model_score_errors": errors}


def disagreement_summary(races: list[RaceMarket]) -> dict[str, dict[str, float | int | None]]:
    def one(label: str, rows: list[RaceMarket]) -> tuple[str, dict[str, float | int | None]]:
        valid = [race for race in rows if race.model_top1 is not None]
        disagreed = [race for race in valid if race.model_top1 != race.market_top1]
        return label, {
            "races": len(disagreed),
            "coverage_of_scored": len(disagreed) / len(valid) if valid else 0.0,
            "model_top1": (
                sum(race.model_top1 == race.winner for race in disagreed) / len(disagreed)
                if disagreed else None
            ),
            "market_top1": (
                sum(race.market_top1 == race.winner for race in disagreed) / len(disagreed)
                if disagreed else None
            ),
        }

    ranges = {
        "all": races,
        "2026": [race for race in races if race.rc_date.startswith("2026")],
        "fresh_from_20260622": [race for race in races if race.rc_date >= "20260622"],
    }
    return dict(one(name, rows) for name, rows in ranges.items())


def write_markdown(payload: dict, path: Path) -> None:
    def pct(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.3f}"

    lines = [
        "# KRA market corpus results",
        "",
        f"generated_at: {payload['generated_at']}",
        f"source_db: {payload['source_db']}",
        f"corpus: {payload['corpus_path']}",
        "",
        "## Probe",
        "- `RaceDetailResult_1` has win/place odds and finish order in the local official DB.",
        "- `API160_1/integratedInfo_1` is the confirmed-dividend route; local shell DNS blocked a fresh live probe, so measurement uses already collected official rows.",
        "",
        "## Corpus",
        f"- races: {payload['corpus_stats']['races']}",
        f"- entries: {payload['corpus_stats']['entries']}",
        f"- races_with_dividend: {payload['corpus_stats']['races_with_dividend']}",
        "",
        "## Win market tiers",
        "| tier | races | coverage | top1 | top3 |",
        "|---|---:|---:|---:|---:|",
    ]
    for tier, row in payload["win_market"]["tiers"].items():
        lines.append(
            f"| {tier} | {row['races']} | {pct(row['coverage'])} | {pct(row['top1'])} | {pct(row['top3'])} |"
        )
    lines.extend(["", "## Win market by field size", "| bucket | tier | races | coverage | top1 | top3 |", "|---|---|---:|---:|---:|---:|"])
    for bucket, tiers in payload["win_market"]["by_field_size"].items():
        for tier, row in tiers.items():
            lines.append(
                f"| {bucket} | {tier} | {row['races']} | {pct(row['coverage'])} | {pct(row['top1'])} | {pct(row['top3'])} |"
            )
    lines.extend(["", "## Exotic market baseline", "| pool | tier | races | coverage | top1 | top3 |", "|---|---|---:|---:|---:|---:|"])
    for pool, pool_payload in payload["exotic_market"].items():
        for tier, row in pool_payload["tiers"].items():
            lines.append(
                f"| {pool} | {tier} | {row['races']} | {pct(row['coverage'])} | {pct(row['top1'])} | {pct(row['top3'])} |"
            )
    lines.extend(["", "## Model-market disagreement", "| split | races | coverage | model_top1 | market_top1 |", "|---|---:|---:|---:|---:|"])
    for split, row in payload["model_market_disagreement"].items():
        lines.append(
            f"| {split} | {row['races']} | {pct(row['coverage_of_scored'])} | {pct(row['model_top1'])} | {pct(row['market_top1'])} |"
        )
    lines.extend(
        [
            "",
            "## Verdict",
            "- Pure measurement only; no promotion claim.",
            "- Exotic coverage starts at 2025-01-03 in the local official dividend table; 2024 exotic boards are absent from the local table.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DEFAULT_SOURCE_DB)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--results-json", type=Path, default=DEFAULT_RESULTS_JSON)
    parser.add_argument("--results-md", type=Path, default=DEFAULT_RESULTS_MD)
    parser.add_argument("--progress", type=Path, default=DEFAULT_PROGRESS)
    args = parser.parse_args()

    append_progress(args.progress, "collection start")
    frame = load_race_frame(args.db)
    races = race_rows(frame)
    corpus_stats = write_corpus(args.corpus, args.db, frame, races, args.progress)
    append_progress(args.progress, f"corpus complete races={corpus_stats['races']} entries={corpus_stats['entries']}")

    scored_races, model_stats = attach_model_picks(args.db, races, args.progress)
    exotic = exotic_rows(args.db, scored_races)
    payload = {
        "generated_at": utc_now(),
        "source_db": str(args.db),
        "corpus_path": str(args.corpus),
        "date_range": {"start": START_DATE, "end": END_DATE},
        "corpus_stats": corpus_stats,
        "model_stats": model_stats,
        "win_market": {
            "tiers": tier_tables(scored_races),
            "by_field_size": field_tier_tables(scored_races),
            "calibration": calibration_curve(scored_races),
        },
        "exotic_market": exotic_summary(exotic),
        "model_market_disagreement": disagreement_summary(scored_races),
        "notes": [
            "학습/선택 개입 없는 순수 측정.",
            "RaceDetailResult_1 winOdds/plcOdds covers all local races.",
            "API160_1 local dividend table covers 2025-01-03..2026-06-21; 2024 exotic board rows are absent.",
            "Production artifact pairwise.enabled=false; disagreement metrics are descriptive only.",
        ],
    }
    args.results_json.parent.mkdir(parents=True, exist_ok=True)
    args.results_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(payload, args.results_md)
    append_progress(args.progress, "measurement complete")
    print(json.dumps({"corpus": str(args.corpus), "results": str(args.results_json)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
