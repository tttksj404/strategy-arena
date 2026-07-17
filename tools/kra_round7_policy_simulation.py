#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

import joblib
import pandas as pd  # noqa: PANDAS_OK - existing KRA analysis stack uses pandas

ROOT: Final = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kra_model_evaluation import race_normalize  # noqa: E402
from kra_training_features import build_features, load_rows  # noqa: E402
from tools.kra_market_corpus_round5b import append_progress  # noqa: E402

DEFAULT_DB: Final = Path("/Users/tttksj/kra/data/kra.db")
DEFAULT_CORPUS: Final = ROOT / "data" / "kra_market_corpus.jsonl"
DEFAULT_MODEL: Final = ROOT / "static" / "models" / "kra_model.joblib"
DEFAULT_RESULTS_JSON: Final = ROOT / "data" / "kra_round7_results.json"
DEFAULT_RESULTS_MD: Final = ROOT / "data" / "kra_round7_results.md"
DEFAULT_PROGRESS: Final = ROOT / "runs" / "kra_corpus_progress.md"
FRESH_FROM: Final = "20260622"


@dataclass(frozen=True, slots=True)
class RaceRecord:
    key: str
    meet: str
    rc_date: str
    rc_no: int
    winner: int
    market_order: tuple[int, ...]
    model_order: tuple[int, ...]
    favorite_odds: float | None
    ratio12: float | None

    @property
    def has_complete_market(self) -> bool:
        return len(self.market_order) >= 2 and self.favorite_odds is not None and self.ratio12 is not None

    @property
    def weak_or_open(self) -> bool:
        return bool(
            self.has_complete_market
            and self.favorite_odds is not None
            and self.ratio12 is not None
            and (self.favorite_odds > 4.0 or self.ratio12 < 1.10)
        )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_model_orders(db_path: Path, model_path: Path) -> dict[str, tuple[int, ...]]:
    model = joblib.load(model_path)
    frame, _ = build_features(load_rows(db_path))
    columns = list(model["cols"])
    medians = pd.Series(model["med"], dtype=float)
    values = frame.reindex(columns=columns, fill_value=0)
    for column in columns:
        if column in medians.index and pd.notna(medians[column]):
            values[column] = values[column].fillna(float(medians[column]))
    values = values.apply(pd.to_numeric, errors="coerce").fillna(0)
    probability = race_normalize(frame, model["win"].predict_proba(values)[:, 1])
    scored = frame[["rk", "chulNo"]].copy()
    scored["probability"] = probability
    scored["chulNo"] = pd.to_numeric(scored["chulNo"], errors="coerce").astype(int)
    orders: dict[str, tuple[int, ...]] = {}
    ranked = scored.sort_values(
        ["rk", "probability", "chulNo"],
        ascending=[True, False, True],
        kind="stable",
    )
    for key, group in ranked.groupby("rk", sort=False):
        orders[str(key)] = tuple(int(value) for value in group["chulNo"].tolist())
    return orders


def market_order(entries: list[dict]) -> tuple[int, ...]:
    valid = []
    for row in entries:
        odds = row.get("winOdds")
        chul_no = row.get("chulNo")
        if odds is None or chul_no is None:
            continue
        odds_value = float(odds)
        if odds_value <= 0.0:
            continue
        valid.append((odds_value, int(chul_no)))
    return tuple(chul_no for _, chul_no in sorted(valid))


def load_records(corpus_path: Path, model_orders: dict[str, tuple[int, ...]]) -> list[RaceRecord]:
    records: list[RaceRecord] = []
    with corpus_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = json.loads(line)
            key = str(raw["race_key"])
            finish = tuple(int(value) for value in raw.get("finish_order", []))
            order = market_order(list(raw.get("entries", [])))
            if not finish or key not in model_orders:
                continue
            odds = [
                float(row["winOdds"])
                for row in raw.get("entries", [])
                if row.get("winOdds") is not None and float(row["winOdds"]) > 0.0
            ]
            sorted_odds = sorted(odds)
            favorite_odds = sorted_odds[0] if len(sorted_odds) >= 2 else None
            ratio12 = sorted_odds[1] / sorted_odds[0] if len(sorted_odds) >= 2 and sorted_odds[0] > 0 else None
            records.append(
                RaceRecord(
                    key=key,
                    meet=str(raw["meet"]),
                    rc_date=str(raw["rcDate"]),
                    rc_no=int(raw["rcNo"]),
                    winner=int(finish[0]),
                    market_order=order,
                    model_order=model_orders[key],
                    favorite_odds=favorite_odds,
                    ratio12=ratio12,
                )
            )
    return records


def selected_order(record: RaceRecord, policy: str) -> tuple[tuple[int, ...], str]:
    match policy:
        case "P0_v4_always":
            return record.model_order, "model"
        case "P1_current_gate":
            return record.model_order, "model"
        case "P2_market_if_odds":
            if record.has_complete_market:
                return record.market_order, "market"
            return record.model_order, "model"
        case "P3_market_except_weak_disagree":
            if not record.has_complete_market:
                return record.model_order, "model"
            if record.weak_or_open and record.market_order[0] != record.model_order[0]:
                return record.model_order, "model"
            return record.market_order, "market"
        case unreachable:
            raise AssertionError(f"unknown policy: {unreachable}")


def metric(records: list[RaceRecord], policy: str) -> dict[str, float | int]:
    if not records:
        return {"races": 0, "top1": 0.0, "top3": 0.0, "market_source_rate": 0.0}
    top1 = 0
    top3 = 0
    market_source = 0
    for record in records:
        order, source = selected_order(record, policy)
        if order and order[0] == record.winner:
            top1 += 1
        if record.winner in order[:3]:
            top3 += 1
        if source == "market":
            market_source += 1
    return {
        "races": len(records),
        "top1": top1 / len(records),
        "top3": top3 / len(records),
        "market_source_rate": market_source / len(records),
    }


def split_records(records: list[RaceRecord]) -> dict[str, list[RaceRecord]]:
    return {
        "all": records,
        "fresh_from_20260622": [record for record in records if record.rc_date >= FRESH_FROM],
    }


def policy_tables(records: list[RaceRecord]) -> dict[str, dict[str, dict[str, float | int]]]:
    policies = (
        "P0_v4_always",
        "P1_current_gate",
        "P2_market_if_odds",
        "P3_market_except_weak_disagree",
    )
    return {
        split: {policy: metric(rows, policy) for policy in policies}
        for split, rows in split_records(records).items()
    }


def choose_default_policy(tables: dict[str, dict[str, dict[str, float | int]]]) -> dict[str, float | str | bool]:
    fresh = tables["fresh_from_20260622"]
    p1_top1 = float(fresh["P1_current_gate"]["top1"])
    candidates = {
        policy: float(fresh[policy]["top1"]) - p1_top1
        for policy in ("P2_market_if_odds", "P3_market_except_weak_disagree")
    }
    winner = max(candidates, key=lambda name: (candidates[name], float(fresh[name]["top3"])))
    return {
        "winner": "market_if_odds" if winner == "P2_market_if_odds" else "market_except_weak_disagree",
        "winner_policy": winner,
        "fresh_lift_vs_p1_pp": candidates[winner] * 100.0,
        "supported": candidates[winner] >= 0.010,
    }


def write_markdown(payload: dict, path: Path) -> None:
    def pct(value: float | int) -> str:
        return f"{float(value) * 100.0:.2f}%"

    lines = [
        "# KRA Round 7 results",
        "",
        f"generated_at: {payload['generated_at']}",
        "",
        "## Current policy gate",
        payload["current_policy_gate"],
        "",
        "## Policy simulation",
    ]
    for split, table in payload["policy_simulation"].items():
        lines.extend(["", f"### {split}", "| policy | races | top1 | top3 | market_source |", "|---|---:|---:|---:|---:|"])
        for policy, row in table.items():
            lines.append(
                f"| {policy} | {row['races']} | {pct(row['top1'])} | {pct(row['top3'])} | {pct(row['market_source_rate'])} |"
            )
    decision = payload["policy_decision"]
    lines.extend(
        [
            "",
            "## Policy decision",
            f"- winner: {decision['winner']}",
            f"- fresh_lift_vs_p1_pp: {float(decision['fresh_lift_vs_p1_pp']):.2f}",
            f"- supported: {decision['supported']}",
            "",
            "## History extension",
            f"- status: {payload['history_extension']['status']}",
            f"- available_start: {payload['history_extension']['available_start']}",
            f"- note: {payload['history_extension']['note']}",
            "",
            "## Training extension",
            f"- status: {payload['training_extension']['status']}",
            f"- enabled: {payload['training_extension']['enabled']}",
            f"- note: {payload['training_extension']['note']}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--results-json", type=Path, default=DEFAULT_RESULTS_JSON)
    parser.add_argument("--results-md", type=Path, default=DEFAULT_RESULTS_MD)
    parser.add_argument("--progress", type=Path, default=DEFAULT_PROGRESS)
    args = parser.parse_args()

    append_progress(args.progress, "round7 policy simulation start")
    model_orders = load_model_orders(args.db, args.model)
    records = load_records(args.corpus, model_orders)
    tables = policy_tables(records)
    decision = choose_default_policy(tables)
    payload = {
        "generated_at": utc_now(),
        "inputs": {
            "db": str(args.db),
            "corpus": str(args.corpus),
            "model": str(args.model),
            "races": len(records),
            "fresh_from": FRESH_FROM,
        },
        "current_policy_gate": (
            "`predict_kra` calls `_kra_prediction_phase`; that returns `live_odds` only when "
            "`meta['odds_snapshot_fresh'] is True` and `_kra_market_probabilities(starters)` "
            "finds a complete positive win-odds board. `score_kra(..., use_market=True)` is "
            "therefore used only for fresh official pre-start odds; missing, partial, stale, "
            "post-result, or unproven odds stay on the model path."
        ),
        "policy_simulation": tables,
        "policy_decision": decision,
        "history_extension": {
            "status": "blocked_missing_network_or_api_access",
            "available_start": None,
            "note": "Updated after the data.go.kr extension probe runs.",
        },
        "training_extension": {
            "status": "not_run_until_history_extension_available",
            "enabled": False,
            "note": "Fresh holdout reserved; no promotion.",
        },
    }
    args.results_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(payload, args.results_md)
    append_progress(
        args.progress,
        "round7 policy simulation complete "
        f"winner={decision['winner']} fresh_lift_pp={float(decision['fresh_lift_vs_p1_pp']):.2f}",
    )
    print(json.dumps({"results": str(args.results_json), "winner": decision["winner"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
