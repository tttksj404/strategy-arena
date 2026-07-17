#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import engine  # noqa: E402


def combo_parts(combo: str) -> tuple[int, int, int] | None:
    if not re.fullmatch(r"[1-7]-[1-7]-[1-7]", str(combo or "")):
        return None
    a, b, c = str(combo).split("-")
    return int(a), int(b), int(c)


def best_board_combo(board: dict) -> tuple[str, float, float] | None:
    valid = []
    for combo, odds in (board or {}).items():
        if not re.fullmatch(r"[1-7]-[1-7]-[1-7]", str(combo)):
            continue
        try:
            o = float(odds)
        except (TypeError, ValueError):
            continue
        if o > 0:
            valid.append((str(combo), o))
    if len(valid) < 150:
        return None
    ranked = sorted(valid, key=lambda item: (item[1], item[0]))
    return ranked[0][0], ranked[0][1], ranked[1][1] / ranked[0][1]


def ymd_of(record: dict) -> str:
    return re.sub(r"\D", "", str(record.get("ymd") or record.get("date") or ""))[:8]


def snapshot_time(record: dict) -> str:
    return str(record.get("fetched_at") or record.get("captured_at") or "")


def race_token(record: dict) -> tuple[str, str, str]:
    return (
        ymd_of(record),
        str(record.get("meet") or "").strip(),
        str(record.get("race_no") or "").strip().lstrip("0") or "0",
    )


def actual_order(raw) -> str:
    if isinstance(raw, (list, tuple)):
        candidate = "-".join(str(value).strip() for value in raw)
    else:
        candidate = str(raw or "").strip()
    return candidate if combo_parts(candidate) else ""


def load_outcomes(path: Path) -> tuple[dict[tuple[str, str, str], str], int, int]:
    outcomes: dict[tuple[str, str, str], str] = {}
    conflicts: set[tuple[str, str, str]] = set()
    records = 0
    if not path.exists():
        return outcomes, records, len(conflicts)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            records += 1
            token = race_token(record)
            order = actual_order(record.get("actual_order"))
            if not order or token in conflicts:
                continue
            previous = outcomes.get(token)
            if previous and previous != order:
                outcomes.pop(token, None)
                conflicts.add(token)
            else:
                outcomes[token] = order
    return outcomes, records, len(conflicts)


def empty_metric() -> dict:
    return {"n": 0, "exact": None, "top1": None, "strong_pull_n": 0, "strong_pull_exact": None, "strong_pull_top1": None}


def finalize_metric(items: list[dict]) -> dict:
    if not items:
        return empty_metric()
    exact = sum(item["best"] == item["actual"] for item in items) / len(items)
    top1 = sum(combo_parts(item["best"])[0] == combo_parts(item["actual"])[0] for item in items) / len(items)
    strong = [item for item in items if item["best_odds"] <= 3.0 and item["gap12"] >= 1.2]
    out = {"n": len(items), "exact": exact, "top1": top1, "strong_pull_n": len(strong)}
    if strong:
        out["strong_pull_exact"] = sum(item["best"] == item["actual"] for item in strong) / len(strong)
        out["strong_pull_top1"] = sum(combo_parts(item["best"])[0] == combo_parts(item["actual"])[0] for item in strong) / len(strong)
    else:
        out["strong_pull_exact"] = None
        out["strong_pull_top1"] = None
    return out


def run(snapshots: Path, outcomes_path: Path) -> dict:
    counts = defaultdict(int)
    by_phase: dict[str, list[dict]] = defaultdict(list)
    phase_candidates: dict[tuple[tuple[str, str, str], str], dict] = {}
    archive_final_proxy: list[dict] = []
    outcome_lookup, outcome_records, outcome_conflicts = load_outcomes(outcomes_path)
    timed_snapshot_candidates = 0
    timed_without_outcome = 0
    post_start_with_outcome = 0
    unknown_phase_with_outcome = 0

    with snapshots.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            source = str(record.get("source") or "unknown")
            counts[f"source:{source}"] += 1
            best = best_board_combo(record.get("board") or {})
            actual = actual_order(record.get("actual_order"))
            actual_parts = combo_parts(actual)
            if not best:
                continue
            best_combo, best_odds, gap12 = best
            row = {"best": best_combo, "best_odds": best_odds, "gap12": gap12, "actual": actual}

            ts = snapshot_time(record)
            if source == "archive_import" and actual_parts:
                archive_final_proxy.append(row)
                continue
            if not ts:
                continue
            policy = engine._kcycle_market_timing_policy(ymd_of(record), record.get("race_no"), ts)
            counts[f"phase:{policy['phase']}"] += 1
            joined_actual = outcome_lookup.get(race_token(record))
            if not joined_actual:
                timed_without_outcome += 1
                continue
            if policy["phase"] == "post_start":
                post_start_with_outcome += 1
                continue
            if policy["phase"] not in {"early", "mid", "late"}:
                unknown_phase_with_outcome += 1
                continue
            timed_snapshot_candidates += 1
            row["actual"] = joined_actual
            row["snapshot_time"] = ts
            phase_key = (race_token(record), str(policy["phase"]))
            previous = phase_candidates.get(phase_key)
            if previous is None or str(previous["snapshot_time"]) < ts:
                phase_candidates[phase_key] = row

    for (_token, phase), row in phase_candidates.items():
        by_phase[phase].append(row)
    phases = {phase: finalize_metric(items) for phase, items in sorted(by_phase.items())}
    timed_with_outcome = sum(len(items) for items in by_phase.values())
    eligible_completed_races = len({token for token, _phase in phase_candidates})
    status = (
        "ready_for_walk_forward_validation"
        if eligible_completed_races >= 2000
        else "waiting_for_2000_timed_completed_races"
    )
    return {
        "status": status,
        "records": sum(v for k, v in counts.items() if k.startswith("source:")),
        "counts": dict(sorted(counts.items())),
        "timed_with_outcome": timed_with_outcome,
        "timed_snapshot_candidates": timed_snapshot_candidates,
        "timed_without_outcome": timed_without_outcome,
        "post_start_with_outcome": post_start_with_outcome,
        "unknown_phase_with_outcome": unknown_phase_with_outcome,
        "eligible_completed_races": eligible_completed_races,
        "outcome_join": {
            "path": str(outcomes_path),
            "records": outcome_records,
            "usable_races": len(outcome_lookup),
            "conflicting_races": outcome_conflicts,
        },
        "phase_metrics": phases,
        "archive_final_proxy": finalize_metric(archive_final_proxy),
        "policy": {
            "early": "30분 초과: 단승 5%, 삼쌍 축/강쏠림 미적용",
            "mid": "10~30분: 단승 15%, 삼쌍 축/강쏠림 미적용",
            "late": "10분 이내: 단승 30%, 삼쌍 축/강쏠림 적용 가능",
            "post_start": "시작 후: 예측 신호 차단",
        },
        "risk_flags": {
            "needs_live_timed_outcome_snapshots": timed_with_outcome == 0,
            "archive_import_is_final_market_proxy_only": True,
        },
        "promotion_gate": {
            "minimum_completed_races": 2000,
            "eligible_completed_races": eligible_completed_races,
            "met": eligible_completed_races >= 2000,
            "rule": "시점별 전광판은 별도 결과 파일과 안전 조인한 뒤, 2,000개 이상의 완료 경주에서만 워크포워드 검증 후보가 된다.",
        },
    }


def write_markdown(out: dict, path: Path) -> None:
    def pct(value):
        return "n/a" if value is None else f"{value * 100:.2f}%"

    lines = [
        "# KCYCLE market timing policy experiment",
        "",
        f"status: {out['status']}",
        f"records: {out['records']}",
        f"timed_with_outcome: {out['timed_with_outcome']}",
        f"eligible_completed_races: {out['eligible_completed_races']}",
        f"timed_without_outcome: {out['timed_without_outcome']}",
        f"post_start_with_outcome: {out['post_start_with_outcome']}",
        "",
        "## Policy",
    ]
    for key, value in out["policy"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Archive final-market proxy"])
    proxy = out["archive_final_proxy"]
    lines.append(
        f"- n={proxy['n']}, exact={pct(proxy['exact'])}, top1={pct(proxy['top1'])}, "
        f"strong_pull_n={proxy['strong_pull_n']}, strong_pull_exact={pct(proxy['strong_pull_exact'])}, "
        f"strong_pull_top1={pct(proxy['strong_pull_top1'])}"
    )
    lines.extend(["", "## Live timed phase metrics"])
    if not out["phase_metrics"]:
        lines.append("- waiting for outcome-linked timed live snapshots")
    for phase, metric in out["phase_metrics"].items():
        lines.append(
            f"- {phase}: n={metric['n']}, exact={pct(metric['exact'])}, top1={pct(metric['top1'])}, "
            f"strong_pull_n={metric['strong_pull_n']}, strong_pull_exact={pct(metric['strong_pull_exact'])}, "
            f"strong_pull_top1={pct(metric['strong_pull_top1'])}"
        )
    lines.extend(["", "## Counts"])
    for key, value in out["counts"].items():
        lines.append(f"- {key}: {value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshots", type=Path, default=ROOT / "data/kcycle_trifecta_snapshots.jsonl")
    parser.add_argument("--outcomes", type=Path, default=ROOT / "data/kcycle_outcomes.jsonl")
    parser.add_argument("--out-json", type=Path, default=ROOT / "data/kcycle_market_timing_policy_results.json")
    parser.add_argument("--out-md", type=Path, default=ROOT / "docs/kcycle_market_timing_policy_results.md")
    args = parser.parse_args()
    out = run(args.snapshots, args.outcomes)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(out, args.out_md)
    print(json.dumps({"event": "kcycle_market_timing_policy_done", "status": out["status"], "records": out["records"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
