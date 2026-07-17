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


def run(snapshots: Path) -> dict:
    counts = defaultdict(int)
    by_phase: dict[str, list[dict]] = defaultdict(list)
    archive_final_proxy: list[dict] = []
    timed_with_outcome = 0
    timed_without_outcome = 0

    with snapshots.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            source = str(record.get("source") or "unknown")
            counts[f"source:{source}"] += 1
            best = best_board_combo(record.get("board") or {})
            actual = str(record.get("actual_order") or "")
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
            if actual_parts:
                timed_with_outcome += 1
                by_phase[str(policy["phase"])].append(row)
            else:
                timed_without_outcome += 1

    phases = {phase: finalize_metric(items) for phase, items in sorted(by_phase.items())}
    status = "ok" if timed_with_outcome else "waiting_for_outcome_linked_timed_snapshots"
    return {
        "status": status,
        "records": sum(v for k, v in counts.items() if k.startswith("source:")),
        "counts": dict(sorted(counts.items())),
        "timed_with_outcome": timed_with_outcome,
        "timed_without_outcome": timed_without_outcome,
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
        f"timed_without_outcome: {out['timed_without_outcome']}",
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
    parser.add_argument("--out-json", type=Path, default=ROOT / "data/kcycle_market_timing_policy_results.json")
    parser.add_argument("--out-md", type=Path, default=ROOT / "docs/kcycle_market_timing_policy_results.md")
    args = parser.parse_args()
    out = run(args.snapshots)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(out, args.out_md)
    print(json.dumps({"event": "kcycle_market_timing_policy_done", "status": out["status"], "records": out["records"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
