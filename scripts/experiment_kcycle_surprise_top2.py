#!/usr/bin/env python3
import argparse
from contextlib import closing
import json
import math
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SNAPSHOTS = ROOT / "data" / "kcycle_trifecta_snapshots.jsonl"
DEFAULT_SQLITE = ROOT / "data" / "strategy_arena.sqlite"
DEFAULT_OUTCOMES = ROOT / "data" / "kcycle_outcomes.jsonl"
DEFAULT_OUT_JSON = ROOT / "data" / "kcycle_surprise_top2_results.json"
DEFAULT_OUT_MD = ROOT / "docs" / "kcycle_surprise_top2_results.md"


def _race_date(value):
    digits = re.sub(r"\D", "", str(value or ""))
    return digits[:8] if len(digits) >= 8 else ""


def _race_no(value):
    text = str(value or "").strip()
    try:
        return str(int(text))
    except ValueError:
        return text.lstrip("0") or text


def _actual_order(value):
    numbers = [int(item) for item in re.findall(r"\d+", str(value or ""))]
    return numbers[:3] if len(numbers) >= 3 else []


def _combo_numbers(value):
    numbers = [int(item) for item in re.findall(r"\d+", str(value or ""))]
    return numbers[:3] if len(numbers) >= 2 else []


def _safe_float(value):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric) or numeric <= 0:
        return None
    return numeric


def _runner_market_scores(board):
    scores = defaultdict(float)
    min_top2_odds = {}
    combo_counts = Counter()
    for selection, raw_odds in (board or {}).items():
        odds = _safe_float(raw_odds)
        numbers = _combo_numbers(selection)
        if odds is None or len(numbers) < 2:
            continue
        weight = 1.0 / odds
        for runner in numbers[:2]:
            scores[runner] += weight
            combo_counts[runner] += 1
            current = min_top2_odds.get(runner)
            if current is None or odds < current:
                min_top2_odds[runner] = odds
    return {runner: float(score) for runner, score in scores.items()}, min_top2_odds, combo_counts


def _market_rank_map(scores):
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return {runner: index + 1 for index, (runner, _) in enumerate(ordered)}


def _best20_top2_presence(row):
    present = Counter()
    for item in row.get("best20") or []:
        selection = item[0] if isinstance(item, (list, tuple)) and item else item
        for runner in _combo_numbers(selection)[:2]:
            present[runner] += 1
    return present


def _snapshot_records(snapshot_path):
    records = []
    path = Path(snapshot_path)
    if not path.exists():
        return records
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            actual = _actual_order(row.get("actual_order"))
            board = row.get("board")
            if len(actual) < 2 or not isinstance(board, dict):
                continue
            scores, min_top2_odds, combo_counts = _runner_market_scores(board)
            if len(scores) < 3:
                continue
            key = (
                "keirin",
                _race_date(row.get("date")),
                str(row.get("meet") or "").strip(),
                _race_no(row.get("race_no")),
            )
            if not all(key):
                continue
            records.append({
                "key": key,
                "actual_top2": actual[:2],
                "actual_order": actual,
                "scores": scores,
                "min_top2_odds": min_top2_odds,
                "combo_counts": combo_counts,
                "best20_top2_presence": _best20_top2_presence(row),
            })
    records.sort(key=lambda item: (item["key"][1], item["key"][2], int(item["key"][3]) if item["key"][3].isdigit() else item["key"][3]))
    return records


def _load_kcycle_outcomes(outcome_path):
    outcomes = {}
    path = Path(outcome_path)
    if not path.exists():
        return outcomes
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            actual = _actual_order(row.get("actual_order"))
            if len(actual) < 2:
                continue
            key = (
                "keirin",
                _race_date(row.get("date")),
                str(row.get("meet") or "").strip(),
                _race_no(row.get("race_no")),
            )
            if all(key):
                outcomes[key] = actual
    return outcomes


def _row_score(row):
    score = _safe_float(row.get("pplc"))
    if score is not None:
        return score
    score = _safe_float(row.get("pwin"))
    return score if score is not None else 0.0


def _prediction_records(sqlite_path, outcome_path):
    db_path = Path(sqlite_path)
    if not db_path.exists():
        return []
    outcomes = _load_kcycle_outcomes(outcome_path)
    latest = {}
    with closing(sqlite3.connect(db_path)) as conn:
        try:
            rows = conn.execute(
                "SELECT id, sport, race_date, meet, race_no, payload_json, created_at "
                "FROM prediction__predictions WHERE sport = 'keirin'"
            ).fetchall()
        except sqlite3.Error:
            return []
    for row_id, sport, race_date, meet, race_no, payload_json, created_at in rows:
        key = (str(sport), _race_date(race_date), str(meet or "").strip(), _race_no(race_no))
        actual = outcomes.get(key)
        if not actual:
            continue
        try:
            payload = json.loads(payload_json)
        except json.JSONDecodeError:
            continue
        participant_rows = payload.get("rows") if isinstance(payload, dict) else None
        if not isinstance(participant_rows, list):
            continue
        scores = {}
        for participant in participant_rows:
            if not isinstance(participant, dict):
                continue
            try:
                runner = int(participant.get("bno") or participant.get("number"))
            except (TypeError, ValueError):
                continue
            scores[runner] = _row_score(participant)
        if len(scores) < 3:
            continue
        marker = (str(created_at or ""), int(row_id or 0))
        current = latest.get(key)
        if current is None or marker > current["marker"]:
            latest[key] = {
                "marker": marker,
                "key": key,
                "actual_top2": actual[:2],
                "actual_order": actual,
                "scores": scores,
                "min_top2_odds": {},
                "combo_counts": {},
                "best20_top2_presence": Counter(),
            }
    records = list(latest.values())
    records.sort(key=lambda item: (item["key"][1], item["key"][2], int(item["key"][3]) if item["key"][3].isdigit() else item["key"][3]))
    return records


def _top_n(scores, n=2):
    return [runner for runner, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:n]]


def _bucket_rank(rank):
    if rank <= 2:
        return "rank_1_2"
    if rank <= 4:
        return "rank_3_4"
    return "rank_5_plus"


def _surprise_rows(records, surprise_rank_min):
    surprises = []
    total_slots = 0
    baseline_hits = 0
    for record in records:
        ranks = _market_rank_map(record["scores"])
        baseline_top2 = set(_top_n(record["scores"], 2))
        actual_top2 = record["actual_top2"]
        total_slots += len(actual_top2)
        baseline_hits += len(set(actual_top2) & baseline_top2)
        for position, runner in enumerate(actual_top2, start=1):
            rank = int(ranks.get(runner, 99))
            if rank < surprise_rank_min:
                continue
            surprises.append({
                "date": record["key"][1],
                "meet": record["key"][2],
                "race_no": record["key"][3],
                "runner": runner,
                "actual_position": position,
                "market_rank": rank,
                "rank_bucket": _bucket_rank(rank),
                "market_score": round(float(record["scores"].get(runner, 0.0)), 10),
                "min_top2_odds": record["min_top2_odds"].get(runner),
                "best20_top2_count": int(record["best20_top2_presence"].get(runner, 0)),
                "top2_combo_count": int(record["combo_counts"].get(runner, 0)),
            })
    return surprises, total_slots, baseline_hits


def _common_gates(surprises):
    counts = Counter(item["runner"] for item in surprises)
    return [
        {
            "gate": gate,
            "count": count,
            "share": count / len(surprises) if surprises else 0.0,
        }
        for gate, count in counts.most_common()
    ]


def _rank_buckets(surprises):
    counts = Counter(item["rank_bucket"] for item in surprises)
    return {key: counts.get(key, 0) for key in ["rank_3_4", "rank_5_plus"]}


def _odds_bucket(value):
    odds = _safe_float(value)
    if odds is None:
        return "unknown"
    if odds <= 20:
        return "odds_le_20"
    if odds <= 50:
        return "odds_20_50"
    if odds <= 100:
        return "odds_50_100"
    return "odds_gt_100"


def _surprise_diagnostics(surprises):
    position_counts = Counter(item["actual_position"] for item in surprises)
    rank_counts = Counter(item["market_rank"] for item in surprises)
    odds_counts = Counter(_odds_bucket(item.get("min_top2_odds")) for item in surprises)
    best20_absent = sum(1 for item in surprises if int(item.get("best20_top2_count") or 0) == 0)
    total = len(surprises)
    return {
        "actual_position_counts": {str(key): position_counts.get(key, 0) for key in [1, 2]},
        "market_rank_counts": {str(key): count for key, count in sorted(rank_counts.items())},
        "min_top2_odds_buckets": dict(sorted(odds_counts.items())),
        "best20_top2_absent_count": best20_absent,
        "best20_top2_absent_rate": best20_absent / total if total else None,
    }


def analyze_snapshot_surprises(snapshot_path, surprise_rank_min=4):
    records = _snapshot_records(snapshot_path)
    surprises, actual_top2_slots, baseline_hits = _surprise_rows(records, surprise_rank_min)
    return {
        "summary": {
            "source": "snapshot_market_proxy",
            "races": len(records),
            "actual_top2_slots": actual_top2_slots,
            "baseline_top2_slots_hit": baseline_hits,
            "baseline_top2_slot_hit_rate": baseline_hits / actual_top2_slots if actual_top2_slots else None,
            "surprise_rank_min": surprise_rank_min,
            "surprise_top2_count": len(surprises),
            "surprise_top2_rate": len(surprises) / actual_top2_slots if actual_top2_slots else None,
        },
        "common_gates": _common_gates(surprises),
        "rank_buckets": _rank_buckets(surprises),
        "diagnostics": _surprise_diagnostics(surprises),
        "surprises": surprises[:200],
        "records": records,
    }


def analyze_prediction_surprises(sqlite_path, outcome_path, surprise_rank_min=4):
    records = _prediction_records(sqlite_path, outcome_path)
    surprises, actual_top2_slots, baseline_hits = _surprise_rows(records, surprise_rank_min)
    return {
        "summary": {
            "source": "prediction_model",
            "races": len(records),
            "actual_top2_slots": actual_top2_slots,
            "baseline_top2_slots_hit": baseline_hits,
            "baseline_top2_slot_hit_rate": baseline_hits / actual_top2_slots if actual_top2_slots else None,
            "surprise_rank_min": surprise_rank_min,
            "surprise_top2_count": len(surprises),
            "surprise_top2_rate": len(surprises) / actual_top2_slots if actual_top2_slots else None,
        },
        "common_gates": _common_gates(surprises),
        "rank_buckets": _rank_buckets(surprises),
        "diagnostics": _surprise_diagnostics(surprises),
        "surprises": surprises[:200],
        "records": records,
    }


def _prior_gate_multiplier(gate_state, gate, global_surprise_rate, min_gate_starts, boost_weight):
    state = gate_state.get(gate)
    if not state or state["actual_top2"] < min_gate_starts:
        return 1.0
    gate_rate = state["surprise_top2"] / state["actual_top2"] if state["actual_top2"] else 0.0
    lift = max(0.0, gate_rate)
    if gate_rate <= global_surprise_rate and state["actual_top2"] < min_gate_starts * 2:
        return 1.0
    return 1.0 + boost_weight * lift


def evaluate_walk_forward_gate_boost(records, surprise_rank_min=4, min_gate_starts=8, boost_weight=0.75):
    gate_state = defaultdict(lambda: {"actual_top2": 0, "surprise_top2": 0})
    baseline_hits = 0
    candidate_hits = 0
    total_slots = 0
    adjusted_races = 0
    prior_actual_top2 = 0
    prior_surprise_top2 = 0
    examples = []

    for record in records:
        scores = dict(record["scores"])
        ranks = _market_rank_map(scores)
        actual_top2 = list(record["actual_top2"])
        baseline_top2 = _top_n(scores, 2)
        global_surprise_rate = prior_surprise_top2 / prior_actual_top2 if prior_actual_top2 else 0.0
        adjusted_scores = {
            gate: score * _prior_gate_multiplier(gate_state, gate, global_surprise_rate, min_gate_starts, boost_weight)
            for gate, score in scores.items()
        }
        candidate_top2 = _top_n(adjusted_scores, 2)

        baseline_hits += len(set(actual_top2) & set(baseline_top2))
        candidate_hits += len(set(actual_top2) & set(candidate_top2))
        total_slots += len(actual_top2)
        adjusted_races += int(candidate_top2 != baseline_top2)
        if candidate_top2 != baseline_top2 and len(examples) < 20:
            examples.append({
                "date": record["key"][1],
                "race_no": record["key"][3],
                "actual_top2": actual_top2,
                "baseline_top2": baseline_top2,
                "candidate_top2": candidate_top2,
            })

        for runner in actual_top2:
            rank = int(ranks.get(runner, 99))
            state = gate_state[runner]
            state["actual_top2"] += 1
            state["surprise_top2"] += int(rank >= surprise_rank_min)
            prior_actual_top2 += 1
            prior_surprise_top2 += int(rank >= surprise_rank_min)

    return {
        "policy": "walk_forward_gate_surprise_boost",
        "races": len(records),
        "actual_top2_slots": total_slots,
        "surprise_rank_min": surprise_rank_min,
        "min_gate_starts": min_gate_starts,
        "boost_weight": boost_weight,
        "baseline_top2_slots_hit": baseline_hits,
        "candidate_top2_slots_hit": candidate_hits,
        "baseline_top2_slot_hit_rate": baseline_hits / total_slots if total_slots else None,
        "candidate_top2_slot_hit_rate": candidate_hits / total_slots if total_slots else None,
        "slot_hit_lift": (candidate_hits - baseline_hits) / total_slots if total_slots else None,
        "adjusted_races": adjusted_races,
        "examples": examples,
    }


def _markdown_report(result):
    summary = result["summary"]
    boost = result["walk_forward_gate_boost"]
    diagnostics = result["diagnostics"]
    prediction_model = result.get("prediction_model")
    gates = result["common_gates"][:7]
    lines = [
        "# KCYCLE surprise top2 experiment",
        "",
        "## Summary",
        f"- source: {summary['source']}",
        f"- races: {summary['races']}",
        f"- baseline_top2_slot_hit_rate: {summary['baseline_top2_slot_hit_rate']}",
        f"- surprise_top2_count: {summary['surprise_top2_count']}",
        f"- surprise_top2_rate: {summary['surprise_top2_rate']}",
        "",
        "## Common surprise gates",
    ]
    for row in gates:
        lines.append(f"- gate {row['gate']}: count={row['count']} share={row['share']:.4f}")
    lines.extend([
        "",
        "## Diagnostics",
        f"- actual_position_counts: {diagnostics['actual_position_counts']}",
        f"- min_top2_odds_buckets: {diagnostics['min_top2_odds_buckets']}",
        f"- best20_top2_absent_rate: {diagnostics['best20_top2_absent_rate']}",
        "",
        "## Walk-forward gate boost",
        f"- baseline_top2_slot_hit_rate: {boost['baseline_top2_slot_hit_rate']}",
        f"- candidate_top2_slot_hit_rate: {boost['candidate_top2_slot_hit_rate']}",
        f"- slot_hit_lift: {boost['slot_hit_lift']}",
        f"- adjusted_races: {boost['adjusted_races']}",
        "",
        "## Interpretation",
    ])
    if boost["slot_hit_lift"] and boost["slot_hit_lift"] > 0.005:
        lines.append("- Gate surprise boost produced a material top2 slot recovery lift in this walk-forward proxy test.")
    elif boost["slot_hit_lift"] and boost["slot_hit_lift"] > 0:
        lines.append("- Gate surprise boost improved top2 slot recovery only marginally; treat it as a diagnostic, not a deployable rule.")
    else:
        lines.append("- Gate-only surprise boost did not improve top2 slot recovery; use cohort diagnostics, not deployment.")
    if prediction_model:
        model_summary = prediction_model["summary"]
        model_boost = prediction_model["walk_forward_gate_boost"]
        lines.extend([
            "",
            "## Prediction-model cohort",
            f"- races: {model_summary['races']}",
            f"- baseline_top2_slot_hit_rate: {model_summary['baseline_top2_slot_hit_rate']}",
            f"- surprise_top2_count: {model_summary['surprise_top2_count']}",
            f"- surprise_top2_rate: {model_summary['surprise_top2_rate']}",
            f"- gate_boost_slot_hit_lift: {model_boost['slot_hit_lift']}",
        ])
    return "\n".join(lines) + "\n"


def run(
    snapshot_path=DEFAULT_SNAPSHOTS,
    surprise_rank_min=4,
    min_gate_starts=8,
    boost_weight=0.75,
    sqlite_path=None,
    outcome_path=None,
):
    analysis = analyze_snapshot_surprises(snapshot_path, surprise_rank_min=surprise_rank_min)
    records = analysis.pop("records")
    boost = evaluate_walk_forward_gate_boost(
        records,
        surprise_rank_min=surprise_rank_min,
        min_gate_starts=min_gate_starts,
        boost_weight=boost_weight,
    )
    analysis["walk_forward_gate_boost"] = boost
    if sqlite_path and outcome_path:
        prediction = analyze_prediction_surprises(sqlite_path, outcome_path, surprise_rank_min=surprise_rank_min)
        prediction_records = prediction.pop("records")
        prediction["walk_forward_gate_boost"] = evaluate_walk_forward_gate_boost(
            prediction_records,
            surprise_rank_min=surprise_rank_min,
            min_gate_starts=min_gate_starts,
            boost_weight=boost_weight,
        )
        analysis["prediction_model"] = prediction
    return analysis


def main():
    parser = argparse.ArgumentParser(description="Find and test KCYCLE unexpected top2 runner patterns.")
    parser.add_argument("--snapshots", default=str(DEFAULT_SNAPSHOTS))
    parser.add_argument("--sqlite", default=str(DEFAULT_SQLITE))
    parser.add_argument("--outcomes", default=str(DEFAULT_OUTCOMES))
    parser.add_argument("--skip-prediction-model", action="store_true")
    parser.add_argument("--surprise-rank-min", type=int, default=4)
    parser.add_argument("--min-gate-starts", type=int, default=8)
    parser.add_argument("--boost-weight", type=float, default=0.75)
    parser.add_argument("--out-json", default=str(DEFAULT_OUT_JSON))
    parser.add_argument("--out-md", default=str(DEFAULT_OUT_MD))
    args = parser.parse_args()

    result = run(
        snapshot_path=Path(args.snapshots),
        surprise_rank_min=args.surprise_rank_min,
        min_gate_starts=args.min_gate_starts,
        boost_weight=args.boost_weight,
        sqlite_path=None if args.skip_prediction_model else Path(args.sqlite),
        outcome_path=None if args.skip_prediction_model else Path(args.outcomes),
    )
    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(_markdown_report(result), encoding="utf-8")
    print(json.dumps({
        "out_json": str(out_json),
        "out_md": str(out_md),
        "summary": result["summary"],
        "walk_forward_gate_boost": result["walk_forward_gate_boost"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
