#!/usr/bin/env python3
import argparse
from contextlib import closing
import datetime as dt
import json
import re
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SQLITE = ROOT / "data" / "strategy_arena.sqlite"
DEFAULT_SNAPSHOTS = ROOT / "data" / "kcycle_trifecta_snapshots.jsonl"
DEFAULT_KCYCLE_OUTCOMES = ROOT / "data" / "kcycle_outcomes.jsonl"
DEFAULT_KRA_DB = Path("/Users/tttksj/kra/data/kra.db")
DEFAULT_PRIORS = ROOT / "data" / "participant_learning_priors.json"
DEFAULT_SUMMARY = ROOT / "data" / "prediction_feedback_summary.json"

DEFAULT_DELTA_LIMIT = 0.08
DEFAULT_MIN_OOS_RACES = 30
DEFAULT_MIN_GRADE_POLICY_RECOMMENDATIONS = 3

DRUG_DISCOVERY_CANDIDATES = [
    {
        "name": "scaffold_hop_shallow",
        "family": "scaffold_hopping",
        "min_starts": 3,
        "alpha": 18.0,
        "weight": 0.15,
        "delta_limit": 0.04,
    },
    {
        "name": "bayesian_shrink_focus",
        "family": "bayesian_surrogate",
        "min_starts": 5,
        "alpha": 12.0,
        "weight": 0.25,
        "delta_limit": 0.06,
    },
    {
        "name": "multi_objective_funnel",
        "family": "multi_objective_funnel",
        "min_starts": 7,
        "alpha": 16.0,
        "weight": 0.20,
        "delta_limit": 0.05,
    },
    {
        "name": "activity_cliff_guard",
        "family": "activity_cliff_guard",
        "min_starts": 8,
        "alpha": 24.0,
        "weight": 0.15,
        "delta_limit": 0.03,
    },
    {
        "name": "potency_probe_guarded",
        "family": "lead_optimization",
        "min_starts": 4,
        "alpha": 8.0,
        "weight": 0.35,
        "delta_limit": 0.08,
    },
]

GRADE_POLICY_BASELINE = {
    "strong_pwin": 0.50,
    "mid_pwin": 0.30,
    "high_gap": 0.25,
    "mixed_gap": 0.08,
    "final_gap": 0.15,
}

GRADE_POLICY_BY_CLASS = {
    "특선": {
        "strong_pwin": 0.55,
        "mid_pwin": 0.35,
        "high_gap": 0.22,
        "mixed_gap": 0.10,
        "final_gap": 0.22,
    },
    "우수": {
        "strong_pwin": 0.50,
        "mid_pwin": 0.30,
        "high_gap": 0.16,
        "mixed_gap": 0.08,
        "final_gap": 0.16,
    },
    "선발": {
        "strong_pwin": 0.45,
        "mid_pwin": 0.28,
        "high_gap": 0.14,
        "mixed_gap": 0.06,
        "final_gap": 0.12,
    },
}


def _race_date(value):
    digits = re.sub(r"\D", "", str(value or ""))
    return digits[:8] if len(digits) >= 8 else ""


def _race_no(value):
    text = str(value or "").strip()
    try:
        return str(int(text))
    except ValueError:
        return text.lstrip("0") or text


def _race_key(sport, date, meet, race_no):
    return (str(sport or "").strip(), _race_date(date), str(meet or "").strip(), _race_no(race_no))


def _actual_order(value):
    numbers = [int(item) for item in re.findall(r"\d+", str(value or ""))]
    return numbers[:3] if len(numbers) >= 3 else []


def _empty_feedback(min_starts_for_live_adjustment, alpha, delta_limit=DEFAULT_DELTA_LIMIT):
    return {
        "version": 1,
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "enabled": False,
        "alpha": alpha,
        "delta_limit": delta_limit,
        "learning_weight": 0.0,
        "min_starts_for_live_adjustment": min_starts_for_live_adjustment,
        "summary": {
            "prediction_rows": 0,
            "raw_prediction_rows": 0,
            "duplicate_prediction_rows_ignored": 0,
            "outcome_rows": 0,
            "matched_races": 0,
            "top1_hits": 0,
            "exact_trifecta_hits": 0,
            "top1_hit_rate": None,
            "exact_trifecta_hit_rate": None,
            "grade_policy_enabled": False,
        },
        "sports": {"keirin": {"participants": {}}, "horse": {"participants": {}}},
    }


def _load_outcomes(snapshot_path):
    outcomes = {}
    if not Path(snapshot_path).exists():
        return outcomes
    with Path(snapshot_path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            actual = _actual_order(row.get("actual_order"))
            if not actual:
                continue
            key = _race_key("keirin", row.get("date"), row.get("meet"), row.get("race_no"))
            if key[1] and key[2] and key[3]:
                outcomes[key] = actual
    return outcomes


def _load_kcycle_outcomes(outcome_path):
    outcomes = {}
    if outcome_path is None or not Path(outcome_path).exists():
        return outcomes
    with Path(outcome_path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            actual = _actual_order(row.get("actual_order"))
            if not actual:
                continue
            key = _race_key("keirin", row.get("date"), row.get("meet"), row.get("race_no"))
            if key[1] and key[2] and key[3]:
                outcomes[key] = actual
    return outcomes


def _load_kra_outcomes(kra_db_path):
    if kra_db_path is None or not Path(kra_db_path).exists():
        return {}
    races = {}
    with closing(sqlite3.connect(kra_db_path)) as conn:
        try:
            rows = conn.execute(
                "SELECT meet, rcDate, rcNo, chulNo, ord FROM race_result "
                "WHERE ord IN ('1', '2', '3', 1, 2, 3)"
            ).fetchall()
        except sqlite3.Error:
            return {}
    for meet, rc_date, rc_no, chul_no, order in rows:
        key = _race_key("horse", rc_date, meet, rc_no)
        try:
            order_no = int(str(order).strip())
            runner_no = int(str(chul_no).strip())
        except ValueError:
            continue
        if order_no not in {1, 2, 3}:
            continue
        races.setdefault(key, {})[order_no] = runner_no
    return {key: [orders[1], orders[2], orders[3]] for key, orders in races.items() if {1, 2, 3} <= set(orders)}


def _load_prediction_rows(sqlite_path):
    if not Path(sqlite_path).exists():
        return []
    with closing(sqlite3.connect(sqlite_path)) as conn:
        try:
            rows = conn.execute(
                "SELECT id, sport, race_date, meet, race_no, payload_json, created_at "
                "FROM prediction__predictions"
            ).fetchall()
        except sqlite3.Error:
            return []
    loaded = []
    for row_id, sport, race_date, meet, race_no, payload_json, created_at in rows:
        key = _race_key(sport, race_date, meet, race_no)
        if not all(key):
            continue
        try:
            payload = json.loads(payload_json)
        except json.JSONDecodeError:
            continue
        loaded.append({
            "key": key,
            "marker": (str(created_at or ""), int(row_id or 0)),
            "payload": payload,
        })
    return loaded


def _load_latest_predictions(sqlite_path):
    latest = {}
    for row in _load_prediction_rows(sqlite_path):
        key = row["key"]
        current = latest.get(key)
        if current is None or row["marker"] > current["marker"]:
            latest[key] = {"marker": row["marker"], "payload": row["payload"]}
    return latest


def _participant_seed():
    return {
        "starts": 0,
        "wins": 0,
        "podiums": 0,
        "expected_win_sum": 0.0,
        "expected_podium_sum": 0.0,
    }


def _participant_name(row):
    name = str(row.get("name") or "").strip()
    return name or f"#{row.get('bno')}"


def _clamp(value, limit):
    return max(-limit, min(limit, value))


def _finish_participants(participants, global_win_rate, global_podium_rate, alpha, delta_limit=DEFAULT_DELTA_LIMIT):
    finished = {}
    for name in sorted(participants):
        row = participants[name]
        starts = int(row["starts"])
        if starts <= 0:
            continue
        expected_win_rate = float(row["expected_win_sum"]) / starts
        expected_podium_rate = float(row["expected_podium_sum"]) / starts
        win_rate = (int(row["wins"]) + alpha * global_win_rate) / (starts + alpha)
        podium_rate = (int(row["podiums"]) + alpha * global_podium_rate) / (starts + alpha)
        finished[name] = {
            "starts": starts,
            "wins": int(row["wins"]),
            "podiums": int(row["podiums"]),
            "win_rate_eb": round(win_rate, 6),
            "podium_rate_eb": round(podium_rate, 6),
            "expected_win_rate": round(expected_win_rate, 6),
            "expected_podium_rate": round(expected_podium_rate, 6),
            "win_delta": round(_clamp(win_rate - expected_win_rate, delta_limit), 6),
            "podium_delta": round(_clamp(podium_rate - expected_podium_rate, delta_limit), 6),
        }
    return finished


def _prediction_order(rows):
    order = []
    for row in rows:
        try:
            order.append(int(row.get("bno")))
        except (TypeError, ValueError):
            continue
    return order


def _probability(value, delta, weight):
    try:
        base = float(value)
    except (TypeError, ValueError):
        base = 0.0
    try:
        shift = float(delta) * float(weight)
    except (TypeError, ValueError):
        shift = 0.0
    return max(0.001, min(0.999, base + shift))


def _adjusted_order(rows, sport, priors_by_sport, candidate):
    participants = priors_by_sport.get(sport, {}).get("participants", {})
    min_starts = int(candidate.get("min_starts") or 5)
    weight = float(candidate.get("weight") or 0.0)
    adjusted = []
    for row in rows:
        out = dict(row)
        prior = participants.get(_participant_name(out))
        if isinstance(prior, dict) and int(prior.get("starts") or 0) >= min_starts:
            out["pwin"] = _probability(out.get("pwin"), prior.get("win_delta"), weight)
            out["pplc"] = _probability(out.get("pplc"), prior.get("podium_delta"), weight)
        adjusted.append(out)
    adjusted.sort(key=lambda item: (-float(item.get("pplc") or 0.0), -float(item.get("pwin") or 0.0)))
    return _prediction_order(adjusted)


def _update_participant_observations(participants, rows, actual):
    actual_top3 = set(actual[:3])
    totals = {"starts": 0, "wins": 0, "podiums": 0}
    for row in rows:
        try:
            bno = int(row.get("bno"))
        except (TypeError, ValueError):
            continue
        item = participants.setdefault(_participant_name(row), _participant_seed())
        item["starts"] += 1
        item["wins"] += int(bno == actual[0])
        item["podiums"] += int(bno in actual_top3)
        item["expected_win_sum"] += float(row.get("pwin") or 0.0)
        item["expected_podium_sum"] += float(row.get("pplc") or 0.0)
        totals["starts"] += 1
        totals["wins"] += int(bno == actual[0])
        totals["podiums"] += int(bno in actual_top3)
    return totals


def _matched_records(predictions, outcomes):
    records = []
    for key, prediction in predictions.items():
        actual = outcomes.get(key)
        if not actual:
            continue
        payload = prediction["payload"]
        rows = payload.get("rows") if isinstance(payload, dict) else None
        if not isinstance(rows, list) or not rows:
            continue
        predicted_order = _prediction_order(rows)
        if not predicted_order:
            continue
        records.append({"key": key, "actual": actual, "rows": rows, "baseline_order": predicted_order})
    records.sort(key=lambda row: (row["key"][1], row["key"][0], row["key"][2], row["key"][3]))
    return records


def _record_grade(rows):
    counts = {}
    for row in rows or []:
        grade = str(row.get("grade") or "").strip()
        if grade in GRADE_POLICY_BY_CLASS:
            counts[grade] = counts.get(grade, 0) + 1
    if not counts:
        return ""
    return max(counts.items(), key=lambda item: (item[1], item[0]))[0]


def _policy_for_record(rows, grade_aware):
    if not grade_aware:
        return GRADE_POLICY_BASELINE
    grade = _record_grade(rows)
    return GRADE_POLICY_BY_CLASS.get(grade) or GRADE_POLICY_BASELINE


def _float_value(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _grade_policy_recommends(rows, grade_aware):
    if not rows:
        return False
    top = rows[0]
    policy = _policy_for_record(rows, grade_aware)
    pwin = _float_value(top.get("pwin"))
    sorted_rows = sorted(rows, key=lambda row: -_float_value(row.get("pwin")))
    gap = 0.0
    if len(sorted_rows) >= 2:
        gap = _float_value(sorted_rows[0].get("pwin")) - _float_value(sorted_rows[1].get("pwin"))
    return pwin >= float(policy["strong_pwin"]) or gap >= float(policy["final_gap"])


def _grade_policy_score(records, grade_aware):
    score = {
        "policy": "grade_context" if grade_aware else "baseline",
        "races": 0,
        "recommended_races": 0,
        "top1_hits": 0,
        "exact_trifecta_hits": 0,
        "top1_precision": None,
        "exact_trifecta_precision": None,
        "coverage": 0.0,
        "grade_counts": {},
    }
    for record in records:
        if record["key"][0] != "keirin":
            continue
        rows = record["rows"]
        actual = record["actual"]
        order = record["baseline_order"]
        if not rows or not actual or not order:
            continue
        score["races"] += 1
        grade = _record_grade(rows) or "unknown"
        score["grade_counts"][grade] = score["grade_counts"].get(grade, 0) + 1
        if not _grade_policy_recommends(rows, grade_aware):
            continue
        score["recommended_races"] += 1
        score["top1_hits"] += int(order[:1] == actual[:1])
        score["exact_trifecta_hits"] += int(order[:3] == actual[:3])
    recommended = score["recommended_races"]
    races = score["races"]
    if recommended:
        score["top1_precision"] = score["top1_hits"] / recommended
        score["exact_trifecta_precision"] = score["exact_trifecta_hits"] / recommended
    score["coverage"] = recommended / races if races else 0.0
    return score


def evaluate_grade_policy(records, min_oos_races=DEFAULT_MIN_OOS_RACES, min_recommendations=DEFAULT_MIN_GRADE_POLICY_RECOMMENDATIONS):
    baseline = _grade_policy_score(records, grade_aware=False)
    candidate = _grade_policy_score(records, grade_aware=True)
    enough = baseline["races"] >= min_oos_races
    enough_recommendations = (
        baseline["recommended_races"] >= min_recommendations
        and candidate["recommended_races"] >= min_recommendations
    )
    baseline_top1 = baseline["top1_precision"]
    candidate_top1 = candidate["top1_precision"]
    baseline_exact = baseline["exact_trifecta_precision"]
    candidate_exact = candidate["exact_trifecta_precision"]
    top1_lift = None if baseline_top1 is None or candidate_top1 is None else candidate_top1 - baseline_top1
    exact_lift = None if baseline_exact is None or candidate_exact is None else candidate_exact - baseline_exact
    deployable = bool(
        enough
        and enough_recommendations
        and top1_lift is not None
        and exact_lift is not None
        and top1_lift > 0.0
        and exact_lift >= 0.0
    )
    if deployable:
        status = "deployable"
    elif not enough:
        status = "insufficient_matched_races"
    elif not enough_recommendations:
        status = "insufficient_recommended_races"
    else:
        status = "no_precision_improvement"
    return {
        "status": status,
        "deployable": deployable,
        "selected_policy": "grade_context" if deployable else "baseline",
        "min_oos_races": min_oos_races,
        "min_recommendations": min_recommendations,
        "baseline": baseline,
        "candidate": candidate,
        "top1_precision_lift": top1_lift,
        "exact_trifecta_precision_lift": exact_lift,
    }


def evaluate_learning_candidates(records, candidates=None, min_oos_races=DEFAULT_MIN_OOS_RACES):
    candidates = list(candidates or DRUG_DISCOVERY_CANDIDATES)
    baseline_top1 = 0
    baseline_exact = 0
    candidate_scores = {
        candidate["name"]: {
            "candidate": candidate,
            "top1_hits": 0,
            "exact_trifecta_hits": 0,
            "adjusted_races": 0,
        }
        for candidate in candidates
    }
    stats_by_candidate = {
        candidate["name"]: {
            "keirin": {"participants": {}, "totals": {"starts": 0, "wins": 0, "podiums": 0}},
            "horse": {"participants": {}, "totals": {"starts": 0, "wins": 0, "podiums": 0}},
        }
        for candidate in candidates
    }

    for record in records:
        actual = record["actual"]
        baseline = record["baseline_order"]
        baseline_top1 += int(baseline[:1] == actual[:1])
        baseline_exact += int(baseline[:3] == actual[:3])
        sport = record["key"][0]

        for candidate in candidates:
            state = stats_by_candidate[candidate["name"]]
            priors = {}
            for item_sport, payload in state.items():
                totals = payload["totals"]
                starts = totals["starts"]
                priors[item_sport] = {
                    "participants": _finish_participants(
                        payload["participants"],
                        totals["wins"] / starts if starts else 0.0,
                        totals["podiums"] / starts if starts else 0.0,
                        float(candidate.get("alpha") or 8.0),
                        float(candidate.get("delta_limit") or DEFAULT_DELTA_LIMIT),
                    )
                }
            order = _adjusted_order(record["rows"], sport, priors, candidate)
            score = candidate_scores[candidate["name"]]
            score["top1_hits"] += int(order[:1] == actual[:1])
            score["exact_trifecta_hits"] += int(order[:3] == actual[:3])
            score["adjusted_races"] += int(order != baseline)
            totals_delta = _update_participant_observations(state[sport]["participants"], record["rows"], actual)
            for field, value in totals_delta.items():
                state[sport]["totals"][field] += value

    n = len(records)
    baseline_summary = {
        "races": n,
        "top1_hits": baseline_top1,
        "exact_trifecta_hits": baseline_exact,
        "top1_hit_rate": baseline_top1 / n if n else None,
        "exact_trifecta_hit_rate": baseline_exact / n if n else None,
    }
    evaluated = []
    for score in candidate_scores.values():
        top1_rate = score["top1_hits"] / n if n else None
        exact_rate = score["exact_trifecta_hits"] / n if n else None
        baseline_top1_rate = baseline_summary["top1_hit_rate"] or 0.0
        baseline_exact_rate = baseline_summary["exact_trifecta_hit_rate"] or 0.0
        top1_lift = (top1_rate - baseline_top1_rate) if top1_rate is not None else None
        exact_lift = (exact_rate - baseline_exact_rate) if exact_rate is not None else None
        weighted_lift = ((top1_lift or 0.0) + 0.35 * (exact_lift or 0.0)) if n else None
        candidate = score["candidate"]
        evaluated.append({
            **candidate,
            "races": n,
            "top1_hits": score["top1_hits"],
            "exact_trifecta_hits": score["exact_trifecta_hits"],
            "top1_hit_rate": top1_rate,
            "exact_trifecta_hit_rate": exact_rate,
            "top1_lift": top1_lift,
            "exact_trifecta_lift": exact_lift,
            "weighted_lift": weighted_lift,
            "adjusted_races": score["adjusted_races"],
        })
    evaluated.sort(key=lambda item: (item["weighted_lift"] or -999.0, item["top1_lift"] or -999.0), reverse=True)
    best = evaluated[0] if evaluated else None
    enough = n >= min_oos_races
    improved = bool(
        enough
        and best
        and (best.get("weighted_lift") or 0.0) > 0.0
        and (best.get("top1_lift") or 0.0) >= 0.0
        and (best.get("exact_trifecta_lift") or 0.0) >= 0.0
    )
    status = "deployable" if improved else ("insufficient_matched_races" if not enough else "no_oos_improvement")
    return {
        "status": status,
        "deployable": improved,
        "min_oos_races": min_oos_races,
        "baseline": baseline_summary,
        "best_candidate": best,
        "candidates": evaluated,
    }


def build_feedback(
    sqlite_path,
    snapshot_path,
    kcycle_outcome_path=DEFAULT_KCYCLE_OUTCOMES,
    kra_db_path=None,
    min_starts_for_live_adjustment=5,
    alpha=8.0,
    min_oos_races=DEFAULT_MIN_OOS_RACES,
    exclude_keys=None,
):
    result = _empty_feedback(min_starts_for_live_adjustment, alpha)
    outcomes = _load_outcomes(snapshot_path)
    outcomes.update(_load_kcycle_outcomes(kcycle_outcome_path))
    outcomes.update(_load_kra_outcomes(kra_db_path))
    raw_predictions = _load_prediction_rows(sqlite_path)
    predictions = _load_latest_predictions(sqlite_path)
    excluded = {
        _race_key(key[0], key[1], key[2], key[3]) if isinstance(key, (list, tuple)) and len(key) >= 4 else key
        for key in (exclude_keys or set())
    }
    if excluded:
        predictions = {key: value for key, value in predictions.items() if key not in excluded}
        outcomes = {key: value for key, value in outcomes.items() if key not in excluded}
    result["summary"]["prediction_rows"] = len(predictions)
    result["summary"]["raw_prediction_rows"] = len(raw_predictions)
    result["summary"]["duplicate_prediction_rows_ignored"] = max(0, len(raw_predictions) - len(predictions))
    result["summary"]["outcome_rows"] = len(outcomes)
    result["summary"]["excluded_roster_mismatch_races"] = len(excluded)

    participants_by_sport = {"keirin": {}, "horse": {}}
    matched_races = 0
    top1_hits = 0
    exact_trifecta_hits = 0
    total_starts = 0
    total_wins = 0
    total_podiums = 0

    records = _matched_records(predictions, outcomes)
    oos_validation = evaluate_learning_candidates(records, min_oos_races=min_oos_races)
    grade_policy_validation = evaluate_grade_policy(records, min_oos_races=min_oos_races)
    best = oos_validation.get("best_candidate") or {}
    if oos_validation.get("deployable"):
        result["enabled"] = True
        result["alpha"] = float(best.get("alpha") or alpha)
        result["delta_limit"] = float(best.get("delta_limit") or DEFAULT_DELTA_LIMIT)
        result["learning_weight"] = float(best.get("weight") or 0.0)
        result["min_starts_for_live_adjustment"] = int(best.get("min_starts") or min_starts_for_live_adjustment)

    for record in records:
        key = record["key"]
        actual = record["actual"]
        rows = record["rows"]
        sport = key[0]
        participants = participants_by_sport.setdefault(sport, {})
        predicted_order = record["baseline_order"]
        if not predicted_order:
            continue
        matched_races += 1
        top1_hits += int(predicted_order[0] == actual[0])
        exact_trifecta_hits += int(predicted_order[:3] == actual[:3])
        actual_top3 = set(actual[:3])
        for row in rows:
            try:
                bno = int(row.get("bno"))
            except (TypeError, ValueError):
                continue
            name = _participant_name(row)
            item = participants.setdefault(name, _participant_seed())
            item["starts"] += 1
            item["wins"] += int(bno == actual[0])
            item["podiums"] += int(bno in actual_top3)
            item["expected_win_sum"] += float(row.get("pwin") or 0.0)
            item["expected_podium_sum"] += float(row.get("pplc") or 0.0)
            total_starts += 1
            total_wins += int(bno == actual[0])
            total_podiums += int(bno in actual_top3)

    global_win_rate = total_wins / total_starts if total_starts else 0.0
    global_podium_rate = total_podiums / total_starts if total_starts else 0.0
    for sport, participants in participants_by_sport.items():
        result["sports"].setdefault(sport, {})["participants"] = _finish_participants(
            participants,
            global_win_rate,
            global_podium_rate,
            float(result["alpha"]),
            float(result["delta_limit"]),
        )

    result["summary"].update({
        "matched_races": matched_races,
        "top1_hits": top1_hits,
        "exact_trifecta_hits": exact_trifecta_hits,
        "top1_hit_rate": top1_hits / matched_races if matched_races else None,
        "exact_trifecta_hit_rate": exact_trifecta_hits / matched_races if matched_races else None,
        "participant_starts": total_starts,
        "global_win_rate": global_win_rate,
        "global_podium_rate": global_podium_rate,
    })
    result["oos_validation"] = oos_validation
    result["grade_policy_validation"] = grade_policy_validation
    result["summary"]["learning_enabled"] = bool(result["enabled"])
    result["summary"]["oos_status"] = oos_validation["status"]
    result["summary"]["learning_weight"] = result["learning_weight"]
    result["summary"]["grade_policy_enabled"] = bool(grade_policy_validation.get("deployable"))
    result["summary"]["grade_policy_status"] = grade_policy_validation.get("status")
    return result


def write_feedback(result, priors_path, summary_path):
    Path(priors_path).parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    Path(priors_path).write_text(text + "\n", encoding="utf-8")
    summary = {
        "generated_at": result["generated_at"],
        "alpha": result["alpha"],
        "delta_limit": result.get("delta_limit"),
        "enabled": result.get("enabled"),
        "learning_weight": result.get("learning_weight"),
        "min_starts_for_live_adjustment": result["min_starts_for_live_adjustment"],
        "oos_validation": result.get("oos_validation"),
        "grade_policy_validation": result.get("grade_policy_validation"),
        **result["summary"],
    }
    Path(summary_path).write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sqlite", default=str(DEFAULT_SQLITE))
    parser.add_argument("--snapshots", default=str(DEFAULT_SNAPSHOTS))
    parser.add_argument("--kcycle-outcomes", default=str(DEFAULT_KCYCLE_OUTCOMES))
    parser.add_argument("--kra-db", default=str(DEFAULT_KRA_DB))
    parser.add_argument("--out-priors", default=str(DEFAULT_PRIORS))
    parser.add_argument("--out-summary", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--min-starts", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=8.0)
    parser.add_argument("--min-oos-races", type=int, default=DEFAULT_MIN_OOS_RACES)
    args = parser.parse_args()
    result = build_feedback(
        Path(args.sqlite),
        Path(args.snapshots),
        kcycle_outcome_path=Path(args.kcycle_outcomes) if args.kcycle_outcomes else None,
        kra_db_path=Path(args.kra_db) if args.kra_db else None,
        min_starts_for_live_adjustment=args.min_starts,
        alpha=args.alpha,
        min_oos_races=args.min_oos_races,
    )
    write_feedback(result, Path(args.out_priors), Path(args.out_summary))
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
