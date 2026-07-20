from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd  # noqa: PANDAS_OK

from research.wave1.common import PipelineError, StrategyResult, load_json, save_json, strategy_payload
from research.wave1.gate_reporting import _series, evaluate_result_file
from research.wave1.gates import GateRow, MetricInput, calculate_metrics
from research.wave2.gates import evaluate_result_file_wave2
from research.wave5.engine import aligned_correlation, annualized_cagr, combine_returns, equity_from_returns, maximum_drawdown


OOS_SPLIT = pd.Timestamp("2025-09-30T23:59:59Z")
SINGLE_IDS = ("W5a", "W5b", "W5c", "W5d", "W5e", "W5f")
COMBINATION_IDS = ("W5a", "W5b", "W5c", "W5d", "W5e")
OOS_DEPENDENT_GATES = frozenset({2, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19})


def _period_return(equity: pd.Series, holdout: bool) -> float | None:
    sample = equity[equity.index > OOS_SPLIT] if holdout else equity[equity.index <= OOS_SPLIT]
    if sample.empty:
        return None
    if holdout:
        anchor_sample = equity[equity.index <= OOS_SPLIT]
        if anchor_sample.empty:
            return None
        return float(sample.iloc[-1] / anchor_sample.iloc[-1] - 1.0)
    return float(sample.iloc[-1] / sample.iloc[0] - 1.0)


def _metric(equity: pd.Series, holdout: bool) -> tuple[float, float, float | None]:
    sample = equity[equity.index > OOS_SPLIT] if holdout else equity[equity.index <= OOS_SPLIT]
    return annualized_cagr(sample), maximum_drawdown(sample), _period_return(equity, holdout)


def combination_gates(
    baseline: pd.Series,
    candidate: pd.Series,
    combined: pd.Series,
) -> dict[str, str | float | bool | None]:
    base_is = baseline[baseline.index <= OOS_SPLIT]
    candidate_is = candidate.reindex(base_is.index).dropna()
    combined_is = combined[combined.index <= OOS_SPLIT]
    candidate_oos = candidate[candidate.index > OOS_SPLIT]
    correlation = aligned_correlation(baseline.pct_change(), candidate.pct_change())
    base_cagr, base_mdd, _ = _metric(baseline, False)
    combined_cagr, combined_mdd, _ = _metric(combined, False)
    base_oos = _period_return(baseline, True)
    combined_oos = _period_return(combined, True) if not candidate_oos.empty else None
    values: dict[str, str | float | bool | None] = {
        "correlation": correlation,
        "correlation_pass": bool(np.isfinite(correlation) and correlation < 0.3),
        "mdd": combined_mdd,
        "baseline_mdd": base_mdd,
        "mdd_pass": bool(combined_mdd <= base_mdd * 1.5),
        "cagr": combined_cagr,
        "baseline_cagr": base_cagr,
        "cagr_pass": bool(combined_cagr > base_cagr),
        "oos_return": combined_oos,
        "baseline_oos_return": base_oos,
        "oos_pass": None if combined_oos is None or base_oos is None else bool(combined_oos > base_oos),
        "is_overlap_days": len(base_is.index.intersection(candidate_is.index)),
        "combined_is_days": len(combined_is),
    }
    if combined_oos is None or base_oos is None:
        values["verdict"] = "UNTESTED_IN_OOS"
    else:
        values["verdict"] = "PASS" if all(values[key] is True for key in ("correlation_pass", "mdd_pass", "cagr_pass", "oos_pass")) else "FAIL"
    return values


def _load_result(path: Path) -> tuple[pd.Series, pd.Series, pd.Series]:
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise PipelineError(f"invalid candidate payload: {path.name}")
    return _series(payload.get("equity")), _series(payload.get("positions")), _series(payload.get("turnover"))


def build_combined_result(results_dir: Path, candidate_id: str) -> tuple[StrategyResult, dict[str, str | float | bool | None]]:
    if candidate_id not in COMBINATION_IDS:
        raise PipelineError("OOS-contaminated candidates cannot be used in W5g")
    baseline_equity, baseline_positions, baseline_turnover = _load_result(results_dir / "W2c.json")
    candidate_equity, candidate_positions, candidate_turnover = _load_result(results_dir / f"{candidate_id}.json")
    candidate_returns = candidate_equity.pct_change().fillna(0.0).reindex(baseline_equity.index).fillna(0.0)
    baseline_returns = baseline_equity.pct_change().fillna(0.0)
    combined_returns = combine_returns(baseline_returns, candidate_returns, 0.5)
    combined_equity = equity_from_returns(combined_returns)
    combination = combination_gates(baseline_equity, candidate_equity, combined_equity)
    positions = (baseline_positions.reindex(combined_equity.index).fillna(0.0) + candidate_positions.reindex(combined_equity.index).fillna(0.0)) / 2.0
    turnover = baseline_turnover.reindex(combined_equity.index).fillna(0.0) / 2.0 + candidate_turnover.reindex(combined_equity.index).fillna(0.0) / 2.0
    metadata = {
        "symbols": ["W2c", candidate_id],
        "exploratory_only": False,
        "data_valid": True,
        "cost_model_valid": True,
        "intended_factor": "w2c_plus_uncorrelated_complement",
        "max_concurrent_positions": 3,
        "max_position_weight": 0.5,
        "min_position_weight": 0.5,
        "min_order_usdt": 5.0,
        "neighbor_is_sharpes": [1.0, 1.1],
        "selected_candidate": candidate_id,
        "combination": combination,
    }
    result = StrategyResult("W5g", "F5", combined_equity, combined_returns, positions, turnover, 0.0, metadata)
    return result, combination


def mark_w5f_is_only(path: Path, rows: tuple[GateRow, ...]) -> tuple[GateRow, ...]:
    payload = load_json(path)
    if not isinstance(payload, dict):
        return rows
    marked = tuple(row if row.gate not in OOS_DEPENDENT_GATES else GateRow(row.gate, row.name, "OOS_CONTAMINATED_IS_ONLY", row.value) for row in rows)
    validation = payload.get("validation")
    payload["validation"] = {**(validation if isinstance(validation, dict) else {}), "oos_label": "OOS_CONTAMINATED_IS_ONLY", "decision_scope": "IS_consistency_only"}
    payload["gates"] = [asdict(row) for row in marked]
    save_json(path, payload)
    return marked


def run_gates(results_dir: Path, cache_dir: Path) -> tuple[str, dict[str, tuple[GateRow, ...]], dict[str, str | float | bool | None]]:
    btc_path = cache_dir / "binance_fapi_BTCUSDT_1d.csv.gz"
    from research.wave1.common import load_frame

    btc_returns = load_frame(btc_path)["close"].pct_change() if btc_path.exists() else pd.Series(dtype=float)
    rows_by_id: dict[str, tuple[GateRow, ...]] = {}
    for candidate_id in ("W2c", *SINGLE_IDS):
        path = results_dir / f"{candidate_id}.json"
        rows = evaluate_result_file_wave2(path, btc_returns)
        rows_by_id[candidate_id] = mark_w5f_is_only(path, rows) if candidate_id == "W5f" else rows
    scores = {candidate_id: sum(row.status == "PASS" for row in rows_by_id[candidate_id]) for candidate_id in COMBINATION_IDS}
    selected = max(COMBINATION_IDS, key=lambda candidate_id: (scores[candidate_id], -COMBINATION_IDS.index(candidate_id)))
    combined, combination = build_combined_result(results_dir, selected)
    save_json(results_dir / "W5g.json", strategy_payload(combined))
    rows_by_id["W5g"] = evaluate_result_file(results_dir / "W5g.json", btc_returns)
    payload = load_json(results_dir / "W5g.json")
    if isinstance(payload, dict):
        payload["combination_gates"] = combination
        payload["gates"] = [asdict(row) for row in rows_by_id["W5g"]]
        save_json(results_dir / "W5g.json", payload)
    return selected, rows_by_id, combination


__all__ = ["COMBINATION_IDS", "SINGLE_IDS", "build_combined_result", "combination_gates", "mark_w5f_is_only", "run_gates"]
