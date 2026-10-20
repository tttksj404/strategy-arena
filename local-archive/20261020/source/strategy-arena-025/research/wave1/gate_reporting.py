# Persists gate evaluations, the candidate registry, and the final verdict report.

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd  # noqa: PANDAS_OK

from research.wave1.common import JsonValue, PipelineError, load_json, save_json
from research.wave1.gates import GateInput, GateRow, MetricInput, Metrics, calculate_metrics, evaluate_gates, kelly_fraction, yearly_returns


@dataclass(frozen=True, slots=True)
class ReportPaths:
    results_dir: Path
    registry_path: Path
    report_path: Path


def write_summary(path: Path, candidate_rows: dict[str, tuple[GateRow, ...]]) -> None:
    lines = ["# Wave-1 gate summary", "", "| Candidate | Gate | Name | Status | Value |", "|---|---:|---|---|---|"]
    for candidate_id, rows in candidate_rows.items():
        lines.extend(f"| {candidate_id} | {row.gate} | {row.name} | {row.status} | {row.value} |" for row in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _series(value: JsonValue) -> pd.Series:
    if not isinstance(value, list):
        raise PipelineError("series payload must be a list")
    records = [item for item in value if isinstance(item, dict) and isinstance(item.get("timestamp"), str)]
    timestamps = pd.DatetimeIndex([pd.Timestamp(item["timestamp"]) for item in records]) if records else pd.DatetimeIndex([], tz="UTC")
    samples = [float(item.get("value", 0.0)) for item in records]
    return pd.Series(samples, index=timestamps, dtype=float).sort_index()


def _floats(value: JsonValue) -> tuple[float, ...]:
    return tuple(float(item) for item in value if isinstance(item, (int, float))) if isinstance(value, list) else ()


def _metric_payload(metrics: Metrics) -> dict[str, JsonValue]:
    return {
        name: float(value) if isinstance(value, (int, float)) and np.isfinite(value) else None
        for name, value in asdict(metrics).items()
    }


def _period_equity(equity: pd.Series, split: pd.Timestamp) -> tuple[pd.Series, pd.Series]:
    in_sample = equity[equity.index <= split]
    holdout = equity[equity.index > split]
    if not in_sample.empty and not holdout.empty:
        holdout = pd.concat([in_sample.iloc[-1:], holdout])
    return in_sample, holdout


def _capacity(metadata: dict[str, JsonValue], trades: tuple[float, ...], positive_ev: bool) -> tuple[bool, str]:
    quarter_kelly = max(0.0, 0.25 * kelly_fraction(trades))
    concurrent = int(metadata.get("max_concurrent_positions", 0))
    max_weight = float(metadata.get("max_position_weight", 0.0))
    min_weight = float(metadata.get("min_position_weight", 0.0))
    minimum_order = float(metadata.get("min_order_usdt", 5.0))
    notional_cap = 150.0 / (300.0 * max_weight) if max_weight > 0.0 else 0.0
    leverage = min(quarter_kelly, 2.0, notional_cap)
    max_notional = 300.0 * leverage * max_weight
    min_notional = 300.0 * leverage * min_weight
    valid = positive_ev and quarter_kelly > 0.0 and 0 < concurrent <= 3 and 0.0 < leverage <= 2.0 and max_notional <= 150.0 and min_notional >= minimum_order
    value = f"quarter_kelly={quarter_kelly:.4f}; leverage={leverage:.4f}; concurrent={concurrent}; notional={min_notional:.2f}-{max_notional:.2f}"
    return valid, value


def evaluate_result_file(path: Path, btc_returns: pd.Series) -> tuple[GateRow, ...]:
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise PipelineError(f"candidate result must be an object: {path.name}")
    equity = _series(payload.get("equity")).resample("1D").last().ffill()
    positions = _series(payload.get("positions"))
    turnover = _series(payload.get("turnover"))
    trades = _series(payload.get("trade_returns"))
    split = pd.Timestamp("2025-09-30T23:59:59Z")
    is_equity, oos_equity = _period_equity(equity, split)
    is_trades = tuple(float(value) for value in trades[trades.index <= split])
    oos_trades = tuple(float(value) for value in trades[trades.index > split])
    is_positions = positions[positions.index <= split]
    oos_positions = positions[positions.index > split]
    is_metrics = calculate_metrics(MetricInput(is_equity, is_trades, float(turnover[turnover.index <= split].sum()), float((is_positions.abs() > 0.0).mean()) if not is_positions.empty else 0.0))
    oos_metrics = calculate_metrics(MetricInput(oos_equity, oos_trades, float(turnover[turnover.index > split].sum()), float((oos_positions.abs() > 0.0).mean()) if not oos_positions.empty else 0.0))
    metadata_value = payload.get("metadata")
    metadata = metadata_value if isinstance(metadata_value, dict) else {}
    neighbor_sharpes = _floats(metadata.get("neighbor_is_sharpes"))
    aligned = pd.concat([equity.pct_change().rename("strategy"), btc_returns.rename("btc")], axis=1, join="inner").dropna()
    correlation = float(aligned["strategy"].corr(aligned["btc"])) if len(aligned) >= 2 else float("nan")
    regimes = {
        label: float(sample.iloc[-1] / sample.iloc[0] - 1.0)
        for year, label in ((2022, "2022_bear"), (2024, "2024_bull"), (2025, "2025_sideways"))
        if len(sample := equity[equity.index.year == year]) >= 2
    }
    capacity_valid, capacity_value = _capacity(metadata, oos_trades, oos_metrics.total_ret > 0.0)
    intended_factor = metadata.get("intended_factor")
    factor_limit = 0.3 if intended_factor == "funding_carry" else 0.8
    factor_valid = isinstance(intended_factor, str) and bool(intended_factor) and bool(np.isfinite(correlation)) and abs(correlation) < factor_limit
    rows = evaluate_gates(GateInput(is_metrics, oos_metrics, oos_trades, neighbor_sharpes, float(payload.get("stress_total_return", 0.0)), yearly_returns(is_equity), regimes, correlation, metadata.get("data_valid") is True, metadata.get("cost_model_valid") is True, capacity_valid, capacity_value, factor_valid))
    payload["metrics"] = {"is": _metric_payload(is_metrics), "oos": _metric_payload(oos_metrics)}
    payload["validation"] = {"neighbor_is_sharpes": list(neighbor_sharpes), "yearly_is_returns": {str(year): value for year, value in yearly_returns(is_equity).items()}, "regime_returns": regimes, "oos_trade_count": len(oos_trades)}
    payload["gates"] = [asdict(row) for row in rows]
    save_json(path, payload)
    return rows


def write_reports(paths: ReportPaths, candidate_ids: tuple[str, ...]) -> None:
    registry_lines = ["# Wave-1 registry", "", "| Candidate | Family | State | Required gates |", "|---|---|---|---|"]
    report_rows: list[str] = []
    deployable: list[tuple[str, float]] = []
    registry_complete = all((paths.results_dir / f"{candidate_id}.json").exists() for candidate_id in candidate_ids)
    for candidate_id in candidate_ids:
        path = paths.results_dir / f"{candidate_id}.json"
        if not path.exists():
            registry_lines.append(f"| {candidate_id} | {candidate_id[:2]} | MISSING | FAIL |")
            continue
        payload = load_json(path)
        if not isinstance(payload, dict):
            continue
        gates_value = payload.get("gates")
        gates = gates_value if isinstance(gates_value, list) else []
        statuses = {int(row["gate"]): str(row["status"]) for row in gates if isinstance(row, dict) and isinstance(row.get("gate"), int)}
        passed = registry_complete and all(statuses.get(gate) == "PASS" for gate in range(1, 20))
        family = str(payload.get("family", candidate_id[:2]))
        state = "EXPLORATORY" if family == "F3" else "EVALUATED"
        registry_lines.append(f"| {candidate_id} | {family} | {state} | {'PASS' if passed else 'FAIL'} |")
        metrics = payload.get("metrics")
        oos = metrics.get("oos") if isinstance(metrics, dict) else None
        calmar = float(oos.get("calmar", 0.0)) if isinstance(oos, dict) and oos.get("calmar") is not None else 0.0
        report_rows.append(f"| {candidate_id} | {family} | {'PASS' if passed else 'FAIL'} | {calmar:.4f} |")
        if passed and family != "F3":
            deployable.append((candidate_id, calmar))
    verdict = "엣지 없음 — wave-2로 이관."
    if deployable:
        winner = max(deployable, key=lambda item: item[1])
        verdict = f"배포 후보: {winner[0]} (OOS Calmar {winner[1]:.4f})"
    paths.registry_path.write_text("\n".join(registry_lines) + "\n", encoding="utf-8")
    report_lines = ["# Wave-1 report", "", f"**Verdict:** {verdict}", "", "| Candidate | Family | Required gates | OOS Calmar |", "|---|---|---|---:|", *report_rows]
    paths.report_path.parent.mkdir(parents=True, exist_ok=True)
    paths.report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
