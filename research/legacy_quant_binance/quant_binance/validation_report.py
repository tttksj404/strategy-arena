from __future__ import annotations

from copy import deepcopy
import json
from dataclasses import asdict, dataclass
from datetime import timezone, datetime, timedelta
from pathlib import Path
from typing import Any

from quant_binance.auto_mode import build_regime_aware_auto_mode
from quant_binance.closed_trade_metrics import (
    aggregate_closed_trades,
    filter_inherited_trades,
    load_closed_trades_jsonl,
)
from quant_binance.performance_report import build_runtime_performance_report, build_runtime_performance_report_from_rows
from quant_binance.policy_evidence import (
    baseline_control_replay_provenance,
    checkpoint_replay_provenance,
    policy_evidence_bucket,
    policy_evidence_bucket_evidence,
    replay_summary_provenance,
    with_policy_evidence_buckets,
)
from quant_binance.policy_lineage import (
    build_policy_lineage_snapshot,
    build_policy_profile_lineage_snapshot,
    build_policy_state_lineage_snapshot,
    policy_lineage_alignment,
)
from quant_binance.symbol_lifecycle import build_symbol_lifecycle, summarize_symbol_lifecycle


@dataclass(frozen=True)
class ValidationCriteriaRow:
    category: str
    rule: str
    action: str


@dataclass(frozen=True)
class WeeklyValidationReport:
    base_dir: str
    generated_at: str
    lookback_days: int
    run_count: int
    period_start: str
    period_end: str
    total_closed_trade_count: int
    total_realized_pnl_usd: float
    total_live_order_count: int
    total_tested_order_count: int
    sample_progress: dict[str, object]
    score_alignment_summary: tuple[dict[str, object], ...]
    symbol_summary: tuple[dict[str, object], ...]
    symbol_scorecard: tuple[dict[str, object], ...]
    regime_summary: tuple[dict[str, object], ...]
    criteria: tuple[ValidationCriteriaRow, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "base_dir": self.base_dir,
            "generated_at": self.generated_at,
            "lookback_days": self.lookback_days,
            "run_count": self.run_count,
            "period_start": self.period_start,
            "period_end": self.period_end,
            "total_closed_trade_count": self.total_closed_trade_count,
            "total_realized_pnl_usd": self.total_realized_pnl_usd,
            "total_live_order_count": self.total_live_order_count,
            "total_tested_order_count": self.total_tested_order_count,
            "sample_progress": dict(self.sample_progress),
            "score_alignment_summary": list(self.score_alignment_summary),
            "symbol_summary": list(self.symbol_summary),
            "symbol_scorecard": list(self.symbol_scorecard),
            "regime_summary": list(self.regime_summary),
            "criteria": [asdict(item) for item in self.criteria],
        }


def _criteria_table() -> tuple[ValidationCriteriaRow, ...]:
    return (
        ValidationCriteriaRow(
            category="prune",
            rule="trade_count >= 3 and expectancy_usd < 0, or repeated thin-edge/liquidity rejections without positive edge",
            action="universe에서 제거 또는 priority 해제",
        ),
        ValidationCriteriaRow(
            category="observe_only",
            rule="cash_count가 높고 LIQUIDITY_TOO_WEAK/EDGE_TOO_THIN 반복, 하지만 trade_count 표본이 부족함",
            action="매매 제외, 관찰만 유지",
        ),
        ValidationCriteriaRow(
            category="keep",
            rule="expectancy_usd >= 0 또는 표본 부족이지만 평균 edge가 양수",
            action="현재 유니버스 유지",
        ),
        ValidationCriteriaRow(
            category="promote",
            rule="trade_count >= 3, expectancy_usd > 0, avg_net_edge_bps > 0, rejection pressure 낮음",
            action="priority_symbols 후보로 승격 검토",
        ),
    )


def _resolve_recent_runs(*, base_dir: Path, lookback_days: int) -> list[Path]:
    mode_root = base_dir / "output" / "paper-live-shell"
    if not mode_root.exists():
        return []
    now = datetime.now(timezone.utc)
    threshold = now - timedelta(days=lookback_days)
    runs: list[Path] = []
    for candidate in mode_root.iterdir():
        if not candidate.is_dir() or candidate.name == "latest":
            continue
        modified = datetime.fromtimestamp(candidate.stat().st_mtime, tz=timezone.utc)
        if modified >= threshold:
            runs.append(candidate)
    runs.sort(key=lambda p: p.stat().st_mtime)
    return runs


def _resolve_latest_run_dir(*, base_dir: Path) -> Path | None:
    mode_root = base_dir / "output" / "paper-live-shell"
    latest = mode_root / "latest"
    if latest.exists():
        return latest
    if not mode_root.exists():
        return None
    runs = [candidate for candidate in mode_root.iterdir() if candidate.is_dir() and candidate.name != "latest"]
    if not runs:
        return None
    return max(runs, key=lambda candidate: candidate.stat().st_mtime)


def _latest_file_under(root: Path, name: str) -> Path | None:
    if not root.exists():
        return None
    matches = sorted(root.rglob(name), key=lambda path: path.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def _load_summary(run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return {}
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return default


_POLICY_BUCKET_EVIDENCE_KEYS = (
    "comparison_verdict",
    "comparison_structural_verdict",
    "comparison_runtime_verdict",
    "comparison_execution_replay_verdict",
    "candidate_vs_current_structural_score_delta",
    "candidate_vs_current_execution_replay_score_delta",
    "candidate_vs_current_score_delta",
    "run_count",
    "total_closed_trade_count",
    "total_live_order_count",
    "total_tested_order_count",
    "runner_total_return_pct",
    "runner_total_realized_pnl_usd",
    "runner_max_drawdown_pct",
    "runner_max_drawdown_usd",
    "runner_drawdown_to_pnl_ratio",
    "runner_shadow_alignment_score",
    "runner_reject_rate",
    "runner_protection_degraded_rate",
    "runner_avg_slippage_bps",
    "runner_avg_realized_edge_bps",
    "runner_avg_edge_retention_ratio",
    "runner_walk_forward_window_count",
    "runner_positive_walk_forward_window_count",
    "runner_positive_walk_forward_ratio",
    "micro_live_gate",
    "recent_retention_window",
    "cumulative_retention_window",
    "sample_quality_watchdog",
    "checkpoint_auto_judge",
    "auto_mode",
    "symbol_lifecycle",
    "symbol_lifecycle_summary",
    "symbol_summary",
    "symbol_scorecard",
    "regime_summary",
    "pruning_recommendations",
    "walk_forward_windows",
    "validation_runs",
    "policy_context_bucket_name",
    "policy_context_bucket_source",
    "policy_context_bucket_available",
    "policy_context_bucket_run_count",
    "policy_context_bucket_decision_count",
    "policy_context_bucket_closed_trade_count",
    "policy_context_bucket_total_realized_pnl_usd",
    "policy_context_bucket_walk_forward_window_count",
    "policy_context_bucket_positive_walk_forward_ratio",
    "policy_context_bucket_validation_runs",
    "policy_context_bucket_walk_forward_windows",
    "policy_context_bucket_symbol_summary",
    "policy_context_bucket_score_alignment_summary",
    "policy_context_bucket_regime_summary",
    "policy_context_bucket_pruning_recommendations",
    "metric_comparisons",
    "current_policy_evidence_alignment",
)


def _bucketize_policy_evidence_payload(payload: dict[str, Any] | None) -> dict[str, object]:
    source = dict(payload or {})
    bucketed: dict[str, object] = {}
    for key in _POLICY_BUCKET_EVIDENCE_KEYS:
        if key in source:
            bucketed[key] = deepcopy(source[key])
    return bucketed


def _policy_evidence_bucket_entry(
    *,
    bucket_name: str,
    source: str,
    available: bool,
    evidence: dict[str, Any] | None = None,
    comparison: dict[str, Any] | None = None,
    policy_lineage: dict[str, Any] | None = None,
    evidence_lineage: dict[str, Any] | None = None,
    alignment: dict[str, Any] | None = None,
    policy_application: dict[str, Any] | None = None,
    replay_summary: dict[str, Any] | None = None,
) -> dict[str, object]:
    return {
        "bucket": bucket_name,
        "source": source,
        "available": bool(available),
        "evidence": _bucketize_policy_evidence_payload(evidence),
        "comparison": deepcopy(dict(comparison or {})),
        "policy_lineage": deepcopy(dict(policy_lineage or {})),
        "evidence_lineage": deepcopy(dict(evidence_lineage or {})),
        "alignment": deepcopy(dict(alignment or {})),
        "policy_application": deepcopy(dict(policy_application or {})),
        "replay_summary": deepcopy(dict(replay_summary or {})),
    }


def _build_policy_evidence_buckets(
    *,
    candidate_policy: dict[str, Any],
    runner_evidence: dict[str, Any],
    current_policy_evidence: dict[str, Any],
    current_active_lineage: dict[str, Any],
    current_policy_evidence_lineage: dict[str, Any],
    current_evidence_lineage_alignment: dict[str, Any],
    baseline_control_comparison: dict[str, Any],
    candidate_policy_application: dict[str, Any],
    current_policy_application: dict[str, Any],
    candidate_replay_summary: dict[str, Any],
    current_replay_summary: dict[str, Any],
) -> dict[str, dict[str, object]]:
    candidate_lineage = build_policy_lineage_snapshot(
        policy=candidate_policy,
        rollout_phase="full",
        policy_status=str(dict(candidate_policy or {}).get("status", "") or ""),
        source="staged_candidate_policy",
    )
    active_available = bool(current_policy_evidence) and bool(current_evidence_lineage_alignment.get("aligned"))
    return {
        "staged_candidate": _policy_evidence_bucket_entry(
            bucket_name="staged_candidate",
            source="policy_comparison_candidate_runner",
            available=bool(runner_evidence),
            evidence=runner_evidence,
            policy_lineage=candidate_lineage,
            policy_application=candidate_policy_application,
            replay_summary=candidate_replay_summary,
        ),
        "active_policy": _policy_evidence_bucket_entry(
            bucket_name="active_policy",
            source=(
                str(dict(current_replay_summary or {}).get("source", "") or "persisted_policy_validation_evidence")
                if current_policy_evidence
                else "active_policy_evidence_unavailable"
            ),
            available=active_available,
            evidence=current_policy_evidence if active_available else {},
            policy_lineage=current_active_lineage,
            evidence_lineage=current_policy_evidence_lineage,
            alignment=current_evidence_lineage_alignment,
            policy_application=current_policy_application,
            replay_summary=current_replay_summary if active_available else {},
        ),
        "baseline_control": _policy_evidence_bucket_entry(
            bucket_name="baseline_control",
            source=str(dict(baseline_control_comparison or {}).get("artifact_path", "") or "baseline_control_comparison"),
            available=bool(dict(baseline_control_comparison or {}).get("available")),
            comparison=baseline_control_comparison,
        ),
    }


def _simple_control_baseline_kind(strategy_name: str) -> str:
    normalized = str(strategy_name or "").strip().lower()
    if not normalized or normalized == "current_strategy":
        return ""
    if normalized == "directional_hold":
        return "directional_hold"
    if normalized == "simple_momentum":
        return "simple_momentum"
    if normalized == "simple_mean_reversion":
        return "simple_mean_reversion"
    if "majors" in normalized and ("only" in normalized or "control" in normalized or "baseline" in normalized):
        return "majors_only_control"
    if normalized.startswith("simple_"):
        return "simple_control"
    if "control" in normalized or "baseline" in normalized or "hold" in normalized:
        return "conservative_control"
    return ""


def _simple_control_baseline_priority(kind: str) -> int:
    return {
        "majors_only_control": 0,
        "directional_hold": 1,
        "simple_momentum": 2,
        "simple_mean_reversion": 3,
        "simple_control": 4,
        "conservative_control": 5,
    }.get(str(kind or ""), 9)


def _baseline_observation_count(row: dict[str, object]) -> int:
    return max(_safe_int(row.get("closed_trade_count")), _safe_int(row.get("trade_count")))


def _runtime_summary_closed_trade_metrics(
    runtime_summary: dict[str, Any] | None,
    *,
    run_dir: Path | None = None,
) -> tuple[int, float]:
    payload = dict(runtime_summary or {})
    raw_trades = list(payload.get("closed_trades", []) or [])
    if raw_trades:
        clean_trades, _ = filter_inherited_trades(raw_trades)
        aggregate = aggregate_closed_trades(clean_trades)
        return aggregate.closed_trade_count, aggregate.realized_pnl_usd
    if run_dir is not None:
        closed_trade_log = run_dir / "logs" / "closed_trades.jsonl"
        if closed_trade_log.exists():
            clean_trades, _ = filter_inherited_trades(load_closed_trades_jsonl(closed_trade_log))
            aggregate = aggregate_closed_trades(clean_trades)
            return aggregate.closed_trade_count, aggregate.realized_pnl_usd
    return (
        _safe_int(payload.get("closed_trade_count")),
        round(_safe_float(payload.get("realized_pnl_usd_estimate")), 6),
    )


def _max_drawdown(values: list[float]) -> float:
    peak = 0.0
    cumulative = 0.0
    max_drawdown = 0.0
    for value in values:
        cumulative += value
        peak = max(peak, cumulative)
        max_drawdown = max(max_drawdown, peak - cumulative)
    return round(max_drawdown, 6)


def _weighted_metric(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return 0.0
    return round(numerator / denominator, 6)


def _safe_ratio(numerator: Any, denominator: Any) -> float:
    denom = _safe_float(denominator)
    if denom <= 0.0:
        return 0.0
    return round(_safe_float(numerator) / denom, 6)


def _extract_policy_lineage_from_payload(
    payload: dict[str, Any] | None,
    *,
    source: str,
    updated_at: object = "",
) -> dict[str, object]:
    item = dict(payload or {})
    explicit_lineage = dict(item.get("policy_lineage", item.get("active_policy_lineage", {})) or {})
    if explicit_lineage:
        return explicit_lineage
    profile = dict(item.get("current_policy_application", item.get("policy_application", {})) or {})
    if profile:
        return build_policy_profile_lineage_snapshot(
            policy_profile=profile,
            updated_at=updated_at or item.get("generated_at", ""),
            source=source,
        )
    active_policy = dict(item.get("active_policy", {}) or {})
    if active_policy:
        return build_policy_state_lineage_snapshot(
            item,
            source=source,
        )
    return {}


def _load_run_policy_lineage(run_dir: Path) -> dict[str, object]:
    state_path = run_dir / "policy_state.json"
    if state_path.exists():
        try:
            payload = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
        if isinstance(payload, dict):
            lineage = _extract_policy_lineage_from_payload(payload, source="run_policy_state")
            if lineage:
                return lineage
    comparison_path = run_dir / "policy_comparison.json"
    if comparison_path.exists():
        try:
            payload = json.loads(comparison_path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
        if isinstance(payload, dict):
            current_policy = dict(payload.get("current_policy", {}) or {})
            lineage = _extract_policy_lineage_from_payload(
                {
                    "policy_application": payload.get("current_policy_application", current_policy.get("policy_application", {})),
                    "active_policy_lineage": payload.get("current_policy_lineage", {}),
                    "generated_at": payload.get("generated_at", ""),
                },
                source="run_policy_comparison",
                updated_at=payload.get("generated_at", ""),
            )
            if lineage:
                return lineage
    return {}


def _rolling_expectancy_stability(expectancies: list[float]) -> float:
    if not expectancies:
        return 0.0
    mean_expectancy = sum(expectancies) / len(expectancies)
    mean_absolute_deviation = sum(abs(value - mean_expectancy) for value in expectancies) / len(expectancies)
    scale = max(abs(mean_expectancy), 1.0)
    return round(max(0.0, 1.0 - min(1.0, mean_absolute_deviation / scale)), 6)


def _symbol_rolling_evidence(
    *,
    aggregate_expectancy: float,
    history: list[dict[str, object]],
) -> dict[str, object]:
    observed_history = [dict(item) for item in history if _safe_int(item.get("trade_count")) > 0]
    if not observed_history:
        return {
            "available": False,
            "observed_run_count": 0,
            "recent_run_count": 0,
            "recent_run_consistency": 0.0,
            "positive_window_count": 0,
            "positive_window_ratio": 0.0,
            "expectancy_stability": 0.0,
        }
    expectancies = [_safe_float(item.get("expectancy_usd")) for item in observed_history]
    recent_expectancies = expectancies[-3:]
    direction = 1 if aggregate_expectancy > 0.0 else (-1 if aggregate_expectancy < 0.0 else 0)
    if direction > 0:
        recent_consistency = _safe_ratio(sum(1 for value in recent_expectancies if value > 0.0), len(recent_expectancies))
    elif direction < 0:
        recent_consistency = _safe_ratio(sum(1 for value in recent_expectancies if value < 0.0), len(recent_expectancies))
    else:
        recent_consistency = 0.0
    positive_window_count = sum(1 for value in expectancies if value > 0.0)
    return {
        "available": True,
        "observed_run_count": len(observed_history),
        "recent_run_count": len(recent_expectancies),
        "recent_run_consistency": round(recent_consistency, 6),
        "positive_window_count": positive_window_count,
        "positive_window_ratio": round(_safe_ratio(positive_window_count, len(observed_history)), 6),
        "expectancy_stability": _rolling_expectancy_stability(expectancies),
    }


def _build_symbol_scorecard(symbol_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in list(symbol_rows):
        symbol = str(row.get("symbol", "") or "")
        recommendation = str(row.get("recommendation", "keep") or "keep")
        expectancy = _safe_float(row.get("expectancy_usd"))
        rolling_evidence = dict(row.get("rolling_evidence", {}) or {})
        observed_run_count = _safe_int(rolling_evidence.get("observed_run_count"))
        recent_run_count = _safe_int(rolling_evidence.get("recent_run_count"))
        recent_run_consistency = _safe_float(rolling_evidence.get("recent_run_consistency"))
        positive_window_ratio = _safe_float(rolling_evidence.get("positive_window_ratio"))
        expectancy_stability = _safe_float(rolling_evidence.get("expectancy_stability"))
        direction = 1 if expectancy > 0.0 else (-1 if expectancy < 0.0 else 0)
        recent_positive_run_ratio = (
            recent_run_consistency
            if direction >= 0
            else round(max(0.0, 1.0 - recent_run_consistency), 6)
        )
        directional_strength = (
            (recent_run_consistency * 0.45)
            + (positive_window_ratio * 0.35)
            + (expectancy_stability * 0.2)
        )
        rolling_score = round(
            0.5 + (directional_strength * 0.5 * direction),
            6,
        ) if direction != 0 else 0.5
        scorecard_recommendation = "keep"
        sample_status = "warmup"
        if recommendation == "promote" and observed_run_count >= 2 and recent_run_consistency >= 0.67 and positive_window_ratio >= 0.6:
            scorecard_recommendation = "promote"
            sample_status = "supportive"
        elif recommendation == "prune" and observed_run_count >= 2 and recent_run_consistency >= 0.67 and positive_window_ratio <= 0.4:
            scorecard_recommendation = "demote"
            sample_status = "supportive"
        elif observed_run_count >= 2:
            sample_status = "mixed"
        rows.append(
            {
                "symbol": symbol,
                "recommendation": scorecard_recommendation,
                "sample_status": sample_status,
                "rolling_score": max(0.0, min(1.0, rolling_score)),
                "trade_count": _safe_int(row.get("trade_count")),
                "trade_run_count": observed_run_count,
                "recent_trade_run_count": recent_run_count,
                "recent_positive_run_ratio": round(recent_positive_run_ratio, 6),
                "positive_window_ratio": round(positive_window_ratio, 6),
                "expectancy_stability": round(expectancy_stability, 6),
            }
        )
    rows.sort(
        key=lambda item: (
            {"demote": 0, "keep": 1, "promote": 2}.get(str(item.get("recommendation", "keep")), 3),
            float(item.get("rolling_score", 0.0)),
            str(item.get("symbol", "")),
        ),
        reverse=True,
    )
    return rows


def _score_bucket_floor(label: str) -> int:
    token = str(label or "").split("-", 1)[0].rstrip("+")
    try:
        return int(token)
    except (TypeError, ValueError):
        return -1


def _score_alignment_signal(score_rows: list[dict[str, object]]) -> dict[str, object]:
    ordered_rows = sorted(
        [dict(row) for row in score_rows if isinstance(row, dict)],
        key=lambda row: _score_bucket_floor(str(row.get("score_bucket_label", "") or "")),
    )
    if len(ordered_rows) < 2:
        expectancy = _safe_float(ordered_rows[0].get("expectancy_usd")) if ordered_rows else 0.0
        return {
            "available": bool(ordered_rows),
            "bucket_count": len(ordered_rows),
            "monotonic_ratio": 0.0,
            "high_vs_low_expectancy_delta_usd": round(expectancy, 6),
            "alignment_score": 0.5 if ordered_rows else 0.0,
        }
    supportive_steps = 0
    adjacent_steps = 0
    for previous, current in zip(ordered_rows, ordered_rows[1:]):
        adjacent_steps += 1
        if _safe_float(current.get("expectancy_usd")) >= _safe_float(previous.get("expectancy_usd")) - 0.05:
            supportive_steps += 1
    midpoint = max(len(ordered_rows) // 2, 1)
    low_rows = ordered_rows[:midpoint]
    high_rows = ordered_rows[midpoint:]
    low_weight = sum(_safe_int(row.get("trade_count")) for row in low_rows)
    high_weight = sum(_safe_int(row.get("trade_count")) for row in high_rows)
    low_expectancy = _weighted_metric(
        sum(_safe_float(row.get("expectancy_usd")) * _safe_int(row.get("trade_count")) for row in low_rows),
        float(max(low_weight, 1)),
    )
    high_expectancy = _weighted_metric(
        sum(_safe_float(row.get("expectancy_usd")) * _safe_int(row.get("trade_count")) for row in high_rows),
        float(max(high_weight, 1)),
    )
    monotonic_ratio = round(_safe_ratio(supportive_steps, adjacent_steps), 6)
    expectancy_delta = round(high_expectancy - low_expectancy, 6)
    alignment_score = _clamp(
        0.5
        + (expectancy_delta / 6.0)
        + ((monotonic_ratio - 0.5) * 0.5),
        lower=0.0,
        upper=1.0,
    )
    return {
        "available": True,
        "bucket_count": len(ordered_rows),
        "monotonic_ratio": monotonic_ratio,
        "high_vs_low_expectancy_delta_usd": expectancy_delta,
        "alignment_score": alignment_score,
    }


def _sample_quality_checkpoint_snapshot(
    *,
    total_closed_trade_count: int,
    total_live_order_count: int,
    total_tested_order_count: int,
    symbol_rows: list[dict[str, object]],
) -> dict[str, object]:
    portfolio_thresholds = [
        {
            "metric": "total_closed_trade_count",
            "threshold": 6,
            "current_value": total_closed_trade_count,
            "reached": total_closed_trade_count >= 6,
        },
        {
            "metric": "total_closed_trade_count",
            "threshold": 10,
            "current_value": total_closed_trade_count,
            "reached": total_closed_trade_count >= 10,
        },
        {
            "metric": "total_live_order_count",
            "threshold": 8,
            "current_value": total_live_order_count,
            "reached": total_live_order_count >= 8,
        },
        {
            "metric": "total_live_order_count",
            "threshold": 12,
            "current_value": total_live_order_count,
            "reached": total_live_order_count >= 12,
        },
        {
            "metric": "total_tested_order_count",
            "threshold": 4,
            "current_value": total_tested_order_count,
            "reached": total_tested_order_count >= 4,
        },
    ]
    symbol_thresholds = [
        {
            "symbol": str(row.get("symbol", "") or ""),
            "trade_count": _safe_int(row.get("trade_count")),
            "validation_threshold": _safe_int(row.get("required_trade_count_for_validation"), 3),
            "validation_ready": _safe_int(row.get("trade_count")) >= _safe_int(row.get("required_trade_count_for_validation"), 3),
        }
        for row in symbol_rows
        if str(row.get("symbol", "") or "")
    ]
    symbol_thresholds.sort(key=lambda item: str(item["symbol"]))
    return {
        "portfolio": portfolio_thresholds,
        "symbols": symbol_thresholds,
    }


def _build_sample_quality_watchdog(
    *,
    run_count: int,
    total_closed_trade_count: int,
    total_live_order_count: int,
    total_tested_order_count: int,
    total_realized_pnl_usd: float,
    symbol_rows: list[dict[str, object]],
    symbol_scorecard: list[dict[str, object]],
    score_alignment_summary: list[dict[str, object]],
    runner_reject_rate: float,
    runner_avg_slippage_bps: float,
    runner_avg_realized_edge_bps: float,
    runner_avg_edge_retention_ratio: float,
    runner_protection_degraded_rate: float,
    runner_walk_forward_window_count: int,
    runner_positive_walk_forward_ratio: float,
) -> dict[str, object]:
    total_symbol_trades = sum(_safe_int(row.get("trade_count")) for row in symbol_rows)
    dominant_symbol_row = max(
        symbol_rows,
        key=lambda row: (_safe_int(row.get("trade_count")), str(row.get("symbol", ""))),
        default={},
    )
    dominant_symbol_trade_count = _safe_int(dominant_symbol_row.get("trade_count"))
    dominant_symbol_trade_share = round(
        dominant_symbol_trade_count / max(total_symbol_trades, 1),
        6,
    ) if total_symbol_trades > 0 else 0.0
    validated_symbol_count = sum(
        1
        for row in symbol_rows
        if _safe_int(row.get("trade_count")) >= _safe_int(row.get("required_trade_count_for_validation"), 3)
    )
    alignment = _score_alignment_signal(score_alignment_summary)
    scorecard_weight = float(sum(_safe_int(row.get("trade_count")) for row in symbol_scorecard))
    weighted_recent_consistency = _weighted_metric(
        sum(_safe_float(row.get("recent_positive_run_ratio")) * _safe_int(row.get("trade_count")) for row in symbol_scorecard),
        scorecard_weight,
    )
    weighted_expectancy_stability = _weighted_metric(
        sum(_safe_float(row.get("expectancy_stability")) * _safe_int(row.get("trade_count")) for row in symbol_scorecard),
        scorecard_weight,
    )
    recent_consistency_score = _clamp(
        (weighted_recent_consistency * 0.55)
        + (weighted_expectancy_stability * 0.45),
        lower=0.0,
        upper=1.0,
    )
    reasons: list[str] = []
    status = "healthy"
    if total_closed_trade_count <= 0 and total_live_order_count <= 0:
        status = "thin"
        reasons.append("NO_SAMPLE_EVIDENCE")
    else:
        if total_closed_trade_count < 6:
            reasons.append("CLOSED_TRADE_SAMPLE_THIN")
        if total_live_order_count < 8:
            reasons.append("LIVE_ORDER_SAMPLE_THIN")
        if run_count < 2:
            reasons.append("RUN_HISTORY_THIN")
        if validated_symbol_count < 2:
            reasons.append("VALIDATED_SYMBOL_BREADTH_THIN")
        if dominant_symbol_trade_share >= 0.68 and total_closed_trade_count >= 4:
            reasons.append("SYMBOL_CONCENTRATION_ELEVATED")
        if (
            total_closed_trade_count < 6
            or total_live_order_count < 8
            or run_count < 2
            or validated_symbol_count < 2
        ):
            status = "thin"
        if total_closed_trade_count >= 6:
            degraded = False
            if dominant_symbol_trade_share >= 0.82:
                degraded = True
                reasons.append("SYMBOL_CONCENTRATION_TOO_HIGH")
            if total_realized_pnl_usd <= 0.0:
                degraded = True
                reasons.append("REALIZED_PNL_NON_POSITIVE")
            if alignment["available"] and float(alignment.get("alignment_score", 0.0) or 0.0) < 0.45:
                degraded = True
                reasons.append("SCORE_TO_PNL_ALIGNMENT_WEAK")
            if runner_avg_edge_retention_ratio < 0.55 and total_live_order_count >= 4:
                degraded = True
                reasons.append("EDGE_RETENTION_WEAK")
            if runner_avg_realized_edge_bps <= 0.0 and total_live_order_count >= 4:
                degraded = True
                reasons.append("REALIZED_EDGE_NON_POSITIVE")
            if runner_reject_rate > 0.12 and total_live_order_count >= 4:
                degraded = True
                reasons.append("REJECT_RATE_HIGH")
            if runner_avg_slippage_bps > 12.0 and total_live_order_count >= 4:
                degraded = True
                reasons.append("SLIPPAGE_HIGH")
            if runner_protection_degraded_rate > 0.12 and total_live_order_count >= 4:
                degraded = True
                reasons.append("EXECUTION_PROTECTION_DEGRADED")
            if runner_walk_forward_window_count >= 2 and runner_positive_walk_forward_ratio < 0.5:
                degraded = True
                reasons.append("RUN_CONSISTENCY_WEAK")
            if recent_consistency_score < 0.5 and validated_symbol_count >= 1:
                degraded = True
                reasons.append("SYMBOL_CONSISTENCY_WEAK")
            if degraded:
                status = "degraded"
        if (
            status != "degraded"
            and total_closed_trade_count >= 10
            and total_live_order_count >= 12
            and validated_symbol_count >= 2
            and dominant_symbol_trade_share <= 0.58
            and total_realized_pnl_usd > 0.0
            and float(alignment.get("alignment_score", 0.0) or 0.0) >= 0.6
            and recent_consistency_score >= 0.62
            and runner_avg_edge_retention_ratio >= 0.68
            and runner_avg_realized_edge_bps > 0.0
            and runner_reject_rate <= 0.06
            and runner_avg_slippage_bps <= 8.0
            and runner_protection_degraded_rate <= 0.08
            and (runner_walk_forward_window_count < 2 or runner_positive_walk_forward_ratio >= 0.67)
        ):
            status = "promote_ready"
            reasons = ["BROAD_SAMPLE_SUPPORTIVE"]
    guardrails = {
        "degraded": {
            "promotion_intensity_cap": 0.65,
            "max_positive_symbols": 1,
            "allow_alt_promotions": False,
            "prefer_majors_only": True,
            "non_major_positive_bias": "observe_only",
        },
        "thin": {
            "promotion_intensity_cap": 0.8,
            "max_positive_symbols": 1,
            "allow_alt_promotions": False,
            "prefer_majors_only": True,
            "non_major_positive_bias": "observe_only",
        },
        "healthy": {
            "promotion_intensity_cap": 1.0,
            "max_positive_symbols": 0,
            "allow_alt_promotions": False,
            "prefer_majors_only": True,
            "non_major_positive_bias": "neutral",
        },
        "promote_ready": {
            "promotion_intensity_cap": 1.05,
            "max_positive_symbols": 2,
            "allow_alt_promotions": True,
            "prefer_majors_only": False,
            "non_major_positive_bias": "neutral",
        },
    }
    return {
        "status": status,
        "reason_codes": reasons,
        "metrics": {
            "run_count": run_count,
            "total_closed_trade_count": total_closed_trade_count,
            "total_live_order_count": total_live_order_count,
            "total_tested_order_count": total_tested_order_count,
            "total_realized_pnl_usd": round(total_realized_pnl_usd, 6),
            "validated_symbol_count": validated_symbol_count,
            "dominant_symbol": str(dominant_symbol_row.get("symbol", "") or ""),
            "dominant_symbol_trade_share": dominant_symbol_trade_share,
            "score_alignment_score": round(float(alignment.get("alignment_score", 0.0) or 0.0), 6),
            "score_alignment_monotonic_ratio": round(float(alignment.get("monotonic_ratio", 0.0) or 0.0), 6),
            "score_alignment_delta_usd": round(float(alignment.get("high_vs_low_expectancy_delta_usd", 0.0) or 0.0), 6),
            "recent_run_consistency": round(weighted_recent_consistency, 6),
            "recent_expectancy_stability": round(weighted_expectancy_stability, 6),
            "recent_consistency_score": recent_consistency_score,
            "runner_reject_rate": round(runner_reject_rate, 6),
            "runner_avg_slippage_bps": round(runner_avg_slippage_bps, 6),
            "runner_avg_realized_edge_bps": round(runner_avg_realized_edge_bps, 6),
            "runner_avg_edge_retention_ratio": round(runner_avg_edge_retention_ratio, 6),
            "runner_protection_degraded_rate": round(runner_protection_degraded_rate, 6),
            "runner_walk_forward_window_count": runner_walk_forward_window_count,
            "runner_positive_walk_forward_ratio": round(runner_positive_walk_forward_ratio, 6),
        },
        "policy_guardrails": guardrails[status],
        "checkpoint_snapshot": _sample_quality_checkpoint_snapshot(
            total_closed_trade_count=total_closed_trade_count,
            total_live_order_count=total_live_order_count,
            total_tested_order_count=total_tested_order_count,
            symbol_rows=symbol_rows,
        ),
    }


def _run_validation_snapshot(*, run_dir: Path) -> dict[str, object]:
    report = build_runtime_performance_report(run_dir=run_dir)
    summary = _load_summary(run_dir)
    live_order_count = _safe_int(summary.get("live_order_count"))
    accepted_live_order_count = _safe_int(summary.get("accepted_live_order_count"))
    rejected_live_order_count = _safe_int(summary.get("rejected_live_order_count"))
    walk_forward = []
    for row in report.walk_forward:
        walk_forward.append(
            {
                "run_id": run_dir.name,
                "window_index": _safe_int(row.get("window_index")),
                "start": str(row.get("start", "")),
                "end": str(row.get("end", "")),
                "decision_count": _safe_int(row.get("decision_count")),
                "futures_count": _safe_int(row.get("futures_count")),
                "spot_count": _safe_int(row.get("spot_count")),
                "cash_count": _safe_int(row.get("cash_count")),
                "avg_score": round(_safe_float(row.get("avg_score")), 6),
                "avg_net_edge_bps": round(_safe_float(row.get("avg_net_edge_bps")), 6),
            }
        )
    return {
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "generated_at": str(summary.get("generated_at", "")),
        "policy_lineage": _load_run_policy_lineage(run_dir),
        "policy_context_buckets": _policy_context_bucket_snapshots(run_dir),
        "closed_trade_count": report.closed_trade_count,
        "realized_pnl_usd": round(report.realized_pnl_usd, 6),
        "live_order_count": live_order_count,
        "accepted_live_order_count": accepted_live_order_count,
        "rejected_live_order_count": rejected_live_order_count,
        "tested_order_count": _safe_int(summary.get("tested_order_count")),
        "avg_slippage_bps": round(_safe_float(summary.get("avg_slippage_bps")), 6),
        "avg_edge_retention_ratio": round(_safe_float(summary.get("avg_edge_retention_ratio")), 6),
        "avg_realized_edge_bps": round(_safe_float(summary.get("avg_realized_edge_bps")), 6),
        "avg_expected_edge_bps": round(_safe_float(summary.get("avg_expected_edge_bps")), 6),
        "reject_rate": round(rejected_live_order_count / max(live_order_count, 1), 6) if live_order_count > 0 else 0.0,
        "protection_degraded_rate": round(_safe_float(summary.get("protection_degraded_rate")), 6),
        "walk_forward": walk_forward,
        "symbol_expectancy": [asdict(item) for item in report.symbol_expectancy],
        "regime_performance": [asdict(item) for item in report.regime_performance],
        "score_bucket_performance": [asdict(item) for item in report.score_bucket_performance],
        "pruning_recommendations": [dict(item) for item in report.pruning_recommendations],
    }


def _policy_context_bucket_name(row: dict[str, Any] | None) -> str:
    payload = dict(row or {})
    bucket_name = str(payload.get("entry_policy_bucket", "") or "")
    if not bucket_name or not bool(payload.get("entry_policy_bucket_available")):
        return ""
    alignment_status = str(payload.get("entry_policy_bucket_alignment_status", "") or "")
    if alignment_status and alignment_status not in {"aligned", "legacy_unbucketed"}:
        return ""
    return bucket_name


def _consensus_policy_lineage(lineages: list[dict[str, object]]) -> dict[str, object]:
    known: dict[str, dict[str, object]] = {}
    for item in lineages:
        payload = dict(item or {})
        if not payload:
            continue
        lineage_key = str(payload.get("structural_key", "") or payload.get("versioned_key", "") or "")
        if not lineage_key:
            continue
        known.setdefault(lineage_key, payload)
    if len(known) == 1:
        return dict(next(iter(known.values())))
    return {}


def _bucket_edge_retention_ratio(realized_edge_bps: object, expected_edge_bps: object) -> float | None:
    realized = _safe_float(realized_edge_bps, None)
    expected = _safe_float(expected_edge_bps, None)
    if realized is None or expected is None:
        return None
    baseline = max(float(expected), 0.0)
    if baseline <= 0.0:
        return None
    return round(max(min(float(realized) / max(baseline, 0.1), 2.0), -2.0), 6)


def _policy_context_bucket_execution_snapshot(
    *,
    live_orders: list[dict[str, Any]],
    tested_orders: list[dict[str, Any]],
    order_errors: list[dict[str, Any]],
) -> dict[str, object]:
    by_symbol: dict[str, dict[str, float | int]] = {}
    live_order_count = 0
    accepted_live_order_count = 0
    rejected_live_order_count = 0
    tested_order_count = 0
    order_error_count = 0
    protection_degraded_count = 0
    slippage_sum = 0.0
    slippage_count = 0
    realized_edge_sum = 0.0
    realized_edge_count = 0
    retention_sum = 0.0
    retention_count = 0

    def symbol_bucket(symbol: str) -> dict[str, float | int]:
        return by_symbol.setdefault(
            symbol,
            {
                "live_order_count": 0,
                "accepted_live_order_count": 0,
                "rejected_live_order_count": 0,
                "tested_order_count": 0,
                "order_error_count": 0,
                "protection_degraded_count": 0,
                "slippage_sum": 0.0,
                "slippage_count": 0,
                "realized_edge_sum": 0.0,
                "realized_edge_count": 0,
                "retention_sum": 0.0,
                "retention_count": 0,
            },
        )

    for row in live_orders:
        symbol = str(row.get("symbol", "") or "")
        if not symbol:
            continue
        bucket = symbol_bucket(symbol)
        live_order_count += 1
        bucket["live_order_count"] = int(bucket["live_order_count"]) + 1
        accepted = bool(row.get("accepted", False))
        if accepted:
            accepted_live_order_count += 1
            bucket["accepted_live_order_count"] = int(bucket["accepted_live_order_count"]) + 1
        else:
            rejected_live_order_count += 1
            bucket["rejected_live_order_count"] = int(bucket["rejected_live_order_count"]) + 1
        slippage = row.get("slippage_bps")
        if slippage not in (None, ""):
            value = _safe_float(slippage)
            slippage_sum += value
            slippage_count += 1
            bucket["slippage_sum"] = float(bucket["slippage_sum"]) + value
            bucket["slippage_count"] = int(bucket["slippage_count"]) + 1
        realized_edge = row.get("realized_edge_bps")
        if realized_edge not in (None, ""):
            value = _safe_float(realized_edge)
            realized_edge_sum += value
            realized_edge_count += 1
            bucket["realized_edge_sum"] = float(bucket["realized_edge_sum"]) + value
            bucket["realized_edge_count"] = int(bucket["realized_edge_count"]) + 1
            retention = _bucket_edge_retention_ratio(
                realized_edge,
                row.get("expected_net_edge_bps", row.get("net_expected_edge_bps")),
            )
            if retention is not None:
                retention_sum += retention
                retention_count += 1
                bucket["retention_sum"] = float(bucket["retention_sum"]) + retention
                bucket["retention_count"] = int(bucket["retention_count"]) + 1
        if row.get("protection_error"):
            protection_degraded_count += 1
            bucket["protection_degraded_count"] = int(bucket["protection_degraded_count"]) + 1

    for row in tested_orders:
        symbol = str(row.get("symbol", "") or "")
        if not symbol:
            continue
        tested_order_count += 1
        bucket = symbol_bucket(symbol)
        bucket["tested_order_count"] = int(bucket["tested_order_count"]) + 1

    for row in order_errors:
        symbol = str(row.get("symbol", "") or "")
        if not symbol:
            continue
        order_error_count += 1
        bucket = symbol_bucket(symbol)
        bucket["order_error_count"] = int(bucket["order_error_count"]) + 1

    symbol_execution_summary = []
    for symbol, bucket in by_symbol.items():
        symbol_live_order_count = int(bucket["live_order_count"])
        symbol_accepted_count = int(bucket["accepted_live_order_count"])
        symbol_rejected_count = int(bucket["rejected_live_order_count"])
        symbol_slippage_count = int(bucket["slippage_count"])
        symbol_realized_edge_count = int(bucket["realized_edge_count"])
        symbol_retention_count = int(bucket["retention_count"])
        symbol_execution_summary.append(
            {
                "symbol": symbol,
                "live_order_count": symbol_live_order_count,
                "accepted_live_order_count": symbol_accepted_count,
                "rejected_live_order_count": symbol_rejected_count,
                "tested_order_count": int(bucket["tested_order_count"]),
                "order_error_count": int(bucket["order_error_count"]),
                "avg_slippage_bps": round(float(bucket["slippage_sum"]) / symbol_slippage_count, 6) if symbol_slippage_count else 0.0,
                "avg_realized_edge_bps": round(float(bucket["realized_edge_sum"]) / symbol_realized_edge_count, 6) if symbol_realized_edge_count else 0.0,
                "avg_edge_retention_ratio": round(float(bucket["retention_sum"]) / symbol_retention_count, 6) if symbol_retention_count else 0.0,
                "reject_rate": round(symbol_rejected_count / max(symbol_live_order_count, 1), 6) if symbol_live_order_count else 0.0,
                "protection_degraded_count": int(bucket["protection_degraded_count"]),
                "protection_degraded_rate": round(
                    int(bucket["protection_degraded_count"]) / max(symbol_live_order_count, 1),
                    6,
                )
                if symbol_live_order_count
                else 0.0,
            }
        )
    symbol_execution_summary.sort(
        key=lambda item: (
            -float(item.get("avg_edge_retention_ratio", 0.0) or 0.0),
            -float(item.get("avg_realized_edge_bps", 0.0) or 0.0),
            str(item.get("symbol", "") or ""),
        )
    )
    return {
        "live_order_count": live_order_count,
        "accepted_live_order_count": accepted_live_order_count,
        "rejected_live_order_count": rejected_live_order_count,
        "tested_order_count": tested_order_count,
        "order_error_count": order_error_count,
        "avg_slippage_bps": round(slippage_sum / slippage_count, 6) if slippage_count else 0.0,
        "avg_realized_edge_bps": round(realized_edge_sum / realized_edge_count, 6) if realized_edge_count else 0.0,
        "avg_edge_retention_ratio": round(retention_sum / retention_count, 6) if retention_count else 0.0,
        "reject_rate": round(rejected_live_order_count / max(live_order_count, 1), 6) if live_order_count else 0.0,
        "protection_degraded_rate": round(protection_degraded_count / max(live_order_count, 1), 6) if live_order_count else 0.0,
        "symbol_execution_summary": symbol_execution_summary,
    }


def _policy_context_bucket_snapshot(
    *,
    run_dir: Path,
    bucket_name: str,
    decisions: list[dict[str, Any]],
    closed_trades: list[dict[str, Any]],
    live_orders: list[dict[str, Any]],
    tested_orders: list[dict[str, Any]],
    order_errors: list[dict[str, Any]],
) -> dict[str, object]:
    report = build_runtime_performance_report_from_rows(
        run_dir=run_dir,
        summary_path=run_dir / "summary.json",
        decisions=decisions,
        closed_trades=closed_trades,
    )
    execution_metrics = _policy_context_bucket_execution_snapshot(
        live_orders=live_orders,
        tested_orders=tested_orders,
        order_errors=order_errors,
    )
    lineages = [
        dict(item.get("entry_policy_lineage", {}) or {})
        for item in [*decisions, *closed_trades, *live_orders]
        if isinstance(item, dict)
    ]
    sources = sorted(
        {
            str(item.get("entry_policy_bucket_source", "") or "")
            for item in [*decisions, *closed_trades, *live_orders, *order_errors]
            if str(item.get("entry_policy_bucket_source", "") or "")
        }
    )
    alignment_statuses = sorted(
        {
            str(item.get("entry_policy_bucket_alignment_status", "") or "")
            for item in [*decisions, *closed_trades, *live_orders, *order_errors]
            if str(item.get("entry_policy_bucket_alignment_status", "") or "")
        }
    )
    return {
        "bucket": bucket_name,
        "available": bool(decisions or closed_trades or live_orders or tested_orders or order_errors),
        "decision_count": len(decisions),
        "closed_trade_count": report.closed_trade_count,
        "realized_pnl_usd": round(report.realized_pnl_usd, 6),
        "live_order_count": int(execution_metrics.get("live_order_count", 0) or 0),
        "accepted_live_order_count": int(execution_metrics.get("accepted_live_order_count", 0) or 0),
        "rejected_live_order_count": int(execution_metrics.get("rejected_live_order_count", 0) or 0),
        "tested_order_count": int(execution_metrics.get("tested_order_count", 0) or 0),
        "order_error_count": int(execution_metrics.get("order_error_count", 0) or 0),
        "avg_slippage_bps": round(_safe_float(execution_metrics.get("avg_slippage_bps")), 6),
        "avg_realized_edge_bps": round(_safe_float(execution_metrics.get("avg_realized_edge_bps")), 6),
        "avg_edge_retention_ratio": round(_safe_float(execution_metrics.get("avg_edge_retention_ratio")), 6),
        "reject_rate": round(_safe_float(execution_metrics.get("reject_rate")), 6),
        "protection_degraded_rate": round(_safe_float(execution_metrics.get("protection_degraded_rate")), 6),
        "walk_forward": [dict(item) for item in report.walk_forward],
        "symbol_expectancy": [asdict(item) for item in report.symbol_expectancy],
        "symbol_execution_summary": list(execution_metrics.get("symbol_execution_summary", []) or []),
        "regime_performance": [asdict(item) for item in report.regime_performance],
        "score_bucket_performance": [asdict(item) for item in report.score_bucket_performance],
        "pruning_recommendations": [dict(item) for item in report.pruning_recommendations],
        "policy_lineage": _consensus_policy_lineage(lineages),
        "sources": sources,
        "alignment_statuses": alignment_statuses,
    }


def _policy_context_bucket_snapshots(run_dir: Path) -> dict[str, dict[str, object]]:
    logs_dir = run_dir / "logs"
    decisions = load_closed_trades_jsonl(logs_dir / "decisions.jsonl")
    closed_trades = load_closed_trades_jsonl(logs_dir / "closed_trades.jsonl")
    live_orders = load_closed_trades_jsonl(logs_dir / "live_orders.jsonl")
    tested_orders = load_closed_trades_jsonl(logs_dir / "tested_orders.jsonl")
    order_errors = load_closed_trades_jsonl(logs_dir / "order_errors.jsonl")
    bucket_names = sorted(
        {
            bucket_name
            for bucket_name in (
                _policy_context_bucket_name(row)
                for row in [*decisions, *closed_trades, *live_orders, *tested_orders, *order_errors]
            )
            if bucket_name
        }
    )
    snapshots: dict[str, dict[str, object]] = {}
    for bucket_name in bucket_names:
        bucket_decisions = [
            dict(item)
            for item in decisions
            if _policy_context_bucket_name(item) == bucket_name
        ]
        bucket_closed_trades = [
            dict(item)
            for item in closed_trades
            if _policy_context_bucket_name(item) == bucket_name
        ]
        bucket_live_orders = [
            dict(item)
            for item in live_orders
            if _policy_context_bucket_name(item) == bucket_name
        ]
        bucket_tested_orders = [
            dict(item)
            for item in tested_orders
            if _policy_context_bucket_name(item) == bucket_name
        ]
        bucket_order_errors = [
            dict(item)
            for item in order_errors
            if _policy_context_bucket_name(item) == bucket_name
        ]
        snapshots[bucket_name] = _policy_context_bucket_snapshot(
            run_dir=run_dir,
            bucket_name=bucket_name,
            decisions=bucket_decisions,
            closed_trades=bucket_closed_trades,
            live_orders=bucket_live_orders,
            tested_orders=bucket_tested_orders,
            order_errors=bucket_order_errors,
        )
    return snapshots


def _build_micro_live_gate(*, live_order_count: int, rejected_live_order_count: int, avg_slippage_bps: float, avg_realized_edge_bps: float, closed_trade_count: int) -> dict[str, object]:
    available = live_order_count > 0 or closed_trade_count > 0
    minimum_live_order_count = 2
    minimum_closed_trade_count = 1
    if not available:
        return {
            "available": False,
            "status": "not_available",
            "required_live_order_count": minimum_live_order_count,
            "required_closed_trade_count": minimum_closed_trade_count,
            "reason": "NO_MICRO_LIVE_EVIDENCE",
        }
    pass_gate = (
        live_order_count >= minimum_live_order_count
        and closed_trade_count >= minimum_closed_trade_count
        and (rejected_live_order_count / max(live_order_count, 1)) <= 0.2
        and avg_slippage_bps <= 15.0
        and avg_realized_edge_bps > 0.0
    )
    reason = "MICRO_LIVE_THRESHOLD_PASSED" if pass_gate else "MICRO_LIVE_THRESHOLD_NOT_MET"
    return {
        "available": True,
        "status": "pass" if pass_gate else "pending",
        "required_live_order_count": minimum_live_order_count,
        "required_closed_trade_count": minimum_closed_trade_count,
        "live_order_count": live_order_count,
        "closed_trade_count": closed_trade_count,
        "reject_rate": round(rejected_live_order_count / max(live_order_count, 1), 6) if live_order_count > 0 else 0.0,
        "avg_slippage_bps": round(avg_slippage_bps, 6),
        "avg_realized_edge_bps": round(avg_realized_edge_bps, 6),
        "reason": reason,
    }


def _runtime_metric_signal(*, candidate: float | None, current: float | None, higher_is_better: bool, tolerance: float) -> tuple[float, float]:
    if candidate is None or current is None:
        return 0.0, 0.0
    delta = round(candidate - current, 6)
    if abs(delta) <= tolerance:
        return 0.0, delta
    return ((1.0 if delta > 0.0 else -1.0) if higher_is_better else (-1.0 if delta > 0.0 else 1.0)), delta


def _aggregate_pruning_recommendations(
    run_snapshots: list[dict[str, object]],
) -> list[dict[str, object]]:
    severity = {"prune": 0, "demote": 1, "observe_only": 2, "keep": 3}
    merged: dict[str, dict[str, object]] = {}
    for snapshot in run_snapshots:
        for item in list(snapshot.get("pruning_recommendations", []) or []):
            symbol = str(item.get("symbol", "") or "")
            if not symbol:
                continue
            candidate = dict(item)
            existing = merged.get(symbol)
            if existing is None:
                merged[symbol] = candidate
                continue
            existing_rank = severity.get(str(existing.get("recommendation", "keep")), 99)
            candidate_rank = severity.get(str(candidate.get("recommendation", "keep")), 99)
            if candidate_rank < existing_rank:
                merged[symbol] = candidate
                continue
            if candidate_rank == existing_rank:
                existing_decisions = _safe_int(existing.get("decision_count"))
                candidate_decisions = _safe_int(candidate.get("decision_count"))
                if candidate_decisions >= existing_decisions:
                    merged[symbol] = candidate
    return sorted(
        merged.values(),
        key=lambda item: (
            severity.get(str(item.get("recommendation", "keep")), 99),
            str(item.get("symbol", "")),
        ),
    )


def _policy_context_bucket_run_snapshots(
    *,
    run_snapshots: list[dict[str, object]],
    bucket_name: str,
) -> list[dict[str, object]]:
    bucket_runs: list[dict[str, object]] = []
    for snapshot in run_snapshots:
        bucket_payload = dict(dict(snapshot.get("policy_context_buckets", {}) or {}).get(bucket_name, {}) or {})
        if not bool(bucket_payload.get("available")):
            continue
        bucket_runs.append(
            {
                "run_id": str(snapshot.get("run_id", "") or ""),
                "run_dir": str(snapshot.get("run_dir", "") or ""),
                "generated_at": str(snapshot.get("generated_at", "") or ""),
                "policy_lineage": dict(bucket_payload.get("policy_lineage", {}) or {}),
                "decision_count": _safe_int(bucket_payload.get("decision_count")),
                "closed_trade_count": _safe_int(bucket_payload.get("closed_trade_count")),
                "realized_pnl_usd": round(_safe_float(bucket_payload.get("realized_pnl_usd")), 6),
                "live_order_count": _safe_int(bucket_payload.get("live_order_count")),
                "accepted_live_order_count": _safe_int(bucket_payload.get("accepted_live_order_count")),
                "rejected_live_order_count": _safe_int(bucket_payload.get("rejected_live_order_count")),
                "tested_order_count": _safe_int(bucket_payload.get("tested_order_count")),
                "avg_slippage_bps": round(_safe_float(bucket_payload.get("avg_slippage_bps")), 6),
                "avg_edge_retention_ratio": round(_safe_float(bucket_payload.get("avg_edge_retention_ratio")), 6),
                "avg_realized_edge_bps": round(_safe_float(bucket_payload.get("avg_realized_edge_bps")), 6),
                "avg_expected_edge_bps": 0.0,
                "reject_rate": round(_safe_float(bucket_payload.get("reject_rate")), 6),
                "protection_degraded_rate": round(_safe_float(bucket_payload.get("protection_degraded_rate")), 6),
                "walk_forward": [dict(item) for item in list(bucket_payload.get("walk_forward", []) or [])],
                "symbol_expectancy": [dict(item) for item in list(bucket_payload.get("symbol_expectancy", []) or [])],
                "symbol_execution_summary": [
                    dict(item)
                    for item in list(bucket_payload.get("symbol_execution_summary", []) or [])
                ],
                "regime_performance": [dict(item) for item in list(bucket_payload.get("regime_performance", []) or [])],
                "score_bucket_performance": [dict(item) for item in list(bucket_payload.get("score_bucket_performance", []) or [])],
                "pruning_recommendations": [dict(item) for item in list(bucket_payload.get("pruning_recommendations", []) or [])],
            }
        )
    return bucket_runs


def _policy_context_bucket_direct_evidence(
    *,
    base_dir: str | Path,
    lookback_days: int,
    generated_at: str,
    run_snapshots: list[dict[str, object]],
    bucket_name: str,
) -> dict[str, object]:
    bucket_runs = _policy_context_bucket_run_snapshots(run_snapshots=run_snapshots, bucket_name=bucket_name)
    if not bucket_runs:
        return {}
    aggregated = _aggregate_weekly_validation_from_run_snapshots(
        base_dir=base_dir,
        run_snapshots=bucket_runs,
        lookback_days=lookback_days,
        generated_at=generated_at,
    )
    walk_forward_windows = [
        dict(window)
        for snapshot in bucket_runs
        for window in list(snapshot.get("walk_forward", []) or [])
    ]
    positive_walk_forward_count = sum(
        1
        for window in walk_forward_windows
        if _safe_float(window.get("avg_net_edge_bps")) > 0.0 and _safe_float(window.get("avg_score")) >= 0.0
    )
    live_order_count = sum(_safe_int(item.get("live_order_count")) for item in bucket_runs)
    accepted_live_order_count = sum(_safe_int(item.get("accepted_live_order_count")) for item in bucket_runs)
    rejected_live_order_count = sum(_safe_int(item.get("rejected_live_order_count")) for item in bucket_runs)
    tested_order_count = sum(_safe_int(item.get("tested_order_count")) for item in bucket_runs)
    slippage_weight = float(sum(_safe_int(item.get("accepted_live_order_count")) for item in bucket_runs))
    retention_weight = float(sum(_safe_int(item.get("live_order_count")) for item in bucket_runs))
    avg_slippage_bps = _weighted_metric(
        sum(_safe_float(item.get("avg_slippage_bps")) * _safe_int(item.get("accepted_live_order_count")) for item in bucket_runs),
        slippage_weight,
    )
    avg_realized_edge_bps = _weighted_metric(
        sum(_safe_float(item.get("avg_realized_edge_bps")) * _safe_int(item.get("live_order_count")) for item in bucket_runs),
        retention_weight,
    )
    avg_edge_retention_ratio = _weighted_metric(
        sum(_safe_float(item.get("avg_edge_retention_ratio")) * _safe_int(item.get("live_order_count")) for item in bucket_runs),
        retention_weight,
    )
    reject_rate = round(rejected_live_order_count / max(live_order_count, 1), 6) if live_order_count > 0 else 0.0
    protection_degraded_rate = _weighted_metric(
        sum(_safe_float(item.get("protection_degraded_rate")) * _safe_int(item.get("live_order_count")) for item in bucket_runs),
        retention_weight,
    )
    pruning_recommendations = _aggregate_pruning_recommendations(bucket_runs)
    symbol_execution_summary = [
        dict(item)
        for snapshot in bucket_runs
        for item in list(snapshot.get("symbol_execution_summary", []) or [])
    ]
    return {
        "policy_context_bucket_name": bucket_name,
        "policy_context_bucket_source": "decision_closed_trade_logs",
        "policy_context_bucket_available": True,
        "policy_context_bucket_run_count": len(bucket_runs),
        "policy_context_bucket_decision_count": sum(_safe_int(item.get("decision_count")) for item in bucket_runs),
        "policy_context_bucket_closed_trade_count": aggregated.get("total_closed_trade_count", 0),
        "policy_context_bucket_live_order_count": live_order_count,
        "policy_context_bucket_tested_order_count": tested_order_count,
        "policy_context_bucket_total_realized_pnl_usd": aggregated.get("total_realized_pnl_usd", 0.0),
        "policy_context_bucket_walk_forward_window_count": len(walk_forward_windows),
        "policy_context_bucket_positive_walk_forward_ratio": round(
            _safe_ratio(positive_walk_forward_count, len(walk_forward_windows)),
            6,
        ),
        "policy_context_bucket_validation_runs": bucket_runs,
        "policy_context_bucket_walk_forward_windows": walk_forward_windows,
        "policy_context_bucket_symbol_summary": list(aggregated.get("symbol_summary", []) or []),
        "policy_context_bucket_symbol_execution_summary": symbol_execution_summary,
        "policy_context_bucket_score_alignment_summary": list(aggregated.get("score_alignment_summary", []) or []),
        "policy_context_bucket_regime_summary": list(aggregated.get("regime_summary", []) or []),
        "policy_context_bucket_pruning_recommendations": pruning_recommendations,
        "run_count": len(bucket_runs),
        "total_closed_trade_count": aggregated.get("total_closed_trade_count", 0),
        "total_live_order_count": live_order_count,
        "total_tested_order_count": tested_order_count,
        "runner_total_realized_pnl_usd": aggregated.get("total_realized_pnl_usd", 0.0),
        "runner_reject_rate": reject_rate,
        "runner_protection_degraded_rate": protection_degraded_rate,
        "runner_avg_slippage_bps": avg_slippage_bps,
        "runner_avg_realized_edge_bps": avg_realized_edge_bps,
        "runner_avg_edge_retention_ratio": avg_edge_retention_ratio,
        "runner_walk_forward_window_count": len(walk_forward_windows),
        "runner_positive_walk_forward_window_count": positive_walk_forward_count,
        "runner_positive_walk_forward_ratio": round(
            _safe_ratio(positive_walk_forward_count, len(walk_forward_windows)),
            6,
        ),
        "live_order_count": live_order_count,
        "accepted_live_order_count": accepted_live_order_count,
        "rejected_live_order_count": rejected_live_order_count,
        "tested_order_count": tested_order_count,
        "avg_slippage_bps": avg_slippage_bps,
        "avg_realized_edge_bps": avg_realized_edge_bps,
        "avg_edge_retention_ratio": avg_edge_retention_ratio,
        "reject_rate": reject_rate,
        "protection_degraded_rate": protection_degraded_rate,
        "symbol_summary": list(aggregated.get("symbol_summary", []) or []),
        "symbol_execution_summary": symbol_execution_summary,
        "symbol_scorecard": list(aggregated.get("symbol_scorecard", []) or []),
        "score_alignment_summary": list(aggregated.get("score_alignment_summary", []) or []),
        "regime_summary": list(aggregated.get("regime_summary", []) or []),
        "pruning_recommendations": pruning_recommendations,
        "walk_forward_windows": walk_forward_windows,
    }


def _policy_context_bucket_direct_evidence_map(
    *,
    base_dir: str | Path,
    lookback_days: int,
    generated_at: str,
    run_snapshots: list[dict[str, object]],
) -> dict[str, dict[str, object]]:
    bucket_names = sorted(
        {
            str(bucket_name)
            for snapshot in run_snapshots
            for bucket_name in dict(snapshot.get("policy_context_buckets", {}) or {})
            if str(bucket_name)
        }
    )
    evidence_by_bucket: dict[str, dict[str, object]] = {}
    for bucket_name in bucket_names:
        evidence = _policy_context_bucket_direct_evidence(
            base_dir=base_dir,
            lookback_days=lookback_days,
            generated_at=generated_at,
            run_snapshots=run_snapshots,
            bucket_name=bucket_name,
        )
        if evidence:
            evidence_by_bucket[bucket_name] = evidence
    return evidence_by_bucket


def _overlay_policy_context_bucket_evidence(
    evidence: dict[str, Any] | None,
    *,
    bucket_evidence: dict[str, Any] | None,
) -> tuple[dict[str, object], dict[str, object]]:
    merged = dict(evidence or {})
    bucket = dict(bucket_evidence or {})
    bucket_runs = [
        dict(item)
        for item in list(bucket.get("policy_context_bucket_validation_runs", []) or [])
        if isinstance(item, dict)
    ]
    bucket_walk_forward_windows = [
        dict(item)
        for item in list(bucket.get("policy_context_bucket_walk_forward_windows", []) or [])
        if isinstance(item, dict)
    ]
    if not bucket:
        return merged, {"applied": False}
    retention_window = _retention_window_signal(
        validation_runs=bucket_runs,
        walk_forward_windows=bucket_walk_forward_windows,
    )
    if bucket_runs:
        merged["validation_runs"] = bucket_runs
    if bucket_walk_forward_windows:
        merged["walk_forward_windows"] = bucket_walk_forward_windows
    if "policy_context_bucket_symbol_summary" in bucket:
        merged["symbol_summary"] = [
            dict(item)
            for item in list(bucket.get("policy_context_bucket_symbol_summary", []) or [])
            if isinstance(item, dict)
        ]
    if "policy_context_bucket_symbol_execution_summary" in bucket:
        merged["symbol_execution_summary"] = [
            dict(item)
            for item in list(bucket.get("policy_context_bucket_symbol_execution_summary", []) or [])
            if isinstance(item, dict)
        ]
    if "policy_context_bucket_score_alignment_summary" in bucket:
        merged["score_alignment_summary"] = [
            dict(item)
            for item in list(bucket.get("policy_context_bucket_score_alignment_summary", []) or [])
            if isinstance(item, dict)
        ]
    if "policy_context_bucket_regime_summary" in bucket:
        merged["regime_summary"] = [
            dict(item)
            for item in list(bucket.get("policy_context_bucket_regime_summary", []) or [])
            if isinstance(item, dict)
        ]
    if "policy_context_bucket_pruning_recommendations" in bucket:
        merged["pruning_recommendations"] = [
            dict(item)
            for item in list(bucket.get("policy_context_bucket_pruning_recommendations", []) or [])
            if isinstance(item, dict)
        ]
    merged.update(
        {
            "run_count": _safe_int(bucket.get("policy_context_bucket_run_count", merged.get("run_count"))),
            "total_closed_trade_count": _safe_int(
                bucket.get("policy_context_bucket_closed_trade_count", merged.get("total_closed_trade_count"))
            ),
            "total_live_order_count": _safe_int(
                bucket.get("policy_context_bucket_live_order_count", merged.get("total_live_order_count"))
            ),
            "total_tested_order_count": _safe_int(
                bucket.get("policy_context_bucket_tested_order_count", merged.get("total_tested_order_count"))
            ),
            "runner_total_realized_pnl_usd": round(
                _safe_float(
                    bucket.get("policy_context_bucket_total_realized_pnl_usd", merged.get("runner_total_realized_pnl_usd"))
                ),
                6,
            ),
            "runner_drawdown_to_pnl_ratio": round(
                _safe_float(retention_window.get("drawdown_to_pnl_ratio", merged.get("runner_drawdown_to_pnl_ratio"))),
                6,
            ),
            "runner_reject_rate": round(_safe_float(bucket.get("reject_rate", merged.get("runner_reject_rate"))), 6),
            "runner_protection_degraded_rate": round(
                _safe_float(bucket.get("protection_degraded_rate", merged.get("runner_protection_degraded_rate"))),
                6,
            ),
            "runner_avg_slippage_bps": round(
                _safe_float(bucket.get("avg_slippage_bps", merged.get("runner_avg_slippage_bps"))),
                6,
            ),
            "runner_avg_realized_edge_bps": round(
                _safe_float(bucket.get("avg_realized_edge_bps", merged.get("runner_avg_realized_edge_bps"))),
                6,
            ),
            "runner_avg_edge_retention_ratio": round(
                _safe_float(bucket.get("avg_edge_retention_ratio", merged.get("runner_avg_edge_retention_ratio"))),
                6,
            ),
            "runner_walk_forward_window_count": _safe_int(
                bucket.get("policy_context_bucket_walk_forward_window_count", merged.get("runner_walk_forward_window_count"))
            ),
            "runner_positive_walk_forward_window_count": _safe_int(
                retention_window.get("positive_walk_forward_count", merged.get("runner_positive_walk_forward_window_count"))
            ),
            "runner_positive_walk_forward_ratio": round(
                _safe_float(
                    bucket.get(
                        "policy_context_bucket_positive_walk_forward_ratio",
                        merged.get("runner_positive_walk_forward_ratio"),
                    )
                ),
                6,
            ),
            "policy_context_bucket_name": str(bucket.get("policy_context_bucket_name", "") or ""),
            "policy_context_bucket_source": str(bucket.get("policy_context_bucket_source", "") or ""),
            "policy_context_bucket_available": bool(bucket.get("policy_context_bucket_available")),
            "preferred_policy_bucket": str(bucket.get("policy_context_bucket_name", "") or ""),
        }
    )
    bucket_live_order_count = _safe_int(bucket.get("policy_context_bucket_live_order_count"))
    bucket_rejected_live_order_count = max(
        bucket_live_order_count - _safe_int(bucket.get("accepted_live_order_count")),
        _safe_int(bucket.get("rejected_live_order_count")),
    )
    bucket_closed_trade_count = _safe_int(bucket.get("policy_context_bucket_closed_trade_count"))
    if bucket_live_order_count > 0 or bucket_closed_trade_count > 0 or bucket_runs:
        merged["micro_live_gate"] = _build_micro_live_gate(
            live_order_count=bucket_live_order_count,
            rejected_live_order_count=bucket_rejected_live_order_count,
            avg_slippage_bps=_safe_float(bucket.get("avg_slippage_bps")),
            avg_realized_edge_bps=_safe_float(bucket.get("avg_realized_edge_bps")),
            closed_trade_count=bucket_closed_trade_count,
        )
    return merged, {
        "applied": True,
        "bucket_name": str(bucket.get("policy_context_bucket_name", "") or ""),
        "source": str(bucket.get("policy_context_bucket_source", "") or ""),
        "used_validation_runs": bool(bucket_runs),
        "used_walk_forward_windows": bool(bucket_walk_forward_windows),
    }


def _compare_runtime_evidence(*, candidate_evidence: dict[str, Any], current_evidence: dict[str, Any]) -> dict[str, object]:
    current_present = bool(current_evidence)
    metric_rows = (
        ("runner_total_realized_pnl_usd", True, 0.5),
        ("runner_drawdown_to_pnl_ratio", False, 0.05),
        ("runner_reject_rate", False, 0.01),
        ("runner_protection_degraded_rate", False, 0.01),
        ("runner_avg_slippage_bps", False, 0.5),
        ("runner_avg_realized_edge_bps", True, 0.25),
        ("runner_avg_edge_retention_ratio", True, 0.02),
        ("runner_shadow_alignment_score", True, 0.03),
        ("runner_positive_walk_forward_ratio", True, 0.05),
    )
    metric_deltas: dict[str, float] = {}
    compared_metrics: list[str] = []
    metric_comparisons: list[dict[str, object]] = []
    runtime_score = 0.0
    for metric_name, higher_is_better, tolerance in metric_rows:
        candidate_value = candidate_evidence.get(metric_name)
        current_value = current_evidence.get(metric_name)
        signal, delta = _runtime_metric_signal(
            candidate=_safe_float(candidate_value) if candidate_value is not None else None,
            current=_safe_float(current_value) if current_value is not None else None,
            higher_is_better=higher_is_better,
            tolerance=tolerance,
        )
        if candidate_value is None or current_value is None:
            continue
        compared_metrics.append(metric_name)
        metric_deltas[f"{metric_name}_delta"] = delta
        runtime_score += signal
        metric_comparisons.append(
            {
                "metric": metric_name,
                "candidate_value": round(_safe_float(candidate_value), 6),
                "current_value": round(_safe_float(current_value), 6),
                "delta": delta,
                "verdict": (
                    "candidate_better"
                    if signal > 0.0
                    else ("candidate_worse" if signal < 0.0 else "keep")
                ),
            }
        )
    verdict = "keep"
    if runtime_score >= 1.0:
        verdict = "candidate_better"
    elif runtime_score <= -1.0:
        verdict = "candidate_worse"
    return {
        "runtime_evidence_available": current_present and bool(compared_metrics),
        "runtime_comparison_verdict": verdict,
        "candidate_vs_current_runtime_score": round(runtime_score, 6),
        "compared_metrics": compared_metrics,
        "metric_comparisons": metric_comparisons,
        **metric_deltas,
    }


def _retention_window_signal(
    *,
    validation_runs: list[dict[str, object]],
    walk_forward_windows: list[dict[str, object]],
    window_size: int | None = None,
) -> dict[str, object]:
    runs = list(validation_runs[-window_size:] if window_size is not None and window_size > 0 else validation_runs)
    windows = list(walk_forward_windows[-window_size:] if window_size is not None and window_size > 0 else walk_forward_windows)
    live_order_count = sum(_safe_int(item.get("live_order_count")) for item in runs)
    accepted_live_order_count = sum(_safe_int(item.get("accepted_live_order_count")) for item in runs)
    rejected_live_order_count = sum(_safe_int(item.get("rejected_live_order_count")) for item in runs)
    closed_trade_count = sum(_safe_int(item.get("closed_trade_count")) for item in runs)
    retention_weight = float(max(live_order_count, 0))
    slippage_weight = float(max(accepted_live_order_count, 0))
    positive_walk_forward_count = sum(
        1
        for item in windows
        if _safe_float(item.get("avg_net_edge_bps")) > 0.0 and _safe_float(item.get("avg_score")) >= 0.0
    )
    pnl_series = [_safe_float(item.get("realized_pnl_usd")) for item in runs]
    return {
        "available": bool(runs or windows),
        "window_size": window_size if window_size is not None else len(runs),
        "run_count": len(runs),
        "walk_forward_window_count": len(windows),
        "live_order_count": live_order_count,
        "accepted_live_order_count": accepted_live_order_count,
        "rejected_live_order_count": rejected_live_order_count,
        "closed_trade_count": closed_trade_count,
        "avg_edge_retention_ratio": _weighted_metric(
            sum(_safe_float(item.get("avg_edge_retention_ratio")) * _safe_int(item.get("live_order_count")) for item in runs),
            retention_weight,
        ),
        "avg_slippage_bps": _weighted_metric(
            sum(_safe_float(item.get("avg_slippage_bps")) * _safe_int(item.get("accepted_live_order_count")) for item in runs),
            slippage_weight,
        ),
        "reject_rate": round(rejected_live_order_count / max(live_order_count, 1), 6) if live_order_count > 0 else 0.0,
        "drawdown_to_pnl_ratio": _max_drawdown(pnl_series),
        "positive_walk_forward_count": positive_walk_forward_count,
        "positive_walk_forward_ratio": round(_safe_ratio(positive_walk_forward_count, len(windows)), 6),
    }


def _policy_rollout_phase_scale(phase: str) -> float:
    return {
        "partial": 0.35,
        "broad": 0.7,
        "full": 1.0,
        "watch": 0.5,
        "rollback": 0.0,
    }.get(str(phase or "full"), 1.0)


def _policy_application_profile(
    *,
    policy: dict[str, Any],
    rollout_phase: str,
    source: str,
) -> dict[str, object]:
    adjustments = list(dict(policy or {}).get("adjustments", []) or [])
    symbols = sorted(str(item.get("symbol", "") or "") for item in adjustments if str(item.get("symbol", "") or ""))
    promote_count = 0
    aggressive_promote_count = 0
    demote_count = 0
    disabled_count = 0
    majors_only_count = 0
    size_values: list[float] = []
    leverage_values: list[float] = []
    entry_values: list[float] = []
    expected_floor_values: list[float] = []
    phase_scale = _policy_rollout_phase_scale(rollout_phase)
    for item in adjustments:
        action = str(item.get("action", "") or "")
        size_multiplier = _safe_float(item.get("size_multiplier"), 1.0)
        leverage_multiplier = _safe_float(item.get("leverage_multiplier"), 1.0)
        entry_threshold_bps = _safe_float(item.get("entry_threshold_bps"))
        expected_profit_floor_bps = _safe_float(item.get("expected_profit_floor_bps"))
        if action in {"promote", "aggressive_promote"}:
            size_multiplier = round(1.0 + ((size_multiplier - 1.0) * phase_scale), 6)
            leverage_multiplier = round(1.0 + ((leverage_multiplier - 1.0) * phase_scale), 6)
            entry_threshold_bps = round(entry_threshold_bps * phase_scale, 6)
            expected_profit_floor_bps = round(expected_profit_floor_bps * phase_scale, 6)
            promote_count += 1
            if action == "aggressive_promote":
                aggressive_promote_count += 1
        elif action == "demote":
            demote_count += 1
        elif action == "disabled":
            disabled_count += 1
        if str(item.get("symbol_bias", "neutral") or "neutral") == "majors_only":
            majors_only_count += 1
        size_values.append(size_multiplier)
        leverage_values.append(leverage_multiplier)
        entry_values.append(entry_threshold_bps)
        expected_floor_values.append(expected_profit_floor_bps)
    adjustment_count = len(adjustments)
    return {
        "source": source,
        "rollout_phase": rollout_phase,
        "phase_application_factor": round(phase_scale, 6),
        "adjustment_count": adjustment_count,
        "symbols": symbols,
        "promote_count": promote_count,
        "aggressive_promote_count": aggressive_promote_count,
        "demote_count": demote_count,
        "disabled_count": disabled_count,
        "majors_only_count": majors_only_count,
        "net_size_delta": round(sum(value - 1.0 for value in size_values), 6) if adjustment_count else 0.0,
        "net_leverage_delta": round(sum(value - 1.0 for value in leverage_values), 6) if adjustment_count else 0.0,
        "entry_aggressiveness_delta_bps": round(sum(-value for value in entry_values), 6) if adjustment_count else 0.0,
        "expected_profit_floor_aggressiveness_delta_bps": round(sum(-value for value in expected_floor_values), 6) if adjustment_count else 0.0,
        "avg_size_multiplier": round(sum(size_values) / adjustment_count, 6) if adjustment_count else 1.0,
        "avg_leverage_multiplier": round(sum(leverage_values) / adjustment_count, 6) if adjustment_count else 1.0,
        "avg_entry_threshold_bps": round(sum(entry_values) / adjustment_count, 6) if adjustment_count else 0.0,
        "avg_expected_profit_floor_bps": round(sum(expected_floor_values) / adjustment_count, 6) if adjustment_count else 0.0,
    }


def _policy_application_delta(
    *,
    candidate_profile: dict[str, object],
    current_profile: dict[str, object],
) -> dict[str, object]:
    metric_names = {
        "avg_size_multiplier": "avg_size_multiplier_delta",
        "avg_leverage_multiplier": "avg_leverage_multiplier_delta",
        "avg_entry_threshold_bps": "avg_entry_threshold_bps_delta",
        "avg_expected_profit_floor_bps": "avg_expected_profit_floor_bps_delta",
        "promote_count": "promote_count_delta",
        "aggressive_promote_count": "aggressive_promote_count_delta",
        "demote_count": "demote_count_delta",
        "disabled_count": "disabled_count_delta",
        "majors_only_count": "majors_only_count_delta",
        "adjustment_count": "adjustment_count_delta",
        "net_size_delta": "net_size_delta",
        "net_leverage_delta": "net_leverage_delta",
        "entry_aggressiveness_delta_bps": "entry_aggressiveness_delta_bps",
        "expected_profit_floor_aggressiveness_delta_bps": "expected_profit_floor_aggressiveness_delta_bps",
    }
    deltas = {
        output_key: round(
            _safe_float(candidate_profile.get(metric_name)) - _safe_float(current_profile.get(metric_name)),
            6,
        )
        for metric_name, output_key in metric_names.items()
    }
    candidate_symbols = set(candidate_profile.get("symbols", []))
    current_symbols = set(current_profile.get("symbols", []))
    return {
        "candidate_rollout_phase": str(candidate_profile.get("rollout_phase", "full") or "full"),
        "current_rollout_phase": str(current_profile.get("rollout_phase", "full") or "full"),
        "candidate_phase_application_factor": round(_safe_float(candidate_profile.get("phase_application_factor"), 1.0), 6),
        "current_phase_application_factor": round(_safe_float(current_profile.get("phase_application_factor"), 1.0), 6),
        "candidate_only_symbols": sorted(candidate_symbols - current_symbols),
        "current_only_symbols": sorted(current_symbols - candidate_symbols),
        "shared_symbols": sorted(candidate_symbols & current_symbols),
        **deltas,
    }


def _clamp(value: float, *, lower: float, upper: float) -> float:
    return round(min(upper, max(lower, value)), 6)


def _policy_application_pressure(profile: dict[str, object]) -> float:
    payload = dict(profile or {})
    pressure = (
        (_safe_float(payload.get("net_size_delta")) * 0.6)
        + (_safe_float(payload.get("net_leverage_delta")) * 0.45)
        + (_safe_float(payload.get("entry_aggressiveness_delta_bps")) * 0.08)
        + (_safe_float(payload.get("expected_profit_floor_aggressiveness_delta_bps")) * 0.06)
        + (_safe_float(payload.get("promote_count")) * 0.12)
        + (_safe_float(payload.get("aggressive_promote_count")) * 0.18)
        - (_safe_float(payload.get("demote_count")) * 0.12)
        - (_safe_float(payload.get("disabled_count")) * 0.2)
        - (_safe_float(payload.get("majors_only_count")) * 0.04)
    )
    return _clamp(pressure, lower=-1.25, upper=1.25)


def _runtime_summary_validation_snapshot(
    runtime_summary: dict[str, Any] | None,
    *,
    run_dir: Path | None = None,
    policy_lineage: dict[str, object] | None = None,
) -> dict[str, object] | None:
    payload = dict(runtime_summary or {})
    live_order_count = _safe_int(payload.get("live_order_count"))
    closed_trade_count, realized_pnl_usd = _runtime_summary_closed_trade_metrics(payload, run_dir=run_dir)
    if live_order_count <= 0 and closed_trade_count <= 0:
        return None
    accepted_live_order_count = _safe_int(payload.get("accepted_live_order_count"))
    rejected_live_order_count = _safe_int(payload.get("rejected_live_order_count"))
    generated_at = str(payload.get("generated_at", ""))
    return {
        "run_id": run_dir.name if run_dir is not None else "current-runtime-summary",
        "run_dir": str(run_dir) if run_dir is not None else "current-runtime-summary",
        "generated_at": generated_at,
        "policy_lineage": dict(policy_lineage or {}),
        "closed_trade_count": closed_trade_count,
        "realized_pnl_usd": realized_pnl_usd,
        "live_order_count": live_order_count,
        "accepted_live_order_count": accepted_live_order_count,
        "rejected_live_order_count": rejected_live_order_count,
        "tested_order_count": _safe_int(payload.get("tested_order_count")),
        "avg_slippage_bps": round(_safe_float(payload.get("avg_slippage_bps")), 6),
        "avg_edge_retention_ratio": round(_safe_float(payload.get("avg_edge_retention_ratio")), 6),
        "avg_realized_edge_bps": round(_safe_float(payload.get("avg_realized_edge_bps")), 6),
        "avg_expected_edge_bps": round(_safe_float(payload.get("avg_expected_edge_bps")), 6),
        "reject_rate": round(rejected_live_order_count / max(live_order_count, 1), 6) if live_order_count > 0 else 0.0,
        "protection_degraded_rate": round(_safe_float(payload.get("protection_degraded_rate")), 6),
        "walk_forward": [],
    }


def _merge_runtime_summary_snapshot(
    validation_runs: list[dict[str, object]],
    runtime_summary_snapshot: dict[str, object] | None,
) -> list[dict[str, object]]:
    merged = [dict(item) for item in validation_runs]
    snapshot = dict(runtime_summary_snapshot or {})
    if not snapshot:
        return merged
    snapshot_run_dir = str(snapshot.get("run_dir", "") or "")
    snapshot_run_id = str(snapshot.get("run_id", "") or "")
    snapshot_generated_at = str(snapshot.get("generated_at", "") or "")
    for index, item in enumerate(merged):
        item_run_dir = str(item.get("run_dir", "") or "")
        item_run_id = str(item.get("run_id", "") or "")
        item_generated_at = str(item.get("generated_at", "") or "")
        if snapshot_run_dir and item_run_dir == snapshot_run_dir:
            merged[index] = snapshot
            return merged
        if snapshot_run_id and item_run_id == snapshot_run_id:
            merged[index] = snapshot
            return merged
        if snapshot_generated_at and snapshot_generated_at == item_generated_at:
            merged[index] = snapshot
            return merged
    merged.append(snapshot)
    return merged


def _project_validation_run(
    run_snapshot: dict[str, object],
    *,
    delta_pressure: float,
) -> dict[str, object]:
    payload = dict(run_snapshot or {})
    risk_shift = max(delta_pressure, 0.0)
    relief_shift = max(-delta_pressure, 0.0)
    live_order_count = _safe_int(payload.get("live_order_count"))
    closed_trade_count = _safe_int(payload.get("closed_trade_count"))
    projected_live_order_count = max(
        0,
        int(round(live_order_count * max(0.2, 1.0 + (delta_pressure * 0.25)))),
    )
    base_reject_rate = _safe_float(payload.get("reject_rate"))
    projected_reject_rate = _clamp(
        base_reject_rate + (risk_shift * 0.035) - (relief_shift * 0.02),
        lower=0.0,
        upper=0.95,
    )
    projected_rejected_live_order_count = min(
        projected_live_order_count,
        max(0, int(round(projected_live_order_count * projected_reject_rate))),
    )
    projected_accepted_live_order_count = max(projected_live_order_count - projected_rejected_live_order_count, 0)
    projected_closed_trade_count = max(
        0,
        int(round(closed_trade_count * max(0.2, 1.0 + (delta_pressure * 0.2)))),
    )
    projected_realized_pnl_usd = round(
        _safe_float(payload.get("realized_pnl_usd")) * max(0.1, 1.0 + (delta_pressure * 0.35)),
        6,
    )
    projected_avg_slippage_bps = round(
        _safe_float(payload.get("avg_slippage_bps")) * max(0.0, 1.0 + (risk_shift * 0.2) - (relief_shift * 0.08)),
        6,
    )
    projected_avg_edge_retention_ratio = _clamp(
        _safe_float(payload.get("avg_edge_retention_ratio")) - (risk_shift * 0.07) + (relief_shift * 0.05),
        lower=-2.0,
        upper=2.0,
    )
    projected_avg_realized_edge_bps = round(
        _safe_float(payload.get("avg_realized_edge_bps")) * max(0.1, 1.0 + (delta_pressure * 0.22)),
        6,
    )
    projected_avg_expected_edge_bps = round(
        _safe_float(payload.get("avg_expected_edge_bps")) * max(0.1, 1.0 + (delta_pressure * 0.16)),
        6,
    )
    projected = dict(payload)
    projected.update(
        {
            "live_order_count": projected_live_order_count,
            "accepted_live_order_count": projected_accepted_live_order_count,
            "rejected_live_order_count": projected_rejected_live_order_count,
            "closed_trade_count": projected_closed_trade_count,
            "realized_pnl_usd": projected_realized_pnl_usd,
            "avg_slippage_bps": projected_avg_slippage_bps,
            "avg_edge_retention_ratio": projected_avg_edge_retention_ratio,
            "avg_realized_edge_bps": projected_avg_realized_edge_bps,
            "avg_expected_edge_bps": projected_avg_expected_edge_bps,
            "reject_rate": projected_reject_rate,
        }
    )
    return projected


def _project_walk_forward_window(
    window: dict[str, object],
    *,
    delta_pressure: float,
) -> dict[str, object]:
    payload = dict(window or {})
    projected = dict(payload)
    projected["decision_count"] = max(
        0,
        int(round(_safe_int(payload.get("decision_count")) * max(0.2, 1.0 + (delta_pressure * 0.15)))),
    )
    projected["futures_count"] = max(
        0,
        int(round(_safe_int(payload.get("futures_count")) * max(0.2, 1.0 + (delta_pressure * 0.2)))),
    )
    projected["spot_count"] = max(
        0,
        int(round(_safe_int(payload.get("spot_count")) * max(0.2, 1.0 + (delta_pressure * 0.1)))),
    )
    projected["cash_count"] = max(
        0,
        int(round(_safe_int(payload.get("cash_count")) * max(0.2, 1.0 - (delta_pressure * 0.1)))),
    )
    projected["avg_score"] = round(_safe_float(payload.get("avg_score")) + (delta_pressure * 5.0), 6)
    projected["avg_net_edge_bps"] = round(
        _safe_float(payload.get("avg_net_edge_bps")) * max(0.1, 1.0 + (delta_pressure * 0.25)),
        6,
    )
    return projected


def _execution_replay_score(metrics: dict[str, object]) -> float:
    payload = dict(metrics or {})
    return round(
        (_safe_float(payload.get("total_realized_pnl_usd")) * 0.2)
        + (_safe_float(payload.get("avg_realized_edge_bps")) * 0.35)
        + (_safe_float(payload.get("avg_edge_retention_ratio")) * 4.0)
        + (_safe_float(payload.get("positive_walk_forward_ratio")) * 2.0)
        - (_safe_float(payload.get("reject_rate")) * 5.0)
        - (_safe_float(payload.get("avg_slippage_bps")) * 0.1)
        - (_safe_float(payload.get("drawdown_to_pnl_ratio")) * 2.0),
        6,
    )


def _execution_metrics_delta(
    *,
    candidate_metrics: dict[str, object],
    current_metrics: dict[str, object],
) -> dict[str, object]:
    metric_names = (
        "live_order_count",
        "accepted_live_order_count",
        "rejected_live_order_count",
        "closed_trade_count",
        "total_realized_pnl_usd",
        "avg_edge_retention_ratio",
        "avg_slippage_bps",
        "avg_realized_edge_bps",
        "avg_expected_edge_bps",
        "reject_rate",
        "drawdown_to_pnl_ratio",
        "positive_walk_forward_ratio",
        "run_count",
        "walk_forward_window_count",
    )
    return {
        f"{metric_name}_delta": round(
            _safe_float(candidate_metrics.get(metric_name)) - _safe_float(current_metrics.get(metric_name)),
            6,
        )
        for metric_name in metric_names
    }


def _execution_replay_summary_from_runs(
    *,
    validation_runs: list[dict[str, object]],
    walk_forward_windows: list[dict[str, object]],
    symbol_summary: list[dict[str, object]] | None,
    regime_summary: list[dict[str, object]] | None,
    policy_application: dict[str, object],
    baseline_policy_application: dict[str, object],
    source: str,
    runtime_summary: dict[str, Any] | None = None,
    runtime_summary_run_dir: Path | None = None,
) -> dict[str, object]:
    runtime_summary_closed_trade_count, runtime_summary_realized_pnl_usd = _runtime_summary_closed_trade_metrics(
        runtime_summary,
        run_dir=runtime_summary_run_dir,
    )
    candidate_pressure = _policy_application_pressure(policy_application)
    baseline_pressure = _policy_application_pressure(baseline_policy_application)
    delta_pressure = round(candidate_pressure - baseline_pressure, 6)
    projected_runs = [
        _project_validation_run(run_snapshot, delta_pressure=delta_pressure)
        for run_snapshot in list(validation_runs or [])
    ]
    projected_windows = [
        _project_walk_forward_window(window, delta_pressure=delta_pressure)
        for window in list(walk_forward_windows or [])
    ]
    retention_window = _retention_window_signal(
        validation_runs=projected_runs,
        walk_forward_windows=projected_windows,
    )
    recent_retention_window = _retention_window_signal(
        validation_runs=projected_runs,
        walk_forward_windows=projected_windows,
        window_size=3,
    )
    total_realized_pnl_usd = round(sum(_safe_float(item.get("realized_pnl_usd")) for item in projected_runs), 6)
    live_order_count = sum(_safe_int(item.get("live_order_count")) for item in projected_runs)
    rejected_live_order_count = sum(_safe_int(item.get("rejected_live_order_count")) for item in projected_runs)
    closed_trade_count = sum(_safe_int(item.get("closed_trade_count")) for item in projected_runs)
    avg_slippage_bps = round(_safe_float(retention_window.get("avg_slippage_bps")), 6)
    avg_realized_edge_bps = _weighted_metric(
        sum(_safe_float(item.get("avg_realized_edge_bps")) * _safe_int(item.get("live_order_count")) for item in projected_runs),
        float(max(live_order_count, 0)),
    )
    avg_expected_edge_bps = _weighted_metric(
        sum(_safe_float(item.get("avg_expected_edge_bps")) * _safe_int(item.get("live_order_count")) for item in projected_runs),
        float(max(live_order_count, 0)),
    )
    micro_live_gate = _build_micro_live_gate(
        live_order_count=live_order_count,
        rejected_live_order_count=rejected_live_order_count,
        avg_slippage_bps=avg_slippage_bps,
        avg_realized_edge_bps=avg_realized_edge_bps,
        closed_trade_count=closed_trade_count,
    )
    execution_metrics = {
        "run_count": _safe_int(retention_window.get("run_count")),
        "walk_forward_window_count": _safe_int(retention_window.get("walk_forward_window_count")),
        "live_order_count": live_order_count,
        "accepted_live_order_count": sum(_safe_int(item.get("accepted_live_order_count")) for item in projected_runs),
        "rejected_live_order_count": rejected_live_order_count,
        "closed_trade_count": closed_trade_count,
        "total_realized_pnl_usd": total_realized_pnl_usd,
        "avg_edge_retention_ratio": round(_safe_float(retention_window.get("avg_edge_retention_ratio")), 6),
        "avg_slippage_bps": avg_slippage_bps,
        "avg_realized_edge_bps": avg_realized_edge_bps,
        "avg_expected_edge_bps": avg_expected_edge_bps,
        "reject_rate": round(_safe_float(retention_window.get("reject_rate")), 6),
        "drawdown_to_pnl_ratio": round(_safe_float(retention_window.get("drawdown_to_pnl_ratio")), 6),
        "positive_walk_forward_ratio": round(_safe_float(retention_window.get("positive_walk_forward_ratio")), 6),
    }
    runtime_summary_anchor = {
        "source": "current_runtime_summary" if runtime_summary else "artifact_only",
        "live_order_count": _safe_int(dict(runtime_summary or {}).get("live_order_count")),
        "closed_trade_count": runtime_summary_closed_trade_count,
        "accepted_live_order_count": _safe_int(dict(runtime_summary or {}).get("accepted_live_order_count")),
        "rejected_live_order_count": _safe_int(dict(runtime_summary or {}).get("rejected_live_order_count")),
        "avg_edge_retention_ratio": round(_safe_float(dict(runtime_summary or {}).get("avg_edge_retention_ratio")), 6),
        "avg_realized_edge_bps": round(_safe_float(dict(runtime_summary or {}).get("avg_realized_edge_bps")), 6),
        "avg_slippage_bps": round(_safe_float(dict(runtime_summary or {}).get("avg_slippage_bps")), 6),
        "realized_pnl_usd": runtime_summary_realized_pnl_usd,
    }
    summary = {
        "source": source,
        "policy_pressure": candidate_pressure,
        "baseline_policy_pressure": baseline_pressure,
        "delta_policy_pressure": delta_pressure,
        "run_count": execution_metrics["run_count"],
        "walk_forward_window_count": execution_metrics["walk_forward_window_count"],
        "positive_walk_forward_window_count": sum(
            1
            for item in projected_windows
            if _safe_float(item.get("avg_net_edge_bps")) > 0.0 and _safe_float(item.get("avg_score")) >= 0.0
        ),
        "positive_walk_forward_ratio": execution_metrics["positive_walk_forward_ratio"],
        "micro_live_gate": micro_live_gate,
        "top_symbols": list(symbol_summary or [])[:3],
        "top_regimes": list(regime_summary or [])[:3],
        "recent_retention_window": recent_retention_window,
        "cumulative_retention_window": retention_window,
        "execution_metrics": execution_metrics,
        "execution_score": _execution_replay_score(execution_metrics),
        "runtime_summary_anchor": runtime_summary_anchor,
    }
    summary["replay_provenance"] = replay_summary_provenance(summary)
    return summary


def _execution_replay_summary_from_bucket_evidence(
    *,
    bucket_evidence: dict[str, Any] | None,
    policy_application: dict[str, object],
    source: str,
    runtime_summary: dict[str, Any] | None = None,
    runtime_summary_run_dir: Path | None = None,
) -> dict[str, object]:
    bucket = dict(bucket_evidence or {})
    validation_runs = [
        dict(item)
        for item in list(
            bucket.get(
                "policy_context_bucket_validation_runs",
                bucket.get("validation_runs", []),
            )
            or []
        )
        if isinstance(item, dict)
    ]
    walk_forward_windows = [
        dict(item)
        for item in list(
            bucket.get(
                "policy_context_bucket_walk_forward_windows",
                bucket.get("walk_forward_windows", []),
            )
            or []
        )
        if isinstance(item, dict)
    ]
    symbol_summary = [
        dict(item)
        for item in list(
            bucket.get(
                "policy_context_bucket_symbol_summary",
                bucket.get("symbol_summary", []),
            )
            or []
        )
        if isinstance(item, dict)
    ]
    regime_summary = [
        dict(item)
        for item in list(
            bucket.get(
                "policy_context_bucket_regime_summary",
                bucket.get("regime_summary", []),
            )
            or []
        )
        if isinstance(item, dict)
    ]
    if not bucket and not validation_runs and not walk_forward_windows and not symbol_summary and not regime_summary:
        return {}
    summary = _execution_replay_summary_from_runs(
        validation_runs=validation_runs,
        walk_forward_windows=walk_forward_windows,
        symbol_summary=symbol_summary,
        regime_summary=regime_summary,
        policy_application=policy_application,
        baseline_policy_application=policy_application,
        source=source,
        runtime_summary=runtime_summary,
        runtime_summary_run_dir=runtime_summary_run_dir,
    )
    positive_walk_forward_window_count = max(
        _safe_int(summary.get("positive_walk_forward_window_count")),
        _safe_int(bucket.get("runner_positive_walk_forward_window_count")),
    )
    execution_metrics = dict(summary.get("execution_metrics", {}) or {})
    execution_metrics.update(
        {
            "run_count": max(
                _safe_int(execution_metrics.get("run_count")),
                _safe_int(bucket.get("policy_context_bucket_run_count", bucket.get("run_count"))),
            ),
            "walk_forward_window_count": max(
                _safe_int(execution_metrics.get("walk_forward_window_count")),
                _safe_int(
                    bucket.get(
                        "policy_context_bucket_walk_forward_window_count",
                        bucket.get("runner_walk_forward_window_count"),
                    )
                ),
            ),
            "live_order_count": max(
                _safe_int(execution_metrics.get("live_order_count")),
                _safe_int(bucket.get("policy_context_bucket_live_order_count", bucket.get("live_order_count"))),
            ),
            "accepted_live_order_count": max(
                _safe_int(execution_metrics.get("accepted_live_order_count")),
                _safe_int(bucket.get("accepted_live_order_count")),
            ),
            "rejected_live_order_count": max(
                _safe_int(execution_metrics.get("rejected_live_order_count")),
                _safe_int(bucket.get("rejected_live_order_count")),
            ),
            "closed_trade_count": max(
                _safe_int(execution_metrics.get("closed_trade_count")),
                _safe_int(
                    bucket.get(
                        "policy_context_bucket_closed_trade_count",
                        bucket.get("closed_trade_count"),
                    )
                ),
            ),
            "total_realized_pnl_usd": round(
                _safe_float(
                    bucket.get(
                        "runner_total_realized_pnl_usd",
                        bucket.get(
                            "policy_context_bucket_total_realized_pnl_usd",
                            execution_metrics.get("total_realized_pnl_usd"),
                        ),
                    )
                ),
                6,
            ),
            "avg_edge_retention_ratio": round(
                _safe_float(
                    bucket.get(
                        "runner_avg_edge_retention_ratio",
                        bucket.get(
                            "avg_edge_retention_ratio",
                            execution_metrics.get("avg_edge_retention_ratio"),
                        ),
                    )
                ),
                6,
            ),
            "avg_slippage_bps": round(
                _safe_float(
                    bucket.get(
                        "runner_avg_slippage_bps",
                        bucket.get("avg_slippage_bps", execution_metrics.get("avg_slippage_bps")),
                    )
                ),
                6,
            ),
            "avg_realized_edge_bps": round(
                _safe_float(
                    bucket.get(
                        "runner_avg_realized_edge_bps",
                        bucket.get(
                            "avg_realized_edge_bps",
                            execution_metrics.get("avg_realized_edge_bps"),
                        ),
                    )
                ),
                6,
            ),
            "avg_expected_edge_bps": round(
                _safe_float(
                    bucket.get("avg_expected_edge_bps", execution_metrics.get("avg_expected_edge_bps"))
                ),
                6,
            ),
            "reject_rate": round(
                _safe_float(
                    bucket.get(
                        "runner_reject_rate",
                        bucket.get("reject_rate", execution_metrics.get("reject_rate")),
                    )
                ),
                6,
            ),
            "drawdown_to_pnl_ratio": round(
                _safe_float(
                    bucket.get(
                        "runner_drawdown_to_pnl_ratio",
                        execution_metrics.get("drawdown_to_pnl_ratio"),
                    )
                ),
                6,
            ),
            "positive_walk_forward_ratio": round(
                _safe_float(
                    bucket.get(
                        "policy_context_bucket_positive_walk_forward_ratio",
                        bucket.get(
                            "runner_positive_walk_forward_ratio",
                            execution_metrics.get("positive_walk_forward_ratio"),
                        ),
                    )
                ),
                6,
            ),
        }
    )
    live_order_count = _safe_int(execution_metrics.get("live_order_count"))
    rejected_live_order_count = _safe_int(execution_metrics.get("rejected_live_order_count"))
    closed_trade_count = _safe_int(execution_metrics.get("closed_trade_count"))
    summary.update(
        {
            "source": source,
            "bucket_name": str(bucket.get("policy_context_bucket_name", "") or ""),
            "run_count": _safe_int(execution_metrics.get("run_count")),
            "walk_forward_window_count": _safe_int(execution_metrics.get("walk_forward_window_count")),
            "positive_walk_forward_window_count": positive_walk_forward_window_count,
            "positive_walk_forward_ratio": round(
                _safe_float(execution_metrics.get("positive_walk_forward_ratio")),
                6,
            ),
            "micro_live_gate": _build_micro_live_gate(
                live_order_count=live_order_count,
                rejected_live_order_count=rejected_live_order_count,
                avg_slippage_bps=_safe_float(execution_metrics.get("avg_slippage_bps")),
                avg_realized_edge_bps=_safe_float(execution_metrics.get("avg_realized_edge_bps")),
                closed_trade_count=closed_trade_count,
            ),
            "top_symbols": symbol_summary[:3],
            "top_regimes": regime_summary[:3],
            "execution_metrics": execution_metrics,
            "execution_score": _execution_replay_score(execution_metrics),
        }
    )
    summary["replay_provenance"] = replay_summary_provenance(summary)
    return summary


def build_weekly_validation_report(*, base_dir: str | Path = "quant_runtime", lookback_days: int = 7) -> WeeklyValidationReport:
    root = Path(base_dir)
    runs = _resolve_recent_runs(base_dir=root, lookback_days=lookback_days)
    generated_at = datetime.now(timezone.utc).isoformat()
    if not runs:
        return WeeklyValidationReport(
            base_dir=str(root),
            generated_at=generated_at,
            lookback_days=lookback_days,
            run_count=0,
            period_start="",
            period_end="",
            total_closed_trade_count=0,
            total_realized_pnl_usd=0.0,
            total_live_order_count=0,
            total_tested_order_count=0,
            sample_progress={
                "status": "no_data",
                "required_total_closed_trade_count": 6,
                "required_total_live_order_count": 8,
                "remaining_closed_trade_count": 6,
                "remaining_live_order_count": 8,
                "ready_for_comparison": False,
            },
            score_alignment_summary=(),
            symbol_summary=(),
            symbol_scorecard=(),
            regime_summary=(),
            criteria=_criteria_table(),
        )

    symbol_buckets: dict[str, dict[str, float | int]] = {}
    symbol_run_history: dict[str, list[dict[str, object]]] = {}
    regime_buckets: dict[str, dict[str, float | int]] = {}
    score_buckets: dict[str, dict[str, float | int]] = {}
    total_closed_trade_count = 0
    total_realized_pnl = 0.0
    total_live_orders = 0
    total_tested_orders = 0

    for run_dir in runs:
        report = build_runtime_performance_report(run_dir=run_dir)
        summary = _load_summary(run_dir)
        total_closed_trade_count += report.closed_trade_count
        total_realized_pnl += report.realized_pnl_usd
        total_live_orders += int(summary.get("live_order_count") or 0)
        total_tested_orders += int(summary.get("tested_order_count") or 0)

        for row in report.symbol_expectancy:
            bucket = symbol_buckets.setdefault(
                row.symbol,
                {
                    "trade_count": 0,
                    "realized_pnl_usd": 0.0,
                    "expectancy_weighted_sum": 0.0,
                    "win_count": 0,
                    "loss_count": 0,
                },
            )
            history = symbol_run_history.setdefault(row.symbol, [])
            history.append(
                {
                    "run_id": run_dir.name,
                    "trade_count": row.trade_count,
                    "expectancy_usd": row.expectancy_usd,
                    "realized_pnl_usd": row.realized_pnl_usd,
                }
            )
            bucket["trade_count"] = int(bucket["trade_count"]) + row.trade_count
            bucket["realized_pnl_usd"] = float(bucket["realized_pnl_usd"]) + row.realized_pnl_usd
            bucket["expectancy_weighted_sum"] = float(bucket["expectancy_weighted_sum"]) + (row.expectancy_usd * max(row.trade_count, 1))
            bucket["win_count"] = int(bucket["win_count"]) + row.win_count
            bucket["loss_count"] = int(bucket["loss_count"]) + row.loss_count

        for row in report.score_bucket_performance:
            bucket = score_buckets.setdefault(
                row.score_bucket_label,
                {
                    "trade_count": 0,
                    "win_count": 0,
                    "loss_count": 0,
                    "realized_pnl_usd": 0.0,
                    "average_return_bps_weighted_sum": 0.0,
                },
            )
            bucket["trade_count"] = int(bucket["trade_count"]) + row.trade_count
            bucket["win_count"] = int(bucket["win_count"]) + row.win_count
            bucket["loss_count"] = int(bucket["loss_count"]) + row.loss_count
            bucket["realized_pnl_usd"] = float(bucket["realized_pnl_usd"]) + row.realized_pnl_usd
            bucket["average_return_bps_weighted_sum"] = float(bucket["average_return_bps_weighted_sum"]) + (row.average_return_bps * max(row.trade_count, 1))

        for row in report.regime_performance:
            bucket = regime_buckets.setdefault(
                row.mode,
                {
                    "decision_count": 0,
                    "score_sum": 0.0,
                    "net_edge_sum": 0.0,
                    "cost_sum": 0.0,
                },
            )
            bucket["decision_count"] = int(bucket["decision_count"]) + row.decision_count
            bucket["score_sum"] = float(bucket["score_sum"]) + (row.avg_score * row.decision_count)
            bucket["net_edge_sum"] = float(bucket["net_edge_sum"]) + (row.avg_net_edge_bps * row.decision_count)
            bucket["cost_sum"] = float(bucket["cost_sum"]) + (row.avg_cost_bps * row.decision_count)

    symbol_rows: list[dict[str, object]] = []
    required_symbol_trade_count = 3
    for symbol, bucket in symbol_buckets.items():
        trade_count = int(bucket["trade_count"])
        expectancy = float(bucket["expectancy_weighted_sum"]) / max(trade_count, 1)
        pnl = float(bucket["realized_pnl_usd"])
        recommendation = "keep"
        sample_status = "warming_up"
        remaining_trade_count = max(required_symbol_trade_count - trade_count, 0)
        if trade_count >= 3 and expectancy < 0:
            recommendation = "prune"
            sample_status = "validated_negative"
        elif trade_count == 0:
            recommendation = "observe_only"
            sample_status = "insufficient_symbol_data"
        elif trade_count >= 3 and expectancy > 0 and pnl > 0:
            recommendation = "promote"
            sample_status = "validated_positive"
        elif trade_count >= 3:
            sample_status = "validated_mixed"
        symbol_rows.append(
            {
                "symbol": symbol,
                "trade_count": trade_count,
                "realized_pnl_usd": round(pnl, 6),
                "expectancy_usd": round(expectancy, 6),
                "win_count": int(bucket["win_count"]),
                "loss_count": int(bucket["loss_count"]),
                "recommendation": recommendation,
                "sample_status": sample_status,
                "remaining_trade_count_for_validation": remaining_trade_count,
                "required_trade_count_for_validation": required_symbol_trade_count,
                "rolling_evidence": _symbol_rolling_evidence(
                    aggregate_expectancy=expectancy,
                    history=list(symbol_run_history.get(symbol, [])),
                ),
            }
        )
    symbol_rows.sort(key=lambda item: (str(item["recommendation"]), float(item["expectancy_usd"])))
    symbol_scorecard_rows = _build_symbol_scorecard(symbol_rows)

    score_alignment_rows: list[dict[str, object]] = []
    for label, bucket in score_buckets.items():
        trade_count = int(bucket["trade_count"])
        realized_pnl = float(bucket["realized_pnl_usd"])
        expectancy = realized_pnl / max(trade_count, 1)
        score_alignment_rows.append(
            {
                "score_bucket_label": label,
                "trade_count": trade_count,
                "win_count": int(bucket["win_count"]),
                "loss_count": int(bucket["loss_count"]),
                "hit_rate": round(int(bucket["win_count"]) / max(trade_count, 1), 6),
                "realized_pnl_usd": round(realized_pnl, 6),
                "expectancy_usd": round(expectancy, 6),
                "average_return_bps": round(float(bucket["average_return_bps_weighted_sum"]) / max(trade_count, 1), 6),
            }
        )
    score_alignment_rows.sort(key=lambda item: str(item["score_bucket_label"]))

    regime_rows: list[dict[str, object]] = []
    for mode, bucket in regime_buckets.items():
        count = int(bucket["decision_count"])
        regime_rows.append(
            {
                "mode": mode,
                "decision_count": count,
                "avg_score": round(float(bucket["score_sum"]) / max(count, 1), 6),
                "avg_net_edge_bps": round(float(bucket["net_edge_sum"]) / max(count, 1), 6),
                "avg_cost_bps": round(float(bucket["cost_sum"]) / max(count, 1), 6),
            }
        )
    regime_rows.sort(key=lambda item: str(item["mode"]))

    required_total_closed_trade_count = 6
    required_total_live_order_count = 8
    sample_progress = {
        "status": (
            "ready_for_comparison"
            if total_closed_trade_count >= required_total_closed_trade_count and total_live_orders >= required_total_live_order_count
            else "collecting_evidence"
        ),
        "required_total_closed_trade_count": required_total_closed_trade_count,
        "required_total_live_order_count": required_total_live_order_count,
        "remaining_closed_trade_count": max(required_total_closed_trade_count - total_closed_trade_count, 0),
        "remaining_live_order_count": max(required_total_live_order_count - total_live_orders, 0),
        "ready_for_comparison": total_closed_trade_count >= required_total_closed_trade_count and total_live_orders >= required_total_live_order_count,
    }

    return WeeklyValidationReport(
        base_dir=str(root),
        generated_at=generated_at,
        lookback_days=lookback_days,
        run_count=len(runs),
        period_start=datetime.fromtimestamp(runs[0].stat().st_mtime, tz=timezone.utc).isoformat(),
        period_end=datetime.fromtimestamp(runs[-1].stat().st_mtime, tz=timezone.utc).isoformat(),
        total_closed_trade_count=total_closed_trade_count,
        total_realized_pnl_usd=round(total_realized_pnl, 6),
        total_live_order_count=total_live_orders,
        total_tested_order_count=total_tested_orders,
        sample_progress=sample_progress,
        score_alignment_summary=tuple(score_alignment_rows),
        symbol_summary=tuple(symbol_rows),
        symbol_scorecard=tuple(symbol_scorecard_rows),
        regime_summary=tuple(regime_rows),
        criteria=_criteria_table(),
    )


def _aggregate_weekly_validation_from_run_snapshots(
    *,
    base_dir: str | Path,
    run_snapshots: list[dict[str, object]],
    lookback_days: int,
    generated_at: str,
) -> dict[str, object]:
    symbol_buckets: dict[str, dict[str, float | int]] = {}
    symbol_run_history: dict[str, list[dict[str, object]]] = {}
    regime_buckets: dict[str, dict[str, float | int]] = {}
    score_buckets: dict[str, dict[str, float | int]] = {}
    total_closed_trade_count = 0
    total_realized_pnl = 0.0
    total_live_orders = 0
    total_tested_orders = 0
    for snapshot in run_snapshots:
        total_closed_trade_count += _safe_int(snapshot.get("closed_trade_count"))
        total_realized_pnl += _safe_float(snapshot.get("realized_pnl_usd"))
        total_live_orders += _safe_int(snapshot.get("live_order_count"))
        total_tested_orders += _safe_int(snapshot.get("tested_order_count"))
        for row in list(snapshot.get("symbol_expectancy", []) or []):
            symbol = str(row.get("symbol", "") or "")
            if not symbol:
                continue
            bucket = symbol_buckets.setdefault(
                symbol,
                {
                    "trade_count": 0,
                    "realized_pnl_usd": 0.0,
                    "expectancy_weighted_sum": 0.0,
                    "win_count": 0,
                    "loss_count": 0,
                },
            )
            history = symbol_run_history.setdefault(symbol, [])
            history.append(
                {
                    "run_id": str(snapshot.get("run_id", "") or ""),
                    "trade_count": _safe_int(row.get("trade_count")),
                    "expectancy_usd": _safe_float(row.get("expectancy_usd")),
                    "realized_pnl_usd": _safe_float(row.get("realized_pnl_usd")),
                }
            )
            trade_count = _safe_int(row.get("trade_count"))
            bucket["trade_count"] = int(bucket["trade_count"]) + trade_count
            bucket["realized_pnl_usd"] = float(bucket["realized_pnl_usd"]) + _safe_float(row.get("realized_pnl_usd"))
            bucket["expectancy_weighted_sum"] = float(bucket["expectancy_weighted_sum"]) + (_safe_float(row.get("expectancy_usd")) * max(trade_count, 1))
            bucket["win_count"] = int(bucket["win_count"]) + _safe_int(row.get("win_count"))
            bucket["loss_count"] = int(bucket["loss_count"]) + _safe_int(row.get("loss_count"))
        for row in list(snapshot.get("score_bucket_performance", []) or []):
            label = str(row.get("score_bucket_label", "") or "")
            if not label:
                continue
            bucket = score_buckets.setdefault(
                label,
                {
                    "trade_count": 0,
                    "win_count": 0,
                    "loss_count": 0,
                    "realized_pnl_usd": 0.0,
                    "average_return_bps_weighted_sum": 0.0,
                },
            )
            trade_count = _safe_int(row.get("trade_count"))
            bucket["trade_count"] = int(bucket["trade_count"]) + trade_count
            bucket["win_count"] = int(bucket["win_count"]) + _safe_int(row.get("win_count"))
            bucket["loss_count"] = int(bucket["loss_count"]) + _safe_int(row.get("loss_count"))
            bucket["realized_pnl_usd"] = float(bucket["realized_pnl_usd"]) + _safe_float(row.get("realized_pnl_usd"))
            bucket["average_return_bps_weighted_sum"] = float(bucket["average_return_bps_weighted_sum"]) + (_safe_float(row.get("average_return_bps")) * max(trade_count, 1))
        for row in list(snapshot.get("regime_performance", []) or []):
            mode = str(row.get("mode", "") or "")
            if not mode:
                continue
            bucket = regime_buckets.setdefault(
                mode,
                {
                    "decision_count": 0,
                    "score_sum": 0.0,
                    "net_edge_sum": 0.0,
                    "cost_sum": 0.0,
                },
            )
            decision_count = _safe_int(row.get("decision_count"))
            bucket["decision_count"] = int(bucket["decision_count"]) + decision_count
            bucket["score_sum"] = float(bucket["score_sum"]) + (_safe_float(row.get("avg_score")) * decision_count)
            bucket["net_edge_sum"] = float(bucket["net_edge_sum"]) + (_safe_float(row.get("avg_net_edge_bps")) * decision_count)
            bucket["cost_sum"] = float(bucket["cost_sum"]) + (_safe_float(row.get("avg_cost_bps")) * decision_count)
    symbol_rows: list[dict[str, object]] = []
    required_symbol_trade_count = 3
    for symbol, bucket in symbol_buckets.items():
        trade_count = int(bucket["trade_count"])
        expectancy = float(bucket["expectancy_weighted_sum"]) / max(trade_count, 1)
        pnl = float(bucket["realized_pnl_usd"])
        recommendation = "keep"
        sample_status = "warming_up"
        remaining_trade_count = max(required_symbol_trade_count - trade_count, 0)
        if trade_count >= 3 and expectancy < 0:
            recommendation = "prune"
            sample_status = "validated_negative"
        elif trade_count == 0:
            recommendation = "observe_only"
            sample_status = "insufficient_symbol_data"
        elif trade_count >= 3 and expectancy > 0 and pnl > 0:
            recommendation = "promote"
            sample_status = "validated_positive"
        elif trade_count >= 3:
            sample_status = "validated_mixed"
        symbol_rows.append(
            {
                "symbol": symbol,
                "trade_count": trade_count,
                "realized_pnl_usd": round(pnl, 6),
                "expectancy_usd": round(expectancy, 6),
                "win_count": int(bucket["win_count"]),
                "loss_count": int(bucket["loss_count"]),
                "recommendation": recommendation,
                "sample_status": sample_status,
                "remaining_trade_count_for_validation": remaining_trade_count,
                "required_trade_count_for_validation": required_symbol_trade_count,
                "rolling_evidence": _symbol_rolling_evidence(
                    aggregate_expectancy=expectancy,
                    history=list(symbol_run_history.get(symbol, [])),
                ),
            }
        )
    symbol_rows.sort(key=lambda item: (str(item["recommendation"]), float(item["expectancy_usd"])))
    symbol_scorecard_rows = _build_symbol_scorecard(symbol_rows)
    score_alignment_rows: list[dict[str, object]] = []
    for label, bucket in score_buckets.items():
        trade_count = int(bucket["trade_count"])
        realized_pnl = float(bucket["realized_pnl_usd"])
        expectancy = realized_pnl / max(trade_count, 1)
        score_alignment_rows.append(
            {
                "score_bucket_label": label,
                "trade_count": trade_count,
                "win_count": int(bucket["win_count"]),
                "loss_count": int(bucket["loss_count"]),
                "hit_rate": round(int(bucket["win_count"]) / max(trade_count, 1), 6),
                "realized_pnl_usd": round(realized_pnl, 6),
                "expectancy_usd": round(expectancy, 6),
                "average_return_bps": round(float(bucket["average_return_bps_weighted_sum"]) / max(trade_count, 1), 6),
            }
        )
    score_alignment_rows.sort(key=lambda item: str(item["score_bucket_label"]))
    regime_rows: list[dict[str, object]] = []
    for mode, bucket in regime_buckets.items():
        count = int(bucket["decision_count"])
        regime_rows.append(
            {
                "mode": mode,
                "decision_count": count,
                "avg_score": round(float(bucket["score_sum"]) / max(count, 1), 6),
                "avg_net_edge_bps": round(float(bucket["net_edge_sum"]) / max(count, 1), 6),
                "avg_cost_bps": round(float(bucket["cost_sum"]) / max(count, 1), 6),
            }
        )
    regime_rows.sort(key=lambda item: str(item["mode"]))
    required_total_closed_trade_count = 6
    required_total_live_order_count = 8
    return {
        "base_dir": str(base_dir),
        "generated_at": generated_at,
        "lookback_days": lookback_days,
        "run_count": len(run_snapshots),
        "total_closed_trade_count": total_closed_trade_count,
        "total_realized_pnl_usd": round(total_realized_pnl, 6),
        "total_live_order_count": total_live_orders,
        "total_tested_order_count": total_tested_orders,
        "sample_progress": {
            "status": (
                "ready_for_comparison"
                if total_closed_trade_count >= required_total_closed_trade_count and total_live_orders >= required_total_live_order_count
                else "collecting_evidence"
            ),
            "required_total_closed_trade_count": required_total_closed_trade_count,
            "required_total_live_order_count": required_total_live_order_count,
            "remaining_closed_trade_count": max(required_total_closed_trade_count - total_closed_trade_count, 0),
            "remaining_live_order_count": max(required_total_live_order_count - total_live_orders, 0),
            "ready_for_comparison": total_closed_trade_count >= required_total_closed_trade_count and total_live_orders >= required_total_live_order_count,
        },
        "score_alignment_summary": score_alignment_rows,
        "symbol_summary": symbol_rows,
        "symbol_scorecard": symbol_scorecard_rows,
        "regime_summary": regime_rows,
    }


def _filter_lineage_aligned_run_snapshots(
    *,
    run_snapshots: list[dict[str, object]],
    active_policy_lineage: dict[str, object] | None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    target = dict(active_policy_lineage or {})
    annotated: list[dict[str, object]] = []
    aligned_rows: list[dict[str, object]] = []
    known_count = 0
    mismatch_count = 0
    unknown_count = 0
    for snapshot in run_snapshots:
        payload = dict(snapshot)
        observed = dict(payload.get("policy_lineage", {}) or {})
        alignment = policy_lineage_alignment(target, observed)
        payload["policy_lineage_alignment"] = alignment
        annotated.append(payload)
        if bool(observed.get("available")):
            known_count += 1
        else:
            unknown_count += 1
        if bool(alignment.get("aligned")):
            aligned_rows.append(payload)
        elif str(alignment.get("status", "")) == "mismatch":
            mismatch_count += 1
    if not bool(target.get("available")):
        return annotated, {
            "applied": False,
            "mode": "unfiltered_no_active_lineage",
            "active_policy_lineage": target,
            "known_run_lineage_count": known_count,
            "unknown_run_lineage_count": unknown_count,
            "aligned_run_count": len(aligned_rows),
            "mismatched_run_count": mismatch_count,
        }
    if known_count <= 0:
        return annotated, {
            "applied": False,
            "mode": "unfiltered_no_derived_run_lineage",
            "active_policy_lineage": target,
            "known_run_lineage_count": known_count,
            "unknown_run_lineage_count": unknown_count,
            "aligned_run_count": len(aligned_rows),
            "mismatched_run_count": mismatch_count,
        }
    return aligned_rows, {
        "applied": True,
        "mode": "filtered_to_active_lineage",
        "active_policy_lineage": target,
        "known_run_lineage_count": known_count,
        "unknown_run_lineage_count": unknown_count,
        "aligned_run_count": len(aligned_rows),
        "mismatched_run_count": mismatch_count,
    }


def _lineage_guarded_baseline_control_comparison(
    baseline_control_comparison: dict[str, object],
    lineage_attribution: dict[str, object],
) -> dict[str, object]:
    payload = dict(baseline_control_comparison or {})
    if not payload:
        return payload
    if not bool(lineage_attribution.get("applied")):
        payload["lineage_scope_status"] = str(lineage_attribution.get("mode", "unfiltered") or "unfiltered")
        return payload
    if int(lineage_attribution.get("aligned_run_count", 0) or 0) > 0:
        payload["lineage_scope_status"] = "aligned"
        return payload
    return {
        "available": False,
        "artifact_path": str(payload.get("artifact_path", "") or ""),
        "reason": "BASELINE_CONTROL_REQUIRES_ALIGNED_POLICY_LINEAGE",
        "verdict": "not_available",
        "expansion_gate": "not_available",
        "expansion_gate_reason": "BASELINE_CONTROL_REQUIRES_ALIGNED_POLICY_LINEAGE",
        "lineage_scope_status": "no_aligned_policy_runs",
    }


def _build_policy_validation_runner_from_run_snapshots(
    *,
    base_dir: str | Path,
    run_snapshots: list[dict[str, object]],
    lookback_days: int,
    generated_at: str,
    baseline_control_comparison: dict[str, object] | None = None,
    lineage_attribution: dict[str, object] | None = None,
) -> dict[str, object]:
    aggregated = _aggregate_weekly_validation_from_run_snapshots(
        base_dir=base_dir,
        run_snapshots=run_snapshots,
        lookback_days=lookback_days,
        generated_at=generated_at,
    )
    policy_context_bucket_evidence = _policy_context_bucket_direct_evidence_map(
        base_dir=base_dir,
        lookback_days=lookback_days,
        generated_at=generated_at,
        run_snapshots=run_snapshots,
    )
    lifecycle_bucket_evidence = dict(
        policy_context_bucket_evidence.get("active_policy", {})
        or policy_context_bucket_evidence.get("staged_candidate", {})
        or {}
    )
    symbol_rows = list(aggregated.get("symbol_summary", []) or [])
    symbol_scorecard = list(aggregated.get("symbol_scorecard", []) or [])
    regime_rows = list(aggregated.get("regime_summary", []) or [])
    pruning_recommendations = _aggregate_pruning_recommendations(run_snapshots)
    walk_forward_windows = [window for snapshot in run_snapshots for window in list(snapshot.get("walk_forward", []) or [])]
    positive_walk_forward_count = sum(
        1
        for window in walk_forward_windows
        if _safe_float(window.get("avg_net_edge_bps")) > 0.0 and _safe_float(window.get("avg_score")) >= 0.0
    )
    promote_count = sum(1 for row in symbol_rows if str(row.get("recommendation", "")) == "promote")
    prune_count = sum(1 for row in symbol_rows if str(row.get("recommendation", "")) == "prune")
    total_symbols = max(len(symbol_rows), 1)
    walk_forward_alignment = (
        positive_walk_forward_count / len(walk_forward_windows)
        if walk_forward_windows
        else (1.0 if int(aggregated.get("run_count", 0) or 0) > 0 and _safe_float(aggregated.get("total_realized_pnl_usd")) > 0.0 else 0.0)
    )
    shadow_alignment_score = max(
        0.0,
        min(1.0, ((promote_count - prune_count + total_symbols) / (2 * total_symbols) * 0.5) + (walk_forward_alignment * 0.5)),
    )
    pnl_series = [_safe_float(snapshot.get("realized_pnl_usd")) for snapshot in run_snapshots]
    total_realized_pnl_usd = round(sum(pnl_series), 6)
    max_drawdown_usd = _max_drawdown(pnl_series)
    drawdown_to_pnl_ratio = round(max_drawdown_usd / max(abs(total_realized_pnl_usd), 1.0), 6)
    live_order_count = sum(_safe_int(snapshot.get("live_order_count")) for snapshot in run_snapshots)
    rejected_live_order_count = sum(_safe_int(snapshot.get("rejected_live_order_count")) for snapshot in run_snapshots)
    closed_trade_count = sum(_safe_int(snapshot.get("closed_trade_count")) for snapshot in run_snapshots)
    slippage_weight = float(sum(_safe_int(snapshot.get("accepted_live_order_count")) for snapshot in run_snapshots))
    retention_weight = float(sum(_safe_int(snapshot.get("live_order_count")) for snapshot in run_snapshots))
    avg_slippage_bps = _weighted_metric(
        sum(_safe_float(snapshot.get("avg_slippage_bps")) * _safe_int(snapshot.get("accepted_live_order_count")) for snapshot in run_snapshots),
        slippage_weight,
    )
    avg_realized_edge_bps = _weighted_metric(
        sum(_safe_float(snapshot.get("avg_realized_edge_bps")) * _safe_int(snapshot.get("live_order_count")) for snapshot in run_snapshots),
        retention_weight,
    )
    avg_edge_retention_ratio = _weighted_metric(
        sum(_safe_float(snapshot.get("avg_edge_retention_ratio")) * _safe_int(snapshot.get("live_order_count")) for snapshot in run_snapshots),
        retention_weight,
    )
    reject_rate = round(rejected_live_order_count / max(live_order_count, 1), 6) if live_order_count > 0 else 0.0
    protection_degraded_rate = _weighted_metric(
        sum(_safe_float(snapshot.get("protection_degraded_rate")) * _safe_int(snapshot.get("live_order_count")) for snapshot in run_snapshots),
        retention_weight,
    )
    micro_live_gate = _build_micro_live_gate(
        live_order_count=live_order_count,
        rejected_live_order_count=rejected_live_order_count,
        avg_slippage_bps=avg_slippage_bps,
        avg_realized_edge_bps=avg_realized_edge_bps,
        closed_trade_count=closed_trade_count,
    )
    recent_retention_window = _retention_window_signal(
        validation_runs=run_snapshots,
        walk_forward_windows=walk_forward_windows,
        window_size=3,
    )
    cumulative_retention_window = _retention_window_signal(
        validation_runs=run_snapshots,
        walk_forward_windows=walk_forward_windows,
    )
    sample_quality_watchdog = _build_sample_quality_watchdog(
        run_count=int(aggregated.get("run_count", 0) or 0),
        total_closed_trade_count=int(aggregated.get("total_closed_trade_count", 0) or 0),
        total_live_order_count=int(aggregated.get("total_live_order_count", 0) or 0),
        total_tested_order_count=int(aggregated.get("total_tested_order_count", 0) or 0),
        total_realized_pnl_usd=total_realized_pnl_usd,
        symbol_rows=symbol_rows,
        symbol_scorecard=symbol_scorecard,
        score_alignment_summary=list(aggregated.get("score_alignment_summary", []) or []),
        runner_reject_rate=reject_rate,
        runner_avg_slippage_bps=avg_slippage_bps,
        runner_avg_realized_edge_bps=avg_realized_edge_bps,
        runner_avg_edge_retention_ratio=avg_edge_retention_ratio,
        runner_protection_degraded_rate=protection_degraded_rate,
        runner_walk_forward_window_count=len(walk_forward_windows),
        runner_positive_walk_forward_ratio=round(_safe_ratio(positive_walk_forward_count, len(walk_forward_windows)), 6),
    )
    guarded_baseline = _lineage_guarded_baseline_control_comparison(
        dict(baseline_control_comparison or {}),
        dict(lineage_attribution or {}),
    )
    symbol_lifecycle = build_symbol_lifecycle(
        symbol_summary=symbol_rows,
        symbol_scorecard=symbol_scorecard,
        pruning_recommendations=pruning_recommendations,
        policy_context_bucket_name=str(lifecycle_bucket_evidence.get("policy_context_bucket_name", "") or ""),
        policy_context_bucket_symbol_summary=list(
            lifecycle_bucket_evidence.get(
                "policy_context_bucket_symbol_summary",
                lifecycle_bucket_evidence.get("symbol_summary", []),
            )
            or []
        ),
        policy_context_bucket_pruning_recommendations=list(
            lifecycle_bucket_evidence.get(
                "policy_context_bucket_pruning_recommendations",
                lifecycle_bucket_evidence.get("pruning_recommendations", []),
            )
            or []
        ),
        sample_quality_watchdog=sample_quality_watchdog,
        baseline_control_comparison=guarded_baseline,
        active_policy={"status": "baseline", "adjustments": []},
        rollout_phase="baseline",
        evaluated_at=generated_at,
    )
    symbol_lifecycle_summary = summarize_symbol_lifecycle(symbol_lifecycle)
    auto_mode = build_regime_aware_auto_mode(
        regime_summary=regime_rows,
        sample_quality_watchdog=sample_quality_watchdog,
        baseline_control_comparison=guarded_baseline,
        execution_quality={
            "run_count": aggregated.get("run_count", 0),
            "total_closed_trade_count": aggregated.get("total_closed_trade_count", 0),
            "total_live_order_count": aggregated.get("total_live_order_count", 0),
            "runner_total_realized_pnl_usd": total_realized_pnl_usd,
            "runner_drawdown_to_pnl_ratio": drawdown_to_pnl_ratio,
            "runner_reject_rate": reject_rate,
            "runner_protection_degraded_rate": protection_degraded_rate,
            "runner_avg_realized_edge_bps": avg_realized_edge_bps,
            "runner_avg_edge_retention_ratio": avg_edge_retention_ratio,
            "runner_walk_forward_window_count": len(walk_forward_windows),
            "runner_positive_walk_forward_ratio": round(_safe_ratio(positive_walk_forward_count, len(walk_forward_windows)), 6),
            "micro_live_gate": micro_live_gate,
            "policy_context_bucket_evidence": policy_context_bucket_evidence,
            "preferred_policy_bucket": str(lifecycle_bucket_evidence.get("policy_context_bucket_name", "") or ""),
        },
        symbol_lifecycle_summary=symbol_lifecycle_summary,
        symbol_lifecycle=symbol_lifecycle,
    )
    total_return_pct = 0.0
    max_drawdown_pct = 0.0
    evidence = {
        "generated_at": generated_at,
        "sample_progress": dict(aggregated.get("sample_progress", {}) or {}),
        "score_alignment_summary": list(aggregated.get("score_alignment_summary", []) or []),
        "total_closed_trade_count": aggregated.get("total_closed_trade_count", 0),
        "total_live_order_count": aggregated.get("total_live_order_count", 0),
        "total_tested_order_count": aggregated.get("total_tested_order_count", 0),
        "runner_total_return_pct": total_return_pct,
        "runner_total_realized_pnl_usd": total_realized_pnl_usd,
        "runner_max_drawdown_pct": max_drawdown_pct,
        "runner_max_drawdown_usd": max_drawdown_usd,
        "runner_drawdown_to_pnl_ratio": drawdown_to_pnl_ratio,
        "runner_shadow_alignment_score": round(shadow_alignment_score, 6),
        "runner_reject_rate": reject_rate,
        "runner_protection_degraded_rate": protection_degraded_rate,
        "runner_avg_slippage_bps": avg_slippage_bps,
        "runner_avg_realized_edge_bps": avg_realized_edge_bps,
        "runner_avg_edge_retention_ratio": avg_edge_retention_ratio,
        "runner_walk_forward_window_count": len(walk_forward_windows),
        "runner_positive_walk_forward_window_count": positive_walk_forward_count,
        "runner_positive_walk_forward_ratio": round(_safe_ratio(positive_walk_forward_count, len(walk_forward_windows)), 6),
        "micro_live_gate": micro_live_gate,
        "recent_retention_window": recent_retention_window,
        "cumulative_retention_window": cumulative_retention_window,
        "sample_quality_watchdog": sample_quality_watchdog,
        "baseline_control_comparison": guarded_baseline,
        "auto_mode": auto_mode,
        "symbol_lifecycle": symbol_lifecycle,
        "symbol_lifecycle_summary": symbol_lifecycle_summary,
        "symbol_scorecard": symbol_scorecard,
        "lineage_attribution": dict(lineage_attribution or {}),
        "policy_context_bucket_evidence": policy_context_bucket_evidence,
    }
    return {
        "generated_at": generated_at,
        "lookback_days": lookback_days,
        "run_count": aggregated.get("run_count", 0),
        "validation_path_mode": "paper_live_walk_forward_artifacts",
        "total_closed_trade_count": aggregated.get("total_closed_trade_count", 0),
        "total_live_order_count": aggregated.get("total_live_order_count", 0),
        "total_tested_order_count": aggregated.get("total_tested_order_count", 0),
        "sample_progress": dict(aggregated.get("sample_progress", {}) or {}),
        "score_alignment_summary": list(aggregated.get("score_alignment_summary", []) or []),
        "runner_total_return_pct": total_return_pct,
        "runner_total_realized_pnl_usd": total_realized_pnl_usd,
        "runner_max_drawdown_pct": max_drawdown_pct,
        "runner_max_drawdown_usd": max_drawdown_usd,
        "runner_drawdown_to_pnl_ratio": drawdown_to_pnl_ratio,
        "runner_shadow_alignment_score": round(shadow_alignment_score, 6),
        "runner_reject_rate": reject_rate,
        "runner_protection_degraded_rate": protection_degraded_rate,
        "runner_avg_slippage_bps": avg_slippage_bps,
        "runner_avg_realized_edge_bps": avg_realized_edge_bps,
        "runner_avg_edge_retention_ratio": avg_edge_retention_ratio,
        "runner_walk_forward_window_count": len(walk_forward_windows),
        "runner_positive_walk_forward_window_count": positive_walk_forward_count,
        "runner_positive_walk_forward_ratio": round(_safe_ratio(positive_walk_forward_count, len(walk_forward_windows)), 6),
        "validation_runs": run_snapshots,
        "walk_forward_windows": walk_forward_windows,
        "symbol_summary": symbol_rows,
        "symbol_scorecard": symbol_scorecard,
        "symbol_lifecycle": symbol_lifecycle,
        "symbol_lifecycle_summary": symbol_lifecycle_summary,
        "regime_summary": regime_rows,
        "pruning_recommendations": pruning_recommendations,
        "policy_context_bucket_evidence": policy_context_bucket_evidence,
        "micro_live_gate": micro_live_gate,
        "recent_retention_window": recent_retention_window,
        "cumulative_retention_window": cumulative_retention_window,
        "sample_quality_watchdog": sample_quality_watchdog,
        "baseline_control_comparison": guarded_baseline,
        "auto_mode": auto_mode,
        "lineage_attribution": dict(lineage_attribution or {}),
        "evidence": evidence,
    }






def _policy_adjustment_score(policy: dict[str, Any]) -> float:
    adjustments = list(dict(policy or {}).get("adjustments", []) or [])
    score = 0.0
    for item in adjustments:
        score += float(item.get("size_multiplier", 1.0) or 1.0) - 1.0
        score += float(item.get("leverage_multiplier", 1.0) or 1.0) - 1.0
        score -= float(item.get("entry_threshold_bps", 0.0) or 0.0) / 10.0
        score -= float(item.get("expected_profit_floor_bps", 0.0) or 0.0) / 10.0
        score += float(item.get("score_delta", 0.0) or 0.0)
    return round(score, 6)


def _replay_summary_from_evidence(
    evidence: dict[str, Any],
    *,
    run_count: int | None = None,
) -> dict[str, object]:
    payload = dict(evidence or {})
    symbol_summary = list(payload.get("symbol_summary", []) or [])
    regime_summary = list(payload.get("regime_summary", []) or [])
    walk_forward_window_count = _safe_int(payload.get("runner_walk_forward_window_count"))
    positive_walk_forward_window_count = _safe_int(payload.get("runner_positive_walk_forward_window_count"))
    return {
        "run_count": _safe_int(run_count if run_count is not None else payload.get("run_count")),
        "walk_forward_window_count": walk_forward_window_count,
        "positive_walk_forward_window_count": positive_walk_forward_window_count,
        "positive_walk_forward_ratio": round(
            _safe_float(
                payload.get(
                    "runner_positive_walk_forward_ratio",
                    _safe_ratio(positive_walk_forward_window_count, walk_forward_window_count),
                )
            ),
            6,
        ),
        "micro_live_gate": dict(payload.get("micro_live_gate", {}) or {}),
        "top_symbols": symbol_summary[:3],
        "top_regimes": regime_summary[:3],
    }


def _replay_summary_available(summary: dict[str, Any]) -> bool:
    payload = dict(summary or {})
    return bool(
        _safe_int(payload.get("run_count")) > 0
        or _safe_int(payload.get("walk_forward_window_count")) > 0
        or list(payload.get("top_symbols", []) or [])
        or list(payload.get("top_regimes", []) or [])
        or bool(dict(payload.get("micro_live_gate", {}) or {}).get("available"))
    )


def _execution_style_path_entry(
    *,
    policy_label: str,
    source: str,
    policy_score: float,
    replay_summary: dict[str, object],
    evidence: dict[str, Any],
    evidence_available: bool,
    policy_application: dict[str, object],
) -> dict[str, object]:
    payload = dict(evidence or {})
    replay_payload = dict(replay_summary or {})
    execution_metrics = dict(replay_payload.get("execution_metrics", {}) or {})
    replay_source = str(replay_payload.get("source", source) or source)
    runtime_summary_anchor = dict(replay_payload.get("runtime_summary_anchor", {}) or {})
    replay_provenance = replay_summary_provenance(replay_payload or {"source": replay_source})
    evidence_basis = {
        "replay_source": replay_source,
        "runtime_summary_anchor_source": str(runtime_summary_anchor.get("source", "") or ""),
        "bucket_name": str(replay_payload.get("bucket_name", "") or ""),
        "validation_run_count": _safe_int(replay_payload.get("run_count")),
        "walk_forward_window_count": _safe_int(replay_payload.get("walk_forward_window_count")),
        "live_order_count": _safe_int(execution_metrics.get("live_order_count")),
        "tested_order_count": _safe_int(execution_metrics.get("tested_order_count")),
        "closed_trade_count": _safe_int(execution_metrics.get("closed_trade_count")),
        "top_symbol_count": len(list(replay_payload.get("top_symbols", []) or [])),
        "top_regime_count": len(list(replay_payload.get("top_regimes", []) or [])),
        "micro_live_status": str(dict(replay_payload.get("micro_live_gate", {}) or {}).get("status", "not_available") or "not_available"),
        "replay_provenance": str(replay_provenance.get("summary", "") or ""),
    }
    execution_path = {
        "policy_label": policy_label,
        "source": replay_source,
        "rollout_phase": str(dict(policy_application or {}).get("rollout_phase", "unknown") or "unknown"),
        "bucket_name": str(replay_payload.get("bucket_name", "") or ""),
        "has_runtime_summary_anchor": bool(runtime_summary_anchor),
        "uses_projected_runtime_replay": replay_source.startswith("projected_"),
        "uses_bucket_log_replay": replay_source.startswith("observed_") or bool(replay_payload.get("bucket_name")),
        "uses_persisted_validation_evidence": replay_source == "persisted_policy_validation_evidence",
        "replay_provenance": str(replay_provenance.get("summary", "") or ""),
    }
    return {
        "policy_label": policy_label,
        "source": replay_source,
        "policy_score": round(policy_score, 6),
        "execution_replay_score": round(_safe_float(replay_payload.get("execution_score")), 6),
        "evidence_available": evidence_available,
        "policy_application": dict(policy_application or {}),
        "micro_live_gate": dict(payload.get("micro_live_gate", {}) or {}),
        "runner_metrics": {
            "total_realized_pnl_usd": round(_safe_float(execution_metrics.get("total_realized_pnl_usd", payload.get("runner_total_realized_pnl_usd"))), 6),
            "drawdown_to_pnl_ratio": round(_safe_float(execution_metrics.get("drawdown_to_pnl_ratio", payload.get("runner_drawdown_to_pnl_ratio"))), 6),
            "reject_rate": round(_safe_float(execution_metrics.get("reject_rate", payload.get("runner_reject_rate"))), 6),
            "avg_slippage_bps": round(_safe_float(execution_metrics.get("avg_slippage_bps", payload.get("runner_avg_slippage_bps"))), 6),
            "avg_realized_edge_bps": round(_safe_float(execution_metrics.get("avg_realized_edge_bps", payload.get("runner_avg_realized_edge_bps"))), 6),
            "avg_edge_retention_ratio": round(_safe_float(execution_metrics.get("avg_edge_retention_ratio", payload.get("runner_avg_edge_retention_ratio"))), 6),
            "positive_walk_forward_ratio": round(_safe_float(execution_metrics.get("positive_walk_forward_ratio", payload.get("runner_positive_walk_forward_ratio"))), 6),
        },
        "execution_metrics": execution_metrics,
        "execution_path": execution_path,
        "replay_evidence_basis": evidence_basis,
        "replay_provenance": replay_provenance,
        "runtime_summary_anchor": runtime_summary_anchor,
        "replay_summary": dict(replay_summary or {}),
    }


def _policy_score_verdict(delta: float) -> str:
    if delta > 0.1:
        return "candidate_better"
    if delta < -0.1:
        return "candidate_worse"
    return "keep"


def _counterfactual_replay_path(
    *,
    validation_mode: str,
    candidate_policy_score: float,
    current_policy_score: float,
    candidate_replay_summary: dict[str, object],
    current_replay_summary: dict[str, object],
    candidate_evidence: dict[str, Any],
    current_evidence: dict[str, Any],
    current_evidence_available: bool,
    candidate_policy_application: dict[str, object],
    current_policy_application: dict[str, object],
) -> dict[str, object]:
    candidate_summary = dict(candidate_replay_summary or {})
    current_summary = dict(current_replay_summary or {})
    score_delta = round(candidate_policy_score - current_policy_score, 6)
    candidate_entry = _execution_style_path_entry(
        policy_label="candidate_policy",
        source="policy_validation_runner_artifact",
        policy_score=candidate_policy_score,
        replay_summary=candidate_summary,
        evidence=candidate_evidence,
        evidence_available=_replay_summary_available(candidate_summary),
        policy_application=candidate_policy_application,
    )
    current_entry = _execution_style_path_entry(
        policy_label="current_policy",
        source="persisted_policy_validation_evidence",
        policy_score=current_policy_score,
        replay_summary=current_summary,
        evidence=current_evidence,
        evidence_available=current_evidence_available and _replay_summary_available(current_summary),
        policy_application=current_policy_application,
    )
    policy_application_delta = _policy_application_delta(
        candidate_profile=candidate_policy_application,
        current_profile=current_policy_application,
    )
    candidate_execution_metrics = dict(candidate_summary.get("execution_metrics", {}) or {})
    current_execution_metrics = dict(current_summary.get("execution_metrics", {}) or {})
    execution_replay_score_delta = round(
        _safe_float(candidate_summary.get("execution_score")) - _safe_float(current_summary.get("execution_score")),
        6,
    )
    execution_replay_verdict = _policy_score_verdict(execution_replay_score_delta)
    return {
        "mode": "counterfactual_current_vs_candidate_policy",
        "validation_mode": validation_mode,
        "execution_style_comparison": {
            "format": "separated_execution_paths",
            "policy_score_delta": score_delta,
            "score_verdict": _policy_score_verdict(score_delta),
            "execution_replay_score_delta": execution_replay_score_delta,
            "execution_replay_verdict": execution_replay_verdict,
            "candidate_path": candidate_entry,
            "current_path": current_entry,
            "comparison_summary": {
                "candidate_evidence_available": bool(candidate_entry.get("evidence_available")),
                "current_evidence_available": bool(current_entry.get("evidence_available")),
                "candidate_micro_live_status": str(candidate_entry["micro_live_gate"].get("status", "not_available") or "not_available"),
                "current_micro_live_status": str(current_entry["micro_live_gate"].get("status", "not_available") or "not_available"),
                "execution_metric_delta": _execution_metrics_delta(
                    candidate_metrics=candidate_execution_metrics,
                    current_metrics=current_execution_metrics,
                ),
                "execution_replay_score_delta": execution_replay_score_delta,
                "execution_replay_verdict": execution_replay_verdict,
                "policy_application_delta": policy_application_delta,
                "policy_application_comparison": {
                    "candidate": dict(candidate_entry.get("policy_application", {}) or {}),
                    "current": dict(current_entry.get("policy_application", {}) or {}),
                    "delta": policy_application_delta,
                },
                "execution_path_comparison": {
                    "candidate": dict(candidate_entry.get("execution_path", {}) or {}),
                    "current": dict(current_entry.get("execution_path", {}) or {}),
                },
                "replay_evidence_comparison": {
                    "candidate": dict(candidate_entry.get("replay_evidence_basis", {}) or {}),
                    "current": dict(current_entry.get("replay_evidence_basis", {}) or {}),
                },
                "candidate_replay_provenance": dict(candidate_entry.get("replay_provenance", {}) or {}),
                "current_replay_provenance": dict(current_entry.get("replay_provenance", {}) or {}),
            },
        },
        "candidate_policy": candidate_entry,
        "current_policy": current_entry,
    }


def _load_recent_baseline_control_comparison(*, base_dir: str | Path) -> dict[str, object]:
    latest_path = _latest_file_under(Path(base_dir) / "output" / "strategy-comparison-recent", "comparison.json")
    if latest_path is None:
        return {
            "available": False,
            "artifact_path": "",
            "evidence_source": "summary_artifact",
            "replay_grounding": "strategy_comparison_recent_summary",
            "reason": "NO_RECENT_BASELINE_CONTROL_ARTIFACT",
            "verdict": "not_available",
        }
    try:
        payload = json.loads(latest_path.read_text(encoding="utf-8"))
    except Exception:
        return {
            "available": False,
            "artifact_path": str(latest_path),
            "evidence_source": "summary_artifact",
            "replay_grounding": "strategy_comparison_recent_summary",
            "reason": "RECENT_BASELINE_CONTROL_ARTIFACT_UNREADABLE",
            "verdict": "not_available",
        }
    strategies = [dict(item) for item in list(payload.get("strategies", []) or []) if isinstance(item, dict)]
    if not strategies:
        return {
            "available": False,
            "artifact_path": str(latest_path),
            "evidence_source": "summary_artifact",
            "replay_grounding": "strategy_comparison_recent_summary",
            "reason": "RECENT_BASELINE_CONTROL_STRATEGIES_MISSING",
            "verdict": "not_available",
        }
    current = next((item for item in strategies if str(item.get("strategy_name", "") or "") == "current_strategy"), {})
    baselines = []
    for item in strategies:
        strategy_name = str(item.get("strategy_name", "") or "")
        baseline_kind = _simple_control_baseline_kind(strategy_name)
        if not baseline_kind:
            continue
        baselines.append(
            dict(
                item,
                baseline_kind=baseline_kind,
                observation_count=_baseline_observation_count(item),
            )
        )
    if not current or not baselines:
        return {
            "available": False,
            "artifact_path": str(latest_path),
            "evidence_source": "summary_artifact",
            "replay_grounding": "strategy_comparison_recent_summary",
            "reason": (
                "RECENT_BASELINE_CONTROL_CURRENT_OR_BASELINE_MISSING"
                if current
                else "RECENT_BASELINE_CONTROL_CURRENT_STRATEGY_MISSING"
            ),
            "verdict": "not_available",
        }
    current_observation_count = _baseline_observation_count(current)
    best_baseline = max(
        baselines,
        key=lambda item: (
            _safe_float(item.get("total_pnl_usd")),
            _safe_float(item.get("realized_pnl_usd")),
            -_safe_float(item.get("max_drawdown_pct")),
            _safe_int(item.get("observation_count")),
            _safe_int(item.get("trade_count")),
            -_simple_control_baseline_priority(str(item.get("baseline_kind", "") or "")),
            str(item.get("strategy_name", "")),
        ),
    )
    baseline_observation_count = _safe_int(best_baseline.get("observation_count"))
    evidence_ready = current_observation_count >= 3 and baseline_observation_count >= 3
    pnl_delta = round(
        _safe_float(current.get("total_pnl_usd")) - _safe_float(best_baseline.get("total_pnl_usd")),
        6,
    )
    return_delta = round(
        _safe_float(current.get("total_return_pct")) - _safe_float(best_baseline.get("total_return_pct")),
        6,
    )
    current_max_drawdown_pct = round(_safe_float(current.get("max_drawdown_pct")), 6)
    baseline_max_drawdown_pct = round(_safe_float(best_baseline.get("max_drawdown_pct")), 6)
    drawdown_advantage_pct = round(baseline_max_drawdown_pct - current_max_drawdown_pct, 6)
    pnl_margin_required = 0.25
    return_margin_required = 0.02
    drawdown_tolerance_pct = 1.0
    if not evidence_ready:
        verdict = "not_available"
        reason = "RECENT_BASELINE_CONTROL_EVIDENCE_THIN"
        expansion_gate = "not_available"
        expansion_gate_reason = "NO_JUSTIFIED_SIMPLE_BASELINE_GATE"
    else:
        current_clearly_better = (
            pnl_delta >= pnl_margin_required
            and return_delta >= return_margin_required
            and current_max_drawdown_pct <= baseline_max_drawdown_pct + drawdown_tolerance_pct
        )
        baseline_clearly_better = (
            pnl_delta <= -pnl_margin_required
            and return_delta <= -return_margin_required
            and baseline_max_drawdown_pct <= current_max_drawdown_pct + drawdown_tolerance_pct
        )
        verdict = "parity"
        reason = "CURRENT_STRATEGY_NOT_CLEARLY_AHEAD_OF_SIMPLE_BASELINE"
        expansion_gate = "block"
        expansion_gate_reason = "SIMPLE_BASELINE_CONTROL_NOT_CLEARLY_BEATEN"
        if baseline_clearly_better:
            verdict = "caution"
            reason = "CURRENT_STRATEGY_UNDERPERFORMS_SIMPLE_BASELINE"
            expansion_gate_reason = "SIMPLE_BASELINE_CONTROL_UNDERPERFORMED"
        elif current_clearly_better:
            verdict = "supportive"
            reason = "CURRENT_STRATEGY_CLEARLY_OUTPERFORMS_SIMPLE_BASELINE"
            expansion_gate = "pass"
            expansion_gate_reason = "SIMPLE_BASELINE_CONTROL_CLEARED"
    return {
        "available": True,
        "artifact_path": str(latest_path),
        "evidence_source": "summary_artifact",
        "replay_grounding": "strategy_comparison_recent_summary",
        "generated_at": payload.get("generated_at"),
        "verdict": verdict,
        "reason": reason,
        "expansion_gate": expansion_gate,
        "expansion_gate_reason": expansion_gate_reason,
        "current_policy_clearly_beats_simple_baseline": verdict == "supportive",
        "simple_control_strategy_names": [
            str(item.get("strategy_name", "") or "")
            for item in sorted(
                baselines,
                key=lambda item: (
                    _simple_control_baseline_priority(str(item.get("baseline_kind", "") or "")),
                    str(item.get("strategy_name", "") or ""),
                ),
            )
        ],
        "simple_control_baseline_count": len(baselines),
        "selected_simple_baseline_kind": str(best_baseline.get("baseline_kind", "") or ""),
        "comparison_thresholds": {
            "minimum_observation_count": 3,
            "total_pnl_usd_margin_required": pnl_margin_required,
            "return_pct_margin_required": return_margin_required,
            "drawdown_tolerance_pct": drawdown_tolerance_pct,
        },
        "current_strategy": {
            "strategy_name": str(current.get("strategy_name", "") or ""),
            "trade_count": _safe_int(current.get("trade_count")),
            "closed_trade_count": _safe_int(current.get("closed_trade_count")),
            "observation_count": current_observation_count,
            "total_pnl_usd": round(_safe_float(current.get("total_pnl_usd")), 6),
            "total_return_pct": round(_safe_float(current.get("total_return_pct")), 6),
            "max_drawdown_pct": current_max_drawdown_pct,
        },
        "best_simple_baseline": {
            "strategy_name": str(best_baseline.get("strategy_name", "") or ""),
            "baseline_kind": str(best_baseline.get("baseline_kind", "") or ""),
            "trade_count": _safe_int(best_baseline.get("trade_count")),
            "closed_trade_count": _safe_int(best_baseline.get("closed_trade_count")),
            "observation_count": baseline_observation_count,
            "total_pnl_usd": round(_safe_float(best_baseline.get("total_pnl_usd")), 6),
            "total_return_pct": round(_safe_float(best_baseline.get("total_return_pct")), 6),
            "max_drawdown_pct": baseline_max_drawdown_pct,
        },
        "current_vs_best_simple_baseline_total_pnl_usd_delta": pnl_delta,
        "current_vs_best_simple_baseline_return_pct_delta": return_delta,
        "current_vs_best_simple_baseline_max_drawdown_pct_delta": drawdown_advantage_pct,
    }


_BASELINE_CONTROL_BUCKET_REPLAY_REQUIREMENTS = (
    ("decision_count", "DECISION_LOGS"),
    ("total_tested_order_count", "TESTED_ORDER_LOGS"),
    ("total_live_order_count", "LIVE_ORDER_LOGS"),
    ("total_closed_trade_count", "CLOSED_TRADE_LOGS"),
)


def _baseline_control_bucket_replay_entry(
    *,
    bucket_name: str,
    bucket_evidence: dict[str, object] | None,
    replay_summary: dict[str, object] | None,
) -> dict[str, object]:
    bucket = dict(bucket_evidence or {})
    summary = dict(replay_summary or {})
    execution_metrics = dict(summary.get("execution_metrics", {}) or {})
    decision_count = max(
        _safe_int(bucket.get("policy_context_bucket_decision_count")),
        _safe_int(bucket.get("decision_count")),
    )
    total_tested_order_count = max(
        _safe_int(bucket.get("policy_context_bucket_tested_order_count")),
        _safe_int(bucket.get("total_tested_order_count")),
        _safe_int(bucket.get("tested_order_count")),
    )
    total_live_order_count = max(
        _safe_int(bucket.get("policy_context_bucket_live_order_count")),
        _safe_int(bucket.get("total_live_order_count")),
        _safe_int(bucket.get("live_order_count")),
        _safe_int(execution_metrics.get("live_order_count")),
    )
    total_closed_trade_count = max(
        _safe_int(bucket.get("policy_context_bucket_closed_trade_count")),
        _safe_int(bucket.get("total_closed_trade_count")),
        _safe_int(bucket.get("closed_trade_count")),
        _safe_int(execution_metrics.get("closed_trade_count")),
    )
    run_count = max(
        _safe_int(bucket.get("policy_context_bucket_run_count")),
        _safe_int(bucket.get("run_count")),
        _safe_int(execution_metrics.get("run_count")),
    )
    available = bool(bucket) or _replay_summary_available(summary)
    replay_requirement_values = {
        "decision_count": decision_count,
        "total_tested_order_count": total_tested_order_count,
        "total_live_order_count": total_live_order_count,
        "total_closed_trade_count": total_closed_trade_count,
        "run_count": run_count,
    }
    missing_surfaces = [
        label.lower()
        for metric_name, label in _BASELINE_CONTROL_BUCKET_REPLAY_REQUIREMENTS
        if _safe_int(replay_requirement_values.get(metric_name)) <= 0
    ]
    bucket_replay_ready = bool(available) and not missing_surfaces
    if bucket_replay_ready:
        bucket_replay_reason = "BASELINE_CONTROL_BUCKET_REPLAY_READY"
    elif available:
        bucket_replay_reason = "BASELINE_CONTROL_BUCKET_REPLAY_MISSING_" + "_AND_".join(
            label
            for metric_name, label in _BASELINE_CONTROL_BUCKET_REPLAY_REQUIREMENTS
            if _safe_int(replay_requirement_values.get(metric_name)) <= 0
        )
    else:
        bucket_replay_reason = "BASELINE_CONTROL_BUCKET_REPLAY_NOT_AVAILABLE"
    return {
        "available": bool(available),
        "bucket_name": bucket_name,
        "run_count": run_count,
        "decision_count": decision_count,
        "total_tested_order_count": total_tested_order_count,
        "total_live_order_count": total_live_order_count,
        "total_closed_trade_count": total_closed_trade_count,
        "micro_live_status": str(dict(summary.get("micro_live_gate", {}) or {}).get("status", "not_available") or "not_available"),
        "replay_source": str(summary.get("source", "") or ""),
        "bucket_replay_ready": bucket_replay_ready,
        "bucket_replay_reason": bucket_replay_reason,
        "missing_surfaces": missing_surfaces,
    }


def _bucket_aware_baseline_control_comparison(
    *,
    baseline_control_comparison: dict[str, object],
    current_policy_bucket_evidence: dict[str, object] | None,
    current_policy_direct_replay_summary: dict[str, object] | None,
    staged_candidate_bucket_evidence: dict[str, object] | None,
    staged_candidate_direct_replay_summary: dict[str, object] | None,
) -> dict[str, object]:
    payload = dict(baseline_control_comparison or {})
    if not payload:
        return payload
    current_policy_bucket_replay = _baseline_control_bucket_replay_entry(
        bucket_name="active_policy",
        bucket_evidence=current_policy_bucket_evidence,
        replay_summary=current_policy_direct_replay_summary,
    )
    staged_candidate_bucket_replay = _baseline_control_bucket_replay_entry(
        bucket_name="staged_candidate",
        bucket_evidence=staged_candidate_bucket_evidence,
        replay_summary=staged_candidate_direct_replay_summary,
    )
    reference_bucket = (
        staged_candidate_bucket_replay
        if bool(staged_candidate_bucket_replay.get("available"))
        else current_policy_bucket_replay
    )
    bucket_replay_ready = bool(reference_bucket.get("bucket_replay_ready"))
    bucket_replay_reason = str(reference_bucket.get("bucket_replay_reason", "BASELINE_CONTROL_BUCKET_REPLAY_NOT_AVAILABLE") or "BASELINE_CONTROL_BUCKET_REPLAY_NOT_AVAILABLE")
    payload["bucket_replay_required_for_expansion"] = True
    payload["bucket_replay_ready"] = bucket_replay_ready
    payload["bucket_replay_reference_bucket"] = str(reference_bucket.get("bucket_name", "not_available") or "not_available")
    payload["bucket_replay_reason"] = bucket_replay_reason
    payload["current_policy_bucket_replay"] = current_policy_bucket_replay
    payload["staged_candidate_bucket_replay"] = staged_candidate_bucket_replay
    if bool(current_policy_bucket_replay.get("available")) or bool(staged_candidate_bucket_replay.get("available")):
        payload["evidence_source"] = "summary_artifact+policy_bucket_replay"
        payload["replay_grounding"] = "strategy_comparison_recent_summary+policy_bucket_replay"
    if str(payload.get("expansion_gate", "not_available") or "not_available") == "pass" and not bucket_replay_ready:
        payload["expansion_gate"] = "not_available"
        payload["expansion_gate_reason"] = bucket_replay_reason
    payload["replay_provenance"] = baseline_control_replay_provenance(payload)
    return payload


def _checkpoint_symbol_lifecycle_actions(
    *,
    symbol_rows: list[dict[str, object]],
    symbol_scorecard: list[dict[str, object]],
    active_adjustments: list[dict[str, object]],
) -> list[dict[str, object]]:
    scorecard_by_symbol = {
        str(item.get("symbol", "") or ""): dict(item)
        for item in list(symbol_scorecard or [])
        if str(item.get("symbol", "") or "")
    }
    active_by_symbol = {
        str(item.get("symbol", "") or ""): dict(item)
        for item in list(active_adjustments or [])
        if str(item.get("symbol", "") or "")
    }
    rows: list[dict[str, object]] = []
    symbols = sorted(
        {
            str(item.get("symbol", "") or "")
            for item in list(symbol_rows or [])
        }.union(active_by_symbol)
    )
    for symbol in symbols:
        if not symbol:
            continue
        row = next((dict(item) for item in symbol_rows if str(item.get("symbol", "") or "") == symbol), {})
        active = dict(active_by_symbol.get(symbol, {}) or {})
        scorecard = dict(scorecard_by_symbol.get(symbol, {}) or {})
        recommendation = str(row.get("recommendation", active.get("action", "keep")) or "keep")
        scorecard_recommendation = str(scorecard.get("recommendation", "keep") or "keep")
        active_action = str(active.get("action", "") or "")
        active_positive = active_action in {"promote", "aggressive_promote"}
        lifecycle_action = ""
        reason_codes: list[str] = []
        if recommendation in {"prune", "demote"} or scorecard_recommendation == "demote":
            lifecycle_action = "rollback" if active_positive else "hold"
            reason_codes.append("SYMBOL_EVIDENCE_DEGRADED")
            if active_positive:
                reason_codes.append("ACTIVE_PROMOTION_SUPPORT_LOST")
        elif recommendation == "observe_only":
            lifecycle_action = "hold"
            reason_codes.append("SYMBOL_OBSERVE_ONLY")
        elif recommendation == "promote":
            validation_ready = _safe_int(row.get("trade_count")) >= _safe_int(row.get("required_trade_count_for_validation"), 3)
            if validation_ready and scorecard_recommendation == "promote":
                lifecycle_action = "expand"
                reason_codes.append("SYMBOL_PROMOTION_SUPPORTED")
            else:
                lifecycle_action = "re_review"
                reason_codes.append("SYMBOL_PROMOTION_REQUIRES_RECHECK")
        elif active_positive:
            lifecycle_action = "re_review"
            reason_codes.append("ACTIVE_SYMBOL_REQUIRES_RECHECK")
        if not lifecycle_action:
            continue
        rows.append(
            {
                "symbol": symbol,
                "lifecycle_action": lifecycle_action,
                "recommendation": recommendation,
                "active_policy_action": active_action or "none",
                "trade_count": _safe_int(row.get("trade_count")),
                "scorecard_recommendation": scorecard_recommendation,
                "sample_status": str(scorecard.get("sample_status", "") or ""),
                "reason_codes": reason_codes,
            }
        )
    rows.sort(
        key=lambda item: (
            {"rollback": 0, "hold": 1, "re_review": 2, "expand": 3}.get(str(item.get("lifecycle_action", "")), 4),
            str(item.get("symbol", "")),
        )
    )
    return rows


def _checkpoint_regime_actions(
    *,
    regime_rows: list[dict[str, object]],
    sample_watchdog_status: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in list(regime_rows or []):
        mode = str(item.get("mode", "") or "")
        decision_count = _safe_int(item.get("decision_count"))
        if not mode or decision_count < 3:
            continue
        avg_net_edge_bps = round(_safe_float(item.get("avg_net_edge_bps")), 6)
        action = "hold"
        reason_codes = ["REGIME_WITHIN_THRESHOLDS"]
        if avg_net_edge_bps <= 0.0:
            action = "tighten"
            reason_codes = ["REGIME_EDGE_NON_POSITIVE"]
        elif sample_watchdog_status == "promote_ready" and avg_net_edge_bps > 1.0:
            action = "expand"
            reason_codes = ["REGIME_EDGE_SUPPORTIVE"]
        elif sample_watchdog_status == "degraded":
            action = "tighten"
            reason_codes = ["REGIME_TIGHTENED_BY_SAMPLE_WATCHDOG"]
        rows.append(
            {
                "mode": mode,
                "action": action,
                "decision_count": decision_count,
                "avg_net_edge_bps": avg_net_edge_bps,
                "reason_codes": reason_codes,
            }
        )
    return rows


def _checkpoint_judge_confidence(
    *,
    run_count: int,
    total_closed_trade_count: int,
    total_live_order_count: int,
    baseline_available: bool,
    comparison_verdict: str,
) -> str:
    if (
        run_count >= 2
        and total_closed_trade_count >= 10
        and total_live_order_count >= 12
        and (baseline_available or comparison_verdict != "keep")
    ):
        return "high"
    if run_count >= 2 and total_closed_trade_count >= 6:
        return "medium"
    return "low"


def _checkpoint_bucket_evidence_map(payload: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    return {
        str(bucket_name): dict(bucket_payload)
        for bucket_name, bucket_payload in dict(dict(payload or {}).get("policy_context_bucket_evidence", {}) or {}).items()
        if str(bucket_name)
    }


def _preferred_checkpoint_bucket_evidence(payload: dict[str, Any] | None) -> tuple[str, dict[str, Any], str]:
    raw_payload = dict(payload or {})
    bucket_payloads = _checkpoint_bucket_evidence_map(raw_payload)
    preferred_bucket = str(raw_payload.get("preferred_policy_bucket", "") or "").strip().lower()
    bucket_order: list[str] = []
    if preferred_bucket:
        bucket_order.append(preferred_bucket)
    bucket_order.extend(
        bucket_name
        for bucket_name in ("staged_candidate", "active_policy", "previous_policy")
        if bucket_name not in bucket_order
    )
    bucket_order.extend(
        bucket_name
        for bucket_name in bucket_payloads
        if bucket_name not in bucket_order
    )
    for bucket_name in bucket_order:
        bucket_payload = dict(bucket_payloads.get(bucket_name, {}) or {})
        if bucket_payload:
            return bucket_name, bucket_payload, "policy_bucket"
    return "", raw_payload, "root"


def _checkpoint_validation_runs(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    return [
        dict(item)
        for item in list(
            dict(payload or {}).get(
                "validation_runs",
                dict(payload or {}).get("policy_context_bucket_validation_runs", []),
            )
            or []
        )
        if isinstance(item, dict)
    ]


def _checkpoint_walk_forward_windows(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    return [
        dict(item)
        for item in list(
            dict(payload or {}).get(
                "walk_forward_windows",
                dict(payload or {}).get("policy_context_bucket_walk_forward_windows", []),
            )
            or []
        )
        if isinstance(item, dict)
    ]


def _checkpoint_drawdown_ratio_from_runs(validation_runs: list[dict[str, Any]]) -> float:
    pnl_values = [_safe_float(item.get("realized_pnl_usd")) for item in validation_runs]
    if not pnl_values:
        return 0.0
    cumulative = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for value in pnl_values:
        cumulative += value
        peak = max(peak, cumulative)
        max_drawdown = max(max_drawdown, peak - cumulative)
    return round(max_drawdown / max(abs(sum(pnl_values)), 1.0), 6)


def _checkpoint_execution_metrics(payload: dict[str, Any] | None) -> dict[str, float | int]:
    evidence = dict(payload or {})
    validation_runs = _checkpoint_validation_runs(evidence)
    walk_forward_windows = _checkpoint_walk_forward_windows(evidence)
    summed_live_order_count = sum(_safe_int(item.get("live_order_count")) for item in validation_runs)
    summed_rejected_live_order_count = sum(_safe_int(item.get("rejected_live_order_count")) for item in validation_runs)
    summed_accepted_live_order_count = sum(_safe_int(item.get("accepted_live_order_count")) for item in validation_runs)
    live_order_weight = max(summed_live_order_count, 0)
    accepted_live_order_weight = max(summed_accepted_live_order_count, 0)
    total_closed_trade_count = max(
        _safe_int(evidence.get("policy_context_bucket_closed_trade_count")),
        _safe_int(evidence.get("total_closed_trade_count")),
        _safe_int(evidence.get("closed_trade_count")),
        _safe_int(dict(evidence.get("micro_live_gate", {}) or {}).get("closed_trade_count")),
        sum(_safe_int(item.get("closed_trade_count")) for item in validation_runs),
    )
    total_live_order_count = max(
        _safe_int(evidence.get("total_live_order_count")),
        _safe_int(evidence.get("live_order_count")),
        _safe_int(dict(evidence.get("micro_live_gate", {}) or {}).get("live_order_count")),
        live_order_weight,
    )
    total_tested_order_count = max(
        _safe_int(evidence.get("policy_context_bucket_tested_order_count")),
        _safe_int(evidence.get("total_tested_order_count")),
        _safe_int(evidence.get("tested_order_count")),
        sum(_safe_int(item.get("tested_order_count")) for item in validation_runs),
    )
    decision_count = max(
        _safe_int(evidence.get("policy_context_bucket_decision_count")),
        _safe_int(evidence.get("decision_count")),
    )
    run_count = max(
        _safe_int(evidence.get("policy_context_bucket_run_count")),
        _safe_int(evidence.get("run_count")),
        len(validation_runs),
    )
    if "runner_total_realized_pnl_usd" in evidence or "policy_context_bucket_total_realized_pnl_usd" in evidence or "realized_pnl_usd" in evidence:
        realized_pnl_usd = round(
            _safe_float(
                evidence.get(
                    "runner_total_realized_pnl_usd",
                    evidence.get(
                        "policy_context_bucket_total_realized_pnl_usd",
                        evidence.get("realized_pnl_usd"),
                    ),
                )
            ),
            6,
        )
    else:
        realized_pnl_usd = round(sum(_safe_float(item.get("realized_pnl_usd")) for item in validation_runs), 6)
    if "runner_drawdown_to_pnl_ratio" in evidence or "drawdown_to_pnl_ratio" in evidence:
        drawdown_to_pnl_ratio = round(
            _safe_float(
                evidence.get(
                    "runner_drawdown_to_pnl_ratio",
                    evidence.get("drawdown_to_pnl_ratio"),
                )
            ),
            6,
        )
    else:
        drawdown_to_pnl_ratio = _checkpoint_drawdown_ratio_from_runs(validation_runs)
    if "runner_reject_rate" in evidence or "reject_rate" in evidence:
        reject_rate = round(
            _safe_float(
                evidence.get(
                    "runner_reject_rate",
                    evidence.get("reject_rate"),
                )
            ),
            6,
        )
    else:
        reject_rate = round(summed_rejected_live_order_count / max(live_order_weight, 1), 6) if live_order_weight > 0 else 0.0
    if "runner_protection_degraded_rate" in evidence or "protection_degraded_rate" in evidence:
        protection_degraded_rate = round(
            _safe_float(
                evidence.get(
                    "runner_protection_degraded_rate",
                    evidence.get("protection_degraded_rate"),
                )
            ),
            6,
        )
    else:
        protection_degraded_rate = (
            round(
                sum(_safe_float(item.get("protection_degraded_rate")) * _safe_int(item.get("live_order_count")) for item in validation_runs)
                / max(live_order_weight, 1),
                6,
            )
            if live_order_weight > 0
            else 0.0
        )
    if "runner_avg_edge_retention_ratio" in evidence or "avg_edge_retention_ratio" in evidence or "avg_retention" in evidence:
        avg_edge_retention_ratio = round(
            _safe_float(
                evidence.get(
                    "runner_avg_edge_retention_ratio",
                    evidence.get(
                        "avg_edge_retention_ratio",
                        evidence.get("avg_retention"),
                    ),
                )
            ),
            6,
        )
    else:
        avg_edge_retention_ratio = (
            round(
                sum(_safe_float(item.get("avg_edge_retention_ratio")) * _safe_int(item.get("live_order_count")) for item in validation_runs)
                / max(live_order_weight, 1),
                6,
            )
            if live_order_weight > 0
            else 0.0
        )
    if "runner_positive_walk_forward_ratio" in evidence or "policy_context_bucket_positive_walk_forward_ratio" in evidence or "positive_walk_forward_ratio" in evidence:
        positive_walk_forward_ratio = round(
            _safe_float(
                evidence.get(
                    "runner_positive_walk_forward_ratio",
                    evidence.get(
                        "policy_context_bucket_positive_walk_forward_ratio",
                        evidence.get("positive_walk_forward_ratio"),
                    ),
                )
            ),
            6,
        )
    else:
        positive_walk_forward_ratio = round(
            _safe_ratio(
                sum(
                    1
                    for item in walk_forward_windows
                    if _safe_float(item.get("avg_net_edge_bps")) > 0.0 and _safe_float(item.get("avg_score")) >= 0.0
                ),
                len(walk_forward_windows),
            ),
            6,
        )
    walk_forward_window_count = max(
        _safe_int(evidence.get("runner_walk_forward_window_count")),
        _safe_int(evidence.get("policy_context_bucket_walk_forward_window_count")),
        _safe_int(evidence.get("walk_forward_window_count")),
        len(walk_forward_windows),
    )
    return {
        "run_count": run_count,
        "decision_count": decision_count,
        "total_closed_trade_count": total_closed_trade_count,
        "total_live_order_count": total_live_order_count,
        "total_tested_order_count": total_tested_order_count,
        "runner_total_realized_pnl_usd": realized_pnl_usd,
        "runner_drawdown_to_pnl_ratio": drawdown_to_pnl_ratio,
        "runner_reject_rate": reject_rate,
        "runner_protection_degraded_rate": protection_degraded_rate,
        "runner_avg_edge_retention_ratio": avg_edge_retention_ratio,
        "runner_positive_walk_forward_ratio": positive_walk_forward_ratio,
        "runner_walk_forward_window_count": walk_forward_window_count,
    }


def _build_checkpoint_auto_judge(
    *,
    current_policy_state: dict[str, Any] | None,
    comparison_verdict: str,
    comparison_delta: float,
    runner_artifact: dict[str, object],
    baseline_control_comparison: dict[str, object],
) -> dict[str, object]:
    payload = dict(runner_artifact or {})
    checkpoint_bucket_name, checkpoint_bucket_evidence, checkpoint_evidence_source = _preferred_checkpoint_bucket_evidence(payload)
    lifecycle_bucket_evidence = dict(checkpoint_bucket_evidence if checkpoint_evidence_source == "policy_bucket" else {})
    sample_watchdog = dict(payload.get("sample_quality_watchdog", {}) or {})
    sample_watchdog_status = str(sample_watchdog.get("status", "not_available") or "not_available")
    active_policy = dict(dict(current_policy_state or {}).get("active_policy", {}) or {})
    active_adjustments = list(active_policy.get("adjustments", []) or [])
    active_status = str(active_policy.get("status", "baseline") or "baseline")
    active_rollout_phase = str(
        dict(current_policy_state or {}).get("rollout_progression", dict(active_policy.get("rollout_progression", {}) or {})).get("execution_phase", "baseline")
        or "baseline"
    )
    checkpoint_metrics = _checkpoint_execution_metrics(checkpoint_bucket_evidence)
    run_count = _safe_int(checkpoint_metrics.get("run_count"))
    decision_count = _safe_int(checkpoint_metrics.get("decision_count"))
    total_closed_trade_count = _safe_int(checkpoint_metrics.get("total_closed_trade_count"))
    total_live_order_count = _safe_int(checkpoint_metrics.get("total_live_order_count"))
    total_tested_order_count = _safe_int(checkpoint_metrics.get("total_tested_order_count"))
    runner_total_realized_pnl_usd = round(_safe_float(checkpoint_metrics.get("runner_total_realized_pnl_usd")), 6)
    runner_drawdown_to_pnl_ratio = round(_safe_float(checkpoint_metrics.get("runner_drawdown_to_pnl_ratio")), 6)
    runner_reject_rate = round(_safe_float(checkpoint_metrics.get("runner_reject_rate")), 6)
    runner_protection_degraded_rate = round(_safe_float(checkpoint_metrics.get("runner_protection_degraded_rate")), 6)
    runner_avg_edge_retention_ratio = round(_safe_float(checkpoint_metrics.get("runner_avg_edge_retention_ratio")), 6)
    runner_positive_walk_forward_ratio = round(_safe_float(checkpoint_metrics.get("runner_positive_walk_forward_ratio")), 6)
    baseline_verdict = str(baseline_control_comparison.get("verdict", "not_available") or "not_available")
    baseline_gate = str(baseline_control_comparison.get("expansion_gate", "not_available") or "not_available")
    baseline_gate_reason = str(
        baseline_control_comparison.get(
            "expansion_gate_reason",
            "NO_JUSTIFIED_SIMPLE_BASELINE_GATE",
        )
        or "NO_JUSTIFIED_SIMPLE_BASELINE_GATE"
    )
    raw_verdict = "hold"
    reason_codes: list[str] = []
    if comparison_verdict == "candidate_worse":
        raw_verdict = "rollback"
        reason_codes.append("POLICY_COMPARISON_CANDIDATE_WORSE")
    elif sample_watchdog_status == "degraded":
        raw_verdict = "tighten"
        reason_codes.append("SAMPLE_QUALITY_WATCHDOG_DEGRADED")
    elif sample_watchdog_status == "promote_ready":
        raw_verdict = "expand"
        reason_codes.append("SAMPLE_QUALITY_WATCHDOG_PROMOTE_READY")
    elif sample_watchdog_status == "healthy":
        raw_verdict = "hold"
        reason_codes.append("SAMPLE_QUALITY_WATCHDOG_HEALTHY")
    else:
        raw_verdict = "hold"
        reason_codes.append("SAMPLE_QUALITY_WATCHDOG_THIN")
    if baseline_gate == "block":
        if baseline_verdict == "caution":
            if raw_verdict == "expand":
                raw_verdict = "hold"
            elif raw_verdict == "hold":
                raw_verdict = "tighten"
        elif baseline_verdict == "parity" and raw_verdict == "expand":
            raw_verdict = "hold"
        reason_codes.append(baseline_gate_reason)
    elif baseline_verdict == "supportive" and baseline_gate == "pass":
        reason_codes.append("SIMPLE_BASELINE_CONTROL_CLEARED")
    if (
        total_closed_trade_count >= 6
        and (
            runner_total_realized_pnl_usd <= 0.0
            or runner_drawdown_to_pnl_ratio > 0.9
            or runner_reject_rate > 0.15
            or runner_protection_degraded_rate > 0.15
            or runner_avg_edge_retention_ratio < 0.4
        )
    ):
        raw_verdict = "rollback"
        reason_codes.append("RUNNER_EVIDENCE_SEVERELY_NEGATIVE")
    elif raw_verdict == "expand" and (
        runner_avg_edge_retention_ratio < 0.68
        or runner_positive_walk_forward_ratio < 0.67
    ):
        raw_verdict = "hold"
        reason_codes.append("RUNNER_EVIDENCE_NOT_STRONG_ENOUGH_TO_EXPAND")
    elif raw_verdict == "hold" and (
        runner_avg_edge_retention_ratio < 0.55
        or runner_reject_rate > 0.08
        or runner_protection_degraded_rate > 0.08
    ):
        raw_verdict = "tighten"
        reason_codes.append("RUNNER_EXECUTION_EVIDENCE_REQUIRES_TIGHTENING")
    if raw_verdict == "expand":
        if decision_count <= 0 or total_tested_order_count <= 0:
            raw_verdict = "hold"
            reason_codes.append("CHECKPOINT_EXPANSION_REQUIRES_DECISION_AND_TESTED_ORDER_LOG_EVIDENCE")
        if checkpoint_evidence_source != "policy_bucket":
            raw_verdict = "hold"
            reason_codes.append("CHECKPOINT_EXPANSION_REQUIRES_POLICY_BUCKET_EVIDENCE")
        elif checkpoint_bucket_name != "staged_candidate":
            raw_verdict = "hold"
            reason_codes.append("CHECKPOINT_EXPANSION_REQUIRES_STAGED_CANDIDATE_BUCKET")
    effective_verdict = raw_verdict
    if raw_verdict == "rollback" and active_status in {"baseline", "keep"} and not active_adjustments:
        effective_verdict = "tighten" if sample_watchdog_status == "degraded" else "hold"
        reason_codes.append("NO_ACTIVE_NON_BASELINE_POLICY_TO_ROLL_BACK")
    if checkpoint_evidence_source == "policy_bucket" and checkpoint_bucket_name:
        reason_codes.append(f"CHECKPOINT_POLICY_BUCKET_{checkpoint_bucket_name.upper()}_USED")
    elif not checkpoint_bucket_name:
        reason_codes.append("CHECKPOINT_ROOT_EVIDENCE_FALLBACK")
    symbol_lifecycle = build_symbol_lifecycle(
        symbol_summary=list(payload.get("symbol_summary", []) or []),
        symbol_scorecard=list(payload.get("symbol_scorecard", []) or []),
        pruning_recommendations=list(payload.get("pruning_recommendations", []) or []),
        policy_context_bucket_name=str(lifecycle_bucket_evidence.get("policy_context_bucket_name", "") or ""),
        policy_context_bucket_symbol_summary=list(
            lifecycle_bucket_evidence.get(
                "policy_context_bucket_symbol_summary",
                lifecycle_bucket_evidence.get("symbol_summary", []),
            )
            or []
        ),
        policy_context_bucket_pruning_recommendations=list(
            lifecycle_bucket_evidence.get(
                "policy_context_bucket_pruning_recommendations",
                lifecycle_bucket_evidence.get("pruning_recommendations", []),
            )
            or []
        ),
        active_adjustments=active_adjustments,
        previous_rows=list(dict(current_policy_state or {}).get("symbol_lifecycle", []) or []),
        checkpoint_auto_judge={
            "verdict": effective_verdict,
            "symbol_actions": _checkpoint_symbol_lifecycle_actions(
                symbol_rows=list(payload.get("symbol_summary", []) or []),
                symbol_scorecard=list(payload.get("symbol_scorecard", []) or []),
                active_adjustments=active_adjustments,
            ),
        },
        sample_quality_watchdog=sample_watchdog,
        baseline_control_comparison=baseline_control_comparison,
        active_policy=active_policy,
        rollout_phase=active_rollout_phase,
        policy_version=dict(current_policy_state or {}).get("version"),
        evaluated_at=payload.get("generated_at", ""),
    )
    replay_provenance = checkpoint_replay_provenance(
        {
            "evidence_source": checkpoint_evidence_source,
            "evidence_policy_bucket": checkpoint_bucket_name or "not_available",
        }
    )
    return {
        "verdict": effective_verdict,
        "raw_verdict": raw_verdict,
        "confidence": _checkpoint_judge_confidence(
            run_count=run_count,
            total_closed_trade_count=total_closed_trade_count,
            total_live_order_count=total_live_order_count,
            baseline_available=bool(baseline_control_comparison.get("available")),
            comparison_verdict=comparison_verdict,
        ),
        "reason_codes": sorted(set(reason_codes)),
        "current_policy_status": active_status,
        "comparison_verdict": comparison_verdict,
        "comparison_score_delta": round(comparison_delta, 6),
        "sample_quality_watchdog_status": sample_watchdog_status,
        "evidence_source": checkpoint_evidence_source,
        "evidence_policy_bucket": checkpoint_bucket_name or "not_available",
        "evidence_bucket_available": checkpoint_evidence_source == "policy_bucket",
        "replay_provenance": replay_provenance,
        "baseline_control_comparison": dict(baseline_control_comparison or {}),
        "policy_guardrails": dict(sample_watchdog.get("policy_guardrails", {}) or {}),
        "evidence": {
            "run_count": run_count,
            "decision_count": decision_count,
            "total_closed_trade_count": total_closed_trade_count,
            "total_live_order_count": total_live_order_count,
            "total_tested_order_count": total_tested_order_count,
            "runner_total_realized_pnl_usd": runner_total_realized_pnl_usd,
            "runner_drawdown_to_pnl_ratio": runner_drawdown_to_pnl_ratio,
            "runner_reject_rate": runner_reject_rate,
            "runner_protection_degraded_rate": runner_protection_degraded_rate,
            "runner_avg_edge_retention_ratio": runner_avg_edge_retention_ratio,
            "runner_positive_walk_forward_ratio": runner_positive_walk_forward_ratio,
            "evidence_source": checkpoint_evidence_source,
            "policy_bucket": checkpoint_bucket_name or "",
        },
        "symbol_actions": [
            {
                "symbol": str(item.get("symbol", "") or ""),
                "lifecycle_action": str(item.get("recommended_action", "keep") or "keep"),
                "recommendation": str(item.get("recommendation", "keep") or "keep"),
                "active_policy_action": str(item.get("active_policy_action", "none") or "none"),
                "trade_count": _safe_int(item.get("trade_count")),
                "scorecard_recommendation": str(item.get("scorecard_recommendation", "keep") or "keep"),
                "sample_status": str(item.get("sample_watchdog_status", "") or ""),
                "reason_codes": list(item.get("reason_codes", []) or []),
            }
            for item in symbol_lifecycle
            if str(item.get("recommended_action", "keep") or "keep") != "keep"
        ],
        "symbol_lifecycle": symbol_lifecycle,
        "symbol_lifecycle_summary": summarize_symbol_lifecycle(symbol_lifecycle),
        "regime_actions": _checkpoint_regime_actions(
            regime_rows=list(payload.get("regime_summary", []) or []),
            sample_watchdog_status=sample_watchdog_status,
        ),
    }


def build_policy_comparison_validation_artifact(*,
    current_policy_state: dict[str, Any] | None,
    candidate_policy: dict[str, Any],
    base_dir: str | Path = "quant_runtime",
    lookback_days: int = 7,
    current_runtime_summary: dict[str, Any] | None = None,
) -> dict[str, object]:
    raw_runner = build_policy_validation_runner_artifact(base_dir=base_dir, lookback_days=lookback_days)
    current_policy = dict(dict(current_policy_state or {}).get("active_policy", {}) or {})
    current_policy_bucket = policy_evidence_bucket(current_policy_state, "active_policy")
    current_policy_evidence = dict(policy_evidence_bucket_evidence(current_policy_state, "active_policy", fallback_to_root=False) or {})
    legacy_current_policy_evidence = dict(dict(dict(current_policy_state or {}).get("policy_validation", {}) or {}).get("evidence", {}) or {})
    if not current_policy_evidence and current_policy_bucket:
        current_policy_evidence = dict(current_policy_bucket.get("evidence", {}) or {})
    if not current_policy_evidence:
        current_policy_evidence = legacy_current_policy_evidence
    current_score = _policy_adjustment_score(current_policy)
    candidate_score = _policy_adjustment_score(candidate_policy)
    current_rollout_phase = str(
        dict(current_policy_state or {}).get("rollout_progression", dict(current_policy.get("rollout_progression", {}) or {})).get("execution_phase", "full")
        or "full"
    )
    if not list(current_policy.get("adjustments", []) or []):
        current_rollout_phase = "baseline"
    candidate_policy_application = _policy_application_profile(
        policy=candidate_policy,
        rollout_phase="full",
        source="candidate_policy_requested",
    )
    current_policy_application = _policy_application_profile(
        policy=current_policy,
        rollout_phase=current_rollout_phase,
        source="current_policy_active",
    )
    current_active_lineage = build_policy_state_lineage_snapshot(
        current_policy_state,
        source="current_policy_state",
    )
    explicit_current_policy_evidence_lineage = dict(
        current_policy_bucket.get(
            "evidence_lineage",
            current_policy_evidence.get("active_policy_lineage", dict(current_policy_state or {}).get("policy_lineage", {})),
        )
        or {}
    )
    current_policy_evidence_lineage = (
        _extract_policy_lineage_from_payload(
            {
                "active_policy_lineage": explicit_current_policy_evidence_lineage,
                "policy_application": current_policy_evidence.get("current_policy_application", current_policy_application),
                "generated_at": current_policy_evidence.get("generated_at", dict(current_policy_state or {}).get("updated_at", "")),
            },
            source="current_policy_evidence",
            updated_at=current_policy_evidence.get("generated_at", dict(current_policy_state or {}).get("updated_at", "")),
        )
        if explicit_current_policy_evidence_lineage
        else dict(current_active_lineage)
    )
    current_evidence_lineage_alignment = policy_lineage_alignment(
        current_active_lineage,
        current_policy_evidence_lineage,
    )
    raw_validation_runs = [dict(item) for item in list(raw_runner.get("validation_runs", []) or [])]
    runtime_summary_run_dir = _resolve_latest_run_dir(base_dir=Path(base_dir)) if current_runtime_summary else None
    runtime_summary_snapshot = _runtime_summary_validation_snapshot(
        current_runtime_summary,
        run_dir=runtime_summary_run_dir,
        policy_lineage=current_active_lineage,
    )
    if runtime_summary_snapshot is not None and (
        (runtime_summary_run_dir is None and not raw_validation_runs)
        or (runtime_summary_run_dir is not None and (runtime_summary_run_dir.name != "latest" or not raw_validation_runs))
    ):
        raw_validation_runs = _merge_runtime_summary_snapshot(raw_validation_runs, runtime_summary_snapshot)
    filtered_validation_runs, lineage_attribution = _filter_lineage_aligned_run_snapshots(
        run_snapshots=raw_validation_runs,
        active_policy_lineage=current_active_lineage,
    )
    current_policy_run_snapshots = [dict(item) for item in filtered_validation_runs]
    if runtime_summary_run_dir is not None:
        current_policy_run_snapshots = _merge_runtime_summary_snapshot(
            current_policy_run_snapshots,
            _run_validation_snapshot(run_dir=runtime_summary_run_dir),
        )
    current_policy_direct_bucket_evidence = _policy_context_bucket_direct_evidence(
        base_dir=base_dir,
        lookback_days=lookback_days,
        generated_at=str(raw_runner.get("generated_at", datetime.now(timezone.utc).isoformat()) or datetime.now(timezone.utc).isoformat()),
        run_snapshots=current_policy_run_snapshots,
        bucket_name="active_policy",
    )
    runner = _build_policy_validation_runner_from_run_snapshots(
        base_dir=base_dir,
        run_snapshots=filtered_validation_runs,
        lookback_days=lookback_days,
        generated_at=str(raw_runner.get("generated_at", datetime.now(timezone.utc).isoformat()) or datetime.now(timezone.utc).isoformat()),
        baseline_control_comparison=dict(raw_runner.get("baseline_control_comparison", {}) or {}),
        lineage_attribution={
            **dict(lineage_attribution or {}),
            "current_policy_lineage": dict(current_active_lineage),
            "current_policy_evidence_lineage": dict(current_policy_evidence_lineage),
            "current_policy_evidence_alignment": dict(current_evidence_lineage_alignment),
        },
    )
    runner_evidence = dict(runner.get("evidence", {}) or {})
    baseline_control_comparison = dict(runner.get("baseline_control_comparison", {}) or {})
    delta = round(candidate_score - current_score, 6)
    structural_verdict = "keep"
    if delta > 0.1:
        structural_verdict = "candidate_better"
    elif delta < -0.1:
        structural_verdict = "candidate_worse"
    current_policy_evidence_for_comparison = (
        current_policy_evidence
        if bool(current_evidence_lineage_alignment.get("aligned"))
        else {}
    )
    current_policy_evidence_for_comparison, current_policy_bucket_overlay = _overlay_policy_context_bucket_evidence(
        current_policy_evidence_for_comparison,
        bucket_evidence=current_policy_direct_bucket_evidence,
    )
    current_policy_bucket_evidence = dict(current_policy_direct_bucket_evidence or {})
    candidate_policy_bucket_evidence = dict(
        dict(runner.get("policy_context_bucket_evidence", {}) or {}).get("staged_candidate", {})
        or {}
    )
    runtime_comparison = _compare_runtime_evidence(
        candidate_evidence=runner_evidence,
        current_evidence=current_policy_evidence_for_comparison,
    )
    runtime_verdict = str(runtime_comparison.get("runtime_comparison_verdict", "keep"))
    shared_validation_runs = [dict(item) for item in list(runner.get("validation_runs", []) or [])]
    shared_walk_forward_windows = [dict(item) for item in list(runner.get("walk_forward_windows", []) or [])]
    current_validation_runs = [
        dict(item)
        for item in list(current_policy_evidence_for_comparison.get("validation_runs", []) or shared_validation_runs)
    ]
    current_walk_forward_windows = [
        dict(item)
        for item in list(current_policy_evidence_for_comparison.get("walk_forward_windows", []) or shared_walk_forward_windows)
    ]
    candidate_symbol_summary = list(runner.get("symbol_summary", []) or [])
    candidate_regime_summary = list(runner.get("regime_summary", []) or [])
    current_symbol_summary = list(current_policy_evidence_for_comparison.get("symbol_summary", []) or candidate_symbol_summary)
    current_regime_summary = list(current_policy_evidence_for_comparison.get("regime_summary", []) or candidate_regime_summary)
    projected_candidate_replay_summary = _execution_replay_summary_from_runs(
        validation_runs=shared_validation_runs,
        walk_forward_windows=shared_walk_forward_windows,
        symbol_summary=candidate_symbol_summary,
        regime_summary=candidate_regime_summary,
        policy_application=candidate_policy_application,
        baseline_policy_application=current_policy_application,
        source="projected_candidate_policy_from_runtime_artifacts",
        runtime_summary=current_runtime_summary,
        runtime_summary_run_dir=runtime_summary_run_dir,
    )
    direct_candidate_replay_summary = _execution_replay_summary_from_bucket_evidence(
        bucket_evidence=candidate_policy_bucket_evidence,
        policy_application=candidate_policy_application,
        source="observed_staged_candidate_policy_bucket_artifacts",
    )
    candidate_replay_summary = (
        direct_candidate_replay_summary
        if _replay_summary_available(direct_candidate_replay_summary)
        else projected_candidate_replay_summary
    )
    projected_current_replay_summary = _execution_replay_summary_from_runs(
        validation_runs=current_validation_runs,
        walk_forward_windows=current_walk_forward_windows,
        symbol_summary=current_symbol_summary,
        regime_summary=current_regime_summary,
        policy_application=current_policy_application,
        baseline_policy_application=current_policy_application,
        source=(
            "observed_runtime_policy_bucket_artifacts"
            if bool(current_policy_bucket_overlay.get("used_validation_runs"))
            or bool(current_policy_bucket_overlay.get("used_walk_forward_windows"))
            else (
                "persisted_policy_validation_evidence"
                if list(current_policy_evidence_for_comparison.get("validation_runs", []) or [])
            else "observed_runtime_artifacts"
            )
        ),
        runtime_summary=current_runtime_summary,
        runtime_summary_run_dir=runtime_summary_run_dir,
    )
    direct_current_replay_summary = _execution_replay_summary_from_bucket_evidence(
        bucket_evidence=current_policy_bucket_evidence,
        policy_application=current_policy_application,
        source="observed_runtime_policy_bucket_artifacts",
        runtime_summary=current_runtime_summary,
        runtime_summary_run_dir=runtime_summary_run_dir,
    )
    current_replay_summary = (
        direct_current_replay_summary
        if _replay_summary_available(direct_current_replay_summary)
        else projected_current_replay_summary
    )
    baseline_control_comparison = _bucket_aware_baseline_control_comparison(
        baseline_control_comparison=baseline_control_comparison,
        current_policy_bucket_evidence=current_policy_bucket_evidence,
        current_policy_direct_replay_summary=direct_current_replay_summary,
        staged_candidate_bucket_evidence=candidate_policy_bucket_evidence,
        staged_candidate_direct_replay_summary=direct_candidate_replay_summary,
    )
    validation_path = {
        "mode": str(runner.get("validation_path_mode", "artifact_walk_forward")),
        "candidate_run_count": _safe_int(runner.get("run_count")),
        "candidate_walk_forward_window_count": _safe_int(runner.get("runner_walk_forward_window_count")),
        "candidate_positive_walk_forward_ratio": round(_safe_float(runner.get("runner_positive_walk_forward_ratio")), 6),
        "candidate_replay_source": str(candidate_replay_summary.get("source", "") or ""),
        "current_evidence_available": bool(runtime_comparison.get("runtime_evidence_available")),
        "current_walk_forward_window_count": _safe_int(current_policy_evidence_for_comparison.get("runner_walk_forward_window_count")),
        "current_positive_walk_forward_ratio": round(_safe_float(current_policy_evidence_for_comparison.get("runner_positive_walk_forward_ratio")), 6),
        "current_replay_source": str(current_replay_summary.get("source", "") or ""),
        "baseline_control_evidence_source": str(baseline_control_comparison.get("evidence_source", "") or ""),
        "baseline_control_replay_grounding": str(baseline_control_comparison.get("replay_grounding", "") or ""),
        "compared_metrics": list(runtime_comparison.get("compared_metrics", [])),
        "lineage_attribution_mode": str(dict(lineage_attribution or {}).get("mode", "unfiltered") or "unfiltered"),
        "current_policy_evidence_alignment": dict(current_evidence_lineage_alignment),
    }
    counterfactual_replay_path = _counterfactual_replay_path(
        validation_mode=str(runner.get("validation_path_mode", "artifact_walk_forward")),
        candidate_policy_score=candidate_score,
        current_policy_score=current_score,
        candidate_replay_summary=candidate_replay_summary,
        current_replay_summary=current_replay_summary,
        candidate_evidence=runner_evidence,
        current_evidence=current_policy_evidence_for_comparison,
        current_evidence_available=bool(runtime_comparison.get("runtime_evidence_available")),
        candidate_policy_application=candidate_policy_application,
        current_policy_application=current_policy_application,
    )
    execution_style_comparison = dict(counterfactual_replay_path.get("execution_style_comparison", {}) or {})
    execution_style_summary = dict(execution_style_comparison.get("comparison_summary", {}) or {})
    execution_replay_delta = round(_safe_float(execution_style_comparison.get("execution_replay_score_delta")), 6)
    execution_replay_verdict = str(execution_style_comparison.get("execution_replay_verdict", "keep") or "keep")
    comparison_delta = execution_replay_delta if execution_replay_verdict != "keep" or shared_validation_runs else delta
    verdict = (
        execution_replay_verdict
        if execution_replay_verdict != "keep"
        else (runtime_verdict if runtime_verdict != "keep" else structural_verdict)
    )
    policy_application_delta = dict(
        execution_style_summary.get("policy_application_delta", {})
        or {}
    )
    policy_evidence_buckets = _build_policy_evidence_buckets(
        candidate_policy=candidate_policy,
        runner_evidence=runner_evidence,
        current_policy_evidence=current_policy_evidence_for_comparison,
        current_active_lineage=current_active_lineage,
        current_policy_evidence_lineage=current_policy_evidence_lineage,
        current_evidence_lineage_alignment=current_evidence_lineage_alignment,
        baseline_control_comparison=baseline_control_comparison,
        candidate_policy_application=candidate_policy_application,
        current_policy_application=current_policy_application,
        candidate_replay_summary=candidate_replay_summary,
        current_replay_summary=current_replay_summary,
    )
    policy_evidence_buckets["staged_candidate"]["evidence"].update(
        {
            "comparison_verdict": verdict,
            "comparison_structural_verdict": structural_verdict,
            "comparison_runtime_verdict": runtime_verdict,
            "comparison_execution_replay_verdict": execution_replay_verdict,
            "candidate_vs_current_structural_score_delta": delta,
            "candidate_vs_current_execution_replay_score_delta": execution_replay_delta,
            "candidate_vs_current_score_delta": comparison_delta,
        }
    )
    checkpoint_auto_judge = _build_checkpoint_auto_judge(
        current_policy_state=current_policy_state,
        comparison_verdict=verdict,
        comparison_delta=comparison_delta,
        runner_artifact=runner,
        baseline_control_comparison=baseline_control_comparison,
    )
    symbol_lifecycle = list(checkpoint_auto_judge.get("symbol_lifecycle", runner.get("symbol_lifecycle", [])) or [])
    symbol_lifecycle_summary = dict(
        checkpoint_auto_judge.get("symbol_lifecycle_summary", runner.get("symbol_lifecycle_summary", {})) or {}
    )
    auto_mode = build_regime_aware_auto_mode(
        regime_summary=list(runner.get("regime_summary", []) or []),
        sample_quality_watchdog=dict(runner.get("sample_quality_watchdog", {}) or {}),
        checkpoint_auto_judge=checkpoint_auto_judge,
        baseline_control_comparison=baseline_control_comparison,
        execution_quality={
            "run_count": runner.get("run_count", 0),
            "total_closed_trade_count": runner.get("total_closed_trade_count", 0),
            "total_live_order_count": runner.get("total_live_order_count", 0),
            "runner_total_realized_pnl_usd": runner.get("runner_total_realized_pnl_usd", 0.0),
            "runner_drawdown_to_pnl_ratio": runner.get("runner_drawdown_to_pnl_ratio", 0.0),
            "runner_reject_rate": runner.get("runner_reject_rate", 0.0),
            "runner_protection_degraded_rate": runner.get("runner_protection_degraded_rate", 0.0),
            "runner_avg_realized_edge_bps": runner.get("runner_avg_realized_edge_bps", 0.0),
            "runner_avg_edge_retention_ratio": runner.get("runner_avg_edge_retention_ratio", 0.0),
            "runner_walk_forward_window_count": runner.get("runner_walk_forward_window_count", 0),
            "runner_positive_walk_forward_ratio": runner.get("runner_positive_walk_forward_ratio", 0.0),
            "micro_live_gate": runner.get("micro_live_gate", {}),
            "policy_context_bucket_evidence": runner.get("policy_context_bucket_evidence", {}),
            "policy_evidence_buckets": policy_evidence_buckets,
            "preferred_policy_bucket": "staged_candidate",
        },
        symbol_lifecycle_summary=symbol_lifecycle_summary,
        symbol_lifecycle=symbol_lifecycle,
    )
    evidence = {
        "comparison_verdict": verdict,
        "comparison_structural_verdict": structural_verdict,
        "comparison_runtime_verdict": runtime_verdict,
        "comparison_execution_replay_verdict": execution_replay_verdict,
        "candidate_policy_score": candidate_score,
        "current_policy_score": current_score,
        "candidate_execution_replay_score": round(_safe_float(candidate_replay_summary.get("execution_score")), 6),
        "current_execution_replay_score": round(_safe_float(current_replay_summary.get("execution_score")), 6),
        "candidate_vs_current_structural_score_delta": delta,
        "candidate_vs_current_execution_replay_score_delta": execution_replay_delta,
        "candidate_vs_current_score_delta": comparison_delta,
        "runner_total_return_pct": runner.get("runner_total_return_pct", 0.0),
        "runner_total_realized_pnl_usd": runner.get("runner_total_realized_pnl_usd", 0.0),
        "runner_max_drawdown_pct": runner.get("runner_max_drawdown_pct", 0.0),
        "runner_max_drawdown_usd": runner.get("runner_max_drawdown_usd", 0.0),
        "runner_drawdown_to_pnl_ratio": runner.get("runner_drawdown_to_pnl_ratio", 0.0),
        "runner_shadow_alignment_score": runner.get("runner_shadow_alignment_score", 0.0),
        "runner_reject_rate": runner.get("runner_reject_rate", 0.0),
        "runner_protection_degraded_rate": runner.get("runner_protection_degraded_rate", 0.0),
        "runner_avg_slippage_bps": runner.get("runner_avg_slippage_bps", 0.0),
        "runner_avg_realized_edge_bps": runner.get("runner_avg_realized_edge_bps", 0.0),
        "runner_avg_edge_retention_ratio": runner.get("runner_avg_edge_retention_ratio", 0.0),
        "runner_positive_walk_forward_ratio": runner.get("runner_positive_walk_forward_ratio", 0.0),
        "micro_live_gate": runner.get("micro_live_gate", {}),
        "recent_retention_window": runner.get("recent_retention_window", {}),
        "cumulative_retention_window": runner.get("cumulative_retention_window", {}),
        "sample_quality_watchdog": runner.get("sample_quality_watchdog", {}),
        "baseline_control_comparison": baseline_control_comparison,
        "checkpoint_auto_judge": checkpoint_auto_judge,
        "auto_mode": auto_mode,
        "symbol_lifecycle": symbol_lifecycle,
        "symbol_lifecycle_summary": symbol_lifecycle_summary,
        "lineage_attribution": dict(runner.get("lineage_attribution", {}) or {}),
        "current_policy_lineage": dict(current_active_lineage),
        "current_policy_evidence_lineage": dict(current_policy_evidence_lineage),
        "current_policy_evidence_alignment": dict(current_evidence_lineage_alignment),
        "candidate_vs_current_validation_path": validation_path,
        "counterfactual_replay_path": counterfactual_replay_path,
        "candidate_replay_summary": candidate_replay_summary,
        "current_replay_summary": current_replay_summary,
        "candidate_policy_application": candidate_policy_application,
        "current_policy_application": current_policy_application,
        "policy_application_delta": policy_application_delta,
        "policy_application_comparison": dict(execution_style_summary.get("policy_application_comparison", {}) or {}),
        "execution_path_comparison": dict(execution_style_summary.get("execution_path_comparison", {}) or {}),
        "replay_evidence_comparison": dict(execution_style_summary.get("replay_evidence_comparison", {}) or {}),
        "execution_replay_metric_delta": dict(execution_style_summary.get("execution_metric_delta", {}) or {}),
        "symbol_summary": runner.get("symbol_summary", []),
        "symbol_scorecard": runner.get("symbol_scorecard", []),
        "regime_summary": runner.get("regime_summary", []),
        "pruning_recommendations": runner.get("pruning_recommendations", []),
        "walk_forward_windows": runner.get("walk_forward_windows", []),
        "validation_runs": runner.get("validation_runs", []),
        **runtime_comparison,
    }
    evidence = with_policy_evidence_buckets(evidence, policy_evidence_buckets)
    return {
        "generated_at": runner.get("generated_at"),
        "comparison_verdict": verdict,
        "candidate_policy_score": candidate_score,
        "current_policy_score": current_score,
        "candidate_execution_replay_score": round(_safe_float(candidate_replay_summary.get("execution_score")), 6),
        "current_execution_replay_score": round(_safe_float(current_replay_summary.get("execution_score")), 6),
        "candidate_vs_current_structural_score_delta": delta,
        "candidate_vs_current_execution_replay_score_delta": execution_replay_delta,
        "candidate_vs_current_score_delta": comparison_delta,
        "comparison_execution_replay_verdict": execution_replay_verdict,
        "runner_total_return_pct": runner.get("runner_total_return_pct", 0.0),
        "runner_total_realized_pnl_usd": runner.get("runner_total_realized_pnl_usd", 0.0),
        "runner_max_drawdown_pct": runner.get("runner_max_drawdown_pct", 0.0),
        "runner_max_drawdown_usd": runner.get("runner_max_drawdown_usd", 0.0),
        "runner_drawdown_to_pnl_ratio": runner.get("runner_drawdown_to_pnl_ratio", 0.0),
        "runner_shadow_alignment_score": runner.get("runner_shadow_alignment_score", 0.0),
        "runner_reject_rate": runner.get("runner_reject_rate", 0.0),
        "runner_protection_degraded_rate": runner.get("runner_protection_degraded_rate", 0.0),
        "runner_avg_slippage_bps": runner.get("runner_avg_slippage_bps", 0.0),
        "runner_avg_realized_edge_bps": runner.get("runner_avg_realized_edge_bps", 0.0),
        "runner_avg_edge_retention_ratio": runner.get("runner_avg_edge_retention_ratio", 0.0),
        "runner_positive_walk_forward_ratio": runner.get("runner_positive_walk_forward_ratio", 0.0),
        "sample_quality_watchdog": runner.get("sample_quality_watchdog", {}),
        "baseline_control_comparison": baseline_control_comparison,
        "checkpoint_auto_judge": checkpoint_auto_judge,
        "auto_mode": auto_mode,
        "symbol_lifecycle": symbol_lifecycle,
        "symbol_lifecycle_summary": symbol_lifecycle_summary,
        "recent_retention_window": runner.get("recent_retention_window", {}),
        "cumulative_retention_window": runner.get("cumulative_retention_window", {}),
        "counterfactual_replay_path": counterfactual_replay_path,
        "candidate_replay_summary": candidate_replay_summary,
        "current_replay_summary": current_replay_summary,
        "validation_path": validation_path,
        "lineage_attribution": dict(runner.get("lineage_attribution", {}) or {}),
        "policy_evidence_buckets": policy_evidence_buckets,
        "evidence": evidence,
    }


def write_policy_comparison_validation_artifact(*,
    current_policy_state: dict[str, Any] | None,
    candidate_policy: dict[str, Any],
    base_dir: str | Path = "quant_runtime",
    output_path: str | Path,
    lookback_days: int = 7,
    current_runtime_summary: dict[str, Any] | None = None,
) -> Path:
    artifact = build_policy_comparison_validation_artifact(
        current_policy_state=current_policy_state,
        candidate_policy=candidate_policy,
        base_dir=base_dir,
        lookback_days=lookback_days,
        current_runtime_summary=current_runtime_summary,
    )
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    return target

def build_policy_validation_runner_artifact(*, base_dir: str | Path = "quant_runtime", lookback_days: int = 7) -> dict[str, object]:
    runs = _resolve_recent_runs(base_dir=Path(base_dir), lookback_days=lookback_days)
    run_snapshots = [_run_validation_snapshot(run_dir=run_dir) for run_dir in runs]
    baseline_control_comparison = _load_recent_baseline_control_comparison(base_dir=base_dir)
    return _build_policy_validation_runner_from_run_snapshots(
        base_dir=base_dir,
        run_snapshots=run_snapshots,
        lookback_days=lookback_days,
        generated_at=datetime.now(timezone.utc).isoformat(),
        baseline_control_comparison=baseline_control_comparison,
        lineage_attribution={
            "applied": False,
            "mode": "unfiltered_runner_artifact",
            "known_run_lineage_count": sum(1 for snapshot in run_snapshots if bool(dict(snapshot.get("policy_lineage", {}) or {}).get("available"))),
            "unknown_run_lineage_count": sum(1 for snapshot in run_snapshots if not bool(dict(snapshot.get("policy_lineage", {}) or {}).get("available"))),
            "aligned_run_count": 0,
            "mismatched_run_count": 0,
        },
    )


def write_policy_validation_runner_artifact(*, base_dir: str | Path = "quant_runtime", output_path: str | Path, lookback_days: int = 7) -> Path:
    artifact = build_policy_validation_runner_artifact(base_dir=base_dir, lookback_days=lookback_days)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    return target

def write_weekly_validation_report(*, report: WeeklyValidationReport, output_path: str | Path) -> Path:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report.as_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return target
