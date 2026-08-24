"""Core analysis engine: reads trade data, computes parameter adjustments."""
from __future__ import annotations

import json
import sys
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any

from quant_binance.autotuner.parameter_registry import (
    ALL_TUNABLE_PARAMS,
    TIER1_ENTRY,
    TIER2_EXIT,
    TIER3_SIZING,
    TunableParam,
)
from quant_binance.autotuner.safety import RevertMonitor, validate_deltas
from quant_binance.autotuner.writer import OverrideWriter, _get_nested


def _load_all_closed_trades(base_dir: Path) -> list[dict[str, Any]]:
    """Load closed trades from all sessions' JSONL logs."""
    trades: list[dict[str, Any]] = []
    for f in sorted(base_dir.rglob("logs/closed_trades.jsonl")):
        try:
            for line in f.open(encoding="utf-8"):
                line = line.strip()
                if line:
                    trades.append(json.loads(line))
        except (json.JSONDecodeError, OSError):
            continue
    return trades


def _valid_trades(trades: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Filter to trades with valid PnL data."""
    return [
        t for t in trades
        if abs(float(t.get("realized_return_bps_estimate", 0) or 0)) >= 0.01
        and float(t.get("entry_predictability_score", 0) or t.get("latest_predictability_score", 0) or 0) > 0
    ]


# ---------------------------------------------------------------------------
# Tier 1: Entry threshold analysis
# ---------------------------------------------------------------------------

def _analyze_entry_thresholds(
    trades: list[dict[str, Any]],
    current_config: dict[str, Any],
) -> list[dict[str, Any]]:
    deltas = []

    # Group by score bucket (10-point buckets)
    buckets: dict[int, list[float]] = defaultdict(list)
    for t in trades:
        score = float(t.get("entry_predictability_score", 0) or t.get("latest_predictability_score", 0) or 0)
        if score <= 0:
            continue
        bucket = int(score // 10) * 10
        ret = float(t.get("realized_return_bps_estimate", 0) or 0)
        buckets[bucket].append(ret)

    # Analyze futures_score_min
    current_min = float(_get_nested(current_config, ("mode_thresholds", "futures_score_min"), 75.0))
    param = next(p for p in TIER1_ENTRY if p.path[-1] == "futures_score_min")
    below_bucket = int(current_min // 10) * 10 - 10
    at_bucket = int(current_min // 10) * 10

    if below_bucket in buckets and len(buckets[below_bucket]) >= 10:
        below_returns = buckets[below_bucket]
        win_rate = len([r for r in below_returns if r > 0]) / len(below_returns)
        avg_ret = mean(below_returns)
        if win_rate > 0.50 and avg_ret > 0:
            # Below-threshold bucket is profitable -> lower threshold
            new_val = param.clamp(current_min - param.max_delta(current_min))
            confidence = min(1.0, len(below_returns) / 20) * win_rate
            deltas.append({
                "path": list(param.path),
                "current_value": current_min,
                "proposed_value": round(new_val, 2),
                "delta_pct": round((new_val - current_min) / max(current_min, 1) * 100, 2),
                "confidence": round(confidence, 3),
                "sample_size": len(below_returns),
                "risk_tier": param.risk_tier,
                "min_trades": param.min_trades,
                "evidence": f"bucket {below_bucket}-{below_bucket+9}: n={len(below_returns)} win={win_rate:.0%} avg={avg_ret:+.1f}bps",
            })

    if at_bucket in buckets and len(buckets[at_bucket]) >= 10:
        at_returns = buckets[at_bucket]
        win_rate = len([r for r in at_returns if r > 0]) / len(at_returns)
        avg_ret = mean(at_returns)
        if win_rate < 0.40 and avg_ret < 0:
            # At-threshold bucket is losing -> raise threshold
            new_val = param.clamp(current_min + param.max_delta(current_min))
            confidence = min(1.0, len(at_returns) / 20) * (1 - win_rate)
            deltas.append({
                "path": list(param.path),
                "current_value": current_min,
                "proposed_value": round(new_val, 2),
                "delta_pct": round((new_val - current_min) / max(current_min, 1) * 100, 2),
                "confidence": round(confidence, 3),
                "sample_size": len(at_returns),
                "risk_tier": param.risk_tier,
                "min_trades": param.min_trades,
                "evidence": f"bucket {at_bucket}-{at_bucket+9}: n={len(at_returns)} win={win_rate:.0%} avg={avg_ret:+.1f}bps",
            })

    return deltas


# ---------------------------------------------------------------------------
# Tier 2: Stop/Exit analysis
# ---------------------------------------------------------------------------

def _analyze_stop_exit(
    trades: list[dict[str, Any]],
    current_config: dict[str, Any],
) -> list[dict[str, Any]]:
    deltas = []

    # Stop waste analysis
    stop_trades = [t for t in trades if "STOP" in str(t.get("exit_reason", ""))]
    if len(stop_trades) >= 15:
        stop_waste_count = 0
        for t in stop_trades:
            best = abs(float(t.get("best_return_bps", 0) or 0))
            worst = abs(float(t.get("worst_return_bps", 0) or 0))
            if best > worst * 0.5:
                stop_waste_count += 1
        waste_ratio = stop_waste_count / len(stop_trades)

        param = next(p for p in TIER2_EXIT if p.path[-1] == "atr_multiple_for_stop")
        current_atr = float(_get_nested(current_config, param.path, 2.0))

        if waste_ratio > 0.30:
            # Too many premature stops -> widen
            new_val = param.clamp(current_atr + param.max_delta(current_atr))
            deltas.append({
                "path": list(param.path),
                "current_value": current_atr,
                "proposed_value": round(new_val, 3),
                "delta_pct": round((new_val - current_atr) / max(current_atr, 0.01) * 100, 2),
                "confidence": round(min(0.9, waste_ratio), 3),
                "sample_size": len(stop_trades),
                "risk_tier": param.risk_tier,
                "min_trades": param.min_trades,
                "evidence": f"stop_waste={waste_ratio:.0%} ({stop_waste_count}/{len(stop_trades)}): widen stop",
            })
        elif waste_ratio < 0.15:
            new_val = param.clamp(current_atr - param.max_delta(current_atr))
            deltas.append({
                "path": list(param.path),
                "current_value": current_atr,
                "proposed_value": round(new_val, 3),
                "delta_pct": round((new_val - current_atr) / max(current_atr, 0.01) * 100, 2),
                "confidence": round(min(0.9, 1 - waste_ratio), 3),
                "sample_size": len(stop_trades),
                "risk_tier": param.risk_tier,
                "min_trades": param.min_trades,
                "evidence": f"stop_waste={waste_ratio:.0%}: stops are efficient, tighten",
            })

    # Holding time analysis
    timed_trades = [t for t in trades if float(t.get("holding_minutes", 0) or 0) > 0]
    if len(timed_trades) >= 20:
        sorted_by_time = sorted(timed_trades, key=lambda t: float(t.get("holding_minutes", 0)))
        n = len(sorted_by_time)
        q3_start = int(n * 0.75)
        q3_trades = sorted_by_time[q3_start:]
        q3_avg_ret = mean([float(t.get("realized_return_bps_estimate", 0) or 0) for t in q3_trades])

        if q3_avg_ret < 0 and len(q3_trades) >= 5:
            param = next(p for p in TIER2_EXIT if p.path[-1] == "futures_max_holding_minutes")
            current_max = float(_get_nested(current_config, param.path, 240.0))
            new_val = param.clamp(current_max - param.max_delta(current_max))
            deltas.append({
                "path": list(param.path),
                "current_value": current_max,
                "proposed_value": round(new_val, 1),
                "delta_pct": round((new_val - current_max) / max(current_max, 1) * 100, 2),
                "confidence": round(min(0.8, len(q3_trades) / 15), 3),
                "sample_size": len(q3_trades),
                "risk_tier": param.risk_tier,
                "min_trades": param.min_trades,
                "evidence": f"longest quartile avg={q3_avg_ret:+.1f}bps (n={len(q3_trades)}): reduce max holding",
            })

    return deltas


# ---------------------------------------------------------------------------
# Tier 3: Position sizing analysis
# ---------------------------------------------------------------------------

def _analyze_sizing(
    trades: list[dict[str, Any]],
    current_config: dict[str, Any],
) -> list[dict[str, Any]]:
    deltas = []
    pnls = [float(t.get("realized_pnl_usd_estimate", 0) or 0) for t in trades]
    if len(pnls) < 30:
        return deltas

    # Compute cumulative PnL and max drawdown
    cumulative = 0.0
    peak = 0.0
    max_dd = 0.0
    for p in pnls:
        cumulative += p
        peak = max(peak, cumulative)
        dd = peak - cumulative
        max_dd = max(max_dd, dd)

    total_pnl = sum(pnls)
    calmar = total_pnl / max(max_dd, 0.01)

    param = next(p for p in TIER3_SIZING if p.path[-1] == "per_trade_equity_risk")
    current_risk = float(_get_nested(current_config, param.path, 0.005))

    if calmar > 1.5 and max_dd < abs(total_pnl) * 0.5:
        new_val = param.clamp(current_risk + param.max_delta(current_risk))
        deltas.append({
            "path": list(param.path),
            "current_value": current_risk,
            "proposed_value": round(new_val, 5),
            "delta_pct": round((new_val - current_risk) / max(current_risk, 0.0001) * 100, 2),
            "confidence": round(min(0.85, calmar / 3), 3),
            "sample_size": len(pnls),
            "risk_tier": param.risk_tier,
            "min_trades": param.min_trades,
            "evidence": f"calmar={calmar:.2f} dd=${max_dd:.2f} pnl=${total_pnl:.2f}: increase risk",
        })
    elif calmar < 0.5 or max_dd > abs(total_pnl) * 0.7:
        new_val = param.clamp(current_risk - param.max_delta(current_risk))
        deltas.append({
            "path": list(param.path),
            "current_value": current_risk,
            "proposed_value": round(new_val, 5),
            "delta_pct": round((new_val - current_risk) / max(current_risk, 0.0001) * 100, 2),
            "confidence": 0.85,
            "sample_size": len(pnls),
            "risk_tier": param.risk_tier,
            "min_trades": param.min_trades,
            "evidence": f"calmar={calmar:.2f} dd=${max_dd:.2f} pnl=${total_pnl:.2f}: reduce risk",
        })

    return deltas


# ---------------------------------------------------------------------------
# Main analysis cycle
# ---------------------------------------------------------------------------

def run_analysis_cycle(
    *,
    base_dir: str | Path,
    override_path: str | Path,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run one full auto-tuning analysis cycle."""
    base_dir = Path(base_dir)
    override_path = Path(override_path)
    autotuner_dir = base_dir / "artifacts" / "autotuner"

    writer = OverrideWriter(override_path=override_path, autotuner_dir=autotuner_dir)
    monitor = RevertMonitor(state_path=autotuner_dir / "revert_monitor.json")

    # Check if revert is needed from previous cycle
    if monitor.active and monitor.state.revert_triggered:
        reverted = writer.revert_to_baseline(reason=monitor.state.revert_reason)
        monitor.clear()
        return {"action": "reverted", "reason": monitor.state.revert_reason, "success": reverted}

    # Load all trades
    paper_shell_dir = base_dir / "output" / "paper-live-shell"
    all_trades = _load_all_closed_trades(paper_shell_dir)
    valid = _valid_trades(all_trades)

    result: dict[str, Any] = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "total_trades": len(all_trades),
        "valid_trades": len(valid),
        "dry_run": dry_run,
        "deltas_proposed": [],
        "deltas_approved": [],
        "rejections": [],
        "action": "none",
    }

    if len(valid) < 30:
        result["action"] = "insufficient_data"
        return result

    current_config = writer.read_current()

    # Run all analyses
    all_deltas: list[dict[str, Any]] = []
    all_deltas.extend(_analyze_entry_thresholds(valid, current_config))
    all_deltas.extend(_analyze_stop_exit(valid, current_config))
    all_deltas.extend(_analyze_sizing(valid, current_config))

    result["deltas_proposed"] = all_deltas

    if not all_deltas:
        result["action"] = "no_changes_needed"
        return result

    # Safety validation
    approved, rejections = validate_deltas(
        all_deltas,
        total_trades=len(valid),
        audit_path=autotuner_dir / "audit.jsonl",
    )
    result["deltas_approved"] = approved
    result["rejections"] = rejections

    if not approved:
        result["action"] = "all_rejected"
        return result

    # Apply
    change_id = f"auto-{datetime.now(tz=timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    applied = writer.apply_deltas(approved, change_id=change_id, dry_run=dry_run)

    if applied:
        # Start monitoring
        monitor.start_monitoring(
            change_id=change_id,
            deltas=approved,
            pre_trades=valid[-30:],  # Last 30 trades as baseline
        )
        result["action"] = "applied"
        result["change_id"] = change_id
    else:
        result["action"] = "dry_run_logged"

    return result


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Auto-tuner analysis cycle")
    parser.add_argument("--base-dir", required=True, help="quant_runtime directory")
    parser.add_argument("--override-path", default=None, help="strategy_override.approved.json path")
    parser.add_argument("--dry-run", action="store_true", help="Log proposals without applying")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    override_path = Path(args.override_path) if args.override_path else base_dir / "artifacts" / "strategy_override.approved.json"

    result = run_analysis_cycle(
        base_dir=base_dir,
        override_path=override_path,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
