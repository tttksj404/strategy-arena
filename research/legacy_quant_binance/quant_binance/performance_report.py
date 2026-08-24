from __future__ import annotations

import json
from collections import Counter, defaultdict
from math import floor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from quant_binance.closed_trade_metrics import (
    aggregate_closed_trades,
    filter_inherited_trades,
    load_closed_trades_jsonl,
)

import logging

logger = logging.getLogger(__name__)
_SEEN_CONTAMINATION_WARNINGS: set[tuple[str, str, tuple[str, ...], int]] = set()


@dataclass(frozen=True)
class SymbolExpectancy:
    symbol: str
    trade_count: int
    win_count: int
    loss_count: int
    hit_rate: float
    realized_pnl_usd: float
    average_pnl_usd: float
    average_return_bps: float
    expectancy_usd: float


@dataclass(frozen=True)
class RegimePerformance:
    mode: str
    decision_count: int
    avg_score: float
    avg_net_edge_bps: float
    avg_cost_bps: float


@dataclass(frozen=True)
class ScoreBucketPerformance:
    score_bucket_label: str
    trade_count: int
    win_count: int
    loss_count: int
    hit_rate: float
    realized_pnl_usd: float
    expectancy_usd: float
    average_return_bps: float


@dataclass(frozen=True)
class RuntimePerformanceReport:
    run_dir: str
    summary_path: str
    symbol_expectancy: tuple[SymbolExpectancy, ...]
    regime_performance: tuple[RegimePerformance, ...]
    score_bucket_performance: tuple[ScoreBucketPerformance, ...]
    walk_forward: tuple[dict[str, object], ...]
    pruning_recommendations: tuple[dict[str, object], ...]
    closed_trade_count: int
    realized_pnl_usd: float

    def as_dict(self) -> dict[str, object]:
        return {
            "run_dir": self.run_dir,
            "summary_path": self.summary_path,
            "symbol_expectancy": [asdict(item) for item in self.symbol_expectancy],
            "regime_performance": [asdict(item) for item in self.regime_performance],
            "score_bucket_performance": [asdict(item) for item in self.score_bucket_performance],
            "walk_forward": list(self.walk_forward),
            "pruning_recommendations": list(self.pruning_recommendations),
            "closed_trade_count": self.closed_trade_count,
            "realized_pnl_usd": self.realized_pnl_usd,
        }


def _score_bucket_label(score: float) -> str:
    bounded = max(0.0, min(float(score), 100.0))
    lower = int(floor(bounded / 10.0) * 10)
    upper = min(lower + 9, 100)
    if lower >= 100:
        return "100"
    return f"{lower:02d}-{upper:02d}"


def _warn_inherited_trade_contamination_once(
    *,
    source: str,
    run_dir: Path,
    inherited: list[dict[str, Any]],
) -> None:
    if not inherited:
        return
    symbols = tuple(sorted({str(trade.get("symbol", "?")) for trade in inherited}))
    fingerprint = (source, str(run_dir), symbols, len(inherited))
    if fingerprint in _SEEN_CONTAMINATION_WARNINGS:
        return
    _SEEN_CONTAMINATION_WARNINGS.add(fingerprint)
    if source == "from_rows":
        logger.warning(
            "[POLICY_VALIDATION_CONTAMINATION] %d inherited manual-close trade(s) excluded "
            "from policy validation (from_rows path). Symbols: %s.",
            len(inherited),
            list(symbols),
        )
        return
    logger.warning(
        "[POLICY_VALIDATION_CONTAMINATION] %d inherited manual-close trade(s) excluded "
        "from policy validation - these positions were entered in a prior session and "
        "closed via MANUAL_CLOSE_SYNCED, not by the current strategy. "
        "Symbols: %s. Excluding to prevent false rollback verdict.",
        len(inherited),
        list(symbols),
    )


def build_runtime_performance_report_from_rows(
    *,
    run_dir: str | Path,
    summary_path: str | Path | None = None,
    decisions: list[dict[str, Any]],
    closed_trades: list[dict[str, Any]],
) -> RuntimePerformanceReport:
    """Build a RuntimePerformanceReport from pre-loaded row lists.

    Callers that have already loaded decisions and closed_trades from their own
    sources (e.g. filtered by policy bucket) can use this variant instead of
    ``build_runtime_performance_report`` which always reads from disk.

    Inherited manual-close trades are filtered out here as well.
    """
    root = Path(run_dir)
    _summary_path = Path(summary_path) if summary_path is not None else root / "summary.json"
    _raw_closed_trades = list(closed_trades)
    clean_trades, _inherited = filter_inherited_trades(_raw_closed_trades)
    _warn_inherited_trade_contamination_once(
        source="from_rows",
        run_dir=root,
        inherited=_inherited,
    )
    return _build_report_from_clean_rows(
        root=root,
        summary_path=_summary_path,
        decisions=list(decisions),
        closed_trades=clean_trades,
    )


def build_runtime_performance_report(*, run_dir: str | Path) -> RuntimePerformanceReport:
    root = Path(run_dir)
    summary_path = root / "summary.json"
    _raw_closed_trades = load_closed_trades_jsonl(root / "logs" / "closed_trades.jsonl")
    closed_trades, _inherited = filter_inherited_trades(_raw_closed_trades)
    _warn_inherited_trade_contamination_once(
        source="disk",
        run_dir=root,
        inherited=_inherited,
    )
    decisions = load_closed_trades_jsonl(root / "logs" / "decisions.jsonl")
    return _build_report_from_clean_rows(
        root=root,
        summary_path=summary_path,
        decisions=decisions,
        closed_trades=closed_trades,
    )


def _build_report_from_clean_rows(
    *,
    root: Path,
    summary_path: Path,
    decisions: list[dict[str, Any]],
    closed_trades: list[dict[str, Any]],
) -> RuntimePerformanceReport:
    closed_trade_metrics = aggregate_closed_trades(closed_trades)

    by_symbol: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {
            "trade_count": 0,
            "win_count": 0,
            "loss_count": 0,
            "realized_pnl_usd": 0.0,
            "return_bps_sum": 0.0,
        }
    )
    score_buckets: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {
            "trade_count": 0,
            "win_count": 0,
            "loss_count": 0,
            "realized_pnl_usd": 0.0,
            "return_bps_sum": 0.0,
        }
    )
    for trade in closed_trades:
        symbol = str(trade.get("symbol", ""))
        pnl = float(trade.get("realized_pnl_usd_estimate", 0.0))
        bps = float(trade.get("realized_return_bps_estimate", 0.0))
        if not symbol:
            continue
        bucket = by_symbol[symbol]
        bucket["trade_count"] = int(bucket["trade_count"]) + 1
        bucket["realized_pnl_usd"] = float(bucket["realized_pnl_usd"]) + pnl
        bucket["return_bps_sum"] = float(bucket["return_bps_sum"]) + bps
        if pnl > 0:
            bucket["win_count"] = int(bucket["win_count"]) + 1
        elif pnl < 0:
            bucket["loss_count"] = int(bucket["loss_count"]) + 1
        bucket_score = float(
            trade.get("entry_predictability_score")
            if trade.get("entry_predictability_score") is not None
            else trade.get("latest_predictability_score", 0.0)
        )
        score_bucket = score_buckets[_score_bucket_label(bucket_score)]
        score_bucket["trade_count"] = int(score_bucket["trade_count"]) + 1
        score_bucket["realized_pnl_usd"] = float(score_bucket["realized_pnl_usd"]) + pnl
        score_bucket["return_bps_sum"] = float(score_bucket["return_bps_sum"]) + bps
        if pnl > 0:
            score_bucket["win_count"] = int(score_bucket["win_count"]) + 1
        elif pnl < 0:
            score_bucket["loss_count"] = int(score_bucket["loss_count"]) + 1

    symbol_rows: list[SymbolExpectancy] = []
    for symbol, bucket in by_symbol.items():
        trade_count = int(bucket["trade_count"])
        realized_pnl = float(bucket["realized_pnl_usd"])
        avg_pnl = realized_pnl / trade_count if trade_count else 0.0
        avg_bps = float(bucket["return_bps_sum"]) / trade_count if trade_count else 0.0
        win_count = int(bucket["win_count"])
        loss_count = int(bucket["loss_count"])
        hit_rate = win_count / trade_count if trade_count else 0.0
        symbol_rows.append(
            SymbolExpectancy(
                symbol=symbol,
                trade_count=trade_count,
                win_count=win_count,
                loss_count=loss_count,
                hit_rate=round(hit_rate, 6),
                realized_pnl_usd=round(realized_pnl, 6),
                average_pnl_usd=round(avg_pnl, 6),
                average_return_bps=round(avg_bps, 6),
                expectancy_usd=round(avg_pnl, 6),
            )
        )
    symbol_rows.sort(key=lambda item: (item.expectancy_usd, item.realized_pnl_usd), reverse=True)

    regime_buckets: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {"decision_count": 0, "score_sum": 0.0, "net_edge_sum": 0.0, "cost_sum": 0.0}
    )
    for decision in decisions:
        mode = str(decision.get("final_mode", "unknown"))
        bucket = regime_buckets[mode]
        bucket["decision_count"] = int(bucket["decision_count"]) + 1
        bucket["score_sum"] = float(bucket["score_sum"]) + float(decision.get("predictability_score", 0.0))
        bucket["net_edge_sum"] = float(bucket["net_edge_sum"]) + float(decision.get("net_expected_edge_bps", 0.0))
        bucket["cost_sum"] = float(bucket["cost_sum"]) + float(decision.get("estimated_round_trip_cost_bps", 0.0))

    regime_rows: list[RegimePerformance] = []
    for mode, bucket in regime_buckets.items():
        count = int(bucket["decision_count"])
        regime_rows.append(
            RegimePerformance(
                mode=mode,
                decision_count=count,
                avg_score=round(float(bucket["score_sum"]) / count, 6) if count else 0.0,
                avg_net_edge_bps=round(float(bucket["net_edge_sum"]) / count, 6) if count else 0.0,
                avg_cost_bps=round(float(bucket["cost_sum"]) / count, 6) if count else 0.0,
            )
        )
    regime_rows.sort(key=lambda item: item.mode)

    score_bucket_rows: list[ScoreBucketPerformance] = []
    for label, bucket in score_buckets.items():
        trade_count = int(bucket["trade_count"])
        realized_pnl = float(bucket["realized_pnl_usd"])
        win_count = int(bucket["win_count"])
        loss_count = int(bucket["loss_count"])
        score_bucket_rows.append(
            ScoreBucketPerformance(
                score_bucket_label=label,
                trade_count=trade_count,
                win_count=win_count,
                loss_count=loss_count,
                hit_rate=round(win_count / trade_count, 6) if trade_count else 0.0,
                realized_pnl_usd=round(realized_pnl, 6),
                expectancy_usd=round(realized_pnl / trade_count, 6) if trade_count else 0.0,
                average_return_bps=round(float(bucket["return_bps_sum"]) / trade_count, 6) if trade_count else 0.0,
            )
        )
    score_bucket_rows.sort(key=lambda item: item.score_bucket_label)

    decision_times = sorted(
        item.get("timestamp")
        for item in decisions
        if isinstance(item.get("timestamp"), str)
    )
    walk_forward: list[dict[str, object]] = []
    if decision_times:
        timestamps = [item for item in decision_times if item]
        window_size = max(len(timestamps) // 3, 1)
        for index in range(0, len(timestamps), window_size):
            window_timestamps = timestamps[index : index + window_size]
            start = window_timestamps[0]
            end = window_timestamps[-1]
            window_decisions = [
                item
                for item in decisions
                if start <= str(item.get("timestamp", "")) <= end
            ]
            window_modes = Counter(str(item.get("final_mode", "unknown")) for item in window_decisions)
            walk_forward.append(
                {
                    "window_index": len(walk_forward) + 1,
                    "start": start,
                    "end": end,
                    "decision_count": len(window_decisions),
                    "futures_count": window_modes.get("futures", 0),
                    "spot_count": window_modes.get("spot", 0),
                    "cash_count": window_modes.get("cash", 0),
                    "avg_score": round(
                        sum(float(item.get("predictability_score", 0.0)) for item in window_decisions)
                        / len(window_decisions),
                        6,
                    )
                    if window_decisions
                    else 0.0,
                    "avg_net_edge_bps": round(
                        sum(float(item.get("net_expected_edge_bps", 0.0)) for item in window_decisions)
                        / len(window_decisions),
                        6,
                    )
                    if window_decisions
                    else 0.0,
                }
            )

    rejection_by_symbol: dict[str, Counter[str]] = defaultdict(Counter)
    mode_by_symbol: dict[str, Counter[str]] = defaultdict(Counter)
    edge_sum_by_symbol: dict[str, float] = defaultdict(float)
    decision_count_by_symbol: Counter[str] = Counter()
    for decision in decisions:
        symbol = str(decision.get("symbol", ""))
        if not symbol:
            continue
        decision_count_by_symbol[symbol] += 1
        mode_by_symbol[symbol][str(decision.get("final_mode", "unknown"))] += 1
        edge_sum_by_symbol[symbol] += float(decision.get("net_expected_edge_bps", 0.0))
        for reason in decision.get("rejection_reasons", []) or []:
            rejection_by_symbol[symbol][str(reason)] += 1

    expectancy_by_symbol = {row.symbol: row for row in symbol_rows}
    pruning_recommendations: list[dict[str, object]] = []
    for symbol, decision_count in decision_count_by_symbol.items():
        expectancy = expectancy_by_symbol.get(symbol)
        avg_edge = edge_sum_by_symbol[symbol] / max(decision_count, 1)
        rejections = rejection_by_symbol[symbol]
        liquidity_weak = rejections.get("LIQUIDITY_TOO_WEAK", 0)
        edge_thin = rejections.get("EDGE_TOO_THIN", 0)
        cash_count = mode_by_symbol[symbol].get("cash", 0)
        recommendation = "keep"
        reason = "no clear downgrade signal"
        if expectancy is not None and expectancy.trade_count >= 2 and expectancy.expectancy_usd < 0:
            recommendation = "demote"
            reason = "negative expectancy on realized trades"
        elif cash_count >= 3 and (liquidity_weak + edge_thin) >= 3 and avg_edge <= 0:
            recommendation = "prune"
            reason = "repeated weak-liquidity or thin-edge rejections without positive edge"
        elif cash_count >= 3 and (liquidity_weak + edge_thin) >= 3:
            recommendation = "observe_only"
            reason = "repeated weak-liquidity or thin-edge rejections"
        pruning_recommendations.append(
            {
                "symbol": symbol,
                "recommendation": recommendation,
                "reason": reason,
                "decision_count": decision_count,
                "cash_count": cash_count,
                "avg_net_edge_bps": round(avg_edge, 6),
                "trade_count": expectancy.trade_count if expectancy is not None else 0,
                "expectancy_usd": expectancy.expectancy_usd if expectancy is not None else 0.0,
                "liquidity_too_weak_count": liquidity_weak,
                "edge_too_thin_count": edge_thin,
            }
        )
    pruning_recommendations.sort(
        key=lambda item: (
            {"prune": 0, "demote": 1, "observe_only": 2, "keep": 3}.get(str(item["recommendation"]), 4),
            float(item["expectancy_usd"]),
            float(item["avg_net_edge_bps"]),
        )
    )

    return RuntimePerformanceReport(
        run_dir=str(root),
        summary_path=str(summary_path),
        symbol_expectancy=tuple(symbol_rows),
        regime_performance=tuple(regime_rows),
        score_bucket_performance=tuple(score_bucket_rows),
        walk_forward=tuple(walk_forward),
        pruning_recommendations=tuple(pruning_recommendations),
        closed_trade_count=closed_trade_metrics.closed_trade_count,
        realized_pnl_usd=closed_trade_metrics.realized_pnl_usd,
    )


def write_runtime_performance_report(*, report: RuntimePerformanceReport, output_path: str | Path) -> Path:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report.as_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return target
