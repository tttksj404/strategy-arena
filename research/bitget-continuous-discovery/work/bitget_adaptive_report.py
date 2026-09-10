"""Report helpers for the adaptive Bitget counterfactual search."""
# pyright: reportImplicitRelativeImport=false

from __future__ import annotations

from typing import Final

import numpy as np

from alt_search_models import (
    BASE_FEE,
    BASE_SLIPPAGE,
    STRESS_SLIPPAGE,
    MarketData,
    PathResult,
    SegmentMetrics,
)
from bitget_adaptive_counterfactual import AdaptiveConfig, family, identifier
from bitget_directional_change_payloads import number_payload
from bitget_path_survival import (
    NoSurvivalEventsError,
    SurvivalMetrics,
    SurvivalSpec,
    evaluate_signal_survival,
)
from bitget_search_metrics import fixed_notional_metrics, metric_payload
from selection_bias_stats import SelectionBiasError

BASE_ROUND_TRIP_COST: Final = 2.0 * (BASE_FEE + BASE_SLIPPAGE)
STRESS_ROUND_TRIP_COST: Final = 2.0 * (BASE_FEE + STRESS_SLIPPAGE)
type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]


def _metrics(returns: np.ndarray, start: int, end: int, *, bars_per_year: float) -> SegmentMetrics:
    return fixed_notional_metrics(returns[start:end], bars_per_year=bars_per_year)


def _segment_hits(path: PathResult, start: int, end: int) -> tuple[int, ...]:
    """Filter official engine hit indices, with compatibility for legacy objects."""

    hits = path.liquidation_indices
    if not hits and path.first_liquidation_index >= 0:
        hits = (path.first_liquidation_index,)
    return tuple(index for index in hits if start <= index < end)


def _liquidation_payload(path: PathResult, start: int, end: int) -> dict[str, JsonValue]:
    """Return every official market-OHLC proxy hit in one segment."""

    hits = _segment_hits(path, start, end)
    return {"hit_count": len(hits), "indices": list(hits), "first_index": hits[0] if hits else None}


def _survival_at_cost(data: MarketData, positions: np.ndarray, spec: SurvivalSpec) -> SurvivalMetrics | None:
    """Return one survival result, treating only a known empty episode as empty."""

    try:
        return evaluate_signal_survival(data, positions, spec)
    except NoSurvivalEventsError:
        return None


def _survival_rows(
    data: MarketData,
    positions: np.ndarray,
    start: int,
    end: int,
    *,
    bar_minutes: int,
) -> list[JsonValue]:
    rows: list[JsonValue] = []
    for horizon in (3, 6, 12, 24, 48, 96):
        if end - start < horizon or start < 1:
            continue
        base = _survival_at_cost(data, positions, SurvivalSpec(start, end, horizon, 5.0, BASE_ROUND_TRIP_COST, 0.02))
        stress = _survival_at_cost(data, positions, SurvivalSpec(start, end, horizon, 5.0, STRESS_ROUND_TRIP_COST, 0.02))
        if base is None or stress is None:
            continue
        rows.append({"horizon_bars": horizon, "horizon_minutes": horizon * bar_minutes, "base": _survival_metric_payload(base), "stress": _survival_metric_payload(stress)})
    return rows


def _survival_metric_payload(metrics: SurvivalMetrics) -> dict[str, JsonValue]:
    """Serialize a typed survival result without an untyped dataclass dump."""

    return {
        "event_count": metrics.event_count,
        "cost_adjusted_terminal_win_rate": metrics.cost_adjusted_terminal_win_rate,
        "market_proxy_survival_rate": metrics.market_proxy_survival_rate,
        "profitable_survivor_rate": metrics.profitable_survivor_rate,
        "eventual_winner_market_proxy_liquidation_rate": metrics.eventual_winner_market_proxy_liquidation_rate,
        "mean_signal_equity_delta": metrics.mean_signal_equity_delta,
        "mae_median": metrics.mae_median,
        "mae_p95": metrics.mae_p95,
        "mfe_median": metrics.mfe_median,
        "survivor_time_to_profit_bars_median": metrics.survivor_time_to_profit_bars_median,
        "p95_market_proxy_leverage_boundary": metrics.p95_market_proxy_leverage_boundary,
        "exact_bitget_liquidation": metrics.exact_bitget_liquidation,
    }


def _candidate_payload(
    config: AdaptiveConfig,
    data: MarketData,
    positions: np.ndarray,
    path: PathResult,
    train_end: int,
    development_end: int,
    bars_per_year: float,
) -> dict[str, JsonValue]:
    validation_base = _metrics(path.returns, train_end, development_end, bars_per_year=bars_per_year)
    validation_stress = _metrics(path.stress_returns, train_end, development_end, bars_per_year=bars_per_year)
    train_base = _metrics(path.returns, 0, train_end, bars_per_year=bars_per_year)
    train_stress = _metrics(path.stress_returns, 0, train_end, bars_per_year=bars_per_year)
    return {
        "id": identifier(config),
        "family": family(config).value,
        "validation_base_fixed_notional": number_payload(metric_payload(validation_base)),
        "validation_stress_fixed_notional": number_payload(metric_payload(validation_stress)),
        "train_base_fixed_notional": number_payload(metric_payload(train_base)),
        "train_stress_fixed_notional": number_payload(metric_payload(train_stress)),
        "train_liquidation_proxy": _liquidation_payload(path, 1, train_end),
        "validation_liquidation_proxy": _liquidation_payload(path, train_end, development_end),
    }


def candidate_manifest(
    records: list[dict[str, JsonValue]],
    scores: dict[str, float],
    eligible: dict[str, bool],
) -> list[JsonValue]:
    """Return all candidate identifiers and validation scores used for selection."""

    return [
        {
            "id": row["id"],
            "family": row["family"],
            "validation_base_net_return": _metric_net_return(row, "validation_base_fixed_notional"),
            "validation_stress_net_return": _metric_net_return(row, "validation_stress_fixed_notional"),
            "validation_score": scores[str(row["id"])],
            "validation_score_stress_net_return": scores[str(row["id"])],
            "eligible": eligible[str(row["id"])],
        }
        for row in records
    ]


def _metric_net_return(record: dict[str, JsonValue], key: str) -> float:
    """Read a metric field from a report record at the serialization boundary."""

    payload = record.get(key)
    if not isinstance(payload, dict):
        raise SelectionBiasError(f"candidate record is missing metric mapping: {key}")
    value = payload.get("net_return")
    if not isinstance(value, (float, int)) or isinstance(value, bool):
        raise SelectionBiasError(f"candidate record has invalid metric value: {key}")
    return float(value)


def split_payload(data: MarketData, train_end: int, development_end: int) -> dict[str, JsonValue]:
    """Serialize chronological split bounds with timestamps."""

    return {
        "train": {"start": data.dates[0].isoformat(), "end": data.dates[train_end - 1].isoformat(), "rows": train_end},
        "validation": {"start": data.dates[train_end].isoformat(), "end": data.dates[development_end - 1].isoformat(), "rows": development_end - train_end},
        "locked_test": {"start": data.dates[development_end].isoformat(), "end": data.dates[-1].isoformat(), "rows": len(data.dates) - development_end},
    }
