from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from quant_binance.data.rest_seed import seed_market_store_from_rest
from quant_binance.features.extractor import MarketFeatureExtractor
from quant_binance.features.primitive import build_feature_vector_from_primitives
from quant_binance.overlays import (
    apply_altcoin_overlay,
    apply_macro_overlay,
    apply_sentiment_overlay,
    load_altcoin_inputs,
    load_macro_inputs,
)
from quant_binance.settings import Settings
from quant_binance.strategy.regime import evaluate_snapshot
from quant_binance.data.snapshot_builder import SnapshotBuilder


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


@dataclass(frozen=True)
class CandidateResult:
    name: str
    objective_score: float
    futures_count: int
    spot_count: int
    cash_count: int
    total_positive_net_edge_bps: float
    total_negative_net_edge_bps: float
    symbols: list[dict[str, object]]
    overrides: dict[str, Any]


def _candidate_objective(decisions: list[dict[str, object]]) -> tuple[float, int, int, int, float, float]:
    futures_count = sum(1 for item in decisions if item["final_mode"] == "futures")
    spot_count = sum(1 for item in decisions if item["final_mode"] == "spot")
    cash_count = sum(1 for item in decisions if item["final_mode"] == "cash")
    positive_net = sum(max(float(item["net_expected_edge_bps"]), 0.0) for item in decisions)
    negative_net = sum(min(float(item["net_expected_edge_bps"]), 0.0) for item in decisions)
    objective = (
        futures_count * 7.0
        + spot_count * 3.0
        + positive_net * 0.25
        + negative_net * 0.05
        - cash_count * 0.25
    )
    return round(objective, 6), futures_count, spot_count, cash_count, round(positive_net, 6), round(negative_net, 6)


def _build_candidates() -> list[tuple[str, dict[str, Any]]]:
    return [
        ("active-baseline", {}),
        (
            "active-futures-lean",
            {
                "futures_exposure": {
                    "macro_score_relaxation": 6.0,
                    "macro_min_entry_net_edge_bps": 0.0,
                    "macro_edge_to_cost_multiple_min": 0.75,
                    "priority_score_relaxation": 6.5,
                    "priority_min_entry_net_edge_bps": 0.0,
                    "priority_edge_to_cost_multiple_min": 0.65,
                    "priority_volatility_relaxation": 0.15,
                    "priority_allow_caution": True,
                }
            },
        ),
        (
            "active-alt-spot-lean",
            {
                "spot_support": {
                    "priority_liquidity_relaxation": 0.2,
                    "priority_edge_relaxation_bps": 4.0,
                }
            },
        ),
        (
            "active-rotation-lean",
            {
                "futures_exposure": {
                    "priority_score_relaxation": 5.5,
                    "priority_min_entry_net_edge_bps": 0.25,
                },
                "portfolio_focus": {
                    "spot_top_n": 1,
                    "futures_top_n": 1,
                    "min_score_advantage_to_replace": 2.0,
                    "min_net_edge_advantage_bps": 1.0,
                    "min_incremental_pnl_usd": 0.5,
                },
            },
        ),
    ]


def run_sandbox_optimization(
    *,
    config_path: str | Path,
    client: Any,
    output_dir: str | Path,
) -> Path:
    raw = json.loads(Path(config_path).read_text(encoding="utf-8"))
    raw = _deep_merge(raw, raw.get("strategy_profiles", {}).get("active", {}))
    raw["strategy_profile"] = "active"
    macro_inputs = load_macro_inputs()
    altcoin_inputs = load_altcoin_inputs()

    base_settings = Settings.from_dict(raw)
    store = seed_market_store_from_rest(
        client=client,
        symbols=base_settings.universe,
        intervals=base_settings.klines,
    )
    extractor = MarketFeatureExtractor(base_settings)

    now = datetime.now(timezone.utc)
    interval = base_settings.decision_engine.decision_interval_minutes
    candidate_results: list[CandidateResult] = []
    for name, overrides in _build_candidates():
        candidate_raw = _deep_merge(raw, overrides)
        settings = Settings.from_dict(candidate_raw)
        builder = SnapshotBuilder(settings)
        rows: list[dict[str, object]] = []
        for symbol in settings.universe:
            state = store.get(symbol)
            if state is None:
                continue
            history = extractor.build_history_context(state)
            primitives = extractor.build_primitive_inputs(state)
            features = build_feature_vector_from_primitives(inputs=primitives, history=history, settings=settings)
            features = extractor.enrich_feature_vector(state=state, features=features)
            features = apply_macro_overlay(features, macro_inputs)
            features = apply_altcoin_overlay(features, symbol=symbol, altcoin_inputs=altcoin_inputs)
            features = apply_sentiment_overlay(features)
            last = state.last_update_time
            boundary = last.replace(second=0, microsecond=0)
            step = interval - (boundary.minute % interval)
            if step == 0:
                step = interval
            boundary = boundary + timedelta(minutes=step)
            snapshot = builder.build(state, features, boundary)
            decision = evaluate_snapshot(
                snapshot,
                settings,
                equity_usd=1000.0,
                remaining_portfolio_capacity_usd=500.0,
                cash_reserve_fraction=settings.cash_reserve.when_futures_enabled,
            )
            rows.append(
                {
                    "symbol": symbol,
                    "final_mode": decision.final_mode,
                    "predictability_score": decision.predictability_score,
                    "gross_expected_edge_bps": decision.gross_expected_edge_bps,
                    "net_expected_edge_bps": decision.net_expected_edge_bps,
                    "rejection_reasons": list(decision.rejection_reasons),
                }
            )
        objective, futures_count, spot_count, cash_count, positive_net, negative_net = _candidate_objective(rows)
        candidate_results.append(
            CandidateResult(
                name=name,
                objective_score=objective,
                futures_count=futures_count,
                spot_count=spot_count,
                cash_count=cash_count,
                total_positive_net_edge_bps=positive_net,
                total_negative_net_edge_bps=negative_net,
                symbols=rows,
                overrides=overrides,
            )
        )

    candidate_results.sort(key=lambda item: item.objective_score, reverse=True)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    output_path = output_root / f"sandbox-optimization-{timestamp}.json"
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "best_candidate": asdict(candidate_results[0]) if candidate_results else None,
        "candidates": [asdict(item) for item in candidate_results],
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    latest_path = output_root / "latest.json"
    latest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return output_path
