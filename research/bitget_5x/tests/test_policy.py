from __future__ import annotations

from datetime import UTC, datetime, timedelta
import json
from pathlib import Path

from research.bitget_5x.candidates import PRECOMMITTED_REPLAY_RULES
from research.bitget_5x.intraday_long_forecast import (
    REQUESTED_LEVERAGE,
    InstrumentProfile,
    IntradayLongPlan,
    evaluate_fixed_leverage_feasibility,
)
from research.bitget_5x.market_timeframes import CandleGranularity


def test_research_policy_is_fixed_at_five_x() -> None:
    assert REQUESTED_LEVERAGE == 5


def test_all_current_candidates_are_frozen_in_the_portable_registry() -> None:
    assert tuple(rule.name for rule in PRECOMMITTED_REPLAY_RULES) == (
        "breakout-volume-1.0",
        "breakout-volume-1.25",
        "breakout-volume-1.5",
        "breakout-volume-2.0",
        "breakout-technical-score-v1-volume-1.0",
        "breakout-bollinger-squeeze-v1-volume-1.0",
    )


def test_candidate_registry_matches_the_committed_contract() -> None:
    contract_path = Path(__file__).parents[1] / "results" / "2026-08-24-candidate-contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))

    assert tuple(candidate["name"] for candidate in contract["candidates"]) == tuple(
        rule.name for rule in PRECOMMITTED_REPLAY_RULES
    )


def test_all_in_stop_loss_gate_rejects_an_oversized_five_x_loss() -> None:
    starts_at = datetime(2026, 8, 12, 12, tzinfo=UTC)
    plan = IntradayLongPlan(
        profile=InstrumentProfile.STANDARD,
        trigger_granularity=CandleGranularity.FIVE_MINUTES,
        entry_price=100,
        target_price=101.4,
        stop_price=98.25,
        score=90,
        starts_at=starts_at,
        expires_at=starts_at + timedelta(minutes=60),
    )

    feasibility = evaluate_fixed_leverage_feasibility(plan, taker_fee_rate=0.0006)

    assert not feasibility.passes
    assert "total_stop_loss_cap_exceeded" in feasibility.reasons


def test_exchange_adapter_exposes_no_private_order_or_account_route() -> None:
    adapter = Path(__file__).parents[1] / "bitget_market.py"
    source = adapter.read_text(encoding="utf-8")

    assert "/api/v2/mix/market/" in source
    assert "/api/v2/mix/order" not in source
    assert "/api/v2/mix/account" not in source
    assert "Authorization" not in source
