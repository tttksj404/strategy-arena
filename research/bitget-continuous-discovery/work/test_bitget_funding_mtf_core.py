"""Regression tests for funding-aware, public-data paper calculations."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np

from alt_search_models import MarketData, RiskOverlay
from bitget_funding_mtf_core import (
    FundingCandidate,
    FundingFamily,
    FundingSettlement,
    SignalInputs,
    ValidationRecord,
    build_positions,
    evaluate_funding_path,
    funding_component,
    FundingPathInputs,
    select_validation_winners,
)


def test_contrarian_positions_use_only_prior_funding_observation() -> None:
    """Given extreme funding, when a signal is built, then its first bar remains flat."""

    candidate = FundingCandidate(FundingFamily.CONTRARIAN, 1, 0.0, 1, 1, 1)
    close = np.full((4, 2), 100.0)
    funding = np.array(((0.001, -0.001), (0.001, -0.001), (0.001, -0.001), (0.001, -0.001)))

    positions = build_positions(SignalInputs(close, funding, 24.0), candidate)

    assert tuple(positions[0]) == (0.0, 0.0)
    assert tuple(positions[1]) == (-0.5, 0.5)


def test_funding_component_credits_short_positive_and_long_negative_rate() -> None:
    """Given a neutral carry pair, when funding settles, then both favorable legs credit PnL."""

    positions = np.array(((0.0, 0.0), (-0.5, 0.5)))
    funding = np.array(((0.0, 0.0), (0.01, -0.01)))
    leverage = np.array((0.0, 2.0))

    result = funding_component(FundingSettlement(positions, funding, leverage, -1))

    assert tuple(result) == (0.0, 0.02)


def test_validation_selection_ignores_unavailable_locked_test_values() -> None:
    """Given validation records, when winners are selected, then only their validation score decides."""

    low = FundingCandidate(FundingFamily.CONTRARIAN, 1, 0.0, 1, 1, 1)
    high = FundingCandidate(FundingFamily.CONTRARIAN, 3, 0.0, 1, 1, 1)
    records = (
        ValidationRecord(low, 3.0, 0.2, True),
        ValidationRecord(high, 3.0, 0.3, True),
    )

    winners = select_validation_winners(records)

    assert tuple(record.candidate.identifier for record in winners) == (high.identifier,)


def test_funding_path_reports_the_realized_leverage_not_only_its_cap() -> None:
    """Given no signal, when a capped path is evaluated, then its realized leverage remains zero."""

    prices = np.full((3, 1), 100.0)
    market = MarketData(
        (datetime.fromtimestamp(0, UTC), datetime.fromtimestamp(1, UTC), datetime.fromtimestamp(2, UTC)),
        ("ALTUSDT",), prices, prices, prices, prices, prices,
    )
    path = evaluate_funding_path(FundingPathInputs(market, prices * 0.0, prices * 0.0, RiskOverlay(5.0, 0.005, 2)))

    assert path.max_realized_leverage == 0.0
