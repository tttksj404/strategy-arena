"""Observable boundary tests for the public Bitget funding study runner."""

from __future__ import annotations

from bitget_funding_mtf_core import PaperMetrics
from bitget_mtf_core import MarketSnapshot
from run_bitget_funding_mtf_search import _metrics_payload, market_gate, risk_overlay


def test_market_gate_requires_an_active_liquid_tight_symbol() -> None:
    """Given public tickers, when a final signal is gated, then empty or wide markets fail closed."""

    markets = {
        "ETHUSDT": MarketSnapshot(5_000_000.0, 0.0005),
        "ALTUSDT": MarketSnapshot(5_000_000.0, 0.0020),
    }

    assert market_gate(("ETHUSDT",), markets)
    assert not market_gate((), markets)
    assert not market_gate(("ALTUSDT",), markets)


def test_risk_overlay_scales_the_daily_target_to_bar_duration() -> None:
    """Given an interval, when risk is built, then the cap stays fixed and bar target scales by time."""

    overlay = risk_overlay(5.0, 4.0)

    assert overlay.max_leverage == 5.0
    assert overlay.target_daily_vol == 0.005 * (4.0 / 24.0) ** 0.5


def test_metrics_payload_serializes_a_frozen_slots_value_object() -> None:
    """Given internal metrics, when report data is formed, then slot-backed fields remain available."""

    payload = _metrics_payload(PaperMetrics(0.1, 0.02, 1.5, 3, 20))

    assert payload.net_return == 0.1
    assert payload.observations == 20
