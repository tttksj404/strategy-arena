from __future__ import annotations

from bitget_directional_change_payloads import dsr_payload, number_payload, pbo_payload
from selection_bias_stats import DeflatedSharpeResult, PboResult


def test_directional_change_payload_helpers_preserve_numeric_contract() -> None:
    assert number_payload({"annual_return": 0.12, "trades": 3}) == {"annual_return": 0.12, "trades": 3}
    assert pbo_payload(PboResult(0.25, 70, 48, -0.4))["candidates"] == 48
    assert dsr_payload(DeflatedSharpeResult(1.2, 0.3, 0.96, 48, 100))["probability"] == 0.96
