# Fee, slippage, funding-transfer, and stress-cost calculations.

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


PERP_TAKER_RATE: Final = 0.0006
SPOT_TAKER_RATE: Final = 0.0010


@dataclass(frozen=True, slots=True)
class LegCost:
    fee_rate: float
    slippage_rate: float


def transaction_cost(notional: float, leg: LegCost) -> float:
    return abs(notional) * (leg.fee_rate + leg.slippage_rate)


def funding_cashflow(notional: float, funding_rate: float, position: float) -> float:
    return -abs(notional) * funding_rate * position


def f1_round_trip_cost(notional: float, slippage_rate: float) -> float:
    spot = LegCost(fee_rate=SPOT_TAKER_RATE, slippage_rate=slippage_rate)
    perp = LegCost(fee_rate=PERP_TAKER_RATE, slippage_rate=slippage_rate)
    return 2.0 * (transaction_cost(notional, spot) + transaction_cost(notional, perp))


def slippage_rate(symbol: str, stress_multiplier: float = 1.0) -> float:
    normalized = symbol.upper().removesuffix("USDT")
    if normalized in {"BTC", "ETH", "SOL"}:
        basis_points = 1.0
    elif normalized in {"SPY", "QQQ", "TSLA", "NVDA", "MSTR"}:
        basis_points = 5.0
    else:
        basis_points = 3.0
    return basis_points * 0.0001 * stress_multiplier
