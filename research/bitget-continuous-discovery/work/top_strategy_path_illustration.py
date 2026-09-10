"""Illustrative path risk; its chosen barrier is not a Bitget liquidation formula."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PathRiskExample:
    prices: tuple[float, ...]
    barrier: float
    breach_index: int
    recovery_index: int


def path_risk_example() -> PathRiskExample:
    return PathRiskExample((100., 98., 94., 89., 81., 77., 83., 92., 101., 104., 106., 108., 109.),
                           barrier=82., breach_index=4, recovery_index=9)
