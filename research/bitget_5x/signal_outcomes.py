from __future__ import annotations

from enum import StrEnum


class SignalOutcome(StrEnum):
    """A settled replay outcome, deliberately independent of the StockPulse database."""

    PENDING = "pending"
    TARGET = "target"
    STOP = "stop"
    UNRESOLVED = "unresolved"
    AMBIGUOUS = "ambiguous"
