from __future__ import annotations

from datetime import timedelta
from enum import StrEnum
from typing import Final, assert_never


ONE_MINUTE_INTERVAL: Final = timedelta(minutes=1)
FIVE_MINUTE_INTERVAL: Final = timedelta(minutes=5)
FIFTEEN_MINUTE_INTERVAL: Final = timedelta(minutes=15)
FOUR_HOUR_INTERVAL: Final = timedelta(hours=4)


class CandleGranularity(StrEnum):
    """Supported Bitget market-candle intervals used by this project."""

    ONE_MINUTE = "1m"
    FIVE_MINUTES = "5m"
    FIFTEEN_MINUTES = "15m"
    FOUR_HOURS = "4H"

    @property
    def interval(self) -> timedelta:
        match self:
            case CandleGranularity.ONE_MINUTE:
                return ONE_MINUTE_INTERVAL
            case CandleGranularity.FIVE_MINUTES:
                return FIVE_MINUTE_INTERVAL
            case CandleGranularity.FIFTEEN_MINUTES:
                return FIFTEEN_MINUTE_INTERVAL
            case CandleGranularity.FOUR_HOURS:
                return FOUR_HOUR_INTERVAL
            case unreachable:
                assert_never(unreachable)
