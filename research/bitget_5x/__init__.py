"""Standalone, public-data-only extraction of StockPulse's Bitget 5x research core."""

import sys

if sys.version_info < (3, 11):
    raise RuntimeError("research.bitget_5x requires Python 3.11 or later")

from .candidates import PRECOMMITTED_REPLAY_RULES
from .intraday_long_forecast import REQUESTED_LEVERAGE, STRATEGY_VERSION

__all__ = ["PRECOMMITTED_REPLAY_RULES", "REQUESTED_LEVERAGE", "STRATEGY_VERSION"]
