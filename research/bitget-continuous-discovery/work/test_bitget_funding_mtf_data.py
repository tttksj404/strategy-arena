"""No-lookahead boundary tests for public funding-event alignment."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np

from alt_search_models import MarketData
from bitget_funding_mtf_data import FundingEvent, TimeframeMarket, funding_matrix


def test_funding_matrix_defers_a_settlement_to_the_next_closed_bar() -> None:
    """Given a settlement at a bar open, when it is aligned, then that bar cannot signal on it."""

    prices = np.full((3, 1), 100.0)
    market = MarketData(
        (datetime.fromtimestamp(0, UTC), datetime.fromtimestamp(0.1, UTC), datetime.fromtimestamp(0.2, UTC)),
        ("ALTUSDT",), prices, prices, prices, prices, prices,
    )
    timeframe = TimeframeMarket((100, 200, 300), market)
    event = FundingEvent.model_validate({"fundingTime": "100", "fundingRate": "0.01"})

    rates = funding_matrix(timeframe, {"ALTUSDT": (event,)})

    assert tuple(rates[:, 0]) == (0.0, 0.01, 0.0)
