import json
from datetime import UTC, datetime, timedelta
from urllib.parse import parse_qs, urlparse

from django.test import SimpleTestCase

from pulse.services.bitget_history import HistoricalRangeQuery, fetch_historical_candle_range
from pulse.services.bitget_market import BitgetContract
from pulse.services.market_timeframes import CandleGranularity


def _row(opened_at: datetime) -> list[str]:
    return [
        str(int(opened_at.timestamp() * 1000)),
        "100",
        "101",
        "99",
        "100.5",
        "10",
        "1000",
    ]


class HistoricalRangeTests(SimpleTestCase):
    def test_paginates_backwards_without_duplicate_or_future_candles(self):
        # Given: a public history endpoint that returns two reverse-time pages bounded by its 200-row maximum.
        starts_at = datetime(2026, 7, 1, tzinfo=UTC)
        ends_at = starts_at + timedelta(minutes=5)
        first_page_end = int(ends_at.timestamp() * 1000)
        second_page_end = int((starts_at + timedelta(minutes=3)).timestamp() * 1000)
        requested_end_times: list[int] = []

        def fetch_text(url: str) -> str:
            end_time = int(parse_qs(urlparse(url).query)["endTime"][0])
            requested_end_times.append(end_time)
            if end_time == first_page_end:
                rows = [_row(starts_at + timedelta(minutes=3)), _row(starts_at + timedelta(minutes=4))]
            elif end_time == second_page_end:
                rows = [_row(starts_at), _row(starts_at + timedelta(minutes=1)), _row(starts_at + timedelta(minutes=2))]
            else:
                rows = []
            return json.dumps({"code": "00000", "data": rows})

        # When: the range adapter requests the full time window.
        candles = fetch_historical_candle_range(
            HistoricalRangeQuery(
                contract=BitgetContract("TSLAUSDT", "TSLA", 20, 8),
                granularity=CandleGranularity.ONE_MINUTE,
                starts_at=starts_at,
                ends_at=ends_at,
            ),
            fetch_text,
        )

        # Then: all and only completed candles in the requested interval are returned once in chronological order.
        self.assertEqual([candle.opened_at for candle in candles], [starts_at + timedelta(minutes=index) for index in range(5)])
        self.assertEqual(requested_end_times, [first_page_end, second_page_end])
