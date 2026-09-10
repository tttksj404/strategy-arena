from __future__ import annotations

from dataclasses import asdict
import json

from bitget_mtf_core import Candle, MarketSnapshot, analyze_interval


def _descending_market_data() -> tuple[dict[str, list[Candle]], dict[str, MarketSnapshot]]:
    candles: dict[str, list[Candle]] = {}
    markets: dict[str, MarketSnapshot] = {}
    for symbol_index in range(10):
        symbol = f"ALT{symbol_index:02d}USDT"
        initial_close = 100.0 + symbol_index
        candles[symbol] = [
            Candle(
                timestamp_ms=bar * 3_600_000,
                open_price=initial_close * (0.999**bar),
                high_price=initial_close * (0.999**bar) * 1.001,
                low_price=initial_close * (0.999**bar) * 0.999,
                close_price=initial_close * (0.999**bar),
                quote_volume_usdt=20_000_000.0,
            )
            for bar in range(100)
        ]
        markets[symbol] = MarketSnapshot(quote_volume_usdt=20_000_000.0, spread_fraction=0.0001)
    return candles, markets


def test_regime_off_result_is_strict_json_serializable() -> None:
    candles, markets = _descending_market_data()

    result = analyze_interval(interval="1H", candles_by_symbol=candles, markets=markets)

    assert type(result.regime_on) is bool
    json.dumps(asdict(result), allow_nan=False)
