from __future__ import annotations

from bitget_mtf_core import Candle, MarketSnapshot, analyze_interval


def test_long_interval_with_fewer_than_30_oos_returns_is_rejected() -> None:
    day_ms = 86_400_000
    candles: dict[str, list[Candle]] = {}
    markets: dict[str, MarketSnapshot] = {}
    for symbol_index in range(10):
        symbol = f"ALT{symbol_index:02d}USDT"
        start = 100.0 + symbol_index
        candles[symbol] = [
            Candle(
                timestamp_ms=bar * 3 * day_ms,
                open_price=start * (0.999**bar),
                high_price=start * (0.999**bar) * 1.001,
                low_price=start * (0.999**bar) * 0.999,
                close_price=start * (0.999**bar),
                quote_volume_usdt=20_000_000.0,
            )
            for bar in range(70)
        ]
        markets[symbol] = MarketSnapshot(quote_volume_usdt=20_000_000.0, spread_fraction=0.0001)

    result = analyze_interval(interval="3D", candles_by_symbol=candles, markets=markets)

    assert "oos_observations_short" in result.gate_reasons
