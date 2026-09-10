from __future__ import annotations

from dataclasses import asdict
import json

from bitget_mtf_core import Candle, MarketSnapshot, analyze_interval, build_weights, gate_reasons


def test_build_weights_when_ranked_signals_have_two_extremes_then_book_is_delta_neutral() -> None:
    signals = {
        "AAAUSDT": 0.50,
        "BBBUSDT": 0.30,
        "CCCUSDT": 0.10,
        "DDDUSDT": -0.10,
        "EEEUSDT": -0.30,
        "FFFUSDT": -0.50,
    }

    weights = build_weights(signals, top_k=2)

    assert weights == {
        "AAAUSDT": 0.5,
        "BBBUSDT": 0.5,
        "EEEUSDT": -0.5,
        "FFFUSDT": -0.5,
    }
    assert sum(weights.values()) == 0.0


def test_gate_reasons_when_a_leg_is_illiquid_then_entry_is_rejected() -> None:
    markets = {
        "AAAUSDT": MarketSnapshot(quote_volume_usdt=20_000_000.0, spread_fraction=0.0001),
        "BBBUSDT": MarketSnapshot(quote_volume_usdt=4_999_999.0, spread_fraction=0.0001),
    }

    reasons = gate_reasons(
        regime_on=True,
        stress_distance=0.02,
        candidate_symbols=("AAAUSDT", "BBBUSDT"),
        markets=markets,
    )

    assert "liquidity" in reasons


def test_analyze_interval_when_current_regime_is_down_then_result_serializes_as_json() -> None:
    symbols = tuple(f"ALT{index:02d}USDT" for index in range(10))
    candles = {
        symbol: tuple(
            Candle(
                timestamp_ms=bar * 3_600_000,
                open_price=100.0 - bar * 0.2 + index * 0.01,
                high_price=100.1 - bar * 0.2 + index * 0.01,
                low_price=99.9 - bar * 0.2 + index * 0.01,
                close_price=100.0 - bar * 0.2 + index * 0.01,
                quote_volume_usdt=20_000_000.0,
            )
            for bar in range(100)
        )
        for index, symbol in enumerate(symbols)
    }
    markets = {
        symbol: MarketSnapshot(quote_volume_usdt=20_000_000.0, spread_fraction=0.0001)
        for symbol in symbols
    }

    result = analyze_interval(interval="1H", candles_by_symbol=candles, markets=markets)

    assert result.regime_on is False
    json.dumps({"results": [asdict(result)]}, allow_nan=False)
