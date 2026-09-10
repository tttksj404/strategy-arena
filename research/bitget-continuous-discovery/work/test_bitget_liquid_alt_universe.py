from __future__ import annotations

from bitget_liquid_alt_universe import Contract, Ticker, select_liquid_alt_universe


def _ticker(symbol: str, turnover: float = 20_000_000.0, bid: float = 100.0, ask: float = 100.05) -> Ticker:
    return Ticker(category="USDT-FUTURES", symbol=symbol, ts=1, bid1Price=bid, ask1Price=ask, turnover24h=turnover)


def _contract(symbol: str, *, rwa: str = "NO", status: str = "normal", contract_type: str = "perpetual") -> Contract:
    return Contract(symbol=symbol, baseCoin=symbol.removesuffix("USDT"), quoteCoin="USDT", symbolType=contract_type, symbolStatus=status, isRwa=rwa, fundInterval="8")


def test_select_liquid_alt_universe_filters_nontradable_noncrypto_and_wide_spreads() -> None:
    rows = select_liquid_alt_universe(
        (
            _ticker("BTCUSDT", turnover=100_000_000.0),
            _ticker("ETHUSDT", turnover=40_000_000.0),
            _ticker("SOLUSDT", turnover=30_000_000.0),
            _ticker("RWAUSDT", turnover=90_000_000.0),
            _ticker("WIDEUSDT", turnover=80_000_000.0, bid=90.0, ask=100.0),
            _ticker("LOWUSDT", turnover=9_999_999.0),
            _ticker("OFFUSDT", turnover=70_000_000.0),
        ),
        {
            "BTCUSDT": _contract("BTCUSDT"),
            "ETHUSDT": _contract("ETHUSDT"),
            "SOLUSDT": _contract("SOLUSDT"),
            "RWAUSDT": _contract("RWAUSDT", rwa="YES"),
            "WIDEUSDT": _contract("WIDEUSDT"),
            "LOWUSDT": _contract("LOWUSDT"),
            "OFFUSDT": _contract("OFFUSDT", status="suspend"),
        },
    )

    assert tuple(row.symbol for row in rows) == ("ETHUSDT", "SOLUSDT")
    assert all(row.spread_fraction <= 0.001 for row in rows)
