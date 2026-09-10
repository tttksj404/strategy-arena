#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx2[http2,brotli,zstd]", "pydantic>=2"]
# ///

"""Snapshot a reproducible, liquid non-RWA Bitget altcoin futures universe."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, TypeVar

import httpx2
from pydantic import BaseModel, ConfigDict, ValidationError

API_ROOT: Final = "https://api.bitget.com"
TICKER_ROUTE: Final = "/api/v3/market/tickers"
CONTRACT_ROUTE: Final = "/api/v2/mix/market/contracts"
MIN_TURNOVER_USDT: Final = 10_000_000.0
MAX_SPREAD_FRACTION: Final = 0.001


class UniverseError(RuntimeError):
    """A public universe snapshot cannot be trusted."""


class Ticker(BaseModel):
    model_config = ConfigDict(frozen=True)
    category: str
    symbol: str
    ts: int
    bid1Price: float
    ask1Price: float
    turnover24h: float


class Contract(BaseModel):
    model_config = ConfigDict(frozen=True)
    symbol: str
    baseCoin: str
    quoteCoin: str
    symbolType: str
    symbolStatus: str
    isRwa: str
    fundInterval: str


class TickerEnvelope(BaseModel):
    model_config = ConfigDict(frozen=True)
    code: str
    msg: str
    data: tuple[Ticker, ...]


class ContractEnvelope(BaseModel):
    model_config = ConfigDict(frozen=True)
    code: str
    msg: str
    data: tuple[Contract, ...]


Envelope = TypeVar("Envelope", TickerEnvelope, ContractEnvelope)


@dataclass(frozen=True, slots=True)
class UniverseEntry:
    symbol: str
    base_coin: str
    ticker_timestamp_ms: int
    turnover_24h_usdt: float
    spread_fraction: float
    funding_interval_hours: int


def select_liquid_alt_universe(tickers: tuple[Ticker, ...], contracts: dict[str, Contract]) -> tuple[UniverseEntry, ...]:
    """Apply only live-tradability filters; historical survivor bias remains explicit."""
    selected: list[UniverseEntry] = []
    for ticker in tickers:
        contract = contracts.get(ticker.symbol)
        if contract is None or ticker.symbol == "BTCUSDT" or not ticker.symbol.isalnum() or not ticker.symbol.endswith("USDT"):
            continue
        if contract.symbolType != "perpetual" or contract.symbolStatus != "normal" or contract.isRwa != "NO" or contract.quoteCoin != "USDT":
            continue
        if ticker.turnover24h < MIN_TURNOVER_USDT or ticker.bid1Price <= 0.0 or ticker.ask1Price <= ticker.bid1Price:
            continue
        spread = (ticker.ask1Price - ticker.bid1Price) / ticker.ask1Price
        if spread > MAX_SPREAD_FRACTION:
            continue
        try:
            funding_interval = int(contract.fundInterval)
        except ValueError as error:
            raise UniverseError(f"{ticker.symbol}: invalid funding interval") from error
        selected.append(UniverseEntry(ticker.symbol, contract.baseCoin, ticker.ts, ticker.turnover24h, spread, funding_interval))
    return tuple(sorted(selected, key=lambda item: (-item.turnover_24h_usdt, item.symbol)))


def _fetch(client: httpx2.Client, route: str, params: dict[str, str], model: type[Envelope]) -> Envelope:
    try:
        response = client.get(route, params=params)
        response.raise_for_status()
        envelope = model.model_validate_json(response.text)
    except (httpx2.HTTPError, ValidationError) as error:
        raise UniverseError(f"{route}: invalid public response") from error
    if envelope.code != "00000":
        raise UniverseError(f"{route}: {envelope.msg}")
    return envelope


def snapshot() -> tuple[UniverseEntry, ...]:
    with httpx2.Client(base_url=API_ROOT, timeout=30.0, http2=True) as client:
        tickers = _fetch(client, TICKER_ROUTE, {"category": "USDT-FUTURES"}, TickerEnvelope)
        contracts = _fetch(client, CONTRACT_ROUTE, {"productType": "USDT-FUTURES"}, ContractEnvelope)
    return select_liquid_alt_universe(tickers.data, {contract.symbol: contract for contract in contracts.data})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("bitget_liquid_alt_universe_20260825.json"))
    arguments = parser.parse_args()
    try:
        rows = snapshot()
    except UniverseError as error:
        print(f"UNIVERSE_ERROR={error}")
        return 1
    payload = {
        "mode": "public-api-paper-only-no-live-orders",
        "observed_at": datetime.now(UTC).isoformat(),
        "criteria": {
            "minimum_turnover_24h_usdt": MIN_TURNOVER_USDT,
            "maximum_spread_fraction": MAX_SPREAD_FRACTION,
            "exclude_btc": True,
            "perpetual_only": True,
            "normal_status_only": True,
            "non_rwa_only": True,
            "survivorship_bias_warning": True,
        },
        "entries": [asdict(row) for row in rows],
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    partial = arguments.output.with_suffix(arguments.output.suffix + ".partial")
    partial.write_text(json.dumps(payload, allow_nan=False, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(partial, arguments.output)
    print(f"UNIVERSE={arguments.output}")
    print(f"LIQUID_ALT_SYMBOLS={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
