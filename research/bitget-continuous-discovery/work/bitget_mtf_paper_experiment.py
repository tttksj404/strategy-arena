#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "httpx2[http2,brotli,zstd]",
#     "numpy",
#     "pydantic>=2",
# ]
# ///

# ─── How to run ───
# 1. Install uv (if not installed):
#      powershell -NoProfile -Command "irm https://astral.sh/uv/install.ps1 | iex"
# 2. Run directly (no venv or key required):
#      uv run work/bitget_mtf_paper_experiment.py
# ──────────────────

"""Read-only Bitget multi-timeframe paper experiment for a fixed altcoin strategy."""

from __future__ import annotations

import json
import socket
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import httpx2
from pydantic import BaseModel, ConfigDict

from bitget_mtf_core import Candle, IntervalResult, MarketSnapshot, analyze_interval


API_ROOT: Final = "https://api.bitget.com"
PRODUCT_TYPE: Final = "USDT-FUTURES"
LIMIT: Final = 1000
PACING_SECONDS: Final = 0.15
RATE_LIMIT_RETRIES: Final = 3
RATE_LIMIT_BACKOFF_SECONDS: Final = 2.0
MAX_QUERY_DAYS: Final = 90
INTERVALS: Final = (
    "1m", "3m", "5m", "15m", "30m", "1H", "2H", "4H", "6H", "12H", "1D", "3D", "1W", "1M",
)
INTERVAL_MS: Final = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1H": 3_600_000,
    "2H": 7_200_000,
    "4H": 14_400_000,
    "6H": 21_600_000,
    "12H": 43_200_000,
    "1D": 86_400_000,
    "3D": 259_200_000,
    "1W": 604_800_000,
    "1M": 2_592_000_000,
}
HISTORY_DAYS_BY_INTERVAL: Final = {
    "1m": 30,
    "3m": 30,
    "5m": 30,
    "15m": 52,
    "30m": 62,
    "1H": 83,
    "2H": 120,
    "4H": 240,
    "6H": 360,
    "12H": 365,
    "1D": 730,
    "3D": 365,
    "1W": 900,
    "1M": 3600,
}
# The existing paper runner's fixed universe, excluding BTC to keep the experiment altcoin-focused.
ALT_UNIVERSE: Final = (
    "ETHUSDT", "BNBUSDT", "ADAUSDT", "SOLUSDT", "DOGEUSDT", "XRPUSDT", "DOTUSDT", "LTCUSDT",
    "LINKUSDT", "AVAXUSDT", "UNIUSDT", "ATOMUSDT", "NEARUSDT",
)


@dataclass(frozen=True, slots=True)
class BitgetRequestError(Exception):
    """A public Bitget request that cannot support a trustworthy paper verdict."""

    route: str
    detail: str

    def __str__(self) -> str:
        return f"Bitget {self.route}: {self.detail}"


class CandleEnvelope(BaseModel):
    """Validated v2 candle payload at the external API boundary."""

    model_config = ConfigDict(frozen=True)
    code: str
    msg: str
    data: tuple[tuple[int, float, float, float, float, float, float], ...]


class TickerRow(BaseModel):
    """The public ticker fields used for a current liquidity gate."""

    model_config = ConfigDict(frozen=True)
    symbol: str
    quoteVolume: float | None = None
    usdtVolume: float | None = None
    bidPr: float | None = None
    askPr: float | None = None
    lastPr: float | None = None


class TickerEnvelope(BaseModel):
    """Validated v2 all-tickers payload at the external API boundary."""

    model_config = ConfigDict(frozen=True)
    code: str
    msg: str
    data: tuple[TickerRow, ...]


def create_client() -> httpx2.Client:
    """Create the documented HTTP/2 client with bounded retries and timeouts."""

    limits = httpx2.Limits(max_connections=200, max_keepalive_connections=40, keepalive_expiry=30.0)
    timeout = httpx2.Timeout(connect=5.0, read=30.0, write=10.0, pool=10.0)
    transport = httpx2.HTTPTransport(
        http2=True,
        retries=3,
        limits=limits,
        socket_options=[(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)],
    )
    return httpx2.Client(base_url=API_ROOT, transport=transport, timeout=timeout, follow_redirects=True)


def _get_text(client: httpx2.Client, route: str, params: dict[str, str]) -> str:
    for attempt in range(RATE_LIMIT_RETRIES + 1):
        try:
            response = client.get(route, params=params)
            response.raise_for_status()
        except httpx2.HTTPStatusError as error:
            if error.response.status_code == 429 and attempt < RATE_LIMIT_RETRIES:
                time.sleep(RATE_LIMIT_BACKOFF_SECONDS * 2**attempt)
                continue
            raise BitgetRequestError(route=route, detail=str(error)) from error
        except httpx2.HTTPError as error:
            raise BitgetRequestError(route=route, detail=str(error)) from error
        time.sleep(PACING_SECONDS)
        return response.text
    raise AssertionError("bounded rate-limit loop must return or raise")


def fetch_markets(client: httpx2.Client) -> dict[str, MarketSnapshot]:
    """Fetch one public ticker snapshot and convert it into entry-gate values."""

    payload = TickerEnvelope.model_validate_json(_get_text(client, "/api/v2/mix/market/tickers", {"productType": "usdt-futures"}))
    if payload.code != "00000":
        raise BitgetRequestError(route="tickers", detail=payload.msg)
    markets: dict[str, MarketSnapshot] = {}
    for row in payload.data:
        volume = row.quoteVolume if row.quoteVolume is not None else row.usdtVolume
        spread = (row.askPr - row.bidPr) / row.lastPr if row.bidPr and row.askPr and row.lastPr else float("inf")
        markets[row.symbol] = MarketSnapshot(quote_volume_usdt=volume or 0.0, spread_fraction=spread)
    return markets


def fetch_candles(client: httpx2.Client, symbol: str, interval: str, now_ms: int) -> tuple[Candle, ...]:
    """Page backwards through the public retention window and exclude the open candle."""

    lower_bound = now_ms - HISTORY_DAYS_BY_INTERVAL[interval] * 86_400_000
    upper_bound = now_ms
    rows: dict[int, Candle] = {}
    while upper_bound > lower_bound:
        request_start = max(lower_bound, upper_bound - MAX_QUERY_DAYS * 86_400_000)
        params = {
            "symbol": symbol,
            "productType": "usdt-futures",
            "granularity": interval,
            "startTime": str(request_start),
            "endTime": str(upper_bound),
            "limit": str(LIMIT),
        }
        payload = CandleEnvelope.model_validate_json(_get_text(client, "/api/v2/mix/market/candles", params))
        if payload.code != "00000":
            raise BitgetRequestError(route=f"candles/{interval}/{symbol}", detail=payload.msg)
        if not payload.data:
            upper_bound = request_start - INTERVAL_MS[interval]
            continue
        for row in payload.data:
            rows[row[0]] = Candle(
                timestamp_ms=row[0], open_price=row[1], high_price=row[2], low_price=row[3], close_price=row[4], quote_volume_usdt=row[6]
            )
        oldest = min(row[0] for row in payload.data)
        if len(payload.data) >= LIMIT and oldest > request_start:
            upper_bound = oldest - INTERVAL_MS[interval]
        elif request_start > lower_bound:
            upper_bound = request_start - INTERVAL_MS[interval]
        else:
            break
    closed = tuple(sorted(rows.values(), key=lambda candle: candle.timestamp_ms))
    return closed[:-1]


def run() -> tuple[IntervalResult, ...]:
    """Run the predeclared full-timeframe experiment without any authenticated endpoint."""

    now_ms = int(time.time() * 1000)
    with create_client() as client:
        markets = fetch_markets(client)
        results: list[IntervalResult] = []
        for interval in INTERVALS:
            print(f"[Bitget public] {interval}: collecting {len(ALT_UNIVERSE)} alt contracts", flush=True)
            candle_map: dict[str, tuple[Candle, ...]] = {}
            for count, symbol in enumerate(ALT_UNIVERSE, start=1):
                candle_map[symbol] = fetch_candles(client, symbol, interval, now_ms)
                if count % 4 == 0 or count == len(ALT_UNIVERSE):
                    print(f"  {interval}: {count}/{len(ALT_UNIVERSE)}", flush=True)
            result = analyze_interval(interval=interval, candles_by_symbol=candle_map, markets=markets)
            results.append(result)
            label = "PASS" if not result.gate_reasons else ",".join(result.gate_reasons)
            print(f"  {interval}: {label}; lev={result.leverage:.2f}; test={result.test.net_return:+.2%}; mdd={result.test.max_drawdown:.2%}", flush=True)
    return tuple(results)


def save(results: tuple[IntervalResult, ...]) -> Path:
    """Persist only the derived evidence summary, never account data or credentials."""

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = Path(__file__).with_name(f"bitget_mtf_paper_{stamp}.json")
    payload = {"observed_at": datetime.now(UTC).isoformat(), "mode": "public-api-paper-only", "results": [asdict(result) for result in results]}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def main() -> int:
    """Execute the experiment and expose a compact evidence location."""

    try:
        results = run()
    except BitgetRequestError as error:
        print(str(error))
        return 1
    evidence = save(results)
    passes = [result.interval for result in results if not result.gate_reasons]
    print(f"EVIDENCE={evidence}")
    print(f"PASS_INTERVALS={','.join(passes) if passes else 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
