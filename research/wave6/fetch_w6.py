# New wave-6 data fetchers: Binance 1h klines (BTC/ETH/SOL), Yahoo 1h SPY, and Bitget 1h
# SPY/QQQ tokens (reuse wave-1 cache + increment). Funding and Bitget contract metadata are read
# from existing wave-1/wave-3 caches, never re-derived here.

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
import sys
from typing import Final

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

import pandas as pd  # noqa: PANDAS_OK
import requests

from research.wave1.common import JsonValue, PipelineError, integrity_report, load_frame, load_json, report_payload, request_json, save_frame, save_json
from research.wave1.fetch_binance import BinanceKlineRequest, cached_frame, fetch_klines
from research.wave1.fetch_bitget import BitgetCandleRequest, fetch_candles, fetch_contracts
from research.wave6.engine_w6 import CACHE_DIR, REPORT_DIR, RESULTS_DIR, WAVE1_CACHE_DIR, WAVE3_CACHE_DIR, eligible_listing_symbols


YAHOO_BASE: Final = "https://query1.finance.yahoo.com"
BINANCE_START_MS: Final = int(pd.Timestamp("2019-09-01T00:00:00Z").timestamp() * 1000)
BITGET_FALLBACK_START_MS: Final = int(pd.Timestamp("2023-01-01T00:00:00Z").timestamp() * 1000)
CRYPTO_SYMBOLS: Final = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
BITGET_STOCK_SYMBOLS: Final = ("SPYUSDT", "QQQUSDT")


def ensure_output_dirs() -> None:
    for directory in (CACHE_DIR, RESULTS_DIR, REPORT_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def _now_ms() -> int:
    return int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)


def fetch_binance_1h(symbol: str, session: requests.Session, force: bool) -> pd.DataFrame:
    path = CACHE_DIR / f"binance_fapi_{symbol}_1h.csv.gz"
    end_ms = _now_ms()
    return cached_frame(path, force, lambda: fetch_klines(BinanceKlineRequest(symbol, "1h", BINANCE_START_MS, end_ms), session))


def _parse_yahoo_chart(payload: JsonValue) -> pd.DataFrame:
    if not isinstance(payload, dict):
        raise PipelineError("Yahoo response must be an object")
    chart = payload.get("chart")
    if not isinstance(chart, dict) or not isinstance(chart.get("result"), list) or not chart["result"]:
        raise PipelineError("Yahoo response has no result")
    result = chart["result"][0]
    if not isinstance(result, dict):
        raise PipelineError("Yahoo result must be an object")
    timestamps = result.get("timestamp")
    indicators = result.get("indicators")
    if not isinstance(timestamps, list) or not isinstance(indicators, dict):
        raise PipelineError("Yahoo result is missing timestamps or indicators")
    quotes = indicators.get("quote")
    if not isinstance(quotes, list) or not quotes or not isinstance(quotes[0], dict):
        raise PipelineError("Yahoo result is missing quote data")
    quote = quotes[0]
    frame = pd.DataFrame({"timestamp": timestamps, **quote})
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="s", utc=True)
    return frame.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()


def fetch_yahoo_spy_1h(session: requests.Session) -> pd.DataFrame:
    # request_json (via common.py) always attaches a User-Agent header; the chart endpoint
    # otherwise returns HTTP 429/999 to header-less clients.
    payload = request_json(session, f"{YAHOO_BASE}/v8/finance/chart/SPY", {"range": "730d", "interval": "1h"})
    return _parse_yahoo_chart(payload)


def fetch_bitget_1h_incremental(symbol: str, session: requests.Session, force: bool) -> pd.DataFrame:
    out_path = CACHE_DIR / f"bitget_{symbol}_1H.csv.gz"
    if out_path.exists() and not force:
        return load_frame(out_path)
    base_path = WAVE1_CACHE_DIR / f"bitget_{symbol}_1H.csv.gz"
    end_ms = _now_ms()
    if base_path.exists():
        base = load_frame(base_path)
        last_ms = int(pd.Timestamp(base.index.max()).timestamp() * 1000) + 1 if not base.empty else BITGET_FALLBACK_START_MS
        increment = fetch_candles(BitgetCandleRequest(symbol, "1H", last_ms, end_ms), session) if last_ms < end_ms else base.iloc[0:0]
        combined = pd.concat([base, increment]).sort_index()
        combined = combined[~combined.index.duplicated(keep="last")]
    else:
        combined = fetch_candles(BitgetCandleRequest(symbol, "1H", BITGET_FALLBACK_START_MS, end_ms), session)
    save_frame(out_path, combined)
    return combined


def load_bitget_contracts(session: requests.Session, force: bool) -> JsonValue:
    wave3_path = WAVE3_CACHE_DIR / "bitget_contracts.json"
    out_path = CACHE_DIR / "bitget_contracts.json"
    if out_path.exists() and not force:
        return load_json(out_path)
    payload = load_json(wave3_path) if wave3_path.exists() else fetch_contracts(session)
    save_json(out_path, payload)
    return payload


def stage_fetch(force: bool = False) -> dict[str, JsonValue]:
    ensure_output_dirs()
    manifest: dict[str, JsonValue] = {"generated_at": pd.Timestamp.now(tz="UTC").isoformat()}
    with requests.Session() as session:
        for symbol in CRYPTO_SYMBOLS:
            frame = fetch_binance_1h(symbol, session, force)
            report = integrity_report(frame, timedelta(hours=1))
            manifest[f"binance_fapi_{symbol}_1h"] = {"rows": len(frame), **report_payload(report)}
            funding_path = WAVE1_CACHE_DIR / f"binance_funding_{symbol}.csv.gz"
            funding_rows = len(load_frame(funding_path)) if funding_path.exists() else 0
            manifest[f"binance_funding_{symbol}_reused"] = {"exists": funding_path.exists(), "rows": funding_rows}
            print(f"fetch: binance 1h {symbol} rows={len(frame)}")
        yahoo_path = CACHE_DIR / "yahoo_SPY_1h.csv.gz"
        yahoo = cached_frame(yahoo_path, force, lambda: fetch_yahoo_spy_1h(session))
        manifest["yahoo_SPY_1h"] = {"rows": len(yahoo), **report_payload(integrity_report(yahoo, timedelta(hours=1)))}
        print(f"fetch: yahoo SPY 1h rows={len(yahoo)}")
        for symbol in BITGET_STOCK_SYMBOLS:
            frame = fetch_bitget_1h_incremental(symbol, session, force)
            manifest[f"bitget_{symbol}_1H"] = {"rows": len(frame), **report_payload(integrity_report(frame, timedelta(hours=1)))}
            print(f"fetch: bitget 1h {symbol} rows={len(frame)}")
        contracts = load_bitget_contracts(session, force)
        eligible = eligible_listing_symbols(contracts)
        manifest["bitget_contracts"] = {
            "total": len(contracts) if isinstance(contracts, list) else 0,
            "launch_time_eligible": len(eligible),
        }
        print(f"fetch: bitget contracts total={manifest['bitget_contracts']['total']} launch_time_eligible={len(eligible)}")
    save_json(CACHE_DIR / "manifest_w6.json", manifest)
    return manifest


__all__ = [
    "BITGET_STOCK_SYMBOLS",
    "CRYPTO_SYMBOLS",
    "ensure_output_dirs",
    "fetch_binance_1h",
    "fetch_bitget_1h_incremental",
    "fetch_yahoo_spy_1h",
    "load_bitget_contracts",
    "stage_fetch",
]
