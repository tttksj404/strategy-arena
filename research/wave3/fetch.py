"""Wave-3 data acquisition stage; the orchestrator owns when it is run."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import hashlib
from pathlib import Path
from collections.abc import Callable
from typing import Final

import pandas as pd  # noqa: PANDAS_OK
import requests

from research.scanner.scan_bitget import STOCK_BASE_COINS
from research.wave1.common import JsonValue, PipelineError
from research.wave1.common import ensure_output_dirs, integrity_report, load_frame, load_json, report_payload, save_frame, save_json, validate_symbol
from research.wave1.fetch_binance import BinanceFundingRequest, BinanceKlineRequest, exchange_symbols, fetch_exchange_info, fetch_funding, fetch_klines, fetch_quote_volumes, fetch_spot_exchange_info, quote_volumes
from research.wave1.fetch_bitget import BitgetCandleRequest, contract_symbols, fetch_candles, fetch_contracts, fetch_funding as fetch_bitget_funding


BASE_DIR: Final = Path(__file__).resolve().parent
WAVE3_CACHE_DIR: Final = BASE_DIR / "cache"
START_MS: Final = int(pd.Timestamp("2019-09-01T00:00:00Z").timestamp() * 1000)
END_MS: Final = int(pd.Timestamp("2026-07-15T00:00:00Z").timestamp() * 1000)


@dataclass(frozen=True, slots=True)
class FetchContext:
    session: requests.Session
    force: bool
    start_ms: int = START_MS
    end_ms: int = END_MS


def _cached(path: Path, context: FetchContext, loader: Callable[[], pd.DataFrame]) -> pd.DataFrame:
    if path.exists() and not context.force:
        return load_frame(path)
    frame = loader()
    if not isinstance(frame, pd.DataFrame):
        raise PipelineError(f"fetch loader did not return a frame for {path.name}")
    save_frame(path, frame)
    return frame


def _fetch_binance_symbol(symbol: str, context: FetchContext) -> dict[str, pd.DataFrame]:
    validate_symbol(symbol)
    paths = {
        "funding": WAVE3_CACHE_DIR / f"binance_funding_{symbol}.csv.gz",
        "perp": WAVE3_CACHE_DIR / f"binance_fapi_{symbol}_1d.csv.gz",
        "spot": WAVE3_CACHE_DIR / f"binance_spot_{symbol}_1d.csv.gz",
    }
    return {
        "funding": _cached(paths["funding"], context, lambda: fetch_funding(BinanceFundingRequest(symbol, context.start_ms, context.end_ms), context.session)),
        "perp": _cached(paths["perp"], context, lambda: fetch_klines(BinanceKlineRequest(symbol, "1d", context.start_ms, context.end_ms), context.session)),
        "spot": _cached(paths["spot"], context, lambda: fetch_klines(BinanceKlineRequest(symbol, "1d", context.start_ms, context.end_ms, "spot"), context.session)),
    }


def _fetch_bitget_stock(symbol: str, context: FetchContext) -> dict[str, pd.DataFrame]:
    validate_symbol(symbol)
    candle_path = WAVE3_CACHE_DIR / f"bitget_{symbol}_1D.csv.gz"
    funding_path = WAVE3_CACHE_DIR / f"bitget_funding_{symbol}.csv.gz"
    return {
        "candles": _cached(candle_path, context, lambda: fetch_candles(BitgetCandleRequest(symbol, "1D", context.start_ms, context.end_ms), context.session)),
        "funding": _cached(funding_path, context, lambda: fetch_bitget_funding(symbol, context.session)),
    }


def _record(path: Path) -> dict[str, JsonValue]:
    frame = load_frame(path)
    report = integrity_report(frame, timedelta(hours=8) if "funding" in path.name else timedelta(days=1))
    return {
        "file": path.name,
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "rows": len(frame),
        "start": pd.Timestamp(frame.index.min()).isoformat() if len(frame) else None,
        "end": pd.Timestamp(frame.index.max()).isoformat() if len(frame) else None,
        "integrity": report_payload(report),
        "source_integrity": report.valid and len(frame) > 0,
    }


def _record_snapshot(path: Path) -> dict[str, JsonValue]:
    return {
        "file": path.name,
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def verify_manifest(cache_dir: Path = WAVE3_CACHE_DIR) -> None:
    """Fail closed when any fetched snapshot or bar file changed or is invalid."""
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.exists():
        raise PipelineError("wave-3 fetch manifest is missing")
    manifest = load_json(manifest_path)
    if not isinstance(manifest, dict) or manifest.get("network_calls") is not True:
        raise PipelineError("wave-3 manifest is not a completed fetch artifact")
    records = [*(manifest.get("snapshots") or []), *(manifest.get("files") or [])]
    if not records:
        raise PipelineError("wave-3 manifest contains no source records")
    root = cache_dir.resolve()
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("file"), str):
            raise PipelineError("wave-3 manifest contains an invalid source record")
        name = record["file"]
        path = (cache_dir / name).resolve()
        if path.parent != root or path.name != name or not path.exists():
            raise PipelineError(f"wave-3 source is missing or escapes cache: {name}")
        if path.stat().st_size != record.get("bytes") or hashlib.sha256(path.read_bytes()).hexdigest() != record.get("sha256"):
            raise PipelineError(f"wave-3 source hash mismatch: {name}")
        if "rows" in record:
            frame = load_frame(path)
            if len(frame) != record["rows"] or record.get("source_integrity") is not True:
                raise PipelineError(f"wave-3 source integrity failed: {name}")


def run_fetch(force: bool = False, session: requests.Session | None = None) -> None:
    """Collect wave-3-only sources and write an auditable, no-lookahead manifest."""
    ensure_output_dirs()
    WAVE3_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    owns_session = session is None
    client = session or requests.Session()
    try:
        context = FetchContext(client, force)
        binance_info = fetch_exchange_info(client)
        spot_payload = fetch_spot_exchange_info(client)
        bitget_payload = fetch_contracts(client)
        volume_payload = fetch_quote_volumes(client)
        save_json(WAVE3_CACHE_DIR / "binance_exchange_info.json", binance_info)
        save_json(WAVE3_CACHE_DIR / "bitget_contracts.json", bitget_payload)
        save_json(WAVE3_CACHE_DIR / "spot_exchange_info.json", spot_payload)
        save_json(WAVE3_CACHE_DIR / "binance_quote_volume_snapshot.json", volume_payload)
        futures = exchange_symbols(binance_info)
        spot = exchange_symbols(spot_payload)
        bitget = contract_symbols(bitget_payload)
        crypto_symbols = tuple(sorted(futures & spot & bitget))
        for symbol in crypto_symbols:
            _fetch_binance_symbol(symbol, context)
        stock_symbols = tuple(sorted(symbol for symbol in bitget if symbol.removesuffix("USDT") in STOCK_BASE_COINS))
        for symbol in stock_symbols:
            _fetch_bitget_stock(symbol, context)
        files = sorted(WAVE3_CACHE_DIR.glob("*.csv.gz"))
        save_json(
            WAVE3_CACHE_DIR / "manifest.json",
            {
                "network_calls": True,
                "frozen_end": "2026-07-14",
                "crypto_symbols": list(crypto_symbols),
                "stock_symbols": list(stock_symbols),
                "snapshot_quote_volume_symbols": len(quote_volumes(volume_payload)),
                "snapshots": [_record_snapshot(WAVE3_CACHE_DIR / name) for name in (
                    "binance_exchange_info.json",
                    "spot_exchange_info.json",
                    "bitget_contracts.json",
                    "binance_quote_volume_snapshot.json",
                )],
                "files": [_record(path) for path in files],
            },
        )
        verify_manifest()
        print(f"fetch: cached {len(crypto_symbols)} crypto and {len(stock_symbols)} stock-token symbols")
    finally:
        if owns_session:
            client.close()


__all__ = ["END_MS", "START_MS", "WAVE3_CACHE_DIR", "run_fetch", "verify_manifest"]
