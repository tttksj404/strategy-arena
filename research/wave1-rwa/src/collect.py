"""Collect Bitget USDT-M RWA contracts, candles, funding and manifest."""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import pandas as pd
import requests

ROOT: Final = Path(__file__).resolve().parents[1]
BASE_URL: Final = "https://api.bitget.com"
PAUSE: Final = 0.12


def progress(phase: str, summary: str) -> None:
    """Append one compact progress record."""
    out = ROOT / "out"
    out.mkdir(exist_ok=True)
    with (out / "PROGRESS.md").open("a", encoding="utf-8") as f:
        f.write(f"{datetime.now(UTC).isoformat()} | {phase} | {summary}\n")


def api_get(path: str, params: dict[str, str]) -> list[dict[str, str]]:
    """Call a public endpoint with bounded retries and rate limiting."""
    last: Exception | None = None
    for attempt in range(3):
        try:
            response = requests.get(BASE_URL + path, params=params, timeout=30)
            response.raise_for_status()
            payload = response.json()
            if payload.get("code") != "00000":
                raise RuntimeError(f"Bitget {path}: {payload.get('code')} {payload.get('msg')}")
            time.sleep(PAUSE)
            data = payload.get("data", [])
            return data if isinstance(data, list) else []
        except (requests.RequestException, ValueError, RuntimeError) as exc:
            last = exc
            time.sleep(2**attempt)
    raise RuntimeError(f"Bitget request failed after retries: {path}") from last


def contracts() -> list[dict[str, str]]:
    """Return all RWA contracts with stable metadata."""
    rows = api_get("/api/v2/mix/market/contracts", {"productType": "usdt-futures"})
    return [
        {
            "symbol": str(row["symbol"]),
            "baseCoin": str(row.get("baseCoin", "")),
            "openTime": str(row.get("openTime", "")),
            "maxLever": str(row.get("maxLever", row.get("maxLeverage", ""))),
            "makerFeeRate": str(row.get("makerFeeRate", "")),
            "takerFeeRate": str(row.get("takerFeeRate", "0.0006")),
            "isRwa": str(row.get("isRwa", "")),
        }
        for row in rows
        if str(row.get("isRwa", "")).upper() == "YES"
    ]


def tickers() -> dict[str, dict[str, float]]:
    """Return measured spread and quote volume for every symbol."""
    rows = api_get("/api/v2/mix/market/tickers", {"productType": "usdt-futures"})
    result: dict[str, dict[str, float]] = {}
    for row in rows:
        try:
            bid, ask = float(row["bidPr"]), float(row["askPr"])
            mid = (bid + ask) / 2
            result[str(row["symbol"])] = {
                "half_spread_bp": max(0.0, (ask - bid) / 2 / mid * 10_000),
                "usdtVolume": float(row.get("usdtVolume", row.get("quoteVolume", 0)) or 0),
            }
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            continue
    return result


def history(symbol: str, granularity: str, end_time: int | None = None) -> pd.DataFrame:
    """Page history candles until the exchange returns no older rows."""
    rows: list[list[str]] = []
    cursor = end_time or int(datetime.now(UTC).timestamp() * 1000)
    while True:
        page = api_get(
            "/api/v2/mix/market/history-candles",
            {"symbol": symbol, "productType": "usdt-futures", "granularity": granularity,
             "limit": "200", "endTime": str(cursor)},
        )
        if not page:
            break
        rows.extend([list(map(str, row)) for row in page])
        oldest = min(int(row[0]) for row in page)
        if len(page) < 200 or oldest >= cursor:
            break
        cursor = oldest - 1
    if not rows:
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "vol_base", "vol_quote"])
    frame = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "vol_base", "vol_quote"])
    frame = frame.drop_duplicates("ts").sort_values("ts")
    frame["ts"] = pd.to_datetime(frame["ts"].astype("int64"), unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "vol_base", "vol_quote"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame.dropna(subset=["ts", "open", "high", "low", "close"])


def funding(symbol: str) -> pd.DataFrame:
    """Page the funding-rate history."""
    rows: list[list[str]] = []
    page = 1
    while True:
        batch = api_get(
            "/api/v2/mix/market/history-fund-rate",
            {"symbol": symbol, "productType": "usdt-futures", "pageSize": "100", "pageNo": str(page)},
        )
        if not batch:
            break
        for row in batch:
            if isinstance(row, dict):
                rows.append([str(row.get("fundingTime", "")), str(row.get("fundingRate", "0"))])
            else:
                rows.append(list(map(str, row)))
        if len(batch) < 100:
            break
        page += 1
    if not rows:
        return pd.DataFrame(columns=["ts", "rate"])
    normalized: list[tuple[int, float]] = []
    for row in rows:
        try:
            ts = int(row[0])
            rate = float(row[1])
        except (IndexError, TypeError, ValueError):
            try:
                ts, rate = int(row[-2]), float(row[-1])
            except (IndexError, TypeError, ValueError):
                continue
        normalized.append((ts, rate))
    frame = pd.DataFrame(normalized, columns=["ts", "rate"]).drop_duplicates("ts")
    frame["ts"] = pd.to_datetime(frame["ts"], unit="ms", utc=True)
    return frame.sort_values("ts")


def quality(frame: pd.DataFrame, granularity: str) -> tuple[str, float, int]:
    """Estimate missing bars against the observed UTC span."""
    if frame.empty:
        return "missing", 1.0, 0
    step = pd.Timedelta(hours=1 if granularity == "1H" else 24)
    expected = max(1, int((frame["ts"].iloc[-1] - frame["ts"].iloc[0]) / step) + 1)
    missing = max(0, expected - len(frame)) / expected
    return ("quality_flag" if missing > 0.05 else "ok"), float(missing), expected


def main() -> int:
    """Collect all RWA market data and write the manifest."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="refetch files")
    args = parser.parse_args()
    for directory in [ROOT / "data/candles_1h", ROOT / "data/candles_1d", ROOT / "data/funding", ROOT / "out"]:
        directory.mkdir(parents=True, exist_ok=True)
    rwa = contracts()
    spread = tickers()
    (ROOT / "data/contracts.json").write_text(json.dumps(rwa, indent=2), encoding="utf-8")
    manifest: list[dict[str, object]] = []
    progress("collect", f"contracts={len(rwa)}")
    for index, contract in enumerate(rwa, 1):
        symbol = contract["symbol"]
        paths = {"1H": ROOT / "data/candles_1h" / f"{symbol}.parquet", "1D": ROOT / "data/candles_1d" / f"{symbol}.parquet"}
        frames: dict[str, pd.DataFrame] = {}
        for granularity, path in paths.items():
            if path.exists() and not args.force:
                frames[granularity] = pd.read_parquet(path)
            else:
                frames[granularity] = history(symbol, granularity)
                frames[granularity].to_parquet(path, index=False)
        fund_path = ROOT / "data/funding" / f"{symbol}.parquet"
        fund = pd.read_parquet(fund_path) if fund_path.exists() and not args.force else funding(symbol)
        fund.to_parquet(fund_path, index=False)
        flag, missing, expected = quality(frames["1H"], "1H")
        info = spread.get(symbol, {"half_spread_bp": 0.0, "usdtVolume": 0.0})
        days = (frames["1H"]["ts"].iloc[-1] - frames["1H"]["ts"].iloc[0]).total_seconds() / 86_400 if not frames["1H"].empty else 0
        tier = "A" if days >= 120 else "B" if days >= 60 else "C"
        manifest.append({**contract, "start": frames["1H"]["ts"].min().isoformat() if not frames["1H"].empty else None,
                         "end": frames["1H"]["ts"].max().isoformat() if not frames["1H"].empty else None,
                         "days": round(days, 2), "missing_rate": round(missing, 6), "expected_1h": expected,
                         "quality_flag": flag, "tier": tier, "half_spread_bp": info["half_spread_bp"],
                         "usdtVolume": info["usdtVolume"], "avg_daily_quote": float(frames["1D"]["vol_quote"].mean()) if not frames["1D"].empty else 0.0,
                         "funding_rows": len(fund)})
        if index == 1 or index % 10 == 0 or index == len(rwa):
            progress("collect", f"{index}/{len(rwa)} symbols")
    (ROOT / "out/data_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    progress("collect", f"manifest={len(manifest)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
