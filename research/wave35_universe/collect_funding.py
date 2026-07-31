#!/usr/bin/env python3
# Wave-35 funding collection for the widened universe.
#
# Funding cannot be skipped or approximated here. At leverage it is charged on NOTIONAL, so a 3x
# position pays 3x the funding, and wave30's engine treats it as a first-order cost for exactly
# that reason. Running the widened universe with funding set to zero would hand every leveraged
# short a subsidy it does not get.
#
# Bitget's history-fund-rate paginates with pageNo (pageSize caps at 100), unlike the candle
# endpoint which pages by endTime. Resumable per symbol, same as collect_bitget.py.

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import sys
import time
from typing import Any, Final

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

import pandas as pd  # noqa: PANDAS_OK
import requests

BITGET: Final = "https://api.bitget.com"
PRODUCT_TYPE: Final = "usdt-futures"
PAGE_SIZE: Final = 100
CACHE_DIR: Final = Path(__file__).resolve().parent / "cache"
REQUEST_PAUSE: Final = 0.06
MAX_RETRIES: Final = 4
MAX_PAGES: Final = 120  # 100 stamps/page x 120 = 12,000 stamps ~ 11 years of 8h funding


class FundingError(RuntimeError):
    pass


def _get(params: dict[str, Any]) -> Any:
    last: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(
                f"{BITGET}/api/v2/mix/market/history-fund-rate", params=params, timeout=30
            )
            if response.status_code == 429:
                time.sleep(1.0 + attempt)
                continue
            payload = response.json()
            if payload.get("code") != "00000":
                raise FundingError(f"{payload.get('code')}: {payload.get('msg')}")
            return payload["data"]
        except (requests.RequestException, ValueError, FundingError) as error:
            last = error
            time.sleep(0.4 * (attempt + 1))
    raise FundingError(f"failed after {MAX_RETRIES} attempts: {last}")


def funding_path(symbol: str) -> Path:
    return CACHE_DIR / f"bitget_{symbol}_funding.csv.gz"


def collect_symbol(symbol: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    seen_oldest: pd.Timestamp | None = None
    for page in range(1, MAX_PAGES + 1):
        rows = _get(
            {"symbol": symbol, "productType": PRODUCT_TYPE, "pageSize": PAGE_SIZE, "pageNo": page}
        )
        time.sleep(REQUEST_PAUSE)
        if not rows:
            break
        frame = pd.DataFrame(rows)
        frame["timestamp"] = pd.to_datetime(frame["fundingTime"].astype("int64"), unit="ms", utc=True)
        frame["funding_rate"] = pd.to_numeric(frame["fundingRate"], errors="coerce")
        frames.append(frame[["timestamp", "funding_rate"]])
        oldest = frame["timestamp"].min()
        if seen_oldest is not None and oldest >= seen_oldest:
            break  # pagination stopped advancing
        seen_oldest = oldest
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True).dropna()
    return out.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect Bitget funding history (public, read-only)")
    parser.add_argument("--budget-seconds", type=float, default=480.0)
    args = parser.parse_args(argv)

    symbols = sorted(
        path.name.replace("bitget_", "").replace("_1H.csv.gz", "")
        for path in CACHE_DIR.glob("bitget_*_1H.csv.gz")
    )
    started = time.time()
    done = skipped = failed = 0
    for symbol in symbols:
        if funding_path(symbol).exists():
            skipped += 1
            continue
        if time.time() - started > args.budget_seconds:
            print("시간 예산 소진 — 다음 실행에서 이어서 수집", flush=True)
            break
        try:
            frame = collect_symbol(symbol)
        except FundingError as error:
            print(f"  {symbol}: 실패 {error}", flush=True)
            failed += 1
            continue
        if frame.empty:
            print(f"  {symbol}: 펀딩 이력 없음", flush=True)
            failed += 1
            continue
        with gzip.open(funding_path(symbol), "wt", encoding="utf-8", newline="\n") as stream:
            frame.to_csv(stream, index=False)
        done += 1
        print(
            f"  {symbol:14s} {len(frame):5d}건 {frame['timestamp'].iloc[0].date()} → "
            f"{frame['timestamp'].iloc[-1].date()} | 평균 {frame['funding_rate'].mean()*3*365:+.2%} APR",
            flush=True,
        )
    total = len(list(CACHE_DIR.glob("bitget_*_funding.csv.gz")))
    print(f"\n신규 {done} · 기존 {skipped} · 실패 {failed} · 누적 {total}/{len(symbols)}종목 "
          f"({time.time()-started:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
