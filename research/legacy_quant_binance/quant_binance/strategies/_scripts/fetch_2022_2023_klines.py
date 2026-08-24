"""
2022-2023 historical 1h klines fetcher (Binance public API, 인증 X).

대상: 2022-2023 24개월 (BTC bear $69k→$15k→ recovery)
Universe: 14 symbols (2022 시점에 존재했던 것만)

저장: ~/Desktop/first_repo/quant_runtime/historical_2022/{SYMBOL}/1h.json
"""
import json
import time
import urllib.request
from pathlib import Path
from datetime import datetime, timezone

OUT = Path.home() / "Desktop" / "first_repo" / "quant_runtime" / "historical_2022"

# 2022-01-01 시점에 이미 상장된 14 알트만
UNIVERSE_2022 = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "SOLUSDT",
    "DOGEUSDT", "DOTUSDT", "LINKUSDT", "LTCUSDT", "AVAXUSDT",
    "MATICUSDT", "NEARUSDT", "UNIUSDT", "XRPUSDT",
]

START_MS = int(datetime(2022, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
END_MS   = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
INTERVAL = "1h"
LIMIT = 1000  # max per request
HOUR_MS = 3600 * 1000

BASE = "https://api.binance.com/api/v3/klines"


def fetch_chunk(symbol, start_ts):
    url = f"{BASE}?symbol={symbol}&interval={INTERVAL}&startTime={start_ts}&limit={LIMIT}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                return json.loads(r.read())
        except Exception as e:
            if attempt == 2: raise
            time.sleep(2 ** attempt)
    return []


def fetch_symbol(symbol):
    out_dir = OUT / symbol
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{INTERVAL}.json"
    if out_path.exists():
        existing = json.loads(out_path.read_text())
        if existing and existing[-1]["open_time"] >= END_MS - HOUR_MS:
            print(f"  {symbol}: cached n={len(existing)} → skip")
            return len(existing)
    all_bars = []
    cur = START_MS
    while cur < END_MS:
        chunk = fetch_chunk(symbol, cur)
        if not chunk:
            break
        for b in chunk:
            if b[0] >= END_MS: break
            all_bars.append({
                "open_time": b[0],
                "open_price": float(b[1]),
                "high_price": float(b[2]),
                "low_price": float(b[3]),
                "close_price": float(b[4]),
                "base_volume": float(b[5]),
                "quote_volume": float(b[7]),
            })
        last = chunk[-1][0]
        if last >= END_MS - HOUR_MS:
            break
        cur = last + HOUR_MS
        time.sleep(0.1)  # gentle rate limit
    out_path.write_text(json.dumps(all_bars))
    return len(all_bars)


def main():
    print(f"Fetching {len(UNIVERSE_2022)} symbols × ~17.5K bars each ({INTERVAL}, 2022-2023)")
    OUT.mkdir(parents=True, exist_ok=True)
    total = 0
    failed = []
    for i, sym in enumerate(UNIVERSE_2022, 1):
        try:
            n = fetch_symbol(sym)
            print(f"[{i}/{len(UNIVERSE_2022)}] {sym}: {n} bars")
            total += n
        except Exception as e:
            print(f"[{i}/{len(UNIVERSE_2022)}] {sym}: FAILED - {e}")
            failed.append(sym)
    print(f"\nTotal bars: {total} | Failed: {failed}")


if __name__ == "__main__":
    main()
