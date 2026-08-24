"""IS25-26 historical 1h klines (374d) — G185 universe 18 alts. Binance public API.

Period: 2025-04-20 → 2026-04-29 (~374 days)
Output: quant_runtime/historical_is25/{SYM}/1h.json
"""
import json, time, urllib.request, sys
from pathlib import Path
from datetime import datetime, timezone

OUT = Path(r"C:\Users\SSAFY\Desktop\first_repo\quant_runtime\historical_is25")

UNIVERSE = [
    "DOGEUSDT","PEPEUSDT","WIFUSDT","ARBUSDT","OPUSDT","AVAXUSDT","SUIUSDT","ADAUSDT",
    "APTUSDT","BNBUSDT","DOTUSDT","LINKUSDT","LTCUSDT","NEARUSDT","SOLUSDT","UNIUSDT","XRPUSDT","BTCUSDT",
]

START_MS = int(datetime(2025, 4, 20, tzinfo=timezone.utc).timestamp() * 1000)
END_MS   = int(datetime(2026, 4, 29, tzinfo=timezone.utc).timestamp() * 1000)
BASE = "https://api.binance.com/api/v3/klines"
HOUR_MS = 3600 * 1000


def fetch(symbol):
    out_dir = OUT / symbol; out_dir.mkdir(parents=True, exist_ok=True)
    out_p = out_dir / "1h.json"
    if out_p.exists():
        try:
            existing = json.loads(out_p.read_text())
            if len(existing) >= 8000:
                return len(existing)
        except Exception:
            pass
    bars = []; cur = START_MS
    while cur < END_MS:
        url = f"{BASE}?symbol={symbol}&interval=1h&startTime={cur}&limit=1000"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                chunk = json.loads(r.read())
        except Exception as e:
            sys.stderr.write(f"  retry {symbol} cur={cur}: {e}\n")
            time.sleep(2); continue
        if not chunk: break
        for b in chunk:
            if b[0] >= END_MS: break
            bars.append({
                "open_time": b[0],
                "open_price": float(b[1]),
                "high_price": float(b[2]),
                "low_price":  float(b[3]),
                "close_price":float(b[4]),
                "base_volume":float(b[5]),
                "quote_volume":float(b[7]),
            })
        last = chunk[-1][0]
        if last >= END_MS - HOUR_MS: break
        cur = last + HOUR_MS
        time.sleep(0.1)
    out_p.write_text(json.dumps(bars))
    return len(bars)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for i, sym in enumerate(UNIVERSE, 1):
        try:
            n = fetch(sym)
            print(f"[{i:2d}/{len(UNIVERSE)}] {sym}: {n} bars")
        except Exception as e:
            print(f"[{i:2d}/{len(UNIVERSE)}] {sym}: FAIL {e}")
