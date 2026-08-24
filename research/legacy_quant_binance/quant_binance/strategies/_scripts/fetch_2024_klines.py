"""2024 historical 1h klines (Binance public API, 인증 X) — 갭 메우기."""
import json, time, urllib.request
from pathlib import Path
from datetime import datetime, timezone

OUT = Path.home() / "Desktop" / "first_repo" / "quant_runtime" / "historical_2024"

UNIVERSE = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "SOLUSDT",
    "DOGEUSDT", "DOTUSDT", "LINKUSDT", "LTCUSDT", "AVAXUSDT",
    "NEARUSDT", "UNIUSDT", "XRPUSDT",
    "OPUSDT", "ARBUSDT", "APTUSDT", "PEPEUSDT", "SUIUSDT",
]

START_MS = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
END_MS   = int(datetime(2025, 4, 1, tzinfo=timezone.utc).timestamp() * 1000)  # 15개월
BASE = "https://api.binance.com/api/v3/klines"
HOUR_MS = 3600 * 1000


def fetch(symbol):
    out_dir = OUT / symbol; out_dir.mkdir(parents=True, exist_ok=True)
    out_p = out_dir / "1h.json"
    if out_p.exists(): return len(json.loads(out_p.read_text()))
    bars = []; cur = START_MS
    while cur < END_MS:
        url = f"{BASE}?symbol={symbol}&interval=1h&startTime={cur}&limit=1000"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                chunk = json.loads(r.read())
        except Exception:
            time.sleep(2); continue
        if not chunk: break
        for b in chunk:
            if b[0] >= END_MS: break
            bars.append({"open_time":b[0],"open_price":float(b[1]),"high_price":float(b[2]),"low_price":float(b[3]),"close_price":float(b[4]),"base_volume":float(b[5]),"quote_volume":float(b[7])})
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
            print(f"[{i}/{len(UNIVERSE)}] {sym}: {n}")
        except Exception as e:
            print(f"[{i}/{len(UNIVERSE)}] {sym}: FAIL {e}")
