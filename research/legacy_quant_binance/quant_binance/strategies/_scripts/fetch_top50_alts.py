"""Top 50 USDT-perp alts (volume-ranked) fetch from Binance public."""
import json, time, urllib.request
from pathlib import Path
from datetime import datetime, timezone

OUT = Path.home() / "Desktop" / "first_repo" / "quant_runtime" / "historical_top50"
START_MS = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
END_MS   = int(datetime(2026, 4, 28, tzinfo=timezone.utc).timestamp() * 1000)
HOUR_MS = 3600 * 1000

# Top 50 USDT spot pairs by typical volume (excluding stablecoins, leveraged tokens)
TOP50 = [
    "BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT","XRPUSDT","DOGEUSDT","ADAUSDT","AVAXUSDT","LINKUSDT","DOTUSDT",
    "MATICUSDT","LTCUSDT","UNIUSDT","NEARUSDT","ATOMUSDT","ETCUSDT","XLMUSDT","BCHUSDT","FILUSDT","ICPUSDT",
    "APTUSDT","ARBUSDT","OPUSDT","SUIUSDT","SEIUSDT","TIAUSDT","INJUSDT","STXUSDT","GRTUSDT","RUNEUSDT",
    "AAVEUSDT","ALGOUSDT","FTMUSDT","SANDUSDT","MANAUSDT","AXSUSDT","THETAUSDT","EOSUSDT","XTZUSDT","NEOUSDT",
    "PEPEUSDT","WIFUSDT","SHIBUSDT","FLOKIUSDT","BONKUSDT","MEMEUSDT","PNUTUSDT","BOMEUSDT","TURBOUSDT","NEIROUSDT",
]

BASE = "https://api.binance.com/api/v3/klines"


def fetch(symbol):
    out_dir = OUT / symbol; out_dir.mkdir(parents=True, exist_ok=True)
    out_p = out_dir / "1h.json"
    if out_p.exists():
        return len(json.loads(out_p.read_text()))
    bars = []; cur = START_MS
    while cur < END_MS:
        url = f"{BASE}?symbol={symbol}&interval=1h&startTime={cur}&limit=1000"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                chunk = json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code == 400: return -1
            time.sleep(2); continue
        except Exception:
            time.sleep(2); continue
        if not chunk: break
        for b in chunk:
            if b[0] >= END_MS: break
            bars.append({"open_time":b[0],"open_price":float(b[1]),"high_price":float(b[2]),"low_price":float(b[3]),"close_price":float(b[4]),"base_volume":float(b[5]),"quote_volume":float(b[7])})
        last = chunk[-1][0]
        if last >= END_MS - HOUR_MS: break
        cur = last + HOUR_MS
        time.sleep(0.05)
    if bars:
        out_p.write_text(json.dumps(bars))
    return len(bars)


import urllib.error

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    success = 0; failed = []
    for i, sym in enumerate(TOP50, 1):
        try:
            n = fetch(sym)
            if n > 0:
                success += 1
                if i % 10 == 0: print(f"[{i}/{len(TOP50)}] {sym}: {n} ✓")
            else:
                failed.append(sym)
        except Exception as e:
            failed.append(sym)
    print(f"\nTotal: {success}/{len(TOP50)} success | failed: {failed}")
