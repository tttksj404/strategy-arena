"""메메코인 1h klines fetch (Binance public, 2024-2026 가용 범위)."""
import json, time, urllib.request, urllib.error
from pathlib import Path
from datetime import datetime, timezone

OUT = Path.home() / "Desktop" / "first_repo" / "quant_runtime" / "historical_meme"
MEMES = ["SHIBUSDT","FLOKIUSDT","BONKUSDT","MEMEUSDT","PNUTUSDT","BOMEUSDT","NOTUSDT","TURBOUSDT","NEIROUSDT","MEWUSDT"]

START_MS = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
END_MS   = int(datetime(2026, 4, 28, tzinfo=timezone.utc).timestamp() * 1000)
BASE = "https://api.binance.com/api/v3/klines"
HOUR_MS = 3600 * 1000


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
            if e.code == 400:
                # symbol 미존재 또는 시작일 너무 이른 것
                return -1
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
        time.sleep(0.1)
    if bars:
        out_p.write_text(json.dumps(bars))
    return len(bars)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for i, sym in enumerate(MEMES, 1):
        try:
            n = fetch(sym)
            print(f"[{i}/{len(MEMES)}] {sym}: {n}")
        except Exception as e:
            print(f"[{i}/{len(MEMES)}] {sym}: FAIL {e}")
