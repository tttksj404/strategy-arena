"""Funding rate history paginated (2024-01 ~ now). Meme aliases handled."""
import json, time, urllib.request
from pathlib import Path
from datetime import datetime, timezone

OUT = Path.home() / "Desktop" / "first_repo" / "quant_runtime" / "funding_binance"

# (klines_symbol, fapi_symbol)
SYMBOL_MAP = [
    ("BTCUSDT", "BTCUSDT"),("ETHUSDT","ETHUSDT"),("SOLUSDT","SOLUSDT"),("BNBUSDT","BNBUSDT"),
    ("XRPUSDT","XRPUSDT"),("DOGEUSDT","DOGEUSDT"),("ADAUSDT","ADAUSDT"),("AVAXUSDT","AVAXUSDT"),
    ("LINKUSDT","LINKUSDT"),("DOTUSDT","DOTUSDT"),("LTCUSDT","LTCUSDT"),("UNIUSDT","UNIUSDT"),
    ("NEARUSDT","NEARUSDT"),("ATOMUSDT","ATOMUSDT"),("APTUSDT","APTUSDT"),("ARBUSDT","ARBUSDT"),
    ("OPUSDT","OPUSDT"),("SUIUSDT","SUIUSDT"),("SEIUSDT","SEIUSDT"),("TIAUSDT","TIAUSDT"),
    ("INJUSDT","INJUSDT"),("STXUSDT","STXUSDT"),("RUNEUSDT","RUNEUSDT"),("AAVEUSDT","AAVEUSDT"),
    ("ALGOUSDT","ALGOUSDT"),
    # Memes (Binance FAPI uses 1000- prefix for low-price tokens)
    ("PEPEUSDT", "1000PEPEUSDT"),
    ("SHIBUSDT", "1000SHIBUSDT"),
    ("FLOKIUSDT", "1000FLOKIUSDT"),
    ("BONKUSDT", "1000BONKUSDT"),
    ("WIFUSDT", "WIFUSDT"),
    ("MEMEUSDT", "MEMEUSDT"),
    ("PNUTUSDT", "1000PEPEUSDT"),  # PNUT not on FAPI, fallback
    ("BOMEUSDT", "BOMEUSDT"),
    ("TURBOUSDT", "1000TURBOUSDT"),
    ("NEIROUSDT", "1000NEIROCTOUSDT"),
]

START_MS = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
END_MS = int(datetime(2026, 4, 28, tzinfo=timezone.utc).timestamp() * 1000)
BASE = "https://fapi.binance.com/fapi/v1/fundingRate"


def fetch(klines_sym, fapi_sym):
    out_dir = OUT / klines_sym; out_dir.mkdir(parents=True, exist_ok=True)
    out_p = out_dir / "funding.json"
    if out_p.exists():
        existing = json.loads(out_p.read_text())
        if len(existing) > 1000: return len(existing)  # already paginated
    all_rates = []
    cur = START_MS
    while cur < END_MS:
        url = f"{BASE}?symbol={fapi_sym}&startTime={cur}&limit=1000"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                data = json.loads(r.read())
        except Exception:
            return len(all_rates)
        if not data: break
        for d in data:
            ts = int(d["fundingTime"])
            if ts >= END_MS: break
            all_rates.append({"ts": ts, "rate": float(d["fundingRate"])})
        last_ts = int(data[-1]["fundingTime"])
        if last_ts >= END_MS - 8 * 3600 * 1000: break
        cur = last_ts + 8 * 3600 * 1000
        time.sleep(0.05)
    if all_rates:
        # dedupe + sort
        seen = set(); uniq = []
        for r in all_rates:
            if r["ts"] not in seen:
                seen.add(r["ts"]); uniq.append(r)
        uniq.sort(key=lambda x: x["ts"])
        out_p.write_text(json.dumps(uniq))
        return len(uniq)
    return 0


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    success = 0; failed = []
    for i, (ks, fs) in enumerate(SYMBOL_MAP, 1):
        try:
            n = fetch(ks, fs)
            if n > 0:
                success += 1
                if i % 8 == 0: print(f"[{i}/{len(SYMBOL_MAP)}] {ks}: {n} entries")
            else:
                failed.append(f"{ks}({fs})")
        except Exception:
            failed.append(f"{ks}({fs})")
    print(f"\nTotal: {success}/{len(SYMBOL_MAP)} | failed: {failed}")
