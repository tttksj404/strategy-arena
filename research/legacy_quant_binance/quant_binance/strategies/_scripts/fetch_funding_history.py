"""Binance FAPI funding rate history (per symbol, last 1000 entries ~11개월).

Bitget v2 API stuck issue 우회. Funding rate 는 양 거래소 거의 동일 (penalty arbitrage).
"""
import json, time, urllib.request, urllib.error
from pathlib import Path

OUT = Path.home() / "Desktop" / "first_repo" / "quant_runtime" / "funding_binance"

SYMBOLS = [
    "BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT","XRPUSDT","DOGEUSDT","ADAUSDT","AVAXUSDT","LINKUSDT","DOTUSDT",
    "MATICUSDT","LTCUSDT","UNIUSDT","NEARUSDT","ATOMUSDT","APTUSDT","ARBUSDT","OPUSDT","SUIUSDT","SEIUSDT",
    "TIAUSDT","INJUSDT","STXUSDT","RUNEUSDT","AAVEUSDT","ALGOUSDT","FTMUSDT","SANDUSDT","AXSUSDT",
    "PEPEUSDT","WIFUSDT","SHIBUSDT","FLOKIUSDT","BONKUSDT","MEMEUSDT","PNUTUSDT","BOMEUSDT","TURBOUSDT","NEIROUSDT",
]

BASE = "https://fapi.binance.com/fapi/v1/fundingRate"


def fetch(symbol):
    out_dir = OUT / symbol; out_dir.mkdir(parents=True, exist_ok=True)
    out_p = out_dir / "funding.json"
    if out_p.exists(): return len(json.loads(out_p.read_text()))
    # Limit max 1000, no pagination needed for our backtest (we have 374-day data, funding ~3/day = 1100/year)
    url = f"{BASE}?symbol={symbol}&limit=1000"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read())
    except urllib.error.HTTPError as e:
        if e.code in (400, 404): return -1
        return 0
    except Exception:
        return 0
    if not data: return 0
    # data items: {symbol, fundingTime, fundingRate, markPrice}
    rates = [{"ts": int(d["fundingTime"]), "rate": float(d["fundingRate"])} for d in data]
    rates.sort(key=lambda x: x["ts"])
    out_p.write_text(json.dumps(rates))
    return len(rates)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    success = 0; failed = []
    for i, sym in enumerate(SYMBOLS, 1):
        try:
            n = fetch(sym)
            if n > 0:
                success += 1
                if i % 10 == 0: print(f"[{i}/{len(SYMBOLS)}] {sym}: {n} entries")
            else:
                failed.append(sym)
        except Exception:
            failed.append(sym)
        time.sleep(0.05)
    print(f"\nTotal: {success}/{len(SYMBOLS)} | failed: {failed}")
