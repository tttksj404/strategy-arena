"""Bitget USDT-perp klines + funding rate history fetch."""
import json, time, urllib.request, urllib.error
from pathlib import Path
from datetime import datetime, timezone

KLINES_OUT = Path.home() / "Desktop" / "first_repo" / "quant_runtime" / "historical_bitget"
FUND_OUT = Path.home() / "Desktop" / "first_repo" / "quant_runtime" / "funding_bitget"

# Bitget USDT-FUTURES top symbols (overlap with Binance top50)
SYMBOLS = [
    "BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT","XRPUSDT","DOGEUSDT","ADAUSDT","AVAXUSDT","LINKUSDT","DOTUSDT",
    "LTCUSDT","UNIUSDT","NEARUSDT","ATOMUSDT","ETCUSDT","XLMUSDT","BCHUSDT","FILUSDT","ICPUSDT",
    "APTUSDT","ARBUSDT","OPUSDT","SUIUSDT","SEIUSDT","TIAUSDT","INJUSDT","STXUSDT","GRTUSDT","RUNEUSDT",
    "AAVEUSDT","ALGOUSDT","SANDUSDT","AXSUSDT",
    "PEPEUSDT","WIFUSDT","SHIBUSDT","FLOKIUSDT","BONKUSDT","MEMEUSDT","PNUTUSDT","BOMEUSDT","TURBOUSDT","NEIROUSDT",
]

START_MS = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
END_MS   = int(datetime(2026, 4, 28, tzinfo=timezone.utc).timestamp() * 1000)
HOUR_MS = 3600 * 1000

KLINES_BASE = "https://api.bitget.com/api/v2/mix/market/candles"
FUND_BASE = "https://api.bitget.com/api/v2/mix/market/history-fund-rate"


def fetch_klines(symbol):
    out_dir = KLINES_OUT / symbol; out_dir.mkdir(parents=True, exist_ok=True)
    out_p = out_dir / "1h.json"
    if out_p.exists(): return len(json.loads(out_p.read_text()))
    bars = []
    cur = START_MS
    while cur < END_MS:
        end_chunk = min(cur + 1000 * HOUR_MS, END_MS)
        url = f"{KLINES_BASE}?symbol={symbol}&granularity=1H&productType=USDT-FUTURES&startTime={cur}&endTime={end_chunk}&limit=1000"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                resp = json.loads(r.read())
        except Exception:
            time.sleep(1); continue
        if resp.get("code") != "00000": break
        chunk = resp.get("data", [])
        if not chunk: break
        for b in chunk:
            ts = int(b[0])
            if ts >= END_MS: break
            bars.append({
                "open_time": ts,
                "open_price": float(b[1]),
                "high_price": float(b[2]),
                "low_price": float(b[3]),
                "close_price": float(b[4]),
                "base_volume": float(b[5]),
                "quote_volume": float(b[6]),
            })
        last_ts = int(chunk[-1][0])
        if last_ts >= END_MS - HOUR_MS: break
        cur = last_ts + HOUR_MS
        time.sleep(0.05)
    if bars:
        bars.sort(key=lambda x: x["open_time"])
        # dedup
        seen = set(); uniq = []
        for b in bars:
            if b["open_time"] not in seen:
                seen.add(b["open_time"]); uniq.append(b)
        out_p.write_text(json.dumps(uniq))
        return len(uniq)
    return 0


def fetch_funding(symbol):
    """Bitget funding rate (8h interval, per symbol). pageSize max 100."""
    out_dir = FUND_OUT / symbol; out_dir.mkdir(parents=True, exist_ok=True)
    out_p = out_dir / "funding.json"
    if out_p.exists(): return len(json.loads(out_p.read_text()))
    # Bitget history-fund-rate: 페이지네이션 X (가장 최근 100개), 큰 데이터 X
    # 대안: 100개씩 가져오면 ~33일분 (8h × 100 = 800h ≈ 33일)
    url = f"{FUND_BASE}?symbol={symbol}&productType=USDT-FUTURES&pageSize=100"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            resp = json.loads(r.read())
    except Exception:
        return 0
    if resp.get("code") != "00000": return 0
    data = resp.get("data", [])
    if not data: return 0
    rates = [{"ts": int(d["fundingTime"]), "rate": float(d["fundingRate"]), "symbol": d["symbol"]} for d in data]
    rates.sort(key=lambda x: x["ts"])
    out_p.write_text(json.dumps(rates))
    return len(rates)


if __name__ == "__main__":
    KLINES_OUT.mkdir(parents=True, exist_ok=True)
    FUND_OUT.mkdir(parents=True, exist_ok=True)
    print("=== Bitget klines fetch ===")
    success_k = 0; failed_k = []
    for i, sym in enumerate(SYMBOLS, 1):
        try:
            n = fetch_klines(sym)
            if n > 0:
                success_k += 1
                if i % 10 == 0: print(f"[{i}/{len(SYMBOLS)}] klines {sym}: {n}")
            else:
                failed_k.append(sym)
        except Exception as e:
            failed_k.append(sym)
    print(f"klines total: {success_k}/{len(SYMBOLS)} | failed: {failed_k}")

    print("\n=== Bitget funding rate fetch ===")
    success_f = 0; failed_f = []
    for i, sym in enumerate(SYMBOLS, 1):
        try:
            n = fetch_funding(sym)
            if n > 0:
                success_f += 1
            else:
                failed_f.append(sym)
        except Exception as e:
            failed_f.append(sym)
    print(f"funding total: {success_f}/{len(SYMBOLS)} | failed: {failed_f}")
