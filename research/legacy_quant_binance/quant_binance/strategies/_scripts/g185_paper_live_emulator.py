"""G185 paper-live emulator — real-time CH1 score on Binance public 1h klines.

전략은 standalone backtest 스크립트 (g070_lottery_design.py / g002_mingogogo_ch1_backtest.py)
의 score_engine 을 그대로 사용. quant_binance.daemon 은 별도 auto-mode 정책이라 G185 와 hook 없음.
이 emulator 가 G185 전략의 paper-live 진실 표본.

Cycle:
  - 매 5분 Binance public klines 조회 (최근 200 bars × 1h × 18 syms)
  - 최근 종가 bar 의 CH1 score, ATR%, BB%B 계산
  - score ≥ 80 + ATR ≤ 8% + 동일 sym 미보유 + 동시 5건 미만 → paper entry
  - hold 24h 후 자동 exit
  - 모든 거래 quant_runtime_g185_paper/g185_emulator/trades.jsonl 기록

Output:
  trades.jsonl       — entry/exit 이벤트
  state.json         — open_positions, equity_usd, cumulative_pnl_usd, last_cycle
  emulator.log       — heartbeat / decision / error
"""
import json
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\Users\SSAFY\Desktop\first_repo")
sys.path.insert(0, str(ROOT / "quant_binance" / "strategies" / "_scripts"))
from g002_mingogogo_ch1_backtest import compute_ch1_score, atr_pct  # type: ignore

OUT = ROOT / "quant_runtime_g185_paper" / "g185_emulator"
OUT.mkdir(parents=True, exist_ok=True)
TRADES = OUT / "trades.jsonl"
STATE = OUT / "state.json"
LOG = OUT / "emulator.log"

UNIVERSE = [
    "DOGEUSDT","PEPEUSDT","WIFUSDT","ARBUSDT","OPUSDT","AVAXUSDT","SUIUSDT","ADAUSDT",
    "APTUSDT","BNBUSDT","DOTUSDT","LINKUSDT","LTCUSDT","NEARUSDT","SOLUSDT","UNIUSDT","XRPUSDT","BTCUSDT",
]

EQUITY_USD = 100.0
SIZE_PCT = 0.40
LEVERAGE = 5.0
THRESHOLD = 80
HOLD_BARS = 24
MAX_CONC = 5
ATR_GUARD_PCT = 8.0
COST_BPS_RT = 16.0
CYCLE_SECONDS = 300

BINANCE_KLINES = "https://api.binance.com/api/v3/klines"


def log(msg: str):
    line = f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_state() -> dict:
    if STATE.exists():
        try:
            return json.loads(STATE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "open_positions": {},
        "closed_count": 0,
        "wins": 0,
        "losses": 0,
        "cumulative_pnl_usd": 0.0,
        "last_cycle": None,
    }


def save_state(state: dict):
    state["last_cycle"] = datetime.now(timezone.utc).isoformat()
    STATE.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def append_trade(event: dict):
    with TRADES.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def fetch_klines(symbol: str, limit: int = 200) -> pd.DataFrame | None:
    url = f"{BINANCE_KLINES}?symbol={symbol}&interval=1h&limit={limit}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as r:
            chunk = json.loads(r.read())
    except Exception as e:
        log(f"  fetch {symbol} FAIL: {e}")
        return None
    if not chunk:
        return None
    bars = []
    for b in chunk[:-1]:  # exclude unfinished current bar
        bars.append({
            "open_time": b[0],
            "open_price": float(b[1]),
            "high_price": float(b[2]),
            "low_price":  float(b[3]),
            "close_price": float(b[4]),
            "base_volume": float(b[5]),
            "quote_volume": float(b[7]),
        })
    return pd.DataFrame(bars)


def evaluate(symbol: str) -> dict | None:
    df = fetch_klines(symbol)
    if df is None or len(df) < 100:
        return None
    score, _ = compute_ch1_score(df)
    atrp = atr_pct(df["high_price"], df["low_price"], df["close_price"], 14)
    last_score = float(score.iloc[-1]) if not pd.isna(score.iloc[-1]) else 0.0
    last_atr = float(atrp.iloc[-1]) if not pd.isna(atrp.iloc[-1]) else 0.0
    last_close = float(df["close_price"].iloc[-1])
    last_ts = int(df["open_time"].iloc[-1])
    return {
        "symbol": symbol,
        "score": last_score,
        "atr_pct": last_atr,
        "close": last_close,
        "bar_ts": last_ts,
    }


def cycle(state: dict):
    now_ms = int(time.time() * 1000)
    open_positions: dict = state["open_positions"]

    # 1) exit any positions past hold
    expired = []
    for sym, pos in list(open_positions.items()):
        exit_ms = pos["entry_ts_ms"] + HOLD_BARS * 3600 * 1000
        if now_ms >= exit_ms:
            expired.append(sym)
    for sym in expired:
        pos = open_positions.pop(sym)
        df = fetch_klines(sym, limit=2)
        if df is None:
            log(f"  EXIT {sym} fetch fail — keeping open")
            open_positions[sym] = pos
            continue
        exit_close = float(df["close_price"].iloc[-1])
        gross_pct = (exit_close / pos["entry_close"] - 1) * 100
        net_pct = gross_pct / 100 - COST_BPS_RT / 10000
        net_pct_levered = net_pct * LEVERAGE
        if net_pct_levered < -0.90:
            net_pct_levered = -0.90  # liquidation cap
        margin = EQUITY_USD * SIZE_PCT
        pnl_usd = round(margin * net_pct_levered, 4)
        state["cumulative_pnl_usd"] = round(state["cumulative_pnl_usd"] + pnl_usd, 4)
        state["closed_count"] += 1
        if pnl_usd > 0:
            state["wins"] += 1
        else:
            state["losses"] += 1
        event = {
            "type": "EXIT", "symbol": sym,
            "entry_ts_ms": pos["entry_ts_ms"], "entry_close": pos["entry_close"],
            "exit_ts_ms": now_ms, "exit_close": exit_close,
            "gross_pct": round(gross_pct, 4), "net_pct_levered": round(net_pct_levered, 4),
            "pnl_usd": pnl_usd,
            "score_at_entry": pos.get("score"),
            "atr_pct_at_entry": pos.get("atr_pct"),
        }
        append_trade(event)
        log(f"  EXIT  {sym}: {gross_pct:+.2f}% gross / {net_pct_levered*100:+.2f}% lev / PnL ${pnl_usd:+.2f}")

    # 2) evaluate all symbols, sort by score desc, attempt entries
    cands = []
    for sym in UNIVERSE:
        if sym in open_positions:
            continue
        info = evaluate(sym)
        if info is None:
            continue
        cands.append(info)
        time.sleep(0.05)
    cands.sort(key=lambda x: x["score"], reverse=True)

    for info in cands:
        sym = info["symbol"]
        if len(open_positions) >= MAX_CONC:
            break
        if info["score"] < THRESHOLD:
            continue
        if info["atr_pct"] > ATR_GUARD_PCT:
            log(f"  SKIP {sym}: score {info['score']:.1f} but ATR {info['atr_pct']:.2f}% > {ATR_GUARD_PCT}%")
            continue
        # ENTRY
        pos = {
            "entry_ts_ms": now_ms,
            "entry_close": info["close"],
            "score": info["score"],
            "atr_pct": info["atr_pct"],
        }
        open_positions[sym] = pos
        event = {
            "type": "ENTRY", "symbol": sym, "entry_ts_ms": now_ms,
            "entry_close": info["close"], "score": info["score"], "atr_pct": info["atr_pct"],
            "size_usd_margin": EQUITY_USD * SIZE_PCT,
            "notional_usd": EQUITY_USD * SIZE_PCT * LEVERAGE,
        }
        append_trade(event)
        log(f"  ENTRY {sym}: score {info['score']:.1f} ATR {info['atr_pct']:.2f}% close {info['close']:.6f}")

    state["open_positions"] = open_positions
    save_state(state)

    top5 = [(c["symbol"], round(c["score"], 1)) for c in cands[:5]]
    log(f"  HEARTBEAT open={len(open_positions)} closed={state['closed_count']} W/L={state['wins']}/{state['losses']} cumPnL=${state['cumulative_pnl_usd']:+.2f} top5={top5}")


def main():
    log(f"=== G185 paper-live emulator START === capital=${EQUITY_USD} size={SIZE_PCT} lev={LEVERAGE}x thr={THRESHOLD} hold={HOLD_BARS}h max_conc={MAX_CONC} atr_guard={ATR_GUARD_PCT}%")
    log(f"universe ({len(UNIVERSE)}): {','.join(UNIVERSE)}")
    log(f"output -> {OUT}")
    state = load_state()
    while True:
        try:
            cycle(state)
        except KeyboardInterrupt:
            log("STOP via KeyboardInterrupt")
            break
        except Exception as e:
            log(f"CYCLE_ERROR: {type(e).__name__}: {e}")
        time.sleep(CYCLE_SECONDS)


if __name__ == "__main__":
    main()
