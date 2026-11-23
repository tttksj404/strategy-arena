#!/usr/bin/env python3
"""Final-lineup paper: G053(h24 L5, Binance 14-alt) $140 + regime-xsec booster $60.

Hourly tick. Slot A recomputes CH1 scores + the 14d/7d gate from fresh Binance
klines each tick, enters score>=70 (slot5, symbol-dedup), holds 24h, 16bps RT.
Slot B refreshes the Bitget daily panel once per UTC day, recomputes regime
(trend20>0 & breadth>=0.6) and rebalances lb40 top5/bottom5 with vol-target 0.08
(max lev 5); off-regime = cash. State/trades/log persist next to this file.
PAPER2_ONCE=1 for a single tick; default loops hourly. Auto-stop after 24h.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/Users/tttksj/first_repo/quant_binance/strategies/_scripts")
from g002_mingogogo_ch1_backtest import compute_ch1_score  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
STATE = HERE / "state2.json"
TRADES = HERE / "trades2.jsonl"
LOG = HERE / "paper2.log"
UNIVERSE = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "SOLUSDT", "DOGEUSDT", "XRPUSDT",
            "DOTUSDT", "LTCUSDT", "LINKUSDT", "AVAXUSDT", "UNIUSDT", "ATOMUSDT", "NEARUSDT"]
G_CAP, B_CAP = 140.0, 60.0
G_LEV, SLOTS, THR, HOLD_H = 5, 5, 70, 24
RT_BPS = 16.0
TV, MAX_LEV, LB, TOPK = 0.08, 5, 40, 5
RUN_HOURS = 24


def log(line: str) -> None:
    with LOG.open("a", encoding="utf-8") as h:
        h.write(f"{datetime.now(UTC).isoformat(timespec='seconds')} {line}\n")


def trade(rec: dict) -> None:
    rec["ts"] = datetime.now(UTC).isoformat(timespec="seconds")
    with TRADES.open("a", encoding="utf-8") as h:
        h.write(json.dumps(rec) + "\n")


def bn_klines(symbol: str, limit: int = 600) -> pd.DataFrame | None:
    url = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval=1h&limit={limit}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        raw = json.load(urllib.request.urlopen(req, timeout=15))
    except Exception:
        return None
    df = pd.DataFrame(raw, columns=["open_time", "open_price", "high_price", "low_price",
                                    "close_price", "base_volume", "ct", "qv", "n", "tb", "tq", "ig"])
    for col in ("open_price", "high_price", "low_price", "close_price", "base_volume"):
        df[col] = pd.to_numeric(df[col])
    return df.iloc[:-1].reset_index(drop=True)  # drop unclosed bar


def g053_tick(slot: dict) -> None:
    now_ms = int(time.time() * 1000)
    frames = {}
    for sym in UNIVERSE:
        df = bn_klines(sym)
        if df is not None and len(df) > 120:
            score, _ = compute_ch1_score(df)
            df["score"] = score.to_numpy()
            frames[sym] = df
        time.sleep(0.08)
    if not frames:
        log("G053 API_FAIL")
        return
    # 게이트: 최근 14d/7d의 '종결된' 진입(24h 경과) net 합
    closed = []
    for sym, df in frames.items():
        fwd = (df.close_price.shift(-HOLD_H) / df.close_price - 1) * 10000
        e = df[(df.score >= THR) & fwd.notna()]
        for ot, net in zip(e.open_time, fwd.loc[e.index] - RT_BPS):
            closed.append((int(ot), float(net)))
    closed.sort()
    g14 = sum(n for t, n in closed if t >= now_ms - 14 * 86400000)
    g7 = sum(n for t, n in closed if t >= now_ms - 7 * 86400000)
    gate_open = (g14 > 0) and not (g7 < -3000)
    # 포지션 관리
    pos = slot.setdefault("positions", [])
    for p in [p for p in pos if now_ms >= p["close_ms"]]:
        px = float(frames[p["sym"]].close_price.iloc[-1]) if p["sym"] in frames else p["entry"]
        ret = G_LEV * (px / p["entry"] - 1) - RT_BPS / 10000 * G_LEV
        slot["equity"] = max(0.0, slot["equity"] * (1 + ret / SLOTS))
        trade({"slot": "g053", "event": "close", "sym": p["sym"], "entry": p["entry"], "exit": px,
               "ret_pct": round(ret / SLOTS * 100, 3), "equity": round(slot["equity"], 2)})
        pos.remove(p)
    if gate_open:
        held = {p["sym"] for p in pos}
        for sym, df in frames.items():
            if len(pos) >= SLOTS:
                break
            if sym in held or float(df.score.iloc[-1]) < THR:
                continue
            entry = float(df.close_price.iloc[-1])
            pos.append({"sym": sym, "entry": entry, "close_ms": now_ms + HOLD_H * 3600000})
            trade({"slot": "g053", "event": "open", "sym": sym, "price": entry,
                   "score": round(float(df.score.iloc[-1]), 1)})
    mark = slot["equity"]
    for p in pos:
        if p["sym"] in frames:
            px = float(frames[p["sym"]].close_price.iloc[-1])
            mark *= 1 + (G_LEV * (px / p["entry"] - 1)) / SLOTS
    slot["mark"] = round(mark, 2)
    log(f"g053 gate={'OPEN' if gate_open else 'CLOSED'} pos={len(pos)} equity={slot['equity']:.2f} mark={slot['mark']}")


def bg_refresh(symbol: str) -> None:
    path = ROOT / "data/candles_1h" / f"{symbol}.parquet"
    if not path.exists():
        return
    c = pd.read_parquet(path)
    url = f"https://api.bitget.com/api/v2/mix/market/candles?symbol={symbol}&productType=usdt-futures&granularity=1H&limit=200"
    try:
        raw = json.load(urllib.request.urlopen(url, timeout=15))["data"]
    except Exception:
        return
    n = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "vb", "vq"])
    n["ts"] = pd.to_datetime(n.ts.astype("int64"), unit="ms", utc=True)
    for col in ("open", "high", "low", "close"):
        n[col] = pd.to_numeric(n[col])
    merged = pd.concat([c, n[c.columns.intersection(n.columns)]]).drop_duplicates("ts").sort_values("ts")
    merged = merged[merged.ts < merged.ts.max()]
    merged.to_parquet(path, index=False)


def booster_tick(slot: dict) -> None:
    today = datetime.now(UTC).date().isoformat()
    mani = {m["symbol"]: m for m in json.loads((ROOT / "out/data_manifest_crypto.json").read_text())}
    uni = [s for s in mani if mani[s].get("tier") == "A" and mani[s].get("usdtVolume", 0) >= 10_000_000]
    if slot.get("last_rebalance") != today:
        for s in uni:
            bg_refresh(s)
            time.sleep(0.06)
    closes = {}
    for s in uni:
        c = pd.read_parquet(ROOT / "data/candles_1h" / f"{s}.parquet").set_index("ts")["close"].resample("1D").last()
        if len(c) >= 120:
            closes[s] = c
    panel = pd.DataFrame(closes).sort_index()
    rets = panel.pct_change()
    idx = (1 + rets.mean(axis=1)).cumprod()
    regime = bool((idx.iloc[-1] / idx.iloc[-21] - 1 > 0) and ((rets.rolling(20).mean() > 0).mean(axis=1).iloc[-1] >= 0.6))
    # 마크: 보유 중이면 전일 대비 수익 반영
    holdings = slot.get("holdings", {})
    if holdings:
        day_ret = sum(w * float(rets[s].iloc[-1]) for s, w in holdings.items() if s in rets and not np.isnan(rets[s].iloc[-1]))
        lev = slot.get("lev", 1.0)
        slot["equity"] = max(0.0, slot["equity"] * (1 + day_ret * lev)) if slot.get("marked_date") != today else slot["equity"]
    if slot.get("last_rebalance") != today:
        sig = panel.pct_change(LB).iloc[-1].dropna()
        raw_hist = (panel.pct_change() * 0).sum(axis=1)  # placeholder index
        if regime and len(sig) >= 2 * TOPK:
            w = {}
            for s in sig.nlargest(TOPK).index:
                w[s] = 1.0 / TOPK
            for s in sig.nsmallest(TOPK).index:
                w[s] = -1.0 / TOPK
            # 변동성 타겟: 최근 20일 전략 원수익 표준편차
            pos_hist = pd.DataFrame(0.0, index=panel.index[-60:], columns=panel.columns)
            for d in pos_hist.index:
                row = panel.pct_change(LB).loc[d].dropna()
                if len(row) < 2 * TOPK:
                    continue
                for s in row.nlargest(TOPK).index:
                    pos_hist.at[d, s] = 1.0 / TOPK
                for s in row.nsmallest(TOPK).index:
                    pos_hist.at[d, s] = -1.0 / TOPK
            raw = (pos_hist.shift(1) * rets.reindex(pos_hist.index)).sum(axis=1)
            realized = float(raw.rolling(20).std().iloc[-1]) or 0.03
            lev = float(np.clip(TV / realized, 0.3, MAX_LEV))
            slot.update(holdings=w, lev=lev, regime="ON")
            trade({"slot": "booster", "event": "rebalance", "regime": "ON", "lev": round(lev, 2),
                   "long": [s for s, x in w.items() if x > 0], "short": [s for s, x in w.items() if x < 0]})
        else:
            slot.update(holdings={}, lev=0.0, regime="OFF")
            trade({"slot": "booster", "event": "rebalance", "regime": "OFF"})
        slot["last_rebalance"] = today
    slot["marked_date"] = today
    slot["mark"] = round(slot["equity"], 2)
    log(f"booster regime={slot.get('regime')} lev={slot.get('lev', 0):.2f} equity={slot['equity']:.2f} holdings={len(slot.get('holdings', {}))}")


def tick() -> None:
    state = json.loads(STATE.read_text()) if STATE.exists() else {
        "started": datetime.now(UTC).isoformat(),
        "g053": {"equity": G_CAP, "positions": []},
        "booster": {"equity": B_CAP},
    }
    try:
        g053_tick(state["g053"])
    except Exception as err:  # noqa: BLE001
        log(f"g053 ERROR {err}")
    try:
        booster_tick(state["booster"])
    except Exception as err:  # noqa: BLE001
        log(f"booster ERROR {err}")
    state["total_mark"] = round(state["g053"].get("mark", state["g053"]["equity"]) +
                                state["booster"].get("mark", state["booster"]["equity"]), 2)
    state["updated"] = datetime.now(UTC).isoformat()
    STATE.write_text(json.dumps(state, indent=2))
    log(f"TOTAL mark={state['total_mark']} (시작 {G_CAP + B_CAP})")


def final_summary() -> None:
    if not STATE.exists():
        return
    d = json.loads(STATE.read_text())
    lines = ["# paper2 (G053+부스터) 24h 최종", "",
             f"기간: {d.get('started', '')} ~ {d.get('updated', '')}",
             f"총 마크: ${d.get('total_mark', 0)} / 시작 $200", "",
             f"- G053: equity {d['g053']['equity']} mark {d['g053'].get('mark')} 보유 {len(d['g053'].get('positions', []))}",
             f"- 부스터: equity {d['booster']['equity']} 국면 {d['booster'].get('regime')} lev {d['booster'].get('lev', 0)}"]
    (HERE / "PAPER2_FINAL.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    if os.environ.get("PAPER2_ONCE"):
        tick()
        return
    deadline = time.time() + RUN_HOURS * 3600
    while time.time() < deadline:
        try:
            tick()
        except Exception as err:  # noqa: BLE001
            log(f"TICK_ERROR {err}")
        nxt = (int(time.time() // 3600) + 1) * 3600 + 120
        time.sleep(max(60.0, min(nxt - time.time(), deadline - time.time() + 1)))
    final_summary()
    log("24h 종료 — PAPER2_FINAL.md 생성")


if __name__ == "__main__":
    main()
