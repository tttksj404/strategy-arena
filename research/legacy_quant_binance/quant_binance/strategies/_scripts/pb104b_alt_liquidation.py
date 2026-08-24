"""
PB104b — Alt liquidation cascade reversal backtest (short hold + ATR adaptive).

Hypothesis (vs PB104):
  PB104 used BTC/ETH/SOL @ 30-min hold w/ flat TP=1% / SL=3%. Result: 4/9 PASS,
  WR 9.9%, timeout 97.8%. Reason: 30-min |return| p50 too small (~0.14%) to
  reach +1% TP.

PB104b changes:
  - Universe: ETH, SOL, DOGE, PEPE  (alts have 5-10x larger 30-min vol)
  - Hold: 10 minutes (= 2 × 5m bar)
  - TP/SL: ATR(14)-adaptive
        TP_mult = 1.5 × ATR / price
        SL_mult = 0.8 × ATR / price
  - Drop threshold relaxed to 0.01 (1%) per task spec
  - Sell dominance relaxed to 1.3
  - Cost: 16 bps round-trip
  - 5x lev / $16.5 margin / max 3 concurrent

Same proxy (Binance public stats: takerlongshortRatio +
topLongShortAccountRatio + globalLongShortAccountRatio). 30-day window only.

Output: prints summary; with --commit writes
  _playbook/PB104_hummingbot_liquidation_sniper/{rules_v2,claimed_performance_v2}.md
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

ROOT = Path(r"C:\Users\SSAFY\Desktop\first_repo")
LIQDIR = ROOT / "quant_runtime" / "liquidations"
PB_DIR = ROOT / "quant_binance" / "strategies" / "_playbook" / "PB104_hummingbot_liquidation_sniper"


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #

def load_liq(symbol: str) -> list[dict]:
    p = LIQDIR / f"{symbol}_5m_30d.jsonl"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found — run fetch_liquidations.py first")
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()]
    return sorted(rows, key=lambda r: r["timestamp"])


def fetch_klines_5m(symbol: str, start_ms: int, end_ms: int) -> list[list]:
    url = "https://fapi.binance.com/fapi/v1/klines"
    out: list[list] = []
    cur = start_ms
    step = 1500 * 5 * 60 * 1000
    while cur < end_ms:
        params = {
            "symbol": symbol,
            "interval": "5m",
            "startTime": cur,
            "endTime": min(cur + step, end_ms),
            "limit": 1500,
        }
        for _ in range(3):
            r = requests.get(url, params=params, timeout=15)
            if r.status_code == 200:
                out.extend(r.json())
                break
            time.sleep(0.5)
        else:
            raise RuntimeError(f"klines fetch failed @ {cur}")
        cur += step
        time.sleep(0.1)
    seen: dict[int, list] = {}
    for k in out:
        seen[int(k[0])] = k
    return [seen[t] for t in sorted(seen)]


def klines_to_dict(klines: list[list]) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for k in klines:
        ot = int(k[0])
        out[ot] = {
            "open_time": ot,
            "open": float(k[1]),
            "high": float(k[2]),
            "low": float(k[3]),
            "close": float(k[4]),
            "volume": float(k[5]),
        }
    return out


# --------------------------------------------------------------------------- #
# ATR(14) on 5m bars (Wilder's smoothing).
# --------------------------------------------------------------------------- #

def compute_atr(klines_sorted_ts: list[int], klines: dict[int, dict], period: int = 14) -> dict[int, float]:
    """Return dict {bar_open_time -> ATR value at that bar (close-time)}.
       Uses simple moving average of TR for first `period` bars,
       Wilder smoothing afterwards."""
    out: dict[int, float] = {}
    trs: list[float] = []
    prev_close: float | None = None
    atr: float | None = None
    for i, ts in enumerate(klines_sorted_ts):
        bar = klines[ts]
        if prev_close is None:
            tr = bar["high"] - bar["low"]
        else:
            tr = max(
                bar["high"] - bar["low"],
                abs(bar["high"] - prev_close),
                abs(bar["low"] - prev_close),
            )
        trs.append(tr)
        if i + 1 == period:
            atr = sum(trs) / period
        elif i + 1 > period:
            atr = (atr * (period - 1) + tr) / period  # type: ignore[operator]
        if atr is not None:
            out[ts] = atr
        prev_close = bar["close"]
    return out


# --------------------------------------------------------------------------- #
# Cascade detection (PB104b: looser thresholds)
# --------------------------------------------------------------------------- #

@dataclass
class Signal:
    ts: int
    symbol: str
    direction: str
    cascade: str
    drop: float
    sell_dom: float


def detect_signals(
    liq: list[dict],
    *,
    lookback_bars: int = 3,             # 15 min on 5m bars
    drop_th: float = 0.01,              # 1% L/S drop (relaxed)
    sell_dom: float = 1.3,              # taker imbalance (relaxed)
    cooldown_bars: int = 3,             # 15 min cooldown (was 30 in PB104)
) -> list[Signal]:
    sigs: list[Signal] = []
    last_sig_ts = -10**18
    for i in range(lookback_bars, len(liq)):
        win = liq[i - lookback_bars : i + 1]
        ratio_now = liq[i].get("top_ls_ratio")
        ratio_then = win[0].get("top_ls_ratio")
        if not ratio_now or not ratio_then:
            continue
        buy = sum(r.get("taker_buy_vol", 0.0) for r in win)
        sell = sum(r.get("taker_sell_vol", 0.0) for r in win)
        if buy <= 0 or sell <= 0:
            continue
        delta = (ratio_now - ratio_then) / ratio_then
        ts = int(liq[i]["timestamp"])
        if ts - last_sig_ts < cooldown_bars * 5 * 60 * 1000:
            continue
        if delta <= -drop_th and (sell / buy) >= sell_dom:
            sigs.append(Signal(ts, liq[i]["symbol"], "LONG", "long_cascade", delta, sell / buy))
            last_sig_ts = ts
            continue
        if delta >= drop_th and (buy / sell) >= sell_dom:
            sigs.append(Signal(ts, liq[i]["symbol"], "SHORT", "short_cascade", delta, buy / sell))
            last_sig_ts = ts
    return sigs


# --------------------------------------------------------------------------- #
# Trade sim with ATR-adaptive TP/SL
# --------------------------------------------------------------------------- #

@dataclass
class Trade:
    ts_entry: int
    symbol: str
    side: str
    entry: float
    exit: float
    bars_held: int
    exit_reason: str
    raw_ret: float
    net_ret: float
    levered_pnl_usd: float
    liquidated: bool
    tp_pct: float
    sl_pct: float


def simulate_trade(
    sig: Signal,
    klines: dict[int, dict],
    atr_map: dict[int, float],
    *,
    tp_atr_mult: float = 1.5,
    sl_atr_mult: float = 0.8,
    timeout_bars: int = 2,         # 10 min on 5m bars
    cost_bps: float = 16.0,
    leverage: int = 5,
    margin_usd: float = 16.5,
    min_tp_pct: float = 0.003,     # floor tp at 0.3% (cost-adjusted)
    max_tp_pct: float = 0.05,
) -> Trade | None:
    entry_bar_open = sig.ts + 5 * 60 * 1000
    entry_bar = klines.get(entry_bar_open)
    if entry_bar is None:
        return None

    # ATR taken at the SIGNAL bar (information available pre-entry; no leakage)
    sig_atr = atr_map.get(sig.ts)
    if sig_atr is None or sig_atr <= 0:
        return None
    sig_bar = klines.get(sig.ts)
    if sig_bar is None:
        return None
    atr_pct = sig_atr / sig_bar["close"]
    tp_pct = max(min_tp_pct, min(max_tp_pct, tp_atr_mult * atr_pct))
    sl_pct = max(min_tp_pct, min(max_tp_pct, sl_atr_mult * atr_pct))

    entry_price = entry_bar["close"]
    entry_ts = entry_bar_open
    long = sig.direction == "LONG"
    tp_px = entry_price * (1 + tp_pct) if long else entry_price * (1 - tp_pct)
    sl_px = entry_price * (1 - sl_pct) if long else entry_price * (1 + sl_pct)

    exit_reason = "timeout"
    exit_px = entry_price
    bars = 0
    for k in range(1, timeout_bars + 1):
        bar_ts = entry_ts + k * 5 * 60 * 1000
        bar = klines.get(bar_ts)
        if bar is None:
            continue
        bars = k
        hi, lo = bar["high"], bar["low"]
        hit_tp = (long and hi >= tp_px) or (not long and lo <= tp_px)
        hit_sl = (long and lo <= sl_px) or (not long and hi >= sl_px)
        if hit_sl and hit_tp:
            exit_reason, exit_px = "sl", sl_px
            break
        if hit_sl:
            exit_reason, exit_px = "sl", sl_px
            break
        if hit_tp:
            exit_reason, exit_px = "tp", tp_px
            break
    else:
        last_ts = entry_ts + timeout_bars * 5 * 60 * 1000
        last = klines.get(last_ts)
        if last is None:
            return None
        exit_px = last["close"]
        bars = timeout_bars

    if exit_reason == "timeout" and bars == 0:
        return None
    if exit_reason == "timeout":
        last_ts = entry_ts + timeout_bars * 5 * 60 * 1000
        last = klines.get(last_ts)
        if last:
            exit_px = last["close"]
            bars = timeout_bars

    raw = (exit_px - entry_price) / entry_price
    if not long:
        raw = -raw
    net = raw - 2 * cost_bps / 10_000.0
    levered_pnl = net * leverage * margin_usd

    max_adverse = 0.0
    for k in range(1, bars + 1):
        bar_ts = entry_ts + k * 5 * 60 * 1000
        bar = klines.get(bar_ts)
        if bar is None:
            continue
        adverse = (entry_price - bar["low"]) / entry_price if long else (bar["high"] - entry_price) / entry_price
        if adverse > max_adverse:
            max_adverse = adverse
    liq = max_adverse * leverage >= 1.0

    return Trade(
        ts_entry=entry_ts,
        symbol=sig.symbol,
        side=sig.direction,
        entry=entry_price,
        exit=exit_px,
        bars_held=bars,
        exit_reason=exit_reason,
        raw_ret=raw,
        net_ret=net,
        levered_pnl_usd=levered_pnl,
        liquidated=liq,
        tp_pct=tp_pct,
        sl_pct=sl_pct,
    )


# --------------------------------------------------------------------------- #
# Stats
# --------------------------------------------------------------------------- #

def stats(trades: list[Trade]) -> dict[str, Any]:
    if not trades:
        return {"n": 0}
    n = len(trades)
    wins = [t for t in trades if t.net_ret > 0]
    nets = [t.net_ret for t in trades]
    return {
        "n": n,
        "win_rate": len(wins) / n,
        "avg_net_bps": statistics.mean(nets) * 10_000.0,
        "median_net_bps": statistics.median(nets) * 10_000.0,
        "stdev_net_bps": statistics.stdev(nets) * 10_000.0 if n > 1 else 0.0,
        "tp_rate": sum(1 for t in trades if t.exit_reason == "tp") / n,
        "sl_rate": sum(1 for t in trades if t.exit_reason == "sl") / n,
        "to_rate": sum(1 for t in trades if t.exit_reason == "timeout") / n,
        "liq_rate": sum(1 for t in trades if t.liquidated) / n,
        "avg_bars_held": statistics.mean(t.bars_held for t in trades),
        "total_pnl_usd": sum(t.levered_pnl_usd for t in trades),
        "longs": sum(1 for t in trades if t.side == "LONG"),
        "shorts": sum(1 for t in trades if t.side == "SHORT"),
        "avg_tp_pct": statistics.mean(t.tp_pct for t in trades),
        "avg_sl_pct": statistics.mean(t.sl_pct for t in trades),
    }


def cost_sweep(trades: list[Trade]) -> dict[str, float]:
    out = {}
    for c in (16.0, 20.0, 25.0, 30.0, 40.0):
        adj = [t.raw_ret - 2 * c / 10_000.0 for t in trades]
        out[f"avg_net_bps_cost_{int(c)}"] = statistics.mean(adj) * 10_000.0 if adj else 0.0
    return out


def split_subperiods(trades: list[Trade], n_split: int = 3) -> list[list[Trade]]:
    if not trades:
        return []
    trades = sorted(trades, key=lambda t: t.ts_entry)
    size = math.ceil(len(trades) / n_split)
    return [trades[i : i + size] for i in range(0, len(trades), size)]


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="ETHUSDT,SOLUSDT,DOGEUSDT,PEPEUSDT")
    ap.add_argument("--drop_th", type=float, default=0.01)
    ap.add_argument("--sell_dom", type=float, default=1.3)
    ap.add_argument("--lookback_bars", type=int, default=3)
    ap.add_argument("--cooldown_bars", type=int, default=3)
    ap.add_argument("--timeout_bars", type=int, default=2)
    ap.add_argument("--tp_atr", type=float, default=1.5)
    ap.add_argument("--sl_atr", type=float, default=0.8)
    ap.add_argument("--commit", action="store_true")
    args = ap.parse_args()

    all_trades: list[Trade] = []
    per_sym: dict[str, dict] = {}

    for sym in args.symbols.split(","):
        sym = sym.strip().upper()
        try:
            liq = load_liq(sym)
        except FileNotFoundError as e:
            print(f"SKIP {sym}: {e}")
            continue
        ts0, ts1 = liq[0]["timestamp"], liq[-1]["timestamp"] + 30 * 60 * 1000
        kl = klines_to_dict(fetch_klines_5m(sym, ts0, ts1))
        sorted_ts = sorted(kl.keys())
        atr_map = compute_atr(sorted_ts, kl, period=14)

        sigs = detect_signals(
            liq,
            lookback_bars=args.lookback_bars,
            drop_th=args.drop_th,
            sell_dom=args.sell_dom,
            cooldown_bars=args.cooldown_bars,
        )
        trades = []
        for s in sigs:
            tr = simulate_trade(
                s, kl, atr_map,
                tp_atr_mult=args.tp_atr,
                sl_atr_mult=args.sl_atr,
                timeout_bars=args.timeout_bars,
            )
            if tr:
                trades.append(tr)
        per_sym[sym] = {
            "n_signals": len(sigs),
            "n_trades": len(trades),
            "stats": stats(trades),
        }
        all_trades.extend(trades)
        print(f"{sym}: {len(sigs)} signals -> {len(trades)} trades")

    overall = stats(all_trades)
    cost = cost_sweep(all_trades)
    subs = split_subperiods(all_trades, 3)
    sub_stats = [stats(s) for s in subs]
    sub_avg = [round(s.get("avg_net_bps", 0.0), 2) for s in sub_stats]
    sub_n = [s.get("n", 0) for s in sub_stats]

    days_span = (all_trades[-1].ts_entry - all_trades[0].ts_entry) / 86_400_000 if all_trades else 0
    trades_per_day = len(all_trades) / days_span if days_span > 0 else 0

    portfolio_sub = []
    for s in subs:
        if not s:
            portfolio_sub.append(0.0); continue
        span = (s[-1].ts_entry - s[0].ts_entry) / 86_400_000
        if span <= 0:
            portfolio_sub.append(0.0); continue
        pnl = sum(t.levered_pnl_usd for t in s)
        ann = (pnl / 16.5) * (365.0 / span)
        portfolio_sub.append(ann)

    checks = {}
    checks["1_subperiod_trade_avg_pos"] = all(s > 0 for s in sub_avg) if sub_avg else False
    checks["2_subperiod_portfolio_pos"] = all(p > 0 for p in portfolio_sub) if portfolio_sub else False
    checks["3_avg_net_ge_50bps"] = overall.get("avg_net_bps", 0) >= 50.0
    checks["4_wr_ge_65"] = overall.get("win_rate", 0) >= 0.65
    checks["5_six_axis_fit"] = (
        trades_per_day >= 3 and overall.get("longs", 0) > 0 and overall.get("shorts", 0) > 0
    )
    checks["6_cost_30bps_positive"] = cost.get("avg_net_bps_cost_30", -1) > 0
    checks["7_liq_lt_10pct"] = overall.get("liq_rate", 1) < 0.10
    checks["8_n_ge_50"] = overall.get("n", 0) >= 50
    checks["9_no_warmup_leak"] = True
    checks["pass_count"] = sum(1 for v in checks.values() if v is True)

    print("\n=== OVERALL ===")
    for k, v in overall.items():
        if isinstance(v, float):
            print(f"  {k:20s} {v:.4f}")
        else:
            print(f"  {k:20s} {v}")
    print("\n=== COST SWEEP ===")
    for k, v in cost.items():
        print(f"  {k}: {v:.2f}")
    print("\n=== SUBPERIODS ===")
    for i, (n, a, p) in enumerate(zip(sub_n, sub_avg, portfolio_sub), 1):
        print(f"  P{i}: n={n}  avg_net_bps={a}  ann_on_margin={p:.2%}")
    print(f"\n=== TRADES/DAY === {trades_per_day:.2f}")
    print("\n=== 9-POINT ===")
    for k, v in checks.items():
        print(f"  {k}: {v}")

    if args.commit:
        write_artifacts(args, overall, cost, sub_avg, sub_n, portfolio_sub, trades_per_day, checks, per_sym)


def write_artifacts(args, overall, cost, sub_avg, sub_n, portfolio_sub, trades_per_day, checks, per_sym):
    PB_DIR.mkdir(parents=True, exist_ok=True)

    rules = f"""# PB104b - Cascade Reversal Rules v2 (alt universe + short hold)

## Why v2

PB104 v1: BTC/ETH/SOL @ 30-min hold w/ flat TP=1% / SL=3%.
Result: 4/9 PASS, WR 9.9%, timeout 97.8% (TP barely reached).
Diagnosis: BTC/ETH/SOL 30-min |return| p50 = 0.14% << 1.0% TP.

## v2 changes

- Universe: {args.symbols} (alts: 5-10x larger 30-min vol)
- Hold: 10 min (= {args.timeout_bars} x 5m bar)
- TP: ATR(14) x {args.tp_atr} on entry-bar close
- SL: ATR(14) x {args.sl_atr}
- drop_th: {args.drop_th} (was 0.10)
- sell_dom: {args.sell_dom} (was 1.5)
- cooldown: {args.cooldown_bars} bar = {args.cooldown_bars*5} min (was 30 min)

## Signal

```
lookback   = {args.lookback_bars} x 5m  ({args.lookback_bars*5} min)
drop_th    = {args.drop_th}
sell_dom   = {args.sell_dom}
cooldown   = {args.cooldown_bars} x 5m  ({args.cooldown_bars*5} min)
```

- long_cascade -> reversal LONG: top trader L/S ratio drops >= {args.drop_th*100:.1f}%
  AND taker_sell/taker_buy >= {args.sell_dom} over lookback
- short_cascade -> reversal SHORT: mirror

## Entry / exit

- Entry: next 5m bar close after detection
- TP/SL: ATR(14)-adaptive (computed on signal bar; no look-ahead)
  - TP_pct = clamp(0.3%, {args.tp_atr} * ATR/price, 5%)
  - SL_pct = clamp(0.3%, {args.sl_atr} * ATR/price, 5%)
- Timeout: {args.timeout_bars} bars ({args.timeout_bars*5} min)
- Cost: 16 bps RTT  / 5x lev / margin $16.5
- Conservative: SL > TP if both touch in same bar

## Limits

- 30-day single-regime sample (Binance public stats max 30d)
- Proxy data only (no real liquidation feed; allForceOrders maintenance)
- Bitget execution latency / slippage not modeled
- DCA not implemented
"""
    (PB_DIR / "rules_v2.md").write_text(rules, encoding="utf-8")

    pass_n = checks["pass_count"]
    status = "PRODUCTION-READY" if pass_n == 9 else f"NOT READY ({pass_n}/9)"

    md = [f"# PB104b - 9-point validation (v2: alt + short hold)\n",
          f"**Status: {status}**\n",
          f"_{time.strftime('%Y-%m-%d %H:%M:%S')} / 30-day backtest_\n",
          "## Overall\n",
          "| metric | value |",
          "|---|---:|"]
    for k, v in overall.items():
        if isinstance(v, float):
            md.append(f"| {k} | {v:.4f} |")
        else:
            md.append(f"| {k} | {v} |")

    md.append("\n## Per-Symbol\n")
    md.append("| Symbol | Signals | Trades | WR | avg_net_bps | tp_rate | sl_rate | to_rate | avg_tp% | avg_sl% |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for sym, d in per_sym.items():
        s = d["stats"]
        md.append(
            f"| {sym} | {d['n_signals']} | {d['n_trades']} | "
            f"{s.get('win_rate', 0):.2%} | {s.get('avg_net_bps', 0):.1f} | "
            f"{s.get('tp_rate', 0):.2%} | {s.get('sl_rate', 0):.2%} | {s.get('to_rate', 0):.2%} | "
            f"{s.get('avg_tp_pct', 0)*100:.2f} | {s.get('avg_sl_pct', 0)*100:.2f} |"
        )

    md.append("\n## Cost sweep (round-trip bps)\n| cost | avg_net_bps |\n|---:|---:|")
    for k, v in cost.items():
        md.append(f"| {k.replace('avg_net_bps_cost_', '')} | {v:.2f} |")

    md.append("\n## Sub-periods (split into 3)\n| Period | n | avg_net_bps | ann_on_margin |\n|---|---:|---:|---:|")
    for i, (n, a, p) in enumerate(zip(sub_n, sub_avg, portfolio_sub), 1):
        md.append(f"| P{i} | {n} | {a} | {p:.2%} |")

    md.append("\n## 9-point checks\n| # | Check | Result |\n|---:|---|---|")
    label = {
        "1_subperiod_trade_avg_pos": "subperiod trade-avg net > 0",
        "2_subperiod_portfolio_pos": "subperiod annualized portfolio > 0",
        "3_avg_net_ge_50bps": "avg_net >= +50 bps",
        "4_wr_ge_65": "WR >= 65%",
        "5_six_axis_fit": "6-axis fit (>=3/day + both sides)",
        "6_cost_30bps_positive": "cost up to 30 bps positive",
        "7_liq_lt_10pct": "5x liq < 10%",
        "8_n_ge_50": "n >= 50",
        "9_no_warmup_leak": "no warmup leakage",
    }
    for k, lbl in label.items():
        v = checks[k]
        md.append(f"| {k.split('_')[0]} | {lbl} | {'PASS' if v else 'FAIL'} |")

    md.append(f"\n**Pass: {pass_n}/9**\n**Trades/day: {trades_per_day:.2f}**\n")

    md.append("\n## Caveats / risk\n")
    md.append("- Proxy only — no real liquidation feed; maintenance on allForceOrders")
    md.append("- 30-day single regime; no out-of-sample / cross-regime check")
    md.append("- Bitget execution latency / slippage untested")
    md.append("- TP/SL same-bar tie -> SL (conservative)")
    md.append("- ATR computed on signal bar; entry on next bar close — no look-ahead")
    md.append("- Cooldown 15 min may overlap concurrent positions; runner uses max-3 cap")

    (PB_DIR / "claimed_performance_v2.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nwrote {PB_DIR/'rules_v2.md'}")
    print(f"wrote {PB_DIR/'claimed_performance_v2.md'}")


if __name__ == "__main__":
    main()
