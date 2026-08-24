"""All-weather strategy search — survive all regimes, maximize risk-adjusted profit.

NOT optimized for win rate. Ranked by:
  1. Calmar ratio (total return / max drawdown)
  2. Profit per trade consistency across regimes
  3. Total absolute profit

Includes:
  - Long AND short
  - All market regimes (bull/bear/sideways)
  - High leverage (5x, 10x, 15x, 20x)
  - All TP/SL/trailing combos
  - Per-regime breakdown to ensure no single regime destroys profits
"""
from __future__ import annotations

import bisect
import json
import os
import sys
import random
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _sim(side, ep, bars, stop_bps, max_bars, tp_type, tp_bps, trail_pct, partial_levels, cost_bps):
    if not bars or ep <= 0:
        return -cost_bps, 0, "NO_DATA"
    peak = 0.0; stop = -stop_bps; realized = 0.0; remain = 1.0; pidx = 0; close_pnl = 0.0
    for i, bar in enumerate(bars[:max_bars]):
        if side == "long":
            hi = ((bar.high_price / ep) - 1) * 10000
            lo = ((bar.low_price / ep) - 1) * 10000
            close_pnl = ((bar.close_price / ep) - 1) * 10000
        else:
            hi = (1 - (bar.low_price / ep)) * 10000
            lo = (1 - (bar.high_price / ep)) * 10000
            close_pnl = (1 - (bar.close_price / ep)) * 10000
        peak = max(peak, hi)
        if lo <= stop:
            return (stop * remain + realized) - cost_bps, i+1, "SL"
        if tp_type == "fixed":
            if tp_bps > 0 and hi >= tp_bps:
                return (tp_bps * remain + realized) - cost_bps, i+1, "TP"
        elif tp_type == "trailing":
            if peak > 0 and trail_pct > 0:
                ns = peak * (1 - trail_pct)
                if ns > stop: stop = ns
                if stop > 0 and close_pnl <= stop:
                    return (stop * remain + realized) - cost_bps, i+1, "TRAIL"
        elif tp_type == "partial_ladder":
            rr = stop_bps; levels = partial_levels or (0.5, 1.0)
            if pidx < len(levels) and hi >= levels[pidx] * rr:
                frac = min(0.33, remain); realized += hi * frac; remain -= frac; pidx += 1
                if pidx == 1: stop = max(stop, 0)
            if pidx >= len(levels) and peak > 0:
                stop = max(stop, peak - peak * 0.6)
                if close_pnl <= stop:
                    return (stop * remain + realized) - cost_bps, i+1, "PARTIAL_TRAIL"
    return (close_pnl * remain + realized) - cost_bps, min(len(bars), max_bars), "HOLD"


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="quant_binance/config.example.json")
    parser.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT")
    parser.add_argument("--output-base", default="quant_runtime")
    parser.add_argument("--equity-usd", type=float, default=66.0)
    parser.add_argument("--cost-bps", type=float, default=16.0)
    parser.add_argument("--min-trades", type=int, default=30)
    parser.add_argument("--top-n", type=int, default=30)
    args = parser.parse_args(argv)

    symbols = [s.strip() for s in args.symbols.split(",")]
    data_dir = Path(args.output_base) / "historical"
    os.environ.setdefault("STRATEGY_OVERRIDE_PATH", str(Path(args.output_base) / "artifacts" / "strategy_override.approved.json"))
    os.environ.setdefault("STRATEGY_PROFILE", "live-ultra-aggressive")
    os.environ.setdefault("EXCHANGE", "bitget")

    from quant_binance.settings import Settings
    from quant_binance.features.extractor import MarketFeatureExtractor
    from quant_binance.cost_calibration import load_cost_calibration
    from quant_binance.backtest.historical_fixture_builder import build_historical_slices, _kline_bars_from_raw
    from quant_binance.data.historical_download import load_historical_klines
    from quant_binance.data.rest_seed import _parse_kline
    from quant_binance.service import PaperTradingService
    from quant_binance.execution.router import ExecutionRouter

    settings = Settings.load(args.config)
    cal_path = Path(args.output_base) / "artifacts" / "cost_calibration.json"
    extractor = MarketFeatureExtractor(settings, cost_calibration=load_cost_calibration(str(cal_path)))
    service = PaperTradingService(settings, router=ExecutionRouter())

    print(f"[allweather] Loading data for {symbols}...")
    all_entries = []
    bars_5m_by_sym = {}

    for symbol in symbols:
        k5m = load_historical_klines(data_dir=data_dir, symbol=symbol, interval="5m")
        k1h = load_historical_klines(data_dir=data_dir, symbol=symbol, interval="1h")
        k4h = load_historical_klines(data_dir=data_dir, symbol=symbol, interval="4h")
        if not k1h or not k5m: continue

        parsed = []
        for row in k5m:
            try: parsed.append(_parse_kline(symbol, "5m", row))
            except: continue
        parsed.sort(key=lambda b: b.close_time)
        bars_5m_by_sym[symbol] = parsed

        slices = build_historical_slices(
            symbol=symbol, klines_5m=k5m, klines_1h=k1h, klines_4h=k4h or [],
            settings=settings, extractor=extractor,
        )

        for sl in slices:
            try:
                d = service.run_cycle(
                    state=sl.state, primitive_inputs=sl.primitive_inputs,
                    history=sl.history, decision_time=sl.decision_time,
                    equity_usd=args.equity_usd,
                    remaining_portfolio_capacity_usd=args.equity_usd * 2.5,
                )
            except: continue
            if d.final_mode not in ("spot", "futures"): continue
            ep = sl.state.last_trade_price
            if ep <= 0: continue

            adx = getattr(sl.primitive_inputs, 'adx_1h', 0.0)
            atr = getattr(sl.primitive_inputs, 'atr_1h', 0.0)
            atr_bps = (atr / ep) * 10000 if atr > 0 else 50.0
            ema = getattr(sl.primitive_inputs, 'ema_cross_signal', 0)
            trend_dir = getattr(sl.primitive_inputs, 'trend_direction', 0)

            # Regime
            bars_1h = sl.state.klines.get("1h", [])
            regime = "sideways"
            if len(bars_1h) >= 48:
                c0, c1 = bars_1h[-48].close_price, bars_1h[-1].close_price
                if c0 > 0:
                    chg = ((c1/c0)-1)*100
                    regime = "bull" if chg > 3 else ("bear" if chg < -3 else "sideways")

            all_entries.append({
                "symbol": symbol, "side": d.side, "score": d.predictability_score,
                "adx": adx, "entry_price": ep, "atr_bps": atr_bps,
                "entry_ms": int(sl.decision_time.timestamp() * 1000),
                "ema_cross": ema, "trend_dir": trend_dir, "regime": regime,
            })

        print(f"  {symbol}: {len([e for e in all_entries if e['symbol']==symbol])} entries")

    all_entries.sort(key=lambda x: x["entry_ms"])
    print(f"  Total: {len(all_entries)} entries")

    # Pre-compute 5m indices
    sym_ts = {s: [int(b.close_time.timestamp()*1000) for b in bars] for s, bars in bars_5m_by_sym.items()}

    # Pre-compute future bars
    future_bars = []
    for feat in all_entries:
        ts = sym_ts.get(feat["symbol"])
        bars = bars_5m_by_sym.get(feat["symbol"])
        if not ts or not bars:
            future_bars.append(None); continue
        idx = bisect.bisect_right(ts, feat["entry_ms"])
        future_bars.append(bars[idx:] if len(bars) - idx >= 3 else None)

    # Grid
    tp_specs = [
        ("partial_ladder", 0.0, 0.0, (0.5, 1.0)),
        ("partial_ladder", 0.0, 0.0, (0.5, 1.5)),
        ("partial_ladder", 0.0, 0.0, (1.0, 2.0)),
        ("trailing", 0.0, 0.3, ()),
        ("trailing", 0.0, 0.5, ()),
        ("trailing", 0.0, 0.7, ()),
        ("fixed", 0.5, 0.0, ()),
        ("fixed", 1.0, 0.0, ()),
        ("fixed", 1.5, 0.0, ()),
        ("fixed", 2.0, 0.0, ()),
    ]
    sl_mults = [1.0, 1.5, 2.0, 3.0, 4.0]
    holds = [6, 12, 24, 48]
    # BOTH sides tested
    side_filters = ["long", "short", "both"]
    adx_mins = [20, 25, 30, 35, 40]
    score_mins = [55, 60, 65, 70, 75, 80]
    leverages = [5, 10, 15, 20]

    # Pre-compute sims
    print(f"[allweather] Pre-computing simulations...")
    unique_params = set()
    for tp_type, tp_r, trail, partial in tp_specs:
        for sl in sl_mults:
            for hold in holds:
                unique_params.add((tp_type, tp_r, sl, trail, partial, hold))

    sim_cache = {}
    for si, feat in enumerate(all_entries):
        fb = future_bars[si]
        if fb is None or feat["side"] == "flat": continue
        ep = feat["entry_price"]
        atr_bps = feat["atr_bps"] if feat["atr_bps"] > 0 else 50.0
        for tp_type, tp_r, sl, trail, partial, hold in unique_params:
            stop = max(atr_bps * sl, 15.0)
            tp_bps = stop * tp_r if tp_type == "fixed" else 0.0
            key = (si, tp_type, tp_r, sl, trail, partial, hold)
            sim_cache[key] = _sim(feat["side"], ep, fb, stop, hold*12, tp_type, tp_bps, trail, partial, args.cost_bps)
        if (si+1) % 3000 == 0:
            print(f"  cached {si+1}/{len(all_entries)}...", flush=True)
    print(f"  Cache: {len(sim_cache):,} sims")

    # Score configs
    total = len(tp_specs) * len(sl_mults) * len(holds) * len(side_filters) * len(adx_mins) * len(score_mins) * len(leverages)
    print(f"\n[allweather] Scoring {total:,} configs...")

    results = []
    ci = 0
    for tp_type, tp_r, trail, partial in tp_specs:
        for sl in sl_mults:
            for hold in holds:
                for side_f in side_filters:
                    for adx in adx_mins:
                        for score in score_mins:
                            for lev in leverages:
                                ci += 1
                                pnls = []
                                regime_pnls = defaultdict(list)
                                sym_pnls = defaultdict(list)

                                for si, feat in enumerate(all_entries):
                                    if feat["score"] < score: continue
                                    if feat["adx"] < adx: continue
                                    side = feat["side"]
                                    if side == "flat": continue
                                    if side_f == "long" and side != "long": continue
                                    if side_f == "short" and side != "short": continue
                                    if future_bars[si] is None: continue

                                    key = (si, tp_type, tp_r, sl, trail, partial, hold)
                                    cached = sim_cache.get(key)
                                    if not cached: continue
                                    p = cached[0] * lev
                                    pnls.append(p)
                                    regime_pnls[feat["regime"]].append(p)
                                    sym_pnls[feat["symbol"]].append(p)

                                if len(pnls) < args.min_trades: continue

                                wins = sum(1 for p in pnls if p > 0)
                                wr = wins / len(pnls)
                                total_pnl = sum(pnls)
                                gw = sum(p for p in pnls if p > 0)
                                gl = abs(sum(p for p in pnls if p <= 0)) or 0.01
                                pf = gw / gl
                                avg = total_pnl / len(pnls)

                                # Max drawdown
                                eq = 0.0; peak_eq = 0.0; mdd = 0.0
                                for p in pnls:
                                    eq += p; peak_eq = max(peak_eq, eq)
                                    mdd = max(mdd, peak_eq - eq)

                                # Calmar ratio (annualized return / max DD)
                                calmar = total_pnl / mdd if mdd > 0 else (999 if total_pnl > 0 else -999)

                                # Regime resilience: worst regime avg PnL
                                worst_regime_avg = 0.0
                                regime_stats = {}
                                for reg in ["bull", "bear", "sideways"]:
                                    rp = regime_pnls.get(reg, [])
                                    if rp:
                                        ravg = sum(rp) / len(rp)
                                        regime_stats[reg] = {"trades": len(rp), "wr": round(sum(1 for p in rp if p>0)/len(rp), 4), "avg": round(ravg, 1), "total": round(sum(rp), 1)}
                                        if ravg < worst_regime_avg:
                                            worst_regime_avg = ravg

                                # Composite score: calmar + regime resilience
                                regime_penalty = abs(worst_regime_avg) / 10 if worst_regime_avg < 0 else 0
                                composite = calmar * 10 + total_pnl / 1000 - regime_penalty + min(len(pnls)/10, 20)

                                sym_stats = {}
                                for sym, sp in sym_pnls.items():
                                    sw = sum(1 for p in sp if p > 0)
                                    sym_stats[sym] = {"trades": len(sp), "wr": round(sw/len(sp), 4), "total": round(sum(sp), 1)}

                                results.append({
                                    "composite": round(composite, 2),
                                    "calmar": round(calmar, 2),
                                    "wr": round(wr, 4), "pf": round(pf, 2),
                                    "trades": len(pnls), "total_bps": round(total_pnl, 1),
                                    "avg_bps": round(avg, 1), "mdd": round(mdd, 1),
                                    "tp_type": tp_type, "tp_r": tp_r, "sl": sl,
                                    "trail": trail, "partial": list(partial),
                                    "hold": hold, "side": side_f, "adx": adx,
                                    "score": score, "lev": lev,
                                    "regime": regime_stats, "sym": sym_stats,
                                })

                    if ci % 50000 == 0:
                        print(f"  {ci:,}/{total:,}...", flush=True)

    # Sort by composite
    results.sort(key=lambda r: r["composite"], reverse=True)

    # Print
    print(f"\n{'='*130}")
    print(f"  ALL-WEATHER STRATEGY SEARCH — TOP {args.top_n} BY RISK-ADJUSTED RETURN")
    print(f"  Ranked by: Calmar ratio + total profit - regime penalty + trade count bonus")
    print(f"{'='*130}")

    print(f"\n{'#':>3} {'Calmar':>7} {'WR':>6} {'PF':>5} {'Trades':>6} {'AvgPnL':>8} {'TotalBps':>10} {'MaxDD':>8} | "
          f"{'TP':<8} {'SL':>4} {'Hold':>5} {'Side':<6} {'ADX':>4} {'Score':>5} {'Lev':>3} | "
          f"{'Bull':>12} {'Bear':>12} {'Side':>12}")
    print("-" * 155)

    for i, r in enumerate(results[:args.top_n]):
        bull = r["regime"].get("bull", {})
        bear = r["regime"].get("bear", {})
        side = r["regime"].get("sideways", {})
        b_str = f"{bull.get('wr',0)*100:.0f}%/{bull.get('trades',0)}" if bull else "N/A"
        br_str = f"{bear.get('wr',0)*100:.0f}%/{bear.get('trades',0)}" if bear else "N/A"
        s_str = f"{side.get('wr',0)*100:.0f}%/{side.get('trades',0)}" if side else "N/A"

        print(f"{i+1:>3} {r['calmar']:>7.2f} {r['wr']*100:>5.1f}% {r['pf']:>5.2f} {r['trades']:>6} "
              f"{r['avg_bps']:>+7.1f} {r['total_bps']:>+9.1f} {r['mdd']:>7.0f} | "
              f"{r['tp_type']:<8} {r['sl']:>4.1f} {r['hold']:>4}h {r['side']:<6} {r['adx']:>4.0f} {r['score']:>5.0f} {r['lev']:>3} | "
              f"{b_str:>12} {br_str:>12} {s_str:>12}")

    # Top 5 detail
    print(f"\n{'='*130}")
    print(f"  TOP 5 DETAILED")
    print(f"{'='*130}")
    for i, r in enumerate(results[:5]):
        print(f"\n  #{i+1}: Calmar={r['calmar']:.2f} | {r['tp_type']} SL={r['sl']}ATR hold={r['hold']}h {r['side']} ADX>={r['adx']} Score>={r['score']} Lev={r['lev']}x")
        print(f"      {r['trades']}건 WR={r['wr']*100:.1f}% PF={r['pf']:.2f} Avg={r['avg_bps']:+.1f} Total={r['total_bps']:+.1f} MaxDD={r['mdd']:.0f}")
        print(f"      Regimes:")
        for reg in ["bull", "bear", "sideways"]:
            rs = r["regime"].get(reg, {})
            if rs:
                print(f"        {reg:<10} {rs['trades']:>4}건 WR={rs['wr']*100:.1f}% avg={rs['avg']:+.1f} total={rs['total']:+.1f}")
        print(f"      Coins:")
        for sym in sorted(r.get("sym", {}).keys()):
            s = r["sym"][sym]
            print(f"        {sym:<12} {s['trades']:>4}건 WR={s['wr']*100:.1f}% total={s['total']:+.1f}")

    # Stats
    print(f"\n  전체 viable: {len(results):,}개")
    both_profit = [r for r in results if all(r["regime"].get(reg, {}).get("total", 0) > 0 for reg in ["bull", "bear", "sideways"] if r["regime"].get(reg))]
    print(f"  모든 구간 수익: {len(both_profit):,}개")
    short_ok = [r for r in results if r["side"] in ("short", "both") and r["total_bps"] > 0]
    print(f"  숏 포함 수익: {len(short_ok):,}개")

    # Save
    out = Path(args.output_base) / "artifacts" / "allweather_results.json"
    with open(out, "w") as f:
        json.dump(results[:200], f, indent=2, ensure_ascii=False)
    print(f"\n[allweather] Saved to {out}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
