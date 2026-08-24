"""Ultra-selective 'one-shot' strategy search — targeting 85%+ win rate.

Extends grid search with:
  - Higher ADX floors (40, 45, 50, 55)
  - Higher score mins (75, 80, 85, 90)
  - Single-coin filters (ETH-only, XRP-only, ETH+XRP)
  - Bull regime filter
  - EMA cross alignment filter
  - Trend strength filter
  - Very tight SL + wide TP combinations
"""
from __future__ import annotations

import bisect
import json
import os
import sys
import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class UltraConfig:
    name: str
    tp_type: str
    tp_r_mult: float
    sl_atr_mult: float
    trail_pct: float
    leverage: int
    hold_bars_1h: int
    side_filter: str
    adx_min: float
    score_min: float
    partial_levels: tuple
    coin_filter: tuple  # empty = all, ("ETHUSDT",) = ETH only
    require_bull: bool
    require_ema_cross: bool
    min_trend_strength: float


def _sim(side, entry_price, bars_5m, stop_bps, max_bars, tp_type, tp_bps, trail_pct, partial_levels, cost_bps):
    if not bars_5m or entry_price <= 0:
        return -cost_bps, 0, "NO_DATA"
    peak = 0.0; stop = -stop_bps; realized = 0.0; remain = 1.0; pidx = 0; close_pnl = 0.0
    for i, bar in enumerate(bars_5m[:max_bars]):
        if side == "long":
            hi = ((bar.high_price / entry_price) - 1) * 10000
            lo = ((bar.low_price / entry_price) - 1) * 10000
            close_pnl = ((bar.close_price / entry_price) - 1) * 10000
        else:
            hi = (1 - (bar.low_price / entry_price)) * 10000
            lo = (1 - (bar.high_price / entry_price)) * 10000
            close_pnl = (1 - (bar.close_price / entry_price)) * 10000
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
            rr = stop_bps; levels = partial_levels or (1.0, 2.0)
            if pidx < len(levels) and hi >= levels[pidx] * rr:
                frac = min(0.33, remain); realized += hi * frac; remain -= frac; pidx += 1
                if pidx == 1: stop = max(stop, 0)
            if pidx >= len(levels) and peak > 0:
                ns = peak - peak * 0.6; stop = max(stop, ns)
                if close_pnl <= stop:
                    return (stop * remain + realized) - cost_bps, i+1, "PARTIAL_TRAIL"
    return (close_pnl * remain + realized) - cost_bps, min(len(bars_5m), max_bars), "HOLD"


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="quant_binance/config.example.json")
    parser.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT")
    parser.add_argument("--output-base", default="quant_runtime")
    parser.add_argument("--equity-usd", type=float, default=66.0)
    parser.add_argument("--cost-bps", type=float, default=16.0)
    parser.add_argument("--min-trades", type=int, default=10)
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

    print(f"[ultra] Loading data...")
    all_entries = []
    bars_5m_by_symbol = {}

    for symbol in symbols:
        k5m = load_historical_klines(data_dir=data_dir, symbol=symbol, interval="5m")
        k1h = load_historical_klines(data_dir=data_dir, symbol=symbol, interval="1h")
        k4h = load_historical_klines(data_dir=data_dir, symbol=symbol, interval="4h")
        if not k1h: continue

        parsed = []
        for row in k5m:
            try: parsed.append(_parse_kline(symbol, "5m", row))
            except: continue
        parsed.sort(key=lambda b: b.close_time)
        bars_5m_by_symbol[symbol] = parsed

        slices = build_historical_slices(
            symbol=symbol, klines_5m=k5m or [], klines_1h=k1h, klines_4h=k4h or [],
            settings=settings, extractor=extractor,
        )

        for sl in slices:
            try:
                decision = service.run_cycle(
                    state=sl.state, primitive_inputs=sl.primitive_inputs,
                    history=sl.history, decision_time=sl.decision_time,
                    equity_usd=args.equity_usd,
                    remaining_portfolio_capacity_usd=args.equity_usd * 2.5,
                )
            except: continue
            if decision.final_mode not in ("spot", "futures"): continue
            ep = sl.state.last_trade_price
            if ep <= 0: continue

            adx = getattr(sl.primitive_inputs, 'adx_1h', 0.0)
            atr = getattr(sl.primitive_inputs, 'atr_1h', 0.0)
            atr_bps = (atr / ep) * 10000 if atr > 0 else 50.0
            ema_cross = getattr(sl.primitive_inputs, 'ema_cross_signal', 0)
            trend_str = getattr(sl.primitive_inputs, 'trend_strength', 0.0)
            trend_dir = getattr(sl.primitive_inputs, 'trend_direction', 0)

            # Bull detection: 48h lookback
            bars_1h = sl.state.klines.get("1h", [])
            bull = False
            if len(bars_1h) >= 48:
                c0 = bars_1h[-48].close_price
                c1 = bars_1h[-1].close_price
                if c0 > 0 and ((c1/c0)-1)*100 > 3:
                    bull = True

            all_entries.append({
                "symbol": symbol, "side": decision.side,
                "score": decision.predictability_score, "adx": adx,
                "entry_price": ep, "atr_bps": atr_bps,
                "entry_ms": int(sl.decision_time.timestamp() * 1000),
                "ema_cross": ema_cross, "trend_strength": trend_str,
                "trend_direction": trend_dir, "bull": bull,
            })

        print(f"  {symbol}: {len([e for e in all_entries if e['symbol']==symbol])} entries, {len(parsed)} 5m bars")

    all_entries.sort(key=lambda x: x["entry_ms"])
    print(f"  Total: {len(all_entries)} entries")

    # Pre-compute 5m indices
    sym_ts = {}
    for sym, bars in bars_5m_by_symbol.items():
        sym_ts[sym] = [int(b.close_time.timestamp() * 1000) for b in bars]

    # Build ultra configs
    tp_specs = [
        ("partial_ladder", 0.0, 0.0, (0.5, 1.0)),
        ("partial_ladder", 0.0, 0.0, (0.5, 1.5)),
        ("partial_ladder", 0.0, 0.0, (0.3, 0.7)),
        ("partial_ladder", 0.0, 0.0, (1.0, 2.0)),
        ("trailing", 0.0, 0.3, ()),
        ("trailing", 0.0, 0.5, ()),
        ("fixed", 0.5, 0.0, ()),
        ("fixed", 1.0, 0.0, ()),
        ("fixed", 1.5, 0.0, ()),
    ]
    sl_mults = [1.5, 2.0, 3.0, 4.0]
    holds = [6, 12, 24, 48]
    adx_mins = [35, 40, 45, 50, 55]
    score_mins = [70, 75, 80, 85, 90]
    coin_filters = [(), ("ETHUSDT",), ("XRPUSDT",), ("ETHUSDT", "XRPUSDT")]
    bull_opts = [False, True]
    ema_opts = [False, True]
    trend_mins = [0.0, 0.6, 0.75]
    leverages = [3, 5, 10]

    total = (len(tp_specs) * len(sl_mults) * len(holds) * len(adx_mins) *
             len(score_mins) * len(coin_filters) * len(bull_opts) * len(ema_opts) *
             len(trend_mins) * len(leverages))
    print(f"\n[ultra] {total:,} combos to test")

    # Pre-compute sim cache
    print(f"[ultra] Pre-computing 5m bar indices...")
    future_bars = []
    for feat in all_entries:
        ts_arr = sym_ts.get(feat["symbol"])
        bars = bars_5m_by_symbol.get(feat["symbol"])
        if not ts_arr or not bars:
            future_bars.append(None); continue
        idx = bisect.bisect_right(ts_arr, feat["entry_ms"])
        future_bars.append(bars[idx:] if len(bars) - idx >= 3 else None)

    # Pre-compute sims
    print(f"[ultra] Pre-computing simulations...")
    unique_sim_params = set()
    for tp_type, tp_r, trail, partial in tp_specs:
        for sl in sl_mults:
            for hold in holds:
                unique_sim_params.add((tp_type, tp_r, sl, trail, partial, hold))

    sim_cache = {}
    for si, feat in enumerate(all_entries):
        fb = future_bars[si]
        if fb is None or feat["side"] == "flat": continue
        ep = feat["entry_price"]
        atr_bps = feat["atr_bps"] if feat["atr_bps"] > 0 else 50.0
        for tp_type, tp_r, sl, trail, partial, hold in unique_sim_params:
            stop = max(atr_bps * sl, 15.0)
            tp_bps = stop * tp_r if tp_type == "fixed" else 0.0
            key = (si, tp_type, tp_r, sl, trail, partial, hold)
            sim_cache[key] = _sim(feat["side"], ep, fb, stop, hold*12, tp_type, tp_bps, trail, partial, args.cost_bps)
        if (si+1) % 2000 == 0:
            print(f"  cached {si+1}/{len(all_entries)}...", flush=True)

    print(f"  Cache: {len(sim_cache):,} sims")

    # Run grid
    print(f"\n[ultra] Scoring {total:,} configs...")
    results = []
    ci = 0
    for tp_type, tp_r, trail, partial in tp_specs:
        for sl in sl_mults:
            for hold in holds:
                for adx in adx_mins:
                    for score in score_mins:
                        for coins in coin_filters:
                            for bull_req in bull_opts:
                                for ema_req in ema_opts:
                                    for tmin in trend_mins:
                                        for lev in leverages:
                                            ci += 1
                                            pnls = []
                                            sym_pnls = defaultdict(list)
                                            for si, feat in enumerate(all_entries):
                                                if feat["score"] < score: continue
                                                if feat["adx"] < adx: continue
                                                if feat["side"] != "long": continue
                                                if coins and feat["symbol"] not in coins: continue
                                                if bull_req and not feat["bull"]: continue
                                                if ema_req and feat["ema_cross"] <= 0: continue
                                                if feat["trend_strength"] < tmin: continue
                                                if future_bars[si] is None: continue

                                                key = (si, tp_type, tp_r, sl, trail, partial, hold)
                                                cached = sim_cache.get(key)
                                                if not cached: continue
                                                p = cached[0] * lev
                                                pnls.append(p)
                                                sym_pnls[feat["symbol"]].append(p)

                                            if len(pnls) < args.min_trades: continue
                                            wins = sum(1 for p in pnls if p > 0)
                                            wr = wins / len(pnls)
                                            total_pnl = sum(pnls)
                                            gw = sum(p for p in pnls if p > 0)
                                            gl = abs(sum(p for p in pnls if p <= 0)) or 0.01
                                            pf = gw / gl

                                            sym_stats = {}
                                            for sym, sp in sym_pnls.items():
                                                sw = sum(1 for p in sp if p > 0)
                                                sym_stats[sym] = {"trades": len(sp), "wr": round(sw/len(sp), 4), "total": round(sum(sp), 1)}

                                            results.append({
                                                "wr": round(wr, 4), "pf": round(pf, 2),
                                                "trades": len(pnls), "total_bps": round(total_pnl, 1),
                                                "tp_type": tp_type, "tp_r": tp_r, "sl": sl,
                                                "trail": trail, "partial": list(partial),
                                                "hold": hold, "adx": adx, "score": score,
                                                "coins": list(coins), "bull": bull_req,
                                                "ema": ema_req, "trend_min": tmin,
                                                "lev": lev, "sym": sym_stats,
                                            })

                    if ci % 50000 == 0:
                        print(f"  {ci:,}/{total:,} tested, {len(results):,} viable...", flush=True)

    # Sort by win rate
    results.sort(key=lambda r: (r["wr"], r["pf"], r["total_bps"]), reverse=True)

    print(f"\n{'='*120}")
    print(f"  ULTRA-SELECTIVE STRATEGY SEARCH — TOP 30 BY WIN RATE")
    print(f"{'='*120}")
    print(f"\n{'#':>3} {'WinR':>6} {'PF':>5} {'Trades':>6} {'TotalBps':>10} | "
          f"{'TP':<8} {'SL':>4} {'Hold':>5} {'ADX':>4} {'Score':>5} {'Coins':<20} {'Bull':>5} {'EMA':>4} {'Trend':>5} {'Lev':>3}")
    print("-" * 130)

    for i, r in enumerate(results[:30]):
        coins_str = ",".join(r["coins"]) if r["coins"] else "ALL"
        print(f"{i+1:>3} {r['wr']*100:>5.1f}% {r['pf']:>5.2f} {r['trades']:>6} {r['total_bps']:>+9.1f} | "
              f"{r['tp_type']:<8} {r['sl']:>4.1f} {r['hold']:>4}h {r['adx']:>4.0f} {r['score']:>5.0f} "
              f"{coins_str:<20} {str(r['bull']):>5} {str(r['ema']):>4} {r['trend_min']:>5.2f} {r['lev']:>3}")

    # Detail top 5
    for i, r in enumerate(results[:5]):
        coins_str = ",".join(r["coins"]) if r["coins"] else "ALL"
        print(f"\n  #{i+1}: WR={r['wr']*100:.1f}% PF={r['pf']:.2f} trades={r['trades']} total={r['total_bps']:+.1f}bps")
        print(f"      {r['tp_type']} SL={r['sl']}ATR hold={r['hold']}h ADX>={r['adx']} Score>={r['score']}")
        print(f"      coins={coins_str} bull={r['bull']} ema={r['ema']} trend>={r['trend_min']} lev={r['lev']}x")
        for sym, s in r.get("sym", {}).items():
            print(f"      {sym}: {s['trades']}건 WR={s['wr']*100:.1f}% total={s['total']:+.1f}bps")

    # Count 85%+ strategies
    wr85 = [r for r in results if r["wr"] >= 0.85]
    wr80 = [r for r in results if r["wr"] >= 0.80]
    print(f"\n  승률 85%+: {len(wr85)}개 조합")
    print(f"  승률 80%+: {len(wr80)}개 조합")
    print(f"  전체 viable: {len(results)}개 / {ci:,} tested")

    # Save
    out_path = Path(args.output_base) / "artifacts" / "ultra_selective_results.json"
    with open(out_path, "w") as f:
        json.dump(results[:100], f, indent=2, ensure_ascii=False)
    print(f"\n[ultra] Results saved to {out_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
