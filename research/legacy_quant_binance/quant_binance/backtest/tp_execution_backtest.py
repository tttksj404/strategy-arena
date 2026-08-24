"""TP execution mode comparison — find optimal exit execution strategy.

Simulates the actual session.py exit logic paths on 5m bars:
  1. BITGET_TP_1R       — current: Bitget plan order at 1.0R (all coins same)
  2. BITGET_TP_PROFILE  — Bitget plan order at coin profile RR
  3. PROACTIVE_ONLY     — no Bitget TP, ROE threshold exits only
  4. TRAILING_ONLY      — no Bitget TP, trailing stop only
  5. PROACTIVE+TRAIL    — proactive partial + trailing on remainder
  6. PARTIAL_LADDER     — our backtest winner (0.5R/1.0R partial + trail)
  7. ADAPTIVE           — trailing activation scales with leverage
  8. HYBRID_CEIL        — Bitget TP as max ceiling, proactive in between

Each mode tested with multiple parameter variants.
All include SL (profile-based) as safety net.
"""
from __future__ import annotations

import bisect, json, os, sys
from collections import defaultdict
from pathlib import Path


def _sim_exit(
    *, side, entry_price, bars_5m, stop_bps, max_bars, cost_bps,
    mode, leverage,
    # Mode params
    tp_r=1.0,                    # BITGET_TP: R multiple
    proactive_thresholds=(),     # PROACTIVE: ROE% thresholds
    proactive_fraction=0.5,      # fraction to close at each threshold
    trail_activation_roe=1.5,    # TRAILING: activate at this ROE%
    trail_lock_ratio=0.5,        # lock this fraction of peak
    partial_levels=(0.5, 1.0),   # PARTIAL_LADDER: R multiples
):
    """Simulate one trade with specific exit execution mode."""
    if not bars_5m or entry_price <= 0:
        return -cost_bps * leverage, 0, "NO_DATA", []

    tp_bps = stop_bps * tp_r if mode in ("BITGET_TP_1R", "BITGET_TP_PROFILE", "HYBRID_CEIL") else 99999
    peak_roe = 0.0
    stop = -stop_bps
    realized = 0.0
    remain = 1.0
    partial_idx = 0
    proactive_idx = 0
    trail_active = False
    exits = []  # log partial exits

    for i, bar in enumerate(bars_5m[:max_bars]):
        if side == "long":
            hi = ((bar.high_price / entry_price) - 1) * 10000
            lo = ((bar.low_price / entry_price) - 1) * 10000
            close_pnl = ((bar.close_price / entry_price) - 1) * 10000
        else:
            hi = (1 - (bar.low_price / entry_price)) * 10000
            lo = (1 - (bar.high_price / entry_price)) * 10000
            close_pnl = ((1 - bar.close_price / entry_price)) * 10000

        # ROE% = pnl_bps / 100 * leverage
        roe_pct = hi / 100 * leverage  # best case this bar
        peak_roe = max(peak_roe, roe_pct)
        current_roe = close_pnl / 100 * leverage

        # SL check
        if lo <= stop:
            final = (stop * remain + realized) - cost_bps
            return final * leverage, i+1, "SL", exits

        # ── Mode-specific exit logic ──

        if mode in ("BITGET_TP_1R", "BITGET_TP_PROFILE"):
            # Simple: full exit at TP
            if hi >= tp_bps:
                final = (tp_bps * remain + realized) - cost_bps
                return final * leverage, i+1, "TP", exits

        elif mode == "PROACTIVE_ONLY":
            # Step exits at ROE thresholds
            if proactive_idx < len(proactive_thresholds) and remain > 0.01:
                thresh = proactive_thresholds[proactive_idx]
                if roe_pct >= thresh:
                    frac = min(proactive_fraction, remain)
                    realized += close_pnl * frac
                    remain -= frac
                    exits.append({"bar": i, "roe": round(roe_pct, 2), "frac": round(frac, 3), "type": f"PROACTIVE_{thresh}"})
                    proactive_idx += 1
                    if remain <= 0.01:
                        final = realized - cost_bps
                        return final * leverage, i+1, "PROACTIVE_FULL", exits

        elif mode == "TRAILING_ONLY":
            # Dynamic trailing based on leverage
            activation = trail_activation_roe
            if peak_roe >= activation and remain > 0.01:
                trail_active = True
                locked_pnl = (peak_roe / leverage * 100) * trail_lock_ratio  # convert back to bps
                trail_stop = locked_pnl
                if trail_stop > stop:
                    stop = trail_stop
                if close_pnl <= stop and stop > 0:
                    final = (stop * remain + realized) - cost_bps
                    return final * leverage, i+1, "TRAIL", exits

        elif mode == "PROACTIVE_TRAIL":
            # Proactive partial exits + trailing on remainder
            if proactive_idx < len(proactive_thresholds) and remain > 0.2:
                thresh = proactive_thresholds[proactive_idx]
                if roe_pct >= thresh:
                    frac = min(proactive_fraction, remain * 0.5)
                    realized += close_pnl * frac
                    remain -= frac
                    exits.append({"bar": i, "roe": round(roe_pct, 2), "frac": round(frac, 3)})
                    proactive_idx += 1
                    # After first proactive exit, move stop to breakeven
                    stop = max(stop, 0)

            # Trail the remainder
            activation = trail_activation_roe
            if peak_roe >= activation:
                locked_pnl = (peak_roe / leverage * 100) * trail_lock_ratio
                if locked_pnl > stop:
                    stop = locked_pnl
                if close_pnl <= stop and stop > 0:
                    final = (stop * remain + realized) - cost_bps
                    return final * leverage, i+1, "TRAIL", exits

        elif mode == "PARTIAL_LADDER":
            rr = stop_bps
            levels = partial_levels
            if partial_idx < len(levels) and hi >= levels[partial_idx] * rr:
                frac = min(0.33, remain)
                realized += hi * frac
                remain -= frac
                exits.append({"bar": i, "level": levels[partial_idx], "frac": round(frac, 3)})
                partial_idx += 1
                if partial_idx == 1:
                    stop = max(stop, 0)
            if partial_idx >= len(levels) and peak_roe > 0:
                locked = (peak_roe / leverage * 100) * 0.6
                new_stop = close_pnl - locked if locked > 0 else stop
                # Actually: trail at 60% of peak bps
                peak_bps = peak_roe / leverage * 100
                trail_stop = peak_bps - peak_bps * 0.6
                stop = max(stop, trail_stop)
                if close_pnl <= stop and stop > 0:
                    final = (stop * remain + realized) - cost_bps
                    return final * leverage, i+1, "PARTIAL_TRAIL", exits

        elif mode == "ADAPTIVE":
            # Trailing activation scales with leverage
            activation = max(3.0, leverage * 0.5)  # 10x→5%, 20x→10%
            lock = 0.50
            if peak_roe >= activation:
                locked_pnl = (peak_roe / leverage * 100) * lock
                if locked_pnl > stop:
                    stop = locked_pnl
                if close_pnl <= stop and stop > 0:
                    final = (stop * remain + realized) - cost_bps
                    return final * leverage, i+1, "ADAPTIVE_TRAIL", exits

        elif mode == "HYBRID_CEIL":
            # Proactive partials + Bitget TP as ceiling
            if proactive_idx < len(proactive_thresholds) and remain > 0.3:
                thresh = proactive_thresholds[proactive_idx]
                if roe_pct >= thresh:
                    frac = min(proactive_fraction, remain * 0.4)
                    realized += close_pnl * frac
                    remain -= frac
                    exits.append({"bar": i, "roe": round(roe_pct, 2), "frac": round(frac, 3)})
                    proactive_idx += 1
                    stop = max(stop, 0)
            # Ceiling TP for remainder
            if hi >= tp_bps:
                final = (tp_bps * remain + realized) - cost_bps
                return final * leverage, i+1, "CEIL_TP", exits
            # Trail remainder
            if peak_roe >= trail_activation_roe:
                locked_pnl = (peak_roe / leverage * 100) * trail_lock_ratio
                if locked_pnl > stop:
                    stop = locked_pnl
                if close_pnl <= stop and stop > 0:
                    final = (stop * remain + realized) - cost_bps
                    return final * leverage, i+1, "HYBRID_TRAIL", exits

    # Max hold
    final = (close_pnl * remain + realized) - cost_bps
    return final * leverage, min(len(bars_5m), max_bars), "HOLD", exits


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="quant_binance/config.example.json")
    parser.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT")
    parser.add_argument("--output-base", default="quant_runtime")
    parser.add_argument("--equity-usd", type=float, default=66.0)
    parser.add_argument("--cost-bps", type=float, default=16.0)
    args = parser.parse_args(argv)

    symbols = [s.strip() for s in args.symbols.split(",")]
    data_dir = Path(args.output_base) / "historical"
    os.environ.setdefault("STRATEGY_OVERRIDE_PATH", str(Path(args.output_base) / "artifacts" / "strategy_override.approved.json"))
    os.environ.setdefault("STRATEGY_PROFILE", "live-ultra-aggressive")
    os.environ.setdefault("EXCHANGE", "bitget")

    from quant_binance.settings import Settings
    from quant_binance.features.extractor import MarketFeatureExtractor
    from quant_binance.cost_calibration import load_cost_calibration
    from quant_binance.backtest.historical_fixture_builder import build_historical_slices
    from quant_binance.data.historical_download import load_historical_klines
    from quant_binance.data.rest_seed import _parse_kline
    from quant_binance.service import PaperTradingService
    from quant_binance.execution.router import ExecutionRouter
    from quant_binance.strategy.coin_profiles import get_profile

    settings = Settings.load(args.config)
    extractor = MarketFeatureExtractor(settings, cost_calibration=load_cost_calibration(str(Path(args.output_base) / "artifacts" / "cost_calibration.json")))
    service = PaperTradingService(settings, router=ExecutionRouter())

    print(f"[tp-exec] Loading data...")
    entries = []
    bars_5m_sym = {}

    for symbol in symbols:
        k5m = load_historical_klines(data_dir=data_dir, symbol=symbol, interval="5m")
        k1h = load_historical_klines(data_dir=data_dir, symbol=symbol, interval="1h")
        k4h = load_historical_klines(data_dir=data_dir, symbol=symbol, interval="4h")
        if not k1h or not k5m: continue

        parsed = [_parse_kline(symbol, "5m", r) for r in k5m if r]
        parsed = [p for p in parsed if p]
        parsed.sort(key=lambda b: b.close_time)
        bars_5m_sym[symbol] = parsed

        slices = build_historical_slices(symbol=symbol, klines_5m=k5m, klines_1h=k1h, klines_4h=k4h or [], settings=settings, extractor=extractor)
        for sl in slices:
            try:
                d = service.run_cycle(state=sl.state, primitive_inputs=sl.primitive_inputs, history=sl.history, decision_time=sl.decision_time, equity_usd=args.equity_usd, remaining_portfolio_capacity_usd=args.equity_usd*2.5)
            except: continue
            if d.final_mode not in ("spot","futures") or d.side == "flat": continue
            ep = sl.state.last_trade_price
            if ep <= 0: continue
            adx = getattr(sl.primitive_inputs, 'adx_1h', 0.0)
            atr = getattr(sl.primitive_inputs, 'atr_1h', 0.0)
            atr_bps = (atr/ep)*10000 if atr > 0 else 50.0
            entries.append({"symbol": symbol, "side": d.side, "score": d.predictability_score, "adx": adx, "entry_price": ep, "atr_bps": atr_bps, "entry_ms": int(sl.decision_time.timestamp()*1000)})

        print(f"  {symbol}: {len([e for e in entries if e['symbol']==symbol])} entries, {len(parsed)} 5m bars")

    entries.sort(key=lambda x: x["entry_ms"])
    print(f"  Total: {len(entries)} entries")

    # Pre-compute future bars
    sym_ts = {s: [int(b.close_time.timestamp()*1000) for b in bars] for s, bars in bars_5m_sym.items()}
    future = []
    for e in entries:
        ts = sym_ts.get(e["symbol"])
        bars = bars_5m_sym.get(e["symbol"])
        if not ts or not bars:
            future.append(None); continue
        idx = bisect.bisect_right(ts, e["entry_ms"])
        future.append(bars[idx:] if len(bars)-idx >= 3 else None)

    # Filter: ADX >= 40, Score >= 55, long only (our all-weather best)
    # Also test with broader filter
    filters = [
        ("BROAD (ADX>=25, Score>=55, both)", 25, 55, "both"),
        ("MEDIUM (ADX>=35, Score>=65, long)", 35, 65, "long"),
        ("TIGHT (ADX>=40, Score>=75, long)", 40, 75, "long"),
    ]

    # Modes to test
    modes = [
        {"name": "BITGET_TP_1R", "mode": "BITGET_TP_1R", "tp_r": 1.0},
        {"name": "BITGET_TP_PROFILE", "mode": "BITGET_TP_PROFILE"},  # uses coin RR
        {"name": "PROACTIVE(6,10,14)", "mode": "PROACTIVE_ONLY", "proactive_thresholds": (6, 10, 14), "proactive_fraction": 0.5},
        {"name": "PROACTIVE(3,6,10)", "mode": "PROACTIVE_ONLY", "proactive_thresholds": (3, 6, 10), "proactive_fraction": 0.5},
        {"name": "PROACTIVE(2,4,8)", "mode": "PROACTIVE_ONLY", "proactive_thresholds": (2, 4, 8), "proactive_fraction": 0.4},
        {"name": "TRAILING(1.5%,50%)", "mode": "TRAILING_ONLY", "trail_activation_roe": 1.5, "trail_lock_ratio": 0.5},
        {"name": "TRAILING(3%,50%)", "mode": "TRAILING_ONLY", "trail_activation_roe": 3.0, "trail_lock_ratio": 0.5},
        {"name": "TRAILING(5%,50%)", "mode": "TRAILING_ONLY", "trail_activation_roe": 5.0, "trail_lock_ratio": 0.5},
        {"name": "TRAILING(5%,60%)", "mode": "TRAILING_ONLY", "trail_activation_roe": 5.0, "trail_lock_ratio": 0.6},
        {"name": "PRO+TRAIL(3,6|3%)", "mode": "PROACTIVE_TRAIL", "proactive_thresholds": (3, 6), "proactive_fraction": 0.3, "trail_activation_roe": 3.0, "trail_lock_ratio": 0.5},
        {"name": "PRO+TRAIL(2,5|5%)", "mode": "PROACTIVE_TRAIL", "proactive_thresholds": (2, 5), "proactive_fraction": 0.25, "trail_activation_roe": 5.0, "trail_lock_ratio": 0.5},
        {"name": "LADDER(0.5R,1R)", "mode": "PARTIAL_LADDER", "partial_levels": (0.5, 1.0)},
        {"name": "LADDER(0.3R,0.7R)", "mode": "PARTIAL_LADDER", "partial_levels": (0.3, 0.7)},
        {"name": "LADDER(1R,2R)", "mode": "PARTIAL_LADDER", "partial_levels": (1.0, 2.0)},
        {"name": "ADAPTIVE(auto,50%)", "mode": "ADAPTIVE", "trail_lock_ratio": 0.5},
        {"name": "ADAPTIVE(auto,60%)", "mode": "ADAPTIVE", "trail_lock_ratio": 0.6},
        {"name": "HYBRID(prof+3,6)", "mode": "HYBRID_CEIL", "proactive_thresholds": (3, 6), "proactive_fraction": 0.25, "trail_activation_roe": 3.0, "trail_lock_ratio": 0.5},
        {"name": "HYBRID(prof+2,5)", "mode": "HYBRID_CEIL", "proactive_thresholds": (2, 5), "proactive_fraction": 0.3, "trail_activation_roe": 5.0, "trail_lock_ratio": 0.5},
    ]

    leverages = [10, 15, 20]

    for filter_name, adx_min, score_min, side_f in filters:
        print(f"\n{'='*130}")
        print(f"  FILTER: {filter_name}")
        print(f"{'='*130}")

        for lev in leverages:
            print(f"\n  Leverage: {lev}x")
            print(f"  {'Mode':<28} {'Trades':>6} {'WR':>6} {'AvgPnL':>9} {'TotalBps':>11} {'PF':>6} {'MaxDD':>9} | {'TP':>5} {'SL':>5} {'TRAIL':>5} {'HOLD':>5}")
            print(f"  {'-'*115}")

            for m in modes:
                pnls = []
                reasons = defaultdict(int)

                for si, e in enumerate(entries):
                    if e["score"] < score_min or e["adx"] < adx_min: continue
                    if side_f == "long" and e["side"] != "long": continue
                    if e["side"] == "flat": continue
                    if future[si] is None: continue

                    cp = get_profile(e["symbol"])
                    atr_bps = e["atr_bps"] if e["atr_bps"] > 0 else 50.0

                    # Use profile-specific params
                    if e["side"] == "short" and cp.short_sl_mult > 0:
                        sl_mult = cp.short_sl_mult
                        rr = cp.short_rr if cp.short_rr > 0 else cp.rr
                    else:
                        sl_mult = cp.sl_atr_mult
                        rr = cp.rr

                    stop = max(atr_bps * sl_mult, 15.0)
                    hold = (cp.short_hold_bars if e["side"] == "short" and cp.short_hold_bars > 0 else cp.hold_bars) * 12

                    tp_r = m.get("tp_r", rr)
                    if m["mode"] == "BITGET_TP_PROFILE":
                        tp_r = rr
                    elif m["mode"] == "HYBRID_CEIL":
                        tp_r = rr * 2  # ceiling at 2x profile RR

                    pnl, _, reason, _ = _sim_exit(
                        side=e["side"], entry_price=e["entry_price"],
                        bars_5m=future[si], stop_bps=stop, max_bars=hold,
                        cost_bps=args.cost_bps, mode=m["mode"], leverage=lev,
                        tp_r=tp_r,
                        proactive_thresholds=m.get("proactive_thresholds", ()),
                        proactive_fraction=m.get("proactive_fraction", 0.5),
                        trail_activation_roe=m.get("trail_activation_roe", 1.5),
                        trail_lock_ratio=m.get("trail_lock_ratio", 0.5),
                        partial_levels=m.get("partial_levels", (0.5, 1.0)),
                    )
                    pnls.append(pnl)
                    reasons[reason] += 1

                if len(pnls) < 10: continue
                wins = sum(1 for p in pnls if p > 0)
                total = sum(pnls)
                gw = sum(p for p in pnls if p > 0)
                gl = abs(sum(p for p in pnls if p <= 0)) or 0.01
                eq = 0; peq = 0; mdd = 0
                for p in pnls:
                    eq += p; peq = max(peq, eq); mdd = max(mdd, peq - eq)

                tp_n = reasons.get("TP",0) + reasons.get("CEIL_TP",0) + reasons.get("PROACTIVE_FULL",0)
                sl_n = reasons.get("SL",0)
                tr_n = reasons.get("TRAIL",0) + reasons.get("PARTIAL_TRAIL",0) + reasons.get("ADAPTIVE_TRAIL",0) + reasons.get("HYBRID_TRAIL",0)
                ho_n = reasons.get("HOLD",0)

                print(f"  {m['name']:<28} {len(pnls):>6} {wins/len(pnls)*100:>5.1f}% {total/len(pnls):>+8.1f} "
                      f"{total:>+10.0f} {gw/gl:>5.2f} {mdd:>8.0f} | {tp_n:>5} {sl_n:>5} {tr_n:>5} {ho_n:>5}")

    # Save
    print(f"\n[tp-exec] Done.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
