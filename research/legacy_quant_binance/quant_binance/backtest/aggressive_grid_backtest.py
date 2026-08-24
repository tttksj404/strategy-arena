"""Aggressive 'one-shot' strategy grid search backtest.

Grid-searches ALL combinations of:
  - Entry filters: ADX floor, score min, trend strength min
  - Exit strategies: TP type x TP multiplier x SL multiplier x trailing %
  - Leverage levels
  - Hold periods
  - Side filters (long-only, short-only, both)

Runs on 5m bars for realistic intra-bar TP/SL simulation.
Outputs top-N combos ranked by: win_rate, profit_factor, total_pnl, risk-adjusted return.
"""
from __future__ import annotations

import itertools
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class GridConfig:
    name: str
    tp_type: str          # "fixed", "trailing", "partial_ladder"
    tp_r_mult: float      # TP distance as multiple of SL
    sl_atr_mult: float    # SL = ATR * this
    trail_pct: float      # trailing: lock this fraction of peak (0 = unused)
    leverage: int
    hold_bars_1h: int
    side_filter: str      # "long", "short", "both"
    adx_min: float
    score_min: float
    partial_levels: tuple  # for partial_ladder: (1.0, 2.0) etc


@dataclass
class GridResult:
    config: GridConfig
    trades: int
    wins: int
    win_rate: float
    avg_pnl_bps: float
    total_pnl_bps: float
    total_pnl_usd: float
    profit_factor: float
    max_drawdown_bps: float
    avg_hold_5m: float
    best_bps: float
    worst_bps: float
    # Per-symbol breakdown
    symbol_stats: dict[str, dict]


def _sim_bar_by_bar(
    *, side: str, entry_price: float, bars_5m: list,
    stop_bps: float, max_bars: int, tp_type: str,
    tp_bps: float, trail_pct: float, partial_levels: tuple,
    cost_bps: float,
) -> tuple[float, float, float, int, str]:
    if not bars_5m or entry_price <= 0:
        return -cost_bps, 0.0, 0.0, 0, "NO_DATA"

    peak_pnl = 0.0
    worst_pnl = 0.0
    stop = -stop_bps
    realized = 0.0
    remain = 1.0
    pidx = 0
    close_pnl = 0.0

    for i, bar in enumerate(bars_5m[:max_bars]):
        if side == "long":
            hi = ((bar.high_price / entry_price) - 1) * 10000
            lo = ((bar.low_price / entry_price) - 1) * 10000
            close_pnl = ((bar.close_price / entry_price) - 1) * 10000
        else:
            hi = (1 - (bar.low_price / entry_price)) * 10000
            lo = (1 - (bar.high_price / entry_price)) * 10000
            close_pnl = (1 - (bar.close_price / entry_price)) * 10000

        peak_pnl = max(peak_pnl, hi)
        worst_pnl = min(worst_pnl, lo)

        # Stop check
        if lo <= stop:
            return (stop * remain + realized) - cost_bps, peak_pnl, worst_pnl, i+1, "SL"

        if tp_type == "fixed":
            if tp_bps > 0 and hi >= tp_bps:
                return (tp_bps * remain + realized) - cost_bps, peak_pnl, worst_pnl, i+1, "TP"

        elif tp_type == "trailing":
            if peak_pnl > 0 and trail_pct > 0:
                new_stop = peak_pnl * (1 - trail_pct)
                if new_stop > stop:
                    stop = new_stop
                if stop > 0 and close_pnl <= stop:
                    return (stop * remain + realized) - cost_bps, peak_pnl, worst_pnl, i+1, "TRAIL"

        elif tp_type == "partial_ladder":
            rr_bps = stop_bps
            levels = partial_levels or (1.0, 2.0)
            if pidx < len(levels) and hi >= levels[pidx] * rr_bps:
                frac = min(0.33, remain)
                realized += hi * frac
                remain -= frac
                pidx += 1
                if pidx == 1:
                    stop = max(stop, 0)
            if pidx >= len(levels) and peak_pnl > 0:
                trail_val = peak_pnl * 0.6
                new_stop = peak_pnl - trail_val
                stop = max(stop, new_stop)
                if close_pnl <= stop:
                    return (stop * remain + realized) - cost_bps, peak_pnl, worst_pnl, i+1, "PARTIAL_TRAIL"

    return (close_pnl * remain + realized) - cost_bps, peak_pnl, worst_pnl, min(len(bars_5m), max_bars), "HOLD"


def build_grid_configs() -> list[GridConfig]:
    """Generate all parameter combinations."""
    configs = []

    # ── TP types and their params ──
    tp_specs = [
        # (tp_type, tp_r_mult, trail_pct, partial_levels)
        ("fixed",          0.5, 0.0, ()),
        ("fixed",          1.0, 0.0, ()),
        ("fixed",          1.5, 0.0, ()),
        ("fixed",          2.0, 0.0, ()),
        ("fixed",          3.0, 0.0, ()),
        ("trailing",       0.0, 0.3, ()),
        ("trailing",       0.0, 0.4, ()),
        ("trailing",       0.0, 0.5, ()),
        ("trailing",       0.0, 0.6, ()),
        ("trailing",       0.0, 0.7, ()),
        ("partial_ladder", 0.0, 0.0, (0.5, 1.0)),
        ("partial_ladder", 0.0, 0.0, (0.5, 1.5)),
        ("partial_ladder", 0.0, 0.0, (1.0, 2.0)),
        ("partial_ladder", 0.0, 0.0, (1.0, 3.0)),
        ("partial_ladder", 0.0, 0.0, (0.75, 1.5)),
    ]

    sl_atr_mults = [0.5, 1.0, 1.5, 2.0, 3.0]
    leverages = [3, 5, 10, 15, 20]
    hold_bars_list = [3, 6, 12, 24, 48]
    side_filters = ["long", "both"]
    adx_mins = [20, 25, 30, 35, 40]
    score_mins = [55, 60, 65, 70, 75]

    idx = 0
    for tp_type, tp_r, trail, partial in tp_specs:
        for sl in sl_atr_mults:
            for lev in leverages:
                for hold in hold_bars_list:
                    for side in side_filters:
                        for adx in adx_mins:
                            for score in score_mins:
                                idx += 1
                                configs.append(GridConfig(
                                    name=f"G{idx}",
                                    tp_type=tp_type,
                                    tp_r_mult=tp_r,
                                    sl_atr_mult=sl,
                                    trail_pct=trail,
                                    leverage=lev,
                                    hold_bars_1h=hold,
                                    side_filter=side,
                                    adx_min=adx,
                                    score_min=score,
                                    partial_levels=partial,
                                ))
    return configs


def _precompute_future_bars(
    features_by_slice: list[dict],
    bars_5m_by_symbol: dict[str, list],
) -> list[list | None]:
    """Pre-compute future 5m bar slices using bisect for O(log n) lookup."""
    import bisect

    # Build sorted timestamp arrays per symbol for binary search
    sym_ts: dict[str, list[int]] = {}
    for sym, bars in bars_5m_by_symbol.items():
        sym_ts[sym] = [int(b.close_time.timestamp() * 1000) for b in bars]

    result = []
    for feat in features_by_slice:
        symbol = feat["symbol"]
        entry_ms = feat["entry_ms"]
        ts_arr = sym_ts.get(symbol)
        bars = bars_5m_by_symbol.get(symbol)
        if not ts_arr or not bars:
            result.append(None)
            continue
        idx = bisect.bisect_right(ts_arr, entry_ms)
        if len(bars) - idx < 3:
            result.append(None)
        else:
            result.append(bars[idx:])  # slice once, reuse many times
    return result


def run_grid_search(
    *,
    slices: list,
    bars_5m_by_symbol: dict[str, list],
    features_by_slice: list[dict],
    equity_usd: float = 66.0,
    cost_bps: float = 16.0,
    top_n: int = 30,
    min_trades: int = 15,
) -> list[GridResult]:
    """Run all grid configs and return top N by composite score."""
    configs = build_grid_configs()
    total = len(configs)
    print(f"  [grid] {total} parameter combinations to test on {len(features_by_slice)} entries")

    # Pre-compute future 5m bars for each entry (once, not per config)
    print(f"  [grid] Pre-computing 5m bar indices...", flush=True)
    future_bars = _precompute_future_bars(features_by_slice, bars_5m_by_symbol)

    # Pre-compute sim results cache: key = (entry_idx, sl_atr_mult, tp_type, tp_r_mult, trail_pct, partial_levels, hold_bars_1h)
    # This avoids re-simulating the same trade for configs that only differ in leverage/filters
    print(f"  [grid] Pre-computing simulation cache...", flush=True)
    sim_cache: dict[tuple, tuple[float, float, float, int, str]] = {}

    # Unique sim params (excluding leverage, side_filter, adx_min, score_min)
    tp_specs_unique = set()
    for cfg in configs:
        tp_specs_unique.add((cfg.tp_type, cfg.tp_r_mult, cfg.sl_atr_mult, cfg.trail_pct, cfg.partial_levels, cfg.hold_bars_1h))

    cache_done = 0
    for si, feat in enumerate(features_by_slice):
        fb = future_bars[si]
        if fb is None:
            continue
        side = feat["side"]
        if side == "flat":
            continue
        entry_price = feat["entry_price"]
        atr_bps = feat["atr_bps"] if feat["atr_bps"] > 0 else 50.0

        for tp_type, tp_r_mult, sl_atr_mult, trail_pct, partial_levels, hold_bars_1h in tp_specs_unique:
            stop_bps = max(atr_bps * sl_atr_mult, 15.0)
            tp_bps = stop_bps * tp_r_mult if tp_type == "fixed" else 0.0
            max_bars_5m = hold_bars_1h * 12

            key = (si, sl_atr_mult, tp_type, tp_r_mult, trail_pct, partial_levels, hold_bars_1h)
            sim_cache[key] = _sim_bar_by_bar(
                side=side, entry_price=entry_price, bars_5m=fb,
                stop_bps=stop_bps, max_bars=max_bars_5m,
                tp_type=tp_type, tp_bps=tp_bps,
                trail_pct=trail_pct, partial_levels=partial_levels,
                cost_bps=cost_bps,
            )
        cache_done += 1
        if cache_done % 200 == 0:
            print(f"  [grid] cached {cache_done}/{len(features_by_slice)} entries...", flush=True)

    print(f"  [grid] Cache built: {len(sim_cache)} simulations. Now scoring {total} configs...", flush=True)

    results: list[GridResult] = []

    for ci, cfg in enumerate(configs):
        trades_pnl = []
        sym_data: dict[str, list[float]] = defaultdict(list)
        holds = []

        for si, feat in enumerate(features_by_slice):
            if feat["score"] < cfg.score_min:
                continue
            if feat["adx"] < cfg.adx_min:
                continue
            side = feat["side"]
            if cfg.side_filter == "long" and side != "long":
                continue
            if cfg.side_filter == "short" and side != "short":
                continue
            if side == "flat":
                continue
            if future_bars[si] is None:
                continue

            key = (si, cfg.sl_atr_mult, cfg.tp_type, cfg.tp_r_mult, cfg.trail_pct, cfg.partial_levels, cfg.hold_bars_1h)
            cached = sim_cache.get(key)
            if cached is None:
                continue

            net_pnl, peak, worst, hbars, reason = cached
            lev_pnl = net_pnl * cfg.leverage
            trades_pnl.append(lev_pnl)
            sym_data[feat["symbol"]].append(lev_pnl)
            holds.append(hbars)

        if len(trades_pnl) < min_trades:
            continue

        wins = [p for p in trades_pnl if p > 0]
        losses = [p for p in trades_pnl if p <= 0]
        gross_win = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 0.01
        total_pnl = sum(trades_pnl)

        # Drawdown
        running = 0.0
        peak_eq = 0.0
        max_dd = 0.0
        for p in trades_pnl:
            running += p
            peak_eq = max(peak_eq, running)
            max_dd = max(max_dd, peak_eq - running)

        # Per-symbol stats
        sym_stats = {}
        for sym, pnls in sym_data.items():
            sw = sum(1 for p in pnls if p > 0)
            sym_stats[sym] = {
                "trades": len(pnls),
                "win_rate": round(sw / len(pnls), 4) if pnls else 0,
                "total_pnl_bps": round(sum(pnls), 2),
                "avg_pnl_bps": round(sum(pnls) / len(pnls), 2),
            }

        wr = len(wins) / len(trades_pnl)
        avg_pnl = total_pnl / len(trades_pnl)
        avg_notional = equity_usd * 0.5
        total_usd = total_pnl / 10000 * avg_notional

        results.append(GridResult(
            config=cfg,
            trades=len(trades_pnl),
            wins=len(wins),
            win_rate=round(wr, 4),
            avg_pnl_bps=round(avg_pnl, 2),
            total_pnl_bps=round(total_pnl, 2),
            total_pnl_usd=round(total_usd, 2),
            profit_factor=round(gross_win / gross_loss, 2),
            max_drawdown_bps=round(max_dd, 2),
            avg_hold_5m=round(sum(holds) / len(holds), 1),
            best_bps=round(max(trades_pnl), 2),
            worst_bps=round(min(trades_pnl), 2),
            symbol_stats=sym_stats,
        ))

        if (ci + 1) % 10000 == 0:
            print(f"  [grid] {ci+1}/{total} configs tested, {len(results)} viable...", flush=True)

    # Sort by composite: prioritize win_rate > 60%, then profit_factor, then total_pnl
    def rank_score(r: GridResult) -> float:
        wr_bonus = 50.0 if r.win_rate >= 0.60 else (25.0 if r.win_rate >= 0.50 else 0.0)
        pf_bonus = r.profit_factor * 20.0
        pnl_bonus = min(r.total_pnl_bps / 100, 50.0)
        dd_penalty = r.max_drawdown_bps / 500
        trade_bonus = min(r.trades / 10, 10.0)  # prefer more trades for statistical validity
        return wr_bonus + pf_bonus + pnl_bonus - dd_penalty + trade_bonus

    results.sort(key=rank_score, reverse=True)
    return results[:top_n]


def print_grid_report(results: list[GridResult], equity_usd: float = 66.0) -> None:
    print("\n" + "=" * 120)
    print("  AGGRESSIVE STRATEGY GRID SEARCH — TOP RESULTS")
    print("=" * 120)

    print(f"\n{'#':>3} {'WinR':>6} {'PF':>5} {'Trades':>6} {'AvgPnL':>8} {'TotalBps':>10} "
          f"{'PnL$':>8} {'MaxDD':>8} {'Hold':>6} | "
          f"{'TP_Type':<9} {'TP_R':>4} {'SL':>4} {'Trail':>5} {'Lev':>3} {'Hold_h':>6} "
          f"{'Side':<5} {'ADX':>4} {'Score':>5} {'Partial'}")
    print("-" * 160)

    for i, r in enumerate(results):
        c = r.config
        partial_str = str(c.partial_levels) if c.partial_levels else "-"
        print(f"{i+1:>3} {r.win_rate*100:>5.1f}% {r.profit_factor:>5.2f} {r.trades:>6} "
              f"{r.avg_pnl_bps:>+7.1f} {r.total_pnl_bps:>+9.1f} "
              f"{r.total_pnl_usd:>+7.2f} {r.max_drawdown_bps:>7.0f} {r.avg_hold_5m:>5.0f}m | "
              f"{c.tp_type:<9} {c.tp_r_mult:>4.1f} {c.sl_atr_mult:>4.1f} {c.trail_pct:>5.1f} "
              f"{c.leverage:>3} {c.hold_bars_1h:>5}h "
              f"{c.side_filter:<5} {c.adx_min:>4.0f} {c.score_min:>5.0f} {partial_str}")

    # Detail for top 5
    print(f"\n{'=' * 120}")
    print(f"  TOP 5 DETAILED BREAKDOWN")
    print(f"{'=' * 120}")

    for i, r in enumerate(results[:5]):
        c = r.config
        print(f"\n  #{i+1}: {c.tp_type} | TP={c.tp_r_mult}R | SL={c.sl_atr_mult}ATR | "
              f"Trail={c.trail_pct} | Lev={c.leverage}x | Hold={c.hold_bars_1h}h | "
              f"Side={c.side_filter} | ADX>={c.adx_min} | Score>={c.score_min}")
        print(f"      Trades={r.trades} WR={r.win_rate*100:.1f}% PF={r.profit_factor:.2f} "
              f"Avg={r.avg_pnl_bps:+.1f}bps Total={r.total_pnl_bps:+.1f}bps "
              f"${r.total_pnl_usd:+.2f} MaxDD={r.max_drawdown_bps:.0f}bps")
        print(f"      Best={r.best_bps:+.1f}bps Worst={r.worst_bps:+.1f}bps")
        print(f"      Per-symbol:")
        for sym in sorted(r.symbol_stats.keys()):
            s = r.symbol_stats[sym]
            print(f"        {sym:<12} {s['trades']:>4}건 WR={s['win_rate']*100:>5.1f}% "
                  f"Avg={s['avg_pnl_bps']:>+7.1f} Total={s['total_pnl_bps']:>+9.1f}bps")

    if results:
        best = results[0]
        c = best.config
        print(f"\n{'=' * 120}")
        print(f"  BEST COMBO FOUND")
        print(f"{'=' * 120}")
        print(f"\n  >>> {c.tp_type} | R:R={c.tp_r_mult} | SL={c.sl_atr_mult}xATR | "
              f"Trail={c.trail_pct} | Lev={c.leverage}x")
        print(f"      ADX >= {c.adx_min} | Score >= {c.score_min} | "
              f"Side={c.side_filter} | Hold={c.hold_bars_1h}h")
        print(f"      승률 {best.win_rate*100:.1f}% | PF {best.profit_factor:.2f} | "
              f"총 {best.total_pnl_usd:+.2f} USD ({best.total_pnl_usd/equity_usd*100:+.1f}%)")
    print()


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Aggressive grid-search backtest")
    parser.add_argument("--config", default="quant_binance/config.example.json")
    parser.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT")
    parser.add_argument("--output-base", default="quant_runtime")
    parser.add_argument("--equity-usd", type=float, default=66.0)
    parser.add_argument("--cost-bps", type=float, default=16.0)
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--min-trades", type=int, default=15)
    args = parser.parse_args(argv)

    symbols = [s.strip() for s in args.symbols.split(",")]
    data_dir = Path(args.output_base) / "historical"

    os.environ.setdefault("STRATEGY_OVERRIDE_PATH",
                          str(Path(args.output_base) / "artifacts" / "strategy_override.approved.json"))
    os.environ.setdefault("STRATEGY_PROFILE", "live-ultra-aggressive")
    os.environ.setdefault("EXCHANGE", "bitget")

    from quant_binance.settings import Settings
    from quant_binance.features.extractor import MarketFeatureExtractor
    from quant_binance.cost_calibration import load_cost_calibration
    from quant_binance.backtest.historical_fixture_builder import build_historical_slices
    from quant_binance.data.historical_download import load_historical_klines, load_funding_rates, load_spot_klines
    from quant_binance.data.rest_seed import _parse_kline
    from quant_binance.service import PaperTradingService
    from quant_binance.execution.router import ExecutionRouter

    settings = Settings.load(args.config)
    cal_path = Path(args.output_base) / "artifacts" / "cost_calibration.json"
    extractor = MarketFeatureExtractor(settings, cost_calibration=load_cost_calibration(str(cal_path)))
    service = PaperTradingService(settings, router=ExecutionRouter())

    print(f"[grid] Loading data for {symbols}...")

    all_entries = []       # (slice, features_dict)
    bars_5m_by_symbol: dict[str, list] = {}

    for symbol in symbols:
        k5m = load_historical_klines(data_dir=data_dir, symbol=symbol, interval="5m")
        k1h = load_historical_klines(data_dir=data_dir, symbol=symbol, interval="1h")
        k4h = load_historical_klines(data_dir=data_dir, symbol=symbol, interval="4h")
        k1m = load_historical_klines(data_dir=data_dir, symbol=symbol, interval="1m")
        spot_1h = load_spot_klines(data_dir=data_dir, symbol=symbol, interval="1h")
        funding = load_funding_rates(data_dir=data_dir, symbol=symbol)

        if not k1h or not k5m:
            print(f"  {symbol}: insufficient data, skipping")
            continue

        parsed_5m = []
        for row in k5m:
            try:
                parsed_5m.append(_parse_kline(symbol, "5m", row))
            except Exception:
                continue
        parsed_5m.sort(key=lambda b: b.close_time)
        bars_5m_by_symbol[symbol] = parsed_5m

        slices = build_historical_slices(
            symbol=symbol, klines_5m=k5m, klines_1h=k1h, klines_4h=k4h,
            klines_1m=k1m, spot_klines_1h=spot_1h, funding_rates=funding,
            settings=settings, extractor=extractor,
        )
        print(f"  {symbol}: {len(slices)} slices, {len(parsed_5m)} 5m bars")

        # Pre-compute decisions and features for each slice
        for sl in slices:
            try:
                decision = service.run_cycle(
                    state=sl.state, primitive_inputs=sl.primitive_inputs,
                    history=sl.history, decision_time=sl.decision_time,
                    equity_usd=args.equity_usd,
                    remaining_portfolio_capacity_usd=args.equity_usd * 2.5,
                )
            except Exception:
                continue

            if decision.final_mode not in ("spot", "futures"):
                continue

            entry_price = sl.state.last_trade_price
            if entry_price <= 0:
                continue

            # Extract ADX
            adx = 0.0
            if hasattr(sl.primitive_inputs, 'adx_1h'):
                adx = sl.primitive_inputs.adx_1h

            # Extract ATR
            atr_bps = 0.0
            if hasattr(sl.primitive_inputs, 'atr_1h') and sl.primitive_inputs.atr_1h > 0:
                atr_bps = (sl.primitive_inputs.atr_1h / entry_price) * 10000
            if atr_bps <= 0:
                klines_1h = sl.state.klines.get("1h", [])
                if len(klines_1h) >= 14:
                    trs = []
                    for j in range(1, min(15, len(klines_1h))):
                        b = klines_1h[-j]
                        tr = max(b.high_price - b.low_price,
                                 abs(b.high_price - klines_1h[-j-1].close_price),
                                 abs(b.low_price - klines_1h[-j-1].close_price))
                        trs.append(tr)
                    atr = sum(trs) / len(trs) if trs else 0
                    atr_bps = (atr / entry_price) * 10000
            if atr_bps <= 0:
                atr_bps = 50.0

            feat = {
                "symbol": symbol,
                "side": decision.side,
                "score": decision.predictability_score,
                "adx": adx,
                "entry_price": entry_price,
                "atr_bps": atr_bps,
                "entry_ms": int(sl.decision_time.timestamp() * 1000),
            }
            all_entries.append((sl, feat))

    all_entries.sort(key=lambda x: x[1]["entry_ms"])
    slices_list = [e[0] for e in all_entries]
    features_list = [e[1] for e in all_entries]

    print(f"  Total entries: {len(all_entries)}")

    if not all_entries:
        print("[ERROR] No entries.")
        return 1

    # Run grid search
    total_configs = len(build_grid_configs())
    print(f"\n[grid] Running grid search: {total_configs} combos x {len(all_entries)} entries...")
    top_results = run_grid_search(
        slices=slices_list,
        bars_5m_by_symbol=bars_5m_by_symbol,
        features_by_slice=features_list,
        equity_usd=args.equity_usd,
        cost_bps=args.cost_bps,
        top_n=args.top_n,
        min_trades=args.min_trades,
    )

    print_grid_report(top_results, equity_usd=args.equity_usd)

    # Save results
    output_path = Path(args.output_base) / "artifacts" / "aggressive_grid_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = []
    for r in top_results:
        c = r.config
        serializable.append({
            "rank": len(serializable) + 1,
            "tp_type": c.tp_type, "tp_r_mult": c.tp_r_mult,
            "sl_atr_mult": c.sl_atr_mult, "trail_pct": c.trail_pct,
            "leverage": c.leverage, "hold_bars_1h": c.hold_bars_1h,
            "side_filter": c.side_filter, "adx_min": c.adx_min,
            "score_min": c.score_min, "partial_levels": list(c.partial_levels),
            "trades": r.trades, "wins": r.wins, "win_rate": r.win_rate,
            "avg_pnl_bps": r.avg_pnl_bps, "total_pnl_bps": r.total_pnl_bps,
            "total_pnl_usd": r.total_pnl_usd, "profit_factor": r.profit_factor,
            "max_drawdown_bps": r.max_drawdown_bps, "avg_hold_5m": r.avg_hold_5m,
            "best_bps": r.best_bps, "worst_bps": r.worst_bps,
            "symbol_stats": r.symbol_stats,
        })
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print(f"[grid] Results saved to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
