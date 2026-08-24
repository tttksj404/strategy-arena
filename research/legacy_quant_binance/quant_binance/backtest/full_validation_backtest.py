"""Full validation backtest — 375 days x 20 coins x walk-forward + Monte Carlo.

Uses 1h bars (375 days available for ALL 20 coins) for comprehensive validation.
Simulates TP/SL on 1h bar high/low within each bar.

Tests:
  1. All 20 coins with the best grid-search strategy
  2. Walk-forward: train on first 250d, test on last 125d
  3. Monte Carlo: shuffle trade order 1000x, measure ruin probability
  4. Cost sensitivity: test at 16, 24, 32 bps
  5. Market regime analysis: bull/bear/sideways
"""
from __future__ import annotations

import bisect
import json
import math
import os
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class TradeResult:
    symbol: str
    side: str
    entry_time: datetime
    entry_price: float
    pnl_bps: float
    exit_reason: str
    hold_bars: int
    regime: str  # "bull" | "bear" | "sideways"


def _detect_regime(bars_1h: list, lookback: int = 48) -> str:
    """Detect market regime from last N 1h bars."""
    if len(bars_1h) < lookback:
        return "sideways"
    first_close = bars_1h[-lookback].close_price
    last_close = bars_1h[-1].close_price
    if first_close <= 0:
        return "sideways"
    change_pct = ((last_close / first_close) - 1) * 100
    if change_pct > 3:
        return "bull"
    elif change_pct < -3:
        return "bear"
    return "sideways"


def _sim_1h_bars(
    *, side: str, entry_price: float, bars: list,
    stop_bps: float, max_bars: int,
    tp_type: str, tp_bps: float, trail_pct: float,
    partial_levels: tuple, cost_bps: float,
) -> tuple[float, int, str]:
    """Simulate trade on 1h bars. Returns (net_pnl_bps, hold_bars, exit_reason)."""
    if not bars or entry_price <= 0:
        return -cost_bps, 0, "NO_DATA"

    peak = 0.0
    stop = -stop_bps
    realized = 0.0
    remain = 1.0
    pidx = 0
    close_pnl = 0.0

    for i, bar in enumerate(bars[:max_bars]):
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
            return (stop * remain + realized) - cost_bps, i + 1, "SL"

        if tp_type == "fixed":
            if tp_bps > 0 and hi >= tp_bps:
                return (tp_bps * remain + realized) - cost_bps, i + 1, "TP"

        elif tp_type == "trailing":
            if peak > 0 and trail_pct > 0:
                new_stop = peak * (1 - trail_pct)
                if new_stop > stop:
                    stop = new_stop
                if stop > 0 and close_pnl <= stop:
                    return (stop * remain + realized) - cost_bps, i + 1, "TRAIL"

        elif tp_type == "partial_ladder":
            rr = stop_bps
            levels = partial_levels or (1.0, 2.0)
            if pidx < len(levels) and hi >= levels[pidx] * rr:
                frac = min(0.33, remain)
                realized += hi * frac
                remain -= frac
                pidx += 1
                if pidx == 1:
                    stop = max(stop, 0)
            if pidx >= len(levels) and peak > 0:
                new_stop = peak - peak * 0.6
                stop = max(stop, new_stop)
                if close_pnl <= stop:
                    return (stop * remain + realized) - cost_bps, i + 1, "PARTIAL_TRAIL"

    return (close_pnl * remain + realized) - cost_bps, min(len(bars), max_bars), "HOLD"


def run_full_validation(
    *,
    data_dir: Path,
    settings: Any,
    equity_usd: float = 66.0,
    cost_bps_list: list[float] = [16.0, 24.0, 32.0],
    monte_carlo_runs: int = 1000,
) -> dict:
    """Run full validation across all coins, regimes, walk-forward, and Monte Carlo."""
    from quant_binance.features.extractor import MarketFeatureExtractor
    from quant_binance.cost_calibration import load_cost_calibration
    from quant_binance.backtest.historical_fixture_builder import build_historical_slices, _kline_bars_from_raw
    from quant_binance.data.historical_download import load_historical_klines
    from quant_binance.service import PaperTradingService
    from quant_binance.execution.router import ExecutionRouter

    cal_path = data_dir.parent / "artifacts" / "cost_calibration.json"
    extractor = MarketFeatureExtractor(settings, cost_calibration=load_cost_calibration(str(cal_path)))
    service = PaperTradingService(settings, router=ExecutionRouter())

    # Best strategy from grid search
    BEST = {
        "tp_type": "partial_ladder",
        "sl_atr_mult": 3.0,
        "trail_pct": 0.0,
        "partial_levels": (0.5, 1.0),
        "hold_bars": 12,
        "side_filter": "long",
        "adx_min": 40,
        "score_min": 70,
        "leverage": 3,
    }

    # Also test a more relaxed version for comparison
    RELAXED = {
        "tp_type": "partial_ladder",
        "sl_atr_mult": 3.0,
        "trail_pct": 0.0,
        "partial_levels": (0.5, 1.0),
        "hold_bars": 12,
        "side_filter": "long",
        "adx_min": 35,
        "score_min": 65,
        "leverage": 3,
    }

    all_symbols = sorted([d.name for d in data_dir.iterdir() if d.is_dir() and (d / "1h.json").exists()])
    print(f"[validation] {len(all_symbols)} coins, data_dir={data_dir}")

    # ── Step 1: Build slices for ALL coins ────────────────────
    print(f"\n[1/5] Building slices for {len(all_symbols)} coins...")
    all_entries = []
    bars_1h_by_symbol: dict[str, list] = {}

    for symbol in all_symbols:
        k1h = load_historical_klines(data_dir=data_dir, symbol=symbol, interval="1h")
        k4h = load_historical_klines(data_dir=data_dir, symbol=symbol, interval="4h")
        k5m = load_historical_klines(data_dir=data_dir, symbol=symbol, interval="5m")
        if not k1h or len(k1h) < 200:
            continue

        # Parse 1h bars for simulation
        parsed_1h = _kline_bars_from_raw(symbol, "1h", k1h)
        parsed_1h.sort(key=lambda b: b.close_time)
        bars_1h_by_symbol[symbol] = parsed_1h

        slices = build_historical_slices(
            symbol=symbol, klines_5m=k5m or k1h[:0], klines_1h=k1h, klines_4h=k4h or k1h[:0],
            settings=settings, extractor=extractor,
        )

        for sl in slices:
            try:
                decision = service.run_cycle(
                    state=sl.state, primitive_inputs=sl.primitive_inputs,
                    history=sl.history, decision_time=sl.decision_time,
                    equity_usd=equity_usd,
                    remaining_portfolio_capacity_usd=equity_usd * 2.5,
                )
            except Exception:
                continue

            if decision.final_mode not in ("spot", "futures"):
                continue

            entry_price = sl.state.last_trade_price
            if entry_price <= 0:
                continue

            adx = getattr(sl.primitive_inputs, 'adx_1h', 0.0)
            atr = getattr(sl.primitive_inputs, 'atr_1h', 0.0)
            atr_bps = (atr / entry_price) * 10000 if atr > 0 and entry_price > 0 else 50.0

            # Detect regime
            visible_1h = sl.state.klines.get("1h", [])
            regime = _detect_regime(visible_1h)

            entry_ms = int(sl.decision_time.timestamp() * 1000)
            all_entries.append({
                "symbol": symbol, "side": decision.side,
                "score": decision.predictability_score, "adx": adx,
                "entry_price": entry_price, "atr_bps": atr_bps,
                "entry_ms": entry_ms, "entry_time": sl.decision_time,
                "regime": regime,
            })

        print(f"  {symbol}: {len([e for e in all_entries if e['symbol'] == symbol])} entries from {len(slices)} slices")

    all_entries.sort(key=lambda x: x["entry_ms"])
    print(f"  Total: {len(all_entries)} entries across {len(all_symbols)} coins")

    # Pre-compute 1h bar indices using bisect
    sym_ts: dict[str, list[int]] = {}
    for sym, bars in bars_1h_by_symbol.items():
        sym_ts[sym] = [int(b.close_time.timestamp() * 1000) for b in bars]

    # ── Step 2: Run strategies on full dataset ────────────────
    print(f"\n[2/5] Running best strategy on full 375d dataset...")
    results_by_cost: dict[float, dict[str, list[TradeResult]]] = {}

    for cost_bps in cost_bps_list:
        strat_results: dict[str, list[TradeResult]] = {"BEST": [], "RELAXED": []}

        for cfg_name, cfg in [("BEST", BEST), ("RELAXED", RELAXED)]:
            for entry in all_entries:
                if entry["score"] < cfg["score_min"]:
                    continue
                if entry["adx"] < cfg["adx_min"]:
                    continue
                if cfg["side_filter"] == "long" and entry["side"] != "long":
                    continue
                if entry["side"] == "flat":
                    continue

                symbol = entry["symbol"]
                ts_arr = sym_ts.get(symbol)
                bars = bars_1h_by_symbol.get(symbol)
                if not ts_arr or not bars:
                    continue

                idx = bisect.bisect_right(ts_arr, entry["entry_ms"])
                if len(bars) - idx < 3:
                    continue
                future_bars = bars[idx:]

                atr_bps = entry["atr_bps"] if entry["atr_bps"] > 0 else 50.0
                stop_bps = max(atr_bps * cfg["sl_atr_mult"], 15.0)

                pnl, hbars, reason = _sim_1h_bars(
                    side=entry["side"], entry_price=entry["entry_price"],
                    bars=future_bars, stop_bps=stop_bps, max_bars=cfg["hold_bars"],
                    tp_type=cfg["tp_type"], tp_bps=0,
                    trail_pct=cfg["trail_pct"], partial_levels=cfg["partial_levels"],
                    cost_bps=cost_bps,
                )

                lev_pnl = pnl * cfg["leverage"]
                strat_results[cfg_name].append(TradeResult(
                    symbol=symbol, side=entry["side"],
                    entry_time=entry["entry_time"], entry_price=entry["entry_price"],
                    pnl_bps=round(lev_pnl, 2), exit_reason=reason,
                    hold_bars=hbars, regime=entry["regime"],
                ))

        results_by_cost[cost_bps] = strat_results
        for name, trades in strat_results.items():
            wins = sum(1 for t in trades if t.pnl_bps > 0)
            total = sum(t.pnl_bps for t in trades)
            wr = wins / len(trades) if trades else 0
            print(f"  [{name}] cost={cost_bps}bps: {len(trades)} trades, WR={wr*100:.1f}%, total={total:+.1f}bps")

    # ── Step 3: Walk-forward validation ───────────────────────
    print(f"\n[3/5] Walk-forward validation (train 250d → test 125d)...")
    # Split by time: first 67% = train, last 33% = test
    if all_entries:
        split_ms = all_entries[int(len(all_entries) * 0.67)]["entry_ms"]
        train_entries = [e for e in all_entries if e["entry_ms"] < split_ms]
        test_entries = [e for e in all_entries if e["entry_ms"] >= split_ms]
        print(f"  Train: {len(train_entries)} entries, Test: {len(test_entries)} entries")

        for period_name, entries in [("TRAIN", train_entries), ("TEST", test_entries)]:
            for cfg_name, cfg in [("BEST", BEST), ("RELAXED", RELAXED)]:
                trades_pnl = []
                for entry in entries:
                    if entry["score"] < cfg["score_min"] or entry["adx"] < cfg["adx_min"]:
                        continue
                    if cfg["side_filter"] == "long" and entry["side"] != "long":
                        continue
                    if entry["side"] == "flat":
                        continue

                    ts_arr = sym_ts.get(entry["symbol"])
                    bars = bars_1h_by_symbol.get(entry["symbol"])
                    if not ts_arr or not bars:
                        continue
                    idx = bisect.bisect_right(ts_arr, entry["entry_ms"])
                    if len(bars) - idx < 3:
                        continue

                    atr_bps = entry["atr_bps"] if entry["atr_bps"] > 0 else 50.0
                    stop_bps = max(atr_bps * cfg["sl_atr_mult"], 15.0)
                    pnl, _, _ = _sim_1h_bars(
                        side=entry["side"], entry_price=entry["entry_price"],
                        bars=bars[idx:], stop_bps=stop_bps, max_bars=cfg["hold_bars"],
                        tp_type=cfg["tp_type"], tp_bps=0,
                        trail_pct=cfg["trail_pct"], partial_levels=cfg["partial_levels"],
                        cost_bps=16.0,
                    )
                    trades_pnl.append(pnl * cfg["leverage"])

                wins = sum(1 for p in trades_pnl if p > 0)
                wr = wins / len(trades_pnl) if trades_pnl else 0
                total = sum(trades_pnl)
                print(f"  [{period_name}] {cfg_name}: {len(trades_pnl)} trades, WR={wr*100:.1f}%, total={total:+.1f}bps")

    # ── Step 4: Monte Carlo simulation ────────────────────────
    print(f"\n[4/5] Monte Carlo simulation ({monte_carlo_runs} runs)...")
    base_trades = results_by_cost.get(16.0, {}).get("BEST", [])
    if base_trades:
        pnl_list = [t.pnl_bps for t in base_trades]
        ruin_count = 0
        max_dd_list = []
        final_pnl_list = []

        for _ in range(monte_carlo_runs):
            shuffled = pnl_list.copy()
            random.shuffle(shuffled)
            equity = 10000.0  # start at 10000 bps (100%)
            peak_eq = equity
            max_dd = 0.0
            ruined = False

            for p in shuffled:
                equity += p
                peak_eq = max(peak_eq, equity)
                dd = peak_eq - equity
                max_dd = max(max_dd, dd)
                if equity <= 0:
                    ruined = True
                    break

            if ruined:
                ruin_count += 1
            max_dd_list.append(max_dd)
            final_pnl_list.append(equity - 10000.0)

        ruin_pct = ruin_count / monte_carlo_runs * 100
        avg_dd = sum(max_dd_list) / len(max_dd_list)
        p95_dd = sorted(max_dd_list)[int(0.95 * len(max_dd_list))]
        avg_final = sum(final_pnl_list) / len(final_pnl_list)
        p5_final = sorted(final_pnl_list)[int(0.05 * len(final_pnl_list))]

        print(f"  Ruin probability: {ruin_pct:.1f}%")
        print(f"  Avg max drawdown: {avg_dd:.0f}bps")
        print(f"  95th pct drawdown: {p95_dd:.0f}bps")
        print(f"  Avg final PnL: {avg_final:+.0f}bps")
        print(f"  5th pct final PnL (worst 5%): {p5_final:+.0f}bps")

    # ── Step 5: Per-coin & regime breakdown ───────────────────
    print(f"\n[5/5] Per-coin and regime breakdown...")
    base_trades = results_by_cost.get(16.0, {}).get("BEST", [])

    # Per coin
    sym_groups: dict[str, list] = defaultdict(list)
    for t in base_trades:
        sym_groups[t.symbol].append(t)

    print(f"\n  {'Coin':<12} {'Trades':>6} {'WinR':>6} {'AvgPnL':>8} {'TotalBps':>10} {'PF':>6}")
    print(f"  {'-' * 52}")
    for sym in sorted(sym_groups.keys()):
        trades = sym_groups[sym]
        wins = sum(1 for t in trades if t.pnl_bps > 0)
        losses = [t for t in trades if t.pnl_bps <= 0]
        total = sum(t.pnl_bps for t in trades)
        gw = sum(t.pnl_bps for t in trades if t.pnl_bps > 0)
        gl = abs(sum(t.pnl_bps for t in losses)) if losses else 0.01
        wr = wins / len(trades) if trades else 0
        pf = gw / gl if gl > 0 else 999
        avg = total / len(trades) if trades else 0
        print(f"  {sym:<12} {len(trades):>6} {wr*100:>5.1f}% {avg:>+7.1f} {total:>+9.1f} {pf:>5.2f}")

    # Per regime
    regime_groups: dict[str, list] = defaultdict(list)
    for t in base_trades:
        regime_groups[t.regime].append(t)

    print(f"\n  {'Regime':<12} {'Trades':>6} {'WinR':>6} {'AvgPnL':>8} {'TotalBps':>10}")
    print(f"  {'-' * 45}")
    for regime in ["bull", "bear", "sideways"]:
        trades = regime_groups.get(regime, [])
        if not trades:
            print(f"  {regime:<12} {'N/A':>6}")
            continue
        wins = sum(1 for t in trades if t.pnl_bps > 0)
        total = sum(t.pnl_bps for t in trades)
        wr = wins / len(trades)
        avg = total / len(trades)
        print(f"  {regime:<12} {len(trades):>6} {wr*100:>5.1f}% {avg:>+7.1f} {total:>+9.1f}")

    # ── Summary ───────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print(f"  FULL VALIDATION SUMMARY")
    print(f"{'=' * 80}")

    for cost in cost_bps_list:
        trades = results_by_cost.get(cost, {}).get("BEST", [])
        if trades:
            wins = sum(1 for t in trades if t.pnl_bps > 0)
            total = sum(t.pnl_bps for t in trades)
            wr = wins / len(trades)
            gw = sum(t.pnl_bps for t in trades if t.pnl_bps > 0)
            gl = abs(sum(t.pnl_bps for t in trades if t.pnl_bps <= 0)) or 0.01
            print(f"  Cost={cost:>4.0f}bps: {len(trades):>4} trades, WR={wr*100:.1f}%, "
                  f"PnL={total:+.0f}bps, PF={gw/gl:.2f}")

    if base_trades:
        total_pnl = sum(t.pnl_bps for t in base_trades)
        est_usd = total_pnl / 10000 * equity_usd * 0.5
        print(f"\n  Estimated USD PnL (16bps cost): {est_usd:+.2f} USD")
        if 'ruin_pct' in dir():
            print(f"  Monte Carlo ruin probability: {ruin_pct:.1f}%")

    # Save
    output = {
        "coins_tested": len(all_symbols),
        "total_entries": len(all_entries),
        "cost_results": {},
        "walk_forward": {},
        "monte_carlo": {},
    }
    for cost, strats in results_by_cost.items():
        for name, trades in strats.items():
            wins = sum(1 for t in trades if t.pnl_bps > 0)
            output["cost_results"][f"{name}_{cost}bps"] = {
                "trades": len(trades), "wins": wins,
                "win_rate": round(wins / max(len(trades), 1), 4),
                "total_pnl_bps": round(sum(t.pnl_bps for t in trades), 2),
            }
    if base_trades and 'ruin_pct' in dir():
        output["monte_carlo"] = {
            "runs": monte_carlo_runs,
            "ruin_pct": round(ruin_pct, 2),
            "avg_max_dd_bps": round(avg_dd, 2),
            "p95_max_dd_bps": round(p95_dd, 2),
            "avg_final_pnl_bps": round(avg_final, 2),
            "p5_final_pnl_bps": round(p5_final, 2),
        }

    return output


def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Full validation backtest")
    parser.add_argument("--config", default="quant_binance/config.example.json")
    parser.add_argument("--output-base", default="quant_runtime")
    parser.add_argument("--equity-usd", type=float, default=66.0)
    parser.add_argument("--mc-runs", type=int, default=1000)
    args = parser.parse_args(argv)

    os.environ.setdefault("STRATEGY_OVERRIDE_PATH",
                          str(Path(args.output_base) / "artifacts" / "strategy_override.approved.json"))
    os.environ.setdefault("STRATEGY_PROFILE", "live-ultra-aggressive")
    os.environ.setdefault("EXCHANGE", "bitget")

    from quant_binance.settings import Settings
    settings = Settings.load(args.config)
    data_dir = Path(args.output_base) / "historical"

    output = run_full_validation(
        data_dir=data_dir, settings=settings,
        equity_usd=args.equity_usd, monte_carlo_runs=args.mc_runs,
    )

    out_path = Path(args.output_base) / "artifacts" / "full_validation_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n[validation] Results saved to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
