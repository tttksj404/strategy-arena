"""Take-profit strategy backtest — compare multiple exit strategies on 5m bars.

Tests 6 strategies:
  1. HOLD_ONLY      — hold for N bars, no TP/SL (current baseline)
  2. FIXED_TP_SL    — fixed R:R take-profit and stop-loss
  3. TRAILING_STOP  — trail stop at X% of peak profit
  4. PARTIAL_LADDER — take partial profits at 1R, 2R, trail rest
  5. TIME_DECAY_TP  — tighten TP target as holding time increases
  6. BREAKEVEN_LOCK — move stop to breakeven after hitting 1R profit

Uses 5m klines for intra-bar resolution on top of the existing
historical_fixture_builder slices.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class TPStrategy:
    name: str
    description: str


@dataclass
class SimulatedTrade:
    symbol: str
    side: str
    entry_time: datetime
    entry_price: float
    exit_time: datetime
    exit_price: float
    exit_reason: str
    pnl_bps: float
    peak_pnl_bps: float
    worst_pnl_bps: float
    hold_bars_5m: int
    leverage: int
    notional_usd: float


@dataclass
class StrategyResult:
    strategy_name: str
    trades: list[SimulatedTrade]
    total_pnl_bps: float
    total_pnl_usd: float
    win_count: int
    loss_count: int
    win_rate: float
    avg_pnl_bps: float
    avg_winner_bps: float
    avg_loser_bps: float
    profit_factor: float
    max_drawdown_bps: float
    avg_hold_bars: float
    best_trade_bps: float
    worst_trade_bps: float


# ── Bar-by-bar simulation engine ──────────────────────────────────────────────

def _simulate_trade_bars(
    *,
    side: str,
    entry_price: float,
    bars_5m: list,
    stop_bps: float,
    max_bars: int,
    strategy: str,
    # Strategy-specific params
    tp_bps: float = 0.0,
    trail_pct: float = 0.5,        # trailing stop: lock this fraction of peak
    partial_levels: list[float] | None = None,  # [1.0, 2.0] = 1R, 2R
    time_decay_factor: float = 0.0,
    breakeven_trigger_bps: float = 0.0,
    cost_bps: float = 16.0,
) -> tuple[float, float, float, int, str]:
    """Simulate a single trade bar-by-bar through 5m candles.

    Returns: (net_pnl_bps, peak_pnl_bps, worst_pnl_bps, hold_bars, exit_reason)
    """
    if not bars_5m or entry_price <= 0:
        return -cost_bps, 0.0, 0.0, 0, "NO_DATA"

    peak_pnl = 0.0
    worst_pnl = 0.0
    current_stop_bps = -stop_bps  # negative = loss
    realized_partial = 0.0
    remaining_fraction = 1.0
    breakeven_armed = False
    partial_idx = 0

    for i, bar in enumerate(bars_5m[:max_bars]):
        # Calculate current PnL in bps
        if side == "long":
            high_pnl = ((bar.high_price / entry_price) - 1) * 10000
            low_pnl = ((bar.low_price / entry_price) - 1) * 10000
            close_pnl = ((bar.close_price / entry_price) - 1) * 10000
        else:
            high_pnl = (1 - (bar.low_price / entry_price)) * 10000
            low_pnl = (1 - (bar.high_price / entry_price)) * 10000
            close_pnl = (1 - (bar.close_price / entry_price)) * 10000

        peak_pnl = max(peak_pnl, high_pnl)
        worst_pnl = min(worst_pnl, low_pnl)

        # ── Check stop-loss first (worst case within bar) ──
        if low_pnl <= current_stop_bps:
            net = (current_stop_bps * remaining_fraction + realized_partial) - cost_bps
            return net, peak_pnl, worst_pnl, i + 1, "STOP_LOSS"

        # ── Strategy-specific exit logic ──
        if strategy == "FIXED_TP_SL":
            if tp_bps > 0 and high_pnl >= tp_bps:
                net = (tp_bps * remaining_fraction + realized_partial) - cost_bps
                return net, peak_pnl, worst_pnl, i + 1, "TAKE_PROFIT"

        elif strategy == "TRAILING_STOP":
            if peak_pnl > 0:
                trail_stop = peak_pnl * trail_pct
                if trail_stop > -current_stop_bps:  # only tighten, never widen
                    current_stop_bps = max(current_stop_bps, peak_pnl - trail_stop)
                    # If trailing stop passed breakeven, lock it
                    if current_stop_bps > 0 and close_pnl <= current_stop_bps:
                        net = (current_stop_bps * remaining_fraction + realized_partial) - cost_bps
                        return net, peak_pnl, worst_pnl, i + 1, "TRAILING_STOP"

        elif strategy == "PARTIAL_LADDER":
            levels = partial_levels or [1.0, 2.0]
            rr_bps = stop_bps  # 1R = stop distance
            if partial_idx < len(levels) and high_pnl >= levels[partial_idx] * rr_bps:
                # Take 33% at each level
                take_fraction = min(0.33, remaining_fraction)
                realized_partial += high_pnl * take_fraction
                remaining_fraction -= take_fraction
                partial_idx += 1
                # Move stop to breakeven after first partial
                if partial_idx == 1:
                    current_stop_bps = max(current_stop_bps, 0)
            # Trail the remainder after all levels hit
            if partial_idx >= len(levels) and peak_pnl > 0:
                trail_val = peak_pnl * 0.6
                current_stop_bps = max(current_stop_bps, peak_pnl - trail_val)
                if close_pnl <= current_stop_bps:
                    net = (current_stop_bps * remaining_fraction + realized_partial) - cost_bps
                    return net, peak_pnl, worst_pnl, i + 1, "PARTIAL_TRAIL_EXIT"

        elif strategy == "TIME_DECAY_TP":
            # Target shrinks over time: full TP at start, tighter as time passes
            progress = min(i / max(max_bars * 0.7, 1), 1.0)
            current_tp = tp_bps * (1.0 - progress * time_decay_factor)
            current_tp = max(current_tp, stop_bps * 0.3)  # floor at 0.3R
            if high_pnl >= current_tp:
                net = (current_tp * remaining_fraction) - cost_bps
                return net, peak_pnl, worst_pnl, i + 1, "TIME_DECAY_TP"

        elif strategy == "BREAKEVEN_LOCK":
            if not breakeven_armed and high_pnl >= breakeven_trigger_bps:
                breakeven_armed = True
                current_stop_bps = max(current_stop_bps, cost_bps * 0.5)  # lock at half-cost
            # Still use fixed TP
            if tp_bps > 0 and high_pnl >= tp_bps:
                net = (tp_bps * remaining_fraction) - cost_bps
                return net, peak_pnl, worst_pnl, i + 1, "TAKE_PROFIT"

    # Hold period expired — exit at last close
    final_pnl = close_pnl if bars_5m else 0.0
    net = (final_pnl * remaining_fraction + realized_partial) - cost_bps
    return net, peak_pnl, worst_pnl, min(len(bars_5m), max_bars), "MAX_HOLD_TIME"


# ── Strategy runner ───────────────────────────────────────────────────────────

def run_tp_backtest(
    *,
    slices: list,
    bars_5m_by_symbol: dict[str, list],
    settings: Any,
    equity_usd: float = 66.0,
    cost_bps: float = 16.0,
) -> dict[str, StrategyResult]:
    """Run all TP strategies on the same set of entry decisions."""
    from quant_binance.service import PaperTradingService
    from quant_binance.execution.router import ExecutionRouter
    from quant_binance.strategy.coin_profiles import get_profile, is_profiled

    service = PaperTradingService(settings, router=ExecutionRouter())

    # Define strategies with their parameters
    strategies = {
        "HOLD_ONLY": {"tp_bps": 0, "trail_pct": 0},
        "FIXED_TP_SL": {},  # uses coin profile RR
        "TRAILING_STOP": {"trail_pct": 0.5},
        "PARTIAL_LADDER": {"partial_levels": [1.0, 2.0]},
        "TIME_DECAY_TP": {"time_decay_factor": 0.7},
        "BREAKEVEN_LOCK": {},  # breakeven at 1R, TP at profile RR
    }

    results: dict[str, list[SimulatedTrade]] = {name: [] for name in strategies}
    entry_count = 0
    skip_count = 0

    for idx, sl in enumerate(slices):
        try:
            decision = service.run_cycle(
                state=sl.state,
                primitive_inputs=sl.primitive_inputs,
                history=sl.history,
                decision_time=sl.decision_time,
                equity_usd=equity_usd,
                remaining_portfolio_capacity_usd=equity_usd * 2.5,
            )
        except Exception:
            continue

        if decision.final_mode not in ("spot", "futures"):
            continue

        symbol = sl.symbol
        side = decision.side
        entry_price = sl.state.last_trade_price
        entry_time = sl.decision_time

        if entry_price <= 0:
            continue

        # Get coin profile for stop/TP calculation
        profile = get_profile(symbol)
        leverage = decision.futures_leverage if hasattr(decision, 'futures_leverage') else profile.optimal_leverage

        # Calculate stop distance from ATR
        atr_bps = 0.0
        if hasattr(sl.primitive_inputs, 'atr_1h') and sl.primitive_inputs.atr_1h > 0:
            atr_bps = (sl.primitive_inputs.atr_1h / entry_price) * 10000
        else:
            # Fallback: estimate from recent 1h bars
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
            atr_bps = 50.0  # fallback

        stop_bps = atr_bps * profile.sl_atr_mult
        stop_bps = max(stop_bps, 20.0)  # minimum stop
        tp_target_bps = stop_bps * profile.rr

        # Get 5m bars after entry for simulation
        all_5m = bars_5m_by_symbol.get(symbol, [])
        entry_ms = int(entry_time.timestamp() * 1000)
        future_5m = [b for b in all_5m if int(b.close_time.timestamp() * 1000) > entry_ms]

        if len(future_5m) < 3:
            skip_count += 1
            continue

        # Max hold in 5m bars (from coin profile hold_bars in 1h)
        max_bars_5m = profile.hold_bars * 12  # 1h = 12 x 5m

        entry_count += 1
        notional = decision.order_intent_notional_usd if hasattr(decision, 'order_intent_notional_usd') else equity_usd * 0.5

        # Run each strategy on the same entry
        for strat_name, params in strategies.items():
            if strat_name == "HOLD_ONLY":
                net_pnl, peak, worst, hold, reason = _simulate_trade_bars(
                    side=side, entry_price=entry_price, bars_5m=future_5m,
                    stop_bps=stop_bps, max_bars=max_bars_5m,
                    strategy="HOLD_ONLY", cost_bps=cost_bps,
                )
            elif strat_name == "FIXED_TP_SL":
                net_pnl, peak, worst, hold, reason = _simulate_trade_bars(
                    side=side, entry_price=entry_price, bars_5m=future_5m,
                    stop_bps=stop_bps, max_bars=max_bars_5m,
                    strategy="FIXED_TP_SL", tp_bps=tp_target_bps, cost_bps=cost_bps,
                )
            elif strat_name == "TRAILING_STOP":
                net_pnl, peak, worst, hold, reason = _simulate_trade_bars(
                    side=side, entry_price=entry_price, bars_5m=future_5m,
                    stop_bps=stop_bps, max_bars=max_bars_5m,
                    strategy="TRAILING_STOP", trail_pct=0.5, cost_bps=cost_bps,
                )
            elif strat_name == "PARTIAL_LADDER":
                net_pnl, peak, worst, hold, reason = _simulate_trade_bars(
                    side=side, entry_price=entry_price, bars_5m=future_5m,
                    stop_bps=stop_bps, max_bars=max_bars_5m,
                    strategy="PARTIAL_LADDER",
                    partial_levels=[1.0, 2.0], cost_bps=cost_bps,
                )
            elif strat_name == "TIME_DECAY_TP":
                net_pnl, peak, worst, hold, reason = _simulate_trade_bars(
                    side=side, entry_price=entry_price, bars_5m=future_5m,
                    stop_bps=stop_bps, max_bars=max_bars_5m,
                    strategy="TIME_DECAY_TP", tp_bps=tp_target_bps,
                    time_decay_factor=0.7, cost_bps=cost_bps,
                )
            elif strat_name == "BREAKEVEN_LOCK":
                net_pnl, peak, worst, hold, reason = _simulate_trade_bars(
                    side=side, entry_price=entry_price, bars_5m=future_5m,
                    stop_bps=stop_bps, max_bars=max_bars_5m,
                    strategy="BREAKEVEN_LOCK", tp_bps=tp_target_bps,
                    breakeven_trigger_bps=stop_bps * 1.0, cost_bps=cost_bps,
                )
            else:
                continue

            exit_price = entry_price * (1 + net_pnl / 10000) if side == "long" else entry_price * (1 - net_pnl / 10000)

            results[strat_name].append(SimulatedTrade(
                symbol=symbol,
                side=side,
                entry_time=entry_time,
                entry_price=entry_price,
                exit_time=entry_time,  # simplified
                exit_price=exit_price,
                exit_reason=reason,
                pnl_bps=round(net_pnl, 2),
                peak_pnl_bps=round(peak, 2),
                worst_pnl_bps=round(worst, 2),
                hold_bars_5m=hold,
                leverage=leverage,
                notional_usd=notional,
            ))

        if entry_count % 200 == 0:
            print(f"  [tp-backtest] {entry_count} entries processed ({idx}/{len(slices)} slices)...", flush=True)

    print(f"  [tp-backtest] Done: {entry_count} entries, {skip_count} skipped (no 5m data)")

    # ── Compile results ───────────────────────────────────────────────────────
    strategy_results = {}
    for strat_name, trades in results.items():
        if not trades:
            strategy_results[strat_name] = StrategyResult(
                strategy_name=strat_name, trades=[], total_pnl_bps=0, total_pnl_usd=0,
                win_count=0, loss_count=0, win_rate=0, avg_pnl_bps=0,
                avg_winner_bps=0, avg_loser_bps=0, profit_factor=0,
                max_drawdown_bps=0, avg_hold_bars=0, best_trade_bps=0, worst_trade_bps=0,
            )
            continue

        winners = [t for t in trades if t.pnl_bps > 0]
        losers = [t for t in trades if t.pnl_bps <= 0]
        total_pnl = sum(t.pnl_bps for t in trades)
        gross_win = sum(t.pnl_bps for t in winners) if winners else 0
        gross_loss = abs(sum(t.pnl_bps for t in losers)) if losers else 0

        # Equity curve drawdown
        equity_curve = []
        running = 0.0
        peak_equity = 0.0
        max_dd = 0.0
        for t in trades:
            running += t.pnl_bps
            equity_curve.append(running)
            peak_equity = max(peak_equity, running)
            dd = peak_equity - running
            max_dd = max(max_dd, dd)

        # Approximate USD PnL (leverage-weighted)
        avg_notional = sum(t.notional_usd for t in trades) / len(trades)
        total_pnl_usd = total_pnl / 10000 * avg_notional

        strategy_results[strat_name] = StrategyResult(
            strategy_name=strat_name,
            trades=trades,
            total_pnl_bps=round(total_pnl, 2),
            total_pnl_usd=round(total_pnl_usd, 2),
            win_count=len(winners),
            loss_count=len(losers),
            win_rate=round(len(winners) / len(trades), 4),
            avg_pnl_bps=round(total_pnl / len(trades), 2),
            avg_winner_bps=round(gross_win / max(len(winners), 1), 2),
            avg_loser_bps=round(-gross_loss / max(len(losers), 1), 2),
            profit_factor=round(gross_win / max(gross_loss, 0.01), 2),
            max_drawdown_bps=round(max_dd, 2),
            avg_hold_bars=round(sum(t.hold_bars_5m for t in trades) / len(trades), 1),
            best_trade_bps=round(max(t.pnl_bps for t in trades), 2),
            worst_trade_bps=round(min(t.pnl_bps for t in trades), 2),
        )

    return strategy_results


# ── Print report ──────────────────────────────────────────────────────────────

def print_report(results: dict[str, StrategyResult], equity_usd: float = 66.0) -> None:
    print("\n" + "=" * 90)
    print("  TAKE-PROFIT STRATEGY COMPARISON BACKTEST")
    print("=" * 90)

    # Sort by total PnL
    sorted_strats = sorted(results.values(), key=lambda r: r.total_pnl_bps, reverse=True)

    print(f"\n{'Strategy':<20} {'Trades':>6} {'WinR':>6} {'AvgPnL':>8} {'TotalBps':>9} "
          f"{'PnL$':>8} {'PF':>6} {'MaxDD':>8} {'AvgHold':>8} {'Best':>8} {'Worst':>8}")
    print("-" * 110)

    for r in sorted_strats:
        hold_str = f"{r.avg_hold_bars:.0f}x5m"
        pnl_sign = "+" if r.total_pnl_usd >= 0 else ""
        print(f"{r.strategy_name:<20} {r.win_count+r.loss_count:>6} "
              f"{r.win_rate*100:>5.1f}% {r.avg_pnl_bps:>+7.1f} {r.total_pnl_bps:>+8.1f} "
              f"{pnl_sign}{r.total_pnl_usd:>7.2f} {r.profit_factor:>5.2f} "
              f"{r.max_drawdown_bps:>7.1f} {hold_str:>8} {r.best_trade_bps:>+7.1f} {r.worst_trade_bps:>+7.1f}")

    # Per-symbol breakdown for best strategy
    best = sorted_strats[0]
    print(f"\n{'─' * 90}")
    print(f"  BEST STRATEGY: {best.strategy_name}")
    print(f"{'─' * 90}")

    # Symbol breakdown
    from collections import defaultdict
    sym_trades: dict[str, list] = defaultdict(list)
    for t in best.trades:
        sym_trades[t.symbol].append(t)

    print(f"\n  {'Symbol':<12} {'Trades':>6} {'WinR':>6} {'AvgPnL':>8} {'TotalBps':>9} {'AvgHold':>8}")
    print(f"  {'-' * 55}")
    for sym in sorted(sym_trades.keys()):
        trades = sym_trades[sym]
        wins = sum(1 for t in trades if t.pnl_bps > 0)
        total = sum(t.pnl_bps for t in trades)
        avg_h = sum(t.hold_bars_5m for t in trades) / len(trades)
        print(f"  {sym:<12} {len(trades):>6} {wins/len(trades)*100:>5.1f}% "
              f"{total/len(trades):>+7.1f} {total:>+8.1f} {avg_h:>6.0f}x5m")

    # Exit reason breakdown for each strategy
    print(f"\n{'─' * 90}")
    print(f"  EXIT REASON BREAKDOWN")
    print(f"{'─' * 90}")

    for r in sorted_strats:
        reason_count: dict[str, int] = defaultdict(int)
        reason_pnl: dict[str, float] = defaultdict(float)
        for t in r.trades:
            reason_count[t.exit_reason] += 1
            reason_pnl[t.exit_reason] += t.pnl_bps

        print(f"\n  {r.strategy_name}:")
        for reason in sorted(reason_count.keys(), key=lambda x: reason_pnl[x], reverse=True):
            cnt = reason_count[reason]
            pnl = reason_pnl[reason]
            avg = pnl / cnt
            print(f"    {reason:<25} {cnt:>4}건  total={pnl:>+8.1f}bps  avg={avg:>+7.1f}bps")

    # Recommendation
    print(f"\n{'=' * 90}")
    print(f"  RECOMMENDATION")
    print(f"{'=' * 90}")
    if best.total_pnl_bps > 0:
        pnl_pct = (best.total_pnl_usd / equity_usd) * 100
        print(f"\n  >>> {best.strategy_name} 전략이 최적 ({best.total_pnl_usd:+.2f} USD, {pnl_pct:+.1f}%)")
        print(f"      승률 {best.win_rate*100:.1f}%, PF {best.profit_factor:.2f}, 최대 DD {best.max_drawdown_bps:.0f}bps")
        if best.strategy_name != "HOLD_ONLY":
            hold_result = results.get("HOLD_ONLY")
            if hold_result:
                improvement = best.total_pnl_bps - hold_result.total_pnl_bps
                print(f"      HOLD_ONLY 대비 {improvement:+.1f}bps 개선")
    else:
        print(f"\n  >>> 모든 전략이 손실. 진입 로직 자체의 개선이 필요합니다.")
    print()


# ── CLI entry point ───────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="TP Strategy Backtest — compare exit strategies")
    parser.add_argument("--config", default="quant_binance/config.example.json")
    parser.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT")
    parser.add_argument("--output-base", default="quant_runtime")
    parser.add_argument("--equity-usd", type=float, default=66.0)
    parser.add_argument("--cost-bps", type=float, default=16.0)
    parser.add_argument("--skip-download", action="store_true", default=True)
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

    settings = Settings.load(args.config)
    cal_path = Path(args.output_base) / "artifacts" / "cost_calibration.json"
    extractor = MarketFeatureExtractor(settings, cost_calibration=load_cost_calibration(str(cal_path)))

    print(f"[tp-backtest] Loading data for {symbols}...")

    all_slices = []
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

        # Parse 5m bars for simulation
        parsed_5m = []
        for row in k5m:
            try:
                parsed_5m.append(_parse_kline(symbol, "5m", row))
            except Exception:
                continue
        parsed_5m.sort(key=lambda b: b.close_time)
        bars_5m_by_symbol[symbol] = parsed_5m

        slices = build_historical_slices(
            symbol=symbol,
            klines_5m=k5m,
            klines_1h=k1h,
            klines_4h=k4h,
            klines_1m=k1m,
            spot_klines_1h=spot_1h,
            funding_rates=funding,
            settings=settings,
            extractor=extractor,
        )
        print(f"  {symbol}: {len(slices)} slices, {len(parsed_5m)} 5m bars")
        all_slices.extend(slices)

    all_slices.sort(key=lambda s: s.decision_time)
    print(f"  Total: {len(all_slices)} slices")

    if not all_slices:
        print("[ERROR] No slices generated.")
        return 1

    # Run backtest
    print(f"\n[tp-backtest] Running 6 TP strategies on {len(all_slices)} slices...")
    results = run_tp_backtest(
        slices=all_slices,
        bars_5m_by_symbol=bars_5m_by_symbol,
        settings=settings,
        equity_usd=args.equity_usd,
        cost_bps=args.cost_bps,
    )

    # Print comparison report
    print_report(results, equity_usd=args.equity_usd)

    # Save results to JSON
    output_path = Path(args.output_base) / "artifacts" / "tp_strategy_backtest_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {}
    for name, r in results.items():
        serializable[name] = {
            "strategy_name": r.strategy_name,
            "total_pnl_bps": r.total_pnl_bps,
            "total_pnl_usd": r.total_pnl_usd,
            "win_count": r.win_count,
            "loss_count": r.loss_count,
            "win_rate": r.win_rate,
            "avg_pnl_bps": r.avg_pnl_bps,
            "avg_winner_bps": r.avg_winner_bps,
            "avg_loser_bps": r.avg_loser_bps,
            "profit_factor": r.profit_factor,
            "max_drawdown_bps": r.max_drawdown_bps,
            "avg_hold_bars_5m": r.avg_hold_bars,
            "best_trade_bps": r.best_trade_bps,
            "worst_trade_bps": r.worst_trade_bps,
            "trade_count": len(r.trades),
        }
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"[tp-backtest] Results saved to {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
