"""Backtest analysis: bucket trades by score/symbol/mode/side, parameter sweep."""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from quant_binance.backtest.batch_backtest import BacktestTrade


# ── Helpers ──────────────────────────────────────────────────────────────────

SCORE_BUCKETS = [(50, 60), (60, 70), (70, 80), (80, 90), (90, 100)]
SWEEP_SCORE_MINS = [55, 60, 65, 70, 75, 80, 85]
SWEEP_COST_BPS = [8, 12, 16, 20]


@dataclass
class BucketStats:
    label: str
    count: int
    win_count: int
    win_rate: float
    avg_gross_bps: float
    avg_net_bps: float
    total_net_pnl_bps: float


def _bucket_stats(label: str, trades: Sequence[BacktestTrade], cost_bps_override: float | None = None) -> BucketStats:
    """Compute stats for a group of trades. Optionally override cost_bps."""
    if not trades:
        return BucketStats(label=label, count=0, win_count=0, win_rate=0.0,
                           avg_gross_bps=0.0, avg_net_bps=0.0, total_net_pnl_bps=0.0)

    nets = []
    grosses = []
    for t in trades:
        g = t.gross_return_bps
        n = g - cost_bps_override if cost_bps_override is not None else t.net_return_bps
        grosses.append(g)
        nets.append(n)

    wins = sum(1 for n in nets if n > 0)
    return BucketStats(
        label=label,
        count=len(trades),
        win_count=wins,
        win_rate=round(wins / len(trades), 4),
        avg_gross_bps=round(sum(grosses) / len(grosses), 2),
        avg_net_bps=round(sum(nets) / len(nets), 2),
        total_net_pnl_bps=round(sum(nets), 2),
    )


def _print_table(title: str, rows: list[BucketStats]) -> None:
    hdr = f"{'Bucket':<20} {'Count':>6} {'Wins':>5} {'WinR%':>6} {'AvgGross':>9} {'AvgNet':>8} {'TotalNet':>10}"
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")
    print(hdr)
    print("-" * 70)
    for r in rows:
        wr = f"{r.win_rate * 100:.1f}" if r.count else "-"
        ag = f"{r.avg_gross_bps:+.1f}" if r.count else "-"
        an = f"{r.avg_net_bps:+.1f}" if r.count else "-"
        tn = f"{r.total_net_pnl_bps:+.0f}" if r.count else "-"
        print(f"{r.label:<20} {r.count:>6} {r.win_count:>5} {wr:>6} {ag:>9} {an:>8} {tn:>10}")
    print()


# ── Build slices + run backtest (reuse run_bootstrap pattern) ────────────────

def _load_trades(args: argparse.Namespace) -> list[BacktestTrade]:
    """Build slices from cached data and run batch backtest once with low score_min."""
    symbols = [s.strip() for s in args.symbols.split(",")]
    intervals = ["5m", "1h", "4h"]
    data_dir = Path(args.output_base) / "historical"
    output_base = Path(args.output_base)

    os.environ.setdefault("STRATEGY_OVERRIDE_PATH",
                          str(output_base / "artifacts" / "strategy_override.approved.json"))
    os.environ.setdefault("STRATEGY_PROFILE", "live-ultra-aggressive")
    os.environ.setdefault("EXCHANGE", "bitget")

    from quant_binance.settings import Settings
    settings = Settings.load(args.config)

    # ── Download (optional) ──
    if not args.skip_download:
        print(f"[1/3] Downloading {args.days}d klines for {symbols}...")
        from quant_binance.execution.client_factory import build_exchange_rest_client
        from quant_binance.data.historical_download import download_all_symbols

        client = build_exchange_rest_client(
            exchange="bitget", allow_insecure_ssl=args.insecure_ssl, allow_missing_credentials=False,
        )
        download_all_symbols(client, symbols=symbols, intervals=intervals,
                             days=args.days, market=args.market, output_dir=data_dir)
    else:
        print("[1/3] Skipping download (--skip-download)")

    # ── Build slices ──
    print("[2/3] Building historical slices...")
    from quant_binance.data.historical_download import load_historical_klines, load_funding_rates, load_spot_klines
    from quant_binance.backtest.historical_fixture_builder import build_historical_slices
    from quant_binance.features.extractor import MarketFeatureExtractor
    from quant_binance.cost_calibration import load_cost_calibration

    cal_path = output_base / "artifacts" / "cost_calibration.json"
    extractor = MarketFeatureExtractor(settings, cost_calibration=load_cost_calibration(str(cal_path)))

    all_slices = []
    for symbol in symbols:
        k5m = load_historical_klines(data_dir=data_dir, symbol=symbol, interval="5m")
        k1h = load_historical_klines(data_dir=data_dir, symbol=symbol, interval="1h")
        k4h = load_historical_klines(data_dir=data_dir, symbol=symbol, interval="4h")
        k1m = load_historical_klines(data_dir=data_dir, symbol=symbol, interval="1m")
        spot_1h = load_spot_klines(data_dir=data_dir, symbol=symbol, interval="1h")
        funding = load_funding_rates(data_dir=data_dir, symbol=symbol)
        if not k1h:
            print(f"  {symbol}: no 1h data, skipping")
            continue
        slices = build_historical_slices(
            symbol=symbol, klines_5m=k5m, klines_1h=k1h, klines_4h=k4h,
            klines_1m=k1m, spot_klines_1h=spot_1h, funding_rates=funding,
            settings=settings, extractor=extractor,
        )
        print(f"  {symbol}: {len(slices)} slices (funding={len(funding or [])}, spot={len(spot_1h or [])}, 1m={len(k1m or [])})")
        all_slices.extend(slices)

    all_slices.sort(key=lambda s: s.decision_time)
    print(f"  Total: {len(all_slices)} slices across {len(symbols)} symbols")

    if not all_slices:
        print("[ERROR] No slices generated. Check data.")
        sys.exit(1)

    # ── Backtest with lowest score_min so we capture all trades ──
    print(f"[3/3] Running batch backtest ({args.holding_period} holding, base cost={args.cost_bps}bps)...")
    from quant_binance.backtest.batch_backtest import run_batch_backtest

    result = run_batch_backtest(
        slices=all_slices,
        settings=settings,
        equity_usd=args.equity_usd,
        capacity_usd=args.equity_usd * 2.5,
        holding_period=args.holding_period,
        cost_bps=args.cost_bps,
    )
    print(f"  Trades: {result.trade_count} | Win rate: {result.win_rate * 100:.1f}% | "
          f"Net PnL: {result.net_pnl_bps:.0f} bps")

    if not result.trades:
        print("[ERROR] No trades generated.")
        sys.exit(1)

    return result.trades


# ── Analysis ─────────────────────────────────────────────────────────────────

def analyze_by_score_bucket(trades: list[BacktestTrade]) -> None:
    rows = []
    for lo, hi in SCORE_BUCKETS:
        group = [t for t in trades if lo <= t.predictability_score < hi]
        rows.append(_bucket_stats(f"[{lo}-{hi})", group))
    # add a total row
    rows.append(_bucket_stats("TOTAL", trades))
    _print_table("TRADES BY PREDICTABILITY SCORE BUCKET", rows)


def analyze_by_symbol(trades: list[BacktestTrade]) -> None:
    by_sym: dict[str, list[BacktestTrade]] = defaultdict(list)
    for t in trades:
        by_sym[t.symbol].append(t)
    rows = [_bucket_stats(sym, trs) for sym, trs in sorted(by_sym.items())]
    rows.append(_bucket_stats("TOTAL", trades))
    _print_table("TRADES BY SYMBOL", rows)


def analyze_by_mode(trades: list[BacktestTrade]) -> None:
    by_mode: dict[str, list[BacktestTrade]] = defaultdict(list)
    for t in trades:
        by_mode[t.mode].append(t)
    rows = [_bucket_stats(mode, trs) for mode, trs in sorted(by_mode.items())]
    rows.append(_bucket_stats("TOTAL", trades))
    _print_table("TRADES BY MODE (futures/spot)", rows)


def analyze_by_side(trades: list[BacktestTrade]) -> None:
    by_side: dict[str, list[BacktestTrade]] = defaultdict(list)
    for t in trades:
        by_side[t.side].append(t)
    rows = [_bucket_stats(side, trs) for side, trs in sorted(by_side.items())]
    rows.append(_bucket_stats("TOTAL", trades))
    _print_table("TRADES BY SIDE (long/short)", rows)


def recommend_score_threshold(trades: list[BacktestTrade], cost_bps: float) -> None:
    """Find the lowest score_min where cumulative net PnL >= 0."""
    print(f"\n{'=' * 70}")
    print(f"  OPTIMAL SCORE_MIN RECOMMENDATION (cost={cost_bps}bps)")
    print(f"{'=' * 70}")

    best_threshold = None
    best_pnl = float("-inf")

    for score_min in range(50, 96):
        above = [t for t in trades if t.predictability_score >= score_min]
        if not above:
            continue
        total_net = sum(t.gross_return_bps - cost_bps for t in above)
        wins = sum(1 for t in above if (t.gross_return_bps - cost_bps) > 0)
        wr = wins / len(above)
        if total_net > best_pnl:
            best_pnl = total_net
            best_threshold = score_min

    # Print table for key thresholds
    hdr = f"{'score_min':>10} {'Count':>6} {'WinR%':>6} {'AvgNet':>8} {'TotalNet':>10}"
    print(hdr)
    print("-" * 45)
    for sm in range(50, 96, 5):
        above = [t for t in trades if t.predictability_score >= sm]
        if not above:
            continue
        nets = [t.gross_return_bps - cost_bps for t in above]
        wins = sum(1 for n in nets if n > 0)
        wr = wins / len(above) * 100
        avg_n = sum(nets) / len(nets)
        tot_n = sum(nets)
        print(f"{sm:>10} {len(above):>6} {wr:>5.1f}% {avg_n:>+7.1f} {tot_n:>+10.0f}")

    # Find lowest threshold where net PnL > 0
    first_positive = None
    for sm in range(50, 96):
        above = [t for t in trades if t.predictability_score >= sm]
        if not above:
            continue
        tot = sum(t.gross_return_bps - cost_bps for t in above)
        if tot > 0 and first_positive is None:
            first_positive = sm

    print()
    if first_positive is not None:
        above = [t for t in trades if t.predictability_score >= first_positive]
        tot = sum(t.gross_return_bps - cost_bps for t in above)
        print(f"  --> Lowest score_min with positive net PnL: {first_positive} "
              f"({len(above)} trades, net={tot:+.0f} bps)")
    else:
        print("  --> No score_min threshold yields positive net PnL at this cost.")

    if best_threshold is not None:
        above = [t for t in trades if t.predictability_score >= best_threshold]
        print(f"  --> Best net PnL at score_min={best_threshold}: "
              f"{best_pnl:+.0f} bps ({len(above)} trades)")
    print()


# ── Parameter Sweep ──────────────────────────────────────────────────────────

@dataclass
class SweepCell:
    score_min: int
    cost_bps: float
    count: int
    win_rate: float
    avg_net_bps: float
    total_net_bps: float


def run_parameter_sweep(trades: list[BacktestTrade]) -> None:
    """Sweep score_min x cost_bps and find profitable parameter combos."""

    print(f"\n{'=' * 80}")
    print(f"  PARAMETER SWEEP: score_min x cost_bps")
    print(f"{'=' * 80}")

    cells: list[SweepCell] = []

    # Header
    header = f"{'score_min':>10}"
    for cb in SWEEP_COST_BPS:
        header += f" | {'cost=' + str(cb):>18}"
    print(header)
    print("-" * (12 + 21 * len(SWEEP_COST_BPS)))

    for sm in SWEEP_SCORE_MINS:
        above = [t for t in trades if t.predictability_score >= sm]
        row = f"{sm:>10}"
        for cb in SWEEP_COST_BPS:
            if not above:
                row += f" | {'-- no trades --':>18}"
                continue
            nets = [t.gross_return_bps - cb for t in above]
            wins = sum(1 for n in nets if n > 0)
            wr = wins / len(above) * 100
            total_net = sum(nets)
            avg_net = total_net / len(above)
            marker = "+" if total_net > 0 else " "
            row += f" |{marker}{len(above):>4}t {wr:>4.0f}% {total_net:>+7.0f}bp"
            cells.append(SweepCell(sm, cb, len(above), wr / 100, round(avg_net, 2), round(total_net, 2)))
        print(row)

    # Find profitable combos
    profitable = [c for c in cells if c.total_net_bps > 0]
    profitable.sort(key=lambda c: c.total_net_bps, reverse=True)

    print(f"\n--- Profitable combinations ({len(profitable)}/{len(cells)}) ---")
    if profitable:
        print(f"{'score_min':>10} {'cost_bps':>9} {'Count':>6} {'WinR%':>6} {'AvgNet':>8} {'TotalNet':>10}")
        print("-" * 55)
        for c in profitable[:15]:
            print(f"{c.score_min:>10} {c.cost_bps:>9.0f} {c.count:>6} "
                  f"{c.win_rate * 100:>5.1f}% {c.avg_net_bps:>+7.1f} {c.total_net_bps:>+10.0f}")
        if len(profitable) > 15:
            print(f"  ... and {len(profitable) - 15} more")

        best = profitable[0]
        print(f"\n  --> BEST: score_min={best.score_min}, cost_bps={best.cost_bps:.0f} "
              f"=> {best.count} trades, WR={best.win_rate * 100:.1f}%, "
              f"net={best.total_net_bps:+.0f} bps")
    else:
        print("  No profitable parameter combinations found.")
    print()


# ── Main ─────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Backtest analysis: bucket + sweep")
    parser.add_argument("--config", default="quant_binance/config.example.json")
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT")
    parser.add_argument("--output-base", default="quant_runtime")
    parser.add_argument("--market", default="futures")
    parser.add_argument("--holding-period", default="4h", choices=["1h", "4h"])
    parser.add_argument("--cost-bps", type=float, default=16.0)
    parser.add_argument("--equity-usd", type=float, default=71.0)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--insecure-ssl", action="store_true")
    parser.add_argument("--sweep-only", action="store_true",
                        help="Skip bucket analysis, run parameter sweep only")
    args = parser.parse_args(argv)

    print(f"{'=' * 70}")
    print(f"  BACKTEST ANALYSIS")
    print(f"  symbols={args.symbols}  days={args.days}  holding={args.holding_period}")
    print(f"  base cost={args.cost_bps}bps  equity=${args.equity_usd}")
    print(f"{'=' * 70}")

    trades = _load_trades(args)

    if not args.sweep_only:
        # ── Bucket analyses ──
        analyze_by_score_bucket(trades)
        analyze_by_symbol(trades)
        analyze_by_mode(trades)
        analyze_by_side(trades)
        recommend_score_threshold(trades, args.cost_bps)

    # ── Parameter sweep ──
    run_parameter_sweep(trades)

    return 0


if __name__ == "__main__":
    sys.exit(main())
