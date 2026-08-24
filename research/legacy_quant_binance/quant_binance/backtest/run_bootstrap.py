"""CLI orchestrator: download historical data → backtest → bootstrap edge table."""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Historical data bootstrap for edge learner")
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
    args = parser.parse_args(argv)

    symbols = [s.strip() for s in args.symbols.split(",")]
    intervals = ["5m", "1h", "4h"]
    data_dir = Path(args.output_base) / "historical"
    output_base = Path(args.output_base)

    os.environ.setdefault("STRATEGY_OVERRIDE_PATH", str(output_base / "artifacts" / "strategy_override.approved.json"))
    os.environ.setdefault("STRATEGY_PROFILE", "live-ultra-aggressive")
    os.environ.setdefault("EXCHANGE", "bitget")

    from quant_binance.settings import Settings
    settings = Settings.load(args.config)
    print(f"[bootstrap] settings loaded: universe={settings.universe} score_min={settings.mode_thresholds.futures_score_min}")

    # ── Step 1: Download ──────────────────────────────────
    if not args.skip_download:
        print(f"\n[1/4] Downloading {args.days}d klines for {symbols}...")
        from quant_binance.execution.client_factory import build_exchange_rest_client
        from quant_binance.data.historical_download import download_all_symbols

        client = build_exchange_rest_client(
            exchange="bitget",
            allow_insecure_ssl=args.insecure_ssl,
            allow_missing_credentials=False,
        )
        counts = download_all_symbols(
            client,
            symbols=symbols,
            intervals=intervals,
            days=args.days,
            market=args.market,
            output_dir=data_dir,
        )
        for sym, ivs in counts.items():
            print(f"  {sym}: {ivs}")
    else:
        print("[1/4] Skipping download (--skip-download)")

    # ── Step 2: Build slices ──────────────────────────────
    print(f"\n[2/4] Building historical slices...")
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
        extra_info = []
        if funding:
            extra_info.append(f"{len(funding)} funding rates")
        if spot_1h:
            extra_info.append(f"{len(spot_1h)} spot bars")
        if k1m:
            extra_info.append(f"{len(k1m)} 1m bars")
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
        print(f"  {symbol}: {len(slices)} slices from {len(k1h)} 1h bars" + (f" + {', '.join(extra_info)}" if extra_info else ""))
        all_slices.extend(slices)

    all_slices.sort(key=lambda s: s.decision_time)
    print(f"  Total: {len(all_slices)} slices across {len(symbols)} symbols")

    if not all_slices:
        print("[ERROR] No slices generated. Check data download.")
        return 1

    # ── Step 3: Backtest ──────────────────────────────────
    print(f"\n[3/4] Running batch backtest ({args.holding_period} holding, {args.cost_bps}bps cost)...")
    from quant_binance.backtest.batch_backtest import run_batch_backtest

    result = run_batch_backtest(
        slices=all_slices,
        settings=settings,
        equity_usd=args.equity_usd,
        capacity_usd=args.equity_usd * 2.5,
        holding_period=args.holding_period,
        cost_bps=args.cost_bps,
    )
    print(f"  Decisions: {result.total_decisions} (cash: {result.cash_decisions})")
    print(f"  Trades: {result.trade_count} (win: {result.win_count}, loss: {result.loss_count})")
    print(f"  Win rate: {result.win_rate * 100:.1f}%")
    print(f"  Gross PnL: {result.gross_pnl_bps:.0f} bps")
    print(f"  Net PnL:   {result.net_pnl_bps:.0f} bps (after {args.cost_bps}bps cost)")

    if not result.trades:
        print("[ERROR] No trades generated. Check strategy thresholds.")
        return 1

    # ── Step 4: Bootstrap edge table ──────────────────────
    print(f"\n[4/4] Bootstrapping edge_table.json...")
    from quant_binance.backtest.bootstrap_edge import bootstrap_edge_table

    edge_output = output_base / "output" / "paper-live-shell" / "bootstrap" / "edge_table.json"
    edge_output.parent.mkdir(parents=True, exist_ok=True)

    diag = bootstrap_edge_table(
        trades=result.trades,
        output_path=edge_output,
    )
    print(f"  Observations: {diag['observation_count']}")
    print(f"  Symbols: {diag['symbols']}")
    print(f"  Learning active: {diag['diagnostics'].get('learning_active')}")
    print(f"  Output: {diag['output_path']}")

    # ── Summary ───────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"BOOTSTRAP COMPLETE")
    print(f"  {result.trade_count} trades → {diag['observation_count']} edge observations")
    print(f"  Win rate: {result.win_rate * 100:.1f}%")
    print(f"  Net PnL: {result.net_pnl_bps:.0f} bps")
    print(f"  Edge table: {edge_output}")
    print(f"  Restart daemon to load: kill daemon → supervisor auto-restart")
    print(f"{'='*50}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
