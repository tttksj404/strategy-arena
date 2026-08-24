"""G140-G145 변형 sweep.

전략:
- G140 (UTC 06-09 filter): G135 trades 를 entry_time UTC hour 로 post-filter
- G141 (hold 1h): re-run with --holding-period 1h
- G143 (short_disabled false): re-run with override
- G144 (UTC + 1h hold): G141 trades 를 post-filter
- G145 SKIP — G135 이미 funding_rate_strategy.enabled=True (duplicate)
- G142 SKIP — analyze_backtest 가 8h hold 미지원

각 변형의 trades 를 직접 in-process 로 backtest 하여 entry_time 까지 보유 후
G135 baseline 대비 비교.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

REPO = Path.home() / "Desktop" / "first_repo"
sys.path.insert(0, str(REPO))

STRATS = REPO / "quant_binance" / "strategies"
G135 = STRATS / "G135_score76" / "overrides.json"

SYMBOLS = ["ETHUSDT", "SOLUSDT", "DOGEUSDT", "PEPEUSDT"]
DAYS = 60
EQUITY = 55.0
COST_BPS = 16.0
SCORE_MIN = 76  # G135 lock
UTC_FILTER_LO = 6
UTC_FILTER_HI = 9  # inclusive (6,7,8,9)


def make_variant(name: str, parent_path: Path, changes: dict, parent_id: str = "G135"):
    """Create variant override dir + json file."""
    src = json.load(open(parent_path, encoding="utf-8"))
    folder = STRATS / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "runs").mkdir(exist_ok=True)
    for path, val in changes.items():
        keys = path.split(".")
        cur = src
        for k in keys[:-1]:
            cur = cur[k]
        cur[keys[-1]] = val
    src["_strategy_id"] = name.split("_")[0]
    src["_parent_id"] = parent_id
    src["_changed_keys_vs_parent"] = list(changes.keys())
    out = folder / "overrides.json"
    json.dump(src, open(out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    return out


def run_backtest(override_path: Path, holding: str = "4h"):
    """Run backtest in-process; return list of BacktestTrade."""
    os.environ["STRATEGY_OVERRIDE_PATH"] = str(override_path)
    os.environ.setdefault("STRATEGY_PROFILE", "live-ultra-aggressive")
    os.environ.setdefault("EXCHANGE", "bitget")

    from quant_binance.settings import Settings
    from quant_binance.data.historical_download import (
        load_historical_klines, load_funding_rates, load_spot_klines,
    )
    from quant_binance.backtest.historical_fixture_builder import build_historical_slices
    from quant_binance.features.extractor import MarketFeatureExtractor
    from quant_binance.cost_calibration import load_cost_calibration
    from quant_binance.backtest.batch_backtest import run_batch_backtest

    settings = Settings.load(REPO / "quant_binance" / "config.example.json")
    output_base = REPO / "quant_runtime"
    data_dir = output_base / "historical"

    cal_path = output_base / "artifacts" / "cost_calibration.json"
    extractor = MarketFeatureExtractor(settings, cost_calibration=load_cost_calibration(str(cal_path)))

    all_slices = []
    for symbol in SYMBOLS:
        k5m = load_historical_klines(data_dir=data_dir, symbol=symbol, interval="5m")
        k1h = load_historical_klines(data_dir=data_dir, symbol=symbol, interval="1h")
        k4h = load_historical_klines(data_dir=data_dir, symbol=symbol, interval="4h")
        k1m = load_historical_klines(data_dir=data_dir, symbol=symbol, interval="1m")
        spot_1h = load_spot_klines(data_dir=data_dir, symbol=symbol, interval="1h")
        funding = load_funding_rates(data_dir=data_dir, symbol=symbol)
        if not k1h:
            print(f"  {symbol}: no 1h, skip")
            continue
        slices = build_historical_slices(
            symbol=symbol, klines_5m=k5m, klines_1h=k1h, klines_4h=k4h,
            klines_1m=k1m, spot_klines_1h=spot_1h, funding_rates=funding,
            settings=settings, extractor=extractor,
        )
        # truncate to last DAYS
        from datetime import timedelta, timezone
        cutoff = datetime.now(timezone.utc) - timedelta(days=DAYS)
        slices = [s for s in slices if s.decision_time >= cutoff]
        print(f"  {symbol}: {len(slices)} slices (last {DAYS}d)")
        all_slices.extend(slices)

    all_slices.sort(key=lambda s: s.decision_time)
    if not all_slices:
        print("  [ERR] no slices")
        return []

    result = run_batch_backtest(
        slices=all_slices, settings=settings,
        equity_usd=EQUITY, capacity_usd=EQUITY * 2.5,
        holding_period=holding, cost_bps=COST_BPS,
    )
    print(f"  -> {result.trade_count} trades, WR {result.win_rate*100:.1f}%, total_net {result.net_pnl_bps:.0f} bps")
    return result.trades


def filter_score(trades, score_min: int):
    return [t for t in trades if t.predictability_score >= score_min]


def filter_utc_hour(trades, lo: int, hi: int):
    """lo/hi inclusive."""
    out = []
    for t in trades:
        h = t.entry_time.hour
        if lo <= h <= hi:
            out.append(t)
    return out


def stats(trades, label: str):
    if not trades:
        return {"label": label, "n": 0, "wr": 0.0, "avg_net": 0.0, "total_net": 0, "annual_pct": 0.0}
    n = len(trades)
    wins = sum(1 for t in trades if t.net_return_bps > 0)
    total_net = sum(t.net_return_bps for t in trades)
    avg_net = total_net / n
    # annual estimate: total_net bps over DAYS days, scaled to 365, applied as % of equity
    # rough: total_net / 10000 * equity_usd / 1 = USD return; annualize *365/DAYS
    # simpler: avg per trade * trades/day * 365 / 10000 * 100 (as %)
    annual_bps = total_net * 365 / DAYS
    annual_pct = annual_bps / 100  # bps to %
    return {
        "label": label, "n": n, "wr": round(wins / n * 100, 1),
        "avg_net": round(avg_net, 1), "total_net": round(total_net, 0),
        "annual_pct": round(annual_pct, 1),
    }


def hour_distribution(trades):
    """Return dict {hour: (n, wins, total_net_bps)}."""
    by_h = defaultdict(lambda: [0, 0, 0.0])
    for t in trades:
        h = t.entry_time.hour
        by_h[h][0] += 1
        if t.net_return_bps > 0:
            by_h[h][1] += 1
        by_h[h][2] += t.net_return_bps
    return by_h


def main():
    print("=" * 78)
    print("  G140-G145 SWEEP")
    print(f"  symbols={SYMBOLS}  days={DAYS}  equity=${EQUITY}  cost={COST_BPS}bps")
    print(f"  score_min={SCORE_MIN} (G135 lock)")
    print("=" * 78)

    # ---------------- G135 baseline (4h hold) ----------------
    print("\n[1/3] G135 baseline backtest (4h hold)")
    trades_4h_all = run_backtest(G135, holding="4h")
    trades_4h = filter_score(trades_4h_all, SCORE_MIN)

    # ---------------- G141 hold 1h ----------------
    print("\n[2/3] G141 hold 1h backtest")
    g141_path = make_variant("G141_hold1h", G135, changes={}, parent_id="G135")
    # holding period is a CLI arg, not in override; we pass to run_batch_backtest
    trades_1h_all = run_backtest(g141_path, holding="1h")
    trades_1h = filter_score(trades_1h_all, SCORE_MIN)

    # ---------------- G143 short enabled ----------------
    print("\n[3/3] G143 short enabled backtest (4h hold)")
    g143_path = make_variant(
        "G143_short_enabled", G135,
        changes={
            "futures_exposure.short_disabled": False,
            "futures_exposure.short_extra_score_floor": 5.0,  # relax
            "futures_exposure.short_extra_edge_bps": 2.0,
        },
        parent_id="G135",
    )
    trades_g143_all = run_backtest(g143_path, holding="4h")
    trades_g143 = filter_score(trades_g143_all, SCORE_MIN)

    # ---------------- variants (post-filter) ----------------
    # G140: G135 + UTC 06-09
    trades_g140 = filter_utc_hour(trades_4h, UTC_FILTER_LO, UTC_FILTER_HI)
    # G144: G141 + UTC 06-09
    trades_g144 = filter_utc_hour(trades_1h, UTC_FILTER_LO, UTC_FILTER_HI)

    # ---------------- summary table ----------------
    rows = [
        stats(trades_4h, "G135 baseline (4h)"),
        stats(trades_g140, "G140 UTC 06-09 filter"),
        stats(trades_1h, "G141 hold 1h"),
        stats(trades_g143, "G143 short enabled"),
        stats(trades_g144, "G144 combo UTC+1h"),
    ]

    print("\n" + "=" * 78)
    print("  RESULTS (60d, score>=76, ETH/SOL/DOGE/PEPE)")
    print("=" * 78)
    print(f"{'variant':<28} {'n':>4} {'WR%':>6} {'avg':>7} {'total':>8} {'annPct':>8}")
    print("-" * 78)
    for r in rows:
        print(f"{r['label']:<28} {r['n']:>4} {r['wr']:>5.1f} "
              f"{r['avg_net']:>+6.1f} {r['total_net']:>+8.0f} {r['annual_pct']:>+7.1f}")

    # ---------------- hour distribution G135 ----------------
    print("\n--- G135 trades hour distribution (UTC) ---")
    print(f"{'hour':>4} {'n':>4} {'wins':>5} {'wr%':>6} {'totalbps':>10}")
    by_h = hour_distribution(trades_4h)
    for h in sorted(by_h):
        n, w, tot = by_h[h]
        wr = w / n * 100 if n else 0
        print(f"{h:>4} {n:>4} {w:>5} {wr:>5.1f} {tot:>+10.0f}")

    # ---------------- save results ----------------
    out_dir = STRATS / "_scripts"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "g140_145_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "config": {"days": DAYS, "symbols": SYMBOLS, "equity": EQUITY,
                        "cost_bps": COST_BPS, "score_min": SCORE_MIN,
                        "utc_filter": [UTC_FILTER_LO, UTC_FILTER_HI]},
            "results": rows,
            "g135_hour_dist": {str(h): list(v) for h, v in by_h.items()},
        }, f, indent=2)
    print(f"\nresults -> {out_dir / 'g140_145_results.json'}")


if __name__ == "__main__":
    main()
