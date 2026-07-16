"""Pluggable weekly strategy-rotation walk-forward over swept combos.

Slots are (symbol, strategy, L) combos from a config-defined pool. Each week the
runner ranks combos by trailing net return using only past bars, holds the top-k
next week, and charges an explicit switch friction when a slot changes.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .engine import Costs, run_backtest
from .strategies import StrategySpec, signal_for, _session

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CONFIG = {
    "manifests": ["out/data_manifest.json", "out/data_manifest_crypto.json"],
    "min_usdt_volume": 1_000_000,
    "tiers": ["A"],
    "strategies": [
        {"name": "B1_donchian", "params": {"lookback": 20}},
        {"name": "B4_vol_breakout", "params": {"bb": 20, "atr_stop": 2, "max_hold": 20}},
        {"name": "B2_mean_reversion", "params": {"lookback": 24, "z_entry": 2, "max_hold": 48}},
        {"name": "WED_SHORT", "params": {"buckets": [81, 82, 83]}},
    ],
    "leverages": [2, 3, 5],
    "trailing_days": 56,
    "top_k": [1, 3, 5],
    "rebalance": "W-MON",
    "initial_equity": 300.0,
}


def load_pool(config: dict) -> list[dict]:
    """Liquid tier-filtered symbols from every configured manifest."""
    pool: list[dict] = []
    for manifest_path in config["manifests"]:
        for row in json.loads((ROOT / manifest_path).read_text(encoding="utf-8")):
            if row.get("tier") in config["tiers"] and row.get("usdtVolume", 0) >= config["min_usdt_volume"]:
                pool.append(row)
    return pool


def combo_returns(meta: dict, spec_cfg: dict, leverage: int) -> pd.Series | None:
    """Hourly levered net-return series for one combo over its full history."""
    symbol = meta["symbol"]
    frame = pd.read_parquet(ROOT / "data/candles_1h" / f"{symbol}.parquet").reset_index(drop=True)
    if len(frame) < 1500:
        return None
    fund_path = ROOT / "data/funding" / f"{symbol}.parquet"
    fund = pd.read_parquet(fund_path) if fund_path.exists() else pd.DataFrame(columns=["ts", "rate"])
    if spec_cfg["name"] == "WED_SHORT":
        signal = -1.0 * _session(frame, set(spec_cfg["params"]["buckets"]), 1.0)
    else:
        signal = signal_for(StrategySpec(spec_cfg["name"], spec_cfg["params"]), frame)
    costs = Costs(float(meta.get("half_spread_bp", 1.0)))
    result = run_backtest(frame, signal, fund, leverage, costs)
    equity = result.equity
    returns = equity.pct_change().fillna(0.0)
    returns.index = pd.DatetimeIndex(frame["ts"].iloc[: len(returns)])
    return returns if not result.liquidated else returns  # liquidation leaves zeros; keep honest path


def walk_forward(panel: pd.DataFrame, config: dict, k: int) -> tuple[pd.Series, list[dict]]:
    """Weekly top-k rotation; selection uses strictly-prior bars only."""
    switch_cost = 2 * (0.0006 + 0.0001 + 0.0001)
    weeks = panel.resample(config["rebalance"]).first().index
    equity, value = [], config["initial_equity"]
    prev_picks: tuple[str, ...] = ()
    picks_log: list[dict] = []
    trailing = pd.Timedelta(days=config["trailing_days"])
    for start, end in zip(weeks[:-1], weeks[1:]):
        history = panel.loc[(panel.index < start) & (panel.index >= start - trailing)]
        live = history.columns[history.notna().sum() > 24 * 21]
        if not len(live):
            continue
        if config.get("score") == "sharpe":
            filled = history[live].fillna(0.0)
            score = filled.mean() / filled.std().replace(0, np.nan)
            score = score.dropna()
        else:
            score = (1 + history[live].fillna(0.0)).prod()
        picks = tuple(score.sort_values(ascending=False).head(k).index)
        week = panel.loc[(panel.index >= start) & (panel.index < end), list(picks)].fillna(0.0)
        week_returns = week.mean(axis=1) if len(picks) else pd.Series(dtype=float)
        changed = len(set(picks) - set(prev_picks))
        lev = int(np.mean([int(p.rsplit("|L", 1)[1]) for p in picks])) if picks else 0
        value *= (1 - switch_cost * lev * changed / max(1, len(picks)))
        for ts, ret in week_returns.items():
            value *= max(0.0, 1 + ret)
            equity.append((ts, value))
        picks_log.append({"week": str(start.date()), "picks": list(picks), "changed": changed})
        prev_picks = picks
    series = pd.Series(dict(equity)).sort_index()
    return series, picks_log


def summarize(name: str, equity: pd.Series, initial: float) -> dict:
    """Headline metrics for one equity curve."""
    if equity.empty:
        return {"name": name, "net": None}
    days = max(1e-9, (equity.index[-1] - equity.index[0]).total_seconds() / 86_400)
    net = equity.iloc[-1] / initial - 1
    daily = (1 + net) ** (1 / days) - 1 if net > -1 else -1.0
    mdd = float((1 - equity / equity.cummax()).max())
    return {"name": name, "net": round(net, 4), "days": round(days, 1), "daily": round(daily, 5),
            "d2x": round(float(np.log(2) / np.log1p(daily)), 1) if daily > 0 else None, "mdd": round(mdd, 4)}


def main() -> int:
    """Build the combo panel, run rotation variants, and write results."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="")
    args = parser.parse_args()
    config = {**DEFAULT_CONFIG, **(json.loads(Path(args.config).read_text(encoding="utf-8")) if args.config else {})}
    cache = ROOT / "out/rotation_panel.parquet"
    if cache.exists():
        panel = pd.read_parquet(cache)
        panel.index = pd.DatetimeIndex(panel.index)
    else:
        pool = load_pool(config)
        series: dict[str, pd.Series] = {}
        for meta, spec_cfg, leverage in itertools.product(pool, DEFAULT_CONFIG["strategies"], DEFAULT_CONFIG["leverages"]):
            returns = combo_returns(meta, spec_cfg, leverage)
            if returns is not None:
                series[f"{meta['symbol']}|{spec_cfg['name']}|L{leverage}"] = returns
        panel = pd.DataFrame(series).sort_index()
        panel.to_parquet(cache)
    keep = [c for c in panel.columns if int(c.rsplit("|L", 1)[1]) in set(config["leverages"])]
    panel = panel[keep]
    results = []
    logs = {}
    for k in config["top_k"]:
        equity, picks_log = walk_forward(panel, config, k)
        results.append(summarize(f"rotation_top{k}", equity, config["initial_equity"]))
        logs[f"top{k}"] = picks_log
    static_names = ["TQQQUSDT|B1_donchian|L2", "XAUUSDT|B1_donchian|L5", "XAUUSDT|B1_donchian|L3", "ETHUSDT|WED_SHORT|L5"]
    for name in static_names:
        if name in panel:
            curve = config["initial_equity"] * (1 + panel[name].fillna(0.0)).cumprod()
            results.append(summarize(f"static_{name}", curve.dropna(), config["initial_equity"]))
    suffix = config.get("out_suffix", "")
    out = {"pool_combos": len(panel.columns), "config": {k: config[k] for k in ("leverages", "top_k", "trailing_days") if k in config} | {"score": config.get("score", "prod")}, "results": results}
    (ROOT / f"out/rotation_results{suffix}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    (ROOT / f"out/rotation_picks{suffix}.json").write_text(json.dumps(logs, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
