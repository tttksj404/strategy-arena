# Wave-27: random-restart search over the USER'S OWN engine.py component space.
# engine.py is imported and driven as-is (no reimplementation) — this file only builds
# component lists, feeds them to run_backtest, and applies wave-27's gates.
# IS = through 2025-09-30 (evolution/search sees only this). OOS is opened once, at the end.

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd  # noqa: PANDAS_OK

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import engine  # the user's pre-existing engine, driven verbatim
from research.wave1.common import load_frame

CACHE = REPO_ROOT / "research" / "wave1" / "cache"
OUT = Path(__file__).resolve().parent / "results"
IS_END = pd.Timestamp("2025-09-30", tz="UTC")

# Measured Bitget costs (wave-13): maker 0.02%/leg, major slippage ~1bp.
# engine.py takes percent units, so 0.02 == 0.02%.
FEE_PCT = 0.02
SLIPPAGE_PCT = 0.01

SIGNALS = [
    ("ema_cross", lambda r: {"fast": r.choice([5, 9, 12, 20]), "slow": r.choice([21, 26, 50, 100])}),
    ("macd_cross", lambda r: {"fast": 12, "slow": 26, "signal": 9}),
    ("boll_bounce", lambda r: {"period": r.choice([14, 20, 30]), "std": r.choice([1.5, 2.0, 2.5])}),
    ("boll_breakout", lambda r: {"period": r.choice([14, 20, 30]), "std": r.choice([1.5, 2.0, 2.5])}),
    ("price_breakout", lambda r: {"period": r.choice([10, 20, 55])}),
    ("volume_breakout", lambda r: {"period": r.choice([10, 20]), "multiplier": r.choice([1.5, 2.0, 3.0])}),
    ("stoch_rsi_cross", lambda r: {"rsi_period": 14, "stoch_period": 14}),
    ("cci_signal", lambda r: {"period": r.choice([14, 20]), "threshold": r.choice([100, 150])}),
    ("obv_divergence", lambda r: {"period": r.choice([14, 20, 30])}),
]
FILTERS = [
    ("adx_filter", lambda r: {"period": 14, "threshold": r.choice([18, 22, 25, 30])}),
    ("trend_filter", lambda r: {"period": r.choice([50, 100, 200])}),
    ("volatility_filter", lambda r: {"period": 14, "min_atr_pct": r.choice([0.3, 0.5, 1.0])}),
    ("volume_filter", lambda r: {"min_ratio": r.choice([0.8, 1.0, 1.5]), "period": 20}),
    ("rsi_range_filter", lambda r: {"period": 14, "low": r.choice([25, 30, 35]), "high": r.choice([65, 70, 75])}),
]
RISKS = [
    ("atr_stop", lambda r: {"period": 14, "multiplier": r.choice([1.5, 2.0, 3.0])}),
    ("fixed_stop", lambda r: {"stop_pct": r.choice([2.0, 3.0, 5.0])}),
    ("trailing_stop", lambda r: {"trail_pct": r.choice([2.0, 3.0, 5.0])}),
]


def random_genome(r: random.Random) -> list[dict]:
    comps: list[dict] = []
    for sid, pf in r.sample(SIGNALS, r.choice([1, 2])):
        comps.append({"id": sid, "category": "signals", "params": pf(r)})
    for fid, pf in r.sample(FILTERS, r.choice([0, 1, 2, 3])):
        comps.append({"id": fid, "category": "filters", "params": pf(r)})
    rid, pf = r.choice(RISKS)
    comps.append({"id": rid, "category": "risk", "params": pf(r)})
    comps.append({"id": "fixed_risk", "category": "sizing", "params": {"risk_pct": r.choice([1.0, 2.0, 5.0])}})
    comps.append({"id": "direction", "category": "sizing", "params": {"direction": r.choice(["both", "long", "short"])}})
    return comps


def load_symbol(symbol: str) -> dict[str, np.ndarray] | None:
    path = CACHE / f"binance_fapi_{symbol}_1d.csv.gz"
    if not path.exists():
        return None
    frame = load_frame(path)
    return {
        "close": frame["close"].to_numpy(dtype=float),
        "high": frame["high"].to_numpy(dtype=float),
        "low": frame["low"].to_numpy(dtype=float),
        "open": frame["open"].to_numpy(dtype=float),
        "volume": frame["volume"].to_numpy(dtype=float),
        # engine.py does int(timestamps[i]) — it wants epoch integers, not datetime64.
        "timestamps": np.asarray(frame.index.view("int64") // 10**9, dtype=np.int64),
        "index": frame.index,
    }


def slice_window(data: dict, oos: bool) -> dict:
    mask = (data["index"] > IS_END) if oos else (data["index"] <= IS_END)
    idx = np.flatnonzero(mask)
    return {k: (v[idx] if isinstance(v, np.ndarray) else v) for k, v in data.items() if k != "index"}


def evaluate(comps: list[dict], markets: dict[str, dict], oos: bool) -> dict:
    """Run the user's backtester per symbol; aggregate to one equity path."""
    curves, trades, dds = [], 0, []
    for data in markets.values():
        win = slice_window(data, oos)
        if len(win["close"]) < 250:
            continue
        res = engine.run_backtest(
            win["close"], win["high"], win["low"], win["volume"], win["timestamps"], win["open"],
            comps, initial_equity=10000.0, fee_pct=FEE_PCT, slippage_pct=SLIPPAGE_PCT, interval="1d",
        )
        if not res.equity_curve:
            continue
        curves.append(np.asarray(res.equity_curve, dtype=float) / 10000.0)
        trades += res.total_trades
        dds.append(res.max_drawdown_pct)
    if not curves:
        return {"fitness": -9.9, "final": 1.0, "trades": 0, "mdd": 0.0, "windows": 0}
    length = min(len(c) for c in curves)
    blended = np.mean([c[:length] for c in curves], axis=0)
    rets = np.diff(blended) / np.maximum(blended[:-1], 1e-9)
    # SPEC fitness: mean of the top-quartile 30-day windows, minus a ruin penalty and a complexity penalty.
    wins = [blended[i + 30] / blended[i] - 1.0 for i in range(0, max(0, len(blended) - 30))]
    if not wins:
        return {"fitness": -9.9, "final": float(blended[-1]), "trades": trades, "mdd": float(np.mean(dds)), "windows": 0}
    arr = np.asarray(wins, dtype=float)
    top = float(np.mean(arr[arr >= np.quantile(arr, 0.75)]))
    ruin = float(np.mean(arr < -0.25))
    fitness = top - 5.0 * ruin - 0.01 * len(comps)
    return {
        "fitness": float(fitness), "final": float(blended[-1]), "trades": trades,
        "mdd": float(np.mean(dds)), "windows": len(arr), "top_quartile": top, "p_ruin": ruin,
        "skew": float(pd.Series(rets).skew()) if len(rets) > 3 else 0.0,
    }


def main() -> int:
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    markets = {s: d for s in symbols if (d := load_symbol(s)) is not None}
    print(f"loaded {len(markets)} symbols", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)

    best_per_seed = []
    for seed in (2027101, 2027102, 2027103, 2027104, 2027105):
        r = random.Random(seed)
        best = None
        for i in range(400):  # random-restart search over the user's component space (IS only)
            comps = random_genome(r)
            score = evaluate(comps, markets, oos=False)
            if best is None or score["fitness"] > best[1]["fitness"]:
                best = (comps, score)
            if (i + 1) % 100 == 0:
                print(f"  seed {seed}: {i+1}/400 best={best[1]['fitness']:.4f}", flush=True)
        best_per_seed.append({"seed": seed, "components": best[0], "is": best[1]})
        print(f"seed {seed} done: fitness={best[1]['fitness']:.4f} final={best[1]['final']:.3f}", flush=True)

    # Final pick = MEDIAN seed by IS fitness (not the max) — avoids crowning the luckiest seed.
    ranked = sorted(best_per_seed, key=lambda b: b["is"]["fitness"])
    final = ranked[len(ranked) // 2]
    final["oos"] = evaluate(final["components"], markets, oos=True)  # OOS opened exactly once
    (OUT / "search_result.json").write_text(
        json.dumps({"per_seed": best_per_seed, "final": final}, indent=2, default=str), encoding="utf-8"
    )
    print("\nFINAL (median seed):")
    print("  components:", [f"{c['id']}" for c in final["components"]])
    print(f"  IS  fitness={final['is']['fitness']:.4f} final={final['is']['final']:.3f} trades={final['is']['trades']}")
    print(f"  OOS fitness={final['oos']['fitness']:.4f} final={final['oos']['final']:.3f} trades={final['oos']['trades']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
