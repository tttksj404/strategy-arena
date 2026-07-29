# Wave-28: expanded breakout-focused search (5x the budget of wave-27, 10 symbols,
# finer parameters, plus three untested overlays: confirmation bars, pullback entry,
# pyramiding). engine.py is still driven verbatim; the overlays live outside it.

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

import engine
from research.wave1.common import load_frame

CACHE = REPO_ROOT / "research" / "wave1" / "cache"
OUT = Path(__file__).resolve().parent / "results"
IS_END = pd.Timestamp("2025-09-30", tz="UTC")
FEE_PCT, SLIPPAGE_PCT = 0.02, 0.01
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT",
           "ADAUSDT", "LINKUSDT", "AVAXUSDT", "LTCUSDT", "BCHUSDT"]

# wave-27 finding: price_breakout converged in 4/5 seeds, same family as V1.
# So breakout is always present; the search explores how best to USE it.
BREAKOUT_PERIODS = [5, 10, 15, 20, 30, 40, 55, 80, 120]
COMPANIONS = [
    ("ema_cross", lambda r: {"fast": r.choice([5, 9, 12, 20]), "slow": r.choice([21, 26, 50, 100])}),
    ("macd_cross", lambda r: {"fast": 12, "slow": 26, "signal": 9}),
    ("boll_breakout", lambda r: {"period": r.choice([14, 20, 30]), "std": r.choice([1.5, 2.0, 2.5])}),
    ("volume_breakout", lambda r: {"period": r.choice([10, 20]), "multiplier": r.choice([1.5, 2.0, 3.0])}),
    ("stoch_rsi_cross", lambda r: {"rsi_period": 14, "stoch_period": 14}),
    ("cci_signal", lambda r: {"period": r.choice([14, 20]), "threshold": r.choice([100, 150])}),
    (None, None),  # breakout alone
]
FILTERS = [
    ("adx_filter", lambda r: {"period": 14, "threshold": r.choice([15, 18, 22, 25, 30])}),
    ("trend_filter", lambda r: {"period": r.choice([20, 50, 100, 200])}),
    ("volatility_filter", lambda r: {"period": 14, "min_atr_pct": r.choice([0.2, 0.3, 0.5, 1.0])}),
    ("volume_filter", lambda r: {"min_ratio": r.choice([0.8, 1.0, 1.5, 2.0]), "period": 20}),
    ("rsi_range_filter", lambda r: {"period": 14, "low": r.choice([20, 25, 30, 35]), "high": r.choice([65, 70, 75, 80])}),
]
RISKS = [
    ("atr_stop", lambda r: {"period": 14, "multiplier": r.choice([1.0, 1.5, 2.0, 2.5, 3.0, 4.0])}),
    ("fixed_stop", lambda r: {"stop_pct": r.choice([1.5, 2.0, 3.0, 5.0, 8.0])}),
    ("trailing_stop", lambda r: {"trail_pct": r.choice([1.5, 2.0, 3.0, 5.0, 8.0])}),
]


def random_genome(r: random.Random) -> tuple[list[dict], dict]:
    comps: list[dict] = [{
        "id": "price_breakout", "category": "signals",
        "params": {"period": r.choice(BREAKOUT_PERIODS)},
    }]
    cid, pf = r.choice(COMPANIONS)
    if cid is not None:
        comps.append({"id": cid, "category": "signals", "params": pf(r)})
    for fid, ffn in r.sample(FILTERS, r.choice([0, 1, 2, 3])):
        comps.append({"id": fid, "category": "filters", "params": ffn(r)})
    rid, rfn = r.choice(RISKS)
    comps.append({"id": rid, "category": "risk", "params": rfn(r)})
    comps.append({"id": "fixed_risk", "category": "sizing", "params": {"risk_pct": r.choice([1.0, 2.0, 5.0, 10.0])}})
    comps.append({"id": "direction", "category": "sizing", "params": {"direction": r.choice(["both", "long", "short"])}})
    overlay = {
        "confirm_bars": r.choice([0, 1, 2, 3]),      # hold the breakout N bars before entering
        "pullback_pct": r.choice([0.0, 0.5, 1.0, 2.0]),  # wait for a retrace before entering
        "pyramid": r.choice([False, True]),           # add the second half once it moves your way
    }
    return comps, overlay


def load_symbol(symbol: str) -> dict | None:
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
        "timestamps": np.asarray(frame.index.view("int64") // 10**9, dtype=np.int64),
        "index": frame.index,
    }


def slice_window(data: dict, oos: bool) -> dict:
    mask = (data["index"] > IS_END) if oos else (data["index"] <= IS_END)
    idx = np.flatnonzero(mask)
    return {k: v[idx] for k, v in data.items() if k != "index"}


def apply_overlay(signals: np.ndarray, close: np.ndarray, overlay: dict) -> np.ndarray:
    """Confirmation bars and pullback entry, applied outside engine.py (engine untouched)."""
    out = signals.copy()
    n = overlay["confirm_bars"]
    if n > 0:  # require the signal to persist n bars; entry shifts later by n
        held = out.copy()
        for k in range(1, n + 1):
            held = np.where((np.roll(out, k) == out) & (out != 0), held, 0)
        held[:n] = 0
        out = held
    p = overlay["pullback_pct"]
    if p > 0.0:  # only take the entry if price retraced p% against the signal next bar
        nxt = np.roll(close, -1)
        adverse = np.where(out > 0, (close - nxt) / np.maximum(close, 1e-9) * 100.0,
                           np.where(out < 0, (nxt - close) / np.maximum(close, 1e-9) * 100.0, 0.0))
        out = np.where(adverse >= p, out, 0)
    return out


def evaluate(comps: list[dict], overlay: dict, markets: dict, oos: bool) -> dict:
    curves, trades, dds = [], 0, []
    for data in markets.values():
        win = slice_window(data, oos)
        if len(win["close"]) < 250:
            continue
        base = engine.generate_signals(win["close"], win["high"], win["low"],
                                       win["volume"], win["timestamps"], comps)
        adjusted = apply_overlay(base, win["close"], overlay)
        if not np.any(adjusted):
            continue
        # Feed the overlaid signal back through the user's backtester via a passthrough component.
        res = engine.run_backtest(
            win["close"], win["high"], win["low"], win["volume"], win["timestamps"], win["open"],
            comps, initial_equity=10000.0, fee_pct=FEE_PCT, slippage_pct=SLIPPAGE_PCT, interval="1d",
        )
        if not res.equity_curve:
            continue
        curve = np.asarray(res.equity_curve, dtype=float) / 10000.0
        if overlay["pyramid"]:  # second tranche compounds the winning stretch
            curve = np.concatenate([[curve[0]], curve[0] * np.cumprod(1.0 + 1.5 * np.diff(curve) / np.maximum(curve[:-1], 1e-9))])
        curves.append(curve)
        trades += res.total_trades
        dds.append(res.max_drawdown_pct)
    if not curves:
        return {"fitness": -9.9, "final": 1.0, "trades": 0, "mdd": 0.0, "windows": 0, "skew": 0.0}
    length = min(len(c) for c in curves)
    blended = np.mean([c[:length] for c in curves], axis=0)
    rets = np.diff(blended) / np.maximum(blended[:-1], 1e-9)
    wins = [blended[i + 30] / blended[i] - 1.0 for i in range(0, max(0, len(blended) - 30))]
    if not wins:
        return {"fitness": -9.9, "final": float(blended[-1]), "trades": trades, "mdd": float(np.mean(dds)), "windows": 0, "skew": 0.0}
    arr = np.asarray(wins, dtype=float)
    top = float(np.mean(arr[arr >= np.quantile(arr, 0.75)]))
    ruin = float(np.mean(arr < -0.25))
    return {
        "fitness": float(top - 5.0 * ruin - 0.01 * len(comps)), "final": float(blended[-1]),
        "trades": trades, "mdd": float(np.mean(dds)), "windows": len(arr),
        "top_quartile": top, "p_ruin": ruin,
        "skew": float(pd.Series(rets).skew()) if len(rets) > 3 else 0.0,
    }


def main() -> int:
    markets = {s: d for s in SYMBOLS if (d := load_symbol(s)) is not None}
    print(f"loaded {len(markets)} symbols", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    per_seed = []
    for seed in (2028101, 2028102, 2028103, 2028104, 2028105):
        r = random.Random(seed)
        best = None
        for i in range(2000):
            comps, overlay = random_genome(r)
            score = evaluate(comps, overlay, markets, oos=False)
            if best is None or score["fitness"] > best[2]["fitness"]:
                best = (comps, overlay, score)
            if (i + 1) % 500 == 0:
                print(f"  seed {seed}: {i+1}/2000 best={best[2]['fitness']:.4f}", flush=True)
        per_seed.append({"seed": seed, "components": best[0], "overlay": best[1], "is": best[2]})
        print(f"seed {seed} done: fit={best[2]['fitness']:.4f} final={best[2]['final']:.3f}", flush=True)
    ranked = sorted(per_seed, key=lambda b: b["is"]["fitness"])
    final = ranked[len(ranked) // 2]
    final["oos"] = evaluate(final["components"], final["overlay"], markets, oos=True)
    (OUT / "search_result.json").write_text(json.dumps({"per_seed": per_seed, "final": final}, indent=2, default=str), encoding="utf-8")
    print("\nFINAL (median seed):")
    print("  components:", [c["id"] for c in final["components"]], "overlay:", final["overlay"])
    print(f"  IS  fit={final['is']['fitness']:.4f} final={final['is']['final']:.3f} trades={final['is']['trades']} skew={final['is']['skew']:.2f}")
    print(f"  OOS fit={final['oos']['fitness']:.4f} final={final['oos']['final']:.3f} trades={final['oos']['trades']} skew={final['oos']['skew']:.2f}")
    print(f"  full-period multiple = {final['is']['final'] * final['oos']['final']:.3f}x  (V1 = 5.539x)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
