# Wave-29: leverage sweep applied to the REAL V1 trades (wave-20 engine, 153 trades).
# Liquidation is modelled from each trade's measured max adverse excursion (MAE),
# recomputed from price data between entry and exit — not approximated from the exit price.

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd  # noqa: PANDAS_OK

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.wave1.common import load_frame
from research.wave20_convex.engine20 import run_v1

CACHE = REPO_ROOT / "research" / "wave1" / "cache"
WAVE6 = REPO_ROOT / "research" / "wave6" / "cache"
OUT = Path(__file__).resolve().parent / "results"
SLEEVE, STABLE = 25.0, 75.0
MAINT = 0.005          # Bitget USDT-M maintenance margin ~0.5%
LEVERAGES = [1, 2, 3, 5, 10]


def price_frame(symbol: str) -> pd.DataFrame:
    """Prefer 1H (tighter MAE); fall back to daily."""
    hourly = WAVE6 / f"binance_fapi_{symbol}_1h.csv.gz"
    if hourly.exists():
        return load_frame(hourly)
    return load_frame(CACHE / f"binance_fapi_{symbol}_1d.csv.gz")


def measure_mae(trades) -> list[dict]:
    """For each trade, the worst adverse move against the position while it was open."""
    frames: dict[str, pd.DataFrame] = {}
    rows = []
    for t in trades:
        if t.symbol not in frames:
            frames[t.symbol] = price_frame(t.symbol)
        df = frames[t.symbol]
        window = df.loc[(df.index >= t.entry_time) & (df.index <= t.exit_time)]
        if window.empty:
            mae = abs(min(0.0, t.pnl_fraction))  # fallback: at least the realised loss
        elif t.direction > 0:
            mae = max(0.0, float((t.entry_price - window["low"].min()) / t.entry_price))
        else:
            mae = max(0.0, float((window["high"].max() - t.entry_price) / t.entry_price))
        rows.append({"pnl_fraction": float(t.pnl_fraction), "mae": mae,
                     "cost_fraction": float(t.cost_usdt / max(t.entry_equity_usdt, 1e-9))})
    return rows


def run_at_leverage(rows: list[dict], lev: float) -> dict:
    """Compound the sleeve; a trade whose MAE breaches the liquidation band wipes the margin."""
    liq_band = max(0.0, 1.0 / lev - MAINT)
    equity, curve, rets, liqs = SLEEVE, [SLEEVE], [], 0
    for r in rows:
        if r["mae"] >= liq_band:
            ret = -1.0
            liqs += 1
        else:
            ret = r["pnl_fraction"] * lev - r["cost_fraction"] * lev
        rets.append(ret)
        equity = max(0.0, equity * (1.0 + ret))
        curve.append(equity)
        if equity <= 0.005:
            break
    arr = np.asarray(curve)
    peak = np.maximum.accumulate(arr)
    return {
        "leverage": lev, "final": float(arr[-1]), "multiple": float(arr[-1] / SLEEVE),
        "trades_taken": len(rets), "liquidations": liqs,
        "mdd": float(((arr - peak) / np.maximum(peak, 1e-9)).min()),
        "returns": rets, "wiped": bool(arr[-1] <= 0.005),
    }


def monte_carlo(rets: list[float], paths: int = 10000) -> dict:
    if len(rets) < 5:
        # Too few trades to bootstrap — the sleeve died almost immediately.
        wiped = bool(rets and min(rets) <= -0.999)
        return {"p05": 0.0 if wiped else float("nan"), "median": 0.0 if wiped else float("nan"),
                "ruin_prob": 0.0, "wipe_prob": 1.0 if wiped else float("nan")}
    rng = np.random.default_rng(20260729)
    src = np.asarray(rets)
    finals = np.empty(paths)
    for i in range(paths):
        eq = SLEEVE
        for r in rng.choice(src, size=len(src), replace=True):
            eq = max(0.0, eq * (1.0 + r))
            if eq <= 0.005:
                break
        finals[i] = eq
    total = finals + STABLE
    return {"p05": float(np.percentile(finals, 5)), "median": float(np.median(finals)),
            "ruin_prob": float((total < 50.0).mean()), "wipe_prob": float((finals <= 0.005).mean())}


def main() -> int:
    result = run_v1()
    rows = measure_mae(result.trades)
    mae = np.asarray([r["mae"] for r in rows])
    print(f"V1 실거래 {len(rows)}건 | MAE 중앙값 {np.median(mae)*100:.2f}% | 90분위 {np.percentile(mae,90)*100:.2f}% | 최대 {mae.max()*100:.2f}%")
    print()
    print(f"{'lev':>4} {'청산선':>7} {'최종':>10} {'배수':>8} {'청산':>5} {'MDD':>8} {'MC p05':>9} {'전멸확률':>8} {'시스템파산':>9}")
    out = []
    for lev in LEVERAGES:
        res = run_at_leverage(rows, lev)
        mc = monte_carlo(res["returns"])
        band = max(0.0, 1.0 / lev - MAINT) * 100
        print(f"{lev:3d}x {band:6.1f}% ${res['final']:9.2f} {res['multiple']:7.2f}x {res['liquidations']:5d} "
              f"{res['mdd']*100:7.1f}% ${mc['p05']:8.2f} {mc['wipe_prob']*100:7.2f}% {mc['ruin_prob']*100:8.2f}%")
        out.append({k: v for k, v in {**res, **{f"mc_{k}": v for k, v in mc.items()}}.items() if k != "returns"})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "v1_leverage.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
