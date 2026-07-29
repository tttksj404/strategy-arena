# Wave-29: leverage sweep on V1 (the convex breakout gamble), 1x..10x, with real
# liquidation modelling. wave-4 swept leverage on CARRY (delta-neutral); V1 is
# directional, so its leverage behaviour has never been measured.

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

CACHE = REPO_ROOT / "research" / "wave1" / "cache"
OUT = Path(__file__).resolve().parent / "results"
SLEEVE = 25.0
MAINT_MARGIN = 0.005          # Bitget USDT-M maintenance margin ~0.5%
TAKER = 0.0006                # liquidation/stop fills are taker
SLIP = 0.0001
VOL_PCTILE = 30               # V1: enter only from a low-volatility regime
ATR_MULT = 2.0
LEVERAGES = [1.0, 2.0, 3.0, 5.0, 10.0]


def load(symbol: str = "BTCUSDT") -> pd.DataFrame:
    return load_frame(CACHE / f"binance_fapi_{symbol}_1d.csv.gz")


def indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    prev = out["close"].shift(1)
    tr = pd.concat([out["high"] - out["low"], (out["high"] - prev).abs(), (out["low"] - prev).abs()], axis=1).max(axis=1)
    out["atr"] = tr.rolling(14).mean()
    ret = out["close"].pct_change()
    out["vol20"] = ret.rolling(20).std()
    out["volpct"] = out["vol20"].rolling(250, min_periods=60).rank(pct=True) * 100.0
    return out


def simulate(df: pd.DataFrame, leverage: float) -> dict:
    """V1 breakout chase at a given leverage, with liquidation checked intrabar."""
    equity = SLEEVE
    curve, trades = [equity], []
    pos = 0          # -1 short, +1 long, 0 flat
    entry = ref = 0.0
    liquidations = 0

    rows = df.dropna(subset=["atr", "volpct"]).itertuples()
    for row in rows:
        atr, close, high, low, opn = row.atr, row.close, row.high, row.low, row.open
        if pos != 0:
            # Liquidation distance: 1/L minus maintenance. Checked against the bar's extreme.
            liq_move = max(0.0, 1.0 / leverage - MAINT_MARGIN)
            adverse = (high - entry) / entry if pos < 0 else (entry - low) / entry
            gap = (opn - entry) / entry if pos < 0 else (entry - opn) / entry
            if gap >= liq_move or adverse >= liq_move:
                equity = 0.0                      # wiped: margin gone
                trades.append(-1.0)
                liquidations += 1
                pos = 0
                curve.append(equity)
                if equity <= 0.005:
                    break
                continue
            # V1 exit: opposite breakout flips the position
            flip_up, flip_dn = ref + ATR_MULT * atr, ref - ATR_MULT * atr
            hit = (close >= flip_up and pos < 0) or (close <= flip_dn and pos > 0)
            if hit:
                move = (close - entry) / entry * pos
                pnl = move * leverage - (TAKER + SLIP) * 2 * leverage
                equity *= max(0.0, 1.0 + pnl)
                trades.append(pnl)
                pos = 0
        if pos == 0 and row.volpct < VOL_PCTILE:
            up, dn = close + ATR_MULT * atr, close - ATR_MULT * atr
            if high >= up:
                pos, entry, ref = 1, up, close
            elif low <= dn:
                pos, entry, ref = -1, dn, close
        curve.append(equity)
        if equity <= 0.005:
            break

    arr = np.asarray(curve, dtype=float)
    peak = np.maximum.accumulate(arr)
    mdd = float(((arr - peak) / np.maximum(peak, 1e-9)).min())
    tr_arr = np.asarray(trades, dtype=float) if trades else np.zeros(1)
    return {
        "leverage": leverage, "final": float(arr[-1]), "multiple": float(arr[-1] / SLEEVE),
        "trades": len(trades), "liquidations": liquidations, "mdd": mdd,
        "win_rate": float((tr_arr > 0).mean()) if trades else 0.0,
        "skew": float(pd.Series(tr_arr).skew()) if len(tr_arr) > 3 else 0.0,
        "ruined": bool(arr[-1] <= 0.005),
    }


def monte_carlo(trade_returns: np.ndarray, leverage: float, paths: int = 10000) -> dict:
    """Bootstrap the realised trades; -1.0 entries represent liquidations."""
    if len(trade_returns) < 5:
        return {"p05": float("nan"), "ruin_prob": float("nan"), "median": float("nan")}
    rng = np.random.default_rng(20260729)
    finals = []
    for _ in range(paths):
        draw = rng.choice(trade_returns, size=len(trade_returns), replace=True)
        eq = SLEEVE
        for r in draw:
            eq *= max(0.0, 1.0 + r)
            if eq <= 0.005:
                break
        finals.append(eq)
    arr = np.asarray(finals)
    # Total capital is \$100 = \$25 sleeve + \$75 stable; ruin = total below \$50.
    total = arr + 75.0
    return {
        "p05": float(np.percentile(arr, 5)), "median": float(np.median(arr)),
        "ruin_prob": float((total < 50.0).mean()), "sleeve_zero_prob": float((arr <= 0.005).mean()),
    }


def main() -> int:
    df = indicators(load())
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    print(f"{'lev':>5} {'최종':>9} {'배수':>8} {'거래':>5} {'청산':>5} {'MDD':>8} {'승률':>6} {'MC p05':>9} {'파산확률':>9}")
    for lev in LEVERAGES:
        res = simulate(df, lev)
        # rebuild trade list for MC
        eq, tr = SLEEVE, []
        r2 = simulate(df, lev)
        mc = monte_carlo(np.asarray(_trade_returns(df, lev)), lev)
        res.update({f"mc_{k}": v for k, v in mc.items()})
        results.append(res)
        print(f"{lev:5.0f}x ${res['final']:8.2f} {res['multiple']:7.2f}x {res['trades']:5d} {res['liquidations']:5d} "
              f"{res['mdd']*100:7.1f}% {res['win_rate']*100:5.1f}% ${mc['p05']:8.2f} {mc['ruin_prob']*100:8.2f}%")
    (OUT / "leverage_sweep.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0


def _trade_returns(df: pd.DataFrame, leverage: float) -> list[float]:
    """Re-run capturing per-trade returns (liquidation = -1.0)."""
    out: list[float] = []
    equity = SLEEVE
    pos = 0
    entry = ref = 0.0
    for row in df.dropna(subset=["atr", "volpct"]).itertuples():
        atr, close, high, low, opn = row.atr, row.close, row.high, row.low, row.open
        if pos != 0:
            liq_move = max(0.0, 1.0 / leverage - MAINT_MARGIN)
            adverse = (high - entry) / entry if pos < 0 else (entry - low) / entry
            gap = (opn - entry) / entry if pos < 0 else (entry - opn) / entry
            if gap >= liq_move or adverse >= liq_move:
                out.append(-1.0)
                pos = 0
                continue
            flip_up, flip_dn = ref + ATR_MULT * atr, ref - ATR_MULT * atr
            if (close >= flip_up and pos < 0) or (close <= flip_dn and pos > 0):
                move = (close - entry) / entry * pos
                out.append(move * leverage - (TAKER + SLIP) * 2 * leverage)
                pos = 0
        if pos == 0 and row.volpct < VOL_PCTILE:
            up, dn = close + ATR_MULT * atr, close - ATR_MULT * atr
            if high >= up:
                pos, entry, ref = 1, up, close
            elif low <= dn:
                pos, entry, ref = -1, dn, close
    return out


if __name__ == "__main__":
    raise SystemExit(main())
