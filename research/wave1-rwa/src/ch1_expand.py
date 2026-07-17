"""Expand G053's CH1 momentum score to the full crypto universe with a hold/threshold sweep.

Beats-the-benchmark search: current best is G053 t70/h24/L5 slot5-dedup at ~+0.63%/day,
MDD 16.6% over 120d. This applies the same 10-indicator CH1 score to every liquid
crypto symbol, sweeps hold {6,12,24,48}h and threshold {65,70,75,80}, keeps the G053
gate + slot5 symbol-dedup execution policy, and reports daily-return/MDD per cell on a
train60/test40 split with path-aware liquidation and full costs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/Users/tttksj/first_repo/quant_binance/strategies/_scripts")
from g002_mingogogo_ch1_backtest import compute_ch1_score  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
COST_BPS_RT = 16.0          # G053 round-trip assumption, bps
MM = 0.005                  # maintenance margin for isolated liquidation
SLOTS = 5                   # execution policy: 5 concurrent, symbol-dedup
GATE_14D_MS = 14 * 86400 * 1000
GATE_7D_MS = 7 * 86400 * 1000


def load_universe(min_vol: float = 3_000_000) -> list[dict]:
    manifest = json.loads((ROOT / "out/data_manifest_crypto.json").read_text(encoding="utf-8"))
    return [m for m in manifest if m.get("tier") == "A" and m.get("usdtVolume", 0) >= min_vol]


def scored_frame(symbol: str) -> pd.DataFrame | None:
    path = ROOT / "data/candles_1h" / f"{symbol}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path).rename(columns={
        "open": "open_price", "high": "high_price", "low": "low_price",
        "close": "close_price", "vol_base": "base_volume"})
    if len(df) < 24 * 40:
        return None
    score, _ = compute_ch1_score(df)
    df["score"] = score.to_numpy()
    df["ts_ms"] = (df["ts"].astype("int64") // 1_000_000)
    return df


def build_events(frames: dict[str, pd.DataFrame], hold: int, thr: float) -> pd.DataFrame:
    """One row per candidate entry across all symbols: ts, symbol, net_bps, path_min_pct."""
    rows = []
    for symbol, df in frames.items():
        fwd = (df["close_price"].shift(-hold) / df["close_price"] - 1) * 10000
        low_fwd = df["low_price"].shift(-1).rolling(hold, min_periods=1).min()
        # forward min over the hold window, aligned to entry bar
        low_min = df["low_price"][::-1].rolling(hold, min_periods=1).min()[::-1].shift(-1)
        cand = df[(df["score"] >= thr) & fwd.notna()].copy()
        if cand.empty:
            continue
        cand["net_bps"] = fwd.loc[cand.index] - COST_BPS_RT
        cand["path_min_pct"] = (low_min.loc[cand.index] / df["close_price"].loc[cand.index]) - 1
        cand["symbol"] = symbol
        rows.append(cand[["ts_ms", "symbol", "net_bps", "path_min_pct"]])
    if not rows:
        return pd.DataFrame(columns=["ts_ms", "symbol", "net_bps", "path_min_pct"])
    return pd.concat(rows).sort_values("ts_ms").reset_index(drop=True)


def simulate(events: pd.DataFrame, hold: int, lever: int, split_ms: int) -> dict:
    """Gate + slot5 dedup execution, path-aware liquidation, daily equity on the test span."""
    open_pos: list[tuple[int, str]] = []
    daily: dict[pd.Timestamp, float] = {}
    trades = liqs = 0
    for row in events.itertuples(index=False):
        ts = row.ts_ms
        g14 = events[(events.ts_ms < ts) & (events.ts_ms >= ts - GATE_14D_MS)].net_bps
        if not (len(g14) and g14.sum() > 0):
            continue
        g7 = events[(events.ts_ms < ts) & (events.ts_ms >= ts - GATE_7D_MS)].net_bps
        if len(g7) and g7.sum() < -3000:
            continue
        open_pos = [p for p in open_pos if p[0] > ts]
        if any(p[1] == row.symbol for p in open_pos) or len(open_pos) >= SLOTS:
            continue
        close_ms = ts + hold * 3600 * 1000
        open_pos.append((close_ms, row.symbol))
        if ts < split_ms:
            continue  # train span: selection context only, not scored
        liq = row.path_min_pct <= -(1 / lever) + MM
        sleeve = -(1 / SLOTS) if liq else (1 / SLOTS) * lever * (row.net_bps / 10000)
        trades += 1
        liqs += int(liq)
        day = pd.Timestamp(close_ms, unit="ms", tz="UTC").normalize()
        daily[day] = daily.get(day, 0.0) + sleeve
    if not daily:
        return {"trades": 0}
    series = pd.Series(daily).sort_index()
    idx = pd.date_range(series.index.min(), series.index.max(), freq="D", tz="UTC")
    ret = series.reindex(idx).fillna(0.0)
    curve = (1 + ret).cumprod()
    net = curve.iloc[-1] - 1
    mdd = float((1 - curve / curve.cummax()).max())
    days = len(ret)
    dpr = (1 + net) ** (1 / days) - 1 if net > -1 and days else -1.0
    return {"trades": trades, "liqs": liqs, "net": net, "mdd": mdd, "days": days,
            "daily": dpr, "d2x": (np.log(2) / np.log1p(dpr)) if dpr > 0 else None}


def main() -> int:
    universe = load_universe()
    print(f"universe: {len(universe)} liquid tier-A crypto", flush=True)
    frames = {}
    for meta in universe:
        f = scored_frame(meta["symbol"])
        if f is not None:
            frames[meta["symbol"]] = f
    print(f"scored: {len(frames)} symbols", flush=True)
    span_min = min(f["ts_ms"].min() for f in frames.values())
    span_max = max(f["ts_ms"].max() for f in frames.values())
    split_ms = int(span_min + (span_max - span_min) * 0.6)
    results = []
    for hold in (6, 12, 24, 48):
        for thr in (65, 70, 75, 80):
            ev = build_events(frames, hold, thr)
            for lever in (3, 5, 7):
                r = simulate(ev, hold, lever, split_ms)
                if r.get("trades", 0) >= 15:
                    r.update(hold=hold, thr=thr, L=lever)
                    results.append(r)
                    print(f"h{hold} t{thr} L{lever}: net={r['net']:+.1%} daily={r['daily']:+.3%} "
                          f"mdd={r['mdd']:.1%} 2x={r['d2x']} trades={r['trades']} liq={r['liqs']}", flush=True)
    df = pd.DataFrame(results).sort_values("daily", ascending=False)
    df.to_csv(ROOT / "out/ch1_expand_results.csv", index=False)
    bench = 0.0063
    winners = df[(df.daily > bench) & (df.mdd <= 0.30)]
    print(f"\n=== 벤치(일 +0.63%, MDD 16.6%) 초과 + MDD<=30%: {len(winners)} ===", flush=True)
    if not winners.empty:
        print(winners.head(8).to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
