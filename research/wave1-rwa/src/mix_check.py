"""G053+XAU 배분 그리드 — 두 ACTIVE 엣지의 합성이 단기 수익속도를 올리는지 판정."""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS = Path("/Users/tttksj/first_repo/quant_binance/strategies/_scripts")
sys.path.insert(0, str(SCRIPTS))
import g041_2022_oos as oos  # noqa: E402

oos.DATA_DIR = Path("/Users/tttksj/Library/Mobile Documents/com~apple~CloudDocs/_session_data/quant_research/bitget_rwa_wave1/g053_recent_data")
from g041_2022_oos import gather_entries  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
W, LEV, SLOTS = 120, 5, 3


def g053_daily() -> pd.Series:
    now_ms = int(time.time() * 1000)
    entries = gather_entries(threshold=70, hold=72)
    entries["net"] = entries["gross_bps"] - 16
    frame = entries[entries.open_time >= now_ms - W * 86400000].sort_values("open_time").reset_index(drop=True)
    open_pos: list[int] = []
    events = []
    for _, row in frame.iterrows():
        ts = row.open_time
        g14 = frame[(frame.open_time < ts) & (frame.open_time >= ts - 14 * 86400000)]
        g7 = frame[(frame.open_time < ts) & (frame.open_time >= ts - 7 * 86400000)]
        if not (len(g14) and g14.net.sum() > 0 and not (len(g7) and g7.net.sum() < -3000)):
            continue
        open_pos = [p for p in open_pos if p > ts]
        if len(open_pos) >= SLOTS:
            continue
        open_pos.append(ts + 72 * 3600000)
        events.append((ts + 72 * 3600000, row.net))
    closes = pd.DataFrame(events, columns=["close_ms", "net_bps"])
    closes["date"] = pd.to_datetime(closes.close_ms, unit="ms", utc=True).dt.normalize()
    daily = closes.groupby("date").net_bps.sum()
    idx = pd.date_range(daily.index.min(), daily.index.max(), freq="D", tz="UTC")
    print(f"G053 체결 {len(closes)}건/{W}d (슬롯{SLOTS}·L{LEV})")
    return daily.reindex(idx).fillna(0) * (1 / SLOTS) * LEV / 10000


def xau_daily() -> pd.Series:
    candles = pd.read_parquet(ROOT / "data/candles_1h/XAUUSDT.parquet").reset_index(drop=True)
    upper = candles.high.rolling(20).max().shift(1)
    lower = candles.low.rolling(20).min().shift(1)
    signal = pd.Series(np.where(candles.close > upper, 1, np.where(candles.close < lower, -1, np.nan))).ffill().fillna(0)
    window = candles.iloc[-24 * W:]
    shifted = signal.iloc[-24 * W:].shift(1).fillna(0).to_numpy()
    opens, closes = window.open.to_numpy(), window.close.to_numpy()
    equity, prev, prev_close, values = 1.0, 0.0, None, []
    for i in range(len(window)):
        if i and prev:
            equity *= 1 + prev * 5 * (opens[i] / prev_close - 1)
        if shifted[i] != prev:
            equity -= equity * 5 * abs(shifted[i] - prev) * 0.0008
            prev = shifted[i]
        if prev:
            equity *= 1 + prev * 5 * (closes[i] / opens[i] - 1)
        prev_close = closes[i]
        values.append(equity)
    series = pd.Series(values, index=pd.DatetimeIndex(window.ts)).resample("1D").last().ffill()
    return series.pct_change().fillna(0)


def main() -> int:
    g, x = g053_daily(), xau_daily()
    common = g.index.intersection(x.index)
    g, x = g[common], x[common]
    print(f"공통 {len(common)}일 | 일간상관 {np.corrcoef(g, x)[0, 1]:+.2f}")
    print(f"{'G053/XAU':10s} {'net':>8s} {'일수익':>8s} {'MDD':>6s} {'속도/MDD':>8s} {'2배':>6s}")
    for weight in (1.0, 0.7, 0.5, 0.3, 0.0):
        blend = weight * g + (1 - weight) * x
        curve = (1 + blend).cumprod()
        net = curve.iloc[-1] - 1
        mdd = (1 - curve / curve.cummax()).max()
        daily = (1 + net) ** (1 / len(common)) - 1
        double_days = np.log(2) / np.log1p(daily) if daily > 0 else float("inf")
        print(f" {int(weight*100):3d}/{int((1-weight)*100):<3d}   {net:+8.1%} {daily:+8.3%} {mdd:6.1%} {daily/mdd*100:8.2f} {double_days:6.0f}d")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
