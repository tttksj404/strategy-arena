"""Bench dashboard: drift-check every validated strategy and print deploy/bench verdicts.

Run any time (`python3 -m src.bench_check`). Each strategy gets: recent 30/90d
net at its chosen leverage, rolling-90d percentile vs its own history, current
signal, and an ACTIVE / BENCH verdict from pre-registered rules (percentile>=10
and 90d net>0 → ACTIVE). Output is a markdown table to stdout and out/BENCH.md.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RATE = 0.0006 + 0.0002


def dch_equity(candles: pd.DataFrame, leverage: int) -> tuple[pd.Series, float]:
    upper = candles.high.rolling(20).max().shift(1)
    lower = candles.low.rolling(20).min().shift(1)
    signal = pd.Series(np.where(candles.close > upper, 1, np.where(candles.close < lower, -1, np.nan))).ffill().fillna(0)
    shifted = signal.shift(1).fillna(0).to_numpy()
    opens, closes = candles.open.to_numpy(), candles.close.to_numpy()
    equity, prev, prev_close, values = 1.0, 0.0, None, []
    for i in range(len(candles)):
        if i and prev:
            equity *= 1 + prev * leverage * (opens[i] / prev_close - 1)
        if shifted[i] != prev:
            equity -= equity * leverage * abs(shifted[i] - prev) * RATE
            prev = shifted[i]
        if prev:
            equity *= 1 + prev * leverage * (closes[i] / opens[i] - 1)
        prev_close = closes[i]
        values.append(max(equity, 1e-9))
    return pd.Series(values, index=pd.DatetimeIndex(candles.ts)), float(signal.iloc[-1])


def wed_equity(candles: pd.DataFrame, leverage: int) -> tuple[pd.Series, float]:
    local = candles.ts.dt.tz_convert(ZoneInfo("America/New_York"))
    bucket = local.dt.dayofweek * 24 + local.dt.hour
    signal = pd.Series(np.where(bucket.isin({81, 82, 83}), -1.0, 0.0))
    shifted = signal.shift(1).fillna(0).to_numpy()
    opens, closes = candles.open.to_numpy(), candles.close.to_numpy()
    equity, prev, prev_close, values = 1.0, 0.0, None, []
    for i in range(len(candles)):
        if i and prev:
            equity *= 1 + prev * leverage * (opens[i] / prev_close - 1)
        if shifted[i] != prev:
            equity -= equity * leverage * abs(shifted[i] - prev) * RATE
            prev = shifted[i]
        if prev:
            equity *= 1 + prev * leverage * (closes[i] / opens[i] - 1)
        prev_close = closes[i]
        values.append(max(equity, 1e-9))
    return pd.Series(values, index=pd.DatetimeIndex(candles.ts)), float(signal.iloc[-1])


def stats(equity: pd.Series) -> tuple[float, float, float]:
    def window_net(days: int) -> float:
        bars = 24 * days
        return equity.iloc[-1] / equity.iloc[-bars] - 1 if len(equity) > bars else np.nan
    rolling = (equity / equity.shift(24 * 90) - 1).dropna()
    pct = float((rolling < rolling.iloc[-1]).mean() * 100) if len(rolling) > 50 else np.nan
    return window_net(30), window_net(90), pct


def verdict(net90: float, pct: float) -> str:
    if np.isnan(pct):
        return "BENCH(표본부족)"
    return "ACTIVE" if net90 > 0 and pct >= 10 else "BENCH"


def main() -> int:
    rows = []
    for name, symbol, lever, kind, regime in (
        ("XAU 돌파 L3", "XAUUSDT", 3, "dch", "귀금속 추세"),
        ("XAU 돌파 L5", "XAUUSDT", 5, "dch", "귀금속 추세(공격)"),
        ("TQQQ 돌파 L2", "TQQQUSDT", 2, "dch", "미 테크 추세(양방향)"),
        ("BTC 돌파 L3", "BTCUSDT", 3, "dch", "크립토 추세"),
        ("ETH 수요숏 L7", "ETHUSDT", 7, "wed", "캘린더/미국세션 플로우"),
    ):
        candles = pd.read_parquet(ROOT / "data/candles_1h" / f"{symbol}.parquet").reset_index(drop=True)
        equity, signal = (dch_equity if kind == "dch" else wed_equity)(candles, lever)
        net30, net90, pct = stats(equity)
        rows.append((name, regime, net30, net90, pct, signal, verdict(net90, pct)))
    lines = [f"# 전략 벤치 대시보드 ({datetime.now(UTC).isoformat(timespec='minutes')})", "",
             "| 전략 | 국면(수익원) | 30d | 90d | 백분위 | 신호 | 판정 |", "|---|---|---|---|---|---|---|"]
    for name, regime, net30, net90, pct, signal, verd in rows:
        sig = "롱" if signal > 0 else "숏" if signal < 0 else "대기"
        lines.append(f"| {name} | {regime} | {net30:+.1%} | {net90:+.1%} | {pct:.0f}% | {sig} | **{verd}** |")
    lines += ["", "- 이벤트형(상장 급등페이드 L2): 상장 캘린더 의존 — paper 러너가 자동 스캔, 최근 4건 1승3패로 BENCH",
              "- 판정 룰(사전등록): 90d>0 AND 자기분포 백분위≥10 → ACTIVE. 그 외 BENCH. 주 1회 실행 권장."]
    output = "\n".join(lines)
    (ROOT / "out/BENCH.md").write_text(output + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
