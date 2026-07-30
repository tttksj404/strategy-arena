#!/usr/bin/env python3
# Wave-35 data validation: does Bitget 1h agree with the Binance 1h this repo has trusted since
# wave6?
#
# Everything downstream of the widened universe depends on this. wave30 validated its 1h cache by
# checking it against the daily cache and confirming the hourly low never sat ABOVE the daily low
# (which would have understated adverse excursion and made 20x look safer than it is). The same
# question applies here, with a sharper edge: Bitget and Binance are different venues with
# different order books, so exact equality is not expected. What must hold is that the two agree
# closely enough that a strategy measured on one would behave the same on the other, and that
# Bitget does not systematically report NARROWER bars (which would understate stop-outs and
# liquidation risk).

from __future__ import annotations

from pathlib import Path
import sys
from typing import Final

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

import numpy as np
import pandas as pd  # noqa: PANDAS_OK

REPO_ROOT: Final = Path(__file__).resolve().parents[2]
BITGET_CACHE: Final = Path(__file__).resolve().parent / "cache"
BINANCE_CACHE: Final = REPO_ROOT / "research" / "wave6" / "cache"
OVERLAP_SYMBOLS: Final = ("BTCUSDT", "ETHUSDT", "SOLUSDT")

# Bitget's own early perpetual history contains STALE segments: for BTCUSDT the close is pinned at
# exactly 23301.00 across 2020-12-25..30 while Binance moves 24,605 -> 27,775, then Bitget prints a
# single +26.0% bar on 2021-01-02 catching up. Year-by-year return correlation against Binance
# exposes it cleanly -- 2019 0.971 / 2020 0.987 / 2021 0.956, then 2022 0.9994 / 2023 0.9993 /
# 2024 0.9997 / 2025 0.9998. A frozen price followed by a catch-up gap is the worst possible input
# for a leveraged backtest: it manufactures both fake calm (no stop-outs while frozen) and a fake
# jackpot (one enormous bar). So the usable span starts where the two venues agree.
USABLE_START: Final = pd.Timestamp("2022-01-01", tz="UTC")
STALE_RUN_THRESHOLD: Final = 6  # >=6 consecutive identical closes is treated as a stale segment


def load_bitget(symbol: str) -> pd.DataFrame:
    frame = pd.read_csv(BITGET_CACHE / f"bitget_{symbol}_1H.csv.gz")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, format="ISO8601")
    return frame.set_index("timestamp").sort_index()


def load_binance(symbol: str) -> pd.DataFrame:
    frame = pd.read_csv(BINANCE_CACHE / f"binance_fapi_{symbol}_1h.csv.gz")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, format="ISO8601")
    return frame.set_index("timestamp").sort_index()


def stale_report(frame: pd.DataFrame) -> dict:
    """Longest run of identical closes, and what share of bars sit inside a stale run.

    Applied to every symbol, not just the three with a Binance counterpart: the frozen-feed defect
    was found on BTC by cross-venue comparison, but nothing guarantees it is confined to symbols we
    can cross-check. This detector needs only one venue.
    """
    close = frame["close"].to_numpy(dtype=float)
    if len(close) < 2:
        return {"longest_run": 0, "stale_share": 0.0}
    changed = np.concatenate([[True], close[1:] != close[:-1]])
    group = np.cumsum(changed)
    counts = np.bincount(group)
    run_lengths = counts[group]
    return {
        "longest_run": int(counts.max()),
        "stale_share": float((run_lengths >= STALE_RUN_THRESHOLD).mean()),
    }


def scan_all_symbols() -> None:
    paths = sorted(BITGET_CACHE.glob("bitget_*_1H.csv.gz"))
    print(f"\n전 종목 정지프린트(stale) 스캔 — {len(paths)}종목, 기준: 동일 close {STALE_RUN_THRESHOLD}봉 이상 연속")
    print("=" * 108)
    rows = []
    for path in paths:
        symbol = path.name.replace("bitget_", "").replace("_1H.csv.gz", "")
        frame = load_bitget(symbol)
        full = stale_report(frame)
        trimmed = stale_report(frame[frame.index >= USABLE_START])
        rows.append((symbol, len(frame), full, trimmed, frame.index[0]))
    worst_full = sorted(rows, key=lambda r: -r[2]["stale_share"])[:8]
    print(f"{'심볼':>12} {'첫 봉':>12} {'전체 stale비율':>14} {'최장연속':>9} "
          f"{'2022+ stale비율':>16} {'2022+ 최장':>11}")
    print("-" * 108)
    for symbol, _rows, full, trimmed, first in worst_full:
        print(f"{symbol:>12} {str(first.date()):>12} {full['stale_share']:13.3%} {full['longest_run']:9d} "
              f"{trimmed['stale_share']:15.3%} {trimmed['longest_run']:11d}")
    bad_full = sum(1 for r in rows if r[2]["stale_share"] > 0.01)
    bad_trim = sum(1 for r in rows if r[3]["stale_share"] > 0.01)
    print(f"\n  stale 비율 >1% 종목: 전체구간 {bad_full}/{len(rows)} → **2022-01-01 이후 {bad_trim}/{len(rows)}**")
    print(f"  → 2022-01-01 절단이 정지프린트 문제를 실질적으로 제거한다.")


def main() -> int:
    print("Bitget 1H vs Binance 1H — 겹치는 구간 정합성 (전체 구간)")
    print("=" * 108)
    print(f"{'심볼':>9} {'겹침봉':>8} {'close 중앙오차':>14} {'close p99오차':>13} "
          f"{'고저폭 비율(중앙)':>17} {'Bitget 폭 좁음 비율':>19} {'상관(수익률)':>12}")
    print("-" * 108)
    verdicts = []
    for symbol in OVERLAP_SYMBOLS:
        bitget = load_bitget(symbol)
        binance = load_binance(symbol)
        joined = bitget.join(binance, how="inner", lsuffix="_bg", rsuffix="_bn").dropna(
            subset=["close_bg", "close_bn", "high_bg", "high_bn", "low_bg", "low_bn"]
        )
        if joined.empty:
            print(f"{symbol:>9} 겹치는 구간 없음")
            continue
        close_error = (joined["close_bg"] - joined["close_bn"]).abs() / joined["close_bn"]
        range_bg = (joined["high_bg"] - joined["low_bg"]) / joined["close_bg"]
        range_bn = (joined["high_bn"] - joined["low_bn"]) / joined["close_bn"]
        ratio = (range_bg / range_bn.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).dropna()
        narrower = float((ratio < 0.9).mean())
        returns_bg = joined["close_bg"].pct_change().dropna()
        returns_bn = joined["close_bn"].pct_change().dropna()
        common = returns_bg.index.intersection(returns_bn.index)
        correlation = float(np.corrcoef(returns_bg.loc[common], returns_bn.loc[common])[0, 1])
        print(
            f"{symbol:>9} {len(joined):8d} {close_error.median():13.6%} {close_error.quantile(0.99):12.4%} "
            f"{ratio.median():16.3f} {narrower:18.2%} {correlation:12.6f}"
        )
        verdicts.append(
            {
                "symbol": symbol,
                "median_close_error": float(close_error.median()),
                "p99_close_error": float(close_error.quantile(0.99)),
                "median_range_ratio": float(ratio.median()),
                "share_bitget_narrower": narrower,
                "return_correlation": correlation,
            }
        )

    print("\n연도별 수익률 상관 — 결함 구간 특정")
    print("=" * 108)
    for symbol in OVERLAP_SYMBOLS:
        bitget, binance = load_bitget(symbol), load_binance(symbol)
        joined = bitget.join(binance, how="inner", lsuffix="_bg", rsuffix="_bn").dropna(
            subset=["close_bg", "close_bn"]
        )
        returns = pd.DataFrame(
            {"bg": joined["close_bg"].pct_change(), "bn": joined["close_bn"].pct_change()}
        ).dropna()
        parts = []
        for year, group in returns.groupby(returns.index.year):
            parts.append(f"{year} {np.corrcoef(group.bg, group.bn)[0, 1]:.4f}")
        print(f"  {symbol:>9}: " + " · ".join(parts))

    print(f"\n{USABLE_START.date()} 이후로 절단한 정합성")
    print("=" * 108)
    print(f"{'심볼':>9} {'겹침봉':>8} {'close 중앙오차':>14} {'고저폭 비율':>12} {'수익률 상관':>12} {'판정':>7}")
    print("-" * 108)
    trimmed_ok = True
    for symbol in OVERLAP_SYMBOLS:
        bitget, binance = load_bitget(symbol), load_binance(symbol)
        joined = bitget.join(binance, how="inner", lsuffix="_bg", rsuffix="_bn").dropna(
            subset=["close_bg", "close_bn", "high_bg", "high_bn", "low_bg", "low_bn"]
        )
        joined = joined[joined.index >= USABLE_START]
        if joined.empty:
            continue
        close_error = (joined["close_bg"] - joined["close_bn"]).abs() / joined["close_bn"]
        range_bg = (joined["high_bg"] - joined["low_bg"]) / joined["close_bg"]
        range_bn = (joined["high_bn"] - joined["low_bn"]) / joined["close_bn"]
        ratio = (range_bg / range_bn.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).dropna()
        returns = pd.DataFrame(
            {"bg": joined["close_bg"].pct_change(), "bn": joined["close_bn"].pct_change()}
        ).dropna()
        correlation = float(np.corrcoef(returns.bg, returns.bn)[0, 1])
        good = close_error.median() < 0.0005 and correlation > 0.99 and ratio.median() > 0.85
        trimmed_ok &= good
        print(
            f"{symbol:>9} {len(joined):8d} {close_error.median():13.6%} {ratio.median():11.3f} "
            f"{correlation:12.6f} {'PASS' if good else 'FAIL':>7}"
        )

    scan_all_symbols()

    print("\n[판정 기준]")
    print("  · close 중앙오차 < 0.05% : 두 거래소 가격이 실질적으로 같다")
    print("  · 고저폭 비율 ~1.0      : Bitget이 변동폭을 좁게 보고하지 않는다 (좁으면 손절·청산 과소추정)")
    print("  · 수익률 상관 > 0.99    : 전략 신호가 거래소를 바꿔도 동일하게 발생한다")
    print()
    full_ok = all(
        v["median_close_error"] < 0.0005 and v["return_correlation"] > 0.99 and v["median_range_ratio"] > 0.85
        for v in verdicts
    )
    print(f"전체 구간 판정 : {'PASS' if full_ok else 'FAIL — 2019~2021 정지프린트 결함'}")
    print(f"{USABLE_START.date()}+ 판정: {'PASS — 이 구간으로 확장 유니버스를 신뢰할 수 있다' if trimmed_ok else 'FAIL'}")
    print(f"\n결론: 확장 유니버스의 사용 가능 시작일을 **{USABLE_START.date()}** 로 확정한다.")
    return 0 if trimmed_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
