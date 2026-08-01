"""wave59 — 배리어 결과를 한 번만 계산해 캐시한다.

핵심 관찰: 배리어(목표/손절) 도달 결과는 **방향에만 의존하고 레짐 필터와 무관**하다.
그러므로 모든 진입 시점에 대해 롱 결과와 숏 결과를 한 번 계산해 저장하면,
이후 어떤 레짐 규칙을 시험해도 불리언 선택만으로 끝난다 (수백 개 규칙이 사실상 무료).

    $V research/wave59_regime/precompute59.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.wave30_qd.dataio30 import SYMBOLS, build_market_cache  # noqa: E402

OUT = Path(__file__).resolve().parent / "cache"
STEP = 6
MAX_HOLD = 24 * 60
CAPITAL = 100.0

# (이름, 목표$, 레버리지, 손절 자본비율)
CONFIGS = [
    ("g30_l5_s90", 30.0, 5.0, 0.90),
    ("g30_l5_s30", 30.0, 5.0, 0.30),
    ("g50_l5_s90", 50.0, 5.0, 0.90),
]

WIN, LOSS, OPEN = 1, -1, 0


def resolve(high, low, close, target_pct, stop_pct, side):
    """모든 진입 시점의 배리어 결과. side=+1 롱, -1 숏."""
    n = len(close)
    entries = np.arange(0, n - 1, STEP)
    out = np.zeros(len(entries), dtype=np.int8)
    for k, entry in enumerate(entries):
        price = close[entry]
        if not np.isfinite(price) or price <= 0:
            continue
        if side > 0:
            take, kill = price * (1 + target_pct), price * (1 - stop_pct)
        else:
            take, kill = price * (1 - target_pct), price * (1 + stop_pct)
        stop_bar = min(entry + MAX_HOLD, n)
        res = OPEN
        for bar in range(entry + 1, stop_bar):
            if side > 0:
                if low[bar] <= kill:
                    res = LOSS
                    break
                if high[bar] >= take:
                    res = WIN
                    break
            else:
                if high[bar] >= kill:
                    res = LOSS
                    break
                if low[bar] <= take:
                    res = WIN
                    break
        out[k] = res
    return entries, out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cache = build_market_cache()
    index = pd.DatetimeIndex(cache.index)

    for name, gain, lev, stop_frac in CONFIGS:
        path = OUT / f"outcomes_{name}.npz"
        if path.exists():
            print(f"skip {name} (이미 있음)")
            continue
        target_pct = gain / (CAPITAL * lev)
        stop_pct = stop_frac / lev
        payload = {}
        for symbol in SYMBOLS:
            arr = cache.arrays[symbol]
            entries, longs = resolve(
                arr.high, arr.low, arr.close, target_pct, stop_pct, +1
            )
            _, shorts = resolve(
                arr.high, arr.low, arr.close, target_pct, stop_pct, -1
            )
            payload[f"{symbol}_entries"] = entries
            payload[f"{symbol}_long"] = longs
            payload[f"{symbol}_short"] = shorts
            print(f"  {name} {symbol}: 롱 승 {(longs == WIN).sum():5d} "
                  f"패 {(longs == LOSS).sum():5d} 미결 {(longs == OPEN).sum():4d} | "
                  f"숏 승 {(shorts == WIN).sum():5d} 패 {(shorts == LOSS).sum():5d}")
        payload["year"] = index.year.to_numpy()
        np.savez_compressed(path, **payload)
        print(f"saved {path.name}")

    print("\n=== 완료 ===")
    for p in sorted(OUT.glob("outcomes_*.npz")):
        print(f"  {p.name}  {p.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
