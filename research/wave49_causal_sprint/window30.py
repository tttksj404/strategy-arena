#!/usr/bin/env python3
# Wave-49 support: evaluate wave30's engine over an ARBITRARY bar range.
#
# engine30.run_genome only offers mode='is' (up to OOS_SPLIT) and mode='full'. A causal walk-forward needs
# neither: it needs "these 365 days" and then "the next 90 days". Adding a range parameter to engine30
# would mean editing a file that wave31's verify31 cross-checked trade-for-trade, so instead the CACHE is
# sliced and the engine is called unmodified on the slice. The engine cannot tell the difference, which is
# exactly the property being relied on -- and it is verified rather than assumed: slicing the full range
# must reproduce run_genome(full) bit for bit.
#
# One subtlety makes this correct rather than merely convenient. SymbolArrays carries precomputed rolling
# features (prior_high/prior_low per lookback) built from the WHOLE series. Slicing keeps each bar's
# already-computed value, so a bar near the window's start still sees features derived from bars BEFORE
# the window. That is not leakage: those bars are past data, known at decision time. Recomputing features
# inside each window would be the incorrect choice, since it would blind the strategy to history a live
# trader would have.

from __future__ import annotations

import dataclasses
from pathlib import Path
import sys
from typing import Final

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

import numpy as np

from research.wave30_qd.dataio30 import MarketCache, SymbolArrays


def _slice_symbol(arrays: SymbolArrays, start: int, end: int) -> SymbolArrays:
    return dataclasses.replace(
        arrays,
        open=arrays.open[start:end],
        high=arrays.high[start:end],
        low=arrays.low[start:end],
        close=arrays.close[start:end],
        tradable=arrays.tradable[start:end],
        funding_at_bar=arrays.funding_at_bar[start:end],
        prior_high={k: v[start:end] for k, v in arrays.prior_high.items()},
        prior_low={k: v[start:end] for k, v in arrays.prior_low.items()},
        ret={k: v[start:end] for k, v in arrays.ret.items()},
        vol={k: v[start:end] for k, v in arrays.vol.items()},
        zscore={k: v[start:end] for k, v in arrays.zscore.items()},
    )


def window_cache(cache: MarketCache, start_bar: int, end_bar: int) -> MarketCache:
    """A MarketCache restricted to bars [start_bar, end_bar).

    day_of_bar indexes into daily_index, so both are re-based together: the daily arrays are cut to the
    days the sliced bars actually touch and day_of_bar is shifted by the first of those days. Getting this
    wrong would silently misalign the stable-sleeve series against the trading bars, which is why the
    round-trip check in verify_full_slice exists.
    """
    if not (0 <= start_bar < end_bar <= cache.n_bars):
        raise ValueError(f"bad window [{start_bar}, {end_bar}) for {cache.n_bars} bars")

    from research.wave30_qd.dataio30 import stable_value_per_dollar

    day_of_bar = cache.day_of_bar[start_bar:end_bar]
    first_day = int(day_of_bar[0])
    last_day = int(day_of_bar[-1])
    sliced_factor = cache.stable_daily_factor[first_day : last_day + 1]
    # stable_per_dollar is CUMULATIVE from the start of the whole series (0.9*cumprod(factor)+0.1), and
    # engine30 multiplies the stable allocation by it directly. Slicing it without rebasing therefore hands
    # a mid-series window an instant windfall: the first observed equity of a window with
    # sleeve_fraction=0.75 came out at $103.65 instead of $100, and that error compounds once per window.
    # Recomputing from the sliced daily factors with the validated helper rebases it correctly -- 90%
    # compounding from THIS window's start, 10% flat -- which is what a fresh deployment here would see.
    return dataclasses.replace(
        cache,
        index=cache.index[start_bar:end_bar],
        arrays={s: _slice_symbol(a, start_bar, end_bar) for s, a in cache.arrays.items()},
        is_mask=cache.is_mask[start_bar:end_bar],
        day_of_bar=day_of_bar - first_day,
        daily_index=cache.daily_index[first_day : last_day + 1],
        stable_daily_factor=sliced_factor,
        stable_per_dollar=stable_value_per_dollar(sliced_factor),
        n_bars=end_bar - start_bar,
    )


def verify_full_slice(cache: MarketCache, genome) -> dict:
    """Slicing the entire range must reproduce the unsliced run exactly.

    Compares trade count, every per-trade return, and the final equity. Any drift here means the slicer
    misaligns something, and every wave49 number would inherit that error.
    """
    from research.wave30_qd.engine30 import run_genome

    reference = run_genome(cache, genome, mode="full")
    sliced = run_genome(window_cache(cache, 0, cache.n_bars), genome, mode="full")
    returns_gap = (
        float(np.max(np.abs(reference.trade_returns - sliced.trade_returns)))
        if len(reference.trade_returns) == len(sliced.trade_returns)
        else float("inf")
    )
    return {
        "reference_trades": len(reference.trades),
        "sliced_trades": len(sliced.trades),
        "trade_count_match": len(reference.trades) == len(sliced.trades),
        "max_trade_return_gap": returns_gap,
        "reference_final": float(reference.total_equity_daily[-1]),
        "sliced_final": float(sliced.total_equity_daily[-1]),
        "final_gap": abs(float(reference.total_equity_daily[-1]) - float(sliced.total_equity_daily[-1])),
    }


def main() -> int:
    from research.wave30_qd.dataio30 import build_market_cache
    from research.wave30_qd.genome30 import Genome

    cache = build_market_cache()
    champion = Genome(
        signal_family="momentum", lookback_bars=6, entry_threshold=2.553694, stop_pct=0.03812488,
        target_r=4.770807, trail_enabled=False, risk_frac=0.11483925, max_hold_bars=48,
        allow_short=True, symbols=("BTCUSDT", "ETHUSDT", "SOLUSDT"), max_concurrent=1,
        cooldown_bars_after_loss=0, sleeve_fraction=1.0,
    )
    print("=== 슬라이서 검증: 전 구간 슬라이스 == 원본 ===")
    report = verify_full_slice(cache, champion)
    for key, value in report.items():
        print(f"  {key}: {value}")
    ok = report["trade_count_match"] and report["max_trade_return_gap"] == 0.0 and report["final_gap"] == 0.0
    print(f"\n  판정: {'일치 (슬라이서 사용 가능)' if ok else '불일치 — 사용 불가'}")

    # Every per-bar field must be sliced. Missing one is not a subtle bug -- it raises on shape
    # mismatch the moment a partial window is used -- but the full-range check above CANNOT catch it,
    # because slicing [0:n] leaves unsliced arrays the right length by coincidence. So the field list is
    # asserted against the dataclass itself rather than trusted.
    per_bar_fields = {"open", "high", "low", "close", "tradable", "funding_at_bar",
                      "prior_high", "prior_low", "ret", "vol", "zscore"}
    declared = {f.name for f in dataclasses.fields(SymbolArrays)} - {"symbol", "cost_rate"}
    missing = declared - per_bar_fields
    print(f"\n=== SymbolArrays 필드 커버리지 ===")
    print(f"  선언된 per-bar 필드 {sorted(declared)}")
    print(f"  슬라이스 누락: {sorted(missing) if missing else '없음'}")
    if missing:
        print("  => 누락된 필드가 있으면 부분 창에서 shape 오류가 난다. 슬라이서를 고쳐야 한다.")
        return 1

    print("\n=== 구간 캐시 동작 확인 (365일 훈련창) ===")
    from research.wave30_qd.engine30 import run_genome

    bars_per_day = 24
    train = window_cache(cache, 0, 365 * bars_per_day)
    result = run_genome(train, champion, mode="full")
    print(f"  훈련창 봉 {train.n_bars:,} · 일 {len(train.daily_index):,} · 거래 {len(result.trades)}")
    print(f"  최종 ${result.total_equity_daily[-1]:,.2f}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
