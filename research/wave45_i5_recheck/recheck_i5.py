#!/usr/bin/env python3
# Wave-45: is the published I5 figure (CAGR 10.27%) exposed to wave44's listing artifact?
#
# wave44 found that a spot leg listing after its perp leg lets the spot's first-day price discovery be
# booked as capturable basis -- WIFUSDT's +25.26% on 2024-03-05 being the clearest case -- and that 64 of
# 332 symbols (19.3%) have that misalignment. On wave38's panel the correction cut the L4-equivalent
# configuration from +8.70% to +6.79% and its 2024 from +15.25% to +2.02%.
#
# I5 is the bar every later wave was measured against, so whether IT carries the same inflation is the
# most consequential open question in the campaign. Earlier I recorded it as unanswerable because
# wave13's cache chain was missing 116 files its own universe needs. That was wrong: every one of those
# files exists in research/wave3/cache, byte-identical where the two caches overlap (verified on row
# counts, columns and close prices). They have been restored, and I5 now reproduces at exactly the
# published +10.27%, which both confirms the published number and makes this test possible.
#
# The test itself has to isolate one variable. So this script rebuilds I5 exactly as
# engine18.run_idle_candidate does -- same universe, same cost model, same overlay layers, same loop --
# and changes ONE thing: the liquidity mask is intersected with a "both legs seasoned" mask.
# build_data_availability_mask requires 30 days of trailing PERP volume and says nothing about the spot
# leg, which is precisely why a late-listing spot leg slips through. No validated file is edited.

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Final

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

import numpy as np
import pandas as pd

from research.wave13_liquidity import universe_liquidity as ul
from research.wave13_liquidity.costs_measured import fit_mapping
from research.wave13_liquidity.engine13 import build_cost_and_liquidity_frames
from research.wave18_idle import engine18
from research.wave18_idle.configs18 import (
    CONFIGS,
    L4_CONFIG,
    LEG_FRACTION,
    OVERLAY_CARRY_CANDIDATE,
    OVERLAY_REVERSE_CANDIDATE,
    TOP_K,
)
from research.wave18_idle.engine18 import (
    ACTIVE_CAPITAL,
    OverlayLayer,
    active_frame_for,
    daily_rate_from_apr,
    reverse_active_frame_for,
)

RESULTS_DIR: Final = Path(__file__).resolve().parent / "results"
MIN_LISTING_AGE_DAYS: Final = 30  # same guard wave44 applied, same justification
PUBLISHED_I5_CAGR: Final = 0.1027


def seasoned_mask(spot_open: pd.DataFrame, perp_open: pd.DataFrame, min_age: int) -> pd.DataFrame:
    """True once BOTH legs have quoted for `min_age` days.

    Age is counted from the first day the pair is jointly observable, so a spot leg arriving months after
    its perp resets the clock rather than inheriting the perp's history.
    """
    both = spot_open.notna() & perp_open.notna()
    out = pd.DataFrame(False, index=both.index, columns=both.columns)
    for column in both.columns:
        valid = np.flatnonzero(both[column].to_numpy())
        if len(valid) == 0:
            continue
        age = np.full(len(both), -1, dtype=np.int64)
        age[valid] = valid - valid[0]
        out[column] = age >= min_age
    return out


def run_i5(apply_guard: bool, lending_apr: float) -> tuple[pd.Series, dict]:
    """I5 exactly as engine18.run_idle_candidate builds it, optionally with the seasoning guard."""
    idle_config = next(config for config in CONFIGS if config.candidate_id == "I5")
    mapping = fit_mapping()
    symbols = ul.verify_cache_and_load_symbols(L4_CONFIG)
    markets = ul.load_markets_for_symbols(symbols)
    (
        spot_open_frame,
        spot_close_frame,
        perp_open_frame,
        perp_close_frame,
        funding_frame,
        raw_score_frame,
    ) = engine18._build_aligned_frames18(markets)
    ranking_score_frame = raw_score_frame.shift(1)
    l4_active_frame = active_frame_for(raw_score_frame, L4_CONFIG.candidate)
    cost_rate_frame, liquidity_ok_frame = build_cost_and_liquidity_frames(
        L4_CONFIG, tuple(spot_open_frame.columns), spot_open_frame.index, mapping
    )

    guard_stats: dict = {"applied": apply_guard}
    if apply_guard:
        seasoned = seasoned_mask(spot_open_frame, perp_open_frame, MIN_LISTING_AGE_DAYS).reindex(
            index=liquidity_ok_frame.index, columns=liquidity_ok_frame.columns
        ).fillna(False)
        before = int(liquidity_ok_frame.to_numpy().sum())
        liquidity_ok_frame = liquidity_ok_frame & seasoned
        after = int(liquidity_ok_frame.to_numpy().sum())
        guard_stats.update({"tradable_cells_before": before, "tradable_cells_after": after,
                            "removed_share": (before - after) / before if before else 0.0})

    overlay_layers: list[OverlayLayer] = []
    if idle_config.uses_carry_overlay:
        carry_active = active_frame_for(raw_score_frame, OVERLAY_CARRY_CANDIDATE)
        overlay_symbols = idle_config.overlay_symbols if idle_config.overlay_symbols is not None else symbols
        overlay_layers.append(OverlayLayer(engine18.LAYER_CARRY_OVERLAY, carry_active, overlay_symbols, 1.0))
    if idle_config.uses_reverse_overlay:
        reverse_active = reverse_active_frame_for(raw_score_frame, OVERLAY_REVERSE_CANDIDATE)
        overlay_layers.append(OverlayLayer(engine18.LAYER_REVERSE_OVERLAY, reverse_active, symbols, -1.0))

    lending_daily_rate = daily_rate_from_apr(lending_apr) if idle_config.uses_lending_fallback else None

    result, _, _ = engine18._run_idle_overlay_loop(
        spot_open_frame,
        spot_close_frame,
        perp_open_frame,
        perp_close_frame,
        funding_frame,
        ranking_score_frame,
        l4_active_frame,
        tuple(overlay_layers),
        TOP_K,
        LEG_FRACTION,
        cost_rate_frame,
        liquidity_ok_frame,
        lending_daily_rate,
    )
    return result.equity, guard_stats


def summarise(equity: pd.Series) -> dict:
    equity = equity.dropna().sort_index()
    days = (equity.index[-1] - equity.index[0]).days
    cagr = float((equity.iloc[-1] / equity.iloc[0]) ** (365.0 / days) - 1.0)
    peak = equity.cummax()
    yearly: dict[int, float] = {}
    previous = equity.iloc[0]
    for year in sorted(set(equity.index.year)):
        segment = equity[equity.index.year == year]
        if len(segment) < 2:
            continue
        yearly[int(year)] = float(segment.iloc[-1] / previous - 1.0)
        previous = segment.iloc[-1]
    recent = [v for y, v in yearly.items() if y >= 2023]
    return {
        "final": float(equity.iloc[-1]),
        "cagr": cagr,
        "mdd": float((1.0 - equity / peak).max()),
        "yearly": yearly,
        "recent_mean": float(np.mean(recent)) if recent else float("nan"),
    }


def main() -> int:
    print("=== wave45: 공표 I5(10.27%)가 상장 인공물에 노출됐는가 ===")
    lending_json = Path("research/wave18_idle/cache/lending.json")
    lending_apr = 0.0191
    if lending_json.exists():
        try:
            lending_apr = float(json.loads(lending_json.read_text(encoding="utf-8"))["usdt_apr"])
        except (json.JSONDecodeError, KeyError, ValueError):
            pass
    print(f"USDT 대여 APR {lending_apr:.4f} (I5의 유휴자본 폴백)\n")

    print("--- (1) 가드 없음: 공표 조건 그대로 ---")
    equity_raw, _ = run_i5(apply_guard=False, lending_apr=lending_apr)
    raw = summarise(equity_raw)
    print(f"  ${ACTIVE_CAPITAL:.2f} -> ${raw['final']:.2f} | CAGR {raw['cagr']:+.2%} | MDD {raw['mdd']:.2%}")
    print(f"  공표치 {PUBLISHED_I5_CAGR:+.2%} 와 일치: {'예' if abs(raw['cagr']-PUBLISHED_I5_CAGR) < 5e-4 else '아니오'}")
    print("  연도별: " + " ".join(f"{k}:{v:+.1%}" for k, v in raw["yearly"].items()))
    print(f"  2023+ 평균 {raw['recent_mean']:+.2%}")

    print(f"\n--- (2) 상장연령 {MIN_LISTING_AGE_DAYS}일 가드 적용 ---")
    equity_fixed, guard = run_i5(apply_guard=True, lending_apr=lending_apr)
    fixed = summarise(equity_fixed)
    print(f"  거래가능 셀 {guard['tradable_cells_before']:,} -> {guard['tradable_cells_after']:,} "
          f"({guard['removed_share']:.2%} 제거)")
    print(f"  ${ACTIVE_CAPITAL:.2f} -> ${fixed['final']:.2f} | CAGR {fixed['cagr']:+.2%} | MDD {fixed['mdd']:.2%}")
    print("  연도별: " + " ".join(f"{k}:{v:+.1%}" for k, v in fixed["yearly"].items()))
    print(f"  2023+ 평균 {fixed['recent_mean']:+.2%}")

    print("\n=== 판정 ===")
    delta = fixed["cagr"] - raw["cagr"]
    delta_recent = fixed["recent_mean"] - raw["recent_mean"]
    print(f"  전기간 CAGR {raw['cagr']:+.2%} -> {fixed['cagr']:+.2%}  ({delta:+.2%}p)")
    print(f"  2023+ 평균  {raw['recent_mean']:+.2%} -> {fixed['recent_mean']:+.2%}  ({delta_recent:+.2%}p)")
    worst_year = min(raw["yearly"], key=lambda y: fixed["yearly"].get(y, 0.0) - raw["yearly"][y])
    print(f"  가장 크게 하락한 연도: {worst_year} "
          f"{raw['yearly'][worst_year]:+.1%} -> {fixed['yearly'].get(worst_year, float('nan')):+.1%}")
    if delta < -0.005:
        print("  => 공표 I5 는 상장 인공물에 노출돼 있었다. 모든 게이트 기준선을 정정해야 한다.")
    elif delta > 0.005:
        print("  => 가드가 오히려 수익을 높였다(인공물이 손실 방향이었음). 기준선은 상향된다.")
    else:
        print("  => 노출이 유의하지 않다. 공표 I5 는 이 결함과 무관하며 기준선은 유효하다.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "i5_recheck.json").write_text(
        json.dumps(
            {
                "wave": "wave45_i5_recheck",
                "published_cagr": PUBLISHED_I5_CAGR,
                "min_listing_age_days": MIN_LISTING_AGE_DAYS,
                "restored_cache_files": 116,
                "no_guard": raw,
                "with_guard": fixed,
                "guard_stats": guard,
                "delta_cagr": delta,
                "delta_recent_mean": delta_recent,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print("\nresults/i5_recheck.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
