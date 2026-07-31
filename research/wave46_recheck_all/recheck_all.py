#!/usr/bin/env python3
# Wave-46: re-run every wave18 candidate under the seasoning guard and see which verdicts move.
#
# wave45 showed the campaign's benchmark was inflated: I5 falls from +10.27% to +8.28% once a late-listing
# spot leg can no longer be traded on its first day. But wave18's promotion rule is RELATIVE --
# "S1-S6 all pass AND full-period CAGR > I0" -- so correcting one candidate settles nothing. If the
# artifact hit I0 harder than I3, I3's rejection (8.72% against I0's 9.37%) could reverse; if it hit them
# equally, the ranking is untouched and only the absolute numbers were wrong. Which of those is true is
# not guessable from wave45 alone, so all six are re-run.
#
# One code path handles all six. engine18's docstring records that its loop with zero overlays and no
# lending reduces to I0 exactly, and tests/test_wave18.py proves it, so routing I0 through the same loop
# as I1-I5 means the guard is applied identically to every candidate rather than through two paths that
# could differ. The frames are built once and shared, since loading 200 symbols' markets dominates
# runtime.
#
# Nothing in the validated engines is edited. The guard is applied by intersecting the liquidity mask,
# exactly as wave45 did.

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
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
from research.wave45_i5_recheck.recheck_i5 import MIN_LISTING_AGE_DAYS, seasoned_mask, summarise

RESULTS_DIR: Final = Path(__file__).resolve().parent / "results"
# wave18's published figures, for a like-for-like before/after table.
PUBLISHED: Final = {"I0": 0.0937, "I1": 0.1024, "I2": 0.0971, "I3": 0.0872, "I4": None, "I5": 0.1027}


def build_shared():
    mapping = fit_mapping()
    symbols = ul.verify_cache_and_load_symbols(L4_CONFIG)
    markets = ul.load_markets_for_symbols(symbols)
    frames = engine18._build_aligned_frames18(markets)
    cost_rate_frame, liquidity_ok_frame = build_cost_and_liquidity_frames(
        L4_CONFIG, tuple(frames[0].columns), frames[0].index, mapping
    )
    seasoned = (
        seasoned_mask(frames[0], frames[2], MIN_LISTING_AGE_DAYS)
        .reindex(index=liquidity_ok_frame.index, columns=liquidity_ok_frame.columns)
        .fillna(False)
    )
    return symbols, frames, cost_rate_frame, liquidity_ok_frame, seasoned


def run_candidate(candidate_id: str, shared, lending_apr: float, apply_guard: bool) -> pd.Series:
    symbols, frames, cost_rate_frame, liquidity_ok_frame, seasoned = shared
    spot_open, spot_close, perp_open, perp_close, funding, raw_score = frames
    idle_config = next(config for config in CONFIGS if config.candidate_id == candidate_id)

    liquidity = liquidity_ok_frame & seasoned if apply_guard else liquidity_ok_frame

    overlay_layers: list[OverlayLayer] = []
    if idle_config.uses_carry_overlay:
        carry_active = active_frame_for(raw_score, OVERLAY_CARRY_CANDIDATE)
        overlay_symbols = idle_config.overlay_symbols if idle_config.overlay_symbols is not None else symbols
        overlay_layers.append(OverlayLayer(engine18.LAYER_CARRY_OVERLAY, carry_active, overlay_symbols, 1.0))
    if idle_config.uses_reverse_overlay:
        reverse_active = reverse_active_frame_for(raw_score, OVERLAY_REVERSE_CANDIDATE)
        overlay_layers.append(OverlayLayer(engine18.LAYER_REVERSE_OVERLAY, reverse_active, symbols, -1.0))

    lending_daily_rate = daily_rate_from_apr(lending_apr) if idle_config.uses_lending_fallback else None

    result, _, _ = engine18._run_idle_overlay_loop(
        spot_open,
        spot_close,
        perp_open,
        perp_close,
        funding,
        raw_score.shift(1),
        active_frame_for(raw_score, L4_CONFIG.candidate),
        tuple(overlay_layers),
        TOP_K,
        LEG_FRACTION,
        cost_rate_frame,
        liquidity,
        lending_daily_rate,
    )
    return result.equity


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="wave46: re-run every wave18 candidate with the guard")
    parser.add_argument("--only", help="single candidate id (I0..I5)")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    started = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "recheck_all.json"
    payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"candidates": {}}

    lending_apr = 0.0191
    lending_json = Path("research/wave18_idle/cache/lending.json")
    if lending_json.exists():
        try:
            lending_apr = float(json.loads(lending_json.read_text(encoding="utf-8"))["usdt_apr"])
        except (json.JSONDecodeError, KeyError, ValueError):
            pass

    print("=== wave46: wave18 전 후보 재검정 (상장연령 가드) ===")
    print(f"판정 규칙: S1~S6 PASS ∧ 전기간 CAGR > I0. 기준선이 바뀌면 순위가 바뀐다.\n")
    shared = build_shared()
    print(f"공유 프레임 구축 {time.time()-started:.0f}s · 종목 {len(shared[0])}\n")

    wanted = [args.only] if args.only else [config.candidate_id for config in CONFIGS]
    for candidate_id in wanted:
        if candidate_id in payload["candidates"] and not args.only:
            print(f"  {candidate_id} (캐시)")
            continue
        record = {}
        for guard in (False, True):
            equity = run_candidate(candidate_id, shared, lending_apr, guard)
            record["with_guard" if guard else "no_guard"] = summarise(equity)
        record["published"] = PUBLISHED.get(candidate_id)
        payload["candidates"][candidate_id] = record
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        raw, fixed = record["no_guard"], record["with_guard"]
        print(
            f"  {candidate_id}: 전기간 {raw['cagr']:+7.2%} -> {fixed['cagr']:+7.2%} "
            f"({fixed['cagr']-raw['cagr']:+.2%}p) | 2023+ {raw['recent_mean']:+6.2%} -> {fixed['recent_mean']:+6.2%}",
            flush=True,
        )

    if args.only:
        print(f"\n{time.time()-started:.0f}s · 나머지는 --only 없이 실행하면 캐시를 재사용한다.")
        return 0

    print("\n=== 정정 전/후 대조 ===")
    print(f"{'후보':>5} {'공표':>8} {'재현':>8} {'가드후':>8} {'차이':>8} {'2023+ 전':>9} {'2023+ 후':>9}")
    table = []
    for candidate_id in [config.candidate_id for config in CONFIGS]:
        record = payload["candidates"].get(candidate_id)
        if not record:
            continue
        raw, fixed = record["no_guard"], record["with_guard"]
        published = record.get("published")
        table.append((candidate_id, published, raw["cagr"], fixed["cagr"], raw["recent_mean"], fixed["recent_mean"]))
        pub = f"{published:+7.2%}" if published is not None else "      -"
        print(f"{candidate_id:>5} {pub} {raw['cagr']:+7.2%} {fixed['cagr']:+7.2%} "
              f"{fixed['cagr']-raw['cagr']:+7.2%}p {raw['recent_mean']:+8.2%} {fixed['recent_mean']:+8.2%}")

    baseline_before = next((r[2] for r in table if r[0] == "I0"), None)
    baseline_after = next((r[3] for r in table if r[0] == "I0"), None)
    print(f"\n=== 판정 이동 (기준선 I0: {baseline_before:+.2%} -> {baseline_after:+.2%}) ===")
    flips = []
    for candidate_id, _published, before, after, _rb, _ra in table:
        if candidate_id == "I0":
            continue
        beat_before = before > baseline_before
        beat_after = after > baseline_after
        mark = ""
        if beat_before != beat_after:
            mark = "  <-- 판정 역전"
            flips.append({"candidate": candidate_id, "beat_before": beat_before, "beat_after": beat_after})
        print(f"  {candidate_id}: 기준선 초과 {('예' if beat_before else '아니오')} -> "
              f"{('예' if beat_after else '아니오')}{mark}")

    ranking_before = [r[0] for r in sorted(table, key=lambda r: -r[2])]
    ranking_after = [r[0] for r in sorted(table, key=lambda r: -r[3])]
    print(f"\n  순위 (전기간, 정정 전): {' > '.join(ranking_before)}")
    print(f"  순위 (전기간, 정정 후): {' > '.join(ranking_after)}")
    print(f"  순위 변동: {'있음' if ranking_before != ranking_after else '없음'}")

    payload["summary"] = {
        "baseline_before": baseline_before,
        "baseline_after": baseline_after,
        "ranking_before": ranking_before,
        "ranking_after": ranking_after,
        "ranking_changed": ranking_before != ranking_after,
        "verdict_flips": flips,
    }
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\n{time.time()-started:.0f}s · results/recheck_all.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
