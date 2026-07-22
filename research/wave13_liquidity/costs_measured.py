# Wave-13 measured cost model (SPEC.md "작업 1b"): replaces every prior wave's
# rank-or-assumption-based slippage table (research.wave1.costs.slippage_rate's flat
# 1bp/3bp/5bp symbol lookup; research.wave12_frontier.costs_tiered's assumed
# 1/3/6/10/20bp rank tiers) with a function fitted DIRECTLY to
# research/wave13_liquidity/cache/measured_spreads.json -- real Bitget order-book
# measurements (collect_spreads.py), not an estimate.
#
# Domain is TRADING VOLUME (USDT), not rank -- SPEC.md is explicit about this ("거래대금 대비
# 회귀/구간중앙값으로 거래대금->슬리피지 매핑 함수 도출"), and the measured data itself is why:
# rank and cost are NOT monotonically related in the raw Bitget snapshot (e.g. AMCUSDT at
# volume-rank ~100 measured 18.2bp, materially worse than several rank-~300 names -- a
# rank-tier table would have silently underpriced it). Volume is the more defensible axis
# because it's the same quantity this repo's engines already track point-in-time per symbol
# per day (research.wave12_frontier.costs_tiered's own `known_avg` rolling-30d-mean-shift(1)
# series) -- no separate cross-sectional rank computation needed downstream.
#
# The raw (volume, effective_slippage_bp) scatter is noisy and NOT monotonic (see above), so
# a plain regression would either underfit the noise or (worse) imply cheaper cost at lower
# volume than an adjacent higher-volume bucket purely from sampling noise. This module
# therefore buckets by log10(volume), takes the per-bucket MEDIAN (robust to the AMC-style
# single-symbol anomalies), and then applies isotonic regression (pool-adjacent-violators,
# L2, non-increasing) across the bucket medians so the FITTED mapping is monotonic even
# though the underlying data is not -- tests/test_wave13.py pins the monotonicity of the
# fitted function, not of the raw measurements (which are reported separately, anomalies
# included, in reporting13.py's spread table).
#
# Unlike costs_tiered.py, there is no separate MAJOR_SYMBOLS override for BTC/ETH: the
# volume-fitted mapping already prices them at the cheapest end on its own (BTC/ETH's own
# measured 24h volume lands them past every other bucket's right edge), which is itself a
# finding worth keeping visible rather than papering over with a hardcoded exception.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Final

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

import numpy as np
import pandas as pd  # noqa: PANDAS_OK

from research.wave2.funding import W2_MAKER_FEE_RATE
from research.wave13_liquidity.collect_spreads import load_measured_spreads

ROLLING_WINDOW_DAYS: Final = 30  # matches SPEC.md "30d 평균 거래대금" / costs_tiered.py's own window
N_BUCKETS: Final = 9  # log-spaced volume buckets fed to the isotonic fit; 66 measured points / 9 ~= 7 points/bucket
MIN_BUCKET_POINTS: Final = 1  # a bucket with >=1 point is usable (median of 1 = that point); empty buckets are dropped, not imputed


@dataclass(frozen=True, slots=True)
class MeasuredCostMapping:
    """A monotonic (non-increasing in volume) piecewise-linear-in-log10(volume) fit over
    Bitget-measured effective_slippage_bp. `anchor_log_volume` is strictly ascending;
    `anchor_bp` is non-increasing and isotonic-regressed (see module docstring) -- both
    arrays are the same length, one entry per non-empty bucket actually observed."""

    anchor_log_volume: np.ndarray
    anchor_bp: np.ndarray
    bucket_counts: tuple[int, ...]
    raw_point_count: int
    source_collected_at_utc: str

    @property
    def worst_bp(self) -> float:
        """Fail-closed fallback for volume=NaN/unknown/<=0 (no trailing history yet, or a
        symbol entirely outside the measured range on the illiquid side): the single most
        expensive anchor, never a cheap default -- same fail-closed convention as
        costs_tiered.TAIL_SLIPPAGE_BP."""
        return float(self.anchor_bp[0])

    @property
    def best_bp(self) -> float:
        return float(self.anchor_bp[-1])


def _pava_non_increasing(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Pool-Adjacent-Violators Algorithm, L2 loss, non-increasing constraint. Standard
    textbook stack algorithm: scan left to right, and whenever the running block's average
    is less than the new block's average (a rise, which violates "must not increase"),
    merge them into one pooled block (weighted average) and keep checking backward. Used
    here (not a canned library) to keep wave13_liquidity dependency-free, matching this
    repo's existing convention of hand-rolled MC/block-shuffle statistics in gates12.py
    rather than pulling in scipy/sklearn for a single call site."""
    stack_sum: list[float] = []
    stack_weight: list[float] = []
    stack_count: list[int] = []
    for value, weight in zip(values, weights):
        stack_sum.append(value * weight)
        stack_weight.append(weight)
        stack_count.append(1)
        while len(stack_sum) >= 2 and (stack_sum[-2] / stack_weight[-2]) < (stack_sum[-1] / stack_weight[-1]):
            merged_sum = stack_sum.pop() + stack_sum[-1]
            merged_weight = stack_weight.pop() + stack_weight[-1]
            merged_count = stack_count.pop() + stack_count[-1]
            stack_sum[-1], stack_weight[-1], stack_count[-1] = merged_sum, merged_weight, merged_count
    output: list[float] = []
    for total, weight, count in zip(stack_sum, stack_weight, stack_count):
        output.extend([total / weight] * count)
    return np.asarray(output, dtype=float)


def _bucket_medians(measurements: list[dict[str, Any]], n_buckets: int) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    volumes = np.array([float(item["usdt_volume_24h"]) for item in measurements], dtype=float)
    slippage = np.array([float(item["effective_slippage_bp"]) for item in measurements], dtype=float)
    log_volume = np.log10(volumes)
    edges = np.linspace(log_volume.min(), log_volume.max(), n_buckets + 1)
    centers: list[float] = []
    medians: list[float] = []
    counts: list[int] = []
    for index in range(n_buckets):
        lower, upper = edges[index], edges[index + 1]
        mask = (log_volume >= lower) & (log_volume <= upper) if index == n_buckets - 1 else (log_volume >= lower) & (log_volume < upper)
        if int(mask.sum()) < MIN_BUCKET_POINTS:
            continue
        centers.append(float(np.median(log_volume[mask])))  # actual observed center, not the geometric bin midpoint
        medians.append(float(np.median(slippage[mask])))
        counts.append(int(mask.sum()))
    order = np.argsort(centers)
    return np.asarray(centers)[order], np.asarray(medians)[order], tuple(np.asarray(counts)[order].tolist())


def fit_mapping(payload: dict[str, Any] | None = None, n_buckets: int = N_BUCKETS) -> MeasuredCostMapping:
    payload = payload if payload is not None else load_measured_spreads()
    measurements = payload["measurements"]
    if len(measurements) < n_buckets:
        raise ValueError(f"need >= {n_buckets} measured points to fit {n_buckets} buckets, got {len(measurements)}")
    anchor_log_volume, bucket_medians, counts = _bucket_medians(measurements, n_buckets)
    weights = np.asarray(counts, dtype=float)
    anchor_bp = _pava_non_increasing(bucket_medians, weights)
    return MeasuredCostMapping(
        anchor_log_volume=anchor_log_volume,
        anchor_bp=anchor_bp,
        bucket_counts=counts,
        raw_point_count=len(measurements),
        source_collected_at_utc=str(payload.get("collected_at_utc", "")),
    )


def slippage_bp_for_volume(volume_usdt: float | None, mapping: MeasuredCostMapping) -> float:
    """Scalar lookup: log-linear interpolation between `mapping`'s isotonic anchor points,
    flat-extrapolated beyond the observed range (never assumes a symbol cheaper/more-liquid
    than the cheapest/most-liquid thing this wave actually measured, and never assumes a
    symbol more expensive than the single worst thing measured either -- both directions
    are held flat, not extrapolated past the data). NaN/None/non-positive volume (no
    trailing history yet) fails closed to `mapping.worst_bp`, never to a cheap default."""
    if volume_usdt is None or not np.isfinite(volume_usdt) or volume_usdt <= 0.0:
        return mapping.worst_bp
    log_volume = np.log10(float(volume_usdt))
    return float(np.interp(log_volume, mapping.anchor_log_volume, mapping.anchor_bp, left=mapping.worst_bp, right=mapping.best_bp))


def cost_rate_from_bp(bp: float | pd.DataFrame, stress_multiplier: float = 1.0) -> float | pd.DataFrame:
    """Identical convention to research.wave12_frontier.costs_tiered.cost_rate_from_bp
    (reimplemented locally rather than imported -- see gates12.py's own precedent of
    reimplementing rather than cross-importing wave-specific cost/gate logic): maker fee
    0.02% on each leg (stress-invariant -- SPEC.md's S5 stress only scales slippage, never
    the maker fee) plus slippage on both legs, scaled by `stress_multiplier` (1.0 base, 3.0
    for wave13's S5 -- SPEC.md: "실측슬리피지 x3", a strictly larger multiplier than wave12's
    S5 (x2), reflecting that this wave's base cost is a real but single calm-market
    snapshot rather than an already-conservative estimate)."""
    return 2.0 * W2_MAKER_FEE_RATE + 2.0 * (bp * 0.0001) * stress_multiplier


def rolling_trailing_avg_volume(quote_volume_frame: pd.DataFrame, window: int = ROLLING_WINDOW_DAYS) -> pd.DataFrame:
    return quote_volume_frame.rolling(window, min_periods=window).mean()


def point_in_time_known_avg(quote_volume_frame: pd.DataFrame, window: int = ROLLING_WINDOW_DAYS) -> pd.DataFrame:
    """Trailing `window`-day average quote_volume AS OF THE PREVIOUS BAR'S CLOSE (shift(1))
    -- what is actually knowable at the moment day t's t+1-open trade is decided. Same lag
    convention as research.wave1.fam_funding's score_frame shift(1) and
    research.wave12_frontier.costs_tiered.point_in_time_known_avg;
    tests/test_wave13.py's lookahead test pins that a future volume spike never moves an
    earlier day's mapped bp, the volume-domain analogue of test_wave12.py's rank version."""
    return rolling_trailing_avg_volume(quote_volume_frame, window).shift(1)


def bp_frame_from_known_avg(known_avg_frame: pd.DataFrame, mapping: MeasuredCostMapping) -> pd.DataFrame:
    """Vectorized research.wave13_liquidity.costs_measured.slippage_bp_for_volume over a
    whole (date x symbol) frame at once -- numerically identical to the scalar function
    (tests/test_wave13.py cross-checks this), just avoiding a Python-level double loop over
    every day/symbol cell in engine13.py's hot path."""
    values = known_avg_frame.to_numpy(dtype=float)
    flat = values.reshape(-1)
    valid = np.isfinite(flat) & (flat > 0.0)
    bp_flat = np.full(flat.shape, mapping.worst_bp, dtype=float)
    if valid.any():
        log_volume = np.log10(flat[valid])
        bp_flat[valid] = np.interp(log_volume, mapping.anchor_log_volume, mapping.anchor_bp, left=mapping.worst_bp, right=mapping.best_bp)
    return pd.DataFrame(bp_flat.reshape(values.shape), index=known_avg_frame.index, columns=known_avg_frame.columns)


def build_cost_rate_frame(
    quote_volume_frame: pd.DataFrame,
    symbols: tuple[str, ...],
    mapping: MeasuredCostMapping,
    stress_multiplier: float = 1.0,
    window: int = ROLLING_WINDOW_DAYS,
) -> pd.DataFrame:
    """One-way cost rate (maker + measured-mapped slippage) per symbol per day, reindexed
    to exactly `symbols`. Never NaN: any symbol/day without a valid trailing average falls
    back to mapping.worst_bp via bp_frame_from_known_avg, so this frame is always safe to
    multiply directly into engine13.py's capital-compounding arithmetic (same contract as
    research.wave12_frontier.engine12.build_cost_frames_for_symbols)."""
    known_avg = point_in_time_known_avg(quote_volume_frame, window).reindex(columns=list(symbols))
    bp_frame = bp_frame_from_known_avg(known_avg, mapping)
    return cost_rate_from_bp(bp_frame, stress_multiplier)


def build_data_availability_mask(quote_volume_frame: pd.DataFrame, symbols: tuple[str, ...], window: int = ROLLING_WINDOW_DAYS) -> pd.DataFrame:
    """L1-L4's liquidity mask: True iff a point-in-time trailing average is even KNOWN that
    day (>= `window` days of history) -- a pure data-availability gate, not a business
    liquidity rule. SPEC.md deliberately does NOT give L1-L4 a dollar liquidity floor the
    way wave12 gave every config a flat $2M floor: the whole point of this wave is to see
    whether the measured cost model's OWN bp pricing (which can be large for thin names) is
    enough to make thin symbols self-limiting, without an extra explicit filter bolted on
    top -- that explicit filter is L5's own distinguishing feature, not a universal rule."""
    known_avg = point_in_time_known_avg(quote_volume_frame, window).reindex(columns=list(symbols))
    return known_avg.notna()


def build_dynamic_liquidity_mask(
    quote_volume_frame: pd.DataFrame,
    symbols: tuple[str, ...],
    mapping: MeasuredCostMapping,
    volume_floor_usdt: float,
    slippage_cap_bp: float,
    window: int = ROLLING_WINDOW_DAYS,
) -> pd.DataFrame:
    """L5's dynamic filter (SPEC.md's core hypothesis): True iff BOTH (a) the point-in-time
    trailing 30d average volume clears `volume_floor_usdt`, AND (b) the measured-mapping bp
    at that same trailing volume is <= `slippage_cap_bp`. Deliberately evaluated fresh every
    day (no persistent "already entered, skip re-check" exemption) -- same rationale
    costs_tiered.py's own liquidity mask documents: engine13.py's `eligible` set is
    recomputed from scratch every timestamp, so there is no separate per-position
    entry-event state to hook a one-time-only check into (see engine13.py's own docstring).
    The bp check is evaluated on the BASE (unstressed) mapping, never
    stress_multiplier-scaled -- this is a structural eligibility rule, not the cost
    conversion step (mirrors wave12's liquidity floor, which is likewise never
    stress-scaled; only build_cost_rate_frame's rate output moves under stress)."""
    known_avg = point_in_time_known_avg(quote_volume_frame, window).reindex(columns=list(symbols))
    bp_frame = bp_frame_from_known_avg(known_avg, mapping)
    volume_ok = known_avg >= volume_floor_usdt
    slippage_ok = bp_frame <= slippage_cap_bp
    return volume_ok & slippage_ok


__all__ = [
    "MIN_BUCKET_POINTS",
    "N_BUCKETS",
    "ROLLING_WINDOW_DAYS",
    "MeasuredCostMapping",
    "bp_frame_from_known_avg",
    "build_cost_rate_frame",
    "build_data_availability_mask",
    "build_dynamic_liquidity_mask",
    "cost_rate_from_bp",
    "fit_mapping",
    "point_in_time_known_avg",
    "rolling_trailing_avg_volume",
    "slippage_bp_for_volume",
]
