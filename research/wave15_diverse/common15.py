# Wave-15 shared plumbing: capital/cost constants, market-frame assembly, and the ONE
# generic daily delta-neutral bookkeeping loop reused by B1/B2/C1 (and, read-only, the L4
# reference recompute). engine_intraday.py (A1-A3) and engine_pairs.py (D1) are structurally
# different enough (settlement-timed state machine; two-perp-leg direction freezing) that
# they get their own loops rather than being force-fit through this one -- but both still
# reuse this module's cost-model wiring and both still emit Wave10Result, so every candidate
# stays compatible with research.wave10_carry100.regime.regime_breakdown and gates15.py
# unmodified.
#
# SPEC.md "공통 규약": 자본 $100, 활성 $90, 레그 $45, 레버리지 1x, 최소주문 $5, wave-13 실측
#비용매핑 재사용, 체결 시그널확정->다음바, IS~2025-09/OOS 2025-10~. Every constant below is a
# read of an EXISTING wave's own frozen constant (research.wave10_carry100.engine), not a new
# literal -- this wave changes the profit MECHANISM, not these numbers.

from __future__ import annotations

from pathlib import Path
import sys
from typing import Final

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

import pandas as pd  # noqa: PANDAS_OK

from research.wave1.fam_funding import FundingMarket
from research.wave10_carry100.engine import (
    ACTIVE_CAPITAL,
    MIN_ORDER_USDT,
    OOS_SPLIT,
    RESERVE_FRACTION,
    TOTAL_CAPITAL,
    Wave10Result,
)
from research.wave12_frontier import universe_frontier as uf12
from research.wave13_liquidity import costs_measured
from research.wave2.funding import W2_MAKER_FEE_RATE

LEG_FRACTION: Final = 0.50  # fraction of ACTIVE_CAPITAL per leg -- SPEC.md "레그 $45" (0.5 * $90)
LEG_USDT: Final = LEG_FRACTION * ACTIVE_CAPITAL  # $45.00
GROSS_USDT: Final = 2.0 * LEG_USDT  # $90.00 == ACTIVE_CAPITAL -> 1x leverage by construction
TOP_K: Final = 1  # SPEC.md "1쌍" every single wave15 candidate

BASE_STRESS_MULTIPLIER: Final = 1.0
STRESS_MULTIPLIER: Final = 3.0  # SPEC.md gate S5 inherits wave13's stress bar (measured-cost snapshot risk)

ENTRY_THRESHOLD_APR: Final = 0.15  # baseline carry entry bar, reused verbatim by B1/B2 and as C1's/A3's slow-mode entry
EXIT_THRESHOLD_APR: Final = ENTRY_THRESHOLD_APR / 2.0  # 7.5% -- carry_position's own hysteresis convention

ASSUMED_FLEXIBLE_EARN_APR: Final = 0.02  # B1 spot leg / B2 USDT collateral -- see earn_apr.py for the (failed) live-fetch attempt this falls back from. NEVER label results using this as "verified."


def leg_usdt() -> float:
    return LEG_USDT


def gross_usdt() -> float:
    return GROSS_USDT


# ---------------------------------------------------------------------------
# Cost model -- wave-13's measured mapping, reused unmodified (SPEC.md: "비용:
# wave-13 실측 매핑 재사용"). Fitted once (network-free -- reads
# research/wave13_liquidity/cache/measured_spreads.json) and shared by every wave15 candidate.
# ---------------------------------------------------------------------------


def fit_measured_cost_mapping() -> costs_measured.MeasuredCostMapping:
    return costs_measured.fit_mapping()


def two_leg_cost_rate_from_bp(bp: float | pd.DataFrame, stress_multiplier: float = 1.0) -> float | pd.DataFrame:
    """Both-legs one-way cost rate: maker 0.02% x2 legs + measured slippage x2 legs. Identical
    formula to costs_measured.cost_rate_from_bp -- re-exposed here under an explicit name
    because B2/D1 below need the CONTRASTING single-leg variant and a two-different-bp-legs
    variant side by side, and "which one applies where" is easier to audit with distinct names."""
    return costs_measured.cost_rate_from_bp(bp, stress_multiplier)


def single_leg_cost_rate_from_bp(bp: float | pd.DataFrame, stress_multiplier: float = 1.0) -> float | pd.DataFrame:
    """One leg's one-way cost rate: maker 0.02% + measured slippage, NOT doubled. B2 has no
    spot leg (SPEC.md: "현물 레그를 USDT 대출 이자로 대체... 숏퍼프만") so it must not be charged
    the spot leg's fee/slippage it never actually pays."""
    return W2_MAKER_FEE_RATE + (bp * 0.0001) * stress_multiplier


def build_cost_rate_frame(
    quote_volume_frame: pd.DataFrame,
    symbols: tuple[str, ...],
    mapping: costs_measured.MeasuredCostMapping,
    stress_multiplier: float = 1.0,
    single_leg: bool = False,
    index: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Point-in-time-safe (research.wave13_liquidity.costs_measured.point_in_time_known_avg's
    own shift(1)-of-trailing-30d-mean convention, unchanged) cost-rate frame, in either the
    normal both-legs form (every wave15 candidate except B2) or the single-leg form (B2 only).

    `index`, when given, reindexes the result onto the CALLER's own price-frame index (e.g.
    engine_daily.run_generic_carry's spot_open_frame.index) with a fail-closed fallback for
    any date the volume series doesn't cover -- mirrors
    research.wave13_liquidity.engine13.build_cost_and_liquidity_frames's own
    reindex+fillna(worst_bp) convention. Omitting `index` returns the frame on
    quote_volume_frame's own (possibly slightly different) date range, which is USUALLY wrong
    for a caller that will do `.loc[timestamp, symbol]` against a different frame's index --
    every call site in this wave passes `index` explicitly; the parameter stays optional only
    so unit tests can inspect the raw unaligned frame."""
    known_avg = costs_measured.point_in_time_known_avg(quote_volume_frame).reindex(columns=list(symbols))
    bp_frame = costs_measured.bp_frame_from_known_avg(known_avg, mapping)
    rate = single_leg_cost_rate_from_bp(bp_frame, stress_multiplier) if single_leg else two_leg_cost_rate_from_bp(bp_frame, stress_multiplier)
    if index is None:
        return rate
    fallback_bp = mapping.worst_bp
    fallback_rate = single_leg_cost_rate_from_bp(fallback_bp, stress_multiplier) if single_leg else two_leg_cost_rate_from_bp(fallback_bp, stress_multiplier)
    return rate.reindex(index=index, columns=list(symbols)).fillna(fallback_rate)


def build_liquidity_mask(quote_volume_frame: pd.DataFrame, symbols: tuple[str, ...]) -> pd.DataFrame:
    """Same L1-L4 philosophy as wave13 (data-availability only, no extra dollar floor bolted
    on -- wave15 isn't testing liquidity breadth, wave13 already did)."""
    return costs_measured.build_data_availability_mask(quote_volume_frame, symbols)


# ---------------------------------------------------------------------------
# Market data -- borrowed read-only from wave12_frontier's own cache-resolution chain
# (CACHE_DIR -> wave11_yield's cache -> wave1's cache; see universe_frontier._first_existing),
# exactly like wave13_liquidity/universe_liquidity.py does. No network fetch of daily
# spot/perp/funding data -- every symbol this wave's daily engines (B1/B2/C1/D1) touch
# already has a complete binance_{spot,fapi}_*_1d.csv.gz + binance_funding_*.csv.gz triple
# on disk (verified interactively before writing this module).
# ---------------------------------------------------------------------------


def load_markets(symbols: tuple[str, ...]) -> dict[str, FundingMarket]:
    missing = uf12.missing_market_files(symbols)
    if missing:
        raise RuntimeError(f"wave15 daily-engine cache incomplete: {', '.join(sorted(set(missing))[:8])}")
    return uf12.load_markets_for_symbols(symbols)


def load_quote_volume_frame(symbols: tuple[str, ...]) -> pd.DataFrame:
    return uf12.load_quote_volume_frame(symbols)


def load_candidate_pool() -> dict:
    return uf12.load_candidate_pool()


def reference_volume_30d(pool: dict, symbol: str) -> float:
    info = pool["symbols"].get(symbol)
    if info is None or not info.get("ok"):
        raise KeyError(f"{symbol} missing/not-ok in wave12_frontier candidate_pool.json")
    return float(info["reference_volume_30d_usdt"])


# ---------------------------------------------------------------------------
# Daily frame assembly -- deliberately a light LOCAL copy of
# research.wave13_liquidity.engine13._build_aligned_frames's price/funding half only (NOT
# its scoring/active half -- every wave15 daily candidate computes its OWN active_frame/
# score_frame from a different signal, see signals15.py), matching this repo's established
# wave10->wave11->wave12->wave13 precedent of copying frame-assembly rather than importing
# a private cross-wave helper.
# ---------------------------------------------------------------------------


def build_price_frames(markets: dict[str, FundingMarket]) -> dict[str, pd.DataFrame]:
    spot_open: dict[str, pd.Series] = {}
    spot_close: dict[str, pd.Series] = {}
    perp_open: dict[str, pd.Series] = {}
    perp_close: dict[str, pd.Series] = {}
    funding_daily: dict[str, pd.Series] = {}
    funding_native: dict[str, pd.Series] = {}
    for symbol, market in markets.items():
        spot_daily = market.spot.resample("1D").agg({"open": "first", "close": "last"}).dropna()
        perp_daily = market.perp.resample("1D").agg({"open": "first", "close": "last"}).dropna()
        spot_open[symbol] = spot_daily["open"]
        spot_close[symbol] = spot_daily["close"]
        perp_open[symbol] = perp_daily["open"]
        perp_close[symbol] = perp_daily["close"]
        funding_daily[symbol] = market.funding.resample("1D").sum()
        funding_native[symbol] = market.funding

    spot_open_frame = pd.DataFrame(spot_open).sort_index()
    return {
        "spot_open": spot_open_frame,
        "spot_close": pd.DataFrame(spot_close).reindex(spot_open_frame.index),
        "perp_open": pd.DataFrame(perp_open).reindex(spot_open_frame.index),
        "perp_close": pd.DataFrame(perp_close).reindex(spot_open_frame.index),
        "funding_daily": pd.DataFrame(funding_daily).reindex(spot_open_frame.index).fillna(0.0),
    }


__all__ = [
    "ACTIVE_CAPITAL",
    "ASSUMED_FLEXIBLE_EARN_APR",
    "BASE_STRESS_MULTIPLIER",
    "ENTRY_THRESHOLD_APR",
    "EXIT_THRESHOLD_APR",
    "GROSS_USDT",
    "LEG_FRACTION",
    "LEG_USDT",
    "MIN_ORDER_USDT",
    "OOS_SPLIT",
    "RESERVE_FRACTION",
    "STRESS_MULTIPLIER",
    "TOP_K",
    "TOTAL_CAPITAL",
    "Wave10Result",
    "build_cost_rate_frame",
    "build_liquidity_mask",
    "build_price_frames",
    "fit_measured_cost_mapping",
    "leg_usdt",
    "gross_usdt",
    "load_candidate_pool",
    "load_markets",
    "load_quote_volume_frame",
    "reference_volume_30d",
    "single_leg_cost_rate_from_bp",
    "two_leg_cost_rate_from_bp",
]
