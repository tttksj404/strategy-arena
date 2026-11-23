# Wave-14 per-venue measured cost model. SPEC.md: "거래소를 추가하면 그 거래소 호가도 같은
# 방식(half-spread + $45 book-walk)으로 실측해 별도 매핑 산출할 것" + "거래소별 수수료
# 실제값 사용(Bitget 메이커 0.02%, Bybit/OKX 각 공시값 확인)".
#
# Two distinct pieces, deliberately kept separate:
#   1. The BINANCE-sourced leg of every candidate (in both engine14 structures) reuses
#      research.wave13_liquidity.costs_measured UNCHANGED (import, not reimplementation) --
#      wave13's Bitget-measured mapping is already this repo's established proxy for what a
#      Binance-listed symbol costs to trade (wave13 measured Bitget order books as a stand-in
#      for Binance execution cost; wave14 does not re-derive or second-guess that choice, per
#      the wave14 task's own "캐리 엔진은 ... 재사용. 룰 변경 금지 — 바꾸는 건 거래소 소스·
#      동시 쌍 수·자본뿐"). `costs_measured.cost_rate_from_bp(bp, stress)` already bakes in
#      BOTH legs of a same-venue spot+perp pair (its own `2.0 *` factor), so it is reused
#      as-is wherever a symbol's BOTH legs sit on Binance.
#   2. The BYBIT-sourced leg (new in this wave) gets its OWN fitted mapping, built by this
#      module from a live Bybit order-book snapshot (research/wave14_multivenue/fetch_venues.py
#      collects it into cache/bybit_spread_spot.json / cache/bybit_spread_linear.json,
#      following collect_spreads.py's exact half-spread/$45-book-walk formula -- see that
#      module's compute_walk_cost_bp, imported not reimplemented). Bybit SPOT and Bybit
#      LINEAR (USDT perpetual) are fit as TWO INDEPENDENT mappings, not one -- SPOT and
#      LINEAR are different order books with different depth/spread profiles on Bybit, and
#      (unlike wave13's Bitget-only world, where a single 0.02% maker rate happened to cover
#      both legs) Bybit's own published fee schedule is NOT symmetric across market types:
#      spot maker 0.10% vs USDT-perpetual (linear) maker 0.02%
#      (https://www.bybit.com/en/announcement-info/fee-rate/ and
#      https://www.bybit.com/en/help-center/article/Trading-Fee-Structure , cross-checked
#      2026-07 -- both regular/non-VIP, non-BB-token-discount tier). Charging the LINEAR
#      leg's rate on the SPOT leg (or vice versa) would misprice one side by 5x, so the two
#      markets get their own fee constant AND their own fitted bp mapping (fit off their own
#      measured order books, not shared).
#
# The isotonic bucket-median fit ITSELF (research.wave13_liquidity.costs_measured.fit_mapping
# / MeasuredCostMapping) is reused verbatim via its own `payload` override parameter -- this
# module builds a Bybit-shaped {"measurements": [...], "collected_at_utc": ...} payload (same
# two keys fit_mapping's bucketing actually reads: usdt_volume_24h, effective_slippage_bp) and
# lets wave13's own PAVA/bucket-median code fit it, rather than re-deriving that algorithm a
# third time (wave13's own module docstring already explains why PAVA-not-scipy: dependency-
# freedom).

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Final, Literal

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

import pandas as pd  # noqa: PANDAS_OK

from research.wave13_liquidity.costs_measured import (
    ROLLING_WINDOW_DAYS,
    MeasuredCostMapping,
    bp_frame_from_known_avg,
    build_data_availability_mask,
    cost_rate_from_bp as binance_leg_pair_cost_rate,  # re-exported: Binance-sourced legs use wave13's cost model unchanged
    fit_mapping,
    point_in_time_known_avg,
)

BybitMarket = Literal["spot", "linear"]

# Published, non-VIP regular-tier maker fees, confirmed 2026-07 (see module docstring for
# sources). Bitget's own 0.02% (research.wave2.funding.W2_MAKER_FEE_RATE, imported inside
# binance_leg_pair_cost_rate above) continues to stand in for the Binance-sourced leg,
# unchanged from wave13.
BYBIT_SPOT_MAKER_FEE_RATE: Final = 0.0010  # 0.10%, Bybit spot regular-tier maker
BYBIT_LINEAR_MAKER_FEE_RATE: Final = 0.0002  # 0.02%, Bybit USDT-perpetual regular-tier maker


@dataclass(frozen=True, slots=True)
class BybitCostMappings:
    spot: MeasuredCostMapping
    linear: MeasuredCostMapping


def fit_bybit_mappings(spread_payload: dict[str, Any], n_buckets: int = 9) -> BybitCostMappings:
    """Fits two independent MeasuredCostMapping objects off
    cache/bybit_spreads.json's own {"spot": {...measurements...}, "linear": {...}} split
    (research/wave14_multivenue/fetch_venues.py writes this shape; see that module for the
    collection methodology, which mirrors research.wave13_liquidity.collect_spreads exactly,
    just pointed at Bybit's REST endpoints instead of Bitget's)."""
    spot_payload = {"measurements": spread_payload["spot"]["measurements"], "collected_at_utc": spread_payload.get("collected_at_utc", "")}
    linear_payload = {"measurements": spread_payload["linear"]["measurements"], "collected_at_utc": spread_payload.get("collected_at_utc", "")}
    return BybitCostMappings(
        spot=fit_mapping(spot_payload, n_buckets=n_buckets),
        linear=fit_mapping(linear_payload, n_buckets=n_buckets),
    )


def bybit_pair_cost_rate(spot_bp: float | pd.DataFrame, linear_bp: float | pd.DataFrame, stress_multiplier: float = 1.0) -> float | pd.DataFrame:
    """One-way cost rate for a Bybit-sourced same-venue spot+perp pair (M1/M3/M4/M5's
    Bybit-routed slots): Bybit spot maker + Bybit-measured spot slippage on the spot leg,
    PLUS Bybit linear maker + Bybit-measured linear slippage on the perp leg. Deliberately
    NOT symmetric-times-2 the way research.wave13_liquidity.costs_measured.cost_rate_from_bp
    is (that function's `2.0 *` is valid only because Bitget's own maker fee happens to be
    identical on both legs it stands in for) -- here the two legs have genuinely different
    fee rates, so they are summed individually. `stress_multiplier` scales slippage only on
    both legs (maker fees stress-invariant), matching wave13's own S5 convention."""
    return (BYBIT_SPOT_MAKER_FEE_RATE + (spot_bp * 0.0001) * stress_multiplier) + (BYBIT_LINEAR_MAKER_FEE_RATE + (linear_bp * 0.0001) * stress_multiplier)


def cross_venue_leg_cost_rate(binance_linear_bp: float | pd.DataFrame, bybit_linear_bp: float | pd.DataFrame, stress_multiplier: float = 1.0) -> float | pd.DataFrame:
    """One-way cost rate for M6/M7's cross-venue funding-spread structure: one perp leg on
    Binance (Bitget-measured proxy + Bitget 0.02% maker, wave13's convention) and one perp
    leg on Bybit (Bybit-measured linear mapping + Bybit 0.02% linear maker) -- no spot leg
    on either side (SPEC.md: "현물 불요, 양쪽 퍼프"). Both exchanges' linear maker fee
    happens to be the same published 0.02% (Bitget and Bybit alike), but that is coincidence,
    not an assumption this function relies on -- each fee constant is looked up from its own
    venue independently."""
    from research.wave2.funding import W2_MAKER_FEE_RATE  # Bitget maker, Binance-leg proxy (wave13 convention)

    binance_leg = W2_MAKER_FEE_RATE + (binance_linear_bp * 0.0001) * stress_multiplier
    bybit_leg = BYBIT_LINEAR_MAKER_FEE_RATE + (bybit_linear_bp * 0.0001) * stress_multiplier
    return binance_leg + bybit_leg


def build_bp_frame_for_market(quote_volume_frame: pd.DataFrame, symbols: tuple[str, ...], mapping: MeasuredCostMapping, window: int = ROLLING_WINDOW_DAYS) -> pd.DataFrame:
    """Point-in-time (shift(1), no lookahead -- same contract as
    research.wave13_liquidity.costs_measured.build_cost_rate_frame) bp lookup for ONE Bybit
    market (spot or linear), reindexed to `symbols`. Exposed separately (rather than folded
    into a single combined-cost-rate builder) because M1/M3/M4/M5 need spot_bp and linear_bp
    as two separate frames to feed bybit_pair_cost_rate, while M6/M7 only ever need the
    linear one."""
    known_avg = point_in_time_known_avg(quote_volume_frame, window).reindex(columns=list(symbols))
    return bp_frame_from_known_avg(known_avg, mapping)


def build_liquidity_mask_for_market(quote_volume_frame: pd.DataFrame, symbols: tuple[str, ...], window: int = ROLLING_WINDOW_DAYS) -> pd.DataFrame:
    """Pure data-availability mask (True iff a full trailing `window`-day average is known
    yet) for one venue's own volume frame -- identical semantics to
    research.wave13_liquidity.costs_measured.build_data_availability_mask (imported, not
    reimplemented), just documented locally so engine14.py's call sites read as "this
    venue's own availability" rather than an opaque cross-wave import."""
    return build_data_availability_mask(quote_volume_frame, symbols, window)


__all__ = [
    "BYBIT_LINEAR_MAKER_FEE_RATE",
    "BYBIT_SPOT_MAKER_FEE_RATE",
    "BybitCostMappings",
    "BybitMarket",
    "binance_leg_pair_cost_rate",
    "build_bp_frame_for_market",
    "build_liquidity_mask_for_market",
    "bybit_pair_cost_rate",
    "cross_venue_leg_cost_rate",
    "fit_bybit_mappings",
]
