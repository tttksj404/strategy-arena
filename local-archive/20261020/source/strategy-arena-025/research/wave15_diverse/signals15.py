# Wave-15 entry/exit signal construction for the DAILY-bar candidates (B1, B2, C1).
# A1-A3's settlement-timed signal lives in engine_intraday.py (it operates on hourly bars
# and a state machine, not this module's daily active/score-frame shape); D1's pair z-score
# + direction freezing lives in engine_pairs.py.

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

from research.wave1.fam_funding import FundingCandidate, FundingMarket, carry_position, funding_score
from research.wave15_diverse.common15 import ENTRY_THRESHOLD_APR, EXIT_THRESHOLD_APR

# ---------------------------------------------------------------------------
# B1/B2: the plain realized-funding threshold signal, UNCHANGED from wave10-13 (window=7d,
# entry 15% APR, exit 7.5% APR, top_k=1) -- B1/B2 change WHAT the return stream is (dual
# yield / directional), never WHEN the position opens. Reuses research.wave1.fam_funding's
# funding_score/carry_position directly rather than reimplementing (SPEC.md's own candidate
# text for B1/B2 only changes the return leg, not the entry rule).
# ---------------------------------------------------------------------------

BASELINE_WINDOW_DAYS: Final = 7
BASELINE_CANDIDATE: Final = FundingCandidate("wave15_baseline_signal", BASELINE_WINDOW_DAYS, ENTRY_THRESHOLD_APR, 1)


def build_realized_funding_signal(markets: dict[str, FundingMarket], index: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame]:
    active: dict[str, pd.Series] = {}
    score: dict[str, pd.Series] = {}
    for symbol, market in markets.items():
        funding_apr = funding_score(market.funding, BASELINE_WINDOW_DAYS).resample("1D").last()
        score[symbol] = funding_apr
        active[symbol] = carry_position(funding_apr, BASELINE_CANDIDATE)
    score_frame = pd.DataFrame(score).reindex(index).shift(1)
    active_frame = pd.DataFrame(active).reindex(index).fillna(0.0)
    return active_frame, score_frame


# ---------------------------------------------------------------------------
# C1: predictive fixed-coefficient composite (SPEC.md: "3개 특징 고정, 계수 사전지정 -- 학습
# 금지"; OI feature dropped -- confirmed live that Binance's /futures/data/openInterestHist
# retains only ~30 days, nowhere near enough for a multi-year backtest spanning the 2020/
# 2021/2024 high-funding years -- so this is the 2-feature fallback SPEC.md pre-approved).
#
# composite = 0.5 * z(7d price momentum) + 0.5 * z(7d funding-score trend), EQUAL weights,
# fixed BEFORE any candidate was run (never adjusted after looking at results -- that would
# be exactly the "학습" this candidate exists to avoid). The z-scoring itself is a CAUSAL
# rolling normalization (90d trailing mean/std, shift-safe), not a fit: it rescales each raw
# feature to comparable units so a 0.5/0.5 sum is meaningful across symbols of very different
# price/funding scale -- it introduces no free parameter that responds to backtest outcomes
# (window/weights/entry bar are all constants declared here, once).
# ---------------------------------------------------------------------------

MOMENTUM_WINDOW_DAYS: Final = 7  # same cadence as the realized 7d APR score this signal is compared/exited against
ZSCORE_WINDOW_DAYS: Final = 90
C1_WEIGHT_MOMENTUM: Final = 0.5
C1_WEIGHT_FUNDING_TREND: Final = 0.5
C1_ENTRY_Z: Final = 1.0  # composite >= 1 std dev above its own trailing norm -- fixed a priori, never tuned
C1_EXIT_APR: Final = EXIT_THRESHOLD_APR  # SPEC.md literal: "7d APR<7.5% 청산"


def _price_momentum(perp_close_daily: pd.Series, window_days: int = MOMENTUM_WINDOW_DAYS) -> pd.Series:
    return perp_close_daily / perp_close_daily.shift(window_days) - 1.0


def _funding_trend(funding_native: pd.Series, window_days: int = MOMENTUM_WINDOW_DAYS) -> pd.Series:
    score = funding_score(funding_native, window_days).resample("1D").last()
    return score - score.shift(window_days)


def _rolling_zscore(raw: pd.Series, window_days: int = ZSCORE_WINDOW_DAYS) -> pd.Series:
    mean = raw.rolling(window_days, min_periods=window_days).mean()
    std = raw.rolling(window_days, min_periods=window_days).std()
    z = (raw - mean) / std
    return z.replace([np.inf, -np.inf], np.nan)


def composite_predictive_score(market: FundingMarket) -> tuple[pd.Series, pd.Series]:
    """Returns (composite_z, realized_apr_7d) both at native daily resolution, NOT yet
    shifted (the outer assembly below applies the point-in-time shift, matching
    research.wave1.fam_funding's own score_frame/active_frame convention)."""
    perp_close_daily = market.perp.resample("1D").agg({"close": "last"})["close"].dropna()
    momentum_z = _rolling_zscore(_price_momentum(perp_close_daily))
    trend_z = _rolling_zscore(_funding_trend(market.funding))
    realized_apr = funding_score(market.funding, BASELINE_WINDOW_DAYS).resample("1D").last()
    composite = C1_WEIGHT_MOMENTUM * momentum_z + C1_WEIGHT_FUNDING_TREND * trend_z
    return composite.reindex(realized_apr.index), realized_apr


def predictive_carry_position(composite_z: pd.Series, realized_apr: pd.Series, entry_z: float = C1_ENTRY_Z, exit_apr: float = C1_EXIT_APR) -> pd.Series:
    """Same hysteresis SHAPE as research.wave1.fam_funding.carry_position (unconditional
    activate-above-bar / deactivate-below-bar scan, shift(1)+fillna(0.0) at the end) but with
    two DIFFERENT input series for the two edges: entry reads the PREDICTIVE composite
    (SPEC.md "선행 진입"), exit reads the REALIZED 7d APR (SPEC.md "7d APR<7.5% 청산") --
    deliberately asymmetric, not a bug."""
    aligned = pd.concat([composite_z.rename("z"), realized_apr.rename("apr")], axis=1)
    values: list[float] = []
    active = 0.0
    for z_value, apr_value in zip(aligned["z"], aligned["apr"]):
        if pd.notna(z_value) and z_value > entry_z:
            active = 1.0
        elif pd.notna(apr_value) and apr_value < exit_apr:
            active = 0.0
        values.append(active)
    return pd.Series(values, index=aligned.index, dtype=float).shift(1).fillna(0.0)


def build_predictive_signal_frames(markets: dict[str, FundingMarket], index: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame]:
    active: dict[str, pd.Series] = {}
    score: dict[str, pd.Series] = {}
    for symbol, market in markets.items():
        composite_z, realized_apr = composite_predictive_score(market)
        active[symbol] = predictive_carry_position(composite_z, realized_apr)
        score[symbol] = composite_z
    active_frame = pd.DataFrame(active).reindex(index).fillna(0.0)
    score_frame = pd.DataFrame(score).reindex(index).shift(1)
    return active_frame, score_frame


__all__ = [
    "BASELINE_CANDIDATE",
    "BASELINE_WINDOW_DAYS",
    "C1_ENTRY_Z",
    "C1_EXIT_APR",
    "C1_WEIGHT_FUNDING_TREND",
    "C1_WEIGHT_MOMENTUM",
    "MOMENTUM_WINDOW_DAYS",
    "ZSCORE_WINDOW_DAYS",
    "build_predictive_signal_frames",
    "build_realized_funding_signal",
    "composite_predictive_score",
    "predictive_carry_position",
]
