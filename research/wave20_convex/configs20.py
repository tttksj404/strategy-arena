# Wave-20 config (frozen 2026-07-23, research/wave20_convex/SPEC.md). Five convex/asymmetric
# gambling candidates (V1-V5) sharing ONE capital contract: $100 total = $25 gambling sleeve
# (this wave's own new engine) + $75 parked in I5 (research/wave18_idle, CAGR 10.27%,
# results/I5.json read verbatim -- NOT re-simulated here, see engine20.load_stable_leg).
#
# Every numeric threshold below is copied 1:1 from SPEC.md's frozen table (lines 12-27) --
# this module is the single place that turns that prose table into code so engine20.py/
# gates20.py never re-litigate a threshold. Per SPEC.md's own promotion rule ("사후 추가·파라미터
# 조정 금지"), nothing here may be tuned after seeing results/*.json.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Final

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

import pandas as pd  # noqa: PANDAS_OK

from research.wave10_carry100.engine import OOS_SPLIT

REPO_ROOT: Final = Path(__file__).resolve().parents[2]
BASE_DIR: Final = Path(__file__).resolve().parent
RESULTS_DIR: Final = BASE_DIR / "results"
REPORT_DIR: Final = BASE_DIR / "report"
REGISTRY_PATH: Final = BASE_DIR / "REGISTRY.md"

# Read-only source caches (SPEC.md "캐시" column / 배경 문단) -- never written to.
WAVE1_CACHE_DIR: Final = REPO_ROOT / "research" / "wave1" / "cache"  # 40+종 일봉+펀딩 (V2)
WAVE3_CACHE_DIR: Final = REPO_ROOT / "research" / "wave3" / "cache"  # 332종 일봉 (V3, V5)
WAVE6_CACHE_DIR: Final = REPO_ROOT / "research" / "wave6" / "cache"  # BTC/ETH/SOL 1H (V1, V4)
I5_RESULTS_PATH: Final = REPO_ROOT / "research" / "wave18_idle" / "results" / "I5.json"

# ---------------------------------------------------------------------------
# Capital contract (SPEC.md "공통 규약").
# ---------------------------------------------------------------------------
TOTAL_CAPITAL: Final = 100.0
GAMBLE_CAPITAL: Final = 25.0  # "도박 배분 최대 25%($25)"
STABLE_CAPITAL: Final = TOTAL_CAPITAL - GAMBLE_CAPITAL  # $75 -> I5
IS_OOS_SPLIT: Final = OOS_SPLIT  # 2025-09-30T23:59:59Z, same constant every prior wave uses
WORST_YEARS: Final[tuple[int, ...]] = (2022, 2025)  # G5 "2022*2025에서 ... 악화 없음"

# Maker fee, reused unmodified from wave2 (research/wave13_liquidity/costs_measured.py's own
# cost_rate_from_bp imports the same constant) -- restated here as a plain float so
# engine20.py's single-leg cost function doesn't need to reach into costs_measured's 2-leg
# (spot+perp carry pair) convention. See engine20.one_leg_cost_rate's docstring for why V1-V5
# use a SINGLE-leg cost (one instrument, directional), not costs_measured.cost_rate_from_bp's
# 2x (that function prices a simultaneous spot+perp carry PAIR, which nothing in this wave is).
MAKER_FEE_RATE: Final = 0.0002

# ---------------------------------------------------------------------------
# Multi-testing disclosure (SPEC.md "다중검정: 누적 121회 DSR 보정").
# ---------------------------------------------------------------------------
DSR_CUMULATIVE_TRIALS: Final = 121

# ---------------------------------------------------------------------------
# Gate thresholds (SPEC.md "게이트" section, G1-G5).
# ---------------------------------------------------------------------------
G1_MAX_LOSS_FRACTION: Final = 1.0  # "최대손실 <= 배분액($25)" -- sleeve equity floor is $0, never negative
G2_MC_PATHS: Final = 10_000
G2_RUIN_FLOOR_USDT: Final = 50.0  # "P(최종<$50)"
G2_RUIN_PROBABILITY_MAX: Final = 0.10  # "< 10%"
G3_MIN_TRADES: Final = 10  # below this, skew/decile-contribution is UNDETERMINED (unreliable with a tiny n)
G3_TOP_DECILE_FRACTION: Final = 0.10  # "상위 10% 거래"
G3_TOP_DECILE_CONTRIBUTION_MIN: Final = 0.50  # "총수익의 50% 이상 기여"
G3_BOOTSTRAP_PATHS: Final = 5_000  # trade-resample count for the skew/one-lucky-trade diagnostic
G4_STABLE_CAGR_REFERENCE_NOTE: Final = "research/wave18_idle/results/I5.json:full_period_annualized (~10.27%, read live, not hardcoded)"

# ---------------------------------------------------------------------------
# V1 -- 양방향 돌파 추격 (long-vol straddle approximation), BTC perp, 1D regime + 1H execution.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class V1Config:
    candidate_id: str
    symbol: str
    atr_window_days: int  # DAILY ATR -- see engine20.run_v1's docstring for why the ±2xATR
    # band uses daily (not hourly) ATR: with a trailing chandelier-style anchor (required for
    # G1 -- see simulate_breakout_reversal's own comment), an hourly ATR is tight enough that
    # ordinary intraday noise triggers a reversal almost every day, which never lets a real
    # trend run and just harvests costs. 1H bars still drive execution TIMING/whipsaw-cost
    # granularity; only the threshold's own magnitude is daily-scale.
    atr_multiplier: float  # "±2×ATR"
    vol_window_days: int  # "realized vol 20d"
    vol_percentile_lookback_days: int  # trailing window the 20d-vol percentile is ranked against
    vol_percentile_threshold: float  # "< 30분위"


V1_CONFIG: Final = V1Config(
    candidate_id="V1",
    symbol="BTCUSDT",
    atr_window_days=14,
    atr_multiplier=2.0,
    vol_window_days=20,
    vol_percentile_lookback_days=365,
    vol_percentile_threshold=0.30,
)

# ---------------------------------------------------------------------------
# V2 -- 꼬리 사냥 (funding-extreme directional long, NOT carry), wave1 40+종 펀딩+가격.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class V2Config:
    candidate_id: str
    funding_window_days: int  # short window -- V2 targets an ACUTE squeeze, not a persistent carry regime
    entry_threshold_apr: float  # "펀딩 > 연 100%"
    exit_threshold_apr: float  # carry_position's own entry/2 hysteresis convention
    top_k: int
    excluded_symbols: tuple[str, ...]  # USD/gold-pegged names for which an "extreme funding squeeze" read is meaningless


V2_CONFIG: Final = V2Config(
    candidate_id="V2",
    funding_window_days=3,
    entry_threshold_apr=1.00,
    exit_threshold_apr=0.50,
    top_k=1,
    excluded_symbols=("USDCUSDT", "PAXGUSDT", "XAUTUSDT"),
)

# ---------------------------------------------------------------------------
# V3 -- 신규상장 첫 7일, wave3 332종 (첫 캔들 = 상장일 대체, Bitget launchTime 공란 -- 배경 참조).
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class V3Config:
    candidate_id: str
    hold_days: int  # "7일 후 청산"
    atr_multiplier: float  # same ±2xATR reversal rule as V1, applied to an expanding intra-window ATR (see engine20 docstring)
    min_listing_gap_days: int  # fetch-window-artifact guard: drop symbols whose first candle sits within this many days of the cache's own global floor
    min_rows_required: int  # a listing needs >= this many daily rows on file to even attempt a hold_days-long trade


V3_CONFIG: Final = V3Config(
    candidate_id="V3",
    hold_days=7,
    atr_multiplier=2.0,
    min_listing_gap_days=14,
    min_rows_required=8,
)

# ---------------------------------------------------------------------------
# V4 -- 청산 캐스케이드 반등 (SYMMETRIC control group), BTC/ETH/SOL 1H.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class V4Config:
    candidate_id: str
    symbols: tuple[str, ...]
    drop_threshold: float  # "-8% 이상 급락"
    take_profit: float  # "+3%"
    stop_loss: float  # "-3%"
    max_hold_bars: int  # "24h 최대보유" (1H bars)


V4_CONFIG: Final = V4Config(
    candidate_id="V4",
    symbols=("BTCUSDT", "ETHUSDT", "SOLUSDT"),
    drop_threshold=-0.08,
    take_profit=0.03,
    stop_loss=-0.03,
    max_hold_bars=24,
)

# ---------------------------------------------------------------------------
# V5 -- 복권 바스켓, wave3 332종, 저가·고변동 5종 point-in-time 매월 재선정.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class V5Config:
    candidate_id: str
    rebalance_days: int  # "30일 보유"
    basket_size: int  # "5종"
    vol_lookback_days: int  # trailing realized-vol window used to rank "고변동"
    min_history_days: int  # a symbol needs this much trailing history before it is even eligible
    cheap_price_percentile: float  # "저가" -- bottom-percentile price filter, applied before the vol rank
    excluded_symbols: tuple[str, ...]  # majors/stable/commodity-pegged names excluded from the lottery universe


V5_CONFIG: Final = V5Config(
    candidate_id="V5",
    rebalance_days=30,
    basket_size=5,
    vol_lookback_days=30,
    min_history_days=90,
    cheap_price_percentile=0.30,
    excluded_symbols=("BTCUSDT", "ETHUSDT", "SOLUSDT", "USDCUSDT", "USDPUSDT", "TUSDUSDT", "FDUSDUSDT", "DAIUSDT", "PAXGUSDT", "XAUTUSDT", "BNBUSDT"),
)

CANDIDATE_IDS: Final[tuple[str, ...]] = ("V1", "V2", "V3", "V4", "V5")

__all__ = [
    "CANDIDATE_IDS",
    "DSR_CUMULATIVE_TRIALS",
    "G1_MAX_LOSS_FRACTION",
    "G2_MC_PATHS",
    "G2_RUIN_FLOOR_USDT",
    "G2_RUIN_PROBABILITY_MAX",
    "G3_BOOTSTRAP_PATHS",
    "G3_MIN_TRADES",
    "G3_TOP_DECILE_CONTRIBUTION_MIN",
    "G3_TOP_DECILE_FRACTION",
    "G4_STABLE_CAGR_REFERENCE_NOTE",
    "GAMBLE_CAPITAL",
    "I5_RESULTS_PATH",
    "IS_OOS_SPLIT",
    "MAKER_FEE_RATE",
    "REGISTRY_PATH",
    "REPORT_DIR",
    "RESULTS_DIR",
    "STABLE_CAPITAL",
    "TOTAL_CAPITAL",
    "V1_CONFIG",
    "V2_CONFIG",
    "V3_CONFIG",
    "V4_CONFIG",
    "V5_CONFIG",
    "WAVE1_CACHE_DIR",
    "WAVE3_CACHE_DIR",
    "WAVE6_CACHE_DIR",
    "WORST_YEARS",
    "V1Config",
    "V2Config",
    "V3Config",
    "V4Config",
    "V5Config",
]
