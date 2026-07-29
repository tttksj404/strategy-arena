# Wave-25 config (frozen 2026-07-29, research/wave25_gamble/SPEC.md). Eight short-horizon
# gambling candidates (B0-B7) sharing ONE capital contract: $100 total = $25 gambling sleeve
# (this wave's own engine25.py) + $75 parked in I5 (research/wave18_idle, results/I5.json read
# verbatim -- NOT re-simulated here, matching research.wave20_convex.engine20.load_stable_leg
# exactly, which this module reuses).
#
# SPEC.md's frozen prose table (B0-B7, lines 14-24) pins the SIGNAL family and its headline
# parameters (e.g. "MACD(12,26,9)", "슈퍼트렌드(10, 3.0)"); it does not spell out every last
# implementation constant (e.g. the MTF breakout's own lookback, the stochastic trend-filter's
# MA window). Per this repo's own established convention (research/wave20_convex/configs20.py's
# own module docstring: "이 표를 코드로 바꾸는 유일한 곳"), THIS module is that single place --
# every numeric constant below is fixed here, BEFORE research/wave25_gamble/run_wave25.py's
# `run` stage is ever executed, and is never adjusted after seeing results/*.json (SPEC.md's own
# freeze condition: "동결, 사후 추가·파라미터 조정 금지"). Where SPEC.md's prose leaves a
# constant unstated, the choice made here is a plain, standard textbook value (e.g. Stochastic
# 20/80 overbought/oversold, a 5-bar/5-day slope lookback) -- never a value searched for or
# tuned against this wave's own results.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Final

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

from research.wave10_carry100.engine import OOS_SPLIT

REPO_ROOT: Final = Path(__file__).resolve().parents[2]
BASE_DIR: Final = Path(__file__).resolve().parent
RESULTS_DIR: Final = BASE_DIR / "results"
REPORT_DIR: Final = BASE_DIR / "report"
REGISTRY_PATH: Final = BASE_DIR / "REGISTRY.md"
LIVE_CACHE_DIR: Final = BASE_DIR / "cache"  # this wave's OWN incremental live-fetch cache; never writes into wave1/wave6's own cache dirs

# Read-only source caches (never written to).
WAVE1_CACHE_DIR: Final = REPO_ROOT / "research" / "wave1" / "cache"  # BTC/ETH/SOL 1D (binance_fapi_*_1d.csv.gz)
WAVE6_CACHE_DIR: Final = REPO_ROOT / "research" / "wave6" / "cache"  # BTC/ETH/SOL 1H
I5_RESULTS_PATH: Final = REPO_ROOT / "research" / "wave18_idle" / "results" / "I5.json"

# ---------------------------------------------------------------------------
# Capital contract (SPEC.md "볼록 구조 강제" -- common to every candidate).
# ---------------------------------------------------------------------------
TOTAL_CAPITAL: Final = 100.0
GAMBLE_CAPITAL: Final = 25.0
STABLE_CAPITAL: Final = TOTAL_CAPITAL - GAMBLE_CAPITAL
IS_OOS_SPLIT: Final = OOS_SPLIT
MAKER_FEE_RATE: Final = 0.0002  # reused unmodified from research.wave20_convex.configs20 / wave13_liquidity.costs_measured's own W2_MAKER_FEE_RATE

SYMBOLS: Final[tuple[str, ...]] = ("BTCUSDT", "ETHUSDT", "SOLUSDT")  # priority scan order (SPEC.md "대상: BTC/ETH/SOL 퍼프")

# ---------------------------------------------------------------------------
# Multi-testing disclosure (SPEC.md "다중검정: 누적 129후보 + GA/GP 82,621평가 반영해 DSR 표기").
# ---------------------------------------------------------------------------
DSR_CUMULATIVE_TRIALS: Final = 129
GA_GP_EVALUATIONS_DISCLOSED: Final = 82_621  # disclosed separately in the report text; NOT fed into DSR's own `trials` (that parameter means independent candidate strategies, not raw GA/GP fitness evaluations within a single search)

# ---------------------------------------------------------------------------
# Shared convex position-lifecycle constants (SPEC.md "볼록 구조 강제", applies to B1-B7; B0
# reuses V1's own native lifecycle verbatim -- see engine25.run_b0). Hourly bars throughout
# (B1-B4/B6/B7 trade 1H; B5 additionally consults 1D for its trend filter).
# ---------------------------------------------------------------------------
RISK_ATR_WINDOW: Final = 14  # uniform ATR(14) used for stop/trailing sizing, independent of any signal's OWN internal ATR window (e.g. Supertrend's own ATR(10))
HARD_STOP_PCT: Final = 0.03  # "-3%"
HARD_STOP_ATR_MULT: Final = 1.0  # "-1xATR" -- stop distance = min(HARD_STOP_PCT*entry_price, HARD_STOP_ATR_MULT*ATR_at_entry), i.e. "중 가까운 쪽"
TRAILING_ACTIVATE_ATR_MULT: Final = 1.0  # trailing arms once favorable move >= 1xATR (then seeds at breakeven -- see engine25 module docstring)
TRAILING_ATR_MULT: Final = 1.5  # once armed, trails 1.5xATR behind the running favorable extreme
MAX_HOLD_DAYS: Final = 14  # "보유 상한 14일"
MAX_HOLD_BARS_1H: Final = MAX_HOLD_DAYS * 24

# ---------------------------------------------------------------------------
# Gate thresholds (SPEC.md "게이트" section, P1-P5).
# ---------------------------------------------------------------------------
P1_MIN_TRADES: Final = 10  # below this, skew/decile-contribution is UNDETERMINED (matches wave20 G3_MIN_TRADES convention)
P1_TOP_DECILE_FRACTION: Final = 0.10
P1_TOP_DECILE_CONTRIBUTION_MIN: Final = 0.50
P1_BOOTSTRAP_PATHS: Final = 5_000
P2_MC_PATHS: Final = 10_000
P2_RUIN_FLOOR_USDT: Final = 50.0
P2_RUIN_PROBABILITY_MAX: Final = 0.10
P2_MAX_SINGLE_TRADE_LOSS_USDT: Final = GAMBLE_CAPITAL  # "단일 최대손실 <= $25" -- an absolute dollar cap on any ONE realized trade, distinct from the MC ruin probability (portfolio-path measure) and from the engine's own equity-floor-at-$0 (which only prevents going NEGATIVE, not a >$25 loss on a since-grown sleeve)
P4_ROLLING_WINDOW_DAYS: Final = 30
P4_TOP_QUARTILE_FRACTION: Final = 0.25
P5_MIN_LEG_USDT: Final = 5.0
P5_STRESS_MULTIPLIER: Final = 3.0  # "실측슬리피지 x3" -- matches wave13_liquidity's own S5 / wave20's stress_multiplier convention exactly

# ---------------------------------------------------------------------------
# B0 -- V1 reproduction (wave20_convex, BTC perp only). No new dataclass: engine25.run_b0
# calls research.wave20_convex.engine20.run_v1(V1_CONFIG) directly and repackages the result
# under wave25's own schema -- see SPEC.md "B0 = V1 재현 ... 이걸 못 이기면 신규 무의미".
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class B1Config:
    candidate_id: str
    fast: int
    slow: int
    signal: int


B1_CONFIG: Final = B1Config(candidate_id="B1", fast=12, slow=26, signal=9)


@dataclass(frozen=True, slots=True)
class B2Config:
    candidate_id: str
    window: int
    adx_threshold: float


B2_CONFIG: Final = B2Config(candidate_id="B2", window=14, adx_threshold=25.0)


@dataclass(frozen=True, slots=True)
class B3Config:
    candidate_id: str
    window: int
    multiplier: float


B3_CONFIG: Final = B3Config(candidate_id="B3", window=10, multiplier=3.0)


@dataclass(frozen=True, slots=True)
class B4Config:
    candidate_id: str
    window: int
    atr_window: int
    multiplier: float


B4_CONFIG: Final = B4Config(candidate_id="B4", window=20, atr_window=20, multiplier=2.0)


@dataclass(frozen=True, slots=True)
class B5Config:
    candidate_id: str
    daily_ma_window: int  # "1D 추세방향(MA50 기울기)"
    daily_slope_lookback: int  # trailing days the MA50 slope is measured over
    breakout_lookback_bars: int  # "1H 돌파" -- fixed-lag momentum window (hours), NOT a Donchian N-bar-extreme channel (that family is already dead per SPEC.md's own 테스트 완료 list)
    breakout_atr_window: int
    breakout_atr_multiplier: float


B5_CONFIG: Final = B5Config(
    candidate_id="B5",
    daily_ma_window=50,
    daily_slope_lookback=5,
    breakout_lookback_bars=24,
    breakout_atr_window=14,
    breakout_atr_multiplier=1.5,
)


@dataclass(frozen=True, slots=True)
class B6Config:
    candidate_id: str
    k_window: int
    d_window: int
    oversold: float
    overbought: float
    trend_ma_window: int
    trend_slope_lookback: int


B6_CONFIG: Final = B6Config(
    candidate_id="B6",
    k_window=14,
    d_window=3,
    oversold=20.0,
    overbought=80.0,
    trend_ma_window=50,
    trend_slope_lookback=5,
)


@dataclass(frozen=True, slots=True)
class B7Config:
    candidate_id: str
    min_agree: int  # "동시 3개 이상 동일 방향 발화"


B7_CONFIG: Final = B7Config(candidate_id="B7", min_agree=3)

CANDIDATE_IDS: Final[tuple[str, ...]] = ("B0", "B1", "B2", "B3", "B4", "B5", "B6", "B7")
NEW_CANDIDATE_IDS: Final[tuple[str, ...]] = ("B1", "B2", "B3", "B4", "B5", "B6", "B7")  # excludes B0 (the baseline itself)
ENSEMBLE_MEMBER_IDS: Final[tuple[str, ...]] = ("B1", "B2", "B3", "B4", "B5", "B6")  # B7's own voting pool

__all__ = [
    "B1Config",
    "B1_CONFIG",
    "B2Config",
    "B2_CONFIG",
    "B3Config",
    "B3_CONFIG",
    "B4Config",
    "B4_CONFIG",
    "B5Config",
    "B5_CONFIG",
    "B6Config",
    "B6_CONFIG",
    "B7Config",
    "B7_CONFIG",
    "CANDIDATE_IDS",
    "DSR_CUMULATIVE_TRIALS",
    "ENSEMBLE_MEMBER_IDS",
    "GA_GP_EVALUATIONS_DISCLOSED",
    "GAMBLE_CAPITAL",
    "HARD_STOP_ATR_MULT",
    "HARD_STOP_PCT",
    "I5_RESULTS_PATH",
    "IS_OOS_SPLIT",
    "LIVE_CACHE_DIR",
    "MAKER_FEE_RATE",
    "MAX_HOLD_BARS_1H",
    "MAX_HOLD_DAYS",
    "NEW_CANDIDATE_IDS",
    "P1_BOOTSTRAP_PATHS",
    "P1_MIN_TRADES",
    "P1_TOP_DECILE_CONTRIBUTION_MIN",
    "P1_TOP_DECILE_FRACTION",
    "P2_MAX_SINGLE_TRADE_LOSS_USDT",
    "P2_MC_PATHS",
    "P2_RUIN_FLOOR_USDT",
    "P2_RUIN_PROBABILITY_MAX",
    "P4_ROLLING_WINDOW_DAYS",
    "P4_TOP_QUARTILE_FRACTION",
    "P5_MIN_LEG_USDT",
    "P5_STRESS_MULTIPLIER",
    "REGISTRY_PATH",
    "REPORT_DIR",
    "RESULTS_DIR",
    "RISK_ATR_WINDOW",
    "STABLE_CAPITAL",
    "SYMBOLS",
    "TOTAL_CAPITAL",
    "TRAILING_ACTIVATE_ATR_MULT",
    "TRAILING_ATR_MULT",
    "WAVE1_CACHE_DIR",
    "WAVE6_CACHE_DIR",
]
