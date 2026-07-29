# Wave-26 config (frozen 2026-07-29, research/wave26_freq/SPEC.md). Eight frequency-control
# candidates (C0-C7) that reuse wave-25's OWN signal logic (indicators25.py, engine25.py)
# UNMODIFIED and add exactly one new thing: a control on WHEN an entry is admitted. SPEC.md's
# own words: "신호를 바꾸는 게 아니라 언제 쓸지만 제한한다" -- this module fixes every numeric
# threshold for that admission control, before research/wave26_freq/run_wave26.py's `run` stage
# is ever executed, and never adjusts them after seeing results/*.json (SPEC.md's own freeze
# condition, same convention as configs25.py/configs20.py before it).
#
# Capital contract, symbols, cost model, convex stop/trailing constants, and P1/P2/P5-style gate
# thresholds are NOT redeclared here -- SPEC.md's own "공통(wave-25 승계, 변경 금지)" clause means
# this module IMPORTS them from research.wave25_gamble.configs25 rather than copying the numbers
# a second time (a second copy could silently drift out of sync with wave25's own frozen values).

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Final

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

from research.wave25_gamble.configs25 import (
    GAMBLE_CAPITAL,
    HARD_STOP_ATR_MULT,
    HARD_STOP_PCT,
    I5_RESULTS_PATH,
    MAKER_FEE_RATE,
    MAX_HOLD_BARS_1H,
    MAX_HOLD_DAYS,
    P1_BOOTSTRAP_PATHS,
    P1_MIN_TRADES,
    P1_TOP_DECILE_CONTRIBUTION_MIN,
    P1_TOP_DECILE_FRACTION,
    P2_MAX_SINGLE_TRADE_LOSS_USDT,
    P2_MC_PATHS,
    P2_RUIN_FLOOR_USDT,
    P2_RUIN_PROBABILITY_MAX,
    P5_MIN_LEG_USDT,
    P5_STRESS_MULTIPLIER,
    RISK_ATR_WINDOW,
    STABLE_CAPITAL,
    SYMBOLS,
    TOTAL_CAPITAL,
    TRAILING_ACTIVATE_ATR_MULT,
    TRAILING_ATR_MULT,
    WAVE1_CACHE_DIR,
    WAVE6_CACHE_DIR,
)

REPO_ROOT: Final = Path(__file__).resolve().parents[2]
BASE_DIR: Final = Path(__file__).resolve().parent
RESULTS_DIR: Final = BASE_DIR / "results"
REPORT_DIR: Final = BASE_DIR / "report"
REGISTRY_PATH: Final = BASE_DIR / "REGISTRY.md"
LIVE_CACHE_DIR: Final = BASE_DIR / "cache"  # this wave's OWN incremental live-fetch cache; never writes into wave1/wave6/wave25's own cache dirs

# Re-exported verbatim from wave25 (SPEC.md "공통 ... 변경 금지") -- listed explicitly (not
# `import *`) so every constant this module actually uses is auditable from this file alone.
__all_reexported__: Final = (
    "GAMBLE_CAPITAL",
    "HARD_STOP_ATR_MULT",
    "HARD_STOP_PCT",
    "I5_RESULTS_PATH",
    "MAKER_FEE_RATE",
    "MAX_HOLD_BARS_1H",
    "MAX_HOLD_DAYS",
    "P1_BOOTSTRAP_PATHS",
    "P1_MIN_TRADES",
    "P1_TOP_DECILE_CONTRIBUTION_MIN",
    "P1_TOP_DECILE_FRACTION",
    "P2_MAX_SINGLE_TRADE_LOSS_USDT",
    "P2_MC_PATHS",
    "P2_RUIN_FLOOR_USDT",
    "P2_RUIN_PROBABILITY_MAX",
    "P5_MIN_LEG_USDT",
    "P5_STRESS_MULTIPLIER",
    "RISK_ATR_WINDOW",
    "STABLE_CAPITAL",
    "SYMBOLS",
    "TOTAL_CAPITAL",
    "TRAILING_ACTIVATE_ATR_MULT",
    "TRAILING_ATR_MULT",
    "WAVE1_CACHE_DIR",
    "WAVE6_CACHE_DIR",
)

# ---------------------------------------------------------------------------
# Multi-testing disclosure (SPEC.md "다중검정: 누적 137후보 반영 DSR 표기" -- wave25's own 129
# cumulative trials + this wave's 8 new candidates (C0-C7) = 137).
# ---------------------------------------------------------------------------
DSR_CUMULATIVE_TRIALS: Final = 137
PRIOR_CUMULATIVE_TRIALS: Final = 129  # wave25's own disclosed count, carried forward for the report's arithmetic to be auditable from this file alone

# ---------------------------------------------------------------------------
# Frequency-control axes (SPEC.md "빈도 통제 3축"). Every candidate's admission rule is built
# from these three knobs -- see engine26.py's module docstring for exactly how each is applied
# (cooldown = dynamic, tracked inside the simulation loop since it depends on WHEN an exit
# actually occurs; ADX/z-score = static per-bar boolean masks, precomputed here-onward).
# ---------------------------------------------------------------------------
ADX_REGIME_WINDOW: Final = 14  # "ADX(14)"
ADX_REGIME_THRESHOLD: Final = 20.0  # "ADX(14) > 20"
Z_SCORE_WINDOW_DAYS: Final = 20  # "신호값 20일 z-score"
Z_SCORE_WINDOW_BARS: Final = Z_SCORE_WINDOW_DAYS * 24  # everything in this engine runs on the hourly calendar (see engine25.run_multi_symbol_convex) -- "20일" = 20*24 hourly bars, the same day->bar convention MAX_HOLD_BARS_1H already uses for "보유 상한 14일"
Z_SCORE_THRESHOLD: Final = 1.0  # "z-score > 1.0"

# ---------------------------------------------------------------------------
# Gate thresholds (SPEC.md "게이트" section, Q1-Q5). Q1/Q2/Q5 reuse P1/P2/P5's own thresholds
# verbatim (imported above, not redeclared). Q3 has no NEW threshold of its own -- it compares
# each candidate directly against C0's own realized final equity (computed at gate-time, not a
# frozen number here). Q4 is the one genuinely new threshold this wave introduces.
# ---------------------------------------------------------------------------
Q4_MAX_COST_FRACTION_OF_SLEEVE: Final = 0.40  # "총비용 <= 슬리브의 40%($10)"
Q4_MAX_COST_USDT: Final = GAMBLE_CAPITAL * Q4_MAX_COST_FRACTION_OF_SLEEVE

# ---------------------------------------------------------------------------
# Candidate definitions (SPEC.md "후보 8개" table, lines 18-27). C0 and C7 are V1-family (reuse
# research.wave20_convex.engine20.simulate_breakout_reversal's own state machine, extended --
# see engine26.simulate_breakout_reversal_controlled); C1-C6 are B-family (reuse
# engine25.run_multi_symbol_convex's own state machine, extended -- see
# engine26.run_multi_symbol_convex_controlled). Base signal families:
#   "V1"         -- ±2xATR breakout/chandelier-reversal (wave20 V1, BTC only, no new stop/trailing)
#   "MACD"       -- engine25.macd_signal (B1's own signal, unmodified)
#   "SUPERTREND" -- engine25.supertrend_signal (B3's own signal, unmodified)
#   "ENSEMBLE"   -- engine25.ensemble_signal_for_symbol (B7's own signal, unmodified)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ControlSpec:
    candidate_id: str
    base_family: str  # "V1" | "MACD" | "SUPERTREND" | "ENSEMBLE"
    cooldown_days: int  # 0 = no cooldown
    adx_gate: bool
    z_gate: bool


C0_SPEC: Final = ControlSpec(candidate_id="C0", base_family="V1", cooldown_days=0, adx_gate=False, z_gate=False)
C1_SPEC: Final = ControlSpec(candidate_id="C1", base_family="MACD", cooldown_days=5, adx_gate=False, z_gate=False)
C2_SPEC: Final = ControlSpec(candidate_id="C2", base_family="MACD", cooldown_days=5, adx_gate=True, z_gate=False)
C3_SPEC: Final = ControlSpec(candidate_id="C3", base_family="MACD", cooldown_days=5, adx_gate=True, z_gate=True)
C4_SPEC: Final = ControlSpec(candidate_id="C4", base_family="SUPERTREND", cooldown_days=5, adx_gate=True, z_gate=False)
C5_SPEC: Final = ControlSpec(candidate_id="C5", base_family="ENSEMBLE", cooldown_days=5, adx_gate=True, z_gate=False)
C6_SPEC: Final = ControlSpec(candidate_id="C6", base_family="ENSEMBLE", cooldown_days=10, adx_gate=True, z_gate=True)
C7_SPEC: Final = ControlSpec(candidate_id="C7", base_family="V1", cooldown_days=5, adx_gate=True, z_gate=False)

CANDIDATE_IDS: Final[tuple[str, ...]] = ("C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7")
NEW_CANDIDATE_IDS: Final[tuple[str, ...]] = ("C1", "C2", "C3", "C4", "C5", "C6", "C7")  # excludes C0 (the baseline itself)
CONTROL_SPECS: Final[dict[str, ControlSpec]] = {
    "C0": C0_SPEC,
    "C1": C1_SPEC,
    "C2": C2_SPEC,
    "C3": C3_SPEC,
    "C4": C4_SPEC,
    "C5": C5_SPEC,
    "C6": C6_SPEC,
    "C7": C7_SPEC,
}

# wave25 counterpart each candidate's BASE SIGNAL reuses unmodified, for the report's
# before/after (wave25 -> wave26) comparison table (SPEC.md "통제 전(wave-25) vs 후(wave-26)
# 거래수·비용·최종액 대비표").
WAVE25_SIGNAL_SOURCE: Final[dict[str, str]] = {
    "C0": "B0 (V1, no signal change)",
    "C1": "B1 (MACD, no signal change)",
    "C2": "B1 (MACD, no signal change)",
    "C3": "B1 (MACD, no signal change)",
    "C4": "B3 (Supertrend, no signal change)",
    "C5": "B7 (Ensemble, no signal change)",
    "C6": "B7 (Ensemble, no signal change)",
    "C7": "B0 (V1, no signal change)",
}

__all__ = [
    "ADX_REGIME_THRESHOLD",
    "ADX_REGIME_WINDOW",
    "BASE_DIR",
    "C0_SPEC",
    "C1_SPEC",
    "C2_SPEC",
    "C3_SPEC",
    "C4_SPEC",
    "C5_SPEC",
    "C6_SPEC",
    "C7_SPEC",
    "CANDIDATE_IDS",
    "CONTROL_SPECS",
    "ControlSpec",
    "DSR_CUMULATIVE_TRIALS",
    "GAMBLE_CAPITAL",
    "HARD_STOP_ATR_MULT",
    "HARD_STOP_PCT",
    "I5_RESULTS_PATH",
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
    "P5_MIN_LEG_USDT",
    "P5_STRESS_MULTIPLIER",
    "PRIOR_CUMULATIVE_TRIALS",
    "Q4_MAX_COST_FRACTION_OF_SLEEVE",
    "Q4_MAX_COST_USDT",
    "REGISTRY_PATH",
    "REPORT_DIR",
    "REPO_ROOT",
    "RESULTS_DIR",
    "RISK_ATR_WINDOW",
    "STABLE_CAPITAL",
    "SYMBOLS",
    "TOTAL_CAPITAL",
    "TRAILING_ACTIVATE_ATR_MULT",
    "TRAILING_ATR_MULT",
    "WAVE1_CACHE_DIR",
    "WAVE25_SIGNAL_SOURCE",
    "WAVE6_CACHE_DIR",
    "Z_SCORE_THRESHOLD",
    "Z_SCORE_WINDOW_BARS",
    "Z_SCORE_WINDOW_DAYS",
]
