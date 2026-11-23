# Wave-18 idle-capital-deployment configs (frozen 2026-07-23, research/wave18_idle/SPEC.md).
# Six candidates (I0-I5), all layered on top of the SAME L4 signal+universe
# (research.wave13_liquidity.configs13's L4 Wave13Config -- imported, not redefined: SPEC.md
# "공통: ... L4 신호·유니버스 승계"). What varies per candidate is ONLY what happens on a day
# L4 itself holds no position (engine18.py's shared day-loop turns this table into six actual
# backtests -- see its module docstring for why one parameterized loop, not six copied loops,
# is the right level of abstraction here: unlike the wave10->...->wave17 lineage (genuinely
# different engines across DIFFERENT waves), these six candidates are tightly-related variants
# WITHIN one wave, sharing the identical primary layer and cost model by SPEC.md construction).

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Final

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

from research.wave1.fam_funding import FundingCandidate
from research.wave13_liquidity.configs13 import Wave13Config
from research.wave13_liquidity.configs13 import get_config as get_wave13_config

L4_CONFIG: Final[Wave13Config] = get_wave13_config("L4")  # primary layer for every I0-I5 candidate, unmodified (top200 breadth, 12mo history, threshold 15%/top_k=1/leg 0.50)

# ---------------------------------------------------------------------------
# Overlay building blocks.
# ---------------------------------------------------------------------------

# I2/I3/I5's carry-overlay signal: SAME window (7d) and top_k (1) as L4 -- SPEC.md
# "L4 엔진의 threshold_apr만 변경" -- ONLY threshold_apr changes, 15% -> 8%. carry_position's
# own hard-coded exit=threshold/2 rule then gives entry 8%/exit 4%, matching wave-11 Y1's own
# convention (research/wave11_yield/SPEC.md: "진입 8% APR / 청산 4%") so the wave-11-vs-wave-18
# comparison in the report is apples to apples on the hysteresis rule itself -- only the
# universe and cost model differ, which is exactly what SPEC.md asks this wave to re-test.
OVERLAY_CARRY_CANDIDATE: Final = FundingCandidate("I18_carry8", 7, 0.08, 1)

# I2/I5's overlay universe: SPEC.md's named BTC/ETH pair -- the same literal 2-symbol set as
# wave13 L1 (research.wave13_liquidity.configs13.CONFIGS[0].fixed_symbols), restated here
# (not imported) because L1's own Wave13Config carries L1-specific fields (universe_kind=
# "fixed", history_months, etc.) wave18 has no use for -- only the symbol tuple itself matters.
MAJORS_ONLY_SYMBOLS: Final[tuple[str, ...]] = ("BTCUSDT", "ETHUSDT")

# I4's reverse-carry overlay: SPEC.md "펀딩 < -15% APR". engine18.reverse_carry_position
# mirrors carry_position's own entry/exit-half hysteresis with the sign flipped (entry when
# score < -threshold_apr, exit when score > -threshold_apr/2), so threshold_apr is carried as
# a plain MAGNITUDE here -- the sign flip lives entirely in engine18.py, not in this candidate
# object (FundingCandidate itself has no sign/direction field).
OVERLAY_REVERSE_CANDIDATE: Final = FundingCandidate("I18_reverse15", 7, 0.15, 1)

LEG_FRACTION: Final = 0.50  # == L4_CONFIG.leg_fraction; restated so callers don't need to reach into configs13 for a bare float at every call site
TOP_K: Final = 1  # == L4_CONFIG.candidate.top_k; every wave18 layer (primary + every overlay) shares this


@dataclass(frozen=True, slots=True)
class IdleConfig:
    candidate_id: str
    uses_carry_overlay: bool  # I2/I3/I5: try an 8%-threshold carry position on L4-idle days
    overlay_symbols: tuple[str, ...] | None  # restrict the carry overlay to these symbols; None == full L4 (top200) universe
    uses_reverse_overlay: bool  # I4 only: try a reverse-carry (funding < -15% APR) position on L4-idle days
    uses_lending_fallback: bool  # I1/I5: after L4 AND any carry/reverse overlay both miss, park idle capital in USDT lending
    note: str


CONFIGS: Final[tuple[IdleConfig, ...]] = (
    IdleConfig("I0", False, None, False, False, "현금 보유(=L4) 기준선 -- 유휴자본에 아무 것도 하지 않음."),
    IdleConfig("I1", False, None, False, True, "USDT 대여(OKX lendingRate 실측, 현재 스냅샷 상수 하한 -- 시계열 미검증)."),
    IdleConfig("I2", True, MAJORS_ONLY_SYMBOLS, False, False, "BTC/ETH 한정, 임계 8% APR 캐리로 하향 진입."),
    IdleConfig("I3", True, None, False, False, "I2 + 알트까지(top200 전체), 임계 8% APR -- wave-11 Y1(MDD13% FAIL) 재검정 대상."),
    IdleConfig("I4", False, None, True, False, "역캐리: 펀딩 < -15% APR(숏이 롱에 지불)일 때 롱퍼프+숏현물."),
    IdleConfig("I5", True, MAJORS_ONLY_SYMBOLS, False, True, "I2(메이저 저임계 캐리) 우선, 없으면 I1(USDT 대여) 폴백 -- 계층적."),
)
CONFIG_IDS: Final[tuple[str, ...]] = tuple(config.candidate_id for config in CONFIGS)


def get_config(candidate_id: str) -> IdleConfig:
    for config in CONFIGS:
        if config.candidate_id == candidate_id:
            return config
    raise KeyError(f"unknown wave18 config: {candidate_id}")


__all__ = [
    "CONFIGS",
    "CONFIG_IDS",
    "L4_CONFIG",
    "LEG_FRACTION",
    "MAJORS_ONLY_SYMBOLS",
    "OVERLAY_CARRY_CANDIDATE",
    "OVERLAY_REVERSE_CANDIDATE",
    "TOP_K",
    "IdleConfig",
    "get_config",
]
