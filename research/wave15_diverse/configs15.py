# Wave-15 frozen candidate registry (SPEC.md, frozen 2026-07-22). Exactly the 7 IDs
# SPEC.md's own candidate table actually enumerates: A1, A2, A3, B1, B2, C1, D1 -- SPEC.md's
# section header says "후보 8개" but its own table lists 7 rows across the 4 families (A:3,
# B:2, C:1, D:1 = 7); the task packet handed to this session likewise names exactly these
# same 7 IDs. SPEC.md itself is frozen ("동결, 사후 추가 금지") -- inventing an 8th candidate
# to match the header count would violate that freeze more than the header's own off-by-one
# does, so this registry implements the 7 IDs that are actually specified and flags the
# header/table mismatch in report/wave15_report.md rather than silently reconciling it either
# way.
#
# B1/B2/C1 share ONE universe (wave12_frontier breadth=200, 12mo history -- BYTE-FOR-BYTE
# L4's own membership rule) so that the ONLY thing that differs between each of them and the
# L4 reference recompute is the single mechanism SPEC.md names for that candidate (dual
# yield / directional collateral / predictive entry) -- never universe breadth, which
# wave12-14 already ran to saturation and which this wave is explicitly not re-testing.

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from research.wave15_diverse.engine_intraday import A_SERIES_SYMBOLS, IntradayConfig

FROZEN_DATE: Final = "2026-07-22"
CANDIDATE_IDS: Final[tuple[str, ...]] = ("A1", "A2", "A3", "B1", "B2", "C1", "D1")

SHARED_DAILY_BREADTH: Final = 200  # == L4's own breadth (research/wave13_liquidity/configs13.py)
SHARED_DAILY_HISTORY_MONTHS: Final = 12.0  # == L4's own history floor

# ---------------------------------------------------------------------------
# A1-A3 (intraday carry) -- engine_intraday.IntradayConfig instances.
# ---------------------------------------------------------------------------

A1_FAST_ENTRY_THRESHOLD: Final = 0.0  # SPEC.md A1: no magnitude filter -- any positive prior-period realized funding
A2_A3_FAST_ENTRY_THRESHOLD: Final = 0.0003  # SPEC.md A2: "직전 펀딩 > 0.03% = 연 33%"
A3_DAILY_ENTRY_THRESHOLD: Final = 0.15  # SPEC.md A3: "7d APR>15%면 일봉 보유 유지" -- literally the baseline entry bar
A3_DAILY_EXIT_THRESHOLD: Final = 0.075  # carry_position's own hysteresis half-bar, reused verbatim

A1_CONFIG: Final = IntradayConfig("A1", A1_FAST_ENTRY_THRESHOLD, None, None)
A2_CONFIG: Final = IntradayConfig("A2", A2_A3_FAST_ENTRY_THRESHOLD, None, None)
A3_CONFIG: Final = IntradayConfig("A3", A2_A3_FAST_ENTRY_THRESHOLD, A3_DAILY_ENTRY_THRESHOLD, A3_DAILY_EXIT_THRESHOLD)

A_SERIES_CONFIGS: Final[dict[str, IntradayConfig]] = {"A1": A1_CONFIG, "A2": A2_CONFIG, "A3": A3_CONFIG}


# ---------------------------------------------------------------------------
# B1/B2/C1 (daily) -- structure + universe declaration.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DailyCandidateConfig:
    candidate_id: str
    structure: str  # "spot_perp" | "perp_only_short" -- engine_daily.Structure
    breadth: int
    history_months: float
    note: str


B1_CONFIG: Final = DailyCandidateConfig(
    "B1",
    "spot_perp",
    SHARED_DAILY_BREADTH,
    SHARED_DAILY_HISTORY_MONTHS,
    "L4와 동일 유니버스(top200,12mo)/동일 진입(15%APR)/동일 구조(spot+perp 델타중립) -- 유일한 차이: "
    "놀고 있는 현물 레그를 Simple Earn Flexible에 넣는다고 가정, 캐리 수익에 가산(ASSUMED APR, "
    "실측 아님 -- earn_apr.py 참조).",
)
B2_CONFIG: Final = DailyCandidateConfig(
    "B2",
    "perp_only_short",
    SHARED_DAILY_BREADTH,
    SHARED_DAILY_HISTORY_MONTHS,
    "L4와 동일 유니버스/동일 진입 신호지만 현물 레그가 아예 없다 -- USDT 담보만 쥐고 숏퍼프 단독. "
    "델타중립 아님(방향노출), 1레그 비용만 부과, USDT 담보도 동일 ASSUMED APR로 이자수취 가정.",
)
C1_CONFIG: Final = DailyCandidateConfig(
    "C1",
    "spot_perp",
    SHARED_DAILY_BREADTH,
    SHARED_DAILY_HISTORY_MONTHS,
    "L4와 동일 유니버스/동일 구조(spot+perp 델타중립), 진입 신호만 실현 7d APR 임계값 대신 "
    "예측 합성 z-score(모멘텀+펀딩추세, 계수 고정)로 교체 -- 청산은 7d APR<7.5% 그대로.",
)

DAILY_CANDIDATE_CONFIGS: Final[dict[str, DailyCandidateConfig]] = {"B1": B1_CONFIG, "B2": B2_CONFIG, "C1": C1_CONFIG}


__all__ = [
    "A1_CONFIG",
    "A1_FAST_ENTRY_THRESHOLD",
    "A2_A3_FAST_ENTRY_THRESHOLD",
    "A2_CONFIG",
    "A3_CONFIG",
    "A3_DAILY_ENTRY_THRESHOLD",
    "A3_DAILY_EXIT_THRESHOLD",
    "A_SERIES_CONFIGS",
    "A_SERIES_SYMBOLS",
    "B1_CONFIG",
    "B2_CONFIG",
    "C1_CONFIG",
    "CANDIDATE_IDS",
    "DAILY_CANDIDATE_CONFIGS",
    "DailyCandidateConfig",
    "FROZEN_DATE",
    "SHARED_DAILY_BREADTH",
    "SHARED_DAILY_HISTORY_MONTHS",
]
