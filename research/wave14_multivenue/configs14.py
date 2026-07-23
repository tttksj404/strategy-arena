# Wave-14 pre-registered multi-venue x concurrent-position x capital-tier configs (frozen
# 2026-07-22, research/wave14_multivenue/SPEC.md). Eight configs, M0-M7. SPEC.md's "공통
# 고정" row applies to every one of them identically: delta-neutral 2-leg, 1x leverage,
# entry 15%APR / exit 7.5%APR (research.wave1.fam_funding.FundingCandidate /
# carry_position's own built-in threshold/2 exit, unchanged), daily rebalance, signal at t
# close -> t+1 open, wave13's measured-cost methodology (bucket-median + isotonic PAVA)
# extended per-venue (research/wave14_multivenue/costs_venue.py), 10% cash reserve.
#
# What varies across M0-M7 is exactly three things, matching SPEC.md's own three
# "unexplored axes" framing (거래소 / 동시 포지션 수 / 자본 티어):
#   1. total_capital        -- $100 / $300 / $1,000 / $3,000
#   2. top_k (concurrent pairs) -- 1 / 3 / 10 / 30
#   3. include_bybit         -- Binance/Bitget-only vs +Bybit opportunity pool
# plus a fourth, qualitatively different axis for M6/M7 only: `structure` switches from
# the ordinary same-venue spot+perp carry (engine14.run_carry_candidate, a multi-venue
# generalization of research.wave13_liquidity.engine13 -- SAME loop body, only the cost
# lookup and symbol pool are venue-aware) to the cross-venue funding-spread structure
# (engine14.run_cross_venue_candidate, a genuinely NEW position structure per SPEC.md:
# "M6/M7은 신규 구조(양쪽 퍼프)라 기존 캐리와 별도 패밀리로 등록").
#
# LEG_USDT is deliberately held at the SAME $45 used by wave13's own $45 book-walk
# measurement (research/wave13_liquidity/collect_spreads.py's ORDER_SIZE_USDT) across
# EVERY capital tier -- SPEC.md's table literally repeats "(레그 $45)" at every tier from
# M2 through M5, not a scaled-up per-tier leg size. This is what makes every one of M0-M7
# land at EXACTLY 1x leverage by construction (gross = 2 * top_k * LEG_USDT; active_capital
# = 0.9 * total_capital; the (total_capital, top_k) pairs SPEC.md registers were chosen so
# 2*top_k*45 == 0.9*total_capital exactly for all eight -- tests/test_wave14.py pins this),
# and -- just as importantly -- it means the wave13 Bitget-measured / wave14 Bybit-measured
# cost mappings (both fitted specifically to $45 orders) stay valid at every capital tier
# without needing a second, differently-sized book-walk measurement. Scaling POSITION COUNT
# (top_k) rather than PER-POSITION SIZE is exactly SPEC.md's "동시 포지션 수" axis.

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

from research.wave1.fam_funding import FundingCandidate

RESERVE_FRACTION: Final = 0.10  # SPEC.md "현금버퍼 10%" -- same convention as wave10-13's fixed $10-of-$100
LEG_USDT: Final = 45.0  # SPEC.md "1쌍 $45/$45" at every tier -- see module docstring
WINDOW_DAYS: Final = 7  # SPEC.md "공통 고정" -- unchanged from wave10-13
THRESHOLD_APR: Final = 0.15  # SPEC.md "진입 15%APR/청산 7.5%" -- carry_position's own threshold/2 gives the 7.5% exit

Structure = Literal["carry", "cross_venue_spread"]


@dataclass(frozen=True, slots=True)
class Wave14Config:
    candidate_id: str
    total_capital: float  # $100 / $300 / $1,000 / $3,000 -- SPEC.md capital-tier column
    top_k: int  # concurrent pairs -- SPEC.md "동시 쌍" column
    include_bybit: bool  # False: Binance/Bitget-cost single venue (M0, M2). True: +Bybit opportunity pool
    structure: Structure  # "carry": engine14.run_carry_candidate. "cross_venue_spread": engine14.run_cross_venue_candidate (M6/M7 only)
    baseline_candidate_id: str | None  # which result (an M-id, or an "_AUX_..." internal reference run) this config's promotion check compares its high-funding annualized return against; None for M0/M2 (they ARE the tier's own single-venue baseline, nothing external to beat)
    note: str

    @property
    def active_capital(self) -> float:
        return self.total_capital * (1.0 - RESERVE_FRACTION)

    @property
    def leg_fraction(self) -> float:
        """Fraction of ACTIVE capital sized into EACH leg of EACH concurrently-active pair
        -- LEG_USDT / active_capital, so the nominal per-leg dollar size stays fixed at $45
        regardless of capital tier (see module docstring)."""
        return LEG_USDT / self.active_capital

    @property
    def candidate(self) -> FundingCandidate:
        return FundingCandidate(self.candidate_id, WINDOW_DAYS, THRESHOLD_APR, self.top_k)


# Auxiliary (non-frozen-8, internal-reference-only) single-venue baselines needed because
# SPEC.md's promotion rule ("고펀딩기 연환산 > 동일 자본 티어의 단일거래소 기준선") requires
# a same-capital-tier/same-top_k single-venue comparison point for M4/M5/M7, but the frozen
# 8-config table itself only DEFINES single-venue configs at the $100/1-pair (M0) and
# $300/3-pair (M2) tiers -- there is no "$1,000 단일" or "$3,000 단일" row in SPEC.md's
# table, and SPEC.md forbids adding a 9th/10th *promoted* candidate post-hoc ("구성 8개
# (동결, 사후 추가 금지)"). These two are therefore built and run (engine14.run_carry_candidate
# with include_bybit=False) purely as read-only reference points for the M4/M5 comparison --
# never scored against S1-S6, never eligible for promotion themselves, exactly mirroring
# research.wave13_liquidity.reporting13's own read-only citation of wave12_frontier's U0/U2
# results as an external, non-owned comparison baseline.
AUX_BASELINE_1000_10 = Wave14Config(
    "_AUX_1000_10", 1_000.0, 10, False, "carry", None,
    "내부 참조 전용(승격 대상 아님) -- M4($1,000/10쌍/+Bybit)의 '동일 자본 티어 단일거래소' 비교 기준선.",
)
AUX_BASELINE_3000_30 = Wave14Config(
    "_AUX_3000_30", 3_000.0, 30, False, "carry", None,
    "내부 참조 전용(승격 대상 아님) -- M5($3,000/30쌍/+Bybit)의 '동일 자본 티어 단일거래소' 비교 기준선.",
)
AUX_BASELINES: Final[tuple[Wave14Config, ...]] = (AUX_BASELINE_1000_10, AUX_BASELINE_3000_30)

CONFIGS: Final[tuple[Wave14Config, ...]] = (
    Wave14Config(
        "M0", 100.0, 1, False, "carry", None,
        "L4 재현 = 기준선. wave13 L4(top200/12mo, 전기간)와 동일 유니버스·비용모델이지만 "
        "Binance+Bybit 겹침 구간(2024-01-01~FROZEN_END)으로 절단해 신규 시작자본으로 재실행 "
        "-- wave13의 전기간 22.01%와 직접 비교 금지, 이 절단판만이 M1-M7의 공정 기준선.",
    ),
    Wave14Config(
        "M1", 100.0, 1, True, "carry", "M0",
        "M0와 동일(자본/쌍수) + Bybit 기회풀 추가. 기회 풀 확대 순효과만 분리 관찰.",
    ),
    Wave14Config(
        "M2", 300.0, 3, False, "carry", None,
        "단일거래소, 동시 3쌍 -- $300/3쌍 티어의 단일거래소 기준선(그 자체).",
    ),
    Wave14Config(
        "M3", 300.0, 3, True, "carry", "M2",
        "M2와 동일(자본/쌍수) + Bybit. 거래소 확대 + 동시포지션 확대 결합 효과.",
    ),
    Wave14Config(
        "M4", 1_000.0, 10, True, "carry", "_AUX_1000_10",
        "자본 티어 확대(10쌍, +Bybit). AUX_BASELINE_1000_10(단일거래소 동일 티어)과 비교.",
    ),
    Wave14Config(
        "M5", 3_000.0, 30, True, "carry", "_AUX_3000_30",
        "상위 자본 티어(30쌍, +Bybit) -- 동시 포지션 포화 지점 탐색. AUX_BASELINE_3000_30과 비교.",
    ),
    Wave14Config(
        "M6", 300.0, 3, True, "cross_venue_spread", "M2",
        "거래소간 펀딩 스프레드(신규 구조, 현물 불요, 양쪽 퍼프) -- 동일 심볼을 펀딩 높은 거래소 "
        "숏퍼프 + 낮은 거래소 롱퍼프. M2(동일 $300/3쌍 단일거래소 통상캐리)와 비교.",
    ),
    Wave14Config(
        "M7", 1_000.0, 10, True, "cross_venue_spread", "_AUX_1000_10",
        "M6의 자본 확대판(10쌍, $1,000). AUX_BASELINE_1000_10(동일 티어 단일거래소 통상캐리)과 비교.",
    ),
)

CONFIG_IDS: Final[tuple[str, ...]] = tuple(config.candidate_id for config in CONFIGS)
CROSS_VENUE_IDS: Final[tuple[str, ...]] = tuple(config.candidate_id for config in CONFIGS if config.structure == "cross_venue_spread")


def get_config(candidate_id: str) -> Wave14Config:
    for config in (*CONFIGS, *AUX_BASELINES):
        if config.candidate_id == candidate_id:
            return config
    raise KeyError(f"unknown wave14 config: {candidate_id}")


__all__ = [
    "AUX_BASELINE_1000_10",
    "AUX_BASELINE_3000_30",
    "AUX_BASELINES",
    "CONFIGS",
    "CONFIG_IDS",
    "CROSS_VENUE_IDS",
    "LEG_USDT",
    "RESERVE_FRACTION",
    "THRESHOLD_APR",
    "WINDOW_DAYS",
    "Structure",
    "Wave14Config",
    "get_config",
]
