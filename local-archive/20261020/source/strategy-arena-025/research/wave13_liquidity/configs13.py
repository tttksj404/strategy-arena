# Wave-13 pre-registered liquidity-constrained-carry configs (frozen 2026-07-22,
# research/wave13_liquidity/SPEC.md). Five configs. Every FundingCandidate below is
# (window_days=7, threshold_apr=0.15, top_k=1) and every leg_fraction is 0.50 -- SPEC.md's
# literal "공통 고정: 델타중립 2레그, 레버리지 1x, 1쌍 $45/$45(활성자본 $90), 진입
# 15%APR/청산 7.5%", byte-for-byte the same candidate research.wave12_frontier.configs12
# registers for U0-U6. The ONLY things that vary across L1-L5 are (a) which symbols are
# eligible on a given day (universe_kind/fixed_symbols/breadth/history_months) and, for L5
# only, (b) the extra per-day dynamic tradability filter
# (dynamic_volume_floor_usdt/dynamic_slippage_cap_bp). The cost model itself (measured,
# not assumed -- research.wave13_liquidity.costs_measured) is NOT a per-config field here,
# same reasoning as configs12.py: SPEC.md requires it applied identically to all five.

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from research.wave1.fam_funding import FundingCandidate

# L5's "랭크 무관" parent candidate pool floor: the loosest funding-history requirement
# used anywhere in this wave (3 months, matching research.wave12_frontier.universe_frontier's
# own LOOSEST_HISTORY_MONTHS for its tier-reference pool) -- NOT a business liquidity rule,
# just "give the daily dynamic filter the broadest possible pool of symbols with at least
# minimal signal history to filter from." The actual admission rule is the per-day dynamic
# filter (volume floor + mapped-slippage cap), evaluated fresh every day in engine13.py.
L5_PARENT_HISTORY_MONTHS: Final = 3.0
L5_DYNAMIC_VOLUME_FLOOR_USDT: Final = 20_000_000.0
L5_DYNAMIC_SLIPPAGE_CAP_BP: Final = 5.0


@dataclass(frozen=True, slots=True)
class Wave13Config:
    candidate: FundingCandidate
    leg_fraction: float  # fraction of ACTIVE capital ($90) sized into EACH leg -- identical semantics to Wave10Config/Wave12Config
    universe_kind: str  # "fixed" (L1) | "breadth" (L2-L4, static snapshot like wave12's U-series) | "dynamic" (L5, re-evaluated every day)
    fixed_symbols: tuple[str, ...] | None  # L1 only: exactly {BTCUSDT, ETHUSDT}, not "whatever breadth=2 ranks highest" (SPEC.md names them explicitly)
    breadth: int | None  # L2-L4 volume-rank cap; None for L1 (uses fixed_symbols) and L5 (uses the dynamic filter, not a static breadth cap)
    history_months: float  # funding-history floor for the STATIC candidate pool this config draws from (12mo for L2-L4 matching wave12's U0-U3; 3mo parent pool for L5; irrelevant but set to 12.0 for L1's fixed 2-symbol list)
    dynamic_volume_floor_usdt: float | None  # L5 only
    dynamic_slippage_cap_bp: float | None  # L5 only
    note: str


CONFIGS: Final[tuple[Wave13Config, ...]] = (
    Wave13Config(
        FundingCandidate("L1", 7, 0.15, 1),
        0.50,
        "fixed",
        ("BTCUSDT", "ETHUSDT"),
        None,
        12.0,
        None,
        None,
        "BTC/ETH 고정 2종 -- 실측상 슬리피지 사실상 0(BTC 0.008bp/ETH 0.026bp 스냅샷 실측)이지만 "
        "top_k=1 신호가 이 2종에서만 발화하므로 기회 자체가 희소함.",
    ),
    Wave13Config(
        FundingCandidate("L2", 7, 0.15, 1),
        0.50,
        "breadth",
        None,
        30,
        12.0,
        None,
        None,
        "거래대금 상위 30, 12mo 히스토리 -- wave12 U0(top100)보다 좁은 폭.",
    ),
    Wave13Config(
        FundingCandidate("L3", 7, 0.15, 1),
        0.50,
        "breadth",
        None,
        100,
        12.0,
        None,
        None,
        "거래대금 상위 100, 12mo -- wave12 U0 대응 (동일 유니버스 선정 규칙, 비용모델만 "
        "실측(costs_measured)으로 교체 -- research.wave12_frontier.costs_tiered의 가정 "
        "1/3/6/10/20bp 계층을 쓰지 않음).",
    ),
    Wave13Config(
        FundingCandidate("L4", 7, 0.15, 1),
        0.50,
        "breadth",
        None,
        200,
        12.0,
        None,
        None,
        "거래대금 상위 200, 12mo -- wave12 정점 U2 대응 (동일 유니버스 규칙, 비용모델만 실측 교체).",
    ),
    Wave13Config(
        FundingCandidate("L5", 7, 0.15, 1),
        0.50,
        "dynamic",
        None,
        None,
        L5_PARENT_HISTORY_MONTHS,
        L5_DYNAMIC_VOLUME_FLOOR_USDT,
        L5_DYNAMIC_SLIPPAGE_CAP_BP,
        "동적 유동성 필터 -- 랭크 무관, 진입(매일) 시점 30d 평균 거래대금 >= $20M 이고 실측매핑 "
        "슬리피지 <= 5bp 인 심볼만 매일 재평가. 이 wave의 핵심 가설: 폭(랭크 컷)이 아니라 "
        "체결가능성(달러유동성+실측비용)으로 거르면 어떻게 되는가.",
    ),
)

CONFIG_IDS: Final[tuple[str, ...]] = tuple(config.candidate.candidate_id for config in CONFIGS)


def get_config(candidate_id: str) -> Wave13Config:
    for config in CONFIGS:
        if config.candidate.candidate_id == candidate_id:
            return config
    raise KeyError(f"unknown wave13 config: {candidate_id}")


__all__ = [
    "L5_DYNAMIC_SLIPPAGE_CAP_BP",
    "L5_DYNAMIC_VOLUME_FLOOR_USDT",
    "L5_PARENT_HISTORY_MONTHS",
    "CONFIGS",
    "CONFIG_IDS",
    "Wave13Config",
    "get_config",
]
