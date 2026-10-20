# Wave-12 pre-registered universe-expansion-frontier configs (frozen 2026-07-22,
# research/wave12_frontier/SPEC.md). Seven configs, varying EXACTLY the two axes SPEC.md
# freezes -- volume-rank breadth and funding-history-months floor -- and nothing else:
# every FundingCandidate below is (window_days=7, threshold_apr=0.15, top_k=1) and every
# leg_fraction is 0.50, i.e. SPEC.md's literal "공통 고정: 델타중립 2레그, 레버리지 1x, 1쌍
# $45/$45(활성자본 $90), 진입 15%APR/청산 7.5%" -- byte-for-byte the same candidate
# research.wave10_carry100.configs registers as C1 / research.wave11_yield.configs
# registers as Y4. The cost model (tiered slippage + liquidity floor, see
# research/wave12_frontier/costs_tiered.py) is NOT a per-config field here because
# SPEC.md requires it applied identically to all seven, U0 included -- it is wired in at
# the engine layer (research/wave12_frontier/engine12.py), not registered per candidate.

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from research.wave1.fam_funding import FundingCandidate


@dataclass(frozen=True, slots=True)
class Wave12Config:
    candidate: FundingCandidate
    leg_fraction: float  # fraction of ACTIVE capital ($90) sized into EACH leg -- identical semantics to Wave10Config/Wave11Config
    breadth: int | None  # volume-rank cap (top-N by reference_volume_30d as of FROZEN_END); None = unlimited (U3)
    history_months: float  # required total funding-history length in months (3.0 / 6.0 / 12.0)
    note: str


CONFIGS: Final[tuple[Wave12Config, ...]] = (
    Wave12Config(
        FundingCandidate("U0", 7, 0.15, 1),
        0.50,
        100,
        12.0,
        "Y4 재현(계층비용 적용) = 새 기준선. 볼륨 top100 + 펀딩히스토리 12mo -- universe "
        "-membership rule identical to research.wave11_yield.fetch_y11.expand_universe_y4; "
        "only the cost model (tiered slippage + $2M 유동성 하한, 기존 알트 일괄 3bp 대체) "
        "differs from Y4's own run.",
    ),
    Wave12Config(
        FundingCandidate("U1", 7, 0.15, 1),
        0.50,
        150,
        12.0,
        "폭↑: 볼륨 top150, 12mo (U0 대비 유니버스 폭만 +50).",
    ),
    Wave12Config(
        FundingCandidate("U2", 7, 0.15, 1),
        0.50,
        200,
        12.0,
        "폭↑↑: 볼륨 top200, 12mo (U0 대비 유니버스 폭만 +100).",
    ),
    Wave12Config(
        FundingCandidate("U3", 7, 0.15, 1),
        0.50,
        None,
        12.0,
        "확장 상한: 무제한(유동성 하한 $2M만 적용), 12mo -- 볼륨 랭크 캡 없이 12mo 히스토리를 "
        "채우는 모든 후보(research/wave12_frontier/cache/candidate_pool.json이 실제로 probe한 "
        "Binance USDT 교차상장 전량). '무제한'은 이론적 전체가 아니라 이 wave가 실제로 조회한 "
        "후보 풀로 조작적 정의됨 -- 한계는 리포트에 명시.",
    ),
    Wave12Config(
        FundingCandidate("U4", 7, 0.15, 1),
        0.50,
        100,
        6.0,
        "신선도↑: 볼륨 top100, 펀딩히스토리 요건만 6mo로 완화 (신규상장 편입, 고펀딩 경향 시험).",
    ),
    Wave12Config(
        FundingCandidate("U5", 7, 0.15, 1),
        0.50,
        200,
        6.0,
        "폭+신선도 결합: 볼륨 top200, 6mo.",
    ),
    Wave12Config(
        FundingCandidate("U6", 7, 0.15, 1),
        0.50,
        200,
        3.0,
        "최대 공격: 볼륨 top200, 펀딩히스토리 3mo -- 신규상장 집중 편입.",
    ),
)

CONFIG_IDS: Final[tuple[str, ...]] = tuple(config.candidate.candidate_id for config in CONFIGS)


def get_config(candidate_id: str) -> Wave12Config:
    for config in CONFIGS:
        if config.candidate.candidate_id == candidate_id:
            return config
    raise KeyError(f"unknown wave12 config: {candidate_id}")


__all__ = [
    "CONFIGS",
    "CONFIG_IDS",
    "Wave12Config",
    "get_config",
]
