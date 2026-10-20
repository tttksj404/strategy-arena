# Wave-16 candidate definitions (E0-E4), frozen per SPEC.md "후보 5개" (2026-07-22). Capital
# sizing/universe/leg-fraction are NOT a new axis in this wave -- every candidate shares wave13's
# L4 config VERBATIM (top200 breadth, 12mo history, FundingCandidate(window=7,threshold=0.15,
# top_k=1), leg_fraction=0.50 -- $45/$45 of $90 active capital) via
# research.wave13_liquidity.configs13.get_config("L4") (SPEC.md: "top200 유니버스(L4 승계)").
#
# The ONLY thing that varies across E0-E4 is how (if at all) the current OKX-lending snapshot
# enters (a) the ranking/hysteresis score and (b) the realized per-day PnL -- both fully captured
# by a single (ranking_lending_discount, pnl_lending_discount) pair:
#
#   ID | ranking_discount | pnl_discount | SPEC.md definition
#   E0 | 0.0              | 0.0          | L4 재현(대여 없음) = 기준선
#   E1 | 0.0              | 1.0          | 랭킹=펀딩만(=L4), 실현수익에 실측 대여이자 가산
#   E2 | 1.0              | 1.0          | 랭킹=펀딩+대여이자 합산 (핵심 가설)
#   E3 | 0.5              | 0.5          | E2 + 대여이자 50% 할인 (랭킹·수익 양쪽)
#   E4 | 1.0              | 0.0          | E2 랭킹 그대로 + 대여이자 0% 가정(대여 실패)
#
# This pair is also the natural cache/memoization key for the backtest loop itself (see
# engine16.py's module docstring): SPEC.md's own gating instruction ("MC/블록셔플은 펀딩 부분에만
# 적용 가능") needs, for EVERY candidate, a "same ranking, but lending stripped from PnL"
# companion series -- which is just (ranking_discount, 0.0), i.e. `funding_only_variant_key`
# below. That this happens to equal E0 for E0/E1 and E4 for E2 is not a special case anywhere in
# the code; it falls out of the memoization key matching, which is the whole point of representing
# candidates this way instead of five independent hand-written definitions.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Final

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

from research.wave13_liquidity.configs13 import Wave13Config
from research.wave13_liquidity.configs13 import get_config as get_wave13_config

L4_CONFIG: Final[Wave13Config] = get_wave13_config("L4")


@dataclass(frozen=True, slots=True)
class DualLayerCandidate:
    candidate_id: str
    ranking_lending_discount: float  # 0.0 = funding-only ranking (=L4's own score); 0.5/1.0 = combined ranking at that discount on the CURRENT lending snapshot
    pnl_lending_discount: float  # realized lending yield discount actually added to daily PnL while held; 0.0/0.5/1.0
    note: str


CANDIDATES: Final[tuple[DualLayerCandidate, ...]] = (
    DualLayerCandidate(
        "E0",
        0.0,
        0.0,
        "L4 재현(대여 없음) = 기준선 -- 동일 단면(top200)·동일 비용(wave13 실측)으로 재산출. "
        "engine13.run_candidate(get_config('L4'))를 그대로 호출한다(재구현 아님).",
    ),
    DualLayerCandidate(
        "E1",
        0.0,
        1.0,
        "랭킹=펀딩APR만(=L4, E0과 동일 트레이드 선택), 진입 후 현물 레그를 실측 대여이자로 "
        "운용(수익만 가산, 선정은 불변) -- funding-only companion은 E0 그 자체.",
    ),
    DualLayerCandidate(
        "E2",
        1.0,
        1.0,
        "랭킹=펀딩+대여이자 합산, 임계 15%는 합산 기준 -- 핵심 가설. funding-only companion은 E4.",
    ),
    DualLayerCandidate(
        "E3",
        0.5,
        0.5,
        "E2 + 대여이자 50% 할인(랭킹·수익 양쪽 모두) -- 플랫폼 스프레드·미체결 보수 가정. "
        "고유한 funding-only companion(ranking_discount=0.5, pnl=0)을 따로 필요로 한다.",
    ),
    DualLayerCandidate(
        "E4",
        1.0,
        0.0,
        "E2 + 대여이자 0% 가정(대여 실패 시나리오) -- 랭킹은 E2와 동일(합산 그대로), 수익은 "
        "펀딩만. 정의상 자기 자신이 곧 자신의 funding-only companion.",
    ),
)
CANDIDATE_IDS: Final[tuple[str, ...]] = tuple(candidate.candidate_id for candidate in CANDIDATES)


def get_candidate(candidate_id: str) -> DualLayerCandidate:
    for candidate in CANDIDATES:
        if candidate.candidate_id == candidate_id:
            return candidate
    raise KeyError(f"unknown wave16 candidate: {candidate_id}")


def funding_only_variant_key(candidate: DualLayerCandidate) -> tuple[float, float]:
    """The 'strip lending out of realized PnL, keep this candidate's OWN ranking/trade-selection
    rule' companion variant key (SPEC.md 방법 4: "MC/블록셔플은 펀딩 부분에만 적용 가능함을
    표기. 대여이자 부분은 게이트 미적용"). gates16.py gates ONLY this companion series for every
    candidate -- never the lending-inclusive headline series."""
    return (candidate.ranking_lending_discount, 0.0)


__all__ = [
    "L4_CONFIG",
    "CANDIDATES",
    "CANDIDATE_IDS",
    "DualLayerCandidate",
    "funding_only_variant_key",
    "get_candidate",
]
