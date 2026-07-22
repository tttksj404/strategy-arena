# Wave-11 pre-registered candidates (frozen 2026-07-21, research/wave11_yield/SPEC.md).
#
# Six candidates, each varying exactly ONE (Y1-Y2-Y4-Y5) or ONE COMBINED (Y3 = Y1+Y2's
# axes together, itself pre-registered as its own row, not a post-hoc combination) of the
# five under-optimized axes SPEC.md names: entry/exit threshold, concurrent-pair count
# (== gross notional split), universe breadth, rebalance cadence, and a dedicated
# funding-spike rule. Delta-neutral 2-leg structure, 1x leverage, the cost model (maker
# 0.02%/leg + slippage 1bp majors/3bp alts), 8h real funding accrual, and the
# t-close-signal -> t+1-open-execution timing are unchanged from research/wave10_carry100
# (S1 in gates_y.py enforces this structurally, same as wave10's gate_a).
#
# leg_fraction is "fraction of ACTIVE capital ($90) sized into EACH leg (spot long, perp
# short) per concurrently-active pair" -- identical semantics to
# research.wave10_carry100.configs.Wave10Config.leg_fraction. Y1-Y5's dollar figures come
# directly from SPEC.md's literal text. Y6's SPEC.md row ("스파이크... 최대 2쌍") gives a
# pair count but no leg dollar amount; this module adopts wave10 C3's already-registered
# precedent for a 2-pair config (25%/leg, gross exactly 1.0x active capital) rather than
# inventing a new number -- called out here explicitly rather than silently picked, the
# same way research/wave10_carry100/configs.py flagged its own C4 ambiguity in place
# instead of hiding it.

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

from research.wave1.fam_funding import FundingCandidate

Axis = Literal["threshold_down", "width_up", "threshold_down_width_up", "universe_up", "speed_up", "spike"]
Universe = Literal["baseline40", "expanded_y4", "majors_8h"]
Bar = Literal["1D", "8h"]


@dataclass(frozen=True, slots=True)
class Wave11Config:
    candidate: FundingCandidate
    leg_fraction: float
    axis: Axis
    universe: Universe
    bar: Bar
    note: str


# Y6-only spike-rule constants (SPEC.md: "단일 8h 펀딩 > 0.05%(연 55%+) 즉시 진입, 3일 보유
# 또는 7d APR<5% 청산"). Not part of FundingCandidate's hysteresis fields (those are
# threshold_apr entry/exit-on-annualized-score, a different mechanism from a single-print
# raw-rate trigger + fixed hold window) -- kept as their own named constants so the
# active-position rule in engine_y.py has one unambiguous source, and reused by
# gates_y.py/tests instead of being re-typed as magic numbers.
Y6_SPIKE_ENTRY_RATE: Final = 0.0005  # single 8h funding print > 0.05%
Y6_SPIKE_EXIT_APR: Final = 0.05  # 7d rolling annualized funding < 5% -> exit
Y6_SPIKE_HOLD_DAYS: Final = 3  # max hold, in daily marks, before forced exit regardless of score

CONFIGS: Final[tuple[Wave11Config, ...]] = (
    Wave11Config(
        FundingCandidate("Y1", 7, 0.08, 1),
        0.50,
        "threshold_down",
        "baseline40",
        "1D",
        "임계↓: 1쌍 $45/$45 (leg_fraction 0.50 @ $90 active), 진입 8%APR / 청산 4%APR "
        "-- carry_position()의 내장 히스테리시스(청산=threshold/2)가 8%/2=4%를 그대로 "
        "재현하므로 exit 값을 별도로 지정할 필요가 없음. C1(15%/7.5%) 대비 임계만 낮춰 "
        "가동률(activity)을 시험한다.",
    ),
    Wave11Config(
        FundingCandidate("Y2", 7, 0.15, 4),
        11.25 / 90.0,
        "width_up",
        "baseline40",
        "1D",
        "폭↑: 4쌍 x $11.25/레그 (leg_fraction 0.125 @ $90 active), gross = 2*4*0.125*90 "
        "= $90 = 1.0x active capital. 임계는 C1과 동일 15%APR/7.5%. 동시기회 포착 폭만 "
        "넓힌다.",
    ),
    Wave11Config(
        FundingCandidate("Y3", 7, 0.08, 4),
        11.25 / 90.0,
        "threshold_down_width_up",
        "baseline40",
        "1D",
        "임계↓+폭↑: Y1의 8%/4% 임계와 Y2의 4쌍x$11.25 폭을 함께 적용 (SPEC.md에 그 자체로 "
        "사전등록된 6번째 후보 중 하나이며, Y1/Y2 결과를 보고 사후에 조합한 것이 아님).",
    ),
    Wave11Config(
        FundingCandidate("Y4", 7, 0.15, 1),
        0.50,
        "universe_up",
        "expanded_y4",
        "1D",
        "유니버스↑: 1쌍 $45/$45, 임계는 C1과 동일 15%/7.5%. 유일한 차이는 종목 풀 -- "
        "펀딩히스토리 요건 24mo→12mo + 볼륨랭킹 top-40→top-100 "
        "(research/wave11_yield/cache/universe_y4.json, fetch_y11.expand_universe_y4가 "
        "생성).",
    ),
    Wave11Config(
        FundingCandidate("Y5", 7, 0.15, 1),
        0.50,
        "speed_up",
        "majors_8h",
        "8h",
        "속도↑: 1쌍 $45/$45, 임계는 C1과 동일 15%/7.5%, 단 리밸런스/체결을 일봉 대신 "
        "8h바(펀딩 정산주기와 정렬)로 수행. 스팟+퍼프 8h 실캔들이 있는 BTC/ETH/SOL로 "
        "유니버스를 제한 (research/wave6/cache의 퍼프 1h + 이 wave가 새로 수집한 스팟 1h, "
        "8h로 리샘플). 다른 심볼은 인트라데이 스팟 데이터가 캐시/수집 범위 어디에도 없어 "
        "제외 -- 가정으로 채우지 않음.",
    ),
    Wave11Config(
        FundingCandidate("Y6", 7, Y6_SPIKE_EXIT_APR, 2),
        0.25,
        "spike",
        "baseline40",
        "1D",
        f"스파이크: 단일 8h 펀딩 > {Y6_SPIKE_ENTRY_RATE:.2%} 즉시 진입(다음 바 체결), "
        f"{Y6_SPIKE_HOLD_DAYS}일 보유 또는 7d APR<{Y6_SPIKE_EXIT_APR:.0%} 청산, 최대 2쌍 "
        "-- 2쌍 x 25%/레그(=$22.5/레그, gross 1.0x)는 wave10 C3의 2쌍 사이징 선례를 그대로 "
        "적용한 것 (SPEC.md에 레그 $ 미지정, 이 파일 상단 docstring 참조). "
        "FundingCandidate.threshold_apr 필드는 여기서는 표준 히스테리시스에 쓰이지 않고 "
        "exit_apr(5%)을 문서화 목적으로만 담는다 -- 실제 진입/청산 로직은 "
        "engine_y.y6_spike_active_builder가 Y6_SPIKE_* 상수로 직접 수행.",
    ),
)

CONFIG_IDS: Final[tuple[str, ...]] = tuple(config.candidate.candidate_id for config in CONFIGS)


def get_config(candidate_id: str) -> Wave11Config:
    for config in CONFIGS:
        if config.candidate.candidate_id == candidate_id:
            return config
    raise KeyError(f"unknown wave11 config: {candidate_id}")


__all__ = [
    "CONFIGS",
    "CONFIG_IDS",
    "Y6_SPIKE_ENTRY_RATE",
    "Y6_SPIKE_EXIT_APR",
    "Y6_SPIKE_HOLD_DAYS",
    "Wave11Config",
    "get_config",
]
