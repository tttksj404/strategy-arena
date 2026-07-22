# Wave-10 pre-registered $100-capital sizing configs (frozen 2026-07-22).
#
# Hypothesis under test: wave-8 ruled the W2c carry engine infeasible at $100 total
# capital because its native sizing (equal-split top-4, up to 100% weight per leg)
# produces gross notional up to 2x active capital ($180 > $90). This wave asks a
# narrower question: does *fixing* the number of concurrent pairs and the per-leg
# weight (instead of deriving weight from 1/top_k) let the same signal clear the
# $100 gross-exposure and min-order bars while still passing the robustness gates?
#
# Only two things are allowed to change relative to W2c: (1) position sizing
# (fixed fraction of active capital per leg instead of equal-split-of-100%), and
# (2) the number of concurrent pairs (top_k). The funding signal (funding_score),
# entry/exit hysteresis (carry_position), universe, and cost/execution rules are
# imported unchanged from research.wave1 / research.wave2.
#
# Exactly 4 configs. Pre-registered. No post-hoc additions or parameter sweeps.

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from research.wave1.fam_funding import FundingCandidate


@dataclass(frozen=True, slots=True)
class Wave10Config:
    candidate: FundingCandidate
    leg_fraction: float  # fraction of ACTIVE capital sized into EACH leg (spot long, perp short)
    #                       per concurrently-active pair. top_k on the candidate is the pair count.
    note: str


# NOTE on C4: the task spec literally reads "진입임계 완화 25%APR (가동률↑ 시도)"
# ("entry threshold RELAXED to 25% APR, attempting to RAISE utilization"). W2c's frozen
# baseline entry threshold is 15% APR. 25% APR is numerically HIGHER than 15%, which
# mechanically makes entry *harder*, not easier -- the opposite of "relaxed" / "raise
# utilization" in this codebase's own vocabulary (compare W2b: W2a's 8%APR -> 5%APR is
# explicitly labeled "가동률↑" in research/wave2/SPEC.md, i.e. *lower*
# threshold = higher utilization). This looks like a drafting error in the source
# instruction (25% vs a plausibly-intended lower number). Per the wave10 contract's own
# "no post-hoc parameter changes" / "don't silently adjust numbers" rule, this runner
# does NOT silently correct it -- C4 is implemented with the literal registered value
# (25% APR) and the resulting utilization/verdict is reported honestly, with this
# inconsistency flagged in the report rather than hidden.
CONFIGS: Final[tuple[Wave10Config, ...]] = (
    Wave10Config(
        FundingCandidate("C1", 7, 0.15, 1),
        0.50,
        "1 pair, 50% active capital per leg ($45/$45 @ $90 active), gross 1.0x",
    ),
    Wave10Config(
        FundingCandidate("C2", 7, 0.15, 1),
        0.40,
        "1 pair, 40% active capital per leg ($36/$36 @ $90 active), gross 0.8x buffer",
    ),
    Wave10Config(
        FundingCandidate("C3", 7, 0.15, 2),
        0.25,
        "2 pairs, 25% active capital per leg each ($22.5/$22.5 x2 @ $90 active), gross 1.0x",
    ),
    Wave10Config(
        FundingCandidate("C4", 7, 0.25, 1),
        0.45,
        "1 pair, 45% active capital per leg ($40.5/$40.5 @ $90 active) + entry threshold "
        "literally 25% APR per spec text (higher than W2c's 15% baseline -- see module "
        "docstring; mechanically LOWERS utilization, contradicting the spec's own "
        "'raise utilization' intent, implemented as-written and flagged, not corrected)",
    ),
)

CONFIG_IDS: Final[tuple[str, ...]] = tuple(config.candidate.candidate_id for config in CONFIGS)


def get_config(candidate_id: str) -> Wave10Config:
    for config in CONFIGS:
        if config.candidate.candidate_id == candidate_id:
            return config
    raise KeyError(f"unknown wave10 config: {candidate_id}")
