# Declares each paper-tracked candidate's REQUIRED live universe and checks a LiveSnapshot
# against it. Exists because of a real incident: G1 (research/paper/candidates.py) is specified
# against a top100-by-volume universe (research/wave21_ga/results/final_candidate.json's genome
# has universe_breadth=100; research/STRATEGY_CARD.md line 254 says the same in prose), but
# research/paper/market_data.py's collect_live_snapshot() was silently returning a funding_series
# covering only ~61 symbols -- most of them with zero or trivial funding. G1 therefore recorded
# "cash" every single day, which reads exactly like "G1 found no opportunity" when the real
# story was "G1 never got to look at the coins that mattered." No exception was raised anywhere;
# the mismatch was only visible by manually counting symbols.
#
# This module is the guard against that happening again, silently, for any candidate: every ID in
# research.paper.candidates.TRACKED_IDS must have an entry in REQUIREMENTS (enforced by
# tests/test_paper_fidelity.py -- adding a candidate without registering its universe fails
# pytest, not just a code review). Every research/paper/track.py run checks the fresh snapshot
# against these requirements and refuses to be quiet about a shortfall (see run_once()'s use of
# check_all -- STATUS.md gets a top-of-file warning banner and a per-row FIDELITY_FAIL marker,
# and the candidate is forced to cash rather than act on a known-incomplete universe).

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
import sys
from typing import Final

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

from research.paper.market_data import LiveSnapshot


class UniverseKind(StrEnum):
    VOLUME_TOP_N = "volume_top_n"  # candidate needs >= min_symbols distinct symbols with usable funding history
    MAJORS_ONLY = "majors_only"  # candidate only ever trades a fixed, small symbol set (e.g. BTC/ETH)
    WAVE3_MOMENTUM = "wave3_momentum"  # candidate needs >= min_symbols entries in wave3_markets


@dataclass(frozen=True, slots=True)
class UniverseRequirement:
    candidate_id: str
    kind: UniverseKind
    min_symbols: int
    required_symbols: tuple[str, ...] = ()  # only meaningful for MAJORS_ONLY
    note: str = ""


@dataclass(frozen=True, slots=True)
class FidelityResult:
    candidate_id: str
    ok: bool
    required: int
    actual: int
    reason: str  # "OK" when ok, else a one-line, STATUS.md-ready explanation


# Single source of truth for "what universe does this candidate's spec actually require". Every
# number here is traceable to a document, not guessed:
#   - G1: research/wave21_ga/results/final_candidate.json final_genome.universe_breadth == 100;
#     research/STRATEGY_CARD.md line 254 "유니버스 top100(top200↓)".
#   - W2c: research/paper/candidates.py's W2c is FundingCandidate(window_days=7,
#     threshold_apr=0.15, top_k=4) -- byte-for-byte the STRATEGY_CARD.md "룰" section (7d APR>15%
#     entry, top4). That configuration IS wave13_liquidity's "L4", whose declared/recommended
#     universe is top200 -- the breadth-sweep saturation point (STRATEGY_CARD.md lines 100/105:
#     "top100 20.3% -> top200 22.0%(정점) -> 동적358 20.6%").
#   - F1e: FundingCandidate(majors_only=True) in research/wave1/fam_funding.py -- BTCUSDT/ETHUSDT
#     only, by construction (carry_position's own majors_only intersection).
#   - W3c/W3d: research.wave3.engine.current_targets(..., volume_limit: int = 150) and
#     research/paper/track.py never overrides that default -- matches
#     research.wave3.universe.VOLUME_LIMIT (150), the wave-3 momentum universe's own spec.
REQUIREMENTS: Final[dict[str, UniverseRequirement]] = {
    "G1": UniverseRequirement(
        "G1", UniverseKind.VOLUME_TOP_N, 100,
        note="wave21_ga final_candidate.json genome universe_breadth=100 (STRATEGY_CARD.md 'top100(top200↓)')",
    ),
    "W2c": UniverseRequirement(
        "W2c", UniverseKind.VOLUME_TOP_N, 200,
        note="W2c params (7d/15%/top4) match wave13 L4 exactly; L4's declared universe is top200 (STRATEGY_CARD.md breadth-sweep saturation point)",
    ),
    "F1e": UniverseRequirement(
        "F1e", UniverseKind.MAJORS_ONLY, 2, required_symbols=("BTCUSDT", "ETHUSDT"),
        note="FundingCandidate(majors_only=True) -- BTC/ETH only",
    ),
    "W3c": UniverseRequirement(
        "W3c", UniverseKind.WAVE3_MOMENTUM, 150,
        note="current_targets default volume_limit=150 (research.wave3.universe.VOLUME_LIMIT)",
    ),
    "W3d": UniverseRequirement(
        "W3d", UniverseKind.WAVE3_MOMENTUM, 150,
        note="current_targets default volume_limit=150 (research.wave3.universe.VOLUME_LIMIT)",
    ),
}


def check_snapshot(snapshot: LiveSnapshot, candidate_id: str) -> FidelityResult:
    """Checks ONE candidate's declared universe requirement against a live snapshot. Raises
    KeyError (loudly, at call time -- not a silent pass) if the candidate has no declared
    requirement; see REQUIREMENTS' docstring and
    tests/test_paper_fidelity.py::test_every_tracked_candidate_has_a_declared_requirement for
    the enforcement this is designed to trigger."""
    requirement = REQUIREMENTS.get(candidate_id)
    if requirement is None:
        raise KeyError(
            f"fidelity: no declared universe requirement for candidate {candidate_id!r} -- "
            "register one in research/paper/fidelity.py REQUIREMENTS before tracking it"
        )
    if requirement.kind is UniverseKind.MAJORS_ONLY:
        actual = sum(1 for symbol in requirement.required_symbols if symbol in snapshot.funding_series)
        required_label = "+".join(requirement.required_symbols)
    elif requirement.kind is UniverseKind.WAVE3_MOMENTUM:
        actual = len(snapshot.wave3_markets)
        required_label = f"top{requirement.min_symbols}"
    else:
        actual = len(snapshot.funding_series)
        required_label = f"top{requirement.min_symbols}"
    ok = actual >= requirement.min_symbols
    reason = "OK" if ok else f"{candidate_id}: 사양 {required_label} 요구, 실제 커버 {actual} — 이 후보의 기록은 무효"
    return FidelityResult(candidate_id, ok, requirement.min_symbols, actual, reason)


def check_all(snapshot: LiveSnapshot, candidate_ids: tuple[str, ...]) -> dict[str, FidelityResult]:
    return {candidate_id: check_snapshot(snapshot, candidate_id) for candidate_id in candidate_ids}


__all__ = [
    "REQUIREMENTS",
    "FidelityResult",
    "UniverseKind",
    "UniverseRequirement",
    "check_all",
    "check_snapshot",
]
