# Single source of truth for which candidates the paper tracker follows.
# Lives apart from track.py so status.py can import the list without a circular import.

from __future__ import annotations

from pathlib import Path
import sys
from typing import Final

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

from research.wave1.fam_funding import FundingCandidate


TRACKED_IDS: Final = ("W2c", "F1e", "W3c", "W3d", "G1")

# G1 = wave-21 GA winner re-scored under the 1x-gross constraint (GA picked top_k=3, which
# breached the leverage gate; the improvement survives at top_k=1). Genome frozen from
# research/wave21_ga/results/final_candidate.json. Forward-validation is a promotion
# condition because its 33.5pp IS-OOS gap is a weak overfitting signal.
G1_CANDIDATE: Final = FundingCandidate(
    candidate_id="G1",
    window_days=14,
    threshold_apr=0.11818955034178509,
    top_k=1,
)
