# Wave-22 frozen genome definitions. G1's 7 gene values are copied byte-for-byte from the
# user's task spec / research/STRATEGY_CARD.md "G1 유전자" row -- DO NOT adjust post-hoc (this
# wave audits G1, it never re-tunes it). G1 == the wave-21 GA final candidate
# (research/wave21_ga/results/final_candidate.json's final_genome) with exactly ONE manual
# change: top_k_pairs restored from the GA's own output (3) back to 1, to satisfy the gross<=1x
# constraint that GA_FINAL failed (wave21_ga's H4 gate) -- see
# research/wave21_ga/report/wave21_report.md "H4 미달은... 순수 구조적 제약" and
# research/STRATEGY_CARD.md "제약 복원 후 재평가 -> G1 확정". Every other gene is untouched GA
# output. This wave-22-local provenance note is exactly why validation #1 (parameter stability)
# and #5 (gene attribution) both treat top_k_pairs specially: G1 sits at the LOWER boundary of
# that gene's range, not at a free interior optimum, and every genome that raises it breaks the
# capital-sizing constraint G1 itself was built to satisfy (see perturb.py / sensitivity.py
# feasibility flags).

from __future__ import annotations

from pathlib import Path
import sys
from typing import Final

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

from research.wave21_ga.genome import Genome, I5_BASELINE_GENOME, L4_BASELINE_GENOME

# G1 -- frozen (task spec / STRATEGY_CARD.md "G1 유전자" row). Reported reference metrics
# (STRATEGY_CARD.md, computed once via this exact engine on 2026-07-27's cache): full-period
# CAGR 12.35%, OOS(self-contained) CAGR 4.04%, MC p05 $165.70, ruin 0%, block-shuffle MDD p95
# 3.89%, x3-stress CAGR 9.64%. wave22 re-derives full/OOS CAGR itself (as a reproduction check,
# see tests/test_wave22.py::test_g1_reproduces_strategy_card_reference_numbers) but does NOT
# redo the MC/block-shuffle/stress simulations -- those are wave21_ga's H4-style gates, already
# settled; wave22's own 6 validations are net-new (sensitivity, rolling, regime, DSR,
# attribution, shuffle-control).
G1_GENOME: Final = Genome(
    entry_threshold_apr=0.11818955034178509,
    exit_threshold_ratio=0.3873937165748336,
    window_days=14,
    top_k_pairs=1,
    leg_fraction=0.5,
    universe_breadth=100,
    idle_mode="usdt_lend",
)

# I5 -- this wave's comparison baseline throughout (task spec: "기준선: I5"), read from
# research/wave21_ga/genome.py's own frozen I5_BASELINE_GENOME rather than re-declared here, so
# a future edit to that module cannot silently desync wave22 from wave21/wave18's own I5.
I5_GENOME: Final = I5_BASELINE_GENOME

# L4 -- background lineage only (I5 = L4 + idle-capital overlay). Not evaluated fresh anywhere
# in wave22 (no validation in the task spec compares against L4); cited in the report's
# background section from its own saved research/wave13_liquidity/results/L4.json.
L4_GENOME: Final = L4_BASELINE_GENOME

G1_REFERENCE_METRICS: Final[dict[str, float]] = {
    "full_period_cagr": 0.1235,
    "oos_cagr_self_contained": 0.0404,
    "mc_p05_usdt": 165.70,
    "ruin_probability": 0.0,
    "block_shuffle_mdd_p95": 0.0389,
    "stress_x3_cagr": 0.0964,
}
I5_REFERENCE_METRICS: Final[dict[str, float]] = {
    "full_period_cagr": 0.1027,
    "oos_cagr_self_contained": 0.0306,
}

__all__ = [
    "G1_GENOME",
    "G1_REFERENCE_METRICS",
    "I5_GENOME",
    "I5_REFERENCE_METRICS",
    "L4_GENOME",
]
