# Wave-22 validation #4 -- DSR recalculation. wave21_ga's own H3 gate computed Deflated Sharpe
# for the RAW GA output (top_k_pairs=3, the genome that then FAILED H4's gross<=1x sizing check)
# at trials=7,500 (this wave's own eval count only) = 0.23594, and separately disclosed a
# reference-only figure at trials=7,621 (=121 prior-wave candidates + 7,500) = 0.23196 -- see
# research/wave21_ga/report/wave21_report.md "DSR / 다중검정". Neither of those two numbers is
# G1's own DSR: G1 differs from that raw GA output by exactly one manual gene change
# (top_k_pairs 3->1, research/wave22_overfit/genomes.py's provenance note), which changes the
# equity curve (different position count -> different daily-return distribution -> different
# Sharpe/skew/kurtosis) and therefore changes the DSR score too. This module recomputes DSR
# directly on G1's OWN equity curve, at the task-specified cumulative trial count
# (121 + 7,500 = 7,621, reusing gates21's own frozen constants rather than re-hardcoding them),
# and -- as an internal-consistency cross-check, since the genome is one JSON load away --
# also recomputes the same thing for the raw GA_FINAL (top_k=3) genome so the report can show
# exactly how much the top_k_pairs fix itself moved the DSR score.
#
# Trial-count scope decision (stated here, not cherry-picked after seeing the score): this
# module counts trials = 121 (wave1-20 registered candidates) + 7,500 (wave-21's own GA
# evaluations) = 7,621, per the task's explicit formula. It does NOT additionally count
# wave22's own ~150 diagnostic evaluations (sensitivity neighbors, shuffled genomes, attribution
# genomes) as extra "trials" -- standard DSR trial-counting counts strategies that competed to be
# SELECTED as the candidate; wave22 runs no selection process at all, it only audits the single,
# already-fixed G1 genome from every angle. Counting audit evaluations as search trials would
# conflate "how hard did we look for a winner" with "how hard did we check one specific answer".

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Final

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

import pandas as pd  # noqa: PANDAS_OK

from research.wave21_ga import fitness, gates21
from research.wave21_ga.genome import from_dict

WAVE21_FINAL_CANDIDATE_PATH: Final = Path(__file__).resolve().parents[1] / "wave21_ga" / "results" / "final_candidate.json"
WAVE21_REPORT_REFERENCE: Final[dict[str, float]] = {
    # Cited from research/wave21_ga/report/wave21_report.md's own "H3" and "DSR / 다중검정"
    # sections (GA_FINAL, top_k=3, BEFORE the top_k_pairs 3->1 sizing fix) -- kept here only as
    # a labeled, disclosed comparison point, never recomputed/overwritten.
    "trials_this_wave_only": 0.23594,
    "trials_cumulative_121_plus_7500": 0.23196,
}


def _dsr_payload(equity: pd.Series, trials: int) -> dict[str, Any] | None:
    result = fitness.deflated_sharpe_for_trials(equity, trials)
    return result


def _load_ga_final_top_k3_genome() -> Any | None:
    if not WAVE21_FINAL_CANDIDATE_PATH.exists():
        return None
    payload = json.loads(WAVE21_FINAL_CANDIDATE_PATH.read_text(encoding="utf-8"))
    return from_dict(payload["final_genome"])


def run(g1_equity: pd.Series, cache: fitness.MarketCache | None = None) -> dict[str, Any]:
    trials_this_wave_only = gates21.GA_TRIALS
    trials_cumulative = gates21.CUMULATIVE_TRIALS_WITH_GA
    if trials_cumulative != gates21.PRIOR_CUMULATIVE_TRIALS + gates21.GA_TRIALS:
        raise AssertionError("dsr.run: gates21.CUMULATIVE_TRIALS_WITH_GA drifted from PRIOR_CUMULATIVE_TRIALS + GA_TRIALS")

    g1_at_this_wave_trials = _dsr_payload(g1_equity, trials_this_wave_only)
    g1_at_cumulative_trials = _dsr_payload(g1_equity, trials_cumulative)

    ga_final_cross_check: dict[str, Any] | None = None
    if cache is not None:
        ga_final_genome = _load_ga_final_top_k3_genome()
        if ga_final_genome is not None:
            ga_final_equity = fitness.run_backtest(ga_final_genome, cache, fitness.MODE_OOS_FINAL)
            ga_final_cross_check = {
                "genome": ga_final_genome.to_dict(),
                "dsr_at_trials_this_wave_only": _dsr_payload(ga_final_equity, trials_this_wave_only),
                "dsr_at_trials_cumulative": _dsr_payload(ga_final_equity, trials_cumulative),
                "full_cagr": fitness.cagr(ga_final_equity),
            }

    return {
        "methodology": {
            "trials_this_wave_only": trials_this_wave_only,
            "trials_cumulative_formula": "gates21.PRIOR_CUMULATIVE_TRIALS (121, wave1-20 registered candidates, frozen figure) + gates21.GA_TRIALS (7,500, wave-21 GA evaluations)",
            "trials_cumulative": trials_cumulative,
            "headline_trial_count": "cumulative (7,621) -- per task instruction to reflect accumulated prior trials, not just this wave's own 7,500",
            "why_wave22_evaluations_excluded_from_trial_count": "wave22 performs no selection search over new candidates; it only audits the single, already-fixed G1 genome (sensitivity neighbors/shuffled genomes/attribution genomes are diagnostic re-evaluations of known points, not competing candidates for promotion)",
            "engine": "fitness.deflated_sharpe_for_trials (research.validation.deep_stats.deflated_sharpe under the hood) on G1's own full equity curve, computed by THIS wave (not reused from wave21_report.md, which scored the pre-sizing-fix top_k=3 genome)",
        },
        "g1_dsr_at_trials_this_wave_only": g1_at_this_wave_trials,
        "g1_dsr_at_trials_cumulative": g1_at_cumulative_trials,
        "g1_dsr_score_cumulative": g1_at_cumulative_trials["score"] if g1_at_cumulative_trials else None,
        "g1_dsr_positive_at_cumulative_trials": bool(g1_at_cumulative_trials and g1_at_cumulative_trials["score"] > 0.0),
        "wave21_report_reference_ga_final_top_k3": WAVE21_REPORT_REFERENCE,
        "ga_final_top_k3_cross_check_this_wave": ga_final_cross_check,
        "limitations": [
            "DSR's own trial-count convention is inherently a judgment call (does a 'trial' count every GA individual, every generation's best, or every genome ever backtested); this module follows wave21_ga's own already-disclosed convention (gates21.GA_TRIALS/PRIOR_CUMULATIVE_TRIALS) rather than inventing a new one, for comparability",
            "the 121 prior-wave figure is a frozen, disclosed constant (gates21.PRIOR_CUMULATIVE_TRIALS), not independently re-derived by this wave -- see gates21.py's own comment for that precedent",
            "DSR assumes daily returns are the right unit and penalizes skew/kurtosis parametrically; it is a multiple-testing correction, not a guarantee -- a positive score is necessary, not sufficient, evidence of a real edge",
        ],
    }


__all__ = ["WAVE21_FINAL_CANDIDATE_PATH", "WAVE21_REPORT_REFERENCE", "run"]
