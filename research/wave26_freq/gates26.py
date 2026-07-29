# Wave-26 promotion gates Q1-Q5 (SPEC.md "게이트" section). Q1/Q2/Q5's underlying statistics
# (skewness, top-decile contribution, bootstrap, MC ruin, max-single-trade-loss, executability)
# are NOT reimplemented here -- they are the exact same generic math research.wave25_gamble.gates25
# already implements and tests (research/wave25_gamble/tests/test_wave25.py), reused verbatim and
# only re-labeled Q1/Q2/Q5 (this wave's own gate IDs) instead of P1/P2/P5. Q3 reuses gates25's own
# "beats a baseline" comparator, called against C0's OWN realized final equity (not a frozen
# number). Q4 is the one genuinely NEW gate this wave introduces (SPEC.md: "비용 효율(신규·핵심)").
#
# Promotion formula differs from wave25's own P1*P2*(P3-or-P4): SPEC.md line 37 --
# "승격 = Q1·Q2·Q4 필수 + Q3" -- reads as ALL FOUR required (Q1, Q2, Q4 called out as
# non-negotiable "필수", Q3 appended with "+" rather than wave25's "or"), a strictly stricter bar
# than wave25's. See evaluate_candidate's own promoted= line below.

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

import numpy as np
import pandas as pd  # noqa: PANDAS_OK

from research.wave25_gamble import gates25
from research.wave25_gamble.gates25 import (
    GateOutcome,
    best_trade_sensitivity,
    bootstrap_skew_diagnostic,
    calendar_year_return,
    daily_returns_array,
    deflated_sharpe_reference,
    full_period_annualized,
    max_single_trade_loss_usdt,
    no_overlapping_positions,
    oos_performance,
    skewness,
    top_decile_contribution,
)
from research.wave26_freq.configs26 import DSR_CUMULATIVE_TRIALS, GAMBLE_CAPITAL, Q4_MAX_COST_FRACTION_OF_SLEEVE, Q4_MAX_COST_USDT


def _status(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def _relabel(outcome: GateOutcome, gate_id: str) -> GateOutcome:
    return GateOutcome(gate_id=gate_id, name=outcome.name, status=outcome.status, detail=outcome.detail)


# ---------------------------------------------------------------------------
# Q1 -- convexity (== wave25 P1, relabeled). Skew>0 AND top-decile trades carry >=50% of gross
# profit AND bootstrap skew lower bound (p05) > 0.
# ---------------------------------------------------------------------------


def gate_q1_convexity(trade_pnls: np.ndarray, final_equity: float, seed: int) -> tuple[GateOutcome, dict[str, Any]]:
    outcome, diagnostics = gates25.gate_p1_convexity(trade_pnls, final_equity, seed)
    return _relabel(outcome, "Q1"), diagnostics


# ---------------------------------------------------------------------------
# Q2 -- bankruptcy defense (== wave25 P2, relabeled, non-negotiable regardless of anything else).
# ---------------------------------------------------------------------------


def gate_q2_bankruptcy(combined_equity: pd.Series, trades_payload: list[dict[str, Any]], seed: int) -> tuple[GateOutcome, dict[str, Any]]:
    outcome, payload = gates25.gate_p2_bankruptcy(combined_equity, trades_payload, seed)
    return _relabel(outcome, "Q2"), payload


# ---------------------------------------------------------------------------
# Q3 -- sleeve final equity beats C0's own realized final equity (SPEC.md "슬리브 최종액 > C0").
# ---------------------------------------------------------------------------


def gate_q3_beats_c0(candidate_final_usdt: float, c0_final_usdt: float) -> GateOutcome:
    outcome = gates25.gate_p3_beats_baseline(candidate_final_usdt, c0_final_usdt)
    detail = f"candidate_sleeve_final=${candidate_final_usdt:.4f} vs C0_sleeve_final=${c0_final_usdt:.4f} (must strictly beat it)"
    return GateOutcome("Q3", "beats_c0_baseline", outcome.status, detail)


# ---------------------------------------------------------------------------
# Q4 -- cost efficiency (NEW, SPEC.md "총비용 <= 슬리브의 40%($10)"). Directly gates the failure
# mode wave25's own report diagnosed: an edge that exists in principle but gets eaten by
# trading-too-often costs.
# ---------------------------------------------------------------------------


def gate_q4_cost_efficiency(total_cost_usdt: float, gamble_capital: float = GAMBLE_CAPITAL, max_fraction: float = Q4_MAX_COST_FRACTION_OF_SLEEVE) -> tuple[GateOutcome, dict[str, Any]]:
    max_cost = gamble_capital * max_fraction
    cost_fraction = (total_cost_usdt / gamble_capital) if gamble_capital > 0.0 else float("nan")
    ok = total_cost_usdt <= max_cost + 1e-9
    detail = f"total_cost=${total_cost_usdt:.4f} ({cost_fraction*100:.1f}% of ${gamble_capital:.0f} sleeve) vs cap=${max_cost:.4f} ({max_fraction*100:.0f}%)"
    payload = {"total_cost_usdt": total_cost_usdt, "cost_fraction_of_sleeve": cost_fraction, "max_cost_usdt": max_cost, "max_fraction": max_fraction}
    return GateOutcome("Q4", "cost_efficiency", _status(ok), detail), payload


# ---------------------------------------------------------------------------
# Q5 -- executability (== wave25 P5, relabeled; disclosed, non-gating -- SPEC.md's own promotion
# formula names only Q1/Q2/Q3/Q4).
# ---------------------------------------------------------------------------


def gate_q5_executable(trades_payload: list[dict[str, Any]], base_final_usdt: float, stressed_final_usdt: float, starting_equity: float = GAMBLE_CAPITAL) -> tuple[GateOutcome, dict[str, Any]]:
    outcome, payload = gates25.gate_p5_executable(trades_payload, base_final_usdt, stressed_final_usdt, starting_equity)
    return _relabel(outcome, "Q5"), payload


# ---------------------------------------------------------------------------
# Orchestration.
# ---------------------------------------------------------------------------


def evaluate_candidate(
    candidate_id: str,
    gamble_equity: pd.Series,
    trades_payload: list[dict[str, Any]],
    combined_equity: pd.Series,
    c0_gamble_equity: pd.Series,
    c0_final_usdt: float,
    stressed_final_usdt: float,
    total_cost_usdt: float,
    seed_offset: int,
) -> dict[str, Any]:
    trade_pnls = np.asarray([float(t["pnl_usdt"]) for t in trades_payload], dtype=float)
    final_gamble_equity = float(gamble_equity.dropna().iloc[-1]) if len(gamble_equity.dropna()) else GAMBLE_CAPITAL

    q1, q1_diagnostics = gate_q1_convexity(trade_pnls, final_gamble_equity, seed=20_260_729 + seed_offset * 101)
    q2, q2_payload = gate_q2_bankruptcy(combined_equity, trades_payload, seed=20_260_730 + seed_offset * 103)
    q3 = gate_q3_beats_c0(final_gamble_equity, c0_final_usdt)
    q4, q4_payload = gate_q4_cost_efficiency(total_cost_usdt)
    q5, q5_payload = gate_q5_executable(trades_payload, final_gamble_equity, stressed_final_usdt)

    gates = (q1, q2, q3, q4, q5)
    core_gates = (q1, q2, q3, q4)  # Q5 disclosed only -- SPEC.md's promotion formula names only Q1-Q4
    # SPEC.md line 37: "승격 = Q1·Q2·Q4 필수 + Q3" -- ALL FOUR required (stricter than wave25's P3-or-P4).
    promoted = q1.status == "PASS" and q2.status == "PASS" and q4.status == "PASS" and q3.status == "PASS"
    any_undetermined = any(gate.status == "UNDETERMINED" for gate in core_gates)

    return {
        "candidate_id": candidate_id,
        "gates": [asdict(gate) for gate in gates],
        "mc_ruin": q2_payload,
        "convexity_diagnostics": q1_diagnostics,
        "q4_detail": q4_payload,
        "q5_detail": q5_payload,
        "reference_metrics": {
            "dsr_gamble_sleeve": deflated_sharpe_reference(gamble_equity, trials=DSR_CUMULATIVE_TRIALS),
            "dsr_combined_system": deflated_sharpe_reference(combined_equity, trials=DSR_CUMULATIVE_TRIALS),
            "total_trials_disclosed": DSR_CUMULATIVE_TRIALS,
            "oos_gamble_sleeve": oos_performance(gamble_equity),
            "oos_combined_system": oos_performance(combined_equity),
        },
        "overall": {
            "status": "UNDETERMINED" if (not promoted and any_undetermined) else _status(promoted),
            "passed": sum(gate.status == "PASS" for gate in gates),
            "undetermined": sum(gate.status == "UNDETERMINED" for gate in gates),
            "total": len(gates),
            "promoted": promoted,
        },
    }


__all__ = [
    "GateOutcome",
    "best_trade_sensitivity",
    "bootstrap_skew_diagnostic",
    "calendar_year_return",
    "daily_returns_array",
    "deflated_sharpe_reference",
    "evaluate_candidate",
    "full_period_annualized",
    "gate_q1_convexity",
    "gate_q2_bankruptcy",
    "gate_q3_beats_c0",
    "gate_q4_cost_efficiency",
    "gate_q5_executable",
    "max_single_trade_loss_usdt",
    "no_overlapping_positions",
    "oos_performance",
    "skewness",
    "top_decile_contribution",
]
