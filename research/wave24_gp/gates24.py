# Wave-24 gate evaluation (SPEC.md "게이트 (승격 조건, 전부 충족)" -- L1-L7).
#
# L1 - GP's best fitness must beat the random-tree control's best fitness in >= 4 of the 5
#      matched seeds (gates the evolutionary MECHANISM itself, not any one tree).
# L2 - the final candidate's OOS (2025-10~) CAGR must beat I5's own OOS CAGR (read read-only from
#      research/wave18_idle/results/I5.json -- never recomputed, never modified).
# L3 - Deflated Sharpe on the final candidate's OWN full equity curve (SPEC.md/task brief:
#      "DSR은 반드시 최종 승격 개체 자신의 equity curve로 계산" -- wave21_ga's own H3 reported
#      DSR from a genome that later failed its own leverage gate, see gates21.py's module
#      docstring; this wave does not repeat that), at CUMULATIVE trials = 121 (wave1-20) + 7,500
#      (wave21_ga) + 15,000 (wave23_ga_short) + this wave's own total eval count (GP 30,000 +
#      random-tree control 30,000 = 60,000) = 82,621. Must be > 0.
# L4 - ruin/tail defense: Monte-Carlo (1e4 paths) p05 > $100 AND P(final < $50) < 5% AND
#      90-day block-shuffle MDD p95 <= 15% (all on the BASE, unstressed equity curve).
# L5 - executability: every leg >= MIN_ORDER_USDT ($5), gross <= ACTIVE_CAPITAL (1x), and sign
#      (end > start) preserved under x3 measured-slippage stress.
# L6 - SPEC-EXECUTION CONSISTENCY, inherited from research.wave23_ga_short.gates23's own K6 (task
#      brief: "wave-23 K6 계승"): are the final tree's terminal INPUTS obtainable live, and does
#      its (fixed, inherited) universe breadth fit what research/paper/market_data.py actually
#      fetches for the carry family? This gate never modifies research/paper/* -- it only checks.
# L7 - NEW (task brief, direct overfitting defense): formula simplicity -- node count <= 15 AND
#      distinct terminal KINDS used <= 5. Enforced STRICTLY (task brief: "L7... 엄격 적용").
#
# Promotion = L1 AND L2 AND L3 AND L4 AND L5 AND L6 AND L7, all-or-nothing (matches every prior
# wave's own all-or-nothing convention).

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from typing import Any, Final, Sequence

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

import numpy as np
import pandas as pd  # noqa: PANDAS_OK

from research.paper.market_data import CARRY_VOLUME_LIMIT
from research.validation.deep_stats import DeepValidationError, TimedValue, deflated_sharpe
from research.wave10_carry100.engine import ACTIVE_CAPITAL, MIN_ORDER_USDT, RESERVE_FRACTION, TOTAL_CAPITAL
from research.wave23_ga_short.gates23 import CUMULATIVE_TRIALS as PRIOR_CUMULATIVE_TRIALS_BEFORE_WAVE24  # 121 + 7,500 + 15,000 = 22,621, frozen -- see that module's own docstring
from research.wave24_gp import gp, random_trees
from research.wave24_gp.fitness24 import FIXED_LEG_FRACTION, FIXED_TOP_K_PAIRS, FIXED_UNIVERSE_BREADTH
from research.wave24_gp.tree import ALL_TERMINAL_KINDS, MAX_DEPTH, Node, depth, node_count, terminal_kinds_used

MC_PATHS: Final = 10_000  # SPEC.md L4 "MC 1e4"
BLOCK_PATHS: Final = 1_000
BLOCK_DAYS: Final = 90
SEED: Final = 24_260_728  # this wave's own freeze date (2026-07-28), wave-24-prefixed to keep its RNG stream distinct from sibling waves' own freeze-date seeds

L4_P05_FLOOR_USDT: Final = 100.0  # SPEC.md "p05 > $100"
L4_RUIN_PROBABILITY_MAX: Final = 0.05  # SPEC.md "P(<$50) < 5%"
L4_RUIN_FLOOR_USDT: Final = 50.0
L4_BLOCK_MDD_P95_MAX: Final = 0.15  # SPEC.md "블록셔플 MDD p95 <= 15%"

L7_MAX_NODE_COUNT: Final = 15  # SPEC.md L7 "노드 <= 15"
L7_MAX_TERMINAL_KINDS: Final = 5  # SPEC.md L7 "터미널 <= 5종"

GP_TRIALS: Final = gp.EVALUATIONS_PER_SEED * len(gp.SEEDS)  # 6,000 x 5 = 30,000
RANDOM_TRIALS: Final = random_trees.N_EVALUATIONS_PER_SEED * len(random_trees.SEEDS)  # 30,000
THIS_WAVE_TOTAL_TRIALS: Final = GP_TRIALS + RANDOM_TRIALS  # 60,000 -- task brief's own "이번 총평가수"
CUMULATIVE_TRIALS: Final = PRIOR_CUMULATIVE_TRIALS_BEFORE_WAVE24 + THIS_WAVE_TOTAL_TRIALS  # 22,621 + 60,000 = 82,621 -- task brief: "누적시행 = 121 + 7,500 + 15,000 + 이번 총평가수"

LEG_USDT: Final = FIXED_LEG_FRACTION * ACTIVE_CAPITAL  # $45 -- position structure is FIXED (fitness24.py module docstring), so this is a constant, not a per-genome function
GROSS_USDT: Final = 2.0 * FIXED_TOP_K_PAIRS * FIXED_LEG_FRACTION * ACTIVE_CAPITAL  # $90 == ACTIVE_CAPITAL exactly (1x by construction)

# L6 -- paper tracker's ACTUAL live-fetched carry universe (research/paper/market_data.py,
# read-only reference; see that module's own collect_live_snapshot: top-CARRY_VOLUME_LIMIT by
# volume, unioned with {BTCUSDT, ETHUSDT}), same +2 convention as research.wave23_ga_short.
# gates23.PAPER_CARRY_UNIVERSE_CAP.
PAPER_CARRY_UNIVERSE_CAP: Final = CARRY_VOLUME_LIMIT + 2

# L6 -- every wave24_gp terminal maps to one of these 3 live-fetchable raw data families;
# collect_live_snapshot/current_funding_rates (research/paper/market_data.py) already fetch
# funding rates, OHLC price, and Bitget mix volume for the live carry universe, so every terminal
# in tree.TERMINAL_VARS is, by construction of this wave's OWN alphabet, backed by a data type
# the live pipeline already produces -- nothing in SPEC.md's terminal table needs a NEW feed.
_TERMINAL_DATA_FAMILY: Final[dict[str, str]] = {
    "funding_1d": "funding_rate", "funding_7d": "funding_rate", "funding_14d": "funding_rate", "funding_30d": "funding_rate",
    "price_ret_1d": "ohlc_price", "price_ret_7d": "ohlc_price", "price_ret_30d": "ohlc_price",
    "realized_vol_20d": "ohlc_price", "atr_14": "ohlc_price", "basis": "ohlc_price",
    "quote_volume_30d": "quote_volume",
}


# ---------------------------------------------------------------------------
# L1: GP vs random-tree control.
# ---------------------------------------------------------------------------


def gate_l1_gp_beats_random(gp_best_by_seed: Sequence[float], random_best_by_seed: Sequence[float]) -> dict[str, Any]:
    if len(gp_best_by_seed) != len(random_best_by_seed) or len(gp_best_by_seed) == 0:
        raise ValueError("gate_l1_gp_beats_random: GP and random seed-count lists must be equal length and non-empty")
    wins = [bool(gp_value > random_value) for gp_value, random_value in zip(gp_best_by_seed, random_best_by_seed)]
    n_wins = int(sum(wins))
    threshold = 4  # SPEC.md: "5시드 중 4+"
    ok = n_wins >= threshold
    return {
        "gp_best_by_seed": [float(v) for v in gp_best_by_seed],
        "random_best_by_seed": [float(v) for v in random_best_by_seed],
        "per_seed_gp_wins": wins,
        "n_wins": n_wins,
        "n_seeds": len(wins),
        "threshold": threshold,
        "status": "PASS" if ok else "FAIL",
    }


# ---------------------------------------------------------------------------
# L2: final candidate OOS vs I5 OOS (read-only, research/wave18_idle/results/I5.json).
# ---------------------------------------------------------------------------


def gate_l2_beats_i5_oos(final_oos_cagr: float | None, i5_oos_cagr: float | None) -> dict[str, Any]:
    ok = final_oos_cagr is not None and i5_oos_cagr is not None and final_oos_cagr > i5_oos_cagr
    gap_pp = (final_oos_cagr - i5_oos_cagr) * 100.0 if (final_oos_cagr is not None and i5_oos_cagr is not None) else None
    return {
        "final_oos_cagr": final_oos_cagr,
        "i5_oos_cagr": i5_oos_cagr,
        "gap_pp": gap_pp,
        "status": "PASS" if ok else "FAIL",
    }


# ---------------------------------------------------------------------------
# L3: DSR, cumulative trials.
# ---------------------------------------------------------------------------


def gate_l3_dsr(full_equity: pd.Series, trials: int = CUMULATIVE_TRIALS) -> dict[str, Any]:
    clean = full_equity.dropna()
    if len(clean) < 4:
        return {"status": "FAIL", "reason": "insufficient observations for DSR (<4)"}
    timed = tuple(TimedValue(pd.Timestamp(ts).to_pydatetime(), float(value)) for ts, value in clean.items())
    try:
        dsr = deflated_sharpe(timed, trials=trials)
    except DeepValidationError as error:
        return {"status": "FAIL", "reason": str(error)}
    ok = dsr.score > 0.0
    return {
        "score": dsr.score,
        "probability": dsr.probability,
        "trials": dsr.trials,
        "observed_sharpe": dsr.observed_sharpe,
        "status": "PASS" if ok else "FAIL",
    }


# ---------------------------------------------------------------------------
# L4: MC principal-preservation + block-shuffle MDD (base equity only -- no stress here, see L5).
# Local reimplementation of gates13/gates18/gates21/gates23's own _daily_returns/_simulate_mc/
# _blocks/_mdd/_block_shuffle (identical formulas, wave24's own numeric bars).
# ---------------------------------------------------------------------------


def _daily_returns(equity: pd.Series) -> tuple[tuple[pd.Timestamp, ...], np.ndarray]:
    clean = equity.dropna().astype(float)
    values = clean.to_numpy()
    if len(values) < 2 or not np.isfinite(values).all() or (values <= 0.0).any():
        raise ValueError("wave24_gp equity series must have >=2 finite, positive observations")
    returns = values[1:] / values[:-1] - 1.0
    if not np.isfinite(returns).all() or (returns <= -1.0).any():
        raise ValueError("wave24_gp equity returns contain invalid values")
    timestamps = tuple(pd.Timestamp(ts) for ts in clean.index[1:])
    return timestamps, returns


def _simulate_mc(returns: np.ndarray, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    finals = np.empty(MC_PATHS, dtype=float)
    chunk_size = 250
    for start in range(0, MC_PATHS, chunk_size):
        stop = min(start + chunk_size, MC_PATHS)
        samples = rng.choice(returns, size=(stop - start, returns.size), replace=True)
        growth = np.prod(1.0 + np.clip(samples, -0.999999, None), axis=1)
        finals[start:stop] = TOTAL_CAPITAL * RESERVE_FRACTION + ACTIVE_CAPITAL * growth
    return {
        "p05": float(np.quantile(finals, 0.05)),
        "ruin_probability": float(np.mean(finals < L4_RUIN_FLOOR_USDT)),
        "mean": float(np.mean(finals)),
        "median": float(np.median(finals)),
        "paths": MC_PATHS,
    }


def _blocks(timestamps: tuple[pd.Timestamp, ...], returns: np.ndarray) -> tuple[np.ndarray, ...]:
    anchor = timestamps[0]
    grouped: dict[int, list[float]] = {}
    for timestamp, value in zip(timestamps, returns):
        index = (timestamp - anchor).days // BLOCK_DAYS
        grouped.setdefault(index, []).append(float(value))
    return tuple(np.asarray(grouped[key], dtype=float) for key in sorted(grouped))


def _mdd_from_returns(values: np.ndarray) -> float:
    curve = TOTAL_CAPITAL * RESERVE_FRACTION + ACTIVE_CAPITAL * np.cumprod(1.0 + np.clip(values, -0.999999, None))
    peaks = np.maximum.accumulate(np.concatenate(([TOTAL_CAPITAL], curve)))
    return float(np.max(1.0 - curve / peaks[1:]))


def _block_shuffle(timestamps: tuple[pd.Timestamp, ...], returns: np.ndarray, seed: int) -> dict[str, float]:
    blocks = _blocks(timestamps, returns)
    rng = np.random.default_rng(seed)
    mdds = np.empty(BLOCK_PATHS, dtype=float)
    finals = np.empty(BLOCK_PATHS, dtype=float)
    for index in range(BLOCK_PATHS):
        path = np.concatenate([blocks[item] for item in rng.permutation(len(blocks))])
        mdds[index] = _mdd_from_returns(path)
        finals[index] = TOTAL_CAPITAL * RESERVE_FRACTION + ACTIVE_CAPITAL * np.prod(1.0 + np.clip(path, -0.999999, None))
    return {
        "block_days": BLOCK_DAYS,
        "block_count": len(blocks),
        "paths": BLOCK_PATHS,
        "mdd_p95": float(np.quantile(mdds, 0.95)),
        "final_p05": float(np.quantile(finals, 0.05)),
    }


def gate_l4_mc_and_block(equity: pd.Series) -> dict[str, Any]:
    try:
        timestamps, returns = _daily_returns(equity)
        mc = _simulate_mc(returns, SEED)
        p05_ok = mc["p05"] > L4_P05_FLOOR_USDT
        ruin_ok = mc["ruin_probability"] < L4_RUIN_PROBABILITY_MAX
        block = _block_shuffle(timestamps, returns, SEED + 103)
        block_mdd_ok = block["mdd_p95"] <= L4_BLOCK_MDD_P95_MAX
    except ValueError as error:
        mc = {"error": str(error)}
        p05_ok = False
        ruin_ok = False
        block = None
        block_mdd_ok = False

    ok = p05_ok and ruin_ok and block_mdd_ok
    return {
        "mc": mc,
        "mc_p05_ok": p05_ok,
        "mc_ruin_ok": ruin_ok,
        "block_mdd_p95": block["mdd_p95"] if block is not None else None,
        "block_mdd_ok": block_mdd_ok,
        "status": "PASS" if ok else "FAIL",
    }


# ---------------------------------------------------------------------------
# L5: executability (leg size, 1x gross, x3-stress sign preservation).
# ---------------------------------------------------------------------------


def gate_l5_executability(equity: pd.Series, stress_equity: pd.Series) -> dict[str, Any]:
    min_order_ok = LEG_USDT >= MIN_ORDER_USDT
    gross_ok = GROSS_USDT <= ACTIVE_CAPITAL + 1e-9  # same 1e-9 float-noise slack every prior wave's own gross-leverage check uses at this exact boundary

    try:
        clean = stress_equity.dropna()
        stress_sign_ok = len(clean) >= 2 and float(clean.iloc[-1]) > float(clean.iloc[0])
    except (TypeError, ValueError):
        stress_sign_ok = False

    ok = min_order_ok and gross_ok and stress_sign_ok
    return {
        "leg_usdt_nominal": LEG_USDT,
        "gross_usdt_nominal": GROSS_USDT,
        "min_order_usdt": MIN_ORDER_USDT,
        "min_order_feasible": min_order_ok,
        "gross_leverage_1x_ok": gross_ok,
        "stress_start_usdt": float(stress_equity.iloc[0]) if len(stress_equity) else None,
        "stress_end_usdt": float(stress_equity.iloc[-1]) if len(stress_equity) else None,
        "stress_sign_preserved": stress_sign_ok,
        "status": "PASS" if ok else "FAIL",
    }


# ---------------------------------------------------------------------------
# L6: spec-execution consistency with the LIVE paper tracker (inherits research.wave23_ga_short.
# gates23's own K6 -- see module docstring).
# ---------------------------------------------------------------------------


def gate_l6_paper_reproducibility(node: Node) -> dict[str, Any]:
    reasons: list[str] = []

    used_kinds = terminal_kinds_used(node)
    market_terminal_kinds = used_kinds - {"const"}
    unknown_kinds = market_terminal_kinds - set(_TERMINAL_DATA_FAMILY)
    data_ok = len(unknown_kinds) == 0
    if not data_ok:
        reasons.append(f"terminal kinds {sorted(unknown_kinds)} have no known live data family mapping")

    breadth_ok = FIXED_UNIVERSE_BREADTH <= PAPER_CARRY_UNIVERSE_CAP
    if not breadth_ok:
        reasons.append(
            f"universe_breadth={FIXED_UNIVERSE_BREADTH} (inherited unchanged from L4/I5's own backtest baseline) > "
            f"paper tracker's actual live-fetched carry universe (cap={PAPER_CARRY_UNIVERSE_CAP} = "
            f"CARRY_VOLUME_LIMIT({CARRY_VOLUME_LIMIT}) + BTC/ETH) -- same class of universe mismatch as the "
            "wave23_ga_short G1 incident this check is inherited from. This is a POSITION-STRUCTURE inheritance "
            "gap (L4/I5's own breadth choice), not something this wave's GP formula search could have avoided: "
            "wave24_gp deliberately holds universe_breadth fixed (task brief -- GP evolves the SIGNAL only)."
        )

    ok = data_ok and breadth_ok
    return {
        "terminal_kinds_used": sorted(market_terminal_kinds),
        "data_family_by_kind": {kind: _TERMINAL_DATA_FAMILY.get(kind) for kind in sorted(market_terminal_kinds)},
        "data_ok": data_ok,
        "universe_breadth": FIXED_UNIVERSE_BREADTH,
        "paper_carry_universe_cap": PAPER_CARRY_UNIVERSE_CAP,
        "universe_breadth_ok": breadth_ok,
        "reasons": reasons,
        "status": "PASS" if ok else "FAIL",
    }


# ---------------------------------------------------------------------------
# L7: formula simplicity (direct overfitting defense -- enforced strictly, task brief).
# ---------------------------------------------------------------------------


def gate_l7_simplicity(node: Node) -> dict[str, Any]:
    n_nodes = node_count(node)
    kinds = terminal_kinds_used(node)
    n_kinds = len(kinds)
    tree_depth = depth(node)
    node_ok = n_nodes <= L7_MAX_NODE_COUNT
    kinds_ok = n_kinds <= L7_MAX_TERMINAL_KINDS
    depth_ok = tree_depth <= MAX_DEPTH  # structural invariant, checked here too as defense-in-depth (never expected to fail given validate_tree gates every individual gp.py/random_trees.py ever produces)
    ok = node_ok and kinds_ok and depth_ok
    return {
        "node_count": n_nodes,
        "max_node_count": L7_MAX_NODE_COUNT,
        "node_count_ok": node_ok,
        "terminal_kinds_used": sorted(kinds),
        "n_terminal_kinds": n_kinds,
        "max_terminal_kinds": L7_MAX_TERMINAL_KINDS,
        "terminal_kinds_ok": kinds_ok,
        "depth": tree_depth,
        "max_depth": MAX_DEPTH,
        "depth_ok": depth_ok,
        "status": "PASS" if ok else "FAIL",
    }


# ---------------------------------------------------------------------------
# Overall promotion.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GateReport:
    l1: dict[str, Any]
    l2: dict[str, Any]
    l3: dict[str, Any]
    l4: dict[str, Any]
    l5: dict[str, Any]
    l6: dict[str, Any]
    l7: dict[str, Any]
    overall: str
    promoted: bool
    failure_reasons: tuple[str, ...]


def evaluate_all_gates(l1: dict[str, Any], l2: dict[str, Any], l3: dict[str, Any], l4: dict[str, Any], l5: dict[str, Any], l6: dict[str, Any], l7: dict[str, Any]) -> GateReport:
    gates = {"L1": l1, "L2": l2, "L3": l3, "L4": l4, "L5": l5, "L6": l6, "L7": l7}
    failure_reasons = tuple(name for name, gate in gates.items() if gate.get("status") != "PASS")
    overall = "PASS" if not failure_reasons else "FAIL"
    return GateReport(l1=l1, l2=l2, l3=l3, l4=l4, l5=l5, l6=l6, l7=l7, overall=overall, promoted=overall == "PASS", failure_reasons=failure_reasons)


def gate_report_payload(report: GateReport) -> dict[str, Any]:
    payload = asdict(report)
    payload["failure_reasons"] = list(report.failure_reasons)
    return payload


__all__ = [
    "ALL_TERMINAL_KINDS",
    "CUMULATIVE_TRIALS",
    "GP_TRIALS",
    "GROSS_USDT",
    "LEG_USDT",
    "L4_BLOCK_MDD_P95_MAX",
    "L4_P05_FLOOR_USDT",
    "L4_RUIN_PROBABILITY_MAX",
    "L7_MAX_NODE_COUNT",
    "L7_MAX_TERMINAL_KINDS",
    "PAPER_CARRY_UNIVERSE_CAP",
    "PRIOR_CUMULATIVE_TRIALS_BEFORE_WAVE24",
    "RANDOM_TRIALS",
    "THIS_WAVE_TOTAL_TRIALS",
    "GateReport",
    "evaluate_all_gates",
    "gate_l1_gp_beats_random",
    "gate_l2_beats_i5_oos",
    "gate_l3_dsr",
    "gate_l4_mc_and_block",
    "gate_l5_executability",
    "gate_l6_paper_reproducibility",
    "gate_l7_simplicity",
    "gate_report_payload",
]
