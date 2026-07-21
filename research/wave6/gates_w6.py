# Wave-6 gate orchestration. Standard candidates (W6a-d) reuse the wave-1 19-gate table via
# wave-2's UNTESTED_IN_OOS wrapper; exploratory candidates (W6e/W6f) get an effect-stats-only
# summary (direction, t-stat, cost-after sign) with no deployment claim, per SPEC.md. Any
# standard survivor (all 19 gates PASS) is additionally checked against W2c using the exact
# wave-5 combination-gate criteria.

from __future__ import annotations

import math
from pathlib import Path
import sys
from typing import Final

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

import pandas as pd  # noqa: PANDAS_OK

from research.wave1.common import JsonValue, PipelineError, load_frame, load_json, save_json
from research.wave1.gate_reporting import _series
from research.wave1.gates import GateRow
from research.wave2.gates import evaluate_result_file_wave2
from research.wave5.engine import combine_returns, equity_from_returns
from research.wave5.gates import combination_gates
from research.wave5.strategies import load_symbols, run_w2c
from research.wave6.engine_w6 import CACHE_DIR, RESULTS_DIR, WAVE1_CACHE_DIR
from research.wave6.strategies_w6 import EXPLORATORY_IDS, STANDARD_IDS


NEXT_AXIS_NOTE: Final = "order_book_depth_forward_collection"


def run_standard_gates(results_dir: Path = RESULTS_DIR, wave6_cache_dir: Path = CACHE_DIR) -> dict[str, tuple[GateRow, ...]]:
    btc_path = wave6_cache_dir / "binance_fapi_BTCUSDT_1h.csv.gz"
    btc_returns = load_frame(btc_path)["close"].pct_change() if btc_path.exists() else pd.Series(dtype=float)
    rows: dict[str, tuple[GateRow, ...]] = {}
    for candidate_id in STANDARD_IDS:
        path = results_dir / f"{candidate_id}.json"
        if not path.exists():
            raise FileNotFoundError(path)
        rows[candidate_id] = evaluate_result_file_wave2(path, btc_returns)
    return rows


def verdict_label(results_dir: Path, candidate_id: str, rows: tuple[GateRow, ...]) -> str:
    payload = load_json(results_dir / f"{candidate_id}.json")
    if isinstance(payload, dict):
        validation = payload.get("validation")
        if isinstance(validation, dict) and isinstance(validation.get("oos_label"), str):
            return str(validation["oos_label"])
    return "PASS" if rows and all(row.status == "PASS" for row in rows) else "FAIL"


def exploratory_summary(results_dir: Path, candidate_id: str) -> dict[str, JsonValue]:
    path = results_dir / f"{candidate_id}.json"
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise PipelineError(f"invalid candidate payload: {path.name}")
    metadata_value = payload.get("metadata")
    metadata = metadata_value if isinstance(metadata_value, dict) else {}
    sample_size = metadata.get("sample_size")
    cost_after = metadata.get("effect_cost_after_mean")
    direction = metadata.get("effect_direction", "undetermined")
    # A candidate only has trustworthy effect stats once strategies_w6 actually computed them
    # (effect_cost_after_mean present + finite + at least one observation); W6f's n<40 branch and
    # W6e's empty-alignment branch deliberately omit these fields, which is the UNDETERMINED signal.
    has_effect_stats = isinstance(cost_after, (int, float)) and math.isfinite(cost_after) and isinstance(sample_size, int) and sample_size > 0
    if not has_effect_stats:
        summary: dict[str, JsonValue] = {
            "candidate_id": candidate_id,
            "verdict": "UNDETERMINED",
            "sample_size": sample_size if isinstance(sample_size, int) else 0,
            "reason": metadata.get("reason", "insufficient sample"),
            "deployment_claim": False,
        }
    else:
        summary = {
            "candidate_id": candidate_id,
            "verdict": "EFFECT_POSITIVE_COST_AFTER" if cost_after > 0.0 else "EFFECT_NEGATIVE_OR_ZERO_COST_AFTER",
            "sample_size": sample_size,
            "direction": direction,
            "t_stat": metadata.get("effect_t_stat"),
            "cost_after_mean": cost_after,
            "deployment_claim": False,
        }
    payload["exploratory_summary"] = summary
    save_json(path, payload)
    return summary


def survivors(results_dir: Path, rows_by_id: dict[str, tuple[GateRow, ...]]) -> tuple[str, ...]:
    return tuple(candidate_id for candidate_id in STANDARD_IDS if verdict_label(results_dir, candidate_id, rows_by_id[candidate_id]) == "PASS")


def build_combination(results_dir: Path, wave1_cache_dir: Path, survivor_id: str) -> dict[str, JsonValue]:
    """Combine a surviving standard candidate with W2c using the wave-5 combination criteria."""
    symbols = load_symbols(wave1_cache_dir)
    baseline = run_w2c(wave1_cache_dir, symbols)
    candidate_payload = load_json(results_dir / f"{survivor_id}.json")
    if not isinstance(candidate_payload, dict):
        raise PipelineError(f"invalid candidate payload: {survivor_id}")
    candidate_equity = _series(candidate_payload.get("equity"))
    candidate_returns = candidate_equity.pct_change().fillna(0.0).reindex(baseline.equity.index).fillna(0.0)
    baseline_returns = baseline.equity.pct_change().fillna(0.0)
    combined_returns = combine_returns(baseline_returns, candidate_returns, 0.5)
    combined_equity = equity_from_returns(combined_returns)
    combination = combination_gates(baseline.equity, candidate_equity, combined_equity)
    payload: dict[str, JsonValue] = {**combination, "survivor": survivor_id}
    save_json(results_dir / "W6_combination.json", payload)
    return payload


def run_gates() -> dict[str, JsonValue]:
    rows_by_id = run_standard_gates(RESULTS_DIR, CACHE_DIR)
    standard_verdicts = {candidate_id: verdict_label(RESULTS_DIR, candidate_id, rows_by_id[candidate_id]) for candidate_id in STANDARD_IDS}
    exploratory = {candidate_id: exploratory_summary(RESULTS_DIR, candidate_id) for candidate_id in EXPLORATORY_IDS}
    winners = survivors(RESULTS_DIR, rows_by_id)
    combination: dict[str, JsonValue] | None = None
    if winners:
        combination = build_combination(RESULTS_DIR, WAVE1_CACHE_DIR, winners[0])
    else:
        (RESULTS_DIR / "W6_combination.json").unlink(missing_ok=True)
    summary: dict[str, JsonValue] = {
        "standard_verdicts": standard_verdicts,
        "survivors": list(winners),
        "combination": combination,
        "exploratory": exploratory,
        "no_edge_next_axis": None if winners else NEXT_AXIS_NOTE,
    }
    save_json(RESULTS_DIR / "gates_summary_w6.json", summary)
    return summary


__all__ = [
    "build_combination",
    "exploratory_summary",
    "run_gates",
    "run_standard_gates",
    "survivors",
    "verdict_label",
]
