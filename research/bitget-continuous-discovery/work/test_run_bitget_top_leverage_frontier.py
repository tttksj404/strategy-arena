from __future__ import annotations

import hashlib
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import work.run_bitget_top_leverage_frontier as frontier
from work.alt_search_models import Candidate, FloatMatrix, MarketData
from work.alt_search_signals import build_positions as real_build_positions
from work.bitget_archive_daily import ArchiveDailyError
from work.run_bitget_top_leverage_frontier import (
    EvaluationCounter,
    FixedPath,
    contract_payload,
    json_text,
    leverage_grid,
    select_validation_winner,
    simulate_fixed_leverage,
)
from work.run_bitget_top_strategy_diagnostic import EXPECTED_SELECTION_SOURCE_SHA256


def _data(close: list[float], low: list[float] | None = None) -> MarketData:
    values = np.asarray(close, dtype=float)[:, None]
    lows = values.copy() if low is None else np.asarray(low, dtype=float)[:, None]
    return MarketData(
        dates=tuple(datetime(2024, 1, i + 1, tzinfo=UTC) for i in range(len(close))),
        symbols=("ALT",),
        open_price=values.copy(),
        high_price=values.copy(),
        low_price=lows,
        close_price=values,
        quote_volume=np.full_like(values, 1_000_000.0),
    )


def test_grid_is_quarter_x_and_includes_both_bounds() -> None:
    grid = leverage_grid()

    assert grid[0] == 0.25
    assert grid[-1] == 5.0
    assert len(grid) == 20
    assert all(round(grid[i + 1] - grid[i], 8) == 0.25 for i in range(len(grid) - 1))


def test_fixed_leverage_marks_intrabar_proxy_hit_from_prior_close() -> None:
    data = _data([100.0, 100.0, 101.0], low=[100.0, 75.0, 101.0])
    positions = np.asarray([[0.0], [1.0], [1.0]])

    result = simulate_fixed_leverage(data, positions, 5.0)

    assert isinstance(result, FixedPath)
    assert result.liquidation_hits == (1,)
    assert result.base_returns[1] == np.float64(-0.0045)
    assert result.continues_after_first_liquidation is True


def test_fixed_leverage_accumulates_adverse_excursion_from_episode_entry() -> None:
    data = _data([100.0, 85.0, 72.25, 72.25], low=[100.0, 85.0, 72.25, 72.25])
    positions = np.asarray([[0.0], [1.0], [1.0], [1.0]])

    result = simulate_fixed_leverage(data, positions, 5.0)

    assert result.liquidation_hits == (2,)


def test_fixed_leverage_reprices_weighted_same_direction_addition() -> None:
    data = _data([100.0, 120.0, 120.0], low=[100.0, 88.0, 110.0])
    positions = np.asarray([[0.0], [0.5], [1.0]])

    result = simulate_fixed_leverage(data, positions, 5.0)

    # The earlier .5x exposure's low=88 must not be replayed onto the
    # 1.0x increase at row 2; only row 2's low=110 is causal for the increase.
    assert result.liquidation_hits == ()


def test_fixed_leverage_detects_causal_drawdown_after_same_direction_addition() -> None:
    data = _data([100.0, 120.0, 120.0], low=[100.0, 88.0, 90.0])
    positions = np.asarray([[0.0], [0.5], [1.0]])

    result = simulate_fixed_leverage(data, positions, 5.0)

    # Weighted entry is 110 at row 2; low=90 on that same increase bar is a
    # genuine 18.18% adverse move and must trigger the 5x, 18% proxy.
    assert result.liquidation_hits == (2,)


def test_validation_selection_uses_only_zero_hit_stress_maximum() -> None:
    rows = (
        {"strategy": "donchian", "leverage": 1.0, "validation_stress_return": 0.2, "validation_liquidation_hits": 0, "development_liquidation_hits": 0},
        {"strategy": "pv", "leverage": 5.0, "validation_stress_return": 0.9, "validation_liquidation_hits": 0, "development_liquidation_hits": 1},
        {"strategy": "pv", "leverage": 2.0, "validation_stress_return": 0.3, "validation_liquidation_hits": 0, "development_liquidation_hits": 0},
    )

    selected = select_validation_winner(rows)

    assert selected["strategy"] == "pv"
    assert selected["leverage"] == 2.0


def test_locked_test_must_be_evaluated_once_by_report_contract() -> None:
    data = _data([100.0, 101.0, 102.0, 103.0])
    positions = np.ones((4, 1))
    result = simulate_fixed_leverage(data, positions, 1.0)

    assert result.evaluations == 1


def test_report_contract_preserves_prior_strategy_selection_and_fail_closed_status() -> None:
    contract = contract_payload()

    assert contract["source_strategy_selection_independent"] is False
    assert contract["source_strategy_search_trials"] == 139_536
    assert contract["prior_global_pbo"] == 0.4142857142857143
    assert contract["prior_global_pbo_threshold"] == 0.05
    assert contract["prior_global_pbo_pass"] is False
    assert contract["prior_donchian_dsr_probability"] == 0.014680027865062376
    assert contract["prior_donchian_dsr_threshold"] == 0.95
    assert contract["prior_donchian_dsr_pass"] is False
    assert "full train+validation" in json_text(contract["selection_development_scope"])
    assert contract["selected_strategy_paper_pass"] is False
    assert contract["selected_strategy_live_promotable"] is False
    assert frontier.json_map(contract["local_pbo_dsr"])["status"] == "not_calculated"
    evidence = frontier.json_map(contract["prior_evidence_source"])
    assert evidence["sha256_verified"] is True
    actual_hash = hashlib.sha256(frontier.PRIOR_PBO_REPORT.read_bytes()).hexdigest()
    assert frontier.json_text(evidence["sha256"]) == EXPECTED_SELECTION_SOURCE_SHA256 == actual_hash
    assert evidence["dsr_field"] == "locked_test_results[*].deflated_sharpe_stress.probability"
    assert contract["fixed_notional_pnl"] is False
    assert contract["exact_bitget_account_return"] is False


def test_selection_source_hash_mismatch_or_missing_is_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(ArchiveDailyError):
        _ = frontier.contract_payload(tmp_path / "missing-selection-report.json")

    mismatched = tmp_path / "mismatched-selection-report.json"
    _ = mismatched.write_text("{}", encoding="utf-8")
    with pytest.raises(ArchiveDailyError):
        _ = frontier.contract_payload(mismatched)


def test_validation_frontier_instrumentation_runs_40_prefix_evaluations(monkeypatch: pytest.MonkeyPatch) -> None:
    data = _data([100.0, 101.0, 102.0, 103.0])
    counter = EvaluationCounter()
    observed_lengths: list[int] = []
    original = real_build_positions

    def spy(input_data: MarketData, candidate: Candidate) -> FloatMatrix:
        observed_lengths.append(len(input_data.dates))
        return original(input_data, candidate)

    monkeypatch.setattr(frontier, "build_positions", spy)

    rows = frontier.candidate_rows(data, 1, 3, counter)

    assert len(rows) == 40
    assert counter.validation_evaluations == 40
    assert counter.locked_test_evaluations == 0
    assert observed_lengths == [3, 3]
