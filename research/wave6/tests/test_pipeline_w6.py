# Integration-style tests for the wave-6 exploratory gate summary and the (currently dead, but
# must stay correct) new-listing aggregation path, using synthetic data and temp directories only
# -- no network access and no dependency on the real cache.

from __future__ import annotations

import pandas as pd  # noqa: PANDAS_OK

from research.wave1.common import save_json
from research.wave1.gates import GateRow
from research.wave6.gates_w6 import exploratory_summary, verdict_label
from research.wave6.strategies_w6 import _aggregate_listing_trades


def test_exploratory_summary_undetermined_below_min_sample(tmp_path) -> None:
    # Given
    save_json(tmp_path / "W6f.json", {"candidate_id": "W6f", "family": "F6", "metadata": {"sample_size": 3, "reason": "too few listings"}})

    # When
    summary = exploratory_summary(tmp_path, "W6f")

    # Then
    assert summary["verdict"] == "UNDETERMINED"
    assert summary["sample_size"] == 3
    assert summary["deployment_claim"] is False


def test_exploratory_summary_flags_positive_cost_after(tmp_path) -> None:
    # Given
    save_json(
        tmp_path / "W6e.json",
        {
            "candidate_id": "W6e",
            "family": "F6",
            "metadata": {"sample_size": 50, "effect_cost_after_mean": 0.001, "effect_direction": "positive", "effect_t_stat": 2.1},
        },
    )

    # When
    summary = exploratory_summary(tmp_path, "W6e")

    # Then
    assert summary["verdict"] == "EFFECT_POSITIVE_COST_AFTER"
    assert summary["deployment_claim"] is False


def test_exploratory_summary_flags_negative_cost_after(tmp_path) -> None:
    # Given
    save_json(
        tmp_path / "W6e.json",
        {
            "candidate_id": "W6e",
            "family": "F6",
            "metadata": {"sample_size": 50, "effect_cost_after_mean": -0.001, "effect_direction": "negative", "effect_t_stat": -1.4},
        },
    )

    # When
    summary = exploratory_summary(tmp_path, "W6e")

    # Then
    assert summary["verdict"] == "EFFECT_NEGATIVE_OR_ZERO_COST_AFTER"


def test_verdict_label_prefers_untested_in_oos_label(tmp_path) -> None:
    # Given: wave-2's UNTESTED_IN_OOS wrapper stamps validation.oos_label when OOS has zero trades.
    save_json(tmp_path / "W6a.json", {"validation": {"oos_label": "UNTESTED_IN_OOS"}})
    rows = (GateRow(1, "data_validation", "PASS", "True"),)

    # When / Then
    assert verdict_label(tmp_path, "W6a", rows) == "UNTESTED_IN_OOS"


def test_verdict_label_pass_requires_every_gate() -> None:
    rows_all_pass = tuple(GateRow(gate, "g", "PASS", "") for gate in range(1, 20))
    rows_one_fail = rows_all_pass[:-1] + (GateRow(19, "factor_exposure", "FAIL", ""),)
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
        results_dir = Path(directory)
        save_json(results_dir / "W6b.json", {"validation": {}})
        assert verdict_label(results_dir, "W6b", rows_all_pass) == "PASS"
        assert verdict_label(results_dir, "W6b", rows_one_fail) == "FAIL"


def test_aggregate_listing_trades_short_profits_from_clean_decline() -> None:
    # Given: 45 synthetic listings (>= the 40-sample minimum) that each decline 10% between D+2
    # open and D+7 close -- this exercises the code path that stays dormant while Bitget's
    # launchTime field is empty in production.
    eligible = tuple((f"SYM{i}USDT", pd.Timestamp("2026-01-01T00:00:00Z") + pd.Timedelta(days=i)) for i in range(45))

    def fake_loader(symbol: str, onboard: pd.Timestamp) -> pd.DataFrame:
        dates = pd.date_range(onboard, periods=10, freq="1D")
        return pd.DataFrame({"open": [10.0] * 10, "close": [9.0] * 10}, index=dates)

    # When
    trades, used_symbols = _aggregate_listing_trades(eligible, fake_loader)

    # Then: net of the (small) round-trip cost, every trade is still profitable.
    assert len(used_symbols) == 45
    assert len(trades) == 45
    assert (trades > 0.0).all()


def test_aggregate_listing_trades_skips_symbols_with_no_data() -> None:
    # Given
    eligible = (("MISSINGUSDT", pd.Timestamp("2026-01-01T00:00:00Z")),)

    def empty_loader(symbol: str, onboard: pd.Timestamp) -> pd.DataFrame | None:
        return None

    # When
    trades, used_symbols = _aggregate_listing_trades(eligible, empty_loader)

    # Then
    assert used_symbols == []
    assert len(trades) == 0
