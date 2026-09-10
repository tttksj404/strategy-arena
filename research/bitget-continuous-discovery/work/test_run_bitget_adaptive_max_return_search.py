from __future__ import annotations

# pyright: reportImplicitRelativeImport=false
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TypeAlias

import numpy as np
import pytest

import bitget_adaptive_report
import run_bitget_adaptive_max_return_search as adaptive_runner
from alt_search_engine import evaluate_counterfactual_path
from alt_search_models import MarketData, PathResult, RiskOverlay
from bitget_adaptive_counterfactual import OlmarCounterfactual
from bitget_adaptive_report import JsonValue
from bitget_olmar import OlmarConfig
from bitget_path_survival import (
    SurvivalMetricError,
    SurvivalSpec,
    evaluate_signal_survival,
)
from run_bitget_adaptive_max_return_search import (
    AdaptiveSearchError,
    _liquidation_payload,
    _split_bounds,
    _survival_rows,
    select_validation_winner,
)

JsonMapping: TypeAlias = dict[str, JsonValue]


def _mapping(value: JsonValue) -> JsonMapping:
    assert isinstance(value, dict)
    return value


def _integer(value: JsonValue) -> int:
    assert isinstance(value, int) and not isinstance(value, bool)
    return value


def test_split_bounds_is_chronological_fifty_twentyfive_twentyfive() -> None:
    assert _split_bounds(100) == (50, 75)
    assert _split_bounds(4) == (2, 3)


def test_validation_winner_uses_fixed_notional_validation_only() -> None:
    scores = {
        "olmar-a": 0.20,
        "olmar-b": 0.30,
        "directional_change-a": 0.25,
    }
    eligible = {identifier: True for identifier in scores}
    winner = select_validation_winner(scores, eligible)
    assert winner == "olmar-b"


def test_validation_winner_rejects_nonfinite_and_ineligible() -> None:
    scores = {"olmar-a": np.nan, "directional_change-a": 0.1}
    eligible = {"olmar-a": True, "directional_change-a": False}
    assert select_validation_winner(scores, eligible) is None


def test_validation_winner_requires_matching_keys() -> None:
    with pytest.raises(AdaptiveSearchError, match="matching"):
        select_validation_winner({"olmar-a": 1.0}, {})


@pytest.mark.parametrize("rows", (0, 3))
def test_split_bounds_rejects_short_inputs_with_domain_error(rows: int) -> None:
    with pytest.raises(AdaptiveSearchError, match="four rows"):
        _split_bounds(rows)


def _ohlc_fixture(rows: int = 6) -> MarketData:
    close = np.full((rows, 1), 100.0)
    open_price = close.copy()
    high = close.copy()
    low = close.copy()
    dates = tuple(datetime(2024, 1, 1, tzinfo=UTC) + timedelta(minutes=5 * index) for index in range(rows))
    return MarketData(dates, ("ALT",), open_price, high, low, close, np.full((rows, 1), 1_000.0))


def test_liquidation_hits_recompute_each_market_bar_not_first_hit_only() -> None:
    data = _ohlc_fixture(4)
    low = data.low_price.copy()
    low[1, 0] = 80.0
    high = data.high_price.copy()
    high[3, 0] = 120.0
    data = MarketData(data.dates, data.symbols, data.open_price, high, low, data.close_price, data.quote_volume)
    positions = np.array([[0.0], [1.0], [0.0], [-1.0]])
    path = evaluate_counterfactual_path(data, positions, RiskOverlay(5.0, 0.20, 5))

    assert _liquidation_payload(path, 0, 4)["indices"] == [1, 3]
    assert _liquidation_payload(path, 1, 3)["indices"] == [1]
    assert _liquidation_payload(path, 2, 4)["indices"] == [3]


def test_survival_rows_propagates_invalid_position_data_instead_of_swallowing() -> None:
    data = _ohlc_fixture()
    invalid_positions = np.zeros((len(data.dates) - 1, 1), dtype=float)

    with pytest.raises(SurvivalMetricError, match="identical"):
        _survival_rows(data, invalid_positions, 1, len(data.dates), bar_minutes=5)


def test_survival_rows_keeps_no_entry_segment_empty() -> None:
    data = _ohlc_fixture(10)
    positions = np.zeros((10, 1), dtype=float)

    assert _survival_rows(data, positions, 1, 10, bar_minutes=5) == []


def test_survival_rows_does_not_swallow_generic_same_message_error(monkeypatch: pytest.MonkeyPatch) -> None:
    data = _ohlc_fixture(10)
    positions = np.zeros((10, 1), dtype=float)

    def raise_generic_error(*args: object, **kwargs: object) -> object:
        raise SurvivalMetricError("no entry events fit the requested segment and horizon")

    monkeypatch.setattr(bitget_adaptive_report, "evaluate_signal_survival", raise_generic_error)
    with pytest.raises(SurvivalMetricError, match="no entry events"):
        bitget_adaptive_report._survival_at_cost(
            data,
            positions,
            SurvivalSpec(1, 10, 3, 5.0, 0.0018, 0.02),
        )


def test_survival_rows_reports_base_and_stress_costs_separately() -> None:
    data = _ohlc_fixture(10)
    positions = np.zeros((10, 1), dtype=float)
    positions[1:, 0] = 1.0

    rows = _survival_rows(data, positions, 1, 10, bar_minutes=5)

    assert rows
    assert isinstance(rows[0], dict)
    assert set(rows[0]) >= {"horizon_bars", "base", "stress"}


def test_survival_engine_rejects_invalid_configuration_fail_closed() -> None:
    data = _ohlc_fixture(10)
    positions = np.zeros((10, 1), dtype=float)

    with pytest.raises(SurvivalMetricError, match="leverage must be positive"):
        evaluate_signal_survival(data, positions, SurvivalSpec(1, 10, 3, 0.0, 0.0018, 0.02))


@pytest.mark.parametrize(("interval", "bar_minutes"), (("5m", 5), ("15m", 15)))
def test_run_v2_report_contract_with_stubbed_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    interval: str,
    bar_minutes: int,
) -> None:
    rows = 64
    symbols = ("BTC", *adaptive_runner.ALT_SYMBOLS)
    prices = np.full((rows, len(symbols)), 100.0)
    dates = tuple(datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=bar_minutes * index) for index in range(rows))
    data = MarketData(dates, symbols, prices.copy(), prices.copy(), prices.copy(), prices.copy(), prices.copy())
    metadata = type(
        "Metadata",
        (),
        {
            "interval": interval,
            "snapshot_ms": 1_700_000_000_000,
            "common_rows": rows,
            "integrity_verified": True,
            "source_hashes": ("stub",),
        },
    )()
    configs = tuple(
        OlmarCounterfactual(
            OlmarConfig(2, 0.01, 0.1, 1, 0.0),
            target_bar_volatility=0.001 + index * 0.00001,
            risk_window=5,
        )
        for index in range(48)
    )
    path_returns = np.linspace(-0.01, 0.02, rows, dtype=float)
    fake_path = PathResult(path_returns, path_returns * 0.9, np.ones(rows), 0, -1)
    positions = np.zeros((rows, len(symbols)))
    positions[32:48, 0] = 1.0
    positions[49:, 0] = 1.0

    monkeypatch.setattr(adaptive_runner, "spec_for_interval", lambda interval: type("Spec", (), {"interval": interval, "bars_per_year": 365.0})())
    monkeypatch.setattr(adaptive_runner, "load_verified_interval_archive", lambda *args, **kwargs: (data, metadata))
    monkeypatch.setattr(adaptive_runner, "adaptive_configs", lambda spec: configs)
    monkeypatch.setattr(adaptive_runner, "positions_for", lambda *args, **kwargs: positions.copy())
    monkeypatch.setattr(adaptive_runner, "evaluate_counterfactual_path", lambda *args, **kwargs: fake_path)

    report = adaptive_runner.run(Path("stub-archive"), interval=interval)

    assert report["schema_version"] == "adaptive-max-return-v2"
    assert report["strategy"] == f"{interval} adaptive OLMAR plus directional-change fixed-notional maximum-return search"
    manifest = _mapping(report["predeclared_search"])["candidate_manifest"]
    assert isinstance(manifest, list)
    assert len(manifest) == 48
    assert len({str(_mapping(row)["id"]) for row in manifest if isinstance(row, dict)}) == 48
    split = _mapping(_mapping(report["data"])["split"])
    assert [_integer(_mapping(split[name])["rows"]) for name in ("train", "validation", "locked_test")] == [32, 16, 16]
    assert all("T" in str(_mapping(split[name])["start"]) and "T" in str(_mapping(split[name])["end"]) for name in split)
    fixed_notional = _mapping(_mapping(report["locked_test"])["fixed_notional"])
    assert set(fixed_notional) == {"base", "stress"}
    survival = _mapping(_mapping(report["locked_test"])["survival"])
    validation_survival = survival["validation"]
    assert isinstance(validation_survival, list)
    assert _mapping(validation_survival[0])["horizon_minutes"] == 3 * bar_minutes
    selection_bias = _mapping(report["selection_bias"])
    assert selection_bias["multiplicity_scope"] == "adaptive_run_local_48"
    assert _mapping(selection_bias["global_search_scope"])["global_dsr_computed"] is False
    assert _mapping(report["locked_test"])["id"] == report["selected_validation_winner"]
    assert report["live_promotable_ids"] == []
