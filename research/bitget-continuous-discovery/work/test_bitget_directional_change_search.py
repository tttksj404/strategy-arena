from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

import run_bitget_directional_change_search as directional_change_search
from alt_search_models import MarketData, PathResult, RiskOverlay
from bitget_archive_panel import ArchivePanelMetadata
from bitget_directional_change import DirectionalChangeConfig
from run_alt_strategy_search import SYMBOLS as ALT_SYMBOLS
from run_bitget_4h_pair_regime_search import spec_for_interval
from run_bitget_directional_change_search import SearchConfig, _search_configs
from selection_bias_stats import PboResult

type FloatMatrix = npt.NDArray[np.float64]
type FloatVector = npt.NDArray[np.float64]
type CacheRunObservations = tuple[list[str], list[FloatMatrix], FloatMatrix, tuple[SearchConfig, ...], float]


def _run_with_position_spies(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> CacheRunObservations:
    configs = _search_configs(spec_for_interval("4H"))
    volatility_caps = tuple(dict.fromkeys(config.btc_volatility_cap for config in configs))
    assert len(volatility_caps) == 2
    restrictive_cap, permissive_cap = volatility_caps
    rows = 16
    symbols = ("BTC", *ALT_SYMBOLS)
    dates = tuple(datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=index) for index in range(rows))
    price = np.ones((rows, len(symbols)), dtype=np.float64)
    data = MarketData(dates, symbols, price, price, price, price, price)
    metadata = ArchivePanelMetadata(0, "4H", rows, True, ())
    raw_positions = np.zeros((rows, len(symbols)), dtype=np.float64)
    raw_positions[:, 1] = 1.0
    position_calls: list[str] = []
    observed_positions: list[FloatMatrix] = []

    def fake_load_verified_interval_archive(
        archive_root: Path,
        assets: tuple[str, ...],
        *,
        interval: str,
    ) -> tuple[MarketData, ArchivePanelMetadata]:
        assert archive_root == tmp_path
        assert assets == symbols
        assert interval == "4H"
        return data, metadata

    def fake_btc_regime_mask(
        close_price: FloatVector,
        *,
        trend_bars: int,
        volatility_bars: int,
        trend_limit: float,
        volatility_cap: float,
    ) -> npt.NDArray[np.bool_]:
        assert close_price.shape == (rows,)
        assert trend_bars > 0
        assert volatility_bars > 0
        assert trend_limit > 0.0
        mask = np.ones(rows, dtype=np.bool_)
        if volatility_cap == restrictive_cap:
            mask[0] = False
        else:
            assert volatility_cap == permissive_cap
        return mask

    def fake_walkforward_positions(
        market: MarketData,
        model: DirectionalChangeConfig,
        *,
        alt_symbols: tuple[str, ...],
    ) -> FloatMatrix:
        assert market is data
        assert alt_symbols == ALT_SYMBOLS
        position_calls.append(model.identifier)
        return raw_positions

    def fake_evaluate_path(market: MarketData, positions: FloatMatrix, _overlay: RiskOverlay) -> PathResult:
        assert market is data
        observed_positions.append(positions.copy())
        returns = np.zeros(rows, dtype=np.float64)
        return PathResult(returns, returns, returns, 0, rows)

    def fake_validation_record(
        _path: PathResult,
        config: SearchConfig,
        *,
        train_end: int,
        development_end: int,
        bars_per_year: float,
    ) -> directional_change_search.ValidationRecord:
        assert train_end < development_end
        assert bars_per_year > 0.0
        return directional_change_search.ValidationRecord(config, float("-inf"))

    def fake_development_moments(
        returns: FloatVector,
        *,
        development_end: int,
        bins: int,
    ) -> tuple[FloatVector, FloatVector, tuple[int, ...]]:
        assert returns.shape == (rows,)
        assert development_end < rows
        assert bins > 0
        return np.asarray([1.0, 1.0]), np.asarray([1.0, 1.0]), (1, 1)

    def fake_cscv_pbo(
        sums: FloatMatrix,
        squares: FloatMatrix,
        counts: tuple[int, ...],
    ) -> PboResult:
        assert sums.shape == (len(configs), 2)
        assert squares.shape == sums.shape
        assert counts == (1, 1)
        return PboResult(1.0, 1, len(configs), 1.0)

    monkeypatch.setattr(directional_change_search, "load_verified_interval_archive", fake_load_verified_interval_archive)
    monkeypatch.setattr(directional_change_search, "btc_regime_mask", fake_btc_regime_mask)
    monkeypatch.setattr(directional_change_search, "walkforward_directional_change_positions", fake_walkforward_positions)
    monkeypatch.setattr(directional_change_search, "evaluate_path", fake_evaluate_path)
    monkeypatch.setattr(directional_change_search, "_validation_record", fake_validation_record)
    monkeypatch.setattr(directional_change_search, "development_moments", fake_development_moments)
    monkeypatch.setattr(directional_change_search, "cscv_pbo_from_moments", fake_cscv_pbo)

    report = directional_change_search.run(tmp_path)

    assert report["selected_validation_winners"] == []
    return position_calls, observed_positions, raw_positions, configs, permissive_cap


def test_directional_change_grid_spans_event_threshold_and_hold_horizons() -> None:
    configs = _search_configs(spec_for_interval("4H"))

    assert len(configs) == 48
    assert {config.model.threshold for config in configs} == {0.0025, 0.005, 0.01}
    assert {config.model.follow_trend for config in configs} == {False, True}
    assert {config.model.rebalance_bars for config in configs} == {3, 6}


def test_directional_change_search_calculates_positions_once_per_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    position_calls, _observed_positions, _raw_positions, configs, _permissive_cap = _run_with_position_spies(monkeypatch, tmp_path)

    model_identifiers = {config.model.identifier for config in configs}

    assert len(position_calls) == len(model_identifiers)
    assert len(position_calls) == len(set(position_calls))
    assert set(position_calls) == model_identifiers


def test_directional_change_search_keeps_cached_positions_unmodified_by_regime_masks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _position_calls, observed_positions, raw_positions, configs, permissive_cap = _run_with_position_spies(monkeypatch, tmp_path)
    first_model = configs[0].model.identifier
    permissive_index = next(
        index
        for index, config in enumerate(configs)
        if config.model.identifier == first_model and config.btc_volatility_cap == permissive_cap
    )

    assert observed_positions[0][0, 1] == 0.0
    assert observed_positions[permissive_index][0, 1] == 1.0
    assert raw_positions[0, 1] == 1.0
