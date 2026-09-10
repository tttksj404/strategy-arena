# pyright: reportImplicitRelativeImport=false

from __future__ import annotations

import csv
import gzip
import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from alt_search_data import CacheIntegrityError, load_frozen_daily_cache
from alt_search_engine import (
    evaluate_counterfactual_path,
    evaluate_path,
    segment_metrics,
    select_validation_winners,
)
from alt_search_models import Candidate, Family, MarketData, RiskOverlay, SegmentMetrics
from alt_search_signals import build_positions
from run_alt_strategy_search import (
    ValidationRecord,
    _select_winners,
    enumerate_candidates,
    enumerate_risk_overlays,
)


def _data(*, rows: int = 80, symbols: tuple[str, ...] = ("ETH", "SOL", "LINK")) -> MarketData:
    dates = tuple(datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=index) for index in range(rows))
    trend = np.arange(rows, dtype=float)[:, None]
    offsets = np.arange(len(symbols), dtype=float)[None, :]
    close = 100.0 + trend + offsets
    return MarketData(
        dates=dates,
        symbols=symbols,
        open_price=close.copy(),
        high_price=close * 1.01,
        low_price=close * 0.99,
        close_price=close,
        quote_volume=np.full_like(close, 10_000_000.0),
    )


def test_cross_sectional_positions_use_only_prior_closes() -> None:
    data = _data()
    candidate = Candidate(Family.CROSS_SECTIONAL, lookback=5, secondary=0, top_k=1, direction=1, pair=None, entry_z=0.0)

    first = build_positions(data, candidate)
    changed = MarketData(
        dates=data.dates,
        symbols=data.symbols,
        open_price=data.open_price.copy(),
        high_price=data.high_price.copy(),
        low_price=data.low_price.copy(),
        close_price=data.close_price.copy(),
        quote_volume=data.quote_volume.copy(),
    )
    changed.close_price[-1, :] *= 100.0
    second = build_positions(changed, candidate)

    assert np.allclose(first[:-1], second[:-1])


def test_bollinger_entry_threshold_blocks_unqualified_positions() -> None:
    data = _data(rows=30)
    candidate = Candidate(Family.BOLLINGER, lookback=5, secondary=0, top_k=1, direction=-1, pair=None, entry_z=100.0)

    positions = build_positions(data, candidate)

    assert not positions.any()


def test_liquidity_floor_uses_only_prior_closed_quote_volume() -> None:
    data = _data(rows=40)
    data.quote_volume[20, 0] = 1.0
    candidate = Candidate(Family.TIME_SERIES, lookback=5, secondary=0, top_k=0, direction=1, pair=None, entry_z=0.0, minimum_daily_quote_volume=5_000_000.0)

    positions = build_positions(data, candidate)
    changed = MarketData(
        dates=data.dates,
        symbols=data.symbols,
        open_price=data.open_price.copy(),
        high_price=data.high_price.copy(),
        low_price=data.low_price.copy(),
        close_price=data.close_price.copy(),
        quote_volume=data.quote_volume.copy(),
    )
    changed.quote_volume[21, 0] = 1.0

    assert positions[21, 0] == 0.0
    assert np.allclose(positions[:22], build_positions(changed, candidate)[:22])


def test_price_volume_signal_skips_zero_volume_history_without_invalid_division() -> None:
    data = _data(rows=40)
    data.quote_volume[:10, 0] = 0.0
    candidate = Candidate(Family.PRICE_VOLUME, lookback=5, secondary=0, top_k=1, direction=1, pair=None, entry_z=0.0)

    with np.errstate(divide="raise", invalid="raise"):
        positions = build_positions(data, candidate)

    assert np.isfinite(positions).all()


def test_five_x_long_path_is_rejected_when_intrabar_move_hits_liquidation() -> None:
    data = _data(rows=60, symbols=("ETH", "SOL"))
    data.low_price[30, 0] = data.close_price[29, 0] * 0.70
    candidate = Candidate(Family.TIME_SERIES, lookback=3, secondary=0, top_k=0, direction=1, pair=None, entry_z=0.0)
    risk = RiskOverlay(max_leverage=5.0, target_daily_vol=0.20, volatility_window=5)

    result = evaluate_path(data, build_positions(data, candidate), risk)

    assert result.liquidations >= 1
    assert result.returns[result.first_liquidation_index] == -1.0


def test_counterfactual_path_records_liquidation_without_stopping_return_path() -> None:
    data = _data(rows=60, symbols=("ETH", "SOL"))
    data.low_price[30, 0] = data.close_price[29, 0] * 0.70
    candidate = Candidate(Family.TIME_SERIES, lookback=3, secondary=0, top_k=0, direction=1, pair=None, entry_z=0.0)
    risk = RiskOverlay(max_leverage=5.0, target_daily_vol=0.20, volatility_window=5)

    result = evaluate_counterfactual_path(data, build_positions(data, candidate), risk)

    assert result.liquidations >= 1
    assert result.returns[result.first_liquidation_index] != -1.0
    assert np.any(result.returns[result.first_liquidation_index + 1 :] != 0.0)


def test_path_result_preserves_all_market_ohlc_liquidation_hit_indices() -> None:
    data = _data(rows=5, symbols=("ETH",))
    data.low_price[1, 0] = data.close_price[0, 0] * 0.80
    data.low_price[3, 0] = data.close_price[2, 0] * 0.80
    positions = np.ones((5, 1), dtype=float)
    risk = RiskOverlay(max_leverage=5.0, target_daily_vol=0.20, volatility_window=5)

    stopped = evaluate_path(data, positions, risk)
    counterfactual = evaluate_counterfactual_path(data, positions, risk)

    assert stopped.liquidation_indices == (1, 3)
    assert counterfactual.liquidation_indices == (1, 3)
    assert stopped.liquidations == counterfactual.liquidations == 2
    assert segment_metrics(counterfactual, 2, 4)[2]


def test_validation_winner_is_chosen_without_locked_test_metric() -> None:
    candidates = (
        Candidate(Family.CROSS_SECTIONAL, 5, 0, 1, 1, None, 0.0),
        Candidate(Family.CROSS_SECTIONAL, 10, 0, 1, 1, None, 0.0),
    )
    validation_scores = {candidates[0].identifier: 0.6, candidates[1].identifier: 0.3}
    validation_passes = {candidates[0].identifier: True, candidates[1].identifier: True}

    winners = select_validation_winners(candidates, validation_scores, validation_passes)

    assert winners == (candidates[0],)


def test_validation_selection_keeps_each_liquidity_lane_separate() -> None:
    metric = SegmentMetrics(net_return=0.1, max_drawdown=0.1, sharpe=1.0, positive_blocks=3, observations=100)
    risk = RiskOverlay(max_leverage=5.0, target_daily_vol=0.005, volatility_window=20)
    candidates = (
        Candidate(Family.TIME_SERIES, 20, 0, 0, 1, None, 0.0, minimum_daily_quote_volume=0.0),
        Candidate(Family.TIME_SERIES, 20, 0, 0, 1, None, 0.0, minimum_daily_quote_volume=5_000_000.0),
    )
    records = tuple(ValidationRecord(candidate, risk, metric, metric, metric, metric, False, False, score) for candidate, score in zip(candidates, (0.6, 0.5), strict=True))

    winners = _select_winners(records)

    assert {winner.candidate.minimum_daily_quote_volume for winner in winners} == {0.0, 5_000_000.0}


def _write_cache_file(path: Path, offset: float) -> str:
    rows = [
        ["2024-01-01 00:00:00+00:00", 100 + offset, 101 + offset, 99 + offset, 100 + offset, 1, 0, 1_000_000],
        ["2024-01-02 00:00:00+00:00", 101 + offset, 102 + offset, 100 + offset, 101 + offset, 1, 0, 1_000_000],
        ["2024-01-03 00:00:00+00:00", 102 + offset, 103 + offset, 101 + offset, 102 + offset, 1, 0, 1_000_000],
    ]
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "open", "high", "low", "close", "volume", "close_time", "quote_volume"])
        writer.writerows(rows)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_loader_accepts_hash_verified_common_daily_cache(tmp_path: Path) -> None:
    eth_path = tmp_path / "binance_fapi_ETHUSDT_1d.csv.gz"
    sol_path = tmp_path / "binance_fapi_SOLUSDT_1d.csv.gz"
    manifest = {
        "frozen_end": "2024-01-03",
        "files": [
            {"file": eth_path.name, "sha256": _write_cache_file(eth_path, 0.0)},
            {"file": sol_path.name, "sha256": _write_cache_file(sol_path, 10.0)},
        ],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    data, metadata = load_frozen_daily_cache(tmp_path, ("ETH", "SOL"))

    assert data.close_price.shape == (3, 2)
    assert metadata.common_rows == 3
    assert metadata.integrity_verified


def test_loader_rejects_hash_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "binance_fapi_ETHUSDT_1d.csv.gz"
    expected_hash = _write_cache_file(path, 0.0)
    manifest = {"frozen_end": "2024-01-03", "files": [{"file": path.name, "sha256": expected_hash}]}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with gzip.open(path, "at", encoding="utf-8") as handle:
        handle.write("tamper")

    with pytest.raises(CacheIntegrityError):
        load_frozen_daily_cache(tmp_path, ("ETH",))


def test_preregistered_broad_grid_has_exact_count_and_five_x_lane() -> None:
    candidates = enumerate_candidates()
    overlays = enumerate_risk_overlays()

    assert len(candidates) == 3_876
    assert len(overlays) == 36
    assert len(candidates) * len(overlays) == 139_536
    assert {overlay.max_leverage for overlay in overlays} == {1.0, 2.0, 3.0, 5.0}
    assert {candidate.minimum_daily_quote_volume for candidate in candidates} == {0.0, 1_000_000.0, 5_000_000.0, 20_000_000.0}
