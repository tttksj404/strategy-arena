from __future__ import annotations

import pandas as pd  # noqa: PANDAS_OK
import pytest

from research.paper import fidelity
from research.paper.candidates import TRACKED_IDS
from research.paper.market_data import LiveSnapshot


def _snapshot(funding_symbols: tuple[str, ...], wave3_symbols: tuple[str, ...] = ()) -> LiveSnapshot:
    funding_series = {symbol: pd.Series([0.0001], index=pd.DatetimeIndex([pd.Timestamp("2026-07-01", tz="UTC")])) for symbol in funding_symbols}
    return LiveSnapshot(
        pd.Timestamp("2026-07-28T00:00:00Z"),
        funding_series,
        {symbol: 0.0001 for symbol in funding_symbols},
        {},
        {},
        {symbol: object() for symbol in wave3_symbols},  # type: ignore[arg-type]  -- fidelity only counts keys, never touches AssetMarket internals
        ("synthetic",),
    )


def test_every_tracked_candidate_has_a_declared_universe_requirement() -> None:
    """The pytest-enforced guard the task asks for: adding a new candidate ID to
    research/paper/candidates.py's TRACKED_IDS without registering its universe requirement in
    research/paper/fidelity.py.REQUIREMENTS must fail CI, not just silently under-check it."""
    missing = [candidate_id for candidate_id in TRACKED_IDS if candidate_id not in fidelity.REQUIREMENTS]
    assert not missing, f"tracked candidates without a declared fidelity.REQUIREMENTS entry: {missing}"


def test_check_snapshot_volume_top_n_passes_when_coverage_meets_the_requirement() -> None:
    snapshot = _snapshot(tuple(f"SYM{i}USDT" for i in range(100)))
    result = fidelity.check_snapshot(snapshot, "G1")
    assert result.ok is True
    assert result.required == 100
    assert result.actual == 100
    assert result.reason == "OK"


def test_check_snapshot_volume_top_n_fails_and_reports_required_vs_actual() -> None:
    # Mirrors the actual incident: 61 symbols against G1's declared top100.
    snapshot = _snapshot(tuple(f"SYM{i}USDT" for i in range(61)))
    result = fidelity.check_snapshot(snapshot, "G1")
    assert result.ok is False
    assert result.required == 100
    assert result.actual == 61
    assert "G1" in result.reason
    assert "100" in result.reason
    assert "61" in result.reason


def test_check_snapshot_w2c_requires_top200() -> None:
    snapshot = _snapshot(tuple(f"SYM{i}USDT" for i in range(150)))
    result = fidelity.check_snapshot(snapshot, "W2c")
    assert result.ok is False
    assert result.required == 200
    assert result.actual == 150


def test_check_snapshot_majors_only_needs_btc_and_eth_specifically() -> None:
    ok_snapshot = _snapshot(("BTCUSDT", "ETHUSDT", "SOLUSDT"))
    ok_result = fidelity.check_snapshot(ok_snapshot, "F1e")
    assert ok_result.ok is True
    assert ok_result.actual == 2

    missing_eth = _snapshot(("BTCUSDT", "SOLUSDT"))
    fail_result = fidelity.check_snapshot(missing_eth, "F1e")
    assert fail_result.ok is False
    assert fail_result.actual == 1


def test_check_snapshot_wave3_momentum_counts_wave3_markets_not_funding_series() -> None:
    # funding_series stays tiny on purpose -- W3c/W3d's requirement reads wave3_markets, and
    # must not be satisfied (or falsely failed) by the unrelated funding-series count.
    snapshot = _snapshot(("BTCUSDT",), wave3_symbols=tuple(f"SYM{i}USDT" for i in range(150)))
    result = fidelity.check_snapshot(snapshot, "W3c")
    assert result.ok is True
    assert result.actual == 150


def test_check_snapshot_raises_for_a_candidate_with_no_declared_requirement() -> None:
    snapshot = _snapshot(("BTCUSDT",))
    with pytest.raises(KeyError):
        fidelity.check_snapshot(snapshot, "NOT_A_REGISTERED_CANDIDATE")


def test_check_all_covers_every_requested_candidate() -> None:
    snapshot = _snapshot(tuple(f"SYM{i}USDT" for i in range(200)) + ("BTCUSDT", "ETHUSDT"), wave3_symbols=tuple(f"SYM{i}USDT" for i in range(150)))
    results = fidelity.check_all(snapshot, TRACKED_IDS)
    assert set(results) == set(TRACKED_IDS)
    assert all(result.ok for result in results.values())
