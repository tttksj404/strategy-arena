# pyright: reportImplicitRelativeImport=false

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from alt_search_models import MarketData
from bitget_archive_daily import ArchiveDailyError
from run_bitget_top_strategy_diagnostic import (
    DEFAULT_ARCHIVE,
    DEFAULT_SELECTION_REPORT,
    daily_bar_diagnostics,
    run,
    selection_audit,
    split_bounds,
    strategy_specs,
)


def _market() -> MarketData:
    rows = 8
    dates = tuple(datetime(2026, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(rows))
    close = np.asarray((100, 102, 110, 105, 105, 115, 112, 120), dtype=float)[:, None]
    open_price = close.copy()
    high = close + 2.0
    low = close - 2.0
    low[1, 0] = 79.0
    return MarketData(dates, ("ALT",), open_price, high, low, close, np.ones_like(close))


def test_strategy_specs_reproduce_the_two_named_candidates() -> None:
    specs = strategy_specs()
    assert len(specs) == 2
    assert specs[0].candidate.identifier == "donchian|lb=90|sec=0|k=0|dir=1|pair=none|z=0|reb=1|liq=1e+06"
    assert specs[0].overlay.identifier == "L=5|tv=0.005|vw=40"
    assert specs[1].candidate.identifier == "price_volume|lb=60|sec=0|k=1|dir=1|pair=none|z=0|reb=2|liq=2e+07"
    assert specs[1].overlay.identifier == "L=5|tv=0.01|vw=40"


def test_split_bounds_are_chronological_fifty_twenty_five_twenty_five() -> None:
    assert split_bounds(8) == (0, 4, 6, 8)


def test_daily_diagnostic_distinguishes_eventual_winner_from_path_liquidation() -> None:
    data = _market()
    positions = np.asarray(((0.0,), (1.0,), (1.0,), (0.0,), (0.0,), (1.0,), (1.0,), (0.0,)))
    diagnostic = daily_bar_diagnostics(data, positions, leverage=5.0, start=1, end=7)
    assert diagnostic["event_count"] == 1
    assert diagnostic["right_censored_event_count"] == 1
    assert diagnostic["completed_event_terminal_direction_winners"] == 1
    assert diagnostic["completed_event_winner_liquidation_proxy"] == 1
    mae_p95 = diagnostic["mae_p95"]
    mae_median = diagnostic["mae_median"]
    assert mae_p95 is not None and mae_median is not None
    assert mae_p95 >= mae_median


def test_daily_diagnostic_uses_leverage_on_the_adverse_bar() -> None:
    """Given leverage rises from 1x to 5x, a later adverse bar must trigger the proxy."""

    data = _market()
    data.low_price[2, 0] = 75.0
    positions = np.asarray(((0.0,), (1.0,), (1.0,), (0.0,), (0.0,), (1.0,), (1.0,), (0.0,)))
    leverage = np.asarray((0.0, 1.0, 5.0, 5.0, 0.0, 1.0, 1.0, 0.0))

    diagnostic = daily_bar_diagnostics(data, positions, leverage=leverage, start=1, end=7)

    assert diagnostic["liquidation_proxy_events"] == 1
    assert diagnostic["completed_event_winner_liquidation_proxy"] == 1


def test_selection_audit_imports_hash_verified_source_gates() -> None:
    """Given the source PBO report, the imported candidate IDs and gates stay exact."""

    source, audit = selection_audit(DEFAULT_SELECTION_REPORT)

    assert source["sha256"] == "dfcd96aead7a059ba32704d4d1c3cd6bdc97377f2db919b511458185fecf8871"
    assert audit["raw_top_donchian"]["paper_pass"] is False
    assert audit["paper_gate_top_price_volume"]["paper_pass"] is True


def test_selection_audit_fails_closed_when_source_changes(tmp_path: Path) -> None:
    """Given a changed source report, selection-gate import must fail closed."""

    changed = tmp_path / "changed.json"
    _ = changed.write_bytes(DEFAULT_SELECTION_REPORT.read_bytes().replace(b'"paper_pass": true', b'"paper_pass": false', 1))

    with pytest.raises(ArchiveDailyError):
        _ = selection_audit(changed)


def test_locked_test_direction_rates_exclude_right_censored_events() -> None:
    """Given the locked archive, right-censored events are outside the win-rate denominator."""

    report = run(DEFAULT_ARCHIVE)

    for strategy_name in ("raw_top_donchian", "paper_gate_top_price_volume"):
        diagnostic = report["strategies"][strategy_name]["locked_test_daily_bar_diagnostic"]
        assert diagnostic["right_censored_event_count"] > 0
        assert diagnostic["event_count"] > 0
        winners = diagnostic["completed_event_terminal_direction_winners"]
        rate = diagnostic.get("completed_event_terminal_direction_win_rate")
        assert rate is not None
        assert rate == pytest.approx(winners / diagnostic["event_count"])


def test_fixed_5x_counterfactual_reproduces_locked_archive_proxy_hits() -> None:
    """Given the locked archive, fixed 5x reports Donchian 1 hit and Price–Volume 6."""

    report = run(DEFAULT_ARCHIVE)

    assert report["strategies"]["raw_top_donchian"]["fixed_5x_counterfactual"]["market_ohlc_proxy_liquidation_hit_count"] == 1
    assert report["strategies"]["paper_gate_top_price_volume"]["fixed_5x_counterfactual"]["market_ohlc_proxy_liquidation_hit_count"] == 6
