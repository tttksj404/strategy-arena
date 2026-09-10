"""Hand-checks and future perturbations for research-only signal adapters."""

import numpy as np
import pytest

import x_discovery_signals as signals
import x_discovery_statistics as statistics
from test_x_discovery_account import panel


def test_volume_exit_preserves_the_article_denominator() -> None:
    # Given: six-hour volume is below 20% of an average six-hour bucket.
    values = np.ones((25, 1)) * 100
    values[18:24] = 1
    # When: the 24 completed hourly bars are available at open 24.
    ratio = signals.volume_ratio(values, 24)
    # Then: it is 4 * 6 / 1806, not 6 / 1806 and not a price stop.
    assert ratio[24, 0] == pytest.approx(4 * 6 / 1806)
    assert ratio[23, 0] == 1


def test_volume_exit_never_reads_the_current_bar() -> None:
    # Given: future volume is radically different after index 30.
    first = np.ones((50, 2))
    changed = first.copy()
    changed[30:] *= 1000
    # When: calculating both hourly volume-ratio histories.
    before, after = signals.volume_ratio(first, 24), signals.volume_ratio(changed, 24)
    # Then: the signal at open 30 and all prior opens are unchanged.
    np.testing.assert_array_equal(before[:31], after[:31])


def test_zero_volume_after_a_full_day_does_not_keep_a_position_open() -> None:
    # Given: a complete day with no recorded turnover.
    values = np.zeros((25, 1))
    # When: the daily denominator is zero after warmup.
    ratio = signals.volume_ratio(values, 24)
    # Then: the exit filter is risk-off rather than interpreting zero as healthy flow.
    assert ratio[24, 0] == 0


def test_donchian_breakout_excludes_the_signal_bar_from_channel() -> None:
    # Given: a fresh high on completed bar 3.
    data = panel([100., 100., 100., 110., 110.])
    # When: using a three-day channel on daily bars and allowing the only asset.
    weights = signals.donchian(data, 1, days=3, first_asset=0)
    # Then: buy at open 4; earlier opens do not know the breakout.
    np.testing.assert_array_equal(weights[:, 0], [0., 0., 0., 0., 1.])


def test_cscv_midrank_makes_identical_paths_uninformative() -> None:
    # Given: all configurations have the exact same compounded path.
    values = np.full((3, 16), .001)
    # When: all 8-bin half splits use a log-growth objective.
    result = statistics.growth_pbo(values)
    # Then: ties do not count as outperformance.
    assert result.pbo == 1
    assert result.median_logit == 0
    assert result.splits == 70


def test_fee_floor_experiment_is_mathematical_not_venue_verification() -> None:
    # Given: the article's unverified fee floor and a $20 flat ticket each way.
    # When: checking the illustrative two-sided fee hurdle.
    hurdle = statistics.flat_ticket_fee_hurdle(20., .0045, .95)
    # Then: the fee floor alone consumes 9.5%, before spread or price-dependent exit fees.
    assert hurdle == pytest.approx(.095)
