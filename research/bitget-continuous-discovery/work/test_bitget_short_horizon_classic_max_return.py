from math import sqrt

from alt_search_models import Family, SegmentMetrics
from run_bitget_4h_pair_regime_search import spec_for_interval
from run_bitget_short_horizon_classic_max_return_search import (
    CLASSIC_FAMILIES,
    ValidationRecord,
    _all_configs,
    select_validation_winner,
)


def test_classic_short_horizon_grid_covers_diverse_signal_families_at_five_x() -> None:
    configs = _all_configs(spec_for_interval("5m"))

    assert {config.candidate.family for config in configs} == set(CLASSIC_FAMILIES)
    assert {config.overlay.max_leverage for config in configs} == {5.0}
    assert any(config.candidate.family is Family.PAIR for config in configs)
    assert any(config.candidate.family is Family.EMA for config in configs)


def test_classic_short_horizon_grid_excludes_zscores_when_window_cannot_reach_them() -> None:
    configs = _all_configs(spec_for_interval("5m"))

    for config in configs:
        if config.candidate.family in {Family.BOLLINGER, Family.PAIR}:
            assert config.candidate.entry_z <= sqrt(config.candidate.lookback - 1)


def test_classic_short_horizon_winner_is_the_highest_validation_pnl() -> None:
    configs = _all_configs(spec_for_interval("5m"))
    lower = SegmentMetrics(0.1, 0.0, 0.0, 0, 1)
    higher = SegmentMetrics(0.2, 1.0, -1.0, 0, 1)
    records = (
        ValidationRecord(configs[0], lower, lower),
        ValidationRecord(configs[1], higher, higher),
    )

    assert select_validation_winner(records) == records[1]
