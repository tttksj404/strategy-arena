import numpy as np

from alt_search_models import SegmentMetrics
from bitget_factor_signals import FactorExposure, FactorFamily
from bitget_search_metrics import fixed_notional_metrics
from run_bitget_4h_pair_regime_search import spec_for_interval
from run_bitget_short_horizon_max_return_search import (
    ValidationRecord,
    _all_configs,
    select_validation_winner,
)


def test_short_horizon_grid_covers_every_factor_family_and_both_exposures() -> None:
    configs = _all_configs(spec_for_interval("5m"))

    assert {config.factor.family for config in configs} == set(FactorFamily)
    assert {config.exposure for config in configs} == set(FactorExposure)
    assert {config.overlay.max_leverage for config in configs} == {5.0}
    assert len(configs) == len(FactorFamily) * len(FactorExposure) * 3 * 2 * 2


def test_short_horizon_grid_keeps_factor_lookbacks_at_two_or_more_bars() -> None:
    configs = _all_configs(spec_for_interval("5m"))

    assert min(config.factor.lookback for config in configs) == 2


def test_fixed_notional_metrics_preserve_pnl_after_a_negative_one_bar() -> None:
    metrics = fixed_notional_metrics(np.asarray((-1.0, 0.20)), bars_per_year=1.0)

    assert metrics.net_return == -0.80


def test_fixed_notional_metrics_never_report_negative_drawdown() -> None:
    metrics = fixed_notional_metrics(np.asarray((0.10, 0.20)), bars_per_year=1.0)

    assert metrics.max_drawdown == 0.0


def test_validation_selection_uses_return_without_liquidation_or_test_gate() -> None:
    metric = SegmentMetrics(net_return=0.10, max_drawdown=0.90, sharpe=0.1, positive_blocks=1, observations=10)
    higher_return = SegmentMetrics(net_return=0.20, max_drawdown=0.99, sharpe=-0.1, positive_blocks=0, observations=10)
    configs = _all_configs(spec_for_interval("5m"))
    records = (
        ValidationRecord(configs[0], metric, metric),
        ValidationRecord(configs[1], higher_return, higher_return),
    )

    assert select_validation_winner(records) == records[1]
