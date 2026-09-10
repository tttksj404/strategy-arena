from __future__ import annotations

import run_bitget_pca_stat_arb_search as runner


def test_pca_stat_arb_grid_spans_model_dimensions_before_risk_overlays() -> None:
    models = runner._pca_configs(runner.spec_for_interval("4H"))

    assert len(models) == 48
    assert {model.factor_count for model in models} == {1, 2, 3}
    assert {model.training_bars for model in models} == {120, 240}
    assert {model.signal_bars for model in models} == {6, 12}
    assert {model.residual_z_window for model in models} == {30, 60}
    assert {model.entry_z for model in models} == {0.5, 1.0}
    assert len(runner._search_configs(runner.spec_for_interval("4H"))) == 192
