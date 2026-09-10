from __future__ import annotations

import run_bitget_hmm_regime_search as runner


def test_hmm_grid_spans_state_model_and_regime_choices() -> None:
    models = runner._hmm_configs(runner.spec_for_interval("4H"))

    assert len(models) == 16
    assert {model.state_count for model in models} == {2, 3}
    assert {model.transition_persistence for model in models} == {0.85, 0.95}
    assert {model.training_bars for model in models} == {120, 240}
    assert {model.trend_lookback for model in models} == {30}
    assert {model.high_volatility_cash for model in models} == {False, True}
    assert len(runner._search_configs(runner.spec_for_interval("4H"))) == 64
