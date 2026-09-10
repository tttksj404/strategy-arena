from __future__ import annotations

from run_bitget_4h_pair_regime_search import spec_for_interval
from run_bitget_dynamic_olmar_screen import DYNAMIC_ASSETS, _search_configs


def test_dynamic_olmar_screen_uses_the_expanded_btc_anchored_universe() -> None:
    configs = _search_configs(spec_for_interval("1D"))

    assert DYNAMIC_ASSETS[0] == "BTC"
    assert len(DYNAMIC_ASSETS) == 29
    assert len(configs) == 48
