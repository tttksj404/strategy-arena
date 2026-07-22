# Wave-13 universe membership resolution. Deliberately thin: all point-in-time-safe
# candidate-pool / market-data plumbing is BORROWED (read-only import, no wave13-local
# copy) from research.wave12_frontier.universe_frontier -- this wave's own new work is the
# COST model (research/wave13_liquidity/collect_spreads.py + costs_measured.py), not a new
# OHLCV/funding data collection. Reusing wave12_frontier's already-fetched, already spot
# -truncation-verified cache (research/wave12_frontier/cache/candidate_pool.json,
# universe_frontier.json, and ~358 symbols' worth of binance_{spot,fapi,funding}_*.csv.gz)
# means L1-L5 reproduce EXACTLY wave12's own point-in-time universe-membership rule (rank by
# reference_volume_30d_usdt as of the same FROZEN_END, 2026-07-14) for L2-L4, and L3/L4
# reproduce U0/U2's membership byte-for-byte -- the only thing that can differ between L3 and
# wave12's U0 is the cost model, which is exactly the controlled comparison SPEC.md asks for.
#
# This module does NOT hit the network. If research/wave12_frontier/cache/ is ever missing
# (e.g. a clean checkout without wave12 having run first), every function below raises
# rather than silently fetching -- SPEC.md forbids modifying anything outside
# research/wave13_liquidity/, and a network re-fetch into wave12_frontier/cache/ from this
# module would violate exactly that boundary.

from __future__ import annotations

from pathlib import Path
import sys
from typing import Final

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

import pandas as pd  # noqa: PANDAS_OK

from research.wave1.fam_funding import FundingMarket
from research.wave12_frontier import universe_frontier as uf12
from research.wave13_liquidity.configs13 import Wave13Config

FROZEN_END: Final = uf12.FROZEN_END  # re-exported for reporting13.py; not redefined


def load_candidate_pool() -> dict:
    """research/wave12_frontier/cache/candidate_pool.json, read-only. Raises (via
    uf12.load_candidate_pool) if wave12_frontier's fetch stage never ran -- wave13 never
    fetches this itself."""
    return uf12.load_candidate_pool()


def symbols_for_config(pool: dict, config: Wave13Config) -> tuple[str, ...]:
    """The static candidate SET a config draws from. For L1 this is literally SPEC.md's
    named pair; for L2-L4 it is wave12_frontier's own breadth-ranked-by-reference-volume
    selection (research.wave12_frontier.universe_frontier.symbols_for_breadth_history,
    unmodified); for L5 it is the broadest (3mo-floor, uncapped breadth) parent pool --
    L5's REAL admission rule is the per-day dynamic filter engine13.py applies on top of
    this parent set, not this static list itself (see configs13.py's L5 note)."""
    if config.universe_kind == "fixed":
        if config.fixed_symbols is None:
            raise ValueError(f"{config.candidate.candidate_id}: universe_kind='fixed' requires fixed_symbols")
        return config.fixed_symbols
    if config.universe_kind == "breadth":
        if config.breadth is None:
            raise ValueError(f"{config.candidate.candidate_id}: universe_kind='breadth' requires breadth")
        return uf12.symbols_for_breadth_history(pool, config.breadth, config.history_months)
    if config.universe_kind == "dynamic":
        return uf12.symbols_for_breadth_history(pool, None, config.history_months)
    raise ValueError(f"{config.candidate.candidate_id}: unknown universe_kind {config.universe_kind!r}")


def verify_cache_and_load_symbols(config: Wave13Config) -> tuple[str, ...]:
    """Fail-closed cache check mirroring research.wave12_frontier.universe_frontier.
    verify_cache_and_load_symbols: no network access, every required market file must
    already exist in wave12_frontier's cache (or the caches it itself falls back to --
    wave11_yield/wave1 -- via uf12._first_existing, transparently)."""
    pool = load_candidate_pool()
    symbols = symbols_for_config(pool, config)
    missing = uf12.missing_market_files(symbols)
    if missing:
        raise RuntimeError(
            f"wave13 cache incomplete for {config.candidate.candidate_id} (borrowed from research/wave12_frontier/cache): "
            f"{', '.join(sorted(set(missing))[:8])} -- run research/wave12_frontier's own `--stage fetch` first"
        )
    return symbols


def load_markets_for_symbols(symbols: tuple[str, ...]) -> dict[str, FundingMarket]:
    return uf12.load_markets_for_symbols(symbols)


def load_quote_volume_frame(symbols: tuple[str, ...]) -> pd.DataFrame:
    """Daily perp quote_volume per symbol -- the sole input to costs_measured's rolling
    point-in-time trailing average. Unlike research.wave12_frontier.costs_tiered (which
    needed a shared cross-sectional pool to compute RANK), costs_measured's mapping is
    purely a function of each symbol's OWN volume, so this is called with exactly a given
    config's own `symbols` -- no broader reference pool required."""
    return uf12.load_quote_volume_frame(symbols)


__all__ = [
    "FROZEN_END",
    "load_candidate_pool",
    "load_markets_for_symbols",
    "load_quote_volume_frame",
    "symbols_for_config",
    "verify_cache_and_load_symbols",
]
