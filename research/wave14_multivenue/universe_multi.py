# Wave-14 union-universe assembly ("universe_multi.py(거래소 합집합 유니버스,
# point-in-time)" per the task contract). Cache-only (no network -- mirrors
# research/wave13_liquidity/universe_liquidity.py's own split); network fetching lives
# entirely in fetch_venues.py.
#
# Two DIFFERENT point-in-time provenances feed this module, and they are NOT equally
# strong -- disclosed here and again in report/wave14_report.md, not smoothed over:
#   - The BINANCE side reuses wave13's own L4 membership verbatim (top200 by
#     reference_volume_30d_usdt as of FROZEN_END=2026-07-14, research.wave13_liquidity's
#     read-only import chain into research.wave12_frontier.universe_frontier) -- a STATIC
#     snapshot ranked as of a fixed historical date, same convention this repo has used
#     since wave12.
#   - The BYBIT side (fetch_venues.discover_bybit_universe) is a LIVE listing snapshot taken
#     at fetch time (whichever symbols Bybit's instruments-info reports as currently
#     Trading, intersected with L4's 200) -- NOT ranked as of a frozen historical date the
#     way the Binance side is. This means a symbol that traded on Bybit for part of the
#     2024-01~FROZEN_END window but was delisted before this wave's own fetch date would be
#     silently absent from the whole backtest (a mild survivorship bias on the Bybit side
#     only), whereas a symbol currently listed but that only started trading on Bybit
#     partway through the window is NOT a problem -- engine14's own availability mask
#     (research/wave14_multivenue/costs_venue.py's build_liquidity_mask_for_market, same
#     data-availability contract as wave13's L1-L4) already handles a late start correctly
#     (ineligible until its own 30d trailing volume is known), exactly like any newly-listed
#     symbol in wave10-13's own engines.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Final

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

import pandas as pd  # noqa: PANDAS_OK

from research.wave1.common import load_frame, load_json
from research.wave1.fam_funding import FundingMarket
from research.wave13_liquidity import universe_liquidity as ul
from research.wave13_liquidity.configs13 import get_config as get_wave13_config
from research.wave14_multivenue.fetch_venues import CACHE_DIR

BYBIT_KEY_SUFFIX: Final = ":BYBIT"


class UniverseError(Exception):
    pass


def venue_of_key(symbol_key: str) -> str:
    return "bybit" if symbol_key.endswith(BYBIT_KEY_SUFFIX) else "binance"


def base_symbol(symbol_key: str) -> str:
    return symbol_key.removesuffix(BYBIT_KEY_SUFFIX)


def bybit_key(symbol: str) -> str:
    return f"{symbol}{BYBIT_KEY_SUFFIX}"


# ---------------------------------------------------------------------------
# Binance side (read-only import chain into wave12_frontier/wave13_liquidity's own cache).
# ---------------------------------------------------------------------------


def load_l4_symbols() -> tuple[str, ...]:
    """wave13 L4's exact static membership (top200/12mo as of FROZEN_END) -- SPEC.md's own
    "L4 재현" instruction for M0, and the Binance-side candidate set for every other
    M-config too (Bybit only ever ADDS to this pool, per SPEC.md's "기회 풀 확대")."""
    l4_config = get_wave13_config("L4")
    pool = ul.load_candidate_pool()
    return ul.symbols_for_config(pool, l4_config)


def load_binance_markets(symbols: tuple[str, ...]) -> dict[str, FundingMarket]:
    return ul.load_markets_for_symbols(symbols)


def load_binance_quote_volume_frame(symbols: tuple[str, ...]) -> pd.DataFrame:
    return ul.load_quote_volume_frame(symbols)


# ---------------------------------------------------------------------------
# Bybit side (cache-only read of what fetch_venues.run_fetch_stage wrote).
# ---------------------------------------------------------------------------


def load_bybit_universe_payload() -> dict:
    path = CACHE_DIR / "bybit_universe.json"
    if not path.exists():
        raise UniverseError(f"{path} missing -- run `--stage fetch` first")
    payload = load_json(path)
    if not isinstance(payload, dict) or "universe_after_fetch" not in payload:
        raise UniverseError(f"{path} is invalid or predates the fetch stage completing")
    return payload


def load_bybit_symbols() -> tuple[str, ...]:
    """The post-fetch Bybit-tradable universe: wave13 L4's 200 symbols intersected with
    Bybit's own spot+linear live listings, minus any symbol that failed the actual market-
    data fetch (fetch_venues.py's own fetch_failures, excluded rather than backfilled)."""
    payload = load_bybit_universe_payload()
    return tuple(sorted(payload["universe_after_fetch"]))


def _bybit_cache_paths(symbol: str) -> dict[str, Path]:
    return {
        "spot": CACHE_DIR / f"bybit_spot_{symbol}_1d.csv.gz",
        "linear": CACHE_DIR / f"bybit_linear_{symbol}_1d.csv.gz",
        "funding": CACHE_DIR / f"bybit_funding_{symbol}.csv.gz",
    }


def missing_bybit_files(symbols: tuple[str, ...]) -> list[str]:
    missing: list[str] = []
    for symbol in symbols:
        for name, path in _bybit_cache_paths(symbol).items():
            if not path.exists():
                missing.append(f"{name}:{symbol}")
    return missing


def load_bybit_markets(symbols: tuple[str, ...]) -> dict[str, FundingMarket]:
    """Keyed by the RAW Bybit symbol (e.g. "BTCUSDT"), not the ":BYBIT"-suffixed union key
    -- callers that need the union-pool key apply bybit_key() themselves (engine14.py),
    keeping this loader identical in shape to
    research.wave13_liquidity.universe_liquidity.load_markets_for_symbols."""
    markets: dict[str, FundingMarket] = {}
    for symbol in symbols:
        paths = _bybit_cache_paths(symbol)
        if all(path.exists() for path in paths.values()):
            markets[symbol] = FundingMarket(
                spot=load_frame(paths["spot"]),
                perp=load_frame(paths["linear"]),
                funding=load_frame(paths["funding"])["funding_rate"],
            )
    return markets


def load_bybit_quote_volume_frame(symbols: tuple[str, ...], market: str) -> pd.DataFrame:
    """Daily quote_volume per symbol for ONE Bybit market ("spot" or "linear") -- the input
    to costs_venue.build_bp_frame_for_market's point-in-time rolling lookup. Two separate
    frames (not one) because SPOT and LINEAR have different turnover levels and are priced
    off their OWN trailing volume, never each other's (see costs_venue.py's module
    docstring)."""
    if market not in {"spot", "linear"}:
        raise UniverseError(f"unknown bybit market: {market!r}")
    series: dict[str, pd.Series] = {}
    for symbol in symbols:
        path = CACHE_DIR / f"bybit_{market}_{symbol}_1d.csv.gz"
        if path.exists():
            frame = load_frame(path)
            series[symbol] = frame["quote_volume"].resample("1D").sum()
    return pd.DataFrame(series).sort_index()


# ---------------------------------------------------------------------------
# Union pool for M0-M5's carry structure (engine14.run_carry_candidate).
# ---------------------------------------------------------------------------


def markets_for_carry_config(include_bybit: bool) -> tuple[dict[str, FundingMarket], dict[str, str]]:
    """Returns (markets keyed by union symbol-key, venue_of mapping). Binance's L4-200 is
    ALWAYS included (every M-config draws from it); Bybit's own universe is unioned in only
    when `include_bybit` (M1/M3/M4/M5/M6/M7) -- M0/M2's single-venue baselines get
    include_bybit=False and therefore see EXACTLY wave13 L4's own candidate set, nothing
    added or removed, which is what makes the M0-vs-M1 / M2-vs-M3 comparisons a clean
    single-variable (venue) diff."""
    l4_symbols = load_l4_symbols()
    binance_markets = load_binance_markets(l4_symbols)
    markets: dict[str, FundingMarket] = dict(binance_markets)
    venue_of: dict[str, str] = {symbol: "binance" for symbol in binance_markets}
    if include_bybit:
        bybit_symbols = load_bybit_symbols()
        missing = missing_bybit_files(bybit_symbols)
        if missing:
            raise UniverseError(f"wave14 bybit cache incomplete: {', '.join(missing[:8])} -- run `--stage fetch` first")
        bybit_markets = load_bybit_markets(bybit_symbols)
        for symbol, market in bybit_markets.items():
            key = bybit_key(symbol)
            markets[key] = market
            venue_of[key] = "bybit"
    return markets, venue_of


# ---------------------------------------------------------------------------
# Paired structure for M6/M7's cross-venue funding-spread engine
# (engine14.run_cross_venue_candidate) -- one entry per symbol that has BOTH a Binance perp
# leg (from L4's own membership) AND a Bybit perp leg (this wave's own fetch), no spot leg
# on either side (SPEC.md: "현물 불요, 양쪽 퍼프").
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CrossVenuePair:
    symbol: str
    binance_perp: pd.DataFrame
    binance_funding: pd.Series
    bybit_perp: pd.DataFrame
    bybit_funding: pd.Series


def cross_venue_pairs() -> dict[str, CrossVenuePair]:
    bybit_symbols = load_bybit_symbols()  # already an L4 ∩ Bybit intersection -- see fetch_venues.discover_bybit_universe
    missing = missing_bybit_files(bybit_symbols)
    if missing:
        raise UniverseError(f"wave14 bybit cache incomplete: {', '.join(missing[:8])} -- run `--stage fetch` first")
    binance_markets = load_binance_markets(bybit_symbols)
    bybit_markets = load_bybit_markets(bybit_symbols)
    pairs: dict[str, CrossVenuePair] = {}
    for symbol in bybit_symbols:
        if symbol not in binance_markets or symbol not in bybit_markets:
            continue
        pairs[symbol] = CrossVenuePair(
            symbol=symbol,
            binance_perp=binance_markets[symbol].perp,
            binance_funding=binance_markets[symbol].funding,
            bybit_perp=bybit_markets[symbol].perp,
            bybit_funding=bybit_markets[symbol].funding,
        )
    return pairs


__all__ = [
    "BYBIT_KEY_SUFFIX",
    "CrossVenuePair",
    "UniverseError",
    "base_symbol",
    "bybit_key",
    "cross_venue_pairs",
    "load_binance_markets",
    "load_binance_quote_volume_frame",
    "load_bybit_markets",
    "load_bybit_quote_volume_frame",
    "load_bybit_symbols",
    "load_bybit_universe_payload",
    "load_l4_symbols",
    "markets_for_carry_config",
    "missing_bybit_files",
    "venue_of_key",
]
