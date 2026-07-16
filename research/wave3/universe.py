"""Point-in-time listings and historical liquidity filtering for wave-3."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final, assert_never

import numpy as np
import pandas as pd  # noqa: PANDAS_OK

from research.wave1.common import JsonValue, PipelineError, validate_symbol


LISTING_AGE_DAYS: Final = 60
VOLUME_WINDOW_DAYS: Final = 30
VOLUME_LIMIT: Final = 150


class AssetType(StrEnum):
    CRYPTO = "crypto"
    STOCK_TOKEN = "stock_token"


@dataclass(frozen=True, slots=True)
class AssetListing:
    """A listing contract with the exchange evidence needed by the backtest."""

    symbol: str
    asset_type: AssetType | str
    onboard_date: pd.Timestamp
    has_spot: bool = False
    has_bitget: bool = False

    def __post_init__(self) -> None:
        validate_symbol(self.symbol)
        normalized = AssetType(self.asset_type)
        onboard = pd.Timestamp(self.onboard_date)
        onboard = onboard.tz_localize("UTC") if onboard.tzinfo is None else onboard.tz_convert("UTC")
        object.__setattr__(self, "asset_type", normalized)
        object.__setattr__(self, "onboard_date", onboard)


def _timestamp_ms(value: JsonValue, field: str) -> pd.Timestamp:
    if not isinstance(value, (int, float, str)):
        raise PipelineError(f"listing {field} is missing or invalid")
    try:
        timestamp = pd.to_datetime(float(value), unit="ms", utc=True)
    except (TypeError, ValueError, OverflowError) as error:
        raise PipelineError(f"listing {field} is invalid: {value!r}") from error
    if pd.isna(timestamp):
        raise PipelineError(f"listing {field} is invalid: {value!r}")
    return pd.Timestamp(timestamp)


def parse_binance_um_listings(
    payload: JsonValue,
    spot_symbols: set[str],
    bitget_symbols: set[str],
    bitget_onboard_dates: dict[str, pd.Timestamp] | None = None,
) -> tuple[AssetListing, ...]:
    """Parse active USDT-M perpetuals while retaining exchange listing dates."""
    if not isinstance(payload, dict) or not isinstance(payload.get("symbols"), list):
        raise PipelineError("Binance exchangeInfo is missing symbols")
    listings: list[AssetListing] = []
    for raw in payload["symbols"]:
        if not isinstance(raw, dict):
            continue
        symbol = raw.get("symbol")
        if (
            raw.get("status") != "TRADING"
            or raw.get("contractType") != "PERPETUAL"
            or raw.get("quoteAsset") != "USDT"
            or not isinstance(symbol, str)
            or symbol not in spot_symbols
            or symbol not in bitget_symbols
        ):
            continue
        onboard = raw.get("onboardDate")
        if onboard is None:
            raise PipelineError(f"Binance listing {symbol} is missing onboardDate")
        onboard_date = _timestamp_ms(onboard, "onboardDate")
        if bitget_onboard_dates is not None and symbol in bitget_onboard_dates:
            onboard_date = max(onboard_date, pd.Timestamp(bitget_onboard_dates[symbol]))
        listings.append(AssetListing(symbol, AssetType.CRYPTO, onboard_date, True, True))
    return tuple(sorted(listings, key=lambda item: item.symbol))


def parse_bitget_stock_listings(
    payload: JsonValue,
    stock_bases: set[str],
    fallback_dates: dict[str, pd.Timestamp] | None = None,
) -> tuple[AssetListing, ...]:
    """Parse Bitget tokenized-stock contracts and their launch timestamps."""
    if not isinstance(payload, list):
        raise PipelineError("Bitget contract payload must be a list")
    listings: list[AssetListing] = []
    for raw in payload:
        if not isinstance(raw, dict):
            continue
        symbol = raw.get("symbol")
        base_coin = raw.get("baseCoin")
        if not isinstance(symbol, str) or not isinstance(base_coin, str) or base_coin.upper() not in stock_bases:
            continue
        launch = raw.get("launchTime", raw.get("openTime"))
        try:
            onboard = _timestamp_ms(launch, "launchTime")
        except PipelineError:
            # Bitget returns an empty launchTime for every tokenized-stock contract; date them
            # from their first cached candle instead (conservative: cannot enter earlier than data).
            onboard = (fallback_dates or {}).get(symbol)
        if onboard is None:
            raise PipelineError(f"Bitget listing {symbol} has no launchTime and no cached candles to date it")
        listings.append(AssetListing(symbol, AssetType.STOCK_TOKEN, onboard, False, True))
    return tuple(sorted(listings, key=lambda item: item.symbol))


def _listing_is_live(listing: AssetListing, day: pd.Timestamp) -> bool:
    match listing.asset_type:
        case AssetType.CRYPTO:
            return listing.onboard_date + pd.Timedelta(days=LISTING_AGE_DAYS) <= day
        case AssetType.STOCK_TOKEN:
            return listing.onboard_date <= day
        case unreachable:
            assert_never(unreachable)


def eligible_symbols_at(
    listings: tuple[AssetListing, ...],
    quote_volume: pd.DataFrame,
    as_of: pd.Timestamp,
    volume_limit: int = VOLUME_LIMIT,
) -> tuple[str, ...]:
    """Return a day-specific universe using only completed prior volume bars."""
    day = pd.Timestamp(as_of)
    day = day.tz_localize("UTC") if day.tzinfo is None else day.tz_convert("UTC")
    day = day.normalize()
    ordered_volume = quote_volume.copy()
    ordered_volume.index = pd.to_datetime(ordered_volume.index, utc=True).normalize()
    ordered_volume = ordered_volume[~ordered_volume.index.duplicated(keep="last")].sort_index()
    prior_days = pd.date_range(day - pd.Timedelta(days=VOLUME_WINDOW_DAYS), periods=VOLUME_WINDOW_DAYS, freq="D", tz="UTC")
    prior = ordered_volume.reindex(prior_days)
    if len(prior) != VOLUME_WINDOW_DAYS or prior.isna().all(axis=1).any():
        return ()
    numeric = prior.apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=float)
    # NaN cells are structural (symbol not listed yet) and dropped column-wise below;
    # only infinities and negative volumes indicate corrupt data.
    if np.isinf(values).any() or (values[np.isfinite(values)] < 0.0).any():
        raise PipelineError("quote-volume bars contain non-finite or negative values")
    live = [listing for listing in listings if _listing_is_live(listing, day)]
    if not live:
        return ()
    means = numeric.loc[:, numeric.notna().all(axis=0)].mean(axis=0).dropna()
    ranked = sorted(
        ((listing.symbol, float(means[listing.symbol])) for listing in live if listing.symbol in means),
        key=lambda item: (-item[1], item[0]),
    )
    return tuple(symbol for symbol, _ in ranked[:volume_limit])


def point_in_time_universe(
    listings: tuple[AssetListing, ...],
    quote_volume: pd.DataFrame,
) -> dict[pd.Timestamp, tuple[str, ...]]:
    """Build the eligible universe for every observed volume date."""
    days = sorted(pd.to_datetime(quote_volume.index, utc=True).normalize().unique())
    return {pd.Timestamp(day): eligible_symbols_at(listings, quote_volume, pd.Timestamp(day)) for day in days}


__all__ = [
    "AssetListing",
    "AssetType",
    "LISTING_AGE_DAYS",
    "VOLUME_LIMIT",
    "VOLUME_WINDOW_DAYS",
    "eligible_symbols_at",
    "parse_binance_um_listings",
    "parse_bitget_stock_listings",
    "point_in_time_universe",
]
