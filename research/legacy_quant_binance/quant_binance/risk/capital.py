from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol

_log = logging.getLogger(__name__)

from quant_binance.settings import Settings


class SupportsCapitalPricing(Protocol):
    def get_exchange_info(self, *, market: str) -> dict[str, Any]:
        ...

    def get_book_ticker(self, *, market: str, symbol: str) -> dict[str, Any]:
        ...


@dataclass(frozen=True)
class AccountCapitalInputs:
    spot_available_balance_usd: float
    spot_recognized_balance_usd: float
    spot_funding_assets: tuple["SpotFundingAsset", ...]
    futures_available_balance_usd: float
    futures_execution_balance_usd: float
    futures_recognized_balance_usd: float
    futures_funding_assets: tuple["FuturesFundingAsset", ...]


@dataclass(frozen=True)
class SpotFundingAsset:
    asset: str
    free: float
    locked: float
    total: float
    free_balance_usd: float
    total_balance_usd: float


@dataclass(frozen=True)
class FuturesFundingAsset:
    asset: str
    free: float
    locked: float
    total: float
    free_balance_usd: float
    total_balance_usd: float
    margin_available: bool


@dataclass(frozen=True)
class CapitalRequirement:
    symbol: str
    market: str
    min_notional_usd: float
    min_quantity: float
    min_equity_usd: float
    buffered_min_equity_usd: float


@dataclass(frozen=True)
class CapitalAdequacyReport:
    spot_available_balance_usd: float
    spot_recognized_balance_usd: float
    spot_funding_assets: tuple[SpotFundingAsset, ...]
    futures_funding_assets: tuple[FuturesFundingAsset, ...]
    spot_execution_routes: tuple["SpotExecutionRoute", ...]
    capital_transfer_routes: tuple["CapitalTransferRoute", ...]
    futures_available_balance_usd: float
    futures_execution_balance_usd: float
    futures_recognized_balance_usd: float
    futures_transferable_execution_balance_usd: float
    futures_collateral_candidate_balance_usd: float
    futures_manual_handling_balance_usd: float
    futures_total_reusable_balance_usd: float
    minimum_operational_balance_usd: float
    minimum_full_universe_balance_usd: float
    recommended_balance_usd: float
    can_trade_any: bool
    can_trade_spot_any: bool
    can_trade_futures_any: bool
    max_spot_to_futures_transfer_usd: float
    max_futures_to_spot_transfer_usd: float
    spot_requirements: tuple[CapitalRequirement, ...]
    futures_requirements: tuple[CapitalRequirement, ...]
    pending_symbols: tuple[str, ...]
    note: str


@dataclass(frozen=True)
class SpotExecutionRoute:
    target_symbol: str
    execution_symbol: str
    base_asset: str
    quote_asset: str
    funding_asset: str
    funding_source_market: str
    route_type: str
    requires_wallet_transfer: bool
    free_balance: float
    free_balance_usd: float
    quote_asset_usd_price: float
    min_notional_usd: float
    min_quantity: float


@dataclass(frozen=True)
class CapitalTransferRoute:
    source_market: str
    target_market: str
    asset: str
    source_free_amount: float
    transferable_usd: float
    route_type: str
    requires_manual_transfer: bool
    capacity_effect: str
    auto_transfer_supported: bool
    note: str


def _transfer_capacity_effect_priority(capacity_effect: str) -> int:
    return {
        "direct_execution": 0,
        "collateral_candidate": 1,
        "manual_conversion": 2,
        "spot_execution": 0,
    }.get(capacity_effect, 3)


def _symbol_min_notional(exchange_info: dict[str, Any], symbol: str) -> float:
    for item in exchange_info["symbols"]:
        if item["symbol"] != symbol:
            continue
        for filt in item["filters"]:
            if filt["filterType"] in {"MIN_NOTIONAL", "NOTIONAL"}:
                value = filt.get("notional") or filt.get("minNotional")
                return float(value)
    raise KeyError(f"min notional not found for {symbol}")


def _symbol_min_quantity(exchange_info: dict[str, Any], symbol: str) -> float:
    for item in exchange_info["symbols"]:
        if item["symbol"] != symbol:
            continue
        for filt in item["filters"]:
            if filt["filterType"] == "LOT_SIZE":
                value = filt.get("minQty") or filt.get("stepSize")
                return float(value)
        raw = item.get("raw", {})
        if isinstance(raw, dict):
            value = raw.get("minTradeNum") or raw.get("minTradeAmount") or raw.get("sizeMultiplier")
            if value is not None:
                return float(value)
    return 0.0


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _recognized_spot_balance_usd(
    *,
    spot_account: dict[str, Any],
    rest_client: SupportsCapitalPricing,
    spot_exchange_info: dict[str, Any],
) -> float:
    return round(
        sum(item.total_balance_usd for item in _spot_funding_assets(spot_account=spot_account, rest_client=rest_client, spot_exchange_info=spot_exchange_info)),
        6,
    )


def _symbol_assets(
    *,
    exchange_info: dict[str, Any],
    symbol: str,
    extra_quote_assets: tuple[str, ...] = (),
) -> tuple[str, str]:
    for item in exchange_info.get("symbols", []):
        if item.get("symbol") != symbol:
            continue
        raw = item.get("raw") if isinstance(item.get("raw"), dict) else {}
        base_asset = (
            item.get("baseAsset")
            or item.get("baseCoin")
            or raw.get("baseAsset")
            or raw.get("baseCoin")
            or ""
        )
        quote_asset = (
            item.get("quoteAsset")
            or item.get("quoteCoin")
            or raw.get("quoteAsset")
            or raw.get("quoteCoin")
            or ""
        )
        if base_asset and quote_asset:
            return str(base_asset).upper(), str(quote_asset).upper()
        break
    quote_assets = tuple(
        dict.fromkeys(
            [
                *extra_quote_assets,
                "USDT",
                "USDC",
                "FDUSD",
                "TUSD",
                "BTC",
                "ETH",
                "BNB",
            ]
        )
    )
    for quote_asset in sorted(quote_assets, key=len, reverse=True):
        if symbol.endswith(quote_asset) and len(symbol) > len(quote_asset):
            return symbol[: -len(quote_asset)].upper(), quote_asset.upper()
    return "", ""


def _spot_asset_usd_price(
    *,
    asset: str,
    supported_symbols: set[str],
    rest_client: SupportsCapitalPricing,
    price_by_symbol: dict[str, float | None],
) -> float | None:
    if asset == "USDT":
        return 1.0
    price_symbol = f"{asset}USDT"
    if price_symbol not in supported_symbols:
        return None
    if price_symbol not in price_by_symbol:
        try:
            ticker = rest_client.get_book_ticker(market="spot", symbol=price_symbol)
        except Exception:
            _log.warning("price fetch failed for %s — treating as unavailable", price_symbol, exc_info=True)
            price_by_symbol[price_symbol] = None
            return None
        bid_price = _optional_float(ticker.get("bidPrice"))
        ask_price = _optional_float(ticker.get("askPrice"))
        price_by_symbol[price_symbol] = bid_price if bid_price is not None and bid_price > 0 else ask_price
    price = price_by_symbol[price_symbol]
    if price is None or price <= 0.0:
        return None
    return price


def _spot_funding_assets(
    *,
    spot_account: dict[str, Any],
    rest_client: SupportsCapitalPricing,
    spot_exchange_info: dict[str, Any],
) -> tuple[SpotFundingAsset, ...]:
    balances = spot_account.get("balances", [])
    if not isinstance(balances, list):
        return ()
    supported_symbols = {str(item.get("symbol", "")) for item in spot_exchange_info.get("symbols", [])}
    price_by_symbol: dict[str, float | None] = {}
    assets: list[SpotFundingAsset] = []
    for item in balances:
        if not isinstance(item, dict):
            continue
        asset = str(item.get("asset", "")).upper()
        if not asset:
            continue
        free = _optional_float(item.get("free")) or 0.0
        locked = _optional_float(item.get("locked")) or 0.0
        total = free + locked
        if total <= 0.0:
            continue
        price = _spot_asset_usd_price(
            asset=asset,
            supported_symbols=supported_symbols,
            rest_client=rest_client,
            price_by_symbol=price_by_symbol,
        )
        free_balance_usd = free * price if price is not None else 0.0
        total_balance_usd = total * price if price is not None else 0.0
        assets.append(
            SpotFundingAsset(
                asset=asset,
                free=round(free, 8),
                locked=round(locked, 8),
                total=round(total, 8),
                free_balance_usd=round(free_balance_usd, 6),
                total_balance_usd=round(total_balance_usd, 6),
            )
        )
    assets.sort(key=lambda item: (-item.free_balance_usd, item.asset))
    return tuple(assets)


def _exchange_supports_internal_transfer(
    *,
    exchange_id: str,
    source_market: str,
    target_market: str,
    asset: str,
) -> bool:
    normalized_exchange = exchange_id.strip().lower()
    normalized_asset = asset.strip().upper()
    route = (source_market, target_market)
    if route not in {("spot", "futures"), ("futures", "spot")}:
        return False
    if normalized_exchange == "binance":
        return bool(normalized_asset)
    if normalized_exchange == "bitget":
        return True  # allow non-USDT spot↔futures transfers (collateral candidates)
    return normalized_asset == "USDT"


def _futures_asset_total_usd(
    *,
    row: dict[str, Any],
    asset: str,
    supported_symbols: set[str],
    rest_client: SupportsCapitalPricing,
    price_by_symbol: dict[str, float | None],
) -> float | None:
    # usdtEquity is always USDT-denominated regardless of the margin coin — trust it first.
    usd_equity = _optional_float(row.get("usdtEquity"))
    if usd_equity is not None and usd_equity > 0.0:
        return usd_equity
    if asset == "USDT":
        # USDT-margined account: accountEquity/equity are also USD-denominated.
        for key in ("accountEquity", "equity"):
            parsed = _optional_float(row.get(key))
            if parsed is not None and parsed > 0.0:
                return parsed
    # For non-USDT coin-margined accounts (COIN-FUTURES), equity/accountEquity are
    # coin-denominated (e.g. 0.01 BTC) — NOT USD. Collect the coin amount and
    # convert to USD via the spot price.
    coin_candidates = (
        row.get("marginBalance"),
        row.get("walletBalance"),
        row.get("crossWalletBalance"),
        row.get("equity"),        # coin-denominated for COIN-FUTURES
        row.get("accountEquity"), # coin-denominated for COIN-FUTURES
        row.get("availableBalance"),
        row.get("maxWithdrawAmount"),
    )
    amount = next(
        (parsed for parsed in (_optional_float(value) for value in coin_candidates) if parsed is not None and parsed > 0.0),
        None,
    )
    if amount is None:
        return None
    if asset == "USDT":
        return amount
    price = _spot_asset_usd_price(
        asset=asset,
        supported_symbols=supported_symbols,
        rest_client=rest_client,
        price_by_symbol=price_by_symbol,
    )
    if price is None or price <= 0.0:
        return None
    return amount * price


def _futures_funding_assets(
    *,
    futures_account: dict[str, Any],
    rest_client: SupportsCapitalPricing,
    spot_exchange_info: dict[str, Any],
) -> tuple[FuturesFundingAsset, ...]:
    rows = futures_account.get("assets")
    if not isinstance(rows, list):
        rows = futures_account.get("accounts")
    if not isinstance(rows, list):
        return ()
    supported_symbols = {str(item.get("symbol", "")) for item in spot_exchange_info.get("symbols", [])}
    price_by_symbol: dict[str, float | None] = {}
    assets: list[FuturesFundingAsset] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        asset = str(row.get("asset") or row.get("marginCoin") or row.get("currency") or "").upper()
        if not asset:
            continue
        free = next(
            (
                parsed
                for parsed in (
                    _optional_float(row.get("maxWithdrawAmount")),
                    _optional_float(row.get("availableBalance")),
                    _optional_float(row.get("available")),
                    _optional_float(row.get("free")),
                )
                if parsed is not None and parsed > 0.0
            ),
            0.0,
        )
        total = next(
            (
                parsed
                for parsed in (
                    _optional_float(row.get("walletBalance")),
                    _optional_float(row.get("marginBalance")),
                    _optional_float(row.get("equity")),
                    _optional_float(row.get("balance")),
                    _optional_float(row.get("crossWalletBalance")),
                    _optional_float(row.get("availableBalance")),
                )
                if parsed is not None and parsed > 0.0
            ),
            0.0,
        )
        if total <= 0.0 and free > 0.0:
            total = free
        if total <= 0.0:
            continue
        total_balance_usd = _futures_asset_total_usd(
            row=row,
            asset=asset,
            supported_symbols=supported_symbols,
            rest_client=rest_client,
            price_by_symbol=price_by_symbol,
        )
        if total_balance_usd is None:
            continue
        free_ratio = min(max(free / total, 0.0), 1.0) if total > 0.0 else 0.0
        free_balance_usd = total_balance_usd * free_ratio
        assets.append(
            FuturesFundingAsset(
                asset=asset,
                free=round(free, 8),
                locked=round(max(total - free, 0.0), 8),
                total=round(total, 8),
                free_balance_usd=round(free_balance_usd, 6),
                total_balance_usd=round(total_balance_usd, 6),
                margin_available=bool(row.get("marginAvailable", asset == "USDT")),
            )
        )
    # Also extract non-USDT assets from Bitget's nested assetList (e.g. BTC collateral).
    # These rows use "coin" instead of "asset"/"marginCoin" so they are skipped above.
    seen_assets = {a.asset for a in assets}
    for row in rows:
        if not isinstance(row, dict):
            continue
        for sub in row.get("assetList", []) or []:
            if not isinstance(sub, dict):
                continue
            sub_asset = str(sub.get("coin") or sub.get("asset") or "").upper()
            if not sub_asset or sub_asset in seen_assets:
                continue
            sub_free = _optional_float(sub.get("available")) or 0.0
            sub_total = _optional_float(sub.get("balance")) or sub_free
            if sub_total <= 0.0:
                continue
            sub_total_usd = _futures_asset_total_usd(
                row=sub,
                asset=sub_asset,
                supported_symbols=supported_symbols,
                rest_client=rest_client,
                price_by_symbol=price_by_symbol,
            )
            if sub_total_usd is None:
                # Fall back to spot-price lookup
                price = _spot_asset_usd_price(
                    asset=sub_asset,
                    supported_symbols=supported_symbols,
                    rest_client=rest_client,
                    price_by_symbol=price_by_symbol,
                )
                sub_total_usd = sub_total * price if price is not None else None
            if sub_total_usd is None or sub_total_usd <= 0.0:
                continue
            sub_free_ratio = min(max(sub_free / sub_total, 0.0), 1.0) if sub_total > 0.0 else 0.0
            assets.append(
                FuturesFundingAsset(
                    asset=sub_asset,
                    free=round(sub_free, 8),
                    locked=round(max(sub_total - sub_free, 0.0), 8),
                    total=round(sub_total, 8),
                    free_balance_usd=round(sub_total_usd * sub_free_ratio, 6),
                    total_balance_usd=round(sub_total_usd, 6),
                    margin_available=bool(sub.get("marginAvailable", False)),
                )
            )
            seen_assets.add(sub_asset)
    assets.sort(key=lambda item: (-item.free_balance_usd, item.asset))
    return tuple(assets)


def _recognized_futures_balance_usd(futures_account: dict[str, Any]) -> float:
    top_level_candidates = (
        futures_account.get("totalMarginBalance"),
        futures_account.get("totalWalletBalance"),
        futures_account.get("totalCrossWalletBalance"),
        futures_account.get("usdtEquity"),
        futures_account.get("accountEquity"),
        futures_account.get("equity"),
    )
    for value in top_level_candidates:
        parsed = _optional_float(value)
        if parsed is not None:
            return parsed
    for row_key in ("accounts", "assets"):
        rows = futures_account.get(row_key)
        if not isinstance(rows, list):
            continue
        recognized_total = 0.0
        found = False
        for item in rows:
            if not isinstance(item, dict):
                continue
            # Only sum USDT-margined rows directly. Non-USDT (COIN-FUTURES) rows hold
            # equity in coin units (e.g. 0.01 BTC ≠ $0.01). Their USD value is computed
            # separately via _futures_funding_assets to avoid both under-valuation and
            # double-counting.
            margin_coin = str(item.get("marginCoin") or item.get("asset") or item.get("currency") or "").upper()
            if margin_coin and margin_coin != "USDT":
                continue
            for key in ("usdtEquity", "accountEquity", "equity", "marginBalance", "walletBalance"):
                parsed = _optional_float(item.get(key))
                if parsed is not None:
                    recognized_total += parsed
                    found = True
                    break
        if found:
            return recognized_total
    return _optional_float(
        futures_account.get("executionAvailableBalance", futures_account.get("availableBalance"))
    ) or 0.0


def extract_account_capital_inputs(
    *,
    spot_account: dict[str, Any],
    futures_account: dict[str, Any],
    rest_client: SupportsCapitalPricing,
) -> AccountCapitalInputs:
    spot_exchange_info = rest_client.get_exchange_info(market="spot")
    balances = spot_account.get("balances", [])
    spot_available_balance_usd = 0.0
    if isinstance(balances, list):
        spot_available_balance_usd = next(
            (
                _optional_float(item.get("free")) or 0.0
                for item in balances
                if isinstance(item, dict) and str(item.get("asset", "")).upper() == "USDT"
            ),
            0.0,
        )
    futures_available_balance_usd = _optional_float(futures_account.get("availableBalance")) or 0.0
    futures_execution_balance_usd = _optional_float(
        futures_account.get("executionAvailableBalance", futures_available_balance_usd)
    ) or futures_available_balance_usd
    spot_funding_assets = _spot_funding_assets(
        spot_account=spot_account,
        rest_client=rest_client,
        spot_exchange_info=spot_exchange_info,
    )
    futures_funding_assets = _futures_funding_assets(
        futures_account=futures_account,
        rest_client=rest_client,
        spot_exchange_info=spot_exchange_info,
    )
    return AccountCapitalInputs(
        spot_available_balance_usd=round(spot_available_balance_usd, 6),
        spot_recognized_balance_usd=round(sum(item.total_balance_usd for item in spot_funding_assets), 6),
        spot_funding_assets=spot_funding_assets,
        futures_available_balance_usd=round(futures_available_balance_usd, 6),
        futures_execution_balance_usd=round(futures_execution_balance_usd, 6),
        futures_recognized_balance_usd=round(_recognized_futures_balance_usd(futures_account), 6),
        futures_funding_assets=futures_funding_assets,
    )


def split_supported_symbols(
    *,
    symbols: tuple[str, ...],
    rest_client: SupportsCapitalPricing,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    spot_info = rest_client.get_exchange_info(market="spot")
    futures_info = rest_client.get_exchange_info(market="futures")
    spot_symbols = {item["symbol"] for item in spot_info["symbols"]}
    futures_symbols = {item["symbol"] for item in futures_info["symbols"]}
    spot_active: list[str] = []
    futures_active: list[str] = []
    pending: list[str] = []
    for symbol in symbols:
        supported = False
        if symbol in spot_symbols:
            spot_active.append(symbol)
            supported = True
        if symbol in futures_symbols:
            futures_active.append(symbol)
            supported = True
        if not supported:
            pending.append(symbol)
    return tuple(spot_active), tuple(futures_active), tuple(pending)


def build_capital_adequacy_report(
    *,
    spot_available_balance_usd: float,
    futures_available_balance_usd: float,
    futures_execution_balance_usd: float | None = None,
    spot_recognized_balance_usd: float | None = None,
    spot_funding_assets: tuple[SpotFundingAsset, ...] = (),
    futures_recognized_balance_usd: float | None = None,
    futures_funding_assets: tuple[FuturesFundingAsset, ...] = (),
    settings: Settings,
    rest_client: SupportsCapitalPricing,
) -> CapitalAdequacyReport:
    spot_exchange_info = rest_client.get_exchange_info(market="spot")
    futures_exchange_info = rest_client.get_exchange_info(market="futures")
    spot_symbols, futures_symbols, pending_symbols = split_supported_symbols(
        symbols=settings.universe,
        rest_client=rest_client,
    )
    cap_fraction = settings.risk.max_symbol_notional_fraction
    spot_requirements: list[CapitalRequirement] = []
    futures_requirements: list[CapitalRequirement] = []
    for symbol in spot_symbols:
        min_notional = _symbol_min_notional(spot_exchange_info, symbol)
        min_equity = min_notional / cap_fraction
        buffered = min_equity * 1.25
        spot_requirements.append(
            CapitalRequirement(
                symbol=symbol,
                market="spot",
                min_notional_usd=min_notional,
                min_quantity=round(_symbol_min_quantity(spot_exchange_info, symbol), 8),
                min_equity_usd=round(min_equity, 6),
                buffered_min_equity_usd=round(buffered, 6),
            )
        )
    funding_assets_by_name = {item.asset: item for item in spot_funding_assets if item.free > 0.0}
    futures_funding_assets_by_name = {item.asset: item for item in futures_funding_assets if item.free > 0.0}
    extra_quote_assets = tuple(dict.fromkeys([*funding_assets_by_name, *futures_funding_assets_by_name]))
    spot_symbol_assets = {
        str(item.get("symbol", "")): _symbol_assets(
            exchange_info=spot_exchange_info,
            symbol=str(item.get("symbol", "")),
            extra_quote_assets=extra_quote_assets,
        )
        for item in spot_exchange_info.get("symbols", [])
        if str(item.get("symbol", ""))
    }
    asset_usd_price = {
        item.asset: (
            item.free_balance_usd / item.free
            if item.free > 0.0
            else (item.total_balance_usd / item.total if item.total > 0.0 else 0.0)
        )
        for item in spot_funding_assets
    }
    for item in futures_funding_assets:
        if item.asset in asset_usd_price:
            continue
        asset_usd_price[item.asset] = (
            item.free_balance_usd / item.free
            if item.free > 0.0
            else (item.total_balance_usd / item.total if item.total > 0.0 else 0.0)
        )
    asset_usd_price.setdefault("USDT", 1.0)
    spot_execution_routes: list[SpotExecutionRoute] = []
    exchange_id = str(getattr(rest_client, "exchange_id", "unknown") or "unknown").lower()

    def append_spot_execution_routes(
        *,
        funding_assets: dict[str, SpotFundingAsset | FuturesFundingAsset],
        funding_source_market: str,
        requires_wallet_transfer: bool,
    ) -> None:
        for requirement in spot_requirements:
            base_asset, target_quote_asset = spot_symbol_assets.get(requirement.symbol, ("", ""))
            if not base_asset:
                continue
            for funding_asset in funding_assets.values():
                if funding_asset.asset == base_asset or funding_asset.free_balance_usd <= 0.0:
                    continue
                if requires_wallet_transfer and not _exchange_supports_internal_transfer(
                    exchange_id=exchange_id,
                    source_market=funding_source_market,
                    target_market="spot",
                    asset=funding_asset.asset,
                ):
                    continue
                execution_symbol = ""
                route_type = "cross_quote"
                if funding_asset.asset == target_quote_asset:
                    execution_symbol = requirement.symbol
                    route_type = "direct"
                else:
                    execution_symbol = next(
                        (
                            symbol
                            for symbol, assets in spot_symbol_assets.items()
                            if assets == (base_asset, funding_asset.asset)
                        ),
                        "",
                    )
                if not execution_symbol:
                    continue
                quote_asset = spot_symbol_assets.get(execution_symbol, ("", ""))[1]
                quote_asset_usd = asset_usd_price.get(quote_asset, 0.0)
                if quote_asset_usd <= 0.0:
                    continue
                route_name = f"transfer_{route_type}" if requires_wallet_transfer else route_type
                spot_execution_routes.append(
                    SpotExecutionRoute(
                        target_symbol=requirement.symbol,
                        execution_symbol=execution_symbol,
                        base_asset=base_asset,
                        quote_asset=quote_asset,
                        funding_asset=funding_asset.asset,
                        funding_source_market=funding_source_market,
                        route_type=route_name,
                        requires_wallet_transfer=requires_wallet_transfer,
                        free_balance=round(funding_asset.free, 8),
                        free_balance_usd=round(funding_asset.free_balance_usd, 6),
                        quote_asset_usd_price=round(quote_asset_usd, 6),
                        min_notional_usd=round(_symbol_min_notional(spot_exchange_info, execution_symbol) * quote_asset_usd, 6),
                        min_quantity=round(_symbol_min_quantity(spot_exchange_info, execution_symbol), 8),
                    )
                )

    append_spot_execution_routes(
        funding_assets=funding_assets_by_name,
        funding_source_market="spot",
        requires_wallet_transfer=False,
    )
    append_spot_execution_routes(
        funding_assets=futures_funding_assets_by_name,
        funding_source_market="futures",
        requires_wallet_transfer=True,
    )
    capital_transfer_routes: list[CapitalTransferRoute] = []
    futures_execution_balance = max(
        0.0,
        futures_available_balance_usd if futures_execution_balance_usd is None else futures_execution_balance_usd,
    )
    existing_futures_collateral_candidate_balance_usd = sum(
        item.free_balance_usd
        for item in futures_funding_assets
        if item.asset != "USDT" and item.margin_available and item.free_balance_usd > 0.0
    )
    existing_futures_manual_handling_balance_usd = sum(
        item.free_balance_usd
        for item in futures_funding_assets
        if item.asset != "USDT" and not item.margin_available and item.free_balance_usd > 0.0
    )
    futures_collateral_candidate_balance_usd = existing_futures_collateral_candidate_balance_usd
    futures_manual_handling_balance_usd = existing_futures_manual_handling_balance_usd
    futures_transferable_execution_balance_usd = 0.0
    for funding_asset in funding_assets_by_name.values():
        if not _exchange_supports_internal_transfer(
            exchange_id=exchange_id,
            source_market="spot",
            target_market="futures",
            asset=funding_asset.asset,
        ):
            continue
        capacity_effect = "direct_execution"
        auto_transfer_supported = True
        note = f"Spot {funding_asset.asset} can be moved into futures as immediate execution balance."
        existing_futures_asset = futures_funding_assets_by_name.get(funding_asset.asset)
        if funding_asset.asset != "USDT":
            auto_transfer_supported = False
            if existing_futures_asset is not None and existing_futures_asset.margin_available:
                capacity_effect = "collateral_candidate"
                note = (
                    f"Spot {funding_asset.asset} can be transferred into futures and is recognized as "
                    "collateral in the current account mode, but it does not become immediate execution cash."
                )
            else:
                capacity_effect = "manual_conversion"
                note = (
                    f"Spot {funding_asset.asset} can be transferred into futures, but non-USDT assets still "
                    "require conversion or manual collateral handling before new futures execution."
                )
        capital_transfer_routes.append(
            CapitalTransferRoute(
                source_market="spot",
                target_market="futures",
                asset=funding_asset.asset,
                source_free_amount=round(funding_asset.free, 8),
                transferable_usd=round(funding_asset.free_balance_usd, 6),
                route_type="wallet_transfer",
                requires_manual_transfer=capacity_effect != "direct_execution",
                capacity_effect=capacity_effect,
                auto_transfer_supported=auto_transfer_supported,
                note=note,
            )
        )
        if capacity_effect == "direct_execution":
            futures_transferable_execution_balance_usd += funding_asset.free_balance_usd
        elif capacity_effect == "collateral_candidate":
            futures_collateral_candidate_balance_usd += funding_asset.free_balance_usd
        else:
            futures_manual_handling_balance_usd += funding_asset.free_balance_usd
    if (
        "USDT" not in funding_assets_by_name
        and spot_available_balance_usd > 0.0
        and _exchange_supports_internal_transfer(
            exchange_id=exchange_id,
            source_market="spot",
            target_market="futures",
            asset="USDT",
        )
    ):
        capital_transfer_routes.append(
            CapitalTransferRoute(
                source_market="spot",
                target_market="futures",
                asset="USDT",
                source_free_amount=round(spot_available_balance_usd, 8),
                transferable_usd=round(spot_available_balance_usd, 6),
                route_type="wallet_transfer",
                requires_manual_transfer=False,
                capacity_effect="direct_execution",
                auto_transfer_supported=True,
                note="Spot USDT can be moved into futures as immediate execution balance.",
            )
        )
        futures_transferable_execution_balance_usd += spot_available_balance_usd
    for funding_asset in futures_funding_assets_by_name.values():
        if not _exchange_supports_internal_transfer(
            exchange_id=exchange_id,
            source_market="futures",
            target_market="spot",
            asset=funding_asset.asset,
        ):
            continue
        note = (
            f"Futures {funding_asset.asset} can be transferred back to spot for spot execution funding."
            if funding_asset.asset == "USDT"
            else f"Futures {funding_asset.asset} can be transferred to spot and used on direct/cross-quote spot routes when the quote asset matches."
        )
        capital_transfer_routes.append(
            CapitalTransferRoute(
                source_market="futures",
                target_market="spot",
                asset=funding_asset.asset,
                source_free_amount=round(funding_asset.free, 8),
                transferable_usd=round(funding_asset.free_balance_usd, 6),
                route_type="wallet_transfer",
                requires_manual_transfer=False,
                capacity_effect="spot_execution",
                auto_transfer_supported=True,
                note=note,
            )
        )
    if (
        "USDT" not in futures_funding_assets_by_name
        and futures_available_balance_usd > 0.0
        and _exchange_supports_internal_transfer(
            exchange_id=exchange_id,
            source_market="futures",
            target_market="spot",
            asset="USDT",
        )
    ):
        capital_transfer_routes.append(
            CapitalTransferRoute(
                source_market="futures",
                target_market="spot",
                asset="USDT",
                source_free_amount=round(futures_available_balance_usd, 8),
                transferable_usd=round(futures_available_balance_usd, 6),
                route_type="wallet_transfer",
                requires_manual_transfer=False,
                capacity_effect="spot_execution",
                auto_transfer_supported=True,
                note="Futures available USDT can be transferred back to spot for spot execution funding.",
            )
        )
    max_spot_to_futures_transfer_usd = max((item.transferable_usd for item in capital_transfer_routes if item.source_market == "spot" and item.target_market == "futures"), default=0.0)
    max_futures_to_spot_transfer_usd = max((item.transferable_usd for item in capital_transfer_routes if item.source_market == "futures" and item.target_market == "spot"), default=0.0)
    for symbol in futures_symbols:
        min_notional = _symbol_min_notional(futures_exchange_info, symbol)
        min_equity = min_notional / cap_fraction
        buffered = min_equity * 1.25
        futures_requirements.append(
            CapitalRequirement(
                symbol=symbol,
                market="futures",
                min_notional_usd=min_notional,
                min_quantity=round(_symbol_min_quantity(futures_exchange_info, symbol), 8),
                min_equity_usd=round(min_equity, 6),
                buffered_min_equity_usd=round(buffered, 6),
            )
        )
    all_requirements = [*spot_requirements, *futures_requirements]
    minimum_operational = min(item.buffered_min_equity_usd for item in all_requirements)
    minimum_full_universe = max(item.buffered_min_equity_usd for item in all_requirements)
    recommended = round(minimum_full_universe * 1.6, 6)
    spot_recognized = spot_available_balance_usd if spot_recognized_balance_usd is None else spot_recognized_balance_usd
    futures_recognized = (
        futures_available_balance_usd if futures_recognized_balance_usd is None else futures_recognized_balance_usd
    )
    futures_total_reusable_balance_usd = (
        futures_recognized
        + existing_futures_collateral_candidate_balance_usd  # pre-existing non-USDT collateral in futures
        + existing_futures_manual_handling_balance_usd        # pre-existing non-USDT manual assets in futures (e.g. BTC)
        + futures_transferable_execution_balance_usd
        + max(futures_collateral_candidate_balance_usd - existing_futures_collateral_candidate_balance_usd, 0.0)
        + max(futures_manual_handling_balance_usd - existing_futures_manual_handling_balance_usd, 0.0)
    )
    if spot_funding_assets:
        can_trade_spot_any = any(
            route.target_symbol == requirement.symbol and route.free_balance_usd >= requirement.buffered_min_equity_usd
            for requirement in spot_requirements
            for route in spot_execution_routes
        )
    else:
        can_trade_spot_any = any(spot_recognized >= item.buffered_min_equity_usd for item in spot_requirements)
    can_trade_futures_any = any(
        futures_total_reusable_balance_usd >= item.buffered_min_equity_usd for item in futures_requirements
    )
    can_trade_any = can_trade_spot_any or can_trade_futures_any
    note = (
        "5% 수익 보장은 불가하며, 아래 금액은 현재 설정과 최소 주문 조건 기준의 운영 가능 자금 추정치입니다."
    )
    if pending_symbols:
        note += f" Pending symbols excluded from capital check: {', '.join(pending_symbols)}."
    return CapitalAdequacyReport(
        spot_available_balance_usd=round(spot_available_balance_usd, 6),
        spot_recognized_balance_usd=round(spot_recognized, 6),
        spot_funding_assets=tuple(spot_funding_assets),
        futures_funding_assets=tuple(futures_funding_assets),
        spot_execution_routes=tuple(
            sorted(
                spot_execution_routes,
                key=lambda item: (
                    item.target_symbol,
                    0 if not item.requires_wallet_transfer else 1,
                    0 if item.route_type.endswith("direct") else 1,
                    -item.free_balance_usd,
                ),
            )
        ),
        capital_transfer_routes=tuple(
            sorted(
                capital_transfer_routes,
                key=lambda item: (
                    item.source_market,
                    item.target_market,
                    _transfer_capacity_effect_priority(item.capacity_effect),
                    -item.transferable_usd,
                ),
            )
        ),
        futures_available_balance_usd=round(futures_available_balance_usd, 6),
        futures_execution_balance_usd=round(futures_execution_balance, 6),
        futures_recognized_balance_usd=round(futures_recognized, 6),
        futures_transferable_execution_balance_usd=round(futures_transferable_execution_balance_usd, 6),
        futures_collateral_candidate_balance_usd=round(futures_collateral_candidate_balance_usd, 6),
        futures_manual_handling_balance_usd=round(futures_manual_handling_balance_usd, 6),
        futures_total_reusable_balance_usd=round(futures_total_reusable_balance_usd, 6),
        minimum_operational_balance_usd=round(minimum_operational, 6),
        minimum_full_universe_balance_usd=round(minimum_full_universe, 6),
        recommended_balance_usd=recommended,
        can_trade_any=can_trade_any,
        can_trade_spot_any=can_trade_spot_any,
        can_trade_futures_any=can_trade_futures_any,
        max_spot_to_futures_transfer_usd=round(max_spot_to_futures_transfer_usd, 6),
        max_futures_to_spot_transfer_usd=round(max_futures_to_spot_transfer_usd, 6),
        spot_requirements=tuple(spot_requirements),
        futures_requirements=tuple(futures_requirements),
        pending_symbols=tuple(pending_symbols),
        note=note,
    )
