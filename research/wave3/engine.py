"""Wave-3 candidate definitions, point-in-time features, and portfolio runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd  # noqa: PANDAS_OK

from research.scanner.scan_bitget import STOCK_BASE_COINS
from research.wave1.common import PipelineError, StrategyResult, load_frame, load_json
from research.wave1.costs import slippage_rate
from research.wave1.fam_tsmom import vol_target_fraction
from research.wave3.fetch import WAVE3_CACHE_DIR
from research.wave3.strategy import RankedSignal, cross_sectional_zscore, select_max_z_candidates, update_hysteresis
from research.wave3.universe import AssetListing, AssetType, eligible_symbols_at, parse_binance_um_listings, parse_bitget_stock_listings


W3_MAKER_FEE_RATE: Final = 0.0002
W3_CANDIDATE_IDS: Final = ("W3a", "W3b", "W3c", "W3d", "W3e", "W3f")


@dataclass(frozen=True, slots=True)
class CandidateConfig:
    candidate_id: str
    factor: str
    top_k: int
    rebalance_days: int
    threshold_apr: float = 0.0
    long_short: bool = False
    long_only: bool = False
    hysteresis: bool = False
    crypto_only: bool = False


W3_CANDIDATES: Final = (
    CandidateConfig("W3a", "carry", 3, 1, 0.05),
    CandidateConfig("W3b", "carry", 5, 1, 0.08),
    CandidateConfig("W3c", "momentum", 3, 7, long_short=True),
    CandidateConfig("W3d", "momentum", 3, 7, long_only=True),
    CandidateConfig("W3e", "max_z", 3, 1, hysteresis=True),
    CandidateConfig("W3f", "max_z", 3, 1, hysteresis=True, crypto_only=True),
)


@dataclass(frozen=True, slots=True)
class AssetMarket:
    listing: AssetListing
    perp: pd.DataFrame
    spot: pd.DataFrame | None
    funding: pd.Series


@dataclass(frozen=True, slots=True)
class Target:
    signal: RankedSignal
    direction: float
    weight: float


def _daily_close(frame: pd.DataFrame) -> pd.Series:
    return frame["close"].resample("1D").last().dropna()


def _daily_quote_volume(frame: pd.DataFrame) -> pd.Series:
    if "quote_volume" not in frame.columns:
        raise PipelineError("wave-3 universe requires quote_volume bars")
    return frame["quote_volume"].resample("1D").sum(min_count=1)


def _carry_apr(funding: pd.Series) -> pd.Series:
    daily = funding.resample("1D").sum(min_count=1)
    return daily.rolling(7, min_periods=7).mean() * 365.0


def _momentum(close: pd.Series) -> pd.Series:
    return close.pct_change(30).replace([np.inf, -np.inf], np.nan)


def _vol_target_targets(targets: tuple[Target, ...], returns: dict[str, pd.Series], day: pd.Timestamp) -> tuple[Target, ...]:
    scaled: list[Target] = []
    for target in targets:
        realized = returns[target.signal.symbol].rolling(20, min_periods=20).std(ddof=1).get(day)
        fraction = float(vol_target_fraction(pd.Series([realized])).iloc[0]) if pd.notna(realized) else 0.0
        if fraction > 0.0:
            scaled.append(Target(target.signal, target.direction, target.weight * fraction))
    gross = sum(target.weight for target in scaled)
    cap = min(1.0, 3.0 / gross) if gross > 3.0 else 1.0
    return tuple(Target(target.signal, target.direction, target.weight * cap) for target in scaled)


def btc_above_ma200(close: pd.Series) -> pd.Series:
    """Return the point-in-time BTC regime used by W3d."""
    ma200 = close.rolling(200, min_periods=200).mean()
    return (close >= ma200).where(ma200.notna(), False)


def load_listings(cache_dir: Path = WAVE3_CACHE_DIR) -> tuple[AssetListing, ...]:
    """Parse the exchange snapshots written by the fetch stage."""
    binance = load_json(cache_dir / "binance_exchange_info.json")
    spot = load_json(cache_dir / "spot_exchange_info.json")
    bitget = load_json(cache_dir / "bitget_contracts.json")
    spot_rows = spot.get("symbols") if isinstance(spot, dict) else None
    spot_symbols = {str(item["symbol"]) for item in spot_rows if isinstance(item, dict) and isinstance(item.get("symbol"), str)} if isinstance(spot_rows, list) else set()
    bitget_symbols = {str(item["symbol"]) for item in bitget if isinstance(item, dict) and isinstance(item.get("symbol"), str)} if isinstance(bitget, list) else set()
    bitget_onboard_dates: dict[str, pd.Timestamp] = {}
    if isinstance(bitget, list):
        for item in bitget:
            if not isinstance(item, dict) or not isinstance(item.get("symbol"), str):
                continue
            launch = item.get("launchTime", item.get("openTime"))
            # launchTime is an empty string for some contracts; those fall back to first-candle dating downstream.
            try:
                launch_ms = float(launch)
            except (TypeError, ValueError):
                continue
            bitget_onboard_dates[item["symbol"]] = pd.to_datetime(launch_ms, unit="ms", utc=True)
    crypto = parse_binance_um_listings(binance, spot_symbols, bitget_symbols, bitget_onboard_dates)
    stock_fallback: dict[str, pd.Timestamp] = {}
    for candle_path in cache_dir.glob("bitget_*USDT_1D.csv.gz"):
        stock_symbol = candle_path.name.removeprefix("bitget_").removesuffix("_1D.csv.gz")
        try:
            stock_fallback[stock_symbol] = pd.Timestamp(load_frame(candle_path).index.min())
        except (PipelineError, ValueError, OSError):
            continue
    stock = parse_bitget_stock_listings(bitget, set(STOCK_BASE_COINS), stock_fallback)
    return tuple(sorted((*crypto, *stock), key=lambda item: item.symbol))


def _source_path(cache_dir: Path, name: str) -> Path:
    wave3_path = (cache_dir / name).resolve()
    if wave3_path.parent != cache_dir.resolve() or not wave3_path.exists():
        raise PipelineError(f"required wave-3 source is missing: {name}")
    return wave3_path


def load_markets(listings: tuple[AssetListing, ...], cache_dir: Path = WAVE3_CACHE_DIR) -> dict[str, AssetMarket]:
    """Load wave-1-compatible crypto sources and wave-3 stock-token sources."""
    markets: dict[str, AssetMarket] = {}
    for listing in listings:
        if listing.asset_type is AssetType.CRYPTO:
            perp_name = f"binance_fapi_{listing.symbol}_1d.csv.gz"
            spot_name = f"binance_spot_{listing.symbol}_1d.csv.gz"
            funding_name = f"binance_funding_{listing.symbol}.csv.gz"
            perp = load_frame(_source_path(cache_dir, perp_name))
            spot = load_frame(_source_path(cache_dir, spot_name))
            funding = load_frame(_source_path(cache_dir, funding_name))["funding_rate"]
        else:
            candle_name = f"bitget_{listing.symbol}_1D.csv.gz"
            funding_name = f"bitget_funding_{listing.symbol}.csv.gz"
            perp = load_frame(_source_path(cache_dir, candle_name))
            spot = None
            funding = load_frame(_source_path(cache_dir, funding_name))["funding_rate"]
        markets[listing.symbol] = AssetMarket(listing, perp.sort_index(), spot, funding.sort_index())
    return markets


def _targets_for_day(
    config: CandidateConfig,
    day: pd.Timestamp,
    eligible: tuple[str, ...],
    carry: pd.DataFrame,
    momentum: pd.DataFrame,
    previous: tuple[RankedSignal, ...],
    regime_above: bool | None = None,
) -> tuple[Target, ...]:
    if config.candidate_id == "W3d" and regime_above is not True:
        return ()
    carry_day = cross_sectional_zscore(carry.loc[day, list(eligible)].dropna()) if day in carry.index else pd.Series(dtype=float)
    momentum_day = cross_sectional_zscore(momentum.loc[day, list(eligible)].dropna()) if day in momentum.index else pd.Series(dtype=float)
    if config.factor == "carry":
        carry_candidates = carry.loc[day, list(eligible)].dropna()
        carry_candidates = carry_candidates[carry_candidates > config.threshold_apr]
        ranked = tuple(
            RankedSignal(symbol, "carry", float(score), rank)
            for rank, (symbol, score) in enumerate(cross_sectional_zscore(carry_candidates).nlargest(config.top_k).items(), 1)
        )
    elif config.factor == "momentum":
        ranked = tuple(
            RankedSignal(symbol, "momentum", float(score), rank)
            for rank, (symbol, score) in enumerate(momentum_day.sort_values(ascending=False).head(config.top_k).items(), 1)
        )
        if config.long_short:
            short = momentum_day.drop(index=momentum_day.nlargest(config.top_k).index, errors="ignore").sort_values().head(config.top_k)
            ranked = tuple((*ranked, *(RankedSignal(symbol, "momentum", float(score), rank + config.top_k) for rank, (symbol, score) in enumerate(short.items(), 1))))
    else:
        ranked = select_max_z_candidates(carry_day, momentum_day, config.top_k * 4)
        if config.hysteresis:
            ranked = update_hysteresis(previous, ranked, config.top_k, 10)
    if config.rebalance_days > 1 and day.weekday() != 4:
        ranked = previous
    target_count = max(1, len(ranked))
    targets: list[Target] = []
    for signal in ranked:
        direction = 1.0
        if signal.factor == "momentum" and config.long_short and signal.rank > config.top_k:
            direction = -1.0
        if signal.factor == "momentum" and config.long_only:
            direction = 1.0
        targets.append(Target(signal, direction, 1.0 / target_count))
    return tuple(targets)


def _markets_data_valid(markets: dict[str, AssetMarket]) -> bool:
    return all(
        not market.perp.empty
        and market.perp["close"].replace([np.inf, -np.inf], np.nan).notna().all()
        and market.perp["quote_volume"].replace([np.inf, -np.inf], np.nan).notna().all()
        and not market.funding.empty
        and market.funding.replace([np.inf, -np.inf], np.nan).notna().all()
        for market in markets.values()
    )


def _run_candidate(
    markets: dict[str, AssetMarket],
    config: CandidateConfig,
    volume_limit: int = 150,
    stress_multiplier: float = 1.0,
) -> StrategyResult:
    """Run one frozen wave-3 candidate with next-day execution semantics."""
    if not markets:
        raise PipelineError("wave-3 market set is empty")
    closes = {symbol: _daily_close(market.perp) for symbol, market in markets.items()}
    volumes = {symbol: _daily_quote_volume(market.perp) for symbol, market in markets.items()}
    quote_frame = pd.DataFrame(volumes).sort_index()
    carry = pd.DataFrame({symbol: _carry_apr(market.funding) for symbol, market in markets.items()})
    for symbol, market in markets.items():
        if market.spot is None:
            carry[symbol] = np.nan
    momentum = pd.DataFrame({symbol: _momentum(close) for symbol, close in closes.items()})
    returns = {symbol: close.pct_change().fillna(0.0) for symbol, close in closes.items()}
    carry_returns = {symbol: pd.Series(0.0, index=closes[symbol].index) for symbol in markets}
    for symbol, market in markets.items():
        if market.spot is not None:
            spot_close = _daily_close(market.spot).reindex(closes[symbol].index)
            carry_returns[symbol] = spot_close.pct_change().fillna(0.0) - returns[symbol] + market.funding.resample("1D").sum().reindex(closes[symbol].index).fillna(0.0)
    days = sorted(pd.DatetimeIndex(pd.concat(closes.values()).index).unique())
    if not days:
        raise PipelineError("wave-3 market bars are empty")
    regime = btc_above_ma200(closes["BTCUSDT"]) if "BTCUSDT" in closes else pd.Series(False, index=pd.DatetimeIndex(days))
    listings = tuple(market.listing for market in markets.values() if not config.crypto_only or market.listing.asset_type is AssetType.CRYPTO)
    capital = 300.0
    equity: list[float] = []
    turnover: list[float] = []
    exposure: list[float] = []
    previous: tuple[RankedSignal, ...] = ()
    previous_weights: dict[str, Target] = {}
    trade_returns: list[float] = []
    trade_times: list[pd.Timestamp] = []
    max_concurrent_positions = 0
    max_position_weight = 0.0
    min_position_weight = float("inf")
    asset_contribution = {AssetType.CRYPTO.value: 0.0, AssetType.STOCK_TOKEN.value: 0.0}
    index = pd.DatetimeIndex(days)
    for day in index:
        day = pd.Timestamp(day)
        eligible = eligible_symbols_at(listings, quote_frame, day, volume_limit)
        targets = _targets_for_day(config, day, eligible, carry.reindex(index), momentum.reindex(index), previous, bool(regime.get(day, False)))
        if config.candidate_id == "W3c":
            targets = _vol_target_targets(targets, returns, day)
        current = {target.signal.symbol: target for target in targets if target.weight > 0.0}
        max_concurrent_positions = max(max_concurrent_positions, len(current))
        if current:
            max_position_weight = max(max_position_weight, *(target.weight for target in current.values()))
            min_position_weight = min(min_position_weight, *(target.weight for target in current.values()))
        daily_change = 0.0
        for symbol, target in previous_weights.items():
            sample = carry_returns[symbol] if target.signal.factor == "carry" else returns[symbol]
            contribution = target.direction * target.weight * float(sample.get(day, 0.0))
            daily_change += contribution
            asset_contribution[markets[symbol].listing.asset_type.value] += contribution
        capital *= 1.0 + daily_change
        changes = 0.0
        cost_return = 0.0
        for symbol in sorted(set(previous_weights) | set(current)):
            old = previous_weights.get(symbol)
            new = current.get(symbol)
            old_signed = old.direction * old.weight if old is not None else 0.0
            new_signed = new.direction * new.weight if new is not None else 0.0
            delta = abs(new_signed - old_signed)
            if delta == 0.0:
                continue
            changes += delta
            factor = new.signal.factor if new is not None else old.signal.factor if old is not None else "momentum"
            legs = 2.0 if factor == "carry" else 1.0
            cost_return += delta * legs * (W3_MAKER_FEE_RATE + stress_multiplier * slippage_rate(symbol))
        capital *= max(0.0, 1.0 - cost_return)
        equity.append(capital)
        turnover.append(changes)
        exposure.append(sum(abs(target.weight) for target in current.values()))
        if daily_change != 0.0:
            trade_returns.append(daily_change)
            trade_times.append(day)
        previous_weights = current
        previous = tuple(target.signal for target in targets)
    equity_series = pd.Series(equity, index=index, dtype=float)
    trade_series = pd.Series(trade_returns, index=pd.DatetimeIndex(trade_times), dtype=float)
    return StrategyResult(config.candidate_id, "F4", equity_series, trade_series, pd.Series(exposure, index=index), pd.Series(turnover, index=index), 0.0, {
        "symbols": sorted(markets),
        "exploratory_only": False,
        "cost_model_valid": W3_MAKER_FEE_RATE > 0.0 and all(slippage_rate(symbol) >= 0.0 for symbol in markets),
        "data_valid": _markets_data_valid(markets),
        "intended_factor": config.factor,
        "candidate_config": asdict(config),
        "max_concurrent_positions": max_concurrent_positions,
        "max_position_weight": max_position_weight,
        "min_position_weight": 0.0 if min_position_weight == float("inf") else min_position_weight,
        "min_order_usdt": 5.0,
        "asset_type_gross_contribution": asset_contribution,
        "crypto_symbols": sorted(symbol for symbol, market in markets.items() if market.listing.asset_type is AssetType.CRYPTO),
        "stock_token_symbols": sorted(symbol for symbol, market in markets.items() if market.listing.asset_type is AssetType.STOCK_TOKEN),
    })


def _annualized_sharpe(returns: pd.Series) -> float:
    clean = returns.replace([np.inf, -np.inf], np.nan).dropna()
    volatility = float(clean.std(ddof=1)) if len(clean) > 1 else 0.0
    return float(clean.mean() / volatility * np.sqrt(365.0)) if volatility > 0.0 else 0.0


def _oos_return(equity: pd.Series) -> float:
    split = pd.Timestamp("2025-09-30T23:59:59Z")
    is_equity = equity[equity.index <= split]
    oos_equity = equity[equity.index > split]
    anchor = float(is_equity.iloc[-1]) if not is_equity.empty else 300.0
    return float(oos_equity.iloc[-1] / anchor - 1.0) if not oos_equity.empty else 0.0


def run_candidate(
    markets: dict[str, AssetMarket],
    config: CandidateConfig,
    volume_limit: int = 150,
    stress_multiplier: float = 1.0,
) -> StrategyResult:
    result = _run_candidate(markets, config, volume_limit, stress_multiplier)
    stress = _run_candidate(markets, config, volume_limit, max(2.0, stress_multiplier))
    metadata = {
        **result.metadata,
        "neighbor_is_sharpes": [_annualized_sharpe(result.trade_returns), _annualized_sharpe(stress.trade_returns)],
        "stress_multiplier": max(2.0, stress_multiplier),
    }
    return replace(result, stress_total_return=_oos_return(stress.equity), metadata=metadata)


__all__ = ["W3_CANDIDATES", "W3_CANDIDATE_IDS", "AssetMarket", "CandidateConfig", "btc_above_ma200", "load_listings", "load_markets", "run_candidate"]
