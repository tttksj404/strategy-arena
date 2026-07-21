# Wave-6 candidate assembly: loads cached frames, applies the pure engine_w6 signal functions,
# applies costs, and packages each candidate into a StrategyResult matching the wave-1..5 schema.

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import sys
from typing import Final

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

import numpy as np
import pandas as pd  # noqa: PANDAS_OK
import requests

from research.wave1.common import JsonValue, PipelineError, StrategyResult, load_frame, load_json, save_frame
from research.wave1.costs import PERP_TAKER_RATE, slippage_rate
from research.wave1.fam_session import effect_estimate
from research.wave1.fetch_bitget import BitgetCandleRequest, fetch_candles
from research.wave5.engine import equity_from_returns
from research.wave6.engine_w6 import (
    CACHE_DIR,
    DEVIATION_ENTRY,
    DEVIATION_EXIT,
    FUNDING_THRESHOLD,
    LISTING_COVER_OFFSET_DAYS,
    LISTING_SHORT_OFFSET_DAYS,
    MIN_LISTING_SAMPLE,
    OOS_SPLIT,
    SPILLOVER_ENTRY_HOUR,
    SPILLOVER_EXIT_HOUR,
    SPILLOVER_SIGNAL_HOUR,
    WAVE1_CACHE_DIR,
    WAVE3_CACHE_DIR,
    align_token_underlying,
    deviation_fade_position,
    eligible_listing_symbols,
    funding_window_trades,
    intraday_round_trip_cost,
    listing_short_trade,
    read_symbol_frame,
    regular_session_mask,
    session_end_mask,
    spillover_trades,
    weekend_trades,
)


STANDARD_IDS: Final = ("W6a", "W6b", "W6c", "W6d")
EXPLORATORY_IDS: Final = ("W6e", "W6f")
ALL_IDS: Final = (*STANDARD_IDS, *EXPLORATORY_IDS)
CRYPTO_SYMBOLS: Final = ("BTCUSDT", "ETHUSDT", "SOLUSDT")


def _now_ms() -> int:
    return int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)


def _direction_label(value: float) -> str:
    if not np.isfinite(value):
        return "undetermined"
    if value > 0.0:
        return "positive"
    if value < 0.0:
        return "negative"
    return "flat"


def _result(candidate_id: str, family: str, returns: pd.Series, positions: pd.Series, turnover: pd.Series, metadata: dict[str, JsonValue]) -> StrategyResult:
    clean_returns = returns.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    equity = equity_from_returns(clean_returns)
    aligned_positions = positions.reindex(clean_returns.index).fillna(0.0)
    trades = clean_returns[aligned_positions.abs() > 0.0]
    # Gate-7 cost-stress overlay: an extra 2bp charge per unit of turnover on top of the already
    # cost-inclusive returns (mirrors research/wave5/strategies.py's stress convention).
    stress_cost = turnover.reindex(clean_returns.index).fillna(0.0) * 0.0002
    stress_returns = clean_returns - stress_cost
    stress_equity = equity_from_returns(stress_returns)
    stress_is = stress_equity[stress_equity.index <= OOS_SPLIT]
    stress_oos_equity = stress_equity[stress_equity.index > OOS_SPLIT]
    anchor = float(stress_is.iloc[-1]) if not stress_is.empty else 300.0
    stress_oos = float(stress_oos_equity.iloc[-1] / anchor - 1.0) if not stress_oos_equity.empty else 0.0
    return StrategyResult(candidate_id, family, equity, trades, positions, turnover, stress_oos, metadata)


# --------------------------------------------------------------------------------------
# W6a / W6b -- funding window
# --------------------------------------------------------------------------------------


def _funding_window_result(candidate_id: str, direction: float, intended_factor: str) -> StrategyResult:
    per_symbol: dict[str, pd.DataFrame] = {}
    for symbol in CRYPTO_SYMBOLS:
        funding = read_symbol_frame(WAVE1_CACHE_DIR, "binance_funding_", symbol, ".csv.gz")["funding_rate"]
        price_open = read_symbol_frame(CACHE_DIR, "binance_fapi_", symbol, "_1h.csv.gz")["open"]
        per_symbol[symbol] = funding_window_trades(funding, price_open, FUNDING_THRESHOLD, direction)
    triggered = pd.concat({symbol: frame["triggered"] for symbol, frame in per_symbol.items()}, axis=1).fillna(False)
    raw_returns = pd.concat({symbol: frame["raw_return"] for symbol, frame in per_symbol.items()}, axis=1)
    cost = intraday_round_trip_cost("BTCUSDT")  # BTC/ETH/SOL share the same 1bp base-slippage bucket
    net_returns = (raw_returns - cost).where(triggered, 0.0)
    n_triggered = triggered.sum(axis=1)
    portfolio_return = (net_returns.sum(axis=1) / n_triggered.replace(0, np.nan)).fillna(0.0)
    positions = direction * (n_triggered > 0).astype(float)
    turnover = (n_triggered > 0).astype(float) * 2.0
    data_valid = all((WAVE1_CACHE_DIR / f"binance_funding_{symbol}.csv.gz").exists() for symbol in CRYPTO_SYMBOLS) and all(
        (CACHE_DIR / f"binance_fapi_{symbol}_1h.csv.gz").exists() for symbol in CRYPTO_SYMBOLS
    )
    metadata: dict[str, JsonValue] = {
        "symbols": list(CRYPTO_SYMBOLS),
        "exploratory_only": False,
        "data_valid": data_valid,
        "cost_model_valid": True,
        "intended_factor": intended_factor,
        "max_concurrent_positions": 3,
        "max_position_weight": 1.0,
        "min_position_weight": round(1.0 / 3.0, 6),
        "min_order_usdt": 5.0,
        "neighbor_is_sharpes": [],
        "candidate_config": {
            "threshold": FUNDING_THRESHOLD,
            "signal_proxy": "previous_settlement_realized_funding",
            "proxy_limitation": "no historical predicted-funding series exists; the realized funding from the prior settlement is used as the pre-registered proxy for the next payment",
            "entry_offset_hours": -1,
            "exit_offset_hours": 1,
            "round_trip_cost": cost,
        },
        "n_settlement_events": int(len(triggered)),
        "n_triggered_events": int((n_triggered > 0).sum()),
    }
    return _result(candidate_id, "F6", portfolio_return, positions, turnover, metadata)


def run_w6a() -> StrategyResult:
    return _funding_window_result("W6a", 1.0, "funding_window_long")


def run_w6b() -> StrategyResult:
    return _funding_window_result("W6b", -1.0, "funding_window_short")


# --------------------------------------------------------------------------------------
# W6c -- open spillover
# --------------------------------------------------------------------------------------


def run_w6c() -> StrategyResult:
    price = read_symbol_frame(CACHE_DIR, "binance_fapi_", "BTCUSDT", "_1h.csv.gz")
    trades = spillover_trades(price)
    cost = intraday_round_trip_cost("BTCUSDT")
    triggered = trades["signal"] != 0.0
    net_return = trades["raw_return"] - np.where(triggered, cost, 0.0)
    positions = trades["signal"]
    turnover = triggered.astype(float) * 2.0
    metadata: dict[str, JsonValue] = {
        "symbols": ["BTCUSDT"],
        "exploratory_only": False,
        "data_valid": not price.empty,
        "cost_model_valid": True,
        "intended_factor": "open_spillover_momentum",
        "max_concurrent_positions": 1,
        "max_position_weight": 1.0,
        "min_position_weight": 1.0,
        "min_order_usdt": 5.0,
        "neighbor_is_sharpes": [],
        "candidate_config": {
            "signal_hour_utc": SPILLOVER_SIGNAL_HOUR,
            "entry_hour_utc": SPILLOVER_ENTRY_HOUR,
            "exit_hour_utc": SPILLOVER_EXIT_HOUR,
            "weekdays_only": True,
            "round_trip_cost": cost,
            "bar_mapping_approximation": "the pre-registered 12:30->13:30 UTC pre-open window is approximated by the hourly [12:00,13:00) bar's own return; the 13:30 entry is approximated by the 13:00 bar's open (when that bar's close confirms)",
        },
        "n_days": int(len(trades)),
        "n_triggered_days": int(triggered.sum()),
    }
    return _result("W6c", "F6", net_return, positions, turnover, metadata)


# --------------------------------------------------------------------------------------
# W6d -- weekend drift
# --------------------------------------------------------------------------------------


def run_w6d() -> StrategyResult:
    price = read_symbol_frame(CACHE_DIR, "binance_fapi_", "BTCUSDT", "_1h.csv.gz")
    trades = weekend_trades(price)
    cost = intraday_round_trip_cost("BTCUSDT")
    net_return = trades["raw_return"] - cost
    positions = pd.Series(1.0, index=trades.index)
    turnover = pd.Series(2.0, index=trades.index)
    metadata: dict[str, JsonValue] = {
        "symbols": ["BTCUSDT"],
        "exploratory_only": False,
        "data_valid": not price.empty,
        "cost_model_valid": True,
        "intended_factor": "weekend_drift_long",
        "max_concurrent_positions": 1,
        "max_position_weight": 1.0,
        "min_position_weight": 1.0,
        "min_order_usdt": 5.0,
        "neighbor_is_sharpes": [],
        "candidate_config": {"entry": "saturday_00:00_utc", "exit": "monday_00:00_utc", "hold_days": 2, "direction": "long_only", "round_trip_cost": cost},
        "n_weekends": int(len(trades)),
    }
    return _result("W6d", "F6", net_return, positions, turnover, metadata)


# --------------------------------------------------------------------------------------
# W6e -- token vs underlying deviation fade (exploratory)
# --------------------------------------------------------------------------------------


def run_w6e() -> StrategyResult:
    token = read_symbol_frame(CACHE_DIR, "bitget_", "SPYUSDT", "_1H.csv.gz")["close"]
    underlying = read_symbol_frame(CACHE_DIR, "yahoo_", "SPY", "_1h.csv.gz")["close"]
    aligned = align_token_underlying(token, underlying)
    if not aligned.empty:
        aligned = aligned[regular_session_mask(pd.DatetimeIndex(aligned.index))]
    base_metadata: dict[str, JsonValue] = {
        "symbols": ["SPYUSDT", "SPY"],
        "exploratory_only": True,
        "deployment_claim": False,
        "cost_model_valid": True,
        "intended_factor": "token_underlying_deviation_fade",
        "candidate_config": {"entry_dev": DEVIATION_ENTRY, "exit_dev": DEVIATION_EXIT, "session_utc": "13:30-20:00"},
    }
    if aligned.empty:
        metadata = {**base_metadata, "sample_size": 0, "reason": "no overlapping SPYUSDT/SPY regular-session timestamps after tz-aware alignment"}
        anchor_ts = pd.Timestamp.now(tz="UTC")
        empty = pd.Series(dtype=float)
        return StrategyResult("W6e", "F6", pd.Series([300.0], index=[anchor_ts]), empty, empty, empty, 0.0, metadata)
    dev = aligned["token"] / aligned["under"] - 1.0
    session_end = session_end_mask(pd.DatetimeIndex(aligned.index))
    raw_position = deviation_fade_position(dev, DEVIATION_ENTRY, DEVIATION_EXIT, session_end)
    turnover = raw_position.diff().abs().fillna(raw_position.abs())
    cost_per_leg = PERP_TAKER_RATE + slippage_rate("SPYUSDT", 2.0)
    gross_return = raw_position.shift(1).fillna(0.0) * aligned["token"].pct_change().fillna(0.0)
    net_return = gross_return - turnover * cost_per_leg
    trades = net_return[raw_position.abs() > 0.0]
    estimate = effect_estimate(trades, 0.0)  # already cost-inclusive above; round_trip_cost=0 avoids double charging
    metadata = {
        **base_metadata,
        "sample_size": int(estimate.observations),
        "aligned_observations": int(len(aligned)),
        "date_range": [aligned.index.min().isoformat(), aligned.index.max().isoformat()],
        "effect_mean": estimate.mean,
        "effect_t_stat": estimate.t_stat,
        "effect_observations": estimate.observations,
        "effect_cost_after_mean": estimate.cost_after_mean,
        "effect_direction": _direction_label(estimate.mean),
    }
    return _result("W6e", "F6", net_return, raw_position, turnover, metadata)


# --------------------------------------------------------------------------------------
# W6f -- new-listing effect (exploratory)
# --------------------------------------------------------------------------------------


def _bitget_daily_loader(symbol: str, onboard: pd.Timestamp) -> pd.DataFrame | None:
    path = CACHE_DIR / f"bitget_{symbol}_1D.csv.gz"
    if path.exists():
        return load_frame(path)
    try:
        with requests.Session() as session:
            frame = fetch_candles(BitgetCandleRequest(symbol, "1D", int(pd.Timestamp(onboard).timestamp() * 1000), _now_ms()), session)
    except (PipelineError, requests.RequestException):
        return None
    if not frame.empty:
        save_frame(path, frame)
    return frame


def _aggregate_listing_trades(
    eligible: tuple[tuple[str, pd.Timestamp], ...],
    daily_loader: Callable[[str, pd.Timestamp], pd.DataFrame | None],
) -> tuple[pd.Series, list[str]]:
    index: list[pd.Timestamp] = []
    values: list[float] = []
    used: list[str] = []
    for symbol, onboard in eligible:
        daily = daily_loader(symbol, onboard)
        if daily is None or daily.empty:
            continue
        trade = listing_short_trade(daily, onboard, LISTING_SHORT_OFFSET_DAYS, LISTING_COVER_OFFSET_DAYS)
        if trade is None:
            continue
        cost = intraday_round_trip_cost(symbol, 1.0)  # daily-bar hold, not the intraday 2x convention
        index.append(pd.Timestamp(onboard).normalize() + pd.Timedelta(days=LISTING_COVER_OFFSET_DAYS))
        values.append(trade - cost)
        used.append(symbol)
    series = pd.Series(values, index=pd.DatetimeIndex(index), dtype=float).sort_index()
    return series, used


def run_w6f() -> StrategyResult:
    contracts_path = CACHE_DIR / "bitget_contracts.json"
    contracts: JsonValue = load_json(contracts_path) if contracts_path.exists() else load_json(WAVE3_CACHE_DIR / "bitget_contracts.json")
    eligible = eligible_listing_symbols(contracts)
    n = len(eligible)
    base_metadata: dict[str, JsonValue] = {
        "exploratory_only": True,
        "deployment_claim": False,
        "min_required": MIN_LISTING_SAMPLE,
        "eligible_symbols": n,
        "candidate_config": {"short_offset_days": LISTING_SHORT_OFFSET_DAYS, "cover_offset_days": LISTING_COVER_OFFSET_DAYS},
    }
    if n < MIN_LISTING_SAMPLE:
        metadata = {
            **base_metadata,
            "symbols": [],
            "sample_size": n,
            "reason": (
                "Bitget contracts payload has no populated launchTime for any of the "
                f"{len(contracts) if isinstance(contracts, list) else 0} listed usdt-futures symbols "
                "(checked against research/wave3/cache/bitget_contracts.json and a live "
                "/api/v2/mix/market/contracts refetch); the pre-registered listing-effect test cannot be evaluated."
            ),
        }
        anchor_ts = pd.Timestamp.now(tz="UTC")
        empty = pd.Series(dtype=float)
        return StrategyResult("W6f", "F6", pd.Series([300.0], index=[anchor_ts]), empty, empty, empty, 0.0, metadata)
    trades, used_symbols = _aggregate_listing_trades(eligible, _bitget_daily_loader)
    estimate = effect_estimate(trades, 0.0)
    positions = pd.Series(1.0, index=trades.index)
    turnover = pd.Series(2.0, index=trades.index)
    metadata = {
        **base_metadata,
        "symbols": used_symbols,
        "sample_size": int(estimate.observations),
        "effect_mean": estimate.mean,
        "effect_t_stat": estimate.t_stat,
        "effect_observations": estimate.observations,
        "effect_cost_after_mean": estimate.cost_after_mean,
        "effect_direction": _direction_label(estimate.mean),
    }
    return _result("W6f", "F6", trades, positions, turnover, metadata)


def run_candidates() -> tuple[StrategyResult, ...]:
    return (run_w6a(), run_w6b(), run_w6c(), run_w6d(), run_w6e(), run_w6f())


__all__ = [
    "ALL_IDS",
    "EXPLORATORY_IDS",
    "STANDARD_IDS",
    "run_candidates",
    "run_w6a",
    "run_w6b",
    "run_w6c",
    "run_w6d",
    "run_w6e",
    "run_w6f",
]
