from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from datetime import timezone, datetime, timedelta
from pathlib import Path
from statistics import mean, pstdev
from typing import Any
from uuid import uuid4

from quant_binance.backtest.comparison import (
    ComparisonService,
    DirectionalHoldBaselineService,
    MeanReversionBaselineService,
    MomentumBaselineService,
)
from quant_binance.data.state import KlineBar, SpotTrade, SymbolMarketState, TopOfBook
from quant_binance.features.extractor import MarketFeatureExtractor
from quant_binance.features.primitive import FeatureHistoryContext, PrimitiveInputs
from quant_binance.models import DecisionIntent
from quant_binance.paths import prepare_run_paths
from quant_binance.settings import Settings
from quant_binance.strategy.normalize import clamp, zscore_to_unit


_REQUIRED_WARMUP_5M_BARS = 252
_SUPPORTED_MODE = "paper-live-shell"


@dataclass(frozen=True)
class RecentComparisonSource:
    run_dir: str
    decisions_path: str
    events_path: str
    summary_path: str
    state_path: str
    decision_count: int
    convertible_decision_count: int
    overlap_symbols: tuple[str, ...]
    first_decision_time: str
    last_decision_time: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PreparedRecentComparison:
    source: RecentComparisonSource
    fixture_path: str
    preparation_report_path: str
    cycle_count: int
    skipped_decision_count: int
    skipped_before_warmup_count: int
    equity_usd: float
    capacity_usd: float
    symbol_cycle_counts: dict[str, int]
    represented_inputs: tuple[str, ...]
    missing_inputs: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "source": self.source.as_dict(),
            "fixture_path": self.fixture_path,
            "preparation_report_path": self.preparation_report_path,
            "cycle_count": self.cycle_count,
            "skipped_decision_count": self.skipped_decision_count,
            "skipped_before_warmup_count": self.skipped_before_warmup_count,
            "equity_usd": self.equity_usd,
            "capacity_usd": self.capacity_usd,
            "symbol_cycle_counts": dict(self.symbol_cycle_counts),
            "represented_inputs": list(self.represented_inputs),
            "missing_inputs": list(self.missing_inputs),
        }


@dataclass(frozen=True)
class _DecisionRecord:
    candidate_mode: str
    config_version: str
    decision_hash: str
    decision_id: str
    divergence_code: str
    exit_reason_code: str
    final_mode: str
    symbol: str
    side: str
    snapshot_id: str
    timestamp: datetime
    estimated_round_trip_cost_bps: float
    gross_expected_edge_bps: float
    liquidity_score: float
    linked_order_ids: tuple[str, ...]
    net_expected_edge_bps: float
    order_intent_notional_usd: float
    volume_confirmation: float
    overheat_penalty: float
    predictability_score: float
    rejection_reasons: tuple[str, ...]
    stop_distance_bps: float
    trend_direction: int
    trend_strength: float
    volatility_penalty: float


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _event_timestamp(payload: dict[str, Any], fallback: datetime) -> datetime:
    data = payload.get("data", {})
    for key in ("E", "time", "T", "t"):
        raw = data.get(key)
        if raw is not None:
            return datetime.fromtimestamp(float(raw) / 1000.0, tz=timezone.utc)
    return fallback


def _floor_interval(start: datetime, *, hours: int) -> datetime:
    floored_hour = start.hour - (start.hour % hours)
    return start.replace(hour=floored_hour, minute=0, second=0, microsecond=0)


def _serialize_bar(bar: KlineBar) -> dict[str, object]:
    return {
        "symbol": bar.symbol,
        "interval": bar.interval,
        "start_time": bar.start_time.isoformat(),
        "close_time": bar.close_time.isoformat(),
        "open_price": bar.open_price,
        "high_price": bar.high_price,
        "low_price": bar.low_price,
        "close_price": bar.close_price,
        "volume": bar.volume,
        "quote_volume": bar.quote_volume,
        "is_closed": bar.is_closed,
    }


def _serialize_trade(trade: SpotTrade) -> dict[str, object]:
    return {
        "symbol": trade.symbol,
        "price": trade.price,
        "quantity": trade.quantity,
        "event_time": trade.event_time.isoformat(),
        "is_buyer_maker": trade.is_buyer_maker,
    }


def _serialize_cycle(
    *,
    decision_time: datetime,
    symbol: str,
    state: SymbolMarketState,
    primitive_inputs: PrimitiveInputs,
    history: FeatureHistoryContext,
) -> dict[str, object]:
    return {
        "decision_time": decision_time.isoformat(),
        "symbol": symbol,
        "state": {
            "top_of_book": {
                "bid_price": state.top_of_book.bid_price,
                "bid_qty": state.top_of_book.bid_qty,
                "ask_price": state.top_of_book.ask_price,
                "ask_qty": state.top_of_book.ask_qty,
                "updated_at": state.top_of_book.updated_at.isoformat(),
            },
            "last_trade_price": state.last_trade_price,
            "funding_rate": state.funding_rate,
            "open_interest": state.open_interest,
            "basis_bps": state.basis_bps,
            "last_update_time": state.last_update_time.isoformat(),
            "trades": [_serialize_trade(trade) for trade in state.trades],
            "klines": {
                interval: [_serialize_bar(bar) for bar in bars]
                for interval, bars in state.klines.items()
            },
            "order_book_imbalance_samples": list(state.order_book_imbalance_samples),
            "funding_rate_samples": list(state.funding_rate_samples),
            "basis_bps_samples": list(state.basis_bps_samples),
            "open_interest_samples": list(state.open_interest_samples),
        },
        "primitive_inputs": asdict(primitive_inputs),
        "history": asdict(history),
    }


def _load_decisions(path: Path) -> list[_DecisionRecord]:
    decisions: list[_DecisionRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            decisions.append(
                _DecisionRecord(
                    candidate_mode=str(raw.get("candidate_mode", raw.get("final_mode", "cash"))),
                    config_version=str(raw.get("config_version", "")),
                    decision_hash=str(raw.get("decision_hash", "")),
                    decision_id=str(raw.get("decision_id", str(uuid4()))),
                    divergence_code=str(raw.get("divergence_code", "")),
                    exit_reason_code=str(raw.get("exit_reason_code", "")),
                    final_mode=str(raw.get("final_mode", "cash")),
                    symbol=raw["symbol"],
                    side=str(raw.get("side", "flat")),
                    snapshot_id=str(raw.get("snapshot_id", "")),
                    timestamp=_parse_timestamp(raw["timestamp"]),
                    estimated_round_trip_cost_bps=float(raw.get("estimated_round_trip_cost_bps", 0.0)),
                    gross_expected_edge_bps=float(raw.get("gross_expected_edge_bps", 0.0)),
                    liquidity_score=float(raw.get("liquidity_score", 0.0)),
                    linked_order_ids=tuple(str(item) for item in raw.get("linked_order_ids", [])),
                    net_expected_edge_bps=float(raw.get("net_expected_edge_bps", 0.0)),
                    order_intent_notional_usd=float(raw.get("order_intent_notional_usd", 0.0)),
                    volume_confirmation=float(raw.get("volume_confirmation", 0.0)),
                    overheat_penalty=float(raw.get("overheat_penalty", 0.0)),
                    predictability_score=float(raw.get("predictability_score", 0.0)),
                    rejection_reasons=tuple(str(item) for item in raw.get("rejection_reasons", [])),
                    stop_distance_bps=float(raw.get("stop_distance_bps", 0.0)),
                    trend_direction=int(raw.get("trend_direction", 0)),
                    trend_strength=float(raw.get("trend_strength", 0.0)),
                    volatility_penalty=float(raw.get("volatility_penalty", 0.0)),
                )
            )
    decisions.sort(key=lambda item: (item.timestamp, item.symbol))
    return decisions


def _load_closed_5m_klines(path: Path) -> dict[str, list[KlineBar]]:
    bars_by_symbol: dict[str, dict[datetime, KlineBar]] = defaultdict(dict)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            payload = raw.get("payload", {})
            data = payload.get("data", {})
            kline = data.get("k")
            if not isinstance(kline, dict):
                continue
            if kline.get("i") != "5m" or not kline.get("x"):
                continue
            symbol = data.get("s")
            if not symbol:
                continue
            start_time = datetime.fromtimestamp(int(kline["t"]) / 1000.0, tz=timezone.utc)
            close_time = datetime.fromtimestamp(int(kline["T"]) / 1000.0, tz=timezone.utc)
            bars_by_symbol[symbol][start_time] = KlineBar(
                symbol=symbol,
                interval="5m",
                start_time=start_time,
                close_time=close_time,
                open_price=float(kline["o"]),
                high_price=float(kline["h"]),
                low_price=float(kline["l"]),
                close_price=float(kline["c"]),
                volume=float(kline["v"]),
                quote_volume=float(kline["q"]),
                is_closed=True,
            )
    return {
        symbol: sorted(symbol_bars.values(), key=lambda bar: bar.start_time)
        for symbol, symbol_bars in bars_by_symbol.items()
    }


def _aggregate_interval(bars_5m: list[KlineBar], *, interval: str, hours: int) -> list[KlineBar]:
    grouped: dict[datetime, list[KlineBar]] = defaultdict(list)
    expected = hours * 12
    for bar in bars_5m:
        grouped[_floor_interval(bar.start_time, hours=hours)].append(bar)
    aggregated: list[KlineBar] = []
    for bucket_start in sorted(grouped):
        bucket = sorted(grouped[bucket_start], key=lambda bar: bar.start_time)
        if len(bucket) != expected:
            continue
        aggregated.append(
            KlineBar(
                symbol=bucket[0].symbol,
                interval=interval,
                start_time=bucket_start,
                close_time=bucket[-1].close_time,
                open_price=bucket[0].open_price,
                high_price=max(item.high_price for item in bucket),
                low_price=min(item.low_price for item in bucket),
                close_price=bucket[-1].close_price,
                volume=round(sum(item.volume for item in bucket), 6),
                quote_volume=round(sum(item.quote_volume for item in bucket), 6),
                is_closed=True,
            )
        )
    return aggregated


def _bars_before(bars: list[KlineBar], decision_time: datetime) -> list[KlineBar]:
    return [bar for bar in bars if bar.start_time < decision_time]


def _estimate_spread_components(
    *,
    estimated_round_trip_cost_bps: float,
    settings: Settings,
) -> tuple[float, float]:
    fee_bps = max(settings.fees.futures_taker_fee_bps, 0.0) * 2.0
    residual = max(estimated_round_trip_cost_bps - fee_bps, 0.0)
    if residual <= 0.0:
        spread_bps = min(settings.feature_thresholds.spread_bps_ceiling * 0.1, 0.8)
        return spread_bps, spread_bps * 1.5
    spread_bps = residual / 2.5
    return round(spread_bps, 6), round(spread_bps * 1.5, 6)


def _synthetic_samples(target: float, *, count: int, scale: float) -> list[float]:
    bounded = clamp(target)
    if count <= 1:
        return [round(bounded * scale, 6)]
    return [round((index / (count - 1)) * scale, 6) for index in range(count)]


def _build_synthetic_state(
    *,
    decision: _DecisionRecord,
    bars_5m: list[KlineBar],
    settings: Settings,
) -> tuple[SymbolMarketState, list[KlineBar], list[KlineBar]]:
    bars_1h = _aggregate_interval(bars_5m, interval="1h", hours=1)
    bars_4h = _aggregate_interval(bars_5m, interval="4h", hours=4)
    if len(bars_1h) < 21 or len(bars_4h) < 2:
        raise ValueError("insufficient aggregated history")

    current_bar = bars_5m[-1]
    current_price = current_bar.close_price
    spread_bps, _ = _estimate_spread_components(
        estimated_round_trip_cost_bps=decision.estimated_round_trip_cost_bps,
        settings=settings,
    )
    spread_price = max(current_price * spread_bps / 10000.0, current_price * 1e-6)
    bid_price = max(current_price - (spread_price / 2.0), current_price * 0.5)
    ask_price = max(current_price + (spread_price / 2.0), bid_price)
    spread_norm = clamp(spread_bps / max(settings.feature_thresholds.spread_bps_ceiling, 1e-9))
    probe_norm = clamp((spread_bps * 1.5) / max(settings.feature_thresholds.slippage_bps_ceiling, 1e-9))
    book_stability = 1.0
    base_liquidity = 0.35 * (1.0 - spread_norm) + 0.20 * (1.0 - probe_norm) + 0.10 * book_stability
    depth_norm = clamp((decision.liquidity_score - base_liquidity) / 0.35)
    depth_usd = max(depth_norm * settings.feature_thresholds.depth_usd_target, current_price * 2.0)
    quantity_per_side = max(depth_usd / max(current_price * 2.0, 1e-9), 1e-6)

    funding_scale = 0.0002
    basis_scale = 25.0
    funding_rate = round(clamp(decision.overheat_penalty) * funding_scale, 8)
    basis_bps = round(clamp(decision.overheat_penalty) * basis_scale, 6)
    open_interest_base = max(mean(bar.quote_volume for bar in bars_1h[-5:]), 1.0)
    open_interest = round(open_interest_base * (1.0 + clamp(decision.overheat_penalty) * 0.15), 6)
    open_interest_samples = [
        round(open_interest_base * (0.9 + (index / 19.0) * 0.2), 6)
        for index in range(20)
    ]

    state = SymbolMarketState(
        symbol=decision.symbol,
        top_of_book=TopOfBook(
            bid_price=round(bid_price, 6),
            bid_qty=round(quantity_per_side, 6),
            ask_price=round(ask_price, 6),
            ask_qty=round(quantity_per_side, 6),
            updated_at=decision.timestamp,
        ),
        last_trade_price=round(current_price, 6),
        funding_rate=funding_rate,
        open_interest=open_interest,
        basis_bps=basis_bps,
        last_update_time=decision.timestamp,
        klines={
            "5m": list(bars_5m),
            "1h": list(bars_1h),
            "4h": list(bars_4h),
        },
        order_book_imbalance_samples=[0.0] * 30,
        funding_rate_samples=_synthetic_samples(decision.overheat_penalty, count=20, scale=funding_scale),
        basis_bps_samples=_synthetic_samples(decision.overheat_penalty, count=20, scale=basis_scale),
        open_interest_samples=open_interest_samples,
    )
    return state, bars_1h, bars_4h


def _target_taker_imbalance_norm(*, decision: _DecisionRecord, history: FeatureHistoryContext, current_5m_quote: float, current_1h_quote: float) -> float:
    mean_5m = mean(history.quote_volume_5m) if history.quote_volume_5m else 0.0
    std_5m = pstdev(history.quote_volume_5m) if len(history.quote_volume_5m) > 1 else 0.0
    mean_1h = mean(history.quote_volume_1h) if history.quote_volume_1h else 0.0
    std_1h = pstdev(history.quote_volume_1h) if len(history.quote_volume_1h) > 1 else 0.0
    vol_z_5m_norm = zscore_to_unit(current_5m_quote, mean_5m, std_5m)
    vol_z_1h_norm = zscore_to_unit(current_1h_quote, mean_1h, std_1h)
    required = (decision.volume_confirmation - (0.40 * vol_z_5m_norm) - (0.35 * vol_z_1h_norm)) / 0.25
    return clamp(required)


def _attach_synthetic_trades(
    *,
    state: SymbolMarketState,
    history: FeatureHistoryContext,
    decision: _DecisionRecord,
) -> None:
    current_5m_quote = state.klines["5m"][-1].quote_volume
    current_1h_quote = state.klines["1h"][-1].quote_volume
    taker_imbalance_norm = _target_taker_imbalance_norm(
        decision=decision,
        history=history,
        current_5m_quote=current_5m_quote,
        current_1h_quote=current_1h_quote,
    )
    buy_share = clamp((taker_imbalance_norm - 0.5) + 0.5)
    total_quote = max(current_5m_quote, state.last_trade_price)
    buy_quote = total_quote * buy_share
    sell_quote = max(total_quote - buy_quote, 0.0)
    price = max(state.last_trade_price, 1e-9)
    state.trades = [
        SpotTrade(
            symbol=state.symbol,
            price=price,
            quantity=round(buy_quote / price, 8),
            event_time=state.last_update_time - timedelta(seconds=10),
            is_buyer_maker=False,
        ),
        SpotTrade(
            symbol=state.symbol,
            price=price,
            quantity=round(sell_quote / price, 8),
            event_time=state.last_update_time - timedelta(seconds=5),
            is_buyer_maker=True,
        ),
    ]


def _primitive_inputs_for_decision(
    *,
    state: SymbolMarketState,
    decision: _DecisionRecord,
    settings: Settings,
) -> tuple[PrimitiveInputs, FeatureHistoryContext]:
    extractor = MarketFeatureExtractor(settings)
    history = extractor.build_history_context(state)
    _attach_synthetic_trades(state=state, history=history, decision=decision)
    base_inputs = extractor.build_primitive_inputs(state)
    lookback = state.klines["1h"][-21:-1]
    desired_trend_direction = decision.trend_direction if decision.trend_direction != 0 else base_inputs.trend_direction
    breakout_reference_price = base_inputs.breakout_reference_price
    if lookback:
        breakout_reference_price = (
            max(bar.high_price for bar in lookback)
            if desired_trend_direction >= 0
            else min(bar.low_price for bar in lookback)
        )
    primitive_inputs = replace(
        base_inputs,
        trend_direction=desired_trend_direction,
        ema_stack_score=1.0 if desired_trend_direction != 0 else 0.0,
        breakout_reference_price=round(breakout_reference_price, 6),
        gross_expected_edge_bps=round(
            decision.gross_expected_edge_bps if decision.gross_expected_edge_bps > 0.0 else base_inputs.gross_expected_edge_bps,
            6,
        ),
        funding_rate=state.funding_rate,
        open_interest=state.open_interest,
        open_interest_ema=round(mean(state.open_interest_samples[-8:]), 6),
        basis_bps=state.basis_bps,
    )
    return primitive_inputs, history


def _load_capital_defaults(summary_path: Path, state_path: Path) -> tuple[float, float]:
    default_equity = 10000.0
    default_capacity = 5000.0
    for path in (summary_path, state_path):
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        capital_report = payload.get("capital_report", {})
        if not isinstance(capital_report, dict):
            continue
        equity = float(capital_report.get("futures_recognized_balance_usd", 0.0) or 0.0)
        execution = float(capital_report.get("futures_execution_balance_usd", 0.0) or 0.0)
        if equity > 0.0:
            default_equity = equity
            default_capacity = equity
            if execution > 0.0:
                default_capacity = max(execution, min(equity, execution))
            break
    return round(default_equity, 6), round(default_capacity, 6)


def _evaluate_recent_candidate(run_dir: Path) -> RecentComparisonSource | None:
    decisions_path = run_dir / "logs" / "decisions.jsonl"
    events_path = run_dir / "logs" / "events.jsonl"
    summary_path = run_dir / "summary.json"
    state_path = run_dir / "summary.state.json"
    if not decisions_path.exists() or not events_path.exists() or not summary_path.exists() or not state_path.exists():
        return None
    decisions = _load_decisions(decisions_path)
    bars_by_symbol = _load_closed_5m_klines(events_path)
    overlap_symbols = tuple(sorted(set(bars_by_symbol) & {item.symbol for item in decisions}))
    convertible = 0
    for decision in decisions:
        bars = bars_by_symbol.get(decision.symbol, [])
        if len(_bars_before(bars, decision.timestamp)) >= _REQUIRED_WARMUP_5M_BARS:
            convertible += 1
    if not decisions or not overlap_symbols or convertible <= 0:
        return None
    return RecentComparisonSource(
        run_dir=str(run_dir),
        decisions_path=str(decisions_path),
        events_path=str(events_path),
        summary_path=str(summary_path),
        state_path=str(state_path),
        decision_count=len(decisions),
        convertible_decision_count=convertible,
        overlap_symbols=overlap_symbols,
        first_decision_time=decisions[0].timestamp.isoformat(),
        last_decision_time=decisions[-1].timestamp.isoformat(),
    )


def select_best_recent_source(*, base_dir: str | Path, mode: str = _SUPPORTED_MODE) -> RecentComparisonSource:
    mode_root = Path(base_dir) / "output" / mode
    candidates: list[tuple[tuple[int, int, float], RecentComparisonSource]] = []
    for run_dir in mode_root.iterdir():
        if not run_dir.is_dir() or run_dir.name == "latest":
            continue
        source = _evaluate_recent_candidate(run_dir)
        if source is None:
            continue
        mtime = (run_dir / "summary.state.json").stat().st_mtime
        score = (source.convertible_decision_count, len(source.overlap_symbols), mtime)
        candidates.append((score, source))
    if not candidates:
        raise FileNotFoundError(f"no recent {mode} run contains both decisions and enough 5m kline history")
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def prepare_recent_comparison_fixture(
    *,
    config_path: str | Path,
    base_dir: str | Path,
    fixture_path: str | Path,
    preparation_report_path: str | Path,
    run_dir: str | Path | None = None,
    mode: str = _SUPPORTED_MODE,
    equity_usd: float | None = None,
    capacity_usd: float | None = None,
) -> PreparedRecentComparison:
    del mode
    settings = Settings.load(config_path)
    source = _evaluate_recent_candidate(Path(run_dir)) if run_dir is not None else select_best_recent_source(base_dir=base_dir)
    if source is None:
        raise FileNotFoundError("the requested run does not contain enough recent local data for comparison prep")

    decisions = _load_decisions(Path(source.decisions_path))
    bars_by_symbol = _load_closed_5m_klines(Path(source.events_path))
    cycles: list[dict[str, object]] = []
    skipped_before_warmup = 0
    symbol_cycle_counts: Counter[str] = Counter()
    for decision in decisions:
        bars = _bars_before(bars_by_symbol.get(decision.symbol, []), decision.timestamp)
        if len(bars) < _REQUIRED_WARMUP_5M_BARS:
            skipped_before_warmup += 1
            continue
        try:
            state, _, _ = _build_synthetic_state(
                decision=decision,
                bars_5m=bars,
                settings=settings,
            )
            primitive_inputs, history = _primitive_inputs_for_decision(
                state=state,
                decision=decision,
                settings=settings,
            )
        except ValueError:
            skipped_before_warmup += 1
            continue
        cycles.append(
            _serialize_cycle(
                decision_time=decision.timestamp,
                symbol=decision.symbol,
                state=state,
                primitive_inputs=primitive_inputs,
                history=history,
            )
        )
        symbol_cycle_counts[decision.symbol] += 1

    if not cycles:
        raise ValueError("recent local data did not yield any convertible comparison cycles")

    fixture_target = Path(fixture_path)
    fixture_target.parent.mkdir(parents=True, exist_ok=True)
    fixture_target.write_text(json.dumps({"cycles": cycles}, indent=2), encoding="utf-8")

    default_equity, default_capacity = _load_capital_defaults(Path(source.summary_path), Path(source.state_path))
    prepared = PreparedRecentComparison(
        source=source,
        fixture_path=str(fixture_target),
        preparation_report_path=str(Path(preparation_report_path)),
        cycle_count=len(cycles),
        skipped_decision_count=len(decisions) - len(cycles),
        skipped_before_warmup_count=skipped_before_warmup,
        equity_usd=round(default_equity if equity_usd is None else equity_usd, 6),
        capacity_usd=round(default_capacity if capacity_usd is None else capacity_usd, 6),
        symbol_cycle_counts=dict(symbol_cycle_counts),
        represented_inputs=(
            "decision timestamps and symbols from decisions.jsonl",
            "5m/1h/4h price and quote-volume history from closed kline events",
            "gross expected edge and trend direction from recorded decisions",
            "synthetic top-of-book spread/depth calibrated from recorded cost and liquidity",
        ),
        missing_inputs=(
            "full historical book-ticker depth ladder",
            "dense trade tape per decision",
            "historical funding, basis, and open-interest streams for every decision time",
            "exact original primitive inputs and feature history from live runtime memory",
        ),
    )
    report_target = Path(preparation_report_path)
    report_target.parent.mkdir(parents=True, exist_ok=True)
    report_target.write_text(json.dumps(prepared.as_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return prepared


def default_recent_comparison_output_root(*, base_dir: str | Path) -> Path:
    run_paths = prepare_run_paths(
        base_dir=base_dir,
        mode="output/strategy-comparison-recent",
    )
    return run_paths.root


class RecordedDecisionComparisonService:
    name = "current_strategy"

    def __init__(self, settings: Settings, decisions: list[_DecisionRecord]) -> None:
        self.settings = settings
        self._decisions = {
            (item.timestamp.isoformat(), item.symbol): item
            for item in decisions
        }

    def run_cycle(
        self,
        *,
        state,
        primitive_inputs,
        history,
        decision_time: datetime,
        equity_usd: float,
        remaining_portfolio_capacity_usd: float,
        cash_reserve_fraction: float = 0.0,
    ) -> DecisionIntent:
        del primitive_inputs, history, equity_usd, remaining_portfolio_capacity_usd, cash_reserve_fraction
        record = self._decisions[(decision_time.isoformat(), state.symbol)]
        return DecisionIntent(
            decision_id=record.decision_id,
            decision_hash=record.decision_hash,
            snapshot_id=record.snapshot_id,
            config_version=record.config_version,
            timestamp=record.timestamp,
            symbol=record.symbol,
            candidate_mode=record.candidate_mode,
            final_mode=record.final_mode,
            side=record.side,
            trend_direction=record.trend_direction,
            trend_strength=record.trend_strength,
            volume_confirmation=record.volume_confirmation,
            liquidity_score=record.liquidity_score,
            volatility_penalty=record.volatility_penalty,
            overheat_penalty=record.overheat_penalty,
            predictability_score=record.predictability_score,
            gross_expected_edge_bps=record.gross_expected_edge_bps,
            net_expected_edge_bps=record.net_expected_edge_bps,
            estimated_round_trip_cost_bps=record.estimated_round_trip_cost_bps,
            order_intent_notional_usd=record.order_intent_notional_usd,
            stop_distance_bps=record.stop_distance_bps,
            linked_order_ids=record.linked_order_ids,
            rejection_reasons=record.rejection_reasons,
            exit_reason_code=record.exit_reason_code,
            divergence_code=record.divergence_code,
        )


def build_recent_comparison_services(
    *,
    settings: Settings,
    decisions_path: str | Path,
) -> tuple[ComparisonService, ...]:
    decisions = _load_decisions(Path(decisions_path))
    return (
        RecordedDecisionComparisonService(settings, decisions),
        DirectionalHoldBaselineService(settings),
        MomentumBaselineService(settings),
        MeanReversionBaselineService(settings),
    )
