from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from quant_binance.backtest.paper_live_fixtures import PaperLiveCycle, load_paper_live_cycles
from quant_binance.data.market_store import MarketStateStore
from quant_binance.execution.router import ExecutionRouter
from quant_binance.live import EventDispatcher, LivePaperRuntime
from quant_binance.models import DecisionIntent
from quant_binance.observability.decision_log import hash_decision_payload
from quant_binance.risk.sizing import position_notional_and_stop_bps, select_futures_leverage
from quant_binance.service import PaperTradingService
from quant_binance.session import LivePaperSession
from quant_binance.settings import Settings


def _clamp(value: float, *, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


def _estimated_round_trip_cost_bps(*, spread_bps: float, probe_slippage_bps: float, taker_fee_bps: float) -> float:
    return round(max(spread_bps, 0.0) + max(probe_slippage_bps, 0.0) + max(taker_fee_bps, 0.0) * 2.0, 6)


def _atr_14_1h_bps(*, atr_14_1h_price: float, last_trade_price: float, floor_bps: float) -> float:
    if atr_14_1h_price <= 0.0 or last_trade_price <= 0.0:
        return floor_bps
    return max((atr_14_1h_price / last_trade_price) * 10000.0, floor_bps)


def _liquidity_score(cycle: PaperLiveCycle) -> float:
    depth_score = _clamp(cycle.primitive_inputs.depth_usd_within_10bps / 300000.0, lower=0.0, upper=1.0)
    spread_score = 1.0 - _clamp(cycle.primitive_inputs.spread_bps / 20.0, lower=0.0, upper=1.0)
    slippage_score = 1.0 - _clamp(cycle.primitive_inputs.probe_slippage_bps / 25.0, lower=0.0, upper=1.0)
    return round(_clamp((depth_score + spread_score + slippage_score) / 3.0, lower=0.05, upper=0.99), 6)


def _volume_confirmation(cycle: PaperLiveCycle) -> float:
    total = max(cycle.primitive_inputs.buy_taker_volume + cycle.primitive_inputs.sell_taker_volume, 1.0)
    imbalance = abs(cycle.primitive_inputs.buy_taker_volume - cycle.primitive_inputs.sell_taker_volume) / total
    return round(_clamp(0.45 + (imbalance * 0.55), lower=0.0, upper=1.0), 6)


def _trend_strength(cycle: PaperLiveCycle) -> float:
    ret_strength = max(abs(cycle.primitive_inputs.ret_1h), abs(cycle.primitive_inputs.ret_4h) / 2.0)
    return round(_clamp(0.35 + (ret_strength * 8.0), lower=0.0, upper=1.0), 6)


def _gross_expected_edge_bps(cycle: PaperLiveCycle, *, strength_multiplier: float) -> float:
    strength = abs(cycle.primitive_inputs.ret_1h) * 600.0 + abs(cycle.primitive_inputs.ret_4h) * 250.0
    return round(_clamp(strength * strength_multiplier, lower=6.0, upper=40.0), 6)


def _predictability_score(
    *,
    cycle: PaperLiveCycle,
    base_score: float,
    signal_strength: float,
) -> float:
    score = base_score + (signal_strength * 20.0) + (_liquidity_score(cycle) * 8.0)
    return round(_clamp(score, lower=35.0, upper=90.0), 6)


def _cash_decision(
    *,
    strategy_name: str,
    cycle: PaperLiveCycle,
    predictability_score: float,
    liquidity_score: float,
    volume_confirmation: float,
    trend_strength: float,
    gross_expected_edge_bps: float,
    estimated_round_trip_cost_bps: float,
    rejection_reason: str,
) -> DecisionIntent:
    net_expected_edge_bps = round(gross_expected_edge_bps - estimated_round_trip_cost_bps, 6)
    payload = {
        "snapshot_id": f"{strategy_name}:{cycle.symbol}:{cycle.decision_time.isoformat()}",
        "config_version": strategy_name,
        "final_mode": "cash",
        "side": "flat",
        "predictability_score": predictability_score,
        "reasons": (rejection_reason,),
    }
    return DecisionIntent(
        decision_id=str(uuid4()),
        decision_hash=hash_decision_payload(payload),
        snapshot_id=payload["snapshot_id"],
        config_version=strategy_name,
        timestamp=cycle.decision_time,
        symbol=cycle.symbol,
        candidate_mode="cash",
        final_mode="cash",
        side="flat",
        trend_direction=cycle.primitive_inputs.trend_direction,
        trend_strength=trend_strength,
        volume_confirmation=volume_confirmation,
        liquidity_score=liquidity_score,
        volatility_penalty=0.25,
        overheat_penalty=0.15,
        predictability_score=predictability_score,
        gross_expected_edge_bps=gross_expected_edge_bps,
        net_expected_edge_bps=net_expected_edge_bps,
        estimated_round_trip_cost_bps=estimated_round_trip_cost_bps,
        order_intent_notional_usd=0.0,
        stop_distance_bps=0.0,
        rejection_reasons=(rejection_reason,),
    )


def _futures_decision(
    *,
    strategy_name: str,
    cycle: PaperLiveCycle,
    settings: Settings,
    side: str,
    equity_usd: float,
    remaining_portfolio_capacity_usd: float,
    gross_expected_edge_bps: float,
    predictability_score: float,
) -> DecisionIntent:
    liquidity_score = _liquidity_score(cycle)
    volume_confirmation = _volume_confirmation(cycle)
    trend_strength = _trend_strength(cycle)
    volatility_penalty = round(_clamp(cycle.primitive_inputs.realized_vol_1h, lower=0.0, upper=1.0), 6)
    overheat_penalty = round(_clamp(abs(cycle.primitive_inputs.ret_1h) * 6.0, lower=0.0, upper=1.0), 6)
    estimated_cost_bps = _estimated_round_trip_cost_bps(
        spread_bps=cycle.primitive_inputs.spread_bps,
        probe_slippage_bps=cycle.primitive_inputs.probe_slippage_bps,
        taker_fee_bps=settings.fees.futures_taker_fee_bps,
    )
    net_expected_edge_bps = round(max(gross_expected_edge_bps - estimated_cost_bps, 0.5), 6)
    leverage = select_futures_leverage(
        predictability_score=predictability_score,
        trend_strength=trend_strength,
        volume_confirmation=volume_confirmation,
        liquidity_score=liquidity_score,
        volatility_penalty=volatility_penalty,
        overheat_penalty=overheat_penalty,
        net_expected_edge_bps=net_expected_edge_bps,
        estimated_round_trip_cost_bps=estimated_cost_bps,
        settings=settings,
    )
    notional_usd, stop_distance_bps = position_notional_and_stop_bps(
        last_trade_price=cycle.state.last_trade_price,
        atr_14_1h_bps=_atr_14_1h_bps(
            atr_14_1h_price=cycle.primitive_inputs.atr_14_1h_price,
            last_trade_price=cycle.state.last_trade_price,
            floor_bps=settings.sizing.stop_floor_bps,
        ),
        equity_usd=equity_usd,
        remaining_portfolio_capacity_usd=remaining_portfolio_capacity_usd,
        settings=settings,
        leverage_multiplier=float(leverage),
        symbol=cycle.symbol,
    )
    payload = {
        "snapshot_id": f"{strategy_name}:{cycle.symbol}:{cycle.decision_time.isoformat()}",
        "config_version": strategy_name,
        "final_mode": "futures",
        "side": side,
        "predictability_score": predictability_score,
    }
    return DecisionIntent(
        decision_id=str(uuid4()),
        decision_hash=hash_decision_payload(payload),
        snapshot_id=payload["snapshot_id"],
        config_version=strategy_name,
        timestamp=cycle.decision_time,
        symbol=cycle.symbol,
        candidate_mode="futures",
        final_mode="futures",
        side=side,
        trend_direction=1 if side == "long" else -1,
        trend_strength=trend_strength,
        volume_confirmation=volume_confirmation,
        liquidity_score=liquidity_score,
        volatility_penalty=volatility_penalty,
        overheat_penalty=overheat_penalty,
        predictability_score=predictability_score,
        gross_expected_edge_bps=gross_expected_edge_bps,
        net_expected_edge_bps=net_expected_edge_bps,
        estimated_round_trip_cost_bps=estimated_cost_bps,
        order_intent_notional_usd=notional_usd,
        stop_distance_bps=stop_distance_bps,
    )


class ComparisonService(Protocol):
    name: str
    settings: Settings

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
        ...


class CurrentStrategyComparisonService:
    name = "current_strategy"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._service = PaperTradingService(settings, router=ExecutionRouter())

    def run_cycle(self, **kwargs) -> DecisionIntent:  # type: ignore[no-untyped-def]
        return self._service.run_cycle(**kwargs)


class DirectionalHoldBaselineService:
    name = "directional_hold"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._side_by_symbol: dict[str, str] = {}

    def run_cycle(  # type: ignore[override]
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
        cycle = PaperLiveCycle(
            decision_time=decision_time,
            symbol=state.symbol,
            state=state,
            primitive_inputs=primitive_inputs,
            history=history,
        )
        side = self._side_by_symbol.get(cycle.symbol)
        if side is None:
            side = "short" if primitive_inputs.trend_direction < 0 else "long"
            self._side_by_symbol[cycle.symbol] = side
        gross_edge = _gross_expected_edge_bps(cycle, strength_multiplier=1.0)
        predictability = _predictability_score(
            cycle=cycle,
            base_score=55.0,
            signal_strength=max(abs(primitive_inputs.ret_1h), 0.01),
        )
        return _futures_decision(
            strategy_name=self.name,
            cycle=cycle,
            settings=self.settings,
            side=side,
            equity_usd=equity_usd,
            remaining_portfolio_capacity_usd=remaining_portfolio_capacity_usd,
            gross_expected_edge_bps=gross_edge,
            predictability_score=predictability,
        )


class MomentumBaselineService:
    name = "simple_momentum"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def run_cycle(  # type: ignore[override]
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
        cycle = PaperLiveCycle(
            decision_time=decision_time,
            symbol=state.symbol,
            state=state,
            primitive_inputs=primitive_inputs,
            history=history,
        )
        gross_edge = _gross_expected_edge_bps(cycle, strength_multiplier=1.1)
        liquidity_score = _liquidity_score(cycle)
        volume_confirmation = _volume_confirmation(cycle)
        trend_strength = _trend_strength(cycle)
        signal_strength = max(abs(primitive_inputs.ret_1h), abs(primitive_inputs.ret_4h) / 2.0)
        if primitive_inputs.ret_1h >= 0.01 and primitive_inputs.trend_direction >= 0:
            side = "long"
        elif primitive_inputs.ret_1h <= -0.01 and primitive_inputs.trend_direction <= 0:
            side = "short"
        else:
            return _cash_decision(
                strategy_name=self.name,
                cycle=cycle,
                predictability_score=_predictability_score(cycle=cycle, base_score=50.0, signal_strength=signal_strength),
                liquidity_score=liquidity_score,
                volume_confirmation=volume_confirmation,
                trend_strength=trend_strength,
                gross_expected_edge_bps=gross_edge,
                estimated_round_trip_cost_bps=_estimated_round_trip_cost_bps(
                    spread_bps=primitive_inputs.spread_bps,
                    probe_slippage_bps=primitive_inputs.probe_slippage_bps,
                    taker_fee_bps=self.settings.fees.futures_taker_fee_bps,
                ),
                rejection_reason="BASELINE_NO_MOMENTUM_SIGNAL",
            )
        return _futures_decision(
            strategy_name=self.name,
            cycle=cycle,
            settings=self.settings,
            side=side,
            equity_usd=equity_usd,
            remaining_portfolio_capacity_usd=remaining_portfolio_capacity_usd,
            gross_expected_edge_bps=gross_edge,
            predictability_score=_predictability_score(cycle=cycle, base_score=58.0, signal_strength=signal_strength),
        )


class MeanReversionBaselineService:
    name = "simple_mean_reversion"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def run_cycle(  # type: ignore[override]
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
        cycle = PaperLiveCycle(
            decision_time=decision_time,
            symbol=state.symbol,
            state=state,
            primitive_inputs=primitive_inputs,
            history=history,
        )
        liquidity_score = _liquidity_score(cycle)
        volume_confirmation = _volume_confirmation(cycle)
        trend_strength = _trend_strength(cycle)
        breakout_reference_price = primitive_inputs.breakout_reference_price or state.last_trade_price
        price_deviation = 0.0
        if breakout_reference_price > 0.0:
            price_deviation = (state.last_trade_price - breakout_reference_price) / breakout_reference_price
        signal_strength = max(abs(price_deviation), abs(primitive_inputs.ret_1h))
        if primitive_inputs.ret_1h >= 0.02 and price_deviation >= 0.01:
            side = "short"
        elif primitive_inputs.ret_1h <= -0.02 and price_deviation <= -0.01:
            side = "long"
        else:
            return _cash_decision(
                strategy_name=self.name,
                cycle=cycle,
                predictability_score=_predictability_score(cycle=cycle, base_score=48.0, signal_strength=signal_strength),
                liquidity_score=liquidity_score,
                volume_confirmation=volume_confirmation,
                trend_strength=trend_strength,
                gross_expected_edge_bps=_gross_expected_edge_bps(cycle, strength_multiplier=0.9),
                estimated_round_trip_cost_bps=_estimated_round_trip_cost_bps(
                    spread_bps=primitive_inputs.spread_bps,
                    probe_slippage_bps=primitive_inputs.probe_slippage_bps,
                    taker_fee_bps=self.settings.fees.futures_taker_fee_bps,
                ),
                rejection_reason="BASELINE_NO_MEAN_REVERSION_SIGNAL",
            )
        return _futures_decision(
            strategy_name=self.name,
            cycle=cycle,
            settings=self.settings,
            side=side,
            equity_usd=equity_usd,
            remaining_portfolio_capacity_usd=remaining_portfolio_capacity_usd,
            gross_expected_edge_bps=_gross_expected_edge_bps(cycle, strength_multiplier=0.9),
            predictability_score=_predictability_score(cycle=cycle, base_score=54.0, signal_strength=signal_strength),
        )


@dataclass(frozen=True)
class StrategyComparisonResult:
    strategy_name: str
    decision_count: int
    trade_count: int
    closed_trade_count: int
    open_position_count: int
    entry_turnover_usd: float
    exit_turnover_usd: float
    turnover_usd: float
    realized_pnl_usd: float
    unrealized_pnl_usd: float
    total_pnl_usd: float
    total_return_pct: float
    max_drawdown_pct: float
    win_count: int
    loss_count: int
    hit_rate: float
    long_trade_count: int
    short_trade_count: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class StrategyComparisonReport:
    config_path: str
    fixture_path: str
    equity_usd: float
    capacity_usd: float
    cycle_count: int
    strategies: tuple[StrategyComparisonResult, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "config_path": self.config_path,
            "fixture_path": self.fixture_path,
            "equity_usd": self.equity_usd,
            "capacity_usd": self.capacity_usd,
            "cycle_count": self.cycle_count,
            "strategies": [item.as_dict() for item in self.strategies],
        }


def _build_session(*, strategy_service: ComparisonService, equity_usd: float, capacity_usd: float) -> LivePaperSession:
    runtime = LivePaperRuntime(
        dispatcher=EventDispatcher(MarketStateStore()),
        paper_service=strategy_service,  # type: ignore[arg-type]
        primitive_builder=lambda symbol, decision_time: None,  # type: ignore[return-value]
        history_provider=lambda symbol, decision_time: None,  # type: ignore[return-value]
        decision_interval_minutes=strategy_service.settings.decision_engine.decision_interval_minutes,
    )
    return LivePaperSession(
        runtime=runtime,
        equity_usd=equity_usd,
        remaining_portfolio_capacity_usd=capacity_usd,
        max_portfolio_capacity_usd=capacity_usd,
        sync_interval_seconds=10**9,
        flush_interval_seconds=10**9,
    )


def _equity_curve_point(*, session: LivePaperSession, starting_equity_usd: float) -> float:
    realized = sum(float(trade.get("realized_pnl_usd_estimate", 0.0)) for trade in session.closed_trades)
    unrealized = sum(position.unrealized_pnl_usd_estimate() for position in session.paper_positions.values())
    return round(starting_equity_usd + realized + unrealized, 6)


def _max_drawdown_pct(equity_curve: list[float]) -> float:
    if not equity_curve:
        return 0.0
    peak = equity_curve[0]
    max_drawdown = 0.0
    for equity in equity_curve:
        peak = max(peak, equity)
        if peak <= 0.0:
            continue
        drawdown = (peak - equity) / peak * 100.0
        max_drawdown = max(max_drawdown, drawdown)
    return round(max_drawdown, 6)


def _evaluate_strategy(
    *,
    strategy_service: ComparisonService,
    cycles: list[PaperLiveCycle],
    equity_usd: float,
    capacity_usd: float,
) -> StrategyComparisonResult:
    session = _build_session(strategy_service=strategy_service, equity_usd=equity_usd, capacity_usd=capacity_usd)
    entry_turnover = 0.0
    exit_turnover = 0.0
    trade_count = 0
    long_trade_count = 0
    short_trade_count = 0
    equity_curve = [equity_usd]
    closed_trade_index = 0
    for cycle in cycles:
        before_symbols = set(session.paper_positions)
        session.run_bootstrap_cycle(
            state=cycle.state,
            primitive_inputs=cycle.primitive_inputs,
            history=cycle.history,
            decision_time=cycle.decision_time,
        )
        after_symbols = set(session.paper_positions)
        for symbol in sorted(after_symbols - before_symbols):
            position = session.paper_positions[symbol]
            trade_count += 1
            if position.side == "short":
                short_trade_count += 1
            else:
                long_trade_count += 1
            entry_turnover += position.entry_price * position.quantity_opened
        closed_trades = list(session.closed_trades)
        new_closed_trades = closed_trades[closed_trade_index:]
        exit_turnover += sum(float(trade.get("exit_price", 0.0)) * float(trade.get("quantity", 0.0)) for trade in new_closed_trades)
        closed_trade_index = len(closed_trades)
        equity_curve.append(_equity_curve_point(session=session, starting_equity_usd=equity_usd))
    realized_pnl = round(sum(float(trade.get("realized_pnl_usd_estimate", 0.0)) for trade in session.closed_trades), 6)
    unrealized_pnl = round(sum(position.unrealized_pnl_usd_estimate() for position in session.paper_positions.values()), 6)
    total_pnl = round(realized_pnl + unrealized_pnl, 6)
    wins = sum(1 for trade in session.closed_trades if float(trade.get("realized_pnl_usd_estimate", 0.0)) > 0.0)
    losses = sum(1 for trade in session.closed_trades if float(trade.get("realized_pnl_usd_estimate", 0.0)) < 0.0)
    closed_trade_count = len(session.closed_trades)
    hit_rate = round((wins / closed_trade_count) if closed_trade_count else 0.0, 6)
    total_return_pct = round((total_pnl / equity_usd) * 100.0, 6) if equity_usd > 0.0 else 0.0
    return StrategyComparisonResult(
        strategy_name=strategy_service.name,
        decision_count=len(session.decisions),
        trade_count=trade_count,
        closed_trade_count=closed_trade_count,
        open_position_count=len(session.paper_positions),
        entry_turnover_usd=round(entry_turnover, 6),
        exit_turnover_usd=round(exit_turnover, 6),
        turnover_usd=round(entry_turnover + exit_turnover, 6),
        realized_pnl_usd=realized_pnl,
        unrealized_pnl_usd=unrealized_pnl,
        total_pnl_usd=total_pnl,
        total_return_pct=total_return_pct,
        max_drawdown_pct=_max_drawdown_pct(equity_curve),
        win_count=wins,
        loss_count=losses,
        hit_rate=hit_rate,
        long_trade_count=long_trade_count,
        short_trade_count=short_trade_count,
    )


def default_comparison_services(settings: Settings) -> tuple[ComparisonService, ...]:
    return (
        CurrentStrategyComparisonService(settings),
        DirectionalHoldBaselineService(settings),
        MomentumBaselineService(settings),
        MeanReversionBaselineService(settings),
    )


def compare_strategies(
    *,
    config_path: str | Path,
    fixture_path: str | Path,
    equity_usd: float,
    capacity_usd: float,
    services: tuple[ComparisonService, ...] | None = None,
) -> StrategyComparisonReport:
    settings = Settings.load(config_path)
    cycles = sorted(
        load_paper_live_cycles(fixture_path),
        key=lambda cycle: (cycle.decision_time, cycle.symbol),
    )
    selected_services = services or default_comparison_services(settings)
    strategies = tuple(
        _evaluate_strategy(
            strategy_service=service,
            cycles=cycles,
            equity_usd=equity_usd,
            capacity_usd=capacity_usd,
        )
        for service in selected_services
    )
    return StrategyComparisonReport(
        config_path=str(Path(config_path)),
        fixture_path=str(Path(fixture_path)),
        equity_usd=equity_usd,
        capacity_usd=capacity_usd,
        cycle_count=len(cycles),
        strategies=strategies,
    )


def render_compact_report(report: StrategyComparisonReport) -> str:
    lines = [
        "Strategy comparison",
        f"fixture={report.fixture_path} cycles={report.cycle_count} equity_usd={report.equity_usd:.2f} capacity_usd={report.capacity_usd:.2f}",
        "PnL is mark-to-market: realized + unrealized at the final cycle.",
        "",
        "strategy                total_pnl   return%   max_dd%   realized   hit_rate   trades   turnover    L/S   open",
    ]
    for item in report.strategies:
        lines.append(
            f"{item.strategy_name:<23} "
            f"{item.total_pnl_usd:>10.2f} "
            f"{item.total_return_pct:>8.2f} "
            f"{item.max_drawdown_pct:>8.2f} "
            f"{item.realized_pnl_usd:>10.2f} "
            f"{item.hit_rate * 100.0:>9.1f}% "
            f"{item.trade_count:>7} "
            f"{item.turnover_usd:>10.2f} "
            f"{item.long_trade_count}/{item.short_trade_count:<3} "
            f"{item.open_position_count:>5}"
        )
    return "\n".join(lines)


def write_comparison_report(path: str | Path, report: StrategyComparisonReport) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report.as_dict(), indent=2, sort_keys=True), encoding="utf-8")
