"""
Funding Rate Contrarian Strategy
=================================
매 8시간마다 펀딩비를 확인하여 역추세 포지션을 진입/청산한다.

- |펀딩비| >= threshold(0.015%) 시 신호 발생
- 펀딩비 양수 → 숏 (과도한 롱 포지션 비용 역추세)
- 펀딩비 음수 → 롱 (과도한 숏 포지션 비용 역추세)
- SL: ATR × atr_stop_multiple
- TP: ATR × atr_tp_multiple
- 최대 보유: max_hold_hours 시간
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from statistics import mean
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quant_binance.data.state import KlineBar, SymbolMarketState

# 동일 심볼 재진입 쿨다운 (8시간)
_ENTRY_COOLDOWN_SECONDS: float = 8 * 3600.0


@dataclass
class FundingRateConfig:
    enabled: bool = False
    threshold: float = 0.00015          # 0.015% = 0.00015
    atr_stop_multiple: float = 1.0
    atr_tp_multiple: float = 2.5
    max_hold_hours: int = 8
    symbols: list[str] = field(default_factory=list)
    notional_usd_per_trade: float = 50.0


@dataclass
class FundingRateSignal:
    symbol: str
    side: str                  # "long" | "short"
    funding_rate: float
    entry_price: float
    atr: float
    stop_price: float
    tp_price: float
    signal_time: datetime


@dataclass
class FundingRatePosition:
    symbol: str
    side: str
    entry_price: float
    atr: float
    stop_price: float
    tp_price: float
    entry_time: datetime
    max_hold_until: datetime
    notional_usd: float
    funding_rate: float
    # 청산 시 채워지는 필드
    pnl_usd: float = 0.0
    exit_reason: str = ""
    exit_price: float = 0.0
    exit_time: datetime | None = None


def _atr_from_bars(bars: list[KlineBar], period: int = 14) -> float:
    """1h kline 목록에서 ATR(period) 계산."""
    if len(bars) < 2:
        return 0.0
    true_ranges: list[float] = []
    for i in range(1, len(bars)):
        b = bars[i]
        prev_close = bars[i - 1].close_price
        tr = max(
            b.high_price - b.low_price,
            abs(b.high_price - prev_close),
            abs(b.low_price - prev_close),
        )
        true_ranges.append(tr)
    if not true_ranges:
        return 0.0
    tail = true_ranges[-period:]
    return mean(tail)


def load_funding_rate_config(override: dict) -> FundingRateConfig:
    """strategy_override JSON에서 funding_rate_strategy 섹션을 파싱."""
    raw = dict(override.get("funding_rate_strategy") or {})
    if not raw:
        return FundingRateConfig(enabled=False)
    return FundingRateConfig(
        enabled=bool(raw.get("enabled", False)),
        threshold=float(raw.get("threshold", 0.00015)),
        atr_stop_multiple=float(raw.get("atr_stop_multiple", 1.0)),
        atr_tp_multiple=float(raw.get("atr_tp_multiple", 2.5)),
        max_hold_hours=int(raw.get("max_hold_hours", 8)),
        symbols=list(raw.get("symbols") or []),
        notional_usd_per_trade=float(raw.get("notional_usd_per_trade", 50.0)),
    )


class FundingRateTracker:
    """
    펀딩비 역추세 전략 포지션 라이프사이클 관리.

    - generate_signals(): 진입 신호 확인 (8h 쿨다운 포함)
    - open_position(): 신호를 실제 포지션으로 등록
    - evaluate_exits(): 보유 포지션의 청산 조건 확인
    - close_position(): 포지션 청산 및 PnL 기록
    """

    def __init__(self, config: FundingRateConfig) -> None:
        self.config = config
        # {symbol: last_entry_time}
        self._last_entry_by_symbol: dict[str, datetime] = {}
        # 현재 오픈 포지션
        self.open_positions: dict[str, FundingRatePosition] = {}
        # 청산 이력
        self.closed_trades: list[dict[str, object]] = []

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def generate_signals(
        self,
        states: dict[str, SymbolMarketState],
        now: datetime,
    ) -> list[FundingRateSignal]:
        """현재 펀딩비를 확인해 진입 신호 목록을 반환."""
        if not self.config.enabled:
            return []

        signals: list[FundingRateSignal] = []

        for symbol in self.config.symbols:
            state = states.get(symbol)
            if state is None:
                continue

            # 이미 보유 중인 포지션은 건너뜀
            if symbol in self.open_positions:
                continue

            # 8시간 쿨다운 체크
            last_entry = self._last_entry_by_symbol.get(symbol)
            if last_entry is not None:
                elapsed = (now - last_entry).total_seconds()
                if elapsed < _ENTRY_COOLDOWN_SECONDS:
                    continue

            funding_rate = state.funding_rate
            if abs(funding_rate) < self.config.threshold:
                continue

            # 펀딩비 양수 → 롱이 과도함 → 숏 진입
            # 펀딩비 음수 → 숏이 과도함 → 롱 진입
            side = "short" if funding_rate > 0 else "long"

            bars_1h = list(state.klines.get("1h") or [])
            atr = _atr_from_bars(bars_1h)
            if atr <= 0:
                continue

            price = state.last_trade_price
            if price <= 0:
                continue

            if side == "long":
                stop_price = price - atr * self.config.atr_stop_multiple
                tp_price = price + atr * self.config.atr_tp_multiple
            else:
                stop_price = price + atr * self.config.atr_stop_multiple
                tp_price = price - atr * self.config.atr_tp_multiple

            signals.append(
                FundingRateSignal(
                    symbol=symbol,
                    side=side,
                    funding_rate=funding_rate,
                    entry_price=price,
                    atr=atr,
                    stop_price=stop_price,
                    tp_price=tp_price,
                    signal_time=now,
                )
            )

        return signals

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------

    def open_position(self, signal: FundingRateSignal) -> FundingRatePosition:
        """신호로부터 포지션을 생성하고 추적 시작."""
        pos = FundingRatePosition(
            symbol=signal.symbol,
            side=signal.side,
            entry_price=signal.entry_price,
            atr=signal.atr,
            stop_price=signal.stop_price,
            tp_price=signal.tp_price,
            entry_time=signal.signal_time,
            max_hold_until=signal.signal_time + timedelta(hours=self.config.max_hold_hours),
            notional_usd=self.config.notional_usd_per_trade,
            funding_rate=signal.funding_rate,
        )
        self.open_positions[signal.symbol] = pos
        self._last_entry_by_symbol[signal.symbol] = signal.signal_time
        return pos

    def evaluate_exits(
        self,
        states: dict[str, SymbolMarketState],
        now: datetime,
    ) -> list[tuple[str, str, float]]:
        """
        청산이 필요한 포지션을 반환.
        Returns: list of (symbol, exit_reason, current_price)
        exit_reason: "stop_loss" | "take_profit" | "max_hold" | "no_state"
        """
        exits: list[tuple[str, str, float]] = []

        for symbol, pos in list(self.open_positions.items()):
            state = states.get(symbol)
            if state is None:
                exits.append((symbol, "no_state", pos.entry_price))
                continue

            price = state.last_trade_price
            if price <= 0:
                continue

            # 최대 보유 시간 초과
            if now >= pos.max_hold_until:
                exits.append((symbol, "max_hold", price))
            elif pos.side == "long":
                if price <= pos.stop_price:
                    exits.append((symbol, "stop_loss", price))
                elif price >= pos.tp_price:
                    exits.append((symbol, "take_profit", price))
            else:  # short
                if price >= pos.stop_price:
                    exits.append((symbol, "stop_loss", price))
                elif price <= pos.tp_price:
                    exits.append((symbol, "take_profit", price))

        return exits

    def close_position(
        self,
        symbol: str,
        exit_reason: str,
        exit_price: float,
        exit_time: datetime,
    ) -> FundingRatePosition | None:
        """포지션 청산 및 PnL 계산."""
        pos = self.open_positions.pop(symbol, None)
        if pos is None:
            return None

        direction = 1 if pos.side == "long" else -1
        pnl_usd = (
            (exit_price - pos.entry_price)
            / max(pos.entry_price, 1e-12)
            * direction
            * pos.notional_usd
        )
        pos.pnl_usd = round(pnl_usd, 6)
        pos.exit_reason = exit_reason
        pos.exit_price = exit_price
        pos.exit_time = exit_time

        self.closed_trades.append(
            {
                "symbol": symbol,
                "side": pos.side,
                "funding_rate": pos.funding_rate,
                "entry_price": pos.entry_price,
                "exit_price": exit_price,
                "exit_reason": exit_reason,
                "pnl_usd": pos.pnl_usd,
                "notional_usd": pos.notional_usd,
                "entry_time": pos.entry_time.isoformat(),
                "exit_time": exit_time.isoformat(),
                "atr": pos.atr,
                "stop_price": pos.stop_price,
                "tp_price": pos.tp_price,
            }
        )
        return pos

    # ------------------------------------------------------------------
    # Summary helpers
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, object]:
        total_pnl = sum(float(t.get("pnl_usd", 0)) for t in self.closed_trades)
        wins = sum(1 for t in self.closed_trades if float(t.get("pnl_usd", 0)) > 0)
        losses = sum(1 for t in self.closed_trades if float(t.get("pnl_usd", 0)) <= 0)
        total = len(self.closed_trades)
        return {
            "total_trades": total,
            "wins": wins,
            "losses": losses,
            "win_rate": round(wins / total, 4) if total else 0.0,
            "total_pnl_usd": round(total_pnl, 4),
            "open_positions": list(self.open_positions.keys()),
        }
