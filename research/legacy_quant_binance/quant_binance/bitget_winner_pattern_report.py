from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any


@dataclass(frozen=True)
class WinnerPatternReport:
    generated_at: str
    winner_count: int
    symbols: tuple[dict[str, object], ...]
    sides: tuple[dict[str, object], ...]
    average_net_profit_usd: float
    median_net_profit_usd: float
    average_hold_minutes: float
    median_hold_minutes: float
    summary: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "generated_at": self.generated_at,
            "winner_count": self.winner_count,
            "symbols": list(self.symbols),
            "sides": list(self.sides),
            "average_net_profit_usd": self.average_net_profit_usd,
            "median_net_profit_usd": self.median_net_profit_usd,
            "average_hold_minutes": self.average_hold_minutes,
            "median_hold_minutes": self.median_hold_minutes,
            "summary": list(self.summary),
        }


def _parse_iso(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def build_winner_pattern_report(*, winners_path: str | Path) -> WinnerPatternReport:
    payload = json.loads(Path(winners_path).read_text(encoding="utf-8"))
    winners = payload.get("winners") or []
    symbol_counter: Counter[str] = Counter()
    side_counter: Counter[str] = Counter()
    net_profits: list[float] = []
    hold_minutes_rows: list[float] = []
    by_symbol_profit: dict[str, list[float]] = defaultdict(list)

    for row in winners:
        symbol = str(row.get("symbol", ""))
        hold_side = str(row.get("hold_side", ""))
        net_profit = float(row.get("net_profit_usd", 0.0))
        symbol_counter[symbol] += 1
        side_counter[hold_side] += 1
        net_profits.append(net_profit)
        by_symbol_profit[symbol].append(net_profit)
        open_time = _parse_iso(str(row.get("open_time", "")))
        close_time = _parse_iso(str(row.get("close_time", "")))
        if open_time is not None and close_time is not None:
            hold_minutes_rows.append(max((close_time - open_time).total_seconds() / 60.0, 0.0))

    symbol_rows = [
        {
            "symbol": symbol,
            "count": count,
            "average_net_profit_usd": round(mean(by_symbol_profit[symbol]), 6),
        }
        for symbol, count in symbol_counter.most_common()
    ]
    side_rows = [
        {"hold_side": side, "count": count}
        for side, count in side_counter.most_common()
    ]
    average_net_profit = round(mean(net_profits), 6) if net_profits else 0.0
    median_net_profit = round(median(net_profits), 6) if net_profits else 0.0
    average_hold_minutes = round(mean(hold_minutes_rows), 6) if hold_minutes_rows else 0.0
    median_hold_minutes = round(median(hold_minutes_rows), 6) if hold_minutes_rows else 0.0

    summary: list[str] = []
    if symbol_rows:
        top_symbol = symbol_rows[0]
        summary.append(
            f"최근 +${payload.get('min_realized_pnl_usd', 0)} 이상 실현 수익은 {top_symbol['symbol']}에 가장 집중됨"
        )
    if side_rows:
        summary.append(f"승리 방향은 {side_rows[0]['hold_side']} 쪽이 우세")
    if hold_minutes_rows:
        summary.append(
            f"중앙값 보유 시간은 약 {round(median_hold_minutes, 1)}분"
        )
    if net_profits:
        summary.append(
            f"승리 거래 평균 순이익은 약 ${average_net_profit:.2f}"
        )

    return WinnerPatternReport(
        generated_at=datetime.now().astimezone().isoformat(),
        winner_count=len(winners),
        symbols=tuple(symbol_rows),
        sides=tuple(side_rows),
        average_net_profit_usd=average_net_profit,
        median_net_profit_usd=median_net_profit,
        average_hold_minutes=average_hold_minutes,
        median_hold_minutes=median_hold_minutes,
        summary=tuple(summary),
    )


def write_winner_pattern_report(*, report: WinnerPatternReport, output_path: str | Path) -> Path:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report.as_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return target
