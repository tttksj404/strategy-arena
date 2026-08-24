from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import timezone, datetime, timedelta
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RealizedWinner:
    symbol: str
    hold_side: str
    realized_pnl_usd: float
    net_profit_usd: float
    open_avg_price: float
    close_avg_price: float
    open_time: str
    close_time: str
    position_id: str


@dataclass(frozen=True)
class BitgetHistoryReport:
    generated_at: str
    start_time: str
    end_time: str
    min_realized_pnl_usd: float
    winner_count: int
    winners: tuple[RealizedWinner, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "generated_at": self.generated_at,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "min_realized_pnl_usd": self.min_realized_pnl_usd,
            "winner_count": self.winner_count,
            "winners": [asdict(item) for item in self.winners],
        }


def _as_iso_utc(value_ms: Any) -> str:
    if value_ms in (None, ""):
        return ""
    try:
        raw = float(value_ms)
    except (TypeError, ValueError):
        return ""
    if raw <= 0:
        return ""
    if raw >= 1_000_000_000_000:
        raw /= 1000.0
    return datetime.fromtimestamp(raw, tz=timezone.utc).isoformat()


def build_bitget_realized_winner_report(
    *,
    client: Any,
    start_time: datetime,
    end_time: datetime,
    min_realized_pnl_usd: float = 20.0,
    symbol: str | None = None,
) -> BitgetHistoryReport:
    start_ms = int(start_time.timestamp() * 1000)
    end_ms = int(end_time.timestamp() * 1000)
    payload = client.get_futures_position_history(
        symbol=symbol,
        start_time_ms=start_ms,
        end_time_ms=end_ms,
    )
    rows = payload.get("positions", [])
    winners: list[RealizedWinner] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        pnl = float(row.get("pnl") or row.get("achievedProfits") or row.get("closeProfits") or 0.0)
        net_profit = float(row.get("netProfit") or pnl)
        threshold_value = net_profit if net_profit != 0.0 else pnl
        if threshold_value < min_realized_pnl_usd:
            continue
        winners.append(
            RealizedWinner(
                symbol=str(row.get("symbol", "")),
                hold_side=str(row.get("holdSide") or row.get("posSide") or row.get("holdMode") or ""),
                realized_pnl_usd=round(pnl, 6),
                net_profit_usd=round(net_profit, 6),
                open_avg_price=float(row.get("openAvgPrice") or row.get("openPriceAvg") or row.get("openPrice") or 0.0),
                close_avg_price=float(row.get("closeAvgPrice") or row.get("closePriceAvg") or row.get("closePrice") or 0.0),
                open_time=_as_iso_utc(row.get("ctime") or row.get("cTime") or row.get("createdTime")),
                close_time=_as_iso_utc(row.get("utime") or row.get("uTime") or row.get("updatedTime")),
                position_id=str(row.get("positionId") or row.get("posId") or ""),
            )
        )
    winners.sort(key=lambda item: item.net_profit_usd, reverse=True)
    return BitgetHistoryReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        start_time=start_time.isoformat(),
        end_time=end_time.isoformat(),
        min_realized_pnl_usd=min_realized_pnl_usd,
        winner_count=len(winners),
        winners=tuple(winners),
    )


def write_bitget_history_report(*, report: BitgetHistoryReport, output_path: str | Path) -> Path:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report.as_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return target
