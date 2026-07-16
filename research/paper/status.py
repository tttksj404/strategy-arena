from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from research.paper.ledger import LedgerEntry, Position, latest_entries


def _position_text(position: Position) -> str:
    return f"{position.symbol} {position.instrument} {position.side} {position.notional_usdt:.2f} USDT @ {position.mark_price:.6g}"


def render_status(entries: tuple[LedgerEntry, ...], path: Path) -> None:
    current = latest_entries(entries)
    generated_at = datetime.now(timezone.utc).isoformat()
    lines = [
        "# Paper forward-validation status",
        "",
        f"생성 시각: `{generated_at}`",
        "",
        "실주문: **금지**. 주문 엔드포인트·API 키·서명 기능을 사용하지 않는다.",
        "메이커 체결가정: 최신 공개 1D 바의 종가를 mid 추정 체결가로 사용하고, 진입·청산 각 leg에 0.02% maker fee를 적용하며 슬리피지는 0으로 둔다.",
        "펀딩: 보유 perp notional × 공개 funding rate × 경과시간/8h로 가상 적립한다. 양수 funding에서 perp short는 수취한다.",
        "",
        "| 후보 | 가상 에쿼티(USDT) | 오픈 포지션 | 누적 펀딩(USDT) | 최근 실행일 |",
        "|---|---:|---|---:|---|",
    ]
    for candidate_id in ("W2c", "F1e", "W3c", "W3d"):
        entry = current.get(candidate_id)
        if entry is None:
            lines.append(f"| {candidate_id} | - | 데이터 없음 | - | - |")
            continue
        positions = "<br>".join(_position_text(position) for position in entry.positions) or "현금"
        lines.append(f"| {candidate_id} | {entry.virtual_equity:.4f} | {positions} | {entry.cumulative_funding:.6f} | {entry.run_date} |")
    lines.extend(("", "## 후보별 신호", ""))
    for candidate_id in ("W2c", "F1e", "W3c", "W3d"):
        entry = current.get(candidate_id)
        if entry is None:
            lines.append(f"- `{candidate_id}`: 미기록")
        else:
            lines.append(f"- `{candidate_id}`: {entry.signal}; 최근 손익 {entry.pnl_delta:.6f} USDT; maker fee {entry.maker_fees:.6f} USDT")
    lines.extend(("", "원장 경로: `research/paper/ledger/paper_ledger.jsonl`", ""))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def render_failure_status(path: Path, observed_at: str, reason: str) -> None:
    lines = [
        "# Paper forward-validation status",
        "",
        f"시도 시각: `{observed_at}`",
        "",
        "실주문: **금지**. 라이브 데이터 수집에 실패해 원장·에쿼티·포지션을 갱신하지 않았다.",
        f"수집 상태: **실패**. `{reason}`",
        "",
        "| 후보 | 가상 에쿼티 | 오픈 포지션 | 누적 펀딩 | 상태 |",
        "|---|---:|---|---:|---|",
    ]
    for candidate_id in ("W2c", "F1e", "W3c", "W3d"):
        lines.append(f"| {candidate_id} | - | - | - | 라이브 데이터 미수집 |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


__all__ = ["render_failure_status", "render_status"]
