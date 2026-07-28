from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from research.paper.candidates import TRACKED_IDS
from research.paper.fidelity import FidelityResult
from research.paper.ledger import LedgerEntry, Position, latest_entries
from research.paper.market_data import LiveSnapshot


def _position_text(position: Position) -> str:
    return f"{position.symbol} {position.instrument} {position.side} {position.notional_usdt:.2f} USDT @ {position.mark_price:.6g}"


def _fidelity_banner_lines(fidelity_results: Mapping[str, FidelityResult] | None) -> list[str]:
    """Task-2 requirement: a shortfall must never pass quietly. Anything that fails
    research/paper/fidelity.py's check gets a banner at the very top of STATUS.md, before even
    the generated-at timestamp, so it's the first thing anyone reading the file sees."""
    if not fidelity_results:
        return []
    failing = [result for result in fidelity_results.values() if not result.ok]
    if not failing:
        return []
    lines = ["## ⚠️ 유니버스 사양 미충족", ""]
    lines.extend(f"⚠️ {result.reason}" for result in failing)
    lines.append("")
    return lines


def _coverage_table_lines(fidelity_results: Mapping[str, FidelityResult] | None) -> list[str]:
    if not fidelity_results:
        return []
    lines = ["## 후보별 유니버스 커버리지", "", "| 후보 | 요구 유니버스 | 실제 커버 | 판정 |", "|---|---:|---:|---|"]
    for candidate_id in TRACKED_IDS:
        result = fidelity_results.get(candidate_id)
        if result is None:
            continue
        verdict = "PASS" if result.ok else "**FIDELITY_FAIL**"
        lines.append(f"| {candidate_id} | {result.required} | {result.actual} | {verdict} |")
    lines.append("")
    return lines


def _stale_fail_notes(current: Mapping[str, LedgerEntry], fidelity_results: Mapping[str, FidelityResult] | None) -> list[str]:
    """Explains an otherwise-confusing juxtaposition: the ledger's per-day dedup (one record per
    candidate per UTC date, unrelated to this task -- see research/paper/ledger.append_entries)
    means a candidate that already has TODAY's record written -- from an earlier run, before a
    fix landed -- keeps showing that record's FIDELITY_FAIL until the next calendar day, even
    though a fresh check moments ago (the coverage table below) shows the universe is fine now.
    Without this note that reads as a contradiction between the two tables; it isn't one."""
    if not fidelity_results:
        return []
    lines: list[str] = []
    for candidate_id in TRACKED_IDS:
        entry = current.get(candidate_id)
        result = fidelity_results.get(candidate_id)
        if entry is None or result is None or entry.fidelity_ok or not result.ok:
            continue
        lines.append(
            f"- `{candidate_id}`: 위 표의 FIDELITY_FAIL은 `{entry.run_date}`에 이미 기록된, 유니버스 요건 미달 상태의 과거 판정이다. "
            f"방금 재확인한 실제 커버리지는 정상(PASS, {result.actual}/{result.required}) — 원장은 하루 1건 정책상 다음 UTC 날짜의 실행부터 새 판정을 반영한다."
        )
    if not lines:
        return []
    return ["## 참고 — 표시된 FIDELITY_FAIL은 과거 기록", ""] + lines + [""]


def _contamination_notes(entries: tuple[LedgerEntry, ...]) -> list[str]:
    """Task-3 requirement: past records written while the universe was known-incomplete are
    never deleted, only flagged (LedgerEntry.fidelity_ok). This surfaces that flag as a
    human-readable "valid record start date" per candidate, generically -- works for any
    candidate that ever gets contaminated, not just G1."""
    by_candidate: dict[str, list[LedgerEntry]] = {}
    for entry in entries:
        by_candidate.setdefault(entry.candidate_id, []).append(entry)
    lines: list[str] = []
    for candidate_id in TRACKED_IDS:
        history = sorted(by_candidate.get(candidate_id, ()), key=lambda item: item.observed_at)
        bad_dates = [item.run_date for item in history if not item.fidelity_ok]
        if not bad_dates:
            continue
        valid_dates = [item.run_date for item in history if item.fidelity_ok]
        first_valid = valid_dates[0] if valid_dates else "아직 없음 — 다음 실행일부터 (오늘 날짜는 이미 오염 기록으로 점유됨)"
        lines.append(f"- `{candidate_id}` 유효 기록 시작일: **{first_valid}** (오염 기록 {len(bad_dates)}건 집계 제외: {', '.join(bad_dates)})")
    if not lines:
        return []
    return ["## 오염 기록 처리", ""] + lines + [""]


def _collection_notes(snapshot: LiveSnapshot | None) -> list[str]:
    if snapshot is None:
        return []
    lines: list[str] = []
    budget_seconds = 300.0
    if snapshot.collection_seconds > budget_seconds:
        lines.append(
            f"⚠️ 수집 시간 {snapshot.collection_seconds:.1f}s가 목표(5분={budget_seconds:.0f}s)를 초과했다."
        )
    else:
        lines.append(f"수집 시간: {snapshot.collection_seconds:.1f}s (목표 5분 이내).")
    if snapshot.universe_failed_symbols:
        preview = ", ".join(snapshot.universe_failed_symbols[:15])
        more = f" 외 {len(snapshot.universe_failed_symbols) - 15}개" if len(snapshot.universe_failed_symbols) > 15 else ""
        lines.append(f"수집 실패/스킵 심볼 {len(snapshot.universe_failed_symbols)}개(추정치로 채우지 않고 제외): {preview}{more}")
    lines.append(f"funding_series 커버 심볼: {len(snapshot.funding_series)}개.")
    return ["## 수집 상태", ""] + lines + [""]


def render_status(
    entries: tuple[LedgerEntry, ...],
    path: Path,
    fidelity_results: Mapping[str, FidelityResult] | None = None,
    snapshot: LiveSnapshot | None = None,
) -> None:
    current = latest_entries(entries)
    generated_at = datetime.now(timezone.utc).isoformat()
    lines = ["# Paper forward-validation status", ""]
    lines.extend(_fidelity_banner_lines(fidelity_results))
    lines.extend(
        [
            f"생성 시각: `{generated_at}`",
            "",
            "실주문: **금지**. 주문 엔드포인트·API 키·서명 기능을 사용하지 않는다.",
            "메이커 체결가정: 최신 공개 1D 바의 종가를 mid 추정 체결가로 사용하고, 진입·청산 각 leg에 0.02% maker fee를 적용하며 슬리피지는 0으로 둔다.",
            "펀딩: 보유 perp notional × 공개 funding rate × 경과시간/8h로 가상 적립한다. 양수 funding에서 perp short는 수취한다.",
            "",
            "| 후보 | 가상 에쿼티(USDT) | 오픈 포지션 | 누적 펀딩(USDT) | 최근 실행일 |",
            "|---|---:|---|---:|---|",
        ]
    )
    for candidate_id in TRACKED_IDS:
        entry = current.get(candidate_id)
        if entry is None:
            lines.append(f"| {candidate_id} | - | 데이터 없음 | - | - |")
            continue
        label = f"{candidate_id} **FIDELITY_FAIL**" if not entry.fidelity_ok else candidate_id
        positions = "<br>".join(_position_text(position) for position in entry.positions) or "현금"
        lines.append(f"| {label} | {entry.virtual_equity:.4f} | {positions} | {entry.cumulative_funding:.6f} | {entry.run_date} |")
    lines.append("")
    lines.extend(_stale_fail_notes(current, fidelity_results))
    lines.extend(("## 후보별 신호", ""))
    for candidate_id in TRACKED_IDS:
        entry = current.get(candidate_id)
        if entry is None:
            lines.append(f"- `{candidate_id}`: 미기록")
        else:
            fail_tag = " [FIDELITY_FAIL]" if not entry.fidelity_ok else ""
            lines.append(f"- `{candidate_id}`{fail_tag}: {entry.signal}; 최근 손익 {entry.pnl_delta:.6f} USDT; maker fee {entry.maker_fees:.6f} USDT")
    lines.append("")
    lines.extend(_coverage_table_lines(fidelity_results))
    lines.extend(_contamination_notes(entries))
    lines.extend(_collection_notes(snapshot))
    lines.extend(("원장 경로: `research/paper/ledger/paper_ledger.jsonl`", ""))
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
    for candidate_id in TRACKED_IDS:
        lines.append(f"| {candidate_id} | - | - | - | 라이브 데이터 미수집 |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


__all__ = ["render_failure_status", "render_status"]
