# Wave-26 report/registry writer. Reads ONLY research/wave26_freq/results/*.json (written by
# run_wave26.py's run+gates stages, plus the separate live_signals.json from the `live` stage)
# and, read-only for the before/after comparison table, research/wave25_gamble/results/*.json
# (wave25's own frozen output -- never modified from here) -- no re-simulation happens in this
# module, matching research.wave25_gamble.reporting25's own run/gates/report stage separation.

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Final

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

from research.wave26_freq.configs26 import (
    CANDIDATE_IDS,
    DSR_CUMULATIVE_TRIALS,
    GAMBLE_CAPITAL,
    PRIOR_CUMULATIVE_TRIALS,
    Q4_MAX_COST_FRACTION_OF_SLEEVE,
    Q4_MAX_COST_USDT,
    REPO_ROOT,
    STABLE_CAPITAL,
    TOTAL_CAPITAL,
    WAVE25_SIGNAL_SOURCE,
)

BASELINE_CANDIDATE: Final = "C0"
WAVE25_RESULTS_DIR: Final = REPO_ROOT / "research" / "wave25_gamble" / "results"


def _load(results_dir: Path, candidate_id: str) -> dict[str, Any] | None:
    path = results_dir / f"{candidate_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _pct(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.{digits}f}%"


def _usd(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "N/A"
    return f"${value:,.{digits}f}"


def _gate_map(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    gates_report = payload.get("gates_report") or {}
    return {g["gate_id"]: g for g in gates_report.get("gates", [])}


def _win_rate(trades: list[dict[str, Any]]) -> float | None:
    if not trades:
        return None
    wins = sum(1 for t in trades if float(t["pnl_usdt"]) > 0.0)
    return wins / len(trades)


def _final_from_series(records: list[dict[str, Any]], default: float) -> float:
    clean = [r for r in records if r.get("value") is not None]
    return float(clean[-1]["value"]) if clean else default


def _summary_row(candidate_id: str, payload: dict[str, Any] | None) -> dict[str, Any]:
    if payload is None:
        return {"candidate_id": candidate_id, "missing": True}
    trades = payload.get("trades", [])
    metadata = payload.get("metadata", {})
    diagnostics = (payload.get("gates_report") or {}).get("convexity_diagnostics", {})
    gates = _gate_map(payload)
    overall = (payload.get("gates_report") or {}).get("overall", {})
    entry_admission = metadata.get("entry_admission", {}) or {}
    return {
        "candidate_id": candidate_id,
        "missing": False,
        "definition": payload.get("definition", ""),
        "control": payload.get("control", {}),
        "base_family": payload.get("base_family"),
        "n_trades": len(trades),
        "win_rate": _win_rate(trades),
        "gamble_final_usdt": _final_from_series(payload.get("gamble_equity", []), GAMBLE_CAPITAL),
        "combined_final_usdt": _final_from_series(payload.get("combined_equity", []), TOTAL_CAPITAL),
        "cagr_gamble_only": payload.get("full_period_cagr_gamble_only"),
        "cagr_combined": payload.get("full_period_cagr_combined"),
        "total_cost_usdt": metadata.get("total_cost_usdt"),
        "skew": diagnostics.get("skew"),
        "top_decile_contribution": diagnostics.get("top_decile_contribution_of_gross_profit"),
        "bootstrap_p05": (diagnostics.get("bootstrap") or {}).get("p05") if diagnostics.get("bootstrap") else None,
        "bootstrap_fraction_positive_skew": (diagnostics.get("bootstrap") or {}).get("fraction_positive") if diagnostics.get("bootstrap") else None,
        "best_trade_pnl_usdt": (diagnostics.get("best_trade_sensitivity") or {}).get("best_trade_pnl_usdt") if diagnostics.get("best_trade_sensitivity") else None,
        "gates": {gid: g["status"] for gid, g in gates.items()},
        "overall_status": overall.get("status", "N/A"),
        "promoted": overall.get("promoted", False),
        "metadata": metadata,
        "entry_admission": entry_admission,
        "gamble_equity_records": payload.get("gamble_equity", []),
        "stress_test": payload.get("stress_test", {}),
    }


def _registry_row(row: dict[str, Any]) -> str:
    if row.get("missing"):
        return f"| {row['candidate_id']} | wave26_freq | NOT_RUN | - | - | - | - |"
    promoted = "YES" if row["promoted"] else "NO"
    gate_str = " ".join(f"{gid}:{status[0]}" for gid, status in sorted(row["gates"].items()))
    return f"| {row['candidate_id']} | wave26_freq | EVALUATED | {_pct(row['cagr_gamble_only'])} | {_usd(row['gamble_final_usdt'], 4)} | {promoted} | {gate_str} |"


def write_registry(results_dir: Path, registry_path: Path) -> None:
    rows = [_summary_row(cid, _load(results_dir, cid)) for cid in CANDIDATE_IDS]
    n_promoted = sum(1 for r in rows if not r.get("missing") and r["candidate_id"] != BASELINE_CANDIDATE and r["promoted"])
    lines = [
        "# Wave-26 registry",
        "",
        "**2차 탐색 산물** -- 이 wave의 가설은 wave-25 결과를 보고 사후에 만들었다 (report/wave26_report.md 서두 참조). 통과 후보도 1차 사전등록보다 신뢰도가 낮다.",
        "",
        "| Candidate | Family | State | 도박sleeve CAGR | 도박sleeve 최종($25 기준) | 승격 | 게이트(Q1-Q5) |",
        "|---|---|---|---|---|---|---|",
    ]
    lines.extend(_registry_row(row) for row in rows)
    lines.append("")
    if n_promoted == 0:
        lines.append("**최종 판정**: 승격 없음 (Q1·Q2·Q4 필수 + Q3 전부 통과한 후보 0개). report/wave26_report.md 참조.")
    else:
        promoted_ids = ", ".join(r["candidate_id"] for r in rows if not r.get("missing") and r["candidate_id"] != BASELINE_CANDIDATE and r["promoted"])
        lines.append(f"**최종 판정**: 부분 승격 ({promoted_ids}) -- 단, 2차 탐색 산물이므로 신뢰도 낮음으로 표기. report/wave26_report.md 참조.")
    registry_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _metrics_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = ["| Candidate | 정의 | 거래수 | 승률 | 도박sleeve 최종액 | 도박sleeve CAGR | 전체시스템 CAGR | 총비용($) | 비용/슬리브 |", "|---|---|---|---|---|---|---|---|---|"]
    for row in rows:
        if row.get("missing"):
            continue
        cost = row["total_cost_usdt"]
        cost_frac = _pct(cost / GAMBLE_CAPITAL) if cost is not None else "N/A"
        lines.append(
            f"| {row['candidate_id']} | {row['definition']} | {row['n_trades']} | {_pct(row['win_rate'])} | {_usd(row['gamble_final_usdt'], 4)} | "
            f"{_pct(row['cagr_gamble_only'])} | {_pct(row['cagr_combined'])} | {_usd(cost)} | {cost_frac} |"
        )
    return lines


def _entry_admission_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = ["| Candidate | 진입기회(원신호 발화) | 실제진입 | 쿨다운 차단 | ADX/z 게이트 차단 | 차단율 |", "|---|---|---|---|---|---|"]
    for row in rows:
        if row.get("missing") or row["candidate_id"] == BASELINE_CANDIDATE:
            continue
        ea = row.get("entry_admission") or {}
        opp = ea.get("entry_opportunities")
        adm = ea.get("entries_admitted")
        cd = ea.get("blocked_by_cooldown")
        gt = ea.get("blocked_by_gate")
        if opp is None:
            lines.append(f"| {row['candidate_id']} | N/A | N/A | N/A | N/A | N/A |")
            continue
        blocked_rate = _pct((cd + gt) / opp) if opp else "N/A"
        lines.append(f"| {row['candidate_id']} | {opp} | {adm} | {cd} | {gt} | {blocked_rate} |")
    lines.append("")
    lines.append("해석: \"진입기회\"는 통제가 전혀 없었다면 열렸을 신규 진입 시도 횟수(원신호가 실제로 발화한 바)다. \"쿨다운 차단\"은 직전 청산 후 대기시간 때문에, \"ADX/z 게이트 차단\"은 레짐/신호강도 조건 미충족 때문에 그 시도가 무산된 횟수다 -- 이 표가 빈도 통제 3축이 실제로 작동했는지의 직접 증거다.")
    return lines


def _convexity_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = ["| Candidate | 왜도(skew) | 상위10% 거래 기여율 | 부트스트랩 skew p05 | 부트스트랩 P(skew>0) | 최대단일거래 PnL | Q1 볼록성 |", "|---|---|---|---|---|---|---|"]
    for row in rows:
        if row.get("missing"):
            continue
        skew = row["skew"]
        skew_str = f"{skew:.3f}" if skew is not None else "N/A"
        decile = row["top_decile_contribution"]
        decile_str = _pct(decile) if decile is not None else "N/A"
        boot_p05 = row["bootstrap_p05"]
        boot_p05_str = f"{boot_p05:.3f}" if boot_p05 is not None else "N/A(표본부족)"
        boot = row["bootstrap_fraction_positive_skew"]
        boot_str = _pct(boot) if boot is not None else "N/A(표본부족)"
        best = row["best_trade_pnl_usdt"]
        best_str = _usd(best, 2) if best is not None else "N/A"
        q1 = row["gates"].get("Q1", "N/A")
        lines.append(f"| {row['candidate_id']} | {skew_str} | {decile_str} | {boot_p05_str} | {boot_str} | {best_str} | {q1} |")
    return lines


def _series_from_records(records: list[dict[str, Any]]):
    import pandas as pd  # noqa: PANDAS_OK

    if not records:
        return pd.Series(dtype=float)
    index = pd.DatetimeIndex([pd.Timestamp(item["timestamp"]) for item in records if item.get("value") is not None])
    values = [float(item["value"]) for item in records if item.get("value") is not None]
    return pd.Series(values, index=index, dtype=float).sort_index()


def _yearly_pnl_table_impl(rows: list[dict[str, Any]]) -> list[str]:
    from research.wave26_freq.gates26 import calendar_year_return

    all_years: set[int] = set()
    per_row_series: dict[str, Any] = {}
    for row in rows:
        if row.get("missing"):
            continue
        series = _series_from_records(row["gamble_equity_records"])
        per_row_series[row["candidate_id"]] = series
        all_years.update(int(y) for y in series.index.year.unique())
    years_sorted = sorted(all_years)
    if not years_sorted:
        return []
    header = "| Candidate | " + " | ".join(str(y) for y in years_sorted) + " |"
    sep = "|---|" + "---|" * len(years_sorted)
    lines = [header, sep]
    for row in rows:
        if row.get("missing"):
            continue
        series = per_row_series[row["candidate_id"]]
        cells = []
        for year in years_sorted:
            ret = calendar_year_return(series, year)
            cells.append(_pct(ret) if ret is not None else "-")
        lines.append(f"| {row['candidate_id']} | " + " | ".join(cells) + " |")
    return lines


def _gate_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = ["| Candidate | Q1 볼록성 | Q2 파산방어 | Q3 C0초과 | Q4 비용효율(<=40%) | Q5 실행가능(참고) | 종합 |", "|---|---|---|---|---|---|---|"]
    for row in rows:
        if row.get("missing"):
            continue
        g = row["gates"]
        verdict = "PROMOTED" if row["promoted"] else row["overall_status"]
        if row["candidate_id"] == BASELINE_CANDIDATE:
            verdict = "기준선(C0, 승격판정 대상 아님)"
        lines.append(f"| {row['candidate_id']} | {g.get('Q1','-')} | {g.get('Q2','-')} | {g.get('Q3','-')} | {g.get('Q4','-')} | {g.get('Q5','-')} | {verdict} |")
    return lines


def _ranking_table(rows_by_id: dict[str, dict[str, Any]], promoted_ids: list[str]) -> list[str]:
    if not promoted_ids:
        return []
    ranked = sorted(promoted_ids, key=lambda cid: (rows_by_id[cid]["cagr_gamble_only"] if rows_by_id[cid]["cagr_gamble_only"] is not None else -1e9), reverse=True)
    lines = ["| 순위 | Candidate | 도박sleeve CAGR | 도박sleeve 최종액 | C0 대비 | 거래수 | 왜도 |", "|---|---|---|---|---|---|---|"]
    baseline_final = rows_by_id.get(BASELINE_CANDIDATE, {}).get("gamble_final_usdt")
    for rank, cid in enumerate(ranked, start=1):
        row = rows_by_id[cid]
        margin = None
        if baseline_final is not None and row["gamble_final_usdt"] is not None and baseline_final:
            margin = row["gamble_final_usdt"] / baseline_final - 1.0
        skew_str = f"{row['skew']:.3f}" if row["skew"] is not None else "N/A"
        margin_str = _pct(margin) if margin is not None else "N/A"
        lines.append(f"| {rank} | {cid} | {_pct(row['cagr_gamble_only'])} | {_usd(row['gamble_final_usdt'], 4)} | {margin_str} | {row['n_trades']} | {skew_str} |")
    return lines


def _wave25_before_row(wave26_candidate_id: str) -> dict[str, Any] | None:
    source_note = WAVE25_SIGNAL_SOURCE.get(wave26_candidate_id, "")
    b_id = source_note.split(" ")[0] if source_note else None
    if not b_id:
        return None
    path = WAVE25_RESULTS_DIR / f"{b_id}.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    trades = payload.get("trades", [])
    metadata = payload.get("metadata", {})
    final = _final_from_series(payload.get("gamble_equity", []), GAMBLE_CAPITAL)
    return {"source_id": b_id, "n_trades": len(trades), "total_cost_usdt": metadata.get("total_cost_usdt"), "gamble_final_usdt": final}


def _before_after_table(rows: list[dict[str, Any]]) -> list[str]:
    """SPEC.md-mandated: "통제 전(wave-25) vs 후(wave-26) 거래수·비용·최종액 대비표(빈도 통제가
    실제로 작동했는지 정량화)"."""
    lines = [
        "| Candidate | wave25 소스 | Before 거래수 | After 거래수 | 거래수 배율 | Before 비용 | After 비용 | 비용 배율 | Before 최종액 | After 최종액 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        if row.get("missing"):
            continue
        before = _wave25_before_row(row["candidate_id"])
        if before is None:
            lines.append(f"| {row['candidate_id']} | N/A | - | {row['n_trades']} | - | - | {_usd(row['total_cost_usdt'])} | - | - | {_usd(row['gamble_final_usdt'], 4)} |")
            continue
        trade_ratio = (row["n_trades"] / before["n_trades"]) if before["n_trades"] else float("nan")
        before_cost = before["total_cost_usdt"] or 0.0
        after_cost = row["total_cost_usdt"] or 0.0
        cost_ratio = (after_cost / before_cost) if before_cost else float("nan")
        lines.append(
            f"| {row['candidate_id']} | {before['source_id']} | {before['n_trades']} | {row['n_trades']} | {trade_ratio:.3f}x | "
            f"{_usd(before_cost)} | {_usd(after_cost)} | {cost_ratio:.3f}x | {_usd(before['gamble_final_usdt'], 4)} | {_usd(row['gamble_final_usdt'], 4)} |"
        )
    return lines


def _live_signal_section(live_payload: dict[str, Any] | None) -> list[str]:
    lines = ["## 오늘 라이브 신호 상태 (즉시 실행 가능 여부)", ""]
    if live_payload is None:
        lines.append("`--stage live`가 아직 실행되지 않았다 -- 라이브 신호 상태 없음.")
        return lines
    lines.append(f"생성 시각(UTC): {live_payload.get('generated_at_utc', 'N/A')} / 네트워크 사용: {live_payload.get('network_used')}")
    lines.append("")
    lines.append("이 표는 신규 진입 신호(또는 현재 보유 중인 포지션, 또는 신호는 있으나 통제에 막힌 상태) 상태만 표시한다 -- 매 실행 시점의 최신 캐시+실시간 데이터로부터 새로 판정한 스냅샷이다.")
    lines.append("")
    lines.append("| Candidate | BTCUSDT | ETHUSDT | SOLUSDT |")
    lines.append("|---|---|---|---|")
    candidates = live_payload.get("candidates", {})
    for cid in CANDIDATE_IDS:
        row = candidates.get(cid)
        if row is None:
            lines.append(f"| {cid} | - | - | - |")
            continue
        cells = []
        for symbol in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
            entry = row.get(symbol)
            if entry is None:
                cells.append("N/A")
                continue
            status = entry.get("status", "N/A")
            if status == "HOLDING":
                cells.append(f"HOLDING {('LONG' if entry.get('direction',0)>0 else 'SHORT')} @ {_usd(entry.get('entry_price'),2)} (unrl {_pct(entry.get('unrealized_pct'))})")
            elif status in ("FRESH_LONG_SIGNAL", "FRESH_SHORT_SIGNAL"):
                cells.append(f"**{status}** (px {_usd(entry.get('current_price'),2)})")
            elif status.startswith("SIGNAL_BLOCKED_COOLDOWN"):
                cells.append(f"signal present, COOLDOWN {entry.get('cooldown_bars_left','?')}bars left (px {_usd(entry.get('current_price'),2)})")
            elif status.startswith("SIGNAL_BLOCKED_GATE"):
                cells.append(f"signal present, GATE blocked (adx_ok={entry.get('adx_ok')}, z_ok={entry.get('z_ok')})")
            elif status == "FLAT_ARMED":
                cells.append(f"ARMED (anchor {_usd(entry.get('anchor'),2)}, px {_usd(entry.get('current_price'),2)}, {entry.get('admission_if_triggered','')})")
            elif status == "FLAT_NOT_ARMED":
                cells.append("not armed")
            else:
                cells.append(status)
        lines.append(f"| {cid} | " + " | ".join(cells) + " |")
    return lines


def write_wave26_report(results_dir: Path, report_dir: Path, registry_path: Path, live_payload: dict[str, Any] | None = None) -> None:
    rows = [_summary_row(cid, _load(results_dir, cid)) for cid in CANDIDATE_IDS]
    rows_by_id = {row["candidate_id"]: row for row in rows}

    promoted = [r["candidate_id"] for r in rows if not r.get("missing") and r["candidate_id"] != BASELINE_CANDIDATE and r["promoted"]]
    baseline_row = rows_by_id.get(BASELINE_CANDIDATE)

    lines: list[str] = []
    lines.append("# Wave-26 리포트 -- 빈도 통제: 볼록 신호를 비용 아래로 (C0-C7)")
    lines.append("")
    lines.append(
        "**출처 고지(정직성, 최우선)**: 이 wave의 가설(\"신호는 좋은데 너무 자주 쓴다 -> 빈도만 제한하면 산다\")은 "
        "**wave-25 결과를 직접 보고 사후에 만들었다** (wave-25 진단: 신규 지표 왜도가 V1보다 높은데도(B7 14.26 등 vs V1 1.76) 거래를 12~38배 해서 비용에 잠식됨). "
        "따라서 이 wave 전체는 **2차 탐색 산물**이며, 아래 통과 후보가 나오더라도 1차 사전등록(wave-25 이전 웨이브들)보다 **신뢰도가 낮다** -- "
        "결과 확인 후 파라미터를 고른 것은 아니지만(SPEC.md 동결 조건 준수), 가설 자체가 결과-유도적이라는 점은 구조적으로 남는다."
    )
    lines.append("")
    lines.append(f"자본구조: 총 ${TOTAL_CAPITAL:.0f} = 도박 ${GAMBLE_CAPITAL:.0f} + 안정 I5 ${STABLE_CAPITAL:.0f} (research/wave18_idle/results/I5.json, 재시뮬레이션 없이 원본 읽음) -- wave25와 동일 계약, 변경 없음.")
    lines.append("")
    lines.append("## 요약")
    lines.append("")
    if baseline_row and not baseline_row.get("missing"):
        baseline_skew_str = f"{baseline_row['skew']:.4f}" if baseline_row["skew"] is not None else "N/A"
        lines.append(f"**C0 기준선(V1 재현, 통제 없음)**: 도박sleeve 최종 {_usd(baseline_row['gamble_final_usdt'], 4)} (CAGR {_pct(baseline_row['cagr_gamble_only'])}), 거래수 {baseline_row['n_trades']}, 왜도 {baseline_skew_str}.")
        c0_before = _wave25_before_row("C0")
        if c0_before is not None:
            match = "일치" if abs(baseline_row["gamble_final_usdt"] - c0_before["gamble_final_usdt"]) < 0.01 and baseline_row["n_trades"] == c0_before["n_trades"] else "**불일치(!!)**"
            lines.append(
                f"**C0 vs wave25 B0 정합성 검증**: wave25 B0 최종 {_usd(c0_before['gamble_final_usdt'], 4)} (거래수 {c0_before['n_trades']}) -- {match} "
                "(C0는 통제가 없으므로 B0와 수치가 100% 같아야 한다; run_b0을 그대로 재사용하므로 동일 코드 경로다)."
            )
    if promoted:
        lines.append(f"**승격 후보**: {', '.join(promoted)} (Q1·Q2·Q4 필수 + Q3 충족) -- **2차 탐색 산물, 신뢰도 낮음으로 표기**")
    else:
        lines.append("**승격 후보**: 없음 (신규 후보(C1~C7) 중 Q1·Q2·Q4 필수 + Q3를 전부 통과한 후보 없음 -- 아래 게이트 표 참조)")
    c7_row = rows_by_id.get("C7")
    if c7_row and not c7_row.get("missing") and baseline_row and not baseline_row.get("missing"):
        c7_vs_c0 = "우위" if c7_row["gamble_final_usdt"] > baseline_row["gamble_final_usdt"] else "열위 또는 동률"
        lines.append(
            f"**C7(기준선+통제) 판정**: 도박sleeve 최종 {_usd(c7_row['gamble_final_usdt'], 4)} (거래수 {c7_row['n_trades']}) -- C0 대비 {c7_vs_c0}. "
            "C7이 C0를 이기면 빈도 통제가 V1 자체도 개선한다는 뜻이라 SPEC.md가 '특히 중요'로 표시한 결과다."
        )
    lines.append("")

    if promoted:
        lines.append("## 승격 후보 기대수익 순위표 (2차 탐색 산물 -- 신뢰도 낮음)")
        lines.append("")
        lines.extend(_ranking_table(rows_by_id, promoted))
        lines.append("")

    lines.append("## 통제 전(wave-25) vs 후(wave-26) 대비표 -- 빈도 통제가 실제로 작동했는가")
    lines.append("")
    lines.extend(_before_after_table(rows))
    lines.append("")
    lines.append("해석: \"거래수 배율\"/\"비용 배율\"이 1.0보다 충분히 작을수록 빈도 통제가 실제로 매매 빈도와 비용을 줄였다는 뜻이다. C0/C7의 소스는 둘 다 B0(V1)이다 -- C0는 무통제 재현(배율 1.0x가 나와야 함), C7만 실제 통제가 적용된다.")
    lines.append("")

    lines.append("## 진입 허용/차단 분해 (쿨다운 vs ADX/z 게이트 -- 어느 축이 실제로 억제했는가)")
    lines.append("")
    lines.extend(_entry_admission_table(rows))
    lines.append("")

    lines.append("## 후보 정의 및 성과 지표")
    lines.append("")
    lines.extend(_metrics_table(rows))
    lines.append("")

    lines.append("## Q1 볼록성 실증 (skew>0 AND 상위10%기여>=50% AND 부트스트랩 skew p05>0)")
    lines.append("")
    lines.extend(_convexity_table(rows))
    lines.append("")

    lines.append("## 연도별 손익 (전체 표기 -- 최악연도 악화는 기각사유 아님, 정직성 표기 의무)")
    lines.append("")
    yearly = _yearly_pnl_table_impl(rows)
    lines.extend(yearly if yearly else ["연도별 데이터 없음."])
    lines.append("")

    lines.append("## 게이트 Q1-Q5")
    lines.append("")
    lines.extend(_gate_table(rows))
    lines.append("")
    lines.append(
        "게이트 정의: Q1 볼록성(왜도>0 AND 상위10%거래가 총이익 50%+ AND 부트스트랩 skew p05>0, wave25 P1과 동일 기준) / "
        "Q2 파산방어(MC 1e4 P(전체자본<$50)<10% AND 단일 최대손실<=$25, 타협불가, wave25 P2와 동일) / Q3 도박sleeve 최종액 > C0 / "
        f"Q4 비용효율(총비용 <= 슬리브의 {Q4_MAX_COST_FRACTION_OF_SLEEVE*100:.0f}%={_usd(Q4_MAX_COST_USDT)}, 이 wave의 신규 핵심 게이트) / "
        "Q5 실행가능(레그>=$5, 무중첩 포지션, 3x슬리피지 부호유지 -- 참고용, 승격판정 미포함). "
        "**승격 = Q1·Q2·Q4 필수 + Q3 (전부 AND -- wave25의 P3-or-P4보다 엄격)**."
    )
    lines.append("")

    lines.append("## 다중검정 (참고용, 승격 판정에는 미적용)")
    lines.append("")
    lines.append(f"누적 {PRIOR_CUMULATIVE_TRIALS}(wave25까지) + 이 wave 8개(C0~C7) = {DSR_CUMULATIVE_TRIALS}후보 반영 DSR 보정 (SPEC.md 사전등록).")
    lines.append(
        "**주의**: DSR/Sharpe는 정규분포 가정 지표라 볼록(오른쪽 꼬리) 수익분포에서는 왜곡되기 쉽다 -- 이 wave의 실제 판정은 DSR이 아니라 Q1(왜도+상위10%기여+부트스트랩) 실측치를 쓴다. 이 표는 참고용일 뿐이다."
    )
    lines.append("")
    lines.append("| Candidate | DSR score(도박sleeve) | probability | DSR score(전체시스템) | probability | OOS 도박sleeve 총수익 |")
    lines.append("|---|---|---|---|---|---|")
    for row in rows:
        if row.get("missing"):
            continue
        payload = _load(results_dir, row["candidate_id"])
        ref = ((payload.get("gates_report") or {}).get("reference_metrics")) or {}
        gamble_dsr = ref.get("dsr_gamble_sleeve") or {}
        combined_dsr = ref.get("dsr_combined_system") or {}
        oos_gamble = ref.get("oos_gamble_sleeve") or {}
        lines.append(
            f"| {row['candidate_id']} | {gamble_dsr.get('score', 'N/A')} | {gamble_dsr.get('probability', 'N/A')} | "
            f"{combined_dsr.get('score', 'N/A')} | {combined_dsr.get('probability', 'N/A')} | {_pct(oos_gamble.get('oos_total_return')) if oos_gamble else 'N/A'} |"
        )
    lines.append("")

    lines.extend(_live_signal_section(live_payload))
    lines.append("")

    lines.append("## 결론")
    lines.append("")
    if not promoted:
        lines.append(
            "C1~C7 중 Q1·Q2·Q4 필수 + Q3를 전부 통과한 후보는 없다. SPEC.md의 사전 합의대로 정직하게 보고한다: "
            "**빈도 통제(쿨다운/ADX 레짐필터/신호강도 z-score)로도 wave-25 신규 지표들의 비용 잠식은 회복되지 않았다** -- "
            "즉 \"신호는 좋은데 너무 자주 쓴다\"는 wave-25발 가설 자체가 이 데이터에서는 기각된다. 위 대비표에서 거래수/비용이 실제로 줄었는지, "
            "줄었는데도 왜 게이트를 통과하지 못했는지(Q1 볼록성 붕괴/Q3 기준선 미달/Q4 비용한도 초과 등)를 후보별로 확인할 것. "
            "**해석: 이 신호들은 비용 구조상 소자본($25 슬리브)에서 성립하지 않는다는 24웨이브+wave25+wave26 누적 관찰과 일치하는 결과다.**"
        )
    else:
        top_id = sorted(promoted, key=lambda cid: (rows_by_id[cid]["cagr_gamble_only"] if rows_by_id[cid]["cagr_gamble_only"] is not None else -1e9), reverse=True)[0]
        top_row = rows_by_id[top_id]
        lines.append(
            f"{', '.join(promoted)}가 Q1·Q2·Q4 필수 + Q3를 통과해 승격 대상이다. 1위 {top_id}: 도박sleeve 최종 {_usd(top_row['gamble_final_usdt'],4)} "
            f"(CAGR {_pct(top_row['cagr_gamble_only'])}), C0 대비 {_usd(top_row['gamble_final_usdt'] - (baseline_row['gamble_final_usdt'] if baseline_row else GAMBLE_CAPITAL), 4)} 우위. "
            "**그러나 이 wave 전체가 wave-25 결과를 보고 만든 2차 탐색 산물이므로, 이 승격은 1차 사전등록보다 약한 증거로 취급해야 한다** -- "
            "독립적인 신규 데이터(향후 기간)에서 재확인되기 전까지는 배포 근거로 단독 사용하지 말 것. 나머지 후보의 실패 사유는 위 게이트 표 참조."
        )
    lines.append("")
    lines.append(
        "사후 추가 후보·파라미터 조정 없음 (SPEC.md 동결 조건 준수). 표본 부족 후보는 UNDETERMINED로 표시했다(위 게이트 표 참조). "
        "Q5는 참고 진단이며 승격 판정에는 포함하지 않았다(SPEC.md 승격식은 Q1~Q4만 명시)."
    )
    lines.append("")

    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "wave26_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_registry(results_dir, registry_path)


__all__ = ["write_registry", "write_wave26_report"]
