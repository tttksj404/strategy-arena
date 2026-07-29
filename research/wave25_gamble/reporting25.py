# Wave-25 report/registry writer. Reads ONLY research/wave25_gamble/results/*.json (written
# by run_wave25.py's run+gates stages, plus the separate live_signals.json from the `live`
# stage) -- no re-simulation happens here, matching research.wave20_convex.reporting20's own
# run/gates/report stage separation.

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Final

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

from research.wave25_gamble.configs25 import (
    CANDIDATE_IDS,
    DSR_CUMULATIVE_TRIALS,
    GA_GP_EVALUATIONS_DISCLOSED,
    GAMBLE_CAPITAL,
    STABLE_CAPITAL,
    TOTAL_CAPITAL,
)

BASELINE_CANDIDATE: Final = "B0"


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
    return {
        "candidate_id": candidate_id,
        "missing": False,
        "definition": payload.get("definition", ""),
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
        "gamble_equity_records": payload.get("gamble_equity", []),
        "stress_test": payload.get("stress_test", {}),
    }


def _registry_row(row: dict[str, Any]) -> str:
    if row.get("missing"):
        return f"| {row['candidate_id']} | wave25_gamble | NOT_RUN | - | - | - | - |"
    promoted = "YES" if row["promoted"] else "NO"
    gate_str = " ".join(f"{gid}:{status[0]}" for gid, status in sorted(row["gates"].items()))
    return f"| {row['candidate_id']} | wave25_gamble | EVALUATED | {_pct(row['cagr_gamble_only'])} | {_usd(row['gamble_final_usdt'], 4)} | {promoted} | {gate_str} |"


def write_registry(results_dir: Path, registry_path: Path) -> None:
    rows = [_summary_row(cid, _load(results_dir, cid)) for cid in CANDIDATE_IDS]
    n_promoted = sum(1 for r in rows if not r.get("missing") and r["candidate_id"] != BASELINE_CANDIDATE and r["promoted"])
    lines = [
        "# Wave-25 registry",
        "",
        "| Candidate | Family | State | 도박sleeve CAGR | 도박sleeve 최종($25 기준) | 승격 | 게이트(P1-P5) |",
        "|---|---|---|---|---|---|---|",
    ]
    lines.extend(_registry_row(row) for row in rows)
    lines.append("")
    if n_promoted == 0:
        lines.append("**최종 판정**: 승격 없음 (P1·P2 필수 + (P3 or P4) 전부 통과한 신규 후보 0개). report/wave25_report.md 참조.")
    else:
        promoted_ids = ", ".join(r["candidate_id"] for r in rows if not r.get("missing") and r["candidate_id"] != BASELINE_CANDIDATE and r["promoted"])
        lines.append(f"**최종 판정**: 부분 승격 ({promoted_ids}). report/wave25_report.md 참조.")
    registry_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _metrics_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = ["| Candidate | 정의 | 거래수 | 승률 | 도박sleeve 최종액 | 도박sleeve CAGR | 전체시스템 CAGR | 총비용($) |", "|---|---|---|---|---|---|---|---|"]
    for row in rows:
        if row.get("missing"):
            continue
        lines.append(
            f"| {row['candidate_id']} | {row['definition']} | {row['n_trades']} | {_pct(row['win_rate'])} | {_usd(row['gamble_final_usdt'], 4)} | "
            f"{_pct(row['cagr_gamble_only'])} | {_pct(row['cagr_combined'])} | {_usd(row['total_cost_usdt'])} |"
        )
    return lines


def _convexity_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = ["| Candidate | 왜도(skew) | 상위10% 거래 기여율 | 부트스트랩 skew p05 | 부트스트랩 P(skew>0) | 최대단일거래 PnL | P1 볼록성 |", "|---|---|---|---|---|---|---|"]
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
        p1 = row["gates"].get("P1", "N/A")
        lines.append(f"| {row['candidate_id']} | {skew_str} | {decile_str} | {boot_p05_str} | {boot_str} | {best_str} | {p1} |")
    return lines


def _honesty_section(rows: list[dict[str, Any]]) -> list[str]:
    lines = ["## 정직성 진단 -- \"운 하나\" vs 구조적 볼록성", ""]
    for row in rows:
        if row.get("missing"):
            continue
        cid = row["candidate_id"]
        best = row["best_trade_pnl_usdt"]
        final = row["gamble_final_usdt"]
        boot = row["bootstrap_fraction_positive_skew"]
        n_trades = row["n_trades"]
        if best is None:
            lines.append(f"- **{cid}**: 거래 없음 또는 진단 불가.")
            continue
        gain = final - GAMBLE_CAPITAL
        share = best / gain if (gain > 0 and best > 0) else None
        boot_note = f"부트스트랩 {n_trades}건 리샘플 {_pct(boot)}에서 skew>0 유지" if boot is not None else f"거래수 {n_trades}건으로 부트스트랩 표본 부족"
        if share is not None:
            share_note = f"최대단일거래가 순이익의 {_pct(share)} 근사 기여"
        elif gain <= 0:
            share_note = f"도박sleeve 순손실({_usd(gain, 2)}) -- 개별 승리거래(최대 {_usd(best, 2)})가 있어도 전체는 마이너스라 기여율 정의 불가"
        else:
            share_note = "단일거래 기여율 정의 불가"
        lines.append(f"- **{cid}**: {share_note}. {boot_note}. (최대거래 {_usd(best, 2)}, 도박sleeve 최종 {_usd(final, 4)})")
    lines.append("")
    lines.append(
        "해석: 부트스트랩 skew p05(P1의 세 번째 조건)가 0 이하면 리샘플 다수 경로에서 볼록성이 사라진다는 뜻 -- 이 경우 P1은 FAIL이다. "
        "최대단일거래 기여율이 100%에 가까우면 사실상 그 거래 하나가 결과를 결정했다는 뜻이며, 이는 P1 PASS 여부와 별개로 강건성을 보여주는 추가 진단이다."
    )
    return lines


def _series_from_records(records: list[dict[str, Any]]):
    import pandas as pd  # noqa: PANDAS_OK

    if not records:
        return pd.Series(dtype=float)
    index = pd.DatetimeIndex([pd.Timestamp(item["timestamp"]) for item in records if item.get("value") is not None])
    values = [float(item["value"]) for item in records if item.get("value") is not None]
    return pd.Series(values, index=index, dtype=float).sort_index()


def _yearly_pnl_table_impl(rows: list[dict[str, Any]]) -> list[str]:
    """SPEC.md: "최악연도 악화는 기각 사유가 아니다 ... 연도별 손익을 전부 표기해 어떤 장세에서
    지는지 드러낸다" -- every calendar year the candidate's own gamble-sleeve equity covers,
    not just a fixed 2-year worst-case subset."""
    from research.wave25_gamble.gates25 import calendar_year_return

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
    lines = ["| Candidate | P1 볼록성 | P2 파산방어 | P3 B0초과 | P4 30일상위25% | P5 실행가능(참고) | 종합 |", "|---|---|---|---|---|---|---|"]
    for row in rows:
        if row.get("missing"):
            continue
        g = row["gates"]
        verdict = "PROMOTED" if row["promoted"] else row["overall_status"]
        if row["candidate_id"] == BASELINE_CANDIDATE:
            verdict = "기준선(B0, 승격판정 대상 아님)"
        lines.append(f"| {row['candidate_id']} | {g.get('P1','-')} | {g.get('P2','-')} | {g.get('P3','-')} | {g.get('P4','-')} | {g.get('P5','-')} | {verdict} |")
    return lines


def _ranking_table(rows_by_id: dict[str, dict[str, Any]], promoted_ids: list[str]) -> list[str]:
    if not promoted_ids:
        return []
    ranked = sorted(promoted_ids, key=lambda cid: (rows_by_id[cid]["cagr_gamble_only"] if rows_by_id[cid]["cagr_gamble_only"] is not None else -1e9), reverse=True)
    lines = ["| 순위 | Candidate | 도박sleeve CAGR | 도박sleeve 최종액 | B0 대비 | 거래수 | 왜도 |", "|---|---|---|---|---|---|---|"]
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


def _live_signal_section(live_payload: dict[str, Any] | None) -> list[str]:
    lines = ["## 오늘 라이브 신호 상태 (즉시 실행 가능 여부)", ""]
    if live_payload is None:
        lines.append("`--stage live`가 아직 실행되지 않았다 -- 라이브 신호 상태 없음.")
        return lines
    lines.append(f"생성 시각(UTC): {live_payload.get('generated_at_utc', 'N/A')} / 네트워크 사용: {live_payload.get('network_used')}")
    lines.append("")
    lines.append("이 표는 신규 진입 신호(또는 현재 보유 중인 포지션) 상태만 표시한다 -- 과거 보유이력을 지속 추적하는 상태머신이 아니라, 매 실행 시점의 최신 캐시+실시간 데이터로부터 새로 판정한 스냅샷이다.")
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
            elif status == "FLAT_ARMED":
                cells.append(f"ARMED (anchor {_usd(entry.get('anchor'),2)}, px {_usd(entry.get('current_price'),2)})")
            elif status == "FLAT_NOT_ARMED":
                cells.append("not armed")
            else:
                cells.append(status)
        lines.append(f"| {cid} | " + " | ".join(cells) + " |")
    return lines


def write_wave25_report(results_dir: Path, report_dir: Path, registry_path: Path, live_payload: dict[str, Any] | None = None) -> None:
    rows = [_summary_row(cid, _load(results_dir, cid)) for cid in CANDIDATE_IDS]
    rows_by_id = {row["candidate_id"]: row for row in rows}

    promoted = [r["candidate_id"] for r in rows if not r.get("missing") and r["candidate_id"] != BASELINE_CANDIDATE and r["promoted"]]
    baseline_row = rows_by_id.get(BASELINE_CANDIDATE)

    lines: list[str] = []
    lines.append("# Wave-25 리포트 -- 단기 도박 토너먼트 (B0-B7, 미탐색 지표 x 볼록 구조 강제)")
    lines.append("")
    lines.append(f"자본구조: 총 ${TOTAL_CAPITAL:.0f} = 도박 ${GAMBLE_CAPITAL:.0f} + 안정 I5 ${STABLE_CAPITAL:.0f} (research/wave18_idle/results/I5.json, 재시뮬레이션 없이 원본 읽음)")
    lines.append("")
    lines.append("## 요약")
    lines.append("")
    if baseline_row and not baseline_row.get("missing"):
        baseline_skew_str = f"{baseline_row['skew']:.4f}" if baseline_row["skew"] is not None else "N/A"
        lines.append(f"**B0 기준선(V1 재현)**: 도박sleeve 최종 {_usd(baseline_row['gamble_final_usdt'], 4)} (CAGR {_pct(baseline_row['cagr_gamble_only'])}), 거래수 {baseline_row['n_trades']}, 왜도 {baseline_skew_str}.")
    if promoted:
        lines.append(f"**승격 후보**: {', '.join(promoted)} (P1·P2 필수 + (P3 or P4) 충족)")
    else:
        lines.append("**승격 후보**: 없음 (신규 후보(B1~B7) 중 P1·P2 필수 + (P3 or P4)를 전부 통과한 후보 없음 -- 아래 게이트 표 참조)")
    lines.append("")

    if promoted:
        lines.append("## 승격 후보 기대수익 순위표")
        lines.append("")
        lines.extend(_ranking_table(rows_by_id, promoted))
        lines.append("")

    lines.append("## 후보 정의 및 성과 지표")
    lines.append("")
    lines.extend(_metrics_table(rows))
    lines.append("")

    lines.append("## P1 볼록성 실증 (이 wave의 핵심 판정 -- skew>0 AND 상위10%기여>=50% AND 부트스트랩 skew p05>0)")
    lines.append("")
    lines.extend(_convexity_table(rows))
    lines.append("")
    lines.extend(_honesty_section(rows))
    lines.append("")

    lines.append("## 연도별 손익 (전체 표기 -- 최악연도 악화는 기각사유 아님, 정직성 표기 의무)")
    lines.append("")
    yearly = _yearly_pnl_table_impl(rows)
    lines.extend(yearly if yearly else ["연도별 데이터 없음."])
    lines.append("")

    lines.append("## 게이트 P1-P5")
    lines.append("")
    lines.extend(_gate_table(rows))
    lines.append("")
    lines.append(
        "게이트 정의: P1 볼록성(왜도>0 AND 상위10%거래가 총이익 50%+ AND 부트스트랩 skew p05>0, 미달=즉시 기각) / "
        "P2 파산방어(MC 1e4 P(전체자본<$50)<10% AND 단일 최대손실<=$25, 타협불가) / P3 도박sleeve 최종액 > B0 / "
        "P4 30일 롤링창 상위25% 평균 > B0 / P5 실행가능(레그>=$5, 무중첩 포지션, 3x슬리피지 부호유지 -- 참고용, 승격판정 미포함). "
        "승격 = P1·P2 필수 + (P3 or P4)."
    )
    lines.append("")

    lines.append("## 다중검정 (참고용, 승격 판정에는 미적용)")
    lines.append("")
    lines.append(f"누적 {DSR_CUMULATIVE_TRIALS}후보 + GA/GP {GA_GP_EVALUATIONS_DISCLOSED:,}평가 반영 DSR 보정 (SPEC.md 사전등록). 이 wave 자체(B0~B7, 8개)가 그 {DSR_CUMULATIVE_TRIALS}후보에 포함되는 신규 시행이다.")
    lines.append(
        "**주의**: DSR/Sharpe는 정규분포 가정 지표라 볼록(오른쪽 꼬리) 수익분포에서는 왜곡되기 쉽다 -- 이 wave의 실제 판정은 DSR이 아니라 P1(왜도+상위10%기여+부트스트랩) 실측치를 쓴다. 이 표는 참고용일 뿐이다."
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

    lines.append("## 왜 B1~B7이 B0에 못 미쳤는가 -- 진단")
    lines.append("")
    new_rows = [r for r in rows if not r.get("missing") and r["candidate_id"] != BASELINE_CANDIDATE]
    if new_rows and baseline_row and not baseline_row.get("missing"):
        avg_trades = sum(r["n_trades"] for r in new_rows) / len(new_rows)
        avg_cost = sum(r["total_cost_usdt"] or 0.0 for r in new_rows) / len(new_rows)
        freq_multiple = avg_trades / baseline_row["n_trades"] if baseline_row["n_trades"] else float("nan")
        cost_fraction_of_sleeve = avg_cost / GAMBLE_CAPITAL
        p1_pass_count = sum(1 for r in new_rows if r["gates"].get("P1") == "PASS")
        p2_pass_count = sum(1 for r in new_rows if r["gates"].get("P2") == "PASS")
        lines.append(
            f"B1~B7 평균 거래수는 {avg_trades:,.0f}건으로 B0({baseline_row['n_trades']}건)의 약 {freq_multiple:,.0f}배다 -- "
            f"B0(V1)는 일봉 저변동성 레짐 필터로 진입을 걸러 저빈도 반전매매만 하는 반면, B1~B7은 시간봉 지표 신호마다(스톱아웃 후 재진입 포함) 즉시 재진입해 훨씬 잦은 매매를 만든다."
        )
        lines.append(
            f"평균 총비용은 슬리브당 {_usd(avg_cost)}로 원금 ${GAMBLE_CAPITAL:.0f}의 {_pct(cost_fraction_of_sleeve)}에 달한다 -- "
            "매 진입·청산마다 메이커 수수료+실측 슬리피지가 붙는 구조에서 이 정도 빈도는 방향성 엣지가 있어도 비용이 잠식하기 쉽다."
        )
        lines.append(
            f"그럼에도 P1(볼록성)은 {p1_pass_count}/{len(new_rows)}개 통과, P2(파산방어)는 {p2_pass_count}/{len(new_rows)}개 통과했다 -- "
            "즉 개별 거래 단위의 손익 '모양'(자주 작게 잃고 가끔 크게 버는 구조)과 파산 방어(손절+포지션 크기 제한)는 대체로 의도대로 작동했지만, "
            "그 볼록한 분포조차 거래당 비용을 이기기엔 빈도가 너무 높았다는 뜻이다 -- **신호 자체(엣지)의 문제라기보다 신호 빈도(비용 잠식)의 문제**로 읽힌다."
        )
        lines.append(
            "**차기 웨이브 제언(이번 wave에서는 시도하지 않음)**: 동일 지표에 재진입 쿨다운이나 일봉 레짐 필터(B0가 쓰는 것과 같은 종류)를 추가하면 "
            "빈도·비용이 줄어 P3/P4 통과 가능성이 있다 -- 그러나 이는 결과를 본 뒤의 파라미터 추가이므로 SPEC.md 동결 조건(\"사후 추가·파라미터 조정 금지\") 위반이다. "
            "이번 wave의 B1~B7은 사전등록된 그대로 재조정 없이 보고하고, 쿨다운/레짐필터 버전은 별도로 사전등록해 다음 wave에서 신규 후보로 시험해야 한다."
        )
    else:
        lines.append("진단에 필요한 데이터가 부족하다.")
    lines.append("")

    lines.append("## 결론")
    lines.append("")
    if not promoted:
        lines.append(
            "B1~B7 중 P1·P2 필수 + (P3 or P4)를 전부 통과한 후보는 없다. SPEC.md의 사전 합의대로 정직하게 보고한다: "
            "미탐색 지표(MACD/ADX/슈퍼트렌드/켈트너/다중시간대/스토캐스틱)에 볼록 구조를 강제해도 B0(wave-20 V1)를 능가하는 새 엣지는 확인되지 않았다 -- "
            "**V1이 이 저장소의 도박 티어 상한**이라는 24웨이브+이번 wave 누적 관찰과 일치하는 결과다."
        )
    else:
        top_id = sorted(promoted, key=lambda cid: (rows_by_id[cid]["cagr_gamble_only"] if rows_by_id[cid]["cagr_gamble_only"] is not None else -1e9), reverse=True)[0]
        top_row = rows_by_id[top_id]
        lines.append(
            f"{', '.join(promoted)}가 P1·P2 필수 + (P3 or P4)를 통과해 승격 대상이다. 1위 {top_id}: 도박sleeve 최종 {_usd(top_row['gamble_final_usdt'],4)} "
            f"(CAGR {_pct(top_row['cagr_gamble_only'])}), B0 대비 {_usd(top_row['gamble_final_usdt'] - (baseline_row['gamble_final_usdt'] if baseline_row else GAMBLE_CAPITAL), 4)} 우위. "
            "나머지 후보의 실패 사유는 위 게이트 표 참조."
        )
    lines.append("")
    lines.append("사후 추가 후보·파라미터 조정 없음 (SPEC.md 동결 조건 준수). 표본 부족 후보는 UNDETERMINED로 표시했다(위 게이트 표 참조). P5는 참고 진단이며 승격 판정에는 포함하지 않았다(SPEC.md 승격식은 P1~P4만 명시).")
    lines.append("")

    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "wave25_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_registry(results_dir, registry_path)


__all__ = ["write_registry", "write_wave25_report"]
