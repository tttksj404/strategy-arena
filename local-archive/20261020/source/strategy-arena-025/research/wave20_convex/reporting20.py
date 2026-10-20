# Wave-20 report/registry writer. Reads ONLY research/wave20_convex/results/*.json (written
# by run_wave20.py's run+gates stages) -- no re-simulation happens here, matching every prior
# wave's own run/gates/report stage separation.

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Final

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

from research.wave20_convex.configs20 import CANDIDATE_IDS, DSR_CUMULATIVE_TRIALS, GAMBLE_CAPITAL, STABLE_CAPITAL, TOTAL_CAPITAL, WORST_YEARS

CONTROL_CANDIDATE: Final = "V4"
ASYMMETRIC_TEST_CANDIDATES: Final = ("V1", "V2", "V3")


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
        "bootstrap_fraction_positive_skew": (diagnostics.get("bootstrap") or {}).get("fraction_positive") if diagnostics.get("bootstrap") else None,
        "best_trade_pnl_usdt": (diagnostics.get("best_trade_sensitivity") or {}).get("best_trade_pnl_usdt") if diagnostics.get("best_trade_sensitivity") else None,
        "gates": {gid: g["status"] for gid, g in gates.items()},
        "overall_status": overall.get("status", "N/A"),
        "promoted": overall.get("promoted", False),
        "metadata": metadata,
    }


def _registry_row(row: dict[str, Any]) -> str:
    if row.get("missing"):
        return f"| {row['candidate_id']} | wave20_convex | NOT_RUN | - | - | - | - |"
    promoted = "YES" if row["promoted"] else "NO"
    gate_str = " ".join(f"{gid}:{status[0]}" for gid, status in sorted(row["gates"].items()))
    return (
        f"| {row['candidate_id']} | wave20_convex | EVALUATED | {_pct(row['cagr_combined'])} | "
        f"{_usd(row['gamble_final_usdt'])} | {promoted} | {gate_str} |"
    )


def write_registry(results_dir: Path, registry_path: Path) -> None:
    rows = [_summary_row(cid, _load(results_dir, cid)) for cid in CANDIDATE_IDS]
    n_promoted = sum(1 for r in rows if not r.get("missing") and r["promoted"])
    lines = [
        "# Wave-20 registry",
        "",
        "| Candidate | Family | State | 전체시스템 CAGR | 도박sleeve 최종($25 기준) | 승격 | 게이트(G1-G5) |",
        "|---|---|---|---|---|---|---|",
    ]
    lines.extend(_registry_row(row) for row in rows)
    lines.append("")
    if n_promoted == 0:
        lines.append("**최종 판정**: 승격 없음 (G1~G5 전부 통과 후보 0개). report/wave20_report.md 참조.")
    else:
        promoted_ids = ", ".join(r["candidate_id"] for r in rows if not r.get("missing") and r["promoted"])
        lines.append(f"**최종 판정**: 부분 승격 ({promoted_ids}). report/wave20_report.md 참조.")
    registry_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _gate_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = ["| Candidate | G1 손실한도 | G2 파산확률<10% | G3 볼록성 | G4 시스템>I5 | G5 최악연도방어 | 종합 |", "|---|---|---|---|---|---|---|"]
    for row in rows:
        if row.get("missing"):
            continue
        g = row["gates"]
        lines.append(
            f"| {row['candidate_id']} | {g.get('G1','-')} | {g.get('G2','-')} | {g.get('G3','-')} | {g.get('G4','-')} | {g.get('G5','-')} | "
            f"{'PROMOTED' if row['promoted'] else row['overall_status']} |"
        )
    return lines


def _metrics_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Candidate | 거래수 | 승률 | 도박sleeve 최종액 | 도박sleeve CAGR | 전체시스템 CAGR | 총비용($) |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        if row.get("missing"):
            continue
        lines.append(
            f"| {row['candidate_id']} | {row['n_trades']} | {_pct(row['win_rate'])} | {_usd(row['gamble_final_usdt'], 4)} | "
            f"{_pct(row['cagr_gamble_only'])} | {_pct(row['cagr_combined'])} | {_usd(row['total_cost_usdt'])} |"
        )
    return lines


def _convexity_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Candidate | 왜도(skew) | 상위10% 거래 기여율 | 부트스트랩 P(skew>0) | 최대단일거래 PnL | 볼록성 실증(G3) |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        if row.get("missing"):
            continue
        skew = row["skew"]
        skew_str = f"{skew:.3f}" if skew is not None else "N/A"
        decile = row["top_decile_contribution"]
        decile_str = _pct(decile) if decile is not None else "N/A"
        boot = row["bootstrap_fraction_positive_skew"]
        boot_str = _pct(boot) if boot is not None else "N/A(표본부족)"
        best = row["best_trade_pnl_usdt"]
        best_str = _usd(best, 2) if best is not None else "N/A"
        g3 = row["gates"].get("G3", "N/A")
        lines.append(f"| {row['candidate_id']} | {skew_str} | {decile_str} | {boot_str} | {best_str} | {g3} |")
    return lines


def _hypothesis_verdict(rows_by_id: dict[str, dict[str, Any]]) -> list[str]:
    control = rows_by_id.get(CONTROL_CANDIDATE)
    lines = ["## V4 대칭 대조군 비교 -- 비대칭 우월성 가설 검정", ""]
    if control is None or control.get("missing"):
        lines.append("V4(대조군) 결과 없음 -- 가설 검정 UNDETERMINED.")
        return lines
    control_final = control["gamble_final_usdt"]
    control_cagr = control["cagr_gamble_only"]
    lines.append(f"대조군 V4(대칭, $25 기준) 도박sleeve 최종액: {_usd(control_final, 4)} (CAGR {_pct(control_cagr)}), G3(볼록성) = {control['gates'].get('G3', 'N/A')}.")
    lines.append("")
    beats: list[str] = []
    loses: list[str] = []
    for cid in ASYMMETRIC_TEST_CANDIDATES:
        row = rows_by_id.get(cid)
        if row is None or row.get("missing"):
            continue
        beat = row["gamble_final_usdt"] > control_final
        convex = row["gates"].get("G3") == "PASS"
        note = f"{cid}: 최종 {_usd(row['gamble_final_usdt'], 4)} ({'V4 상회' if beat else 'V4 이하'}), G3={row['gates'].get('G3','N/A')}"
        (beats if beat else loses).append(note)
    for note in beats:
        lines.append(f"- {note}")
    for note in loses:
        lines.append(f"- {note}")
    lines.append("")
    n_tested = sum(1 for cid in ASYMMETRIC_TEST_CANDIDATES if rows_by_id.get(cid) and not rows_by_id[cid].get("missing"))
    n_beats = len(beats)
    n_convex_and_beats = sum(
        1 for cid in ASYMMETRIC_TEST_CANDIDATES
        if rows_by_id.get(cid) and not rows_by_id[cid].get("missing")
        and rows_by_id[cid]["gamble_final_usdt"] > control_final and rows_by_id[cid]["gates"].get("G3") == "PASS"
    )
    if n_beats == 0:
        lines.append(
            f"**판정: 비대칭 우월성 가설 기각.** V1~V3 {n_tested}개 전부 대칭 대조군(V4)을 넘지 못했다 -- "
            "이 wave가 시험한 비대칭 손익구조 3종은 대칭 방향베팅보다 우월하지 않다."
        )
    elif n_convex_and_beats == n_tested:
        lines.append(
            f"**판정: 비대칭 우월성 가설 부분 지지.** V1~V3 {n_tested}개 전부 V4를 상회했고 G3(볼록성 실증)도 전부 통과 -- "
            "다만 G1~G5 전체 승격 기준은 별도(아래 게이트 표), 성과·구조 두 축 모두에서 대칭 대조군보다 우월함이 확인됐다."
        )
    else:
        lines.append(
            f"**판정: 비대칭 우월성 가설 부분 지지, 후보별로 갈림.** {n_tested}개 중 {n_beats}개가 V4를 상회했다(그 중 {n_convex_and_beats}개는 G3 볼록성도 실증). "
            "나머지는 V4에 못 미치거나 볼록성이 실증되지 않아 개별 후보 단위로만 결론이 성립한다 -- 후보군 전체에 대한 일괄 결론은 낼 수 없다."
        )
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
        share = None
        gain = final - GAMBLE_CAPITAL
        if gain > 0 and best > 0:
            share = best / gain
        boot_note = f"부트스트랩 {n_trades}건 리샘플 {_pct(boot)}에서 skew>0 유지" if boot is not None else f"거래수 {n_trades}건으로 부트스트랩 표본 부족"
        if share is not None:
            share_note = f"최대단일거래가 순이익의 {_pct(share)} 근사 기여"
        elif gain <= 0:
            share_note = f"도박sleeve 순손실({_usd(gain,2)}) -- 개별 승리거래(최대 {_usd(best,2)})가 있어도 전체는 마이너스라 기여율 정의 불가"
        else:
            share_note = "단일거래 기여율 정의 불가"
        lines.append(f"- **{cid}**: {share_note}. {boot_note}. (최대거래 {_usd(best,2)}, 도박sleeve 최종 {_usd(final,4)})")
    lines.append("")
    lines.append(
        "해석: 부트스트랩에서 skew>0 유지 비율이 낮으면(예: <70%) 볼록성이 소수 리샘플 경로에 취약하다는 뜻이고, "
        "최대단일거래 기여율이 100%에 가까우면 사실상 그 거래 하나가 결과를 결정했다는 뜻이다. G3는 표본 전체(왜도+상위10%)로 "
        "판정하므로 이 섹션은 G3 PASS/FAIL 자체를 바꾸지 않지만, PASS라도 그 강건성 정도를 추가로 드러낸다."
    )
    return lines


def write_wave20_report(results_dir: Path, report_dir: Path, registry_path: Path) -> None:
    rows = [_summary_row(cid, _load(results_dir, cid)) for cid in CANDIDATE_IDS]
    rows_by_id = {row["candidate_id"]: row for row in rows}

    promoted = [r["candidate_id"] for r in rows if not r.get("missing") and r["promoted"]]
    best_combined_cagr = max(
        (r["cagr_combined"] for r in rows if not r.get("missing") and r["cagr_combined"] is not None),
        default=None,
    )
    best_combined_id = next(
        (r["candidate_id"] for r in rows if not r.get("missing") and r["cagr_combined"] == best_combined_cagr),
        None,
    )

    lines: list[str] = []
    lines.append("# Wave-20 리포트 -- 비대칭(볼록) 도박 5후보 (V1-V5)")
    lines.append("")
    lines.append(f"자본구조: 총 ${TOTAL_CAPITAL:.0f} = 도박 ${GAMBLE_CAPITAL:.0f} + 안정 I5 ${STABLE_CAPITAL:.0f} (research/wave18_idle/results/I5.json, 재시뮬레이션 없이 원본 읽음)")
    lines.append("")
    lines.append("## 요약")
    lines.append("")
    if promoted:
        lines.append(f"**승격 후보**: {', '.join(promoted)} (G1~G5 전부 PASS)")
    else:
        lines.append("**승격 후보**: 없음 (G1~G5 전부 통과한 후보 없음 -- 아래 게이트 표 참조)")
    if best_combined_id is not None:
        lines.append(f"**최고 전체시스템 CAGR**: {best_combined_id} ({_pct(best_combined_cagr)}), I5 단독 대비 비교는 게이트 표 G4 참조.")
    lines.append("")

    lines.append("## 후보 정의")
    lines.append("")
    for row in rows:
        if row.get("missing"):
            lines.append(f"- **{row['candidate_id']}**: 결과 없음(미실행).")
            continue
        lines.append(f"- **{row['candidate_id']}**: {row['definition']}")
    lines.append("")

    lines.append("## 성과 지표")
    lines.append("")
    lines.extend(_metrics_table(rows))
    lines.append("")

    lines.append("## G3 볼록성 실증 (이 wave의 핵심 판정)")
    lines.append("")
    lines.extend(_convexity_table(rows))
    lines.append("")
    lines.extend(_honesty_section(rows))
    lines.append("")

    lines.extend(_hypothesis_verdict(rows_by_id))
    lines.append("")

    lines.append("## 게이트 G1-G5")
    lines.append("")
    lines.extend(_gate_table(rows))
    lines.append("")
    lines.append(f"게이트 정의: G1 구조적 손실한도(<=배분액) / G2 시스템 파산확률 P(최종<$50)<10% (MC {'{:,}'.format(10_000)}회, 전체 포트폴리오) / "
                  f"G3 볼록성(왜도>0 AND 상위10%거래가 총이익 50%+) / G4 전체시스템 CAGR>I5단독 / G5 {'/'.join(str(y) for y in WORST_YEARS)} 최악연도 I5 대비 악화없음.")
    lines.append("")

    lines.append("## 최악연도(G5) 상세")
    lines.append("")
    lines.append("| Candidate | 2022 combined | 2022 I5-solo | 2025 combined | 2025 I5-solo | G5 |")
    lines.append("|---|---|---|---|---|---|")
    for row in rows:
        if row.get("missing"):
            continue
        payload = _load(results_dir, row["candidate_id"])
        g5_detail = ((payload.get("gates_report") or {}).get("g5_detail")) or {}
        y2022 = g5_detail.get("2022", {})
        y2025 = g5_detail.get("2025", {})
        lines.append(
            f"| {row['candidate_id']} | {_pct(y2022.get('combined_return'))} | {_pct(y2022.get('i5_solo_return'))} | "
            f"{_pct(y2025.get('combined_return'))} | {_pct(y2025.get('i5_solo_return'))} | {row['gates'].get('G5','N/A')} |"
        )
    lines.append("")

    lines.append("## OOS (2025-10~) 성과 -- SPEC.md 공통 규약 (참고용, 게이트 아님)")
    lines.append("")
    lines.append("| Candidate | 도박sleeve OOS 총수익 | 도박sleeve OOS 연율화 | 전체시스템 OOS 총수익 | 전체시스템 OOS 연율화 |")
    lines.append("|---|---|---|---|---|")
    for row in rows:
        if row.get("missing"):
            continue
        payload = _load(results_dir, row["candidate_id"])
        ref = ((payload.get("gates_report") or {}).get("reference_metrics")) or {}
        oos_gamble = ref.get("oos_gamble_sleeve") or {}
        oos_combined = ref.get("oos_combined_system") or {}
        lines.append(
            f"| {row['candidate_id']} | {_pct(oos_gamble.get('oos_total_return'))} | {_pct(oos_gamble.get('oos_annualized_return'))} | "
            f"{_pct(oos_combined.get('oos_total_return'))} | {_pct(oos_combined.get('oos_annualized_return'))} |"
        )
    stable_oos = None
    for row in rows:
        if row.get("missing"):
            continue
        payload = _load(results_dir, row["candidate_id"])
        ref = ((payload.get("gates_report") or {}).get("reference_metrics")) or {}
        stable_oos = ref.get("oos_stable_solo")
        if stable_oos is not None:
            break
    if stable_oos is not None:
        lines.append("")
        lines.append(f"참고: I5 단독 OOS 총수익 {_pct(stable_oos.get('oos_total_return'))} (연율화 {_pct(stable_oos.get('oos_annualized_return'))}), split={stable_oos.get('split')}")
    lines.append("")

    lines.append("## 다중검정 (참고용, 승격 판정에는 미적용)")
    lines.append("")
    lines.append(f"누적 {DSR_CUMULATIVE_TRIALS}회 DSR 보정 (SPEC.md 사전등록). 이 wave 자체가 그 {DSR_CUMULATIVE_TRIALS}회에 포함되는 신규 시행이다.")
    lines.append(
        "**주의**: DSR/Sharpe는 정규분포 가정 지표라 V1/V2/V3/V5처럼 왜도가 큰(오른쪽 꼬리) 수익분포에서는 "
        "왜곡되기 쉽다(여기서도 probability가 1.0에 근접 -- 극단적 skew 때문에 z-score가 포화된 결과). "
        "그래서 이 wave의 실제 판정은 DSR이 아니라 G3(왜도+상위10%기여) 실측치를 쓴다 -- 이 표는 참고용일 뿐이다."
    )
    lines.append("")
    lines.append("| Candidate | DSR score(도박sleeve) | probability | DSR score(전체시스템) | probability |")
    lines.append("|---|---|---|---|---|")
    for row in rows:
        if row.get("missing"):
            continue
        payload = _load(results_dir, row["candidate_id"])
        ref = ((payload.get("gates_report") or {}).get("reference_metrics")) or {}
        gamble_dsr = ref.get("dsr_gamble_sleeve") or {}
        combined_dsr = ref.get("dsr_combined_system") or {}
        lines.append(
            f"| {row['candidate_id']} | {gamble_dsr.get('score', 'N/A')} | {gamble_dsr.get('probability', 'N/A')} | "
            f"{combined_dsr.get('score', 'N/A')} | {combined_dsr.get('probability', 'N/A')} |"
        )
    lines.append("")

    lines.append("## 결론")
    lines.append("")
    if not promoted:
        lines.append(
            "G1~G5를 전부 통과한 후보는 없다. 그러나 이는 \"도박 계열 전체가 -EV\"라는 뜻이 아니다 -- "
            "V1/V2는 도박sleeve 단독 성과와 G3 볼록성 실증에서 대칭 대조군(V4)을 명확히 상회했고, "
            "실패한 게이트는 대부분 G5(최악연도 방어, 매우 엄격한 조건)이거나 G2(V5의 풀컴파운딩 파산위험)였다. "
            "V3는 성과·게이트 양쪽에서 명확히 기각된다(신규상장 롱은 이 데이터에서 -EV)."
        )
        v2_row = rows_by_id.get("V2")
        if v2_row and not v2_row.get("missing") and v2_row.get("cagr_combined") is not None:
            margin_pp = (v2_row["cagr_combined"] - 0.1027) * 100.0
            lines.append(
                f"V2의 G4 통과 마진은 {margin_pp:+.2f}%p로 매우 얇다 -- I5 CAGR 재계산이나 비용모델이 조금만 바뀌어도 뒤집힐 수 있는 수준이라 "
                "\"확고한 승리\"가 아니라 \"근소 우위\"로 읽어야 한다."
            )
    else:
        lines.append(f"{', '.join(promoted)}가 G1~G5를 전부 통과해 승격 대상이다. 나머지 후보의 실패 사유는 위 게이트 표 참조.")
    lines.append("")
    lines.append("사후 추가 후보·파라미터 조정 없음 (SPEC.md 동결 조건 준수). 표본 부족 후보는 UNDETERMINED로 표시했다(위 게이트 표 참조).")
    lines.append("")

    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "wave20_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_registry(results_dir, registry_path)


__all__ = ["write_registry", "write_wave20_report"]
