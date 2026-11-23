# Wave-16 markdown report + registry writer. Pure formatting over already-computed
# results/E{0..4}.json (run_wave16.py --stage run/gates) plus cache/lending_snapshot.json
# (--stage fetch) and a read-only peek at research/wave13_liquidity/results/L4.json (to empirically
# show E0's reproduction claim, never written to).
#
# SPEC.md's non-negotiable labeling requirement is enforced structurally, not just in the title:
# every section that touches a lending-inclusive ("combined") number restates "단면 근거, 시계열
# 미검증" or points back to the header note -- see _header_and_scope_note.

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Final

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

import pandas as pd  # noqa: PANDAS_OK

from research.wave10_carry100.engine import Wave10Result
from research.wave16_duallayer import gates16
from research.wave16_duallayer.configs16 import CANDIDATES, CANDIDATE_IDS

REQUIRED_LABEL: Final = "단면 근거, 시계열 미검증"  # SPEC.md 판정: 이 라벨 없이 "구조 유효"를 쓰지 않는다


def _fmt_pct(value: float | None, digits: int = 2) -> str:
    return "N/A" if value is None else f"{value * 100.0:.{digits}f}%"


def _fmt_usd(value: float | None, digits: int = 2) -> str:
    return "N/A" if value is None else f"${value:,.{digits}f}"


def _load(results_dir: Path, candidate_id: str) -> dict[str, Any]:
    path = results_dir / f"{candidate_id}.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _load_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _series_from_payload(records: list[dict[str, Any]]) -> pd.Series:
    if not records:
        return pd.Series([], index=pd.DatetimeIndex([], tz="UTC"), dtype=float)
    idx = pd.DatetimeIndex([pd.Timestamp(item["timestamp"]) for item in records])
    values = [float(item["value"]) for item in records]
    return pd.Series(values, index=idx, dtype=float).sort_index()


def _minimal_result(equity: pd.Series) -> Wave10Result:
    """regime_breakdown()/gates16 only ever read `.equity` off what they're given in this
    report's own cross-candidate comparisons -- a fully reconstructed Wave10Result (positions/
    turnover/trade_returns) isn't needed here (run_wave16.py's own gates stage already did that
    reconstruction once, and stored ITS output in regime_breakdown_combined -- this local
    re-derivation exists only so evaluate_structure_validity can compare all 5 candidates
    side-by-side from one place without re-plumbing every field through five JSON files)."""
    return Wave10Result(equity=equity, positions=pd.Series(dtype=float), turnover=pd.Series(dtype=float), trade_returns=pd.Series(dtype=float), max_concurrent_positions=0, symbols_used=())


def _annualize(total_return: float, days: float) -> float:
    growth = 1.0 + total_return
    if growth <= 0.0:
        return -1.0
    return float(growth ** (365.0 / max(days, 1.0)) - 1.0)


def _full_period_stats(equity: pd.Series) -> dict[str, Any] | None:
    if len(equity) < 2:
        return None
    start_value = float(equity.iloc[0])
    end_value = float(equity.iloc[-1])
    days = max((pd.Timestamp(equity.index[-1]) - pd.Timestamp(equity.index[0])).total_seconds() / 86_400.0, 1.0)
    total_return = end_value / start_value - 1.0
    return {"start_usdt": start_value, "end_usdt": end_value, "days": days, "total_return": total_return, "annualized_return": _annualize(total_return, days)}


# ---------------------------------------------------------------------------
# Header / methodology
# ---------------------------------------------------------------------------


def _header_and_scope_note(lending_snapshot: dict[str, Any] | None) -> list[str]:
    lines = [
        f"# Wave-16 리포트 -- 이중수익 캐리: 펀딩 + 현물 대여이자 ({REQUIRED_LABEL})",
        "",
        "**이 리포트의 모든 '결합(펀딩+대여이자)' 수치는 현재 시점 대여이자 스냅샷 1회를 "
        "과거 펀딩 시계열에 상수로 얹은 하한 추정치다. '검증된 수익률'이 아니다 -- 검증이라는 "
        "단어는 이 리포트 어디에도 그 의미로 쓰이지 않는다.**",
        "",
        "## 방법론 한계 (필독 -- 아래 모든 수치에 선행하는 전제, SPEC.md 치명적 한계 1-3)",
        "",
        "1. **대여이자 과거 시계열 없음 (시계열 백테스트 불가)**: OKX `lending-rate-summary`는 "
        "현재 스냅샷만 공개한다. 이 wave는 \"과거에 얼마 벌었나\"가 아니라 \"현재 단면에서 "
        "구조가 성립하나\"만 검정한다. 아래 '하한 추정치' 절의 모든 결합 수치는 **현재 "
        "대여이자를 상수로 고정**해 과거 펀딩 시계열에 얹은 것이다 -- 대여이자가 실제로 시간에 "
        "따라 어떻게 움직였는지는 전혀 반영하지 않는다(낙관도 비관도 아니고, 그냥 모른다).",
        "2. **대여 실행 가능성 미검증**: 헤지 중인 현물을 대여하면 청산 시 즉시 회수되는지, "
        "한도·계층이 있는지, avgRate가 대여자 실수취분인지 전부 미확인 -- '미해결 운영 리스크' "
        "절에 각 항목을 명시한다.",
        "3. **거래소 분리**: 대여(OKX)와 실행(Bitget)이 분리되면 wave-14 M6/M7에서 이미 확인한 "
        "거래소간 신용리스크 구조가 재발한다(자본이 두 거래소에 나뉘어 예치됨). Bitget 자체 "
        "스팟자산 대여 상품 공개 확인 결과는 아래 데이터 스냅샷 절 참조 -- 미해결이면 이 구조는 "
        "거래소간 노출로 등급 하향 대상이다.",
        "",
        "## 게이트 적용 범위 (SPEC.md 방법 4 -- 반드시 구분해서 읽을 것)",
        "",
        "- **S1(구조)/S2(MC)/S3(블록셔플MDD)/S4(실행가능)/S5(x3스트레스)는 오직 "
        "funding-only 성분(그 후보의 트레이드 선택은 그대로 두고, 실현 PnL에서 대여이자를 뺀 "
        "series)에만 적용된다.** 대여이자가 들어간 '결합' 수치는 어떤 게이트도 통과한 적이 "
        "없다 -- 상수를 리샘플링해봐야 그 상수의 불확실성은 조금도 검증되지 않기 때문이다.",
        "- **판정('구조 유효')은 게이트 PASS/FAIL이 아니라 SPEC.md의 별도 규칙"
        "(E2 vs E0 vs E3 vs E4 연환산 비교)이다.** 아래 '판정' 절 참조.",
        "- **E3의 '50% 할인'은 랭킹과 실현수익 양쪽 모두에 적용했다** (SPEC.md 원문 \"대여이자를 "
        "50% 할인 적용\"을 대여이자 자체의 절반 축소로 해석 -- 랭킹만 또는 수익만 할인하는 대안 "
        "해석도 가능하나, 이 해석이 \"대여이자가 실제로는 광고값의 절반\"이라는 스트레스 시나리오"
        "의 취지에 가장 부합한다고 판단했다).",
        "- **현재 단면 표(아래)는 순간 펀딩률의 연환산값을 쓴다**, 백테스트가 실제로 쓰는 7일 "
        "실현평균 신호가 아니다(라이브 롤링윈도우를 새로 구축하지 않았으므로) -- 근사치이며, "
        "역사적 백테스트 자체에는 이 근사가 전혀 섞이지 않는다(백테스트는 항상 캐시의 진짜 7일"
        "실현평균 신호를 쓴다).",
        "",
    ]
    if lending_snapshot is not None:
        lines.append(
            f"- **데이터 시차**: 백테스트가 쓰는 캐시(wave12/13 동결 스냅샷)는 2026-07-14에 "
            f"끝난다. 이 리포트의 라이브 스냅샷(OKX/Bitget)은 {lending_snapshot.get('collected_at_utc', 'N/A')}"
            f"에 수집했다 -- 그 사이 실제 시장은 계속 움직였다. '현재 단면' 절은 라이브 스냅샷을, "
            f"'하한 추정치' 절은 캐시 백테스트 + 그 라이브 스냅샷의 대여이자를 쓴다."
        )
        lines.append("")
    return lines


def _candidate_definitions_section() -> list[str]:
    lines = [
        "## 후보 5개 정의 (SPEC.md 동결, 사후 추가 금지)",
        "",
        "공통: $100 자본/$90 활성/$45 레그, 델타중립 1x, wave-13 실측비용, top200 유니버스(L4 승계), "
        "진입 15%APR/청산 7.5%.",
        "",
        "| ID | 랭킹 대여이자 반영 | 실현PnL 대여이자 반영 | 정의 |",
        "|---|---|---|---|",
    ]
    for candidate in CANDIDATES:
        lines.append(
            f"| {candidate.candidate_id} | x{candidate.ranking_lending_discount:.1f} | "
            f"x{candidate.pnl_lending_discount:.1f} | {candidate.note} |"
        )
    lines.append("")
    return lines


# ---------------------------------------------------------------------------
# E0/L4 reproduction check (empirical, read-only peek at wave13's own results/L4.json).
# ---------------------------------------------------------------------------


def _l4_reproduction_section(payloads: dict[str, dict[str, Any]], wave13_results_dir: Path) -> list[str]:
    e0 = payloads.get("E0")
    l4_path = wave13_results_dir / "L4.json"
    l4_payload = _load_optional_json(l4_path)
    lines = ["## E0 재현 검증 (wave13 L4 대비, read-only 비교 -- wave13 파일은 손대지 않음)", ""]
    if e0 is None:
        lines.append("- E0 결과 없음 (run 스테이지 미실행).")
        lines.append("")
        return lines
    e0_high = e0.get("regime_breakdown_combined", {}).get("high_funding_mean_annualized_return")
    e0_final = e0["combined_equity"][-1]["value"] if e0.get("combined_equity") else None
    lines.append(f"- E0 고펀딩기 연환산(결합, 대여 0이므로 순수 펀딩): {_fmt_pct(e0_high)}")
    lines.append(f"- E0 최종 활성자본: {_fmt_usd(e0_final)}")
    if l4_payload is not None:
        l4_high = l4_payload.get("regime_breakdown", {}).get("high_funding_mean_annualized_return")
        l4_final = l4_payload["equity"][-1]["value"] if l4_payload.get("equity") else None
        diff = (e0_high - l4_high) if (e0_high is not None and l4_high is not None) else None
        lines.append(f"- wave13 L4(원본) 고펀딩기 연환산: {_fmt_pct(l4_high)} (참고용, wave13/results/L4.json read-only)")
        lines.append(f"- wave13 L4(원본) 최종 활성자본: {_fmt_usd(l4_final)}")
        lines.append(f"- 차이: {_fmt_pct(diff, digits=6) if diff is not None else 'N/A'} (0%에 가까울수록 '동일 단면·동일 비용 재산출' 주장이 실제로 성립)")
    else:
        lines.append("- wave13/results/L4.json을 찾지 못해 대조 불가 (wave13 run 스테이지가 먼저 실행되어 있어야 함).")
    lines.append("")
    return lines


# ---------------------------------------------------------------------------
# Data snapshot (OKX lending + Bitget funding/loan).
# ---------------------------------------------------------------------------


def _data_snapshot_section(lending_snapshot: dict[str, Any] | None) -> list[str]:
    lines = ["## 데이터 스냅샷 (OKX 대여이자 + Bitget 현재펀딩/대출, 라이브 1회 수집)", ""]
    if lending_snapshot is None:
        lines.append("- lending_snapshot 없음 (fetch 스테이지 미실행).")
        lines.append("")
        return lines
    okx = lending_snapshot["okx_lending"]
    bitget_funding = lending_snapshot["bitget_current_funding"]
    bitget_loan = lending_snapshot["bitget_loan_coininfos"]
    savings_probe = lending_snapshot["bitget_savings_probe"]
    universe = lending_snapshot["universe_summary"]

    def _rate_for(ccy: str) -> float | None:
        row = next((item for item in okx["rows"] if item["ccy"] == ccy), None)
        return row["avg_rate"] if row else None

    highlight_ccys = ["BTC", "ETH", "SOL", "THETA", "GALA", "1INCH"]
    highlight_line = ", ".join(f"{ccy} {_fmt_pct(_rate_for(ccy))}" for ccy in highlight_ccys)

    lines.extend(
        [
            f"- 수집 시각(UTC): {lending_snapshot.get('collected_at_utc', 'N/A')}",
            f"- OKX lending-rate-summary: {okx['raw_count']}종 수신, {okx['kept_count']}종 채택 "
            f"({len(okx['excluded_outliers'])}종 이상치 제외: {', '.join(row['ccy'] for row in okx['excluded_outliers']) or '없음'})",
            f"- OKX 채택분 중앙값(연): {_fmt_pct(okx['median_avg_rate_excl_outliers'])}",
            f"- 주요 코인: {highlight_line}",
            f"- **OKX avgRate 대여자수취분 여부**: 미확인(UNCONFIRMED) -- {okx['lender_side_uncertainty_note']}",
            "",
            f"- Bitget current-fund-rate: {bitget_funding['raw_count']}개 계약 수신, L4 유니버스 "
            f"매칭 {bitget_funding['l4_universe_matched_count']}/{universe['n_symbols']}",
            f"- L4 유니버스 현재 펀딩 APR 중앙값(순간값 연환산, 7일 실현평균 아님): {_fmt_pct(bitget_funding['l4_universe_median_apr'])}",
            "",
            f"- Bitget 자체 대출북(차입자 지불금리, lender 수취 확인용 아님): "
            + ", ".join(f"{row['coin']} 7D {_fmt_pct(row['rate_7d'])}" for row in bitget_loan["rows"][:6]),
            "",
            f"- **Bitget 자체 스팟자산 대여(savings) 공개 확인**: {savings_probe['conclusion']}",
            "",
            f"- L4 유니버스 {universe['n_symbols']}종 중 OKX 대여이자 데이터 존재: "
            f"{universe['n_with_lending']}종 ({universe['n_with_lending_pct']:.1f}%)",
            "",
        ]
    )
    return lines


# ---------------------------------------------------------------------------
# Current cross-sectional pick (method a).
# ---------------------------------------------------------------------------


def _current_snapshot_section(payloads: dict[str, dict[str, Any]]) -> list[str]:
    lines = [
        "## 현재 단면 포트폴리오 (오늘, SPEC.md 방법 2 -- 백테스트 아님, hysteresis 상태 미반영)",
        "",
        "| 후보 | 랭킹방식 | 임계클리어 종목수 | 오늘의 선택 | 펀딩APR(순간) | 대여APR | 합산점수 |",
        "|---|---|---|---|---|---|---|",
    ]
    for candidate_id in CANDIDATE_IDS:
        payload = payloads.get(candidate_id)
        if payload is None:
            lines.append(f"| {candidate_id} | - | - | 결과없음 | - | - | - |")
            continue
        pick = payload.get("current_snapshot_pick", {})
        ranking_mode = "펀딩만" if payload["ranking_lending_discount"] == 0.0 else f"펀딩+대여x{payload['ranking_lending_discount']:.1f}"
        top = pick.get("top_pick")
        if top is None:
            selection = "대기(무포지션 -- 임계 미클리어)"
            funding_cell = lending_cell = score_cell = "-"
        else:
            selection = top["symbol"]
            funding_cell = _fmt_pct(top["funding_apr_current"])
            lending_cell = _fmt_pct(top["lending_apr"]) if top["lending_available"] else "데이터없음(0 가정)"
            score_cell = _fmt_pct(top["ranking_score"])
        lines.append(
            f"| {candidate_id} | {ranking_mode} | {pick.get('n_clearing_threshold', 0)}/{pick.get('universe_n', 0)} | "
            f"{selection} | {funding_cell} | {lending_cell} | {score_cell} |"
        )
    lines.append("")
    return lines


# ---------------------------------------------------------------------------
# Historical lower-bound estimate (method b).
# ---------------------------------------------------------------------------


def _historical_lower_bound_section(payloads: dict[str, dict[str, Any]]) -> list[str]:
    lines = [
        "## 하한 추정치 (과거 펀딩 시계열 + 현재 대여이자 상수, SPEC.md 방법 3 -- 낙관/비관 아님)",
        "",
        "| 후보 | 고펀딩기 연환산(결합) | 전체구간 연환산(결합) | 현재(저펀딩)구간 연환산(결합) | 대여이자 기여분(고펀딩기, %p) | funding-only 고펀딩기 연환산(게이트 대상) |",
        "|---|---|---|---|---|---|",
    ]
    for candidate_id in CANDIDATE_IDS:
        payload = payloads.get(candidate_id)
        if payload is None:
            lines.append(f"| {candidate_id} | 결과없음 | | | | |")
            continue
        combined_regime = payload.get("regime_breakdown_combined", {})
        funding_only_regime = payload.get("regime_breakdown_funding_only", {})
        combined_high = combined_regime.get("high_funding_mean_annualized_return")
        funding_only_high = funding_only_regime.get("high_funding_mean_annualized_return")
        contribution = (combined_high - funding_only_high) * 100.0 if (combined_high is not None and funding_only_high is not None) else None
        combined_equity = _series_from_payload(payload.get("combined_equity", []))
        full_period = _full_period_stats(combined_equity)
        current_regime = combined_regime.get("current_low_funding") or {}
        lines.append(
            f"| {candidate_id} | {_fmt_pct(combined_high)} | {_fmt_pct(full_period['annualized_return']) if full_period else 'N/A'} | "
            f"{_fmt_pct(current_regime.get('annualized_return'))} | "
            f"{f'{contribution:+.2f}%p' if contribution is not None else 'N/A'} | {_fmt_pct(funding_only_high)} |"
        )
    lines.append("")
    lines.append(
        f"참고: funding-only 열은 SPEC.md 방법 4의 게이트 대상 series다 -- E0/E1은 같은 트레이드 "
        f"선택(값도 동일), E2/E4는 같은 트레이드 선택(값도 동일, E4가 정의상 E2의 funding-only "
        f"companion), E3만 고유한 자기 자신의 funding-only companion을 갖는다."
    )
    lines.append("")
    return lines


# ---------------------------------------------------------------------------
# Gates (funding-only only).
# ---------------------------------------------------------------------------


def _gates_section(payloads: dict[str, dict[str, Any]]) -> list[str]:
    lines = [
        "## 게이트 -- funding-only 성분만 (S1-S5, SPEC.md 방법 4)",
        "",
        "| 후보 | S1구조 | S2 MC(p05/ruin) | S3 블록MDD | S4 실행가능 | S5 스트레스 | Overall |",
        "|---|---|---|---|---|---|---|",
    ]
    for candidate_id in CANDIDATE_IDS:
        payload = payloads.get(candidate_id)
        gates = payload.get("gates_funding_only") if payload else None
        if not gates:
            lines.append(f"| {candidate_id} | - | - | - | - | - | 게이트 미실행 |")
            continue
        s2 = gates["gate_s2"]
        s3 = gates["gate_s3"]
        lines.append(
            f"| {candidate_id} | {gates['gate_s1']['status']} | "
            f"{s2['status']}(p05={_fmt_usd(s2.get('p05'))}/ruin={_fmt_pct(s2.get('ruin_probability'))}) | "
            f"{s3['status']}(p95={_fmt_pct(s3.get('mdd_p95'))}) | {gates['gate_s4']['status']} | "
            f"{gates['gate_s5']['status']} | **{gates['overall']}** |"
        )
    lines.append("")
    return lines


# ---------------------------------------------------------------------------
# Verdict -- SPEC.md's own structure-validity rule.
# ---------------------------------------------------------------------------


def _verdict_section(payloads: dict[str, dict[str, Any]]) -> tuple[str, list[str]]:
    combined_results: dict[str, Wave10Result] = {}
    for candidate_id in CANDIDATE_IDS:
        payload = payloads.get(candidate_id)
        if payload is None:
            continue
        combined_results[candidate_id] = _minimal_result(_series_from_payload(payload.get("combined_equity", [])))

    lines = [f"## 판정 -- '구조 유효'? ({REQUIRED_LABEL})", ""]
    if set(combined_results) != set(CANDIDATE_IDS):
        missing = set(CANDIDATE_IDS) - set(combined_results)
        lines.append(f"- 판정 불가: {sorted(missing)} 결과 없음 (run 스테이지 미완료).")
        lines.append("")
        return "PENDING", lines

    verdict = gates16.evaluate_structure_validity(combined_results)
    lines.extend(
        [
            f"- E0 (기준선, 대여 없음) 고펀딩기 연환산: {_fmt_pct(verdict.e0_high_funding_annualized)}",
            f"- E1 (펀딩랭킹+실측대여수익) 고펀딩기 연환산: {_fmt_pct(verdict.e1_high_funding_annualized)}",
            f"- E2 (합산랭킹, 핵심가설) 고펀딩기 연환산: {_fmt_pct(verdict.e2_high_funding_annualized)} "
            f"-> E0 대비 개선: {'YES' if verdict.e2_beats_e0 else 'NO'}",
            f"- E3 (합산랭킹, 대여이자 50%할인) 고펀딩기 연환산: {_fmt_pct(verdict.e3_high_funding_annualized)} "
            f"-> E0 대비 개선 유지: {'YES' if verdict.e3_beats_e0 else 'NO'}",
            f"- E4 (합산랭킹 그대로, 대여이자 0% 실현 -- 대여실패 시나리오) 고펀딩기 연환산: "
            f"{_fmt_pct(verdict.e4_high_funding_annualized)} -> E0 이상 유지: {'YES' if verdict.e4_at_least_e0 else 'NO'}",
            "",
            f"**판정: {'구조 유효 (' + REQUIRED_LABEL + ')' if verdict.structure_valid else '기각'}**",
            "",
        ]
    )
    if not verdict.structure_valid:
        lines.append("기각 사유: " + "; ".join(verdict.reasons))
        lines.append("")
        if not verdict.e4_at_least_e0:
            lines.append(
                "**E4 < E0**: 대여이자를 믿고 랭킹에 반영했지만 실제로 그 수익이 실현되지 않는 "
                "시나리오에서, 순수 펀딩 랭킹(E0)보다 더 나쁜 트레이드를 골랐다는 뜻이다 -- "
                "합산 랭킹 자체가 (대여이자 실현 여부와 무관하게) 트레이드 선택을 악화시켰다는 "
                "증거이므로, SPEC.md 규칙대로 이 가설은 기각한다."
            )
            lines.append("")
    return ("PASS" if verdict.structure_valid else "FAIL"), lines


# ---------------------------------------------------------------------------
# Unresolved operational risks.
# ---------------------------------------------------------------------------


def _unresolved_risk_section(lending_snapshot: dict[str, Any] | None) -> list[str]:
    exposure = gates16.exchange_separation_remaining_fraction()
    lines = [
        "## 미해결 운영 리스크 (실자금 적용 전 확인 필요 -- 이 wave가 해소한 항목 아님)",
        "",
        "- **대여 회수(recall) 지연**: 헤지 청산 신호가 뜬 순간 대여 중인 현물을 즉시 회수해 "
        "매도할 수 있는지 미확인. 회수 지연이 있으면 청산 시점에 현물 없이 퍼프 숏만 남는 "
        "방향노출 구간이 생길 수 있다. 실자금 적용 전 확인 필요.",
        "- **한도·계층**: OKX lending-rate-summary의 avgAmt/avgAmtUsd 필드는 공개 데이터에서 "
        "비어 있어 개인 대여한도·전체 풀 규모 모두 확인 불가. 실자금 적용 전 확인 필요.",
        f"- **OKX avgRate 대여자수취분 여부**: 미확인(UNCONFIRMED) -- 데이터 스냅샷 절 참조. "
        f"E3(50% 할인)는 이 불확실성에 대한 대비책이지 확인이 아니다. 실자금 적용 전 확인 필요.",
        f"- **거래소간 분리 노출 (wave-14 M6/M7 재발)**: 대여(OKX)와 실행(Bitget)이 분리된 "
        f"구조에서 한 거래소 전액 손실(출금중단·파산) 시나리오는 시장 경로와 무관하게 잔존자본 "
        f"{exposure * 100.0:.2f}%로 고정된다(예비금 + 생존 레그, 확률 추정 아님 -- 구조적 노출). "
        f"실자금 적용 전 확인 필요.",
        "- **Bitget 자체 스팟자산 대여 공개 확인 불가**: "
        + (lending_snapshot["bitget_savings_probe"]["conclusion"] if lending_snapshot else "N/A (fetch 미실행)")
        + " -- 거래소 분리를 회피할 단일거래소 대안 경로가 있는지 이 wave에서는 확인하지 못했다. "
        "실자금 적용 전 확인 필요.",
        "- **라이브-캐시 시차**: 백테스트 캐시 종료일(2026-07-14)과 라이브 스냅샷 수집 시각 사이 "
        "실제 시장이 이동한다 -- '현재 단면' 절 수치는 그 시차만큼 이미 낡았을 수 있다. 실행 "
        "직전 재수집 필요.",
        "",
    ]
    return lines


# ---------------------------------------------------------------------------
# Multiple-testing note + write.
# ---------------------------------------------------------------------------


def _dsr_note() -> str:
    return (
        f"다중검정 보정: 누적 시행 {gates16.DSR_CUMULATIVE_TRIALS}회(wave15까지 91회 + 이 "
        "wave의 5개 후보 E0-E4) 기준 DSR(Deflated Sharpe Ratio) 참고치를 각 결과 JSON의 "
        "`reference_metrics.dsr_funding_only`에 기록 (샤프는 참고 지표이며 승격 판정에는 "
        "사용하지 않음, wave10-15와 동일 원칙)."
    )


def write_wave16_report(results_dir: Path, report_dir: Path, registry_path: Path, cache_dir: Path) -> None:
    payloads: dict[str, dict[str, Any]] = {}
    for candidate_id in CANDIDATE_IDS:
        try:
            payloads[candidate_id] = _load(results_dir, candidate_id)
        except FileNotFoundError:
            continue
    lending_snapshot = _load_optional_json(cache_dir / "lending_snapshot.json")
    wave13_results_dir = results_dir.parent.parent / "wave13_liquidity" / "results"

    verdict_state, verdict_lines = _verdict_section(payloads)

    lines: list[str] = [
        *_header_and_scope_note(lending_snapshot),
        *_candidate_definitions_section(),
        *_l4_reproduction_section(payloads, wave13_results_dir),
        *_data_snapshot_section(lending_snapshot),
        *_current_snapshot_section(payloads),
        *_historical_lower_bound_section(payloads),
        *_gates_section(payloads),
        *verdict_lines,
        *_unresolved_risk_section(lending_snapshot),
        "## 다중검정",
        "",
        _dsr_note(),
        "",
    ]
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "wave16_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    registry_lines = [
        "# Wave-16 registry",
        "",
        f"모든 항목의 승격 여부는 '{REQUIRED_LABEL}' 라벨 없이는 성립하지 않는다 -- report/wave16_report.md 참조.",
        "",
        "| Candidate | Family | State | funding-only 게이트 | 고펀딩기 연환산(결합) | E0 대비 | 분류 |",
        "|---|---|---|---|---|---|---|",
    ]
    for candidate_id in CANDIDATE_IDS:
        payload = payloads.get(candidate_id)
        if payload is None:
            registry_lines.append(f"| {candidate_id} | wave16_duallayer | PENDING | - | - | - | run 미실행 |")
            continue
        gates = payload.get("gates_funding_only")
        if not gates:
            registry_lines.append(f"| {candidate_id} | wave16_duallayer | EVALUATED | - | - | - | gates 미실행 |")
            continue
        combined_high = payload.get("regime_breakdown_combined", {}).get("high_funding_mean_annualized_return")
        e0_high = payloads.get("E0", {}).get("regime_breakdown_combined", {}).get("high_funding_mean_annualized_return")
        vs_e0 = "N/A"
        if combined_high is not None and e0_high is not None:
            vs_e0 = f"{(combined_high - e0_high) * 100.0:+.2f}%p"
        if candidate_id == "E0":
            classification = "기준선"
        elif candidate_id in {"E3", "E4"}:
            classification = "스트레스 시나리오"
        else:
            classification = "핵심가설" if candidate_id == "E2" else "참고"
        registry_lines.append(
            f"| {candidate_id} | wave16_duallayer | EVALUATED | {gates['overall']} | {_fmt_pct(combined_high)} | {vs_e0} | {classification} |"
        )
    verdict_label = {
        "PASS": f"구조 유효 ({REQUIRED_LABEL})",
        "FAIL": "기각 (E4<E0 또는 E2/E3가 E0를 넘지 못함 -- SPEC.md 판정 규칙)",
        "PENDING": "미결정 (일부 후보 결과 없음)",
    }[verdict_state]
    registry_lines.append("")
    registry_lines.append(f"**최종 판정**: {verdict_state} -- {verdict_label} (report/wave16_report.md 참조).")
    registry_path.write_text("\n".join(registry_lines) + "\n", encoding="utf-8")


__all__ = ["write_wave16_report"]
