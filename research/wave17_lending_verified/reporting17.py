# Wave-17 markdown report + registry writer. Pure formatting over already-computed
# results/F{0..3,_min}.json (recompute17.run_and_save) and cache/lending_realized.json
# (fetch17.collect_lending_realized), plus a read-only peek at
# research/wave16_duallayer/results/E0.json (to empirically show F0's reproduction claim --
# same convention wave16's own reporting16.py used against wave13's L4.json).

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Final

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

from research.wave17_lending_verified import gates17, volatility17
from research.wave17_lending_verified.recompute17 import CANDIDATE_IDS17, F_MIN

REQUIRED_LABEL: Final = "단면 근거, 시계열 미검증"  # wave16과 동일 라벨 유지 (SPEC.md 판정 -- 이 wave가 없앤 것이 아님)
ALL_RESULT_IDS: Final = (*CANDIDATE_IDS17, F_MIN.candidate_id)


def _fmt_pct(value: float | None, digits: int = 2) -> str:
    return "N/A" if value is None else f"{value * 100.0:.{digits}f}%"


def _fmt_usd(value: float | None, digits: int = 2) -> str:
    return "N/A" if value is None else f"${value:,.{digits}f}"


def _fmt_ratio(value: float | None, digits: int = 3) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


def _load(results_dir: Path, candidate_id: str) -> dict[str, Any]:
    path = results_dir / f"{candidate_id}.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _load_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _minimal_result(equity_records: list[dict[str, Any]]):
    import pandas as pd  # noqa: PANDAS_OK

    from research.wave10_carry100.engine import Wave10Result

    if not equity_records:
        idx = pd.DatetimeIndex([], tz="UTC")
        equity = pd.Series([], index=idx, dtype=float)
    else:
        idx = pd.DatetimeIndex([pd.Timestamp(item["timestamp"]) for item in equity_records])
        equity = pd.Series([float(item["value"]) for item in equity_records], index=idx, dtype=float).sort_index()
    return Wave10Result(equity=equity, positions=pd.Series(dtype=float), turnover=pd.Series(dtype=float), trade_returns=pd.Series(dtype=float), max_concurrent_positions=0, symbols_used=())


# ---------------------------------------------------------------------------
# Header / methodology.
# ---------------------------------------------------------------------------


def _header_and_scope_note(lending_realized: dict[str, Any] | None) -> list[str]:
    lines = [
        f"# Wave-17 리포트 -- 대여이자 실수취(lendingRate) 검증 및 재산출 ({REQUIRED_LABEL})",
        "",
        "**이 리포트는 wave16의 대여수익 가정(OKX `lending-rate-summary`의 `avgRate` = 대여자 수취분)이 "
        "틀렸다는 실측에서 출발한다. avgRate는 차입자 지불 쪽에 가까운 값이고, 대여자 쪽 실현값은 "
        "OKX `lending-rate-history`의 `lendingRate` 필드다. 이 리포트의 모든 F1/F2/F_min 수치는 "
        "`lendingRate`의 4일치(시간당 100건) 관측을 상수로 고정해 과거 펀딩 시계열에 얹은 하한 "
        "추정치다 -- wave16과 마찬가지로 '검증된 수익률'이 아니다.**",
        "",
        "## 방법론 한계 (필독 -- wave16 치명적 한계 1-3과 동일 축, 이 wave가 해소한 것 아님)",
        "",
        "1. **시계열 깊이는 여전히 4일뿐**: `lending-rate-history`가 `avgRate`보다 나은 필드"
        "(`lendingRate`)를 준다는 것이지, 더 긴 과거를 주는 게 아니다. limit=100(시간당 1건) 이상은 "
        "이 엔드포인트에서 얻을 수 없었다 -- 과거(4일 이전) 백테스트는 이 wave 이후에도 여전히 "
        "불가능하다.",
        "2. **4일 중앙값을 연환산 상수로 쓰는 것은 강한 가정이다**: 대여이자가 실제로 연중 어떻게 "
        "움직였는지 전혀 모른다(고펀딩기엔 더 높았을 수도, 더 낮았을 수도 있다 -- 확인 불가). "
        "'변동성' 절의 표준편차·범위는 이 4일 구간 *내부의* 변동만 보여준다 -- 4일 밖의 변동은 여전히 "
        "미지수다.",
        "3. **`lendingRate`가 대여자 실수취(net-of-fee) 값이라는 것도 필드명 기반 추론이다**: OKX "
        "공개 도움말 문서로 교차확인을 시도했다(아래 '운영 리스크' 절 참조) -- 정성적으로는 "
        "일관되는 정황을 찾았지만, `lendingRate` 필드 자체를 '이 값이 곧 net-of-fee 실수취액'이라고 "
        "명시하는 API 문서 원문은 확보하지 못했다. 실계좌로 확인된 사실이 아니다.",
        "4. **거래소 분리·대여 회수 지연 등 wave16의 미해결 운영 리스크는 이 wave에서 그대로 "
        "이어진다** -- 아래 '운영 리스크' 절에서 이번에 새로 확인한 항목(flexible/fixed-term, "
        "수수료 구조)과 여전히 미확인인 항목을 구분해서 표기한다.",
        "",
    ]
    if lending_realized is not None:
        lines.append(
            f"- **데이터 시차**: 백테스트가 쓰는 캐시(wave12/13 동결 스냅샷)는 2026-07-14에 끝난다. "
            f"이 리포트의 lendingRate 라이브 수집은 {lending_realized.get('collected_at_utc', 'N/A')}에 "
            f"실행했다 -- 그 사이 실제 시장은 계속 움직였다."
        )
        lines.append("")
    return lines


def _candidate_definitions_section() -> list[str]:
    from research.wave17_lending_verified.recompute17 import CANDIDATES17

    lines = [
        "## 후보 정의 (SPEC.md 동결, 사후 추가 금지)",
        "",
        "공통: $100 자본/$90 활성/$45 레그, 델타중립 1x, wave-13 실측비용, top200 유니버스(L4/E0 승계), "
        "진입 15%APR/청산 7.5%, **랭킹은 F0-F3 전부 펀딩APR만 사용**(대여이자는 실현PnL에만 가산 -- "
        "wave16 E1과 동일 전제, E2/E3/E4류의 '랭킹에 대여이자 합산'은 이 wave의 대상이 아니다).",
        "",
        "| ID | 대여수익 배율 | 대여이자 소스 | 정의 |",
        "|---|---|---|---|",
    ]
    for candidate in CANDIDATES17:
        lines.append(f"| {candidate.candidate_id} | x{candidate.pnl_lending_discount:.2f} | {candidate.lending_apr_source} | {candidate.note} |")
    lines.append(f"| {F_MIN.candidate_id} (참고, 게이트 미적용) | x{F_MIN.pnl_lending_discount:.2f} | {F_MIN.lending_apr_source} | {F_MIN.note} |")
    lines.append("")
    return lines


def _f0_reproduction_section(payloads: dict[str, dict[str, Any]], wave16_results_dir: Path) -> list[str]:
    f0 = payloads.get("F0")
    e0_path = wave16_results_dir / "E0.json"
    e0_payload = _load_optional_json(e0_path)
    lines = ["## F0 재현 검증 (wave16 E0 대비, read-only 비교 -- wave16 파일은 손대지 않음)", ""]
    if f0 is None:
        lines.append("- F0 결과 없음 (run 스테이지 미실행).")
        lines.append("")
        return lines
    f0_high = f0.get("regime_breakdown_combined", {}).get("high_funding_mean_annualized_return")
    f0_final = f0["combined_equity"][-1]["value"] if f0.get("combined_equity") else None
    lines.append(f"- F0 고펀딩기 연환산: {_fmt_pct(f0_high)}")
    lines.append(f"- F0 최종 활성자본: {_fmt_usd(f0_final)}")
    if e0_payload is not None:
        e0_high = e0_payload.get("regime_breakdown_combined", {}).get("high_funding_mean_annualized_return")
        e0_final = e0_payload["combined_equity"][-1]["value"] if e0_payload.get("combined_equity") else None
        diff = (f0_high - e0_high) if (f0_high is not None and e0_high is not None) else None
        lines.append(f"- wave16 E0(원본) 고펀딩기 연환산: {_fmt_pct(e0_high)} (참고용, wave16/results/E0.json read-only)")
        lines.append(f"- wave16 E0(원본) 최종 활성자본: {_fmt_usd(e0_final)}")
        lines.append(f"- 차이: {_fmt_pct(diff, digits=8) if diff is not None else 'N/A'} (0%에 가까울수록 'L4/E0 재현' 주장이 실제로 성립)")
    else:
        lines.append("- research/wave16_duallayer/results/E0.json을 찾지 못해 대조 불가.")
    lines.append("")
    lines.append(
        "- F3(대여수익 0% 가정)는 정의상 F0과 동일한 (ranking=0.0, pnl=0.0) 변형이다 -- 아래 '판정' "
        "절의 F3==F0 확인이 곧 이 재현성의 두 번째 증거다."
    )
    lines.append("")
    return lines


# ---------------------------------------------------------------------------
# Data snapshot -- ratio comparisons (three denominators, all reported, difference explained).
# ---------------------------------------------------------------------------


def _data_snapshot_section(lending_realized: dict[str, Any] | None) -> list[str]:
    lines = ["## 데이터 스냅샷 (OKX lending-rate-history 실측, 라이브 1회 수집)", ""]
    if lending_realized is None:
        lines.append("- lending_realized 없음 (fetch 스테이지 미실행).")
        lines.append("")
        return lines
    lines.extend(
        [
            f"- 수집 시각(UTC): {lending_realized.get('collected_at_utc', 'N/A')}",
            f"- 대상 코인: wave16 스냅샷 유니버스(lending_available=True) {lending_realized.get('n_target_ccys')}종 "
            f"-> 히스토리 수집 성공 {lending_realized.get('n_history_available')}종, 실패 {lending_realized.get('n_history_failed')}종",
            (
                f"- 관측 구간: 중앙값 {lending_realized.get('span_days_median'):.2f}일"
                + (f" (최소 {lending_realized.get('span_days_min'):.2f}일)" if lending_realized.get("span_days_min") is not None else "")
                if lending_realized.get("span_days_median") is not None
                else "- 관측 구간: N/A"
            ),
            "",
            "### lendingRate/avgRate 비율 -- 3가지 기준, 값이 서로 다른 이유는 본문 참조",
            "",
            "| 기준 | 중앙값 | 평균 | 설명 |",
            "|---|---|---|---|",
            f"| 이번 세션 재수집 avgRate 대비 | {_fmt_ratio(lending_realized.get('ratio_median_across_universe_vs_fresh_avgrate'))} | "
            f"{_fmt_ratio(lending_realized.get('ratio_mean_across_universe_vs_fresh_avgrate'))} | "
            f"lendingRate 4일 중앙값 / 같은 세션에 재수집한 lending-rate-summary avgRate |",
            f"| wave16이 실제로 쓴 avgRate 대비 | {_fmt_ratio(lending_realized.get('ratio_median_across_universe_vs_wave16_avgrate'))} | "
            f"{_fmt_ratio(lending_realized.get('ratio_mean_across_universe_vs_wave16_avgrate'))} | "
            f"**'wave16 E1이 실제로 몇 % 과대평가했나'에 답하는 지표** -- lendingRate 중앙값 / wave16 cache의 avgRate(2026-07-23 수집분) |",
            f"| 동일 API 호출 내부(rate) 대비 | {_fmt_ratio(lending_realized.get('ratio_median_across_universe_vs_history_rate'))} | - | "
            f"lending-rate-history 응답 안의 rate와 lendingRate를 같은 호출·같은 타임스탬프에서 직접 비교 (교차시점 잡음 없음) |",
            "",
            "세 값이 다른 이유: `lending-rate-summary.avgRate`(시스템 전체 매칭 추정 평균)와 "
            "`lending-rate-history.rate`(그 시각의 시간당 값)는 서로 다른 API·다른 계산창일 수 있다 "
            "-- 완전한 동일 지표가 아니다. 이 리포트는 세 가지를 전부 투명하게 남기고, F1/F2 "
            "계산에는 어느 쪽 '비율'도 곱하지 않는다(아래 참고) -- F1은 코인별 lendingRate 실측값을 "
            "직접 쓴다, 비율은 서술용 진단 지표일 뿐이다.",
            "",
        ]
    )
    return lines


def _highlight_coins_section(lending_realized: dict[str, Any] | None) -> list[str]:
    if lending_realized is None:
        return []
    by_ccy = lending_realized["by_ccy"]
    highlight = ["THETA", "BERA", "BTC", "ETH", "SOL"]
    lines = ["### 주요 코인 스팟체크", "", "| 코인 | avgRate(신선) | lendingRate 중앙값 | 비율(신선avgRate 대비) | 비율(동일호출 rate 대비) |", "|---|---|---|---|---|"]
    for ccy in highlight:
        row = by_ccy.get(ccy)
        if row is None or not row.get("history_available"):
            lines.append(f"| {ccy} | - | - | - | 데이터없음 |")
            continue
        lines.append(
            f"| {ccy} | {_fmt_pct(row.get('avg_rate_fresh'))} | {_fmt_pct(row.get('lending_rate_median'))} | "
            f"{_fmt_ratio(row.get('ratio_lendingrate_over_avgrate_fresh'))} | {_fmt_ratio(row.get('ratio_lendingrate_over_historyrate'))} |"
        )
    lines.append("")
    return lines


# ---------------------------------------------------------------------------
# Volatility table (SPEC.md 방법 3).
# ---------------------------------------------------------------------------


def _volatility_section(lending_realized: dict[str, Any] | None) -> list[str]:
    lines = ["## 변동성 정량화 (4일 관측 구간 내부, SPEC.md 방법 3)", ""]
    if lending_realized is None:
        lines.append("- lending_realized 없음.")
        lines.append("")
        return lines
    rows = volatility17.build_volatility_table(lending_realized)
    summary = volatility17.universe_volatility_summary(rows)
    spike_coins = [r.ccy for r in rows if r.lending_rate_max >= 1.0]  # >=100% within-window max -- see note below
    lines.extend(
        [
            f"- 유니버스 {summary['n_coins']}종, 변동계수(CV=표준편차/평균) 중앙값 {_fmt_ratio(summary.get('cv_median'))}, "
            f"최대 {_fmt_ratio(summary.get('cv_max'))}",
            f"- (최대-최소)/중앙값 비율의 중앙값 {_fmt_ratio(summary.get('range_over_median_median'))}, "
            f"최대 {_fmt_ratio(summary.get('range_over_median_max'))} -- 4일 새 관측값이 중앙값 대비 이만큼 흔들렸다는 뜻",
            "",
        ]
    )
    if spike_coins:
        lines.extend(
            [
                f"- **관측된 순간 스파이크**: {', '.join(spike_coins)} {len(spike_coins)}종은 4일 관측 구간 "
                f"중 단 몇 시간(100건 중 일부) 동안 lendingRate가 중앙값의 5~10배 수준(약 200%대)까지 "
                f"튀었다가 되돌아왔다(예: BERA 중앙값 4.12% vs 관측 최고치 203%). 여러 종목이 비슷한 "
                f"자릿수(약 200~210%)로 동시에 튄 점으로 미루어 개별 코인 고유 이벤트라기보다 특정 "
                f"시간대의 플랫폼 전체 매칭 이상치일 가능성이 있다 -- **원인은 확인하지 못했다(추정)**. "
                f"F1은 중앙값을 쓰므로 이 스파이크에 거의 영향받지 않지만, '최고치를 믿고 크게 베팅'하는 "
                f"방식이었다면 위험했을 것이라는 근거로 남긴다.",
                "",
            ]
        )
    lines.extend(
        [
            "### 가장 불안정한 5종 (CV 기준)",
            "",
            "| 코인 | 중앙값 | 평균 | 표준편차 | 최소 | 최대 | CV |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    for row in rows[:5]:
        lines.append(
            f"| {row.ccy} | {_fmt_pct(row.lending_rate_median)} | {_fmt_pct(row.lending_rate_mean)} | "
            f"{_fmt_pct(row.lending_rate_std)} | {_fmt_pct(row.lending_rate_min)} | {_fmt_pct(row.lending_rate_max)} | {_fmt_ratio(row.lending_rate_cv)} |"
        )
    lines.append("")
    lines.append("### 가장 안정적인 5종 (CV 기준)")
    lines.append("")
    lines.append("| 코인 | 중앙값 | 평균 | 표준편차 | 최소 | 최대 | CV |")
    lines.append("|---|---|---|---|---|---|---|")
    for row in rows[-5:]:
        lines.append(
            f"| {row.ccy} | {_fmt_pct(row.lending_rate_median)} | {_fmt_pct(row.lending_rate_mean)} | "
            f"{_fmt_pct(row.lending_rate_std)} | {_fmt_pct(row.lending_rate_min)} | {_fmt_pct(row.lending_rate_max)} | {_fmt_ratio(row.lending_rate_cv)} |"
        )
    lines.append("")
    return lines


# ---------------------------------------------------------------------------
# Results comparison + verdict.
# ---------------------------------------------------------------------------


def _results_table_section(payloads: dict[str, dict[str, Any]]) -> list[str]:
    lines = [
        "## 결과 비교 (하한 추정치 -- 과거 펀딩 시계열 + 실측 lendingRate 상수, 낙관/비관 아님)",
        "",
        "| 후보 | 고펀딩기 연환산(결합) | 전체구간 연환산(결합) | 현재(저펀딩)구간 연환산 | F0 대비(고펀딩기, %p) |",
        "|---|---|---|---|---|",
    ]
    f0_high = payloads.get("F0", {}).get("regime_breakdown_combined", {}).get("high_funding_mean_annualized_return")
    for candidate_id in ALL_RESULT_IDS:
        payload = payloads.get(candidate_id)
        if payload is None:
            lines.append(f"| {candidate_id} | 결과없음 | | | |")
            continue
        combined_regime = payload.get("regime_breakdown_combined", {})
        combined_high = combined_regime.get("high_funding_mean_annualized_return")
        contribution = (combined_high - f0_high) * 100.0 if (combined_high is not None and f0_high is not None) else None
        combined_equity = payload.get("combined_equity", [])
        full_period = None
        if len(combined_equity) >= 2:
            start_v = combined_equity[0]["value"]
            end_v = combined_equity[-1]["value"]
            import pandas as pd  # noqa: PANDAS_OK

            days = max((pd.Timestamp(combined_equity[-1]["timestamp"]) - pd.Timestamp(combined_equity[0]["timestamp"])).total_seconds() / 86_400.0, 1.0)
            growth = end_v / start_v
            full_period = growth ** (365.0 / days) - 1.0 if growth > 0 else None
        current_regime = combined_regime.get("current_low_funding") or {}
        label = candidate_id + (" (참고)" if candidate_id == F_MIN.candidate_id else "")
        lines.append(
            f"| {label} | {_fmt_pct(combined_high)} | {_fmt_pct(full_period)} | "
            f"{_fmt_pct(current_regime.get('annualized_return'))} | "
            f"{f'{contribution:+.2f}%p' if contribution is not None else 'N/A'} |"
        )
    lines.append("")
    return lines


def _verdict_section(payloads: dict[str, dict[str, Any]]) -> tuple[str, list[str]]:
    combined_results = {}
    for candidate_id in CANDIDATE_IDS17:
        payload = payloads.get(candidate_id)
        if payload is None:
            continue
        combined_results[candidate_id] = _minimal_result(payload.get("combined_equity", []))

    lines = [f"## 판정 -- '실수취 기준 재확인 유효'? ({REQUIRED_LABEL})", ""]
    if set(combined_results) != set(CANDIDATE_IDS17):
        missing = set(CANDIDATE_IDS17) - set(combined_results)
        lines.append(f"- 판정 불가: {sorted(missing)} 결과 없음.")
        lines.append("")
        return "PENDING", lines

    verdict = gates17.evaluate_wave17(combined_results)
    lines.extend(
        [
            f"- F0 (기준선, 대여 없음) 고펀딩기 연환산: {_fmt_pct(verdict.f0_high_funding_annualized)}",
            f"- F1 (실측 lendingRate 중앙값) 고펀딩기 연환산: {_fmt_pct(verdict.f1_high_funding_annualized)} "
            f"-> F0 대비 개선: {'YES' if verdict.f1_beats_f0 else 'NO'}",
            f"- F2 (F1 x 50% 보수 할인) 고펀딩기 연환산: {_fmt_pct(verdict.f2_high_funding_annualized)} "
            f"-> F0 대비 개선 유지: {'YES' if verdict.f2_beats_f0 else 'NO'}",
            f"- F3 (대여수익 0%, 무결성 검증) 고펀딩기 연환산: {_fmt_pct(verdict.f3_high_funding_annualized)} "
            f"-> F0와 동일(허용오차 {gates17.IDENTITY_ABS_TOLERANCE}): {'YES' if verdict.f3_equals_f0 else 'NO'} (diff={verdict.f3_abs_diff})",
            "",
            f"**판정: {'실수취 기준 재확인 유효 (' + REQUIRED_LABEL + ')' if verdict.verdict_valid else '기각/일부 실패'}**",
            "",
        ]
    )
    if not verdict.verdict_valid:
        lines.append("미달 사유: " + "; ".join(verdict.reasons))
        lines.append("")
        lines.append("**수치는 조정하지 않았다 -- 위 결과는 실측값을 그대로 반영한다.**")
        lines.append("")
    return ("PASS" if verdict.verdict_valid else "FAIL"), lines


# ---------------------------------------------------------------------------
# Operational risk (SPEC.md 방법 4 -- research, not code; findings hardcoded here with
# citations, UNCONFIRMED markers where the tools available to this task could not verify).
# ---------------------------------------------------------------------------


def _operational_risk_section() -> list[str]:
    lines = [
        "## 운영 리스크 (OKX 공개 도움말·API 문서 조사, 2026-07-24 -- 코드 아님)",
        "",
        "wave16이 전부 UNCONFIRMED로 남겼던 항목을 OKX 공개 도움말 문서(`www.okx.com/en-us/help/`)로 "
        "재조사했다. 확인된 것과 여전히 안 된 것을 구분한다.",
        "",
        "### CONFIRMED (OKX 공식 도움말 문서 기반)",
        "",
        "- **flexible(즉시 회수) vs fixed-term(락업)**: **flexible로 확인.** OKX 도움말 \"Introduction "
        "to OKX Savings and Its Rules\"(www.okx.com/en-us/help/introduction-to-okx-savings-and-its-rules)"
        "는 이 상품을 언제든 인출·회수 가능하다고 설명한다 -- 락업 기간이 있다는 언급 없음. "
        "wave16 미해결 리스크 '대여 회수 지연' 우려는 상품 설계상 락업은 아니라는 점에서 완화되나, "
        "**청산 신호 발생 즉시 회수~매도 체결까지의 실제 지연시간(초/분 단위)은 API 문서로 확인되지 "
        "않는다 -- 이 세부는 여전히 UNCONFIRMED**.",
        "- **한도**: 개인 예치·회수에 **일반적인 한도 없음**(OKX가 시장 상황에 따라 조정할 권리는 "
        "보유). 계층별(예치금액 구간별) 금리 차등 구조는 문서에서 발견하지 못했다 -- 금리는 시간당 "
        "매칭 메커니즘(아래)으로만 결정되는 것으로 보인다.",
        "- **수수료 구조**: OKX가 발생 이자의 **15%를 서비스 수수료로 가져가고, 대여자는 나머지 "
        "85%**를 받는다고 도움말 문서에 명시돼 있다.",
        "- **금리 결정 메커니즘**: 매시 차입 수요를 확인해 대여 호가를 최저~최고 순으로 정렬, 수요가 "
        "채워질 때까지 매칭한다 -- 매칭된 마지막(최고) 호가가 그 시각 '시장 금리'를 결정한다. 공급이 "
        "수요를 초과하면 개별 대여자의 실제 수취금리는 자신이 제시한 희망금리보다 낮아질 수 있다고 "
        "문서 자체가 명시한다.",
        "",
        "### 이 발견이 lendingRate/avgRate 격차를 설명하는가 -- 부분적으로만, 완전 검증 아님",
        "",
        "이번 조사로 찾은 '15% 수수료'만으로는 이 리포트가 실측한 전체 격차(위 표 3가지 기준 모두 "
        "중앙값 약 0.21, 즉 avgRate 대비 lendingRate가 약 79%p 감소)를 전부 설명하지 못한다 -- 15% "
        "수수료는 최대 15%p 감소 요인일 뿐이다(코인별 편차도 크다 -- 위 스팟체크 표에서 BTC는 비율 "
        "0.06까지 낮고 SOL은 1.0에 근접한다). 위 '금리 결정 메커니즘' 서술과 결합하면 그럴듯한 가설은 "
        "이렇다: `rate`"
        "(lending-rate-history)/`avgRate`(lending-rate-summary)는 **그 시각 매칭된 가장 비싼(한계) "
        "호가** 근방을 반영하는 반면, `lendingRate`는 **그보다 낮게 호가해 매칭된 다수 대여자를 포함한 "
        "평균 실현치**일 가능성이 있다(선착순이 아니라 낮은 호가부터 매칭되는 구조이므로) -- 여기에 "
        "15% 수수료가 추가로 얹힌다. **이것은 정합적인 가설이지, 수치로 분해·검증된 사실이 아니다.** "
        "`lendingRate` 필드 자체를 'net-of-fee 대여자 실수취'라고 못박는 API 문서 원문은 이번에도 "
        "확보하지 못했다.",
        "",
        "### UNCONFIRMED (그대로 남음 -- 실자금 적용 전 확인 필요)",
        "",
        "- `lendingRate` 필드가 정확히 무엇의 평균/추정치인지(매칭된 전체 대여자의 실현 평균인지, "
        "다른 계산인지) API 필드 문서 원문 미확보 -- 필드명 기반 추론이다.",
        "- 위에서 설명한 '15%p 수수료 + 매칭평균 효과' 가설의 수치적 분해(격차 중 몇 %p가 수수료이고 "
        "몇 %p가 매칭평균 효과인지) -- 미검증.",
        "- 헤지 청산 신호 발생부터 실제 회수·매도 체결까지의 지연시간 -- 상품이 flexible이라는 것과 "
        "'즉시 체결'은 별개다, 미확인.",
        "- 이 wave가 사용한 `finance/savings/*` 엔드포인트(P2P 마진대여 풀)와 위 도움말 문서가 설명한 "
        "'OKX Savings/Simple Earn Flexible' 상품이 100% 동일 상품인지 -- API 경로(`finance/savings`)와 "
        "문서 제목(\"OKX Savings\")이 일치해 정황상 같은 상품으로 판단했으나, 별도의 명시적 매핑 "
        "문서는 확보하지 못했다.",
        "",
        "### wave16에서 이어지는 미해결 리스크 (이 wave가 새로 해소한 것 아님)",
        "",
        "- 거래소 분리(OKX 대여/Bitget 실행) 노출 -- wave16 M6/M7 재발 구조, 그대로 유지.",
        "- 라이브-캐시 시차(백테스트 캐시 2026-07-14 종료 vs 이 리포트의 라이브 스냅샷).",
        "",
    ]
    return lines


# ---------------------------------------------------------------------------
# Write.
# ---------------------------------------------------------------------------


def write_wave17_report(results_dir: Path, report_dir: Path, registry_path: Path, cache_dir: Path) -> None:
    payloads: dict[str, dict[str, Any]] = {}
    for candidate_id in ALL_RESULT_IDS:
        try:
            payloads[candidate_id] = _load(results_dir, candidate_id)
        except FileNotFoundError:
            continue
    lending_realized = _load_optional_json(cache_dir / "lending_realized.json")
    wave16_results_dir = results_dir.parent.parent / "wave16_duallayer" / "results"

    verdict_state, verdict_lines = _verdict_section(payloads)

    lines: list[str] = [
        *_header_and_scope_note(lending_realized),
        *_candidate_definitions_section(),
        *_f0_reproduction_section(payloads, wave16_results_dir),
        *_data_snapshot_section(lending_realized),
        *_highlight_coins_section(lending_realized),
        *_volatility_section(lending_realized),
        *_results_table_section(payloads),
        *verdict_lines,
        *_operational_risk_section(),
        "## 다중검정",
        "",
        "이 wave는 wave16의 5개 후보(E0-E4) 중 E1류 4개(F0-F3, F_min 포함 5개)만 재산출한다 -- "
        "새로운 통계적 가설검정(MC/블록셔플)을 도입하지 않으므로 wave16의 DSR_CUMULATIVE_TRIALS(96)에 "
        "이 wave의 시행을 별도로 더하지 않는다(S1-S5 재실행이 없어 다중검정 인플레이션 대상 시행이 "
        "아님 -- gates17.py 모듈 docstring 참조).",
        "",
    ]
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "wave17_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    registry_lines = [
        "# Wave-17 registry",
        "",
        f"모든 항목의 승격 여부는 '{REQUIRED_LABEL}' 라벨 없이는 성립하지 않는다 -- report/wave17_report.md 참조.",
        "",
        "| Candidate | Family | State | 고펀딩기 연환산(결합) | F0 대비 | 분류 |",
        "|---|---|---|---|---|---|",
    ]
    f0_high = payloads.get("F0", {}).get("regime_breakdown_combined", {}).get("high_funding_mean_annualized_return")
    for candidate_id in ALL_RESULT_IDS:
        payload = payloads.get(candidate_id)
        if payload is None:
            registry_lines.append(f"| {candidate_id} | wave17_lending_verified | PENDING | - | - | run 미실행 |")
            continue
        combined_high = payload.get("regime_breakdown_combined", {}).get("high_funding_mean_annualized_return")
        vs_f0 = "N/A"
        if combined_high is not None and f0_high is not None:
            vs_f0 = f"{(combined_high - f0_high) * 100.0:+.2f}%p"
        classification = {"F0": "기준선", "F1": "핵심 재산출", "F2": "스트레스(50%할인)", "F3": "무결성 검증", "F_min": "참고(게이트 미적용)"}.get(candidate_id, "참고")
        registry_lines.append(f"| {candidate_id} | wave17_lending_verified | EVALUATED | {_fmt_pct(combined_high)} | {vs_f0} | {classification} |")
    verdict_label = {
        "PASS": f"실수취 기준 재확인 유효 ({REQUIRED_LABEL})",
        "FAIL": "기각/일부 실패 (F1/F2가 F0를 못 넘거나 F3!=F0 -- SPEC.md 판정 규칙)",
        "PENDING": "미결정 (일부 후보 결과 없음)",
    }[verdict_state]
    registry_lines.append("")
    registry_lines.append(f"**최종 판정**: {verdict_state} -- {verdict_label} (report/wave17_report.md 참조).")
    registry_lines.append("")
    registry_lines.append(
        "**wave16과의 관계**: 이 wave는 wave16의 avgRate 기반 E1 대여수익 가정을 lendingRate 기반으로 "
        "정정한 재산출이다 -- wave16 E2/E3/E4(랭킹에 대여이자 합산)는 재검토 대상이 아니다(그 후보들은 "
        "avgRate 문제와 무관하게 이미 기각됐다, research/wave16_duallayer/REGISTRY.md 참조)."
    )
    registry_path.write_text("\n".join(registry_lines) + "\n", encoding="utf-8")


__all__ = ["write_wave17_report"]
