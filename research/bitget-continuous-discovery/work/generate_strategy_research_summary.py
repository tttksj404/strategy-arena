"""Create a compact, reproducible status report from completed paper-search artifacts."""

from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

WORK = Path(__file__).resolve().parent
OUTPUT = WORK.parent / "outputs" / "strategy-arena-algorithm-search-status.md"
ARCHIVE = WORK / "bitget_history_full_intraday_20260824_v5" / "MANIFEST.json"
MARK_ARCHIVE = WORK / "bitget_mark_history_full_intraday_20260825_v1" / "MANIFEST.json"
LIQUID_UNIVERSE = WORK / "bitget_liquid_alt_universe_20260825.json"
LIQUID_DAILY_ARCHIVE = WORK / "bitget_history_liquid_alt_daily_20260825_v1" / "MANIFEST.json"
DYNAMIC_OLMAR_SCREEN = WORK / "bitget_dynamic_daily_olmar_screen_report.json"
SHORT_HORIZON_COUNTERFACTUAL_PATTERN = "bitget_*_short_horizon_*max_return_fixed_notional_report.json"
FAMILIES = (
    ("고전 기술", "bitget_*h_classic_family_search_report.json"),
    ("횡단면 팩터", "bitget_*_factor_family_search_report.json"),
    ("단기 반전", "bitget_*_cross_reversal_report.json"),
    ("동적 페어 MR", "bitget_*h_dynamic_pair_search_report.json"),
    ("워크포워드 Ridge", "bitget_*h_walkforward_ridge_search_report.json"),
    ("복합 CTREND", "bitget_*h_ctrend_search_report.json"),
    ("롱온리 추세", "bitget_*h_long_only_search_report.json"),
    ("저회전 롱온리", "bitget_*h_low_turnover_long_only_search_report.json"),
    ("비선형 RFF", "bitget_*h_walkforward_rff_search_report.json"),
    ("비선형 트리 앙상블", "bitget_*h_walkforward_tree_search_report.json"),
    ("다변량 VAR", "bitget_*h_walkforward_var_search_report.json"),
    ("칼만 페어 MR", "bitget_*h_kalman_pair_search_report.json"),
    ("동적 PCA 잔차 StatArb", "bitget_*h_pca_stat_arb_search_report.json"),
    ("Gaussian HMM 레짐", "bitget_*h_hmm_regime_search_report.json"),
    ("BOCPD 레짐", "bitget_*h_bocpd_regime_search_report.json"),
    ("시장중립 OLMAR", "bitget_*_olmar_search_report.json"),
    ("성숙 트리플 배리어", "bitget_*_triple_barrier_search_report.json"),
    ("방향전환 DC 이벤트", "bitget_*_directional_change_search_report.json"),
    ("리스크 패리티", "bitget_*h_risk_parity_search_report.json"),
)
INTERVAL_ORDER = {"1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30, "1H": 60, "4H": 240, "6H": 360, "12H": 720}


def _winners(report: dict) -> int:
    values = report.get("selected_validation_winners")
    if values is not None:
        return len(values)
    return int(report.get("selected_validation_winner") is not None)


def _eligible(report: dict) -> int:
    value = report.get("eligible_for_mark_price_paper_validation", report.get("eligible_for_further_paper_validation", []))
    return len(value) if isinstance(value, list) else int(bool(value))


def _artifact_rows() -> list[tuple[str, dict]]:
    rows: list[tuple[str, dict]] = []
    for family, pattern in FAMILIES:
        for path in WORK.glob(pattern):
            rows.append((family, json.loads(path.read_text(encoding="utf-8"))))
    return sorted(rows, key=lambda item: (item[0], INTERVAL_ORDER[item[1]["data"]["interval"]]))


def _archive_progress() -> list[str]:
    manifest = json.loads(ARCHIVE.read_text(encoding="utf-8"))
    totals = Counter(Path(item["relative_path"]).parts[1] for item in manifest["partitions"])
    nonempty = Counter(Path(item["relative_path"]).parts[1] for item in manifest["partitions"] if item["rows"] > 0)
    return [f"- `{interval}`: 파티션 {totals[interval]}개, 유효 행 보유 {nonempty[interval]}개" for interval in manifest["intervals"]]


def _mark_archive_progress() -> str:
    if not MARK_ARCHIVE.exists():
        return "- mark-price 장기 아카이브: 수집 준비 중"
    manifest = json.loads(MARK_ARCHIVE.read_text(encoding="utf-8"))
    return (
        f"- mark-price 장기 아카이브: `{len(manifest['partitions'])}`개 파티션, "
        f"완료={str(manifest['complete']).lower()}, 기준시각={manifest['snapshot_ms']}"
    )


def _liquid_alt_progress() -> list[str]:
    if not LIQUID_UNIVERSE.exists():
        return ["- 확장 알트 우주: 스냅샷 준비 중"]
    universe = json.loads(LIQUID_UNIVERSE.read_text(encoding="utf-8"))
    rows = [f"- 확장 알트 우주: Bitget 현재 거래 가능·비RWA·24시간 회전율 1,000만 USDT 이상·스프레드 10bp 이하 기준 `{len(universe['entries'])}`개"]
    if LIQUID_DAILY_ARCHIVE.exists():
        archive = json.loads(LIQUID_DAILY_ARCHIVE.read_text(encoding="utf-8"))
        rows.append(f"- 확장 우주 일봉 아카이브: `{len(archive['partitions'])}`개 파티션, 완료={str(archive['complete']).lower()}")
    rows.append("- 확장 우주는 현재 유동성으로 고른 라이브 후보군이므로, 상장 시점·시계열 유동성으로 생존편향을 제거하기 전에는 승격 근거로 사용하지 않는다.")
    return rows


def _dynamic_olmar_progress() -> list[str]:
    if not DYNAMIC_OLMAR_SCREEN.exists():
        return ["- 확장 동적 OLMAR 화면: 실행 대기"]
    report = json.loads(DYNAMIC_OLMAR_SCREEN.read_text(encoding="utf-8"))
    return [
        "- 확장 동적 OLMAR 일봉 화면: 현재 유동성 스냅샷 기반 생존편향이 남아 있어 **승격 불가 탐색**으로만 기록한다.",
        (
            f"- 동적 화면: 자산 {report['data']['asset_count']}개, 상장 후 대기 {report['data']['minimum_listing_bars']}일, "
            f"조합 {report['predeclared_search']['raw_configurations']}개, PBO {report['selection_bias']['pbo']['pbo']:.2%}, "
            f"검증 승자 {_winners(report)}개, 후속 후보 0개."
        ),
    ]


def _short_horizon_counterfactual_progress() -> list[str]:
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in WORK.glob(SHORT_HORIZON_COUNTERFACTUAL_PATTERN)]
    if not reports:
        return ["- 5분 단기 최대수익 반사실 탐색: 실행 중 또는 결과 생성 대기"]
    lines = [
        "- 이 구간은 청산 중단을 끄고 고정명목 PnL을 합산한 반사실 실험이다. 실계좌 수익률·승격 후보·안전 탐색 합계에 포함하지 않는다.",
    ]
    for report in sorted(reports, key=lambda value: INTERVAL_ORDER[value["data"]["interval"]]):
        winner = report["selected_validation_winner"]
        locked = report["locked_test_result"]
        validation = winner["validation_fixed_notional"]
        test = locked["test_fixed_notional"]
        lane = "고전·페어" if "classic" in report["mode"] else "횡단면 팩터"
        lines.append(
            f"- {report['data']['interval']} {lane} 단기 반사실: 조합 {report['predeclared_search']['raw_configurations']}개, "
            f"PBO {report['selection_bias_diagnostic']['pbo']['pbo']:.2%}, "
            f"검증 고정명목 PnL {validation['fixed_notional_net_pnl']:.6f}, "
            f"잠금 OOS 고정명목 PnL {test['fixed_notional_net_pnl']:.6f}, "
            f"전체 경로 청산 프록시 {locked['all_path_liquidation_proxy_hits']}회."
        )
    return lines


def main() -> None:
    artifacts = _artifact_rows()
    total_configs = sum(report["predeclared_search"]["raw_configurations"] for _, report in artifacts)
    total_winners = sum(_winners(report) for _, report in artifacts)
    total_eligible = sum(_eligible(report) for _, report in artifacts)
    lines = [
        "# Strategy-arena 알고리즘 탐색 현황",
        "",
        f"생성 시각: {datetime.now(UTC).isoformat()}",
        "",
        "모든 수치는 Bitget 공개 API 시장가 캔들 기반의 페이퍼 실험이다. 실주문·계정 변경은 전혀 수행하지 않았으며, 청산 검증은 mark-price 전체 이력 확보 전까지 시장가 캔들 프록시다.",
        "",
        "## 완료된 전략군/OOS 스윕",
        "",
        "| 전략군 | 시간봉 | 전략군 내부 고정 조합 | PBO | 검증 승자 | mark-price 후속 후보 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for family, report in artifacts:
        pbo = report["selection_bias"]["pbo"]["pbo"]
        lines.append(
            f"| {family} | {report['data']['interval']} | {report['predeclared_search']['raw_configurations']} | {pbo:.2%} | {_winners(report)} | {_eligible(report)} |"
        )
    lines.extend(
        (
            "",
            f"합계: **{len(artifacts)}개 보고서, {total_configs:,}개 전략군 내부 고정 조합, 검증 승자 {total_winners}개, mark-price 후속 후보 {total_eligible}개**.",
            "",
            "검증 승자가 있었던 항목도 잠금 OOS·PBO·DSR·스트레스 게이트를 모두 통과하지 못하면 후보로 승격하지 않았다. 따라서 현재 승격 가능한 5배 전략은 없다.",
            "",
            "전략군을 순차로 추가한 탐색 연구이므로 전체 탐색은 `selection_independent=false`다. 현재는 후보가 0개라 승격이 없고, 미래 후보가 생기면 모든 전략군을 가로지르는 글로벌 PBO/DSR을 별도 재계산해야 한다.",
            "",
            "## 별도 단기 최대수익 반사실 탐색",
            "",
            *_short_horizon_counterfactual_progress(),
            "",
            "## 장기 데이터 수집 상태",
            "",
            "1H·4H·6H·12H 공통 표본 검증, 30분 시장가 탐색, mark-price 고시간봉·확장 알트 일봉 수집은 완료됐다. 5분·15분 시장가의 계획 파티션도 준비됐고, 1m·3m 시장가와 모든 고빈도 mark-price 이력은 계속 수집 중이다. 고빈도 mark-price가 끝난 뒤에만 같은 사전등록 게이트로 청산 검증까지 추가한다.",
            "",
            *_archive_progress(),
            _mark_archive_progress(),
            *_liquid_alt_progress(),
            *_dynamic_olmar_progress(),
            "",
            "## 해석 경계",
            "",
            "PBO가 낮아도 검증/잠금 OOS 수익성과 스트레스 낙폭이 불합격이면 통과가 아니다. 반대로 PBO가 높거나 잠금 OOS가 음수인 전략은 추가 최적화로 되살리지 않고 탈락으로 기록한다. 펀딩 캐리는 공식 공개 API 보존 범위가 약 90일이라 장기 OOS 승격 근거로 사용하지 않는다.",
        )
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"SUMMARY={OUTPUT}")
    print(f"REPORTS={len(artifacts)} CONFIGURATIONS={total_configs} ELIGIBLE={total_eligible}")


if __name__ == "__main__":
    main()
