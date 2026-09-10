from generate_strategy_research_summary import (
    FAMILIES,
    MARK_ARCHIVE,
    _artifact_rows,
    _dynamic_olmar_progress,
    _short_horizon_counterfactual_progress,
)


def test_status_summary_includes_all_completed_tree_interval_reports() -> None:
    assert ("비선형 트리 앙상블", "bitget_*h_walkforward_tree_search_report.json") in FAMILIES
    rows = [report for family, report in _artifact_rows() if family == "비선형 트리 앙상블"]
    assert {report["data"]["interval"] for report in rows} == {"1H", "4H", "6H", "12H"}
    assert all(report["predeclared_search"]["raw_configurations"] == 32 for report in rows)


def test_status_summary_includes_all_completed_var_interval_reports() -> None:
    assert ("다변량 VAR", "bitget_*h_walkforward_var_search_report.json") in FAMILIES
    rows = [report for family, report in _artifact_rows() if family == "다변량 VAR"]
    assert {report["data"]["interval"] for report in rows} == {"1H", "4H", "6H", "12H"}
    assert all(report["predeclared_search"]["raw_configurations"] == 32 for report in rows)


def test_status_summary_includes_all_completed_kalman_interval_reports() -> None:
    assert ("칼만 페어 MR", "bitget_*h_kalman_pair_search_report.json") in FAMILIES
    rows = [report for family, report in _artifact_rows() if family == "칼만 페어 MR"]
    assert {report["data"]["interval"] for report in rows} == {"1H", "4H", "6H", "12H"}
    assert all(report["predeclared_search"]["pair_pool"] == 78 for report in rows)


def test_status_summary_includes_all_completed_pca_stat_arb_reports() -> None:
    assert ("동적 PCA 잔차 StatArb", "bitget_*h_pca_stat_arb_search_report.json") in FAMILIES
    rows = [report for family, report in _artifact_rows() if family == "동적 PCA 잔차 StatArb"]
    assert {report["data"]["interval"] for report in rows} == {"1H", "4H", "6H", "12H"}
    assert all(report["predeclared_search"]["raw_configurations"] == 192 for report in rows)


def test_status_summary_includes_all_completed_hmm_regime_reports() -> None:
    assert ("Gaussian HMM 레짐", "bitget_*h_hmm_regime_search_report.json") in FAMILIES
    rows = [report for family, report in _artifact_rows() if family == "Gaussian HMM 레짐"]
    assert {report["data"]["interval"] for report in rows} == {"1H", "4H", "6H", "12H"}
    assert all(report["predeclared_search"]["raw_configurations"] == 64 for report in rows)


def test_status_summary_registers_bocpd_regime_reports() -> None:
    assert ("BOCPD 레짐", "bitget_*h_bocpd_regime_search_report.json") in FAMILIES
    rows = [report for family, report in _artifact_rows() if family == "BOCPD 레짐"]
    assert {report["data"]["interval"] for report in rows} == {"1H", "4H", "6H", "12H"}
    assert all(report["predeclared_search"]["raw_configurations"] == 64 for report in rows)


def test_status_summary_registers_all_cross_reversal_interval_reports() -> None:
    assert ("단기 반전", "bitget_*_cross_reversal_report.json") in FAMILIES
    rows = [report for family, report in _artifact_rows() if family == "단기 반전"]
    assert {report["data"]["interval"] for report in rows} == {"5m", "1H", "4H", "6H", "12H"}
    by_interval = {report["data"]["interval"]: report for report in rows}
    assert by_interval["5m"]["predeclared_search"]["raw_configurations"] == 288


def test_status_summary_registers_all_factor_family_interval_reports() -> None:
    assert ("횡단면 팩터", "bitget_*_factor_family_search_report.json") in FAMILIES
    rows = [report for family, report in _artifact_rows() if family == "횡단면 팩터"]
    assert {report["data"]["interval"] for report in rows} == {"5m", "1H", "4H", "6H", "12H"}
    by_interval = {report["data"]["interval"]: report for report in rows}
    assert by_interval["5m"]["predeclared_search"]["raw_configurations"] == 256


def test_status_summary_registers_all_olmar_interval_reports() -> None:
    assert ("시장중립 OLMAR", "bitget_*_olmar_search_report.json") in FAMILIES
    rows = [report for family, report in _artifact_rows() if family == "시장중립 OLMAR"]
    assert {report["data"]["interval"] for report in rows} == {"5m", "15m", "30m", "1H", "4H", "6H", "12H"}
    assert all(report["predeclared_search"]["raw_configurations"] == 48 for report in rows)


def test_status_summary_registers_triple_barrier_reports() -> None:
    assert ("성숙 트리플 배리어", "bitget_*_triple_barrier_search_report.json") in FAMILIES
    rows = [report for family, report in _artifact_rows() if family == "성숙 트리플 배리어"]
    assert {report["data"]["interval"] for report in rows} == {"1H", "4H", "6H", "12H"}
    assert all(report["predeclared_search"]["raw_configurations"] == 64 for report in rows)


def test_status_summary_registers_directional_change_reports() -> None:
    assert ("방향전환 DC 이벤트", "bitget_*_directional_change_search_report.json") in FAMILIES
    rows = [report for family, report in _artifact_rows() if family == "방향전환 DC 이벤트"]
    assert {report["data"]["interval"] for report in rows} == {"5m", "15m", "30m", "1H", "4H", "6H", "12H"}
    assert all(report["predeclared_search"]["raw_configurations"] == 48 for report in rows)


def test_status_summary_keeps_dynamic_current_universe_screen_non_promotable() -> None:
    lines = _dynamic_olmar_progress()

    assert any("승격 불가" in line for line in lines)


def test_short_horizon_counterfactual_is_excluded_from_safe_strategy_totals() -> None:
    assert all("short_horizon_max_return" not in pattern for _, pattern in FAMILIES)
    lines = _short_horizon_counterfactual_progress()
    assert any("반사실" in line for line in lines)
    assert any("횡단면 팩터" in line for line in lines)
    assert any("고전·페어" in line for line in lines)


def test_status_summary_reads_the_active_intraday_mark_archive() -> None:
    assert MARK_ARCHIVE.parent.name == "bitget_mark_history_full_intraday_20260825_v1"
