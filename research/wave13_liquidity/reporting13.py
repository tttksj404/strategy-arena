# Wave-13 markdown report + registry writer. Pure formatting over already-computed
# results/L{1..5}.json payloads written by run_wave13.py --stage run/gates, plus
# cache/measured_spreads.json (work item 1's raw data) and a read-only citation of
# research/wave12_frontier's own U0/U2 results (for the L3-vs-U0 / L4-vs-U2 comparison
# SPEC.md asks for -- read fresh off disk rather than hardcoded, unlike wave12's own
# Y4-citation constants, since wave12's results are still on disk and don't need freezing
# into a literal here).

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from research.wave1.common import load_json
from research.wave13_liquidity.configs13 import CONFIG_IDS, get_config
from research.wave13_liquidity.gates13 import DSR_CUMULATIVE_TRIALS, S3_BLOCK_MDD_P95_MAX, S5_BLOCK_MDD_P95_MAX

WAVE12_RESULTS_DIR = Path(__file__).resolve().parents[1] / "wave12_frontier" / "results"


def _load(results_dir: Path, candidate_id: str) -> dict[str, Any]:
    path = results_dir / f"{candidate_id}.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _load_wave12_baseline(candidate_id: str) -> dict[str, Any] | None:
    path = WAVE12_RESULTS_DIR / f"{candidate_id}.json"
    if not path.exists():
        return None
    try:
        return load_json(path)  # type: ignore[return-value]
    except (OSError, ValueError):
        return None


def _fmt_pct(value: float | None, digits: int = 2) -> str:
    return "N/A" if value is None else f"{value * 100.0:.{digits}f}%"


def _fmt_usd(value: float | None, digits: int = 2) -> str:
    return "N/A" if value is None else f"${value:,.{digits}f}"


def _fmt_bp(value: float | None, digits: int = 3) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}bp"


def _fmt_universe(config) -> str:
    if config.universe_kind == "fixed":
        return "/".join(sym.removesuffix("USDT") for sym in (config.fixed_symbols or ()))
    if config.universe_kind == "breadth":
        return f"top{config.breadth}"
    return f"동적(>=${config.dynamic_volume_floor_usdt/1e6:.0f}M ∧ <={config.dynamic_slippage_cap_bp:.0f}bp)"


# ---------------------------------------------------------------------------
# Section: measured spreads (work item 1's raw data + fitted mapping).
# ---------------------------------------------------------------------------


def _limitations_section() -> list[str]:
    return [
        "## 방법론 한계 (필독 -- 아래 모든 수치에 선행하는 전제)",
        "",
        "- 실측 스프레드는 **2026-07-22 단일 시점 스냅샷(평온장)** 이다. 고펀딩기=고변동기에는 "
        "스프레드가 확대되는 것이 시장미시구조 일반론이므로, 이 실측을 과거 전체 백테스트 "
        "구간(수년)에 균일 적용하는 것은 근사이며 고변동기 비용을 과소평가할 가능성이 있다. "
        "이 한계를 상쇄하기 위해 S5 게이트에서 실측치 x3 스트레스를 강제한다.",
        "- 백테스트의 OHLCV/펀딩비 데이터는 새로 수집하지 않았다 -- `research/wave12_frontier/cache/`의 "
        "기존 Binance 소스 캐시를 그대로 재사용한다 (이 wave가 새로 수집한 것은 Bitget 실측 "
        "스프레드뿐). 매핑 함수는 Bitget 24h 거래대금 기준으로 적합했고, 백테스트 적용 시에는 "
        "Binance 소스 30일 평균 거래대금(t-1까지 시점기준)을 같은 척도로 대입한다 -- 거래소가 "
        "다르므로 절대 수준 차이가 있을 수 있으나, 대형>중형>소형의 상대적 유동성 서열은 "
        "크립토 무기한선물 시장 전반의 공유된 구조로 가정했다.",
        "- 이 근사들 때문에 아래 결과는 \"현재 실측 비용구조가 과거에도 구조적으로 비슷했다\"는 "
        "가정 위에 있다. 절대적으로 정확한 과거 실행비용 재구성이 아니다.",
    ]


def _measured_spread_table(payload: dict[str, Any]) -> list[str]:
    measurements = payload.get("measurements", [])
    lines = [
        "## 실측 스프레드 (작업 1 원자료 -- Bitget 라이브 오더북)",
        "",
        f"수집 시각(UTC): {payload.get('collected_at_utc', 'N/A')} · 대상: Bitget USDT-M 무기한선물 "
        f"{payload.get('total_live_contracts', 'N/A')}종 중 {payload.get('measured_count', len(measurements))}종 실측 "
        f"(목표 랭크 {payload.get('target_rank_count', 'N/A')}개, 미해결 {len(payload.get('failed_ranks', []))}개) · "
        f"주문크기 ${payload.get('order_size_usdt', 45.0):.0f} (전략의 실제 레그 사이즈와 동일).",
        "",
        payload.get("snapshot_limitation", ""),
        "",
        "| 랭크 | 심볼 | 24h거래대금 | half-spread | walk-cost($45) | 실효슬리피지 | 최상단깊이($) | 부족 |",
        "|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for item in measurements:
        insuf = "부족" if item.get("insufficient_depth") else "-"
        lines.append(
            f"| {item['rank']} | {item['symbol']} | {_fmt_usd(item['usdt_volume_24h'], 0)} | "
            f"{_fmt_bp(item['half_spread_bp'])} | {_fmt_bp(item['walk_cost_bp'])} | "
            f"{_fmt_bp(item['effective_slippage_bp'])} | {_fmt_usd(item['depth_usdt_top'], 1)} | {insuf} |"
        )
    return lines


def _anomaly_callouts(payload: dict[str, Any], anchors: dict[str, Any]) -> list[str]:
    """Flags symbols whose measured cost is a large POSITIVE residual above what the
    fitted mapping predicts for their own volume level -- i.e. genuine outliers the
    bucket-median + isotonic fit had to smooth over (the AMCUSDT-style anomaly this wave's
    background section describes), not just "any two rows out of order." Comparing against
    the FITTED curve (one predicted value per symbol) rather than pairwise against every
    other row avoids the earlier degenerate version of this function, which kept
    re-flagging dozens of rows against the same single global-cheapest neighbor."""
    measurements = payload.get("measurements", [])
    anchor_log_volume = anchors.get("anchor_log_volume")
    anchor_bp = anchors.get("anchor_bp")
    if not measurements or not anchor_log_volume or not anchor_bp:
        return []

    residuals: list[tuple[float, dict[str, Any], float]] = []
    for item in measurements:
        volume = float(item["usdt_volume_24h"])
        predicted_bp = float(np.interp(np.log10(volume), anchor_log_volume, anchor_bp, left=anchor_bp[0], right=anchor_bp[-1]))
        actual_bp = float(item["effective_slippage_bp"])
        residual = actual_bp - predicted_bp
        # Only flag a MEANINGFUL, absolute overshoot -- both a relative bar (actual >=
        # 2x predicted) and an absolute floor (>= 2bp) so tiny buckets near zero bp don't
        # trigger on noise (e.g. 0.02bp actual vs 0.01bp predicted is a 2x ratio but
        # economically irrelevant).
        if residual >= 2.0 and actual_bp >= predicted_bp * 2.0:
            residuals.append((residual, item, predicted_bp))
    if not residuals:
        return []
    residuals.sort(key=lambda entry: entry[0], reverse=True)
    lines = [
        "",
        "**비단조성/이상치 사례** (매핑 함수가 그 거래대금 수준에서 예측하는 값보다 실측이 "
        "훨씬 비싼 경우 -- 구간중앙값+등위회귀는 이런 개별 이상치를 스무딩하지만 원자료에는 "
        "이렇게 남아있다):",
        "",
    ]
    for residual, item, predicted_bp in residuals[:8]:
        lines.append(
            f"- **{item['symbol']}**(랭크{item['rank']}, 24h거래대금 {_fmt_usd(item['usdt_volume_24h'], 0)}): "
            f"실측 {_fmt_bp(item['effective_slippage_bp'])} vs 매핑 예측 {_fmt_bp(predicted_bp)} "
            f"(잔차 +{residual:.2f}bp) -- 거래대금만으로는 예측 못하는 개별 유동성 결함."
        )
    return lines


def _mapping_table(anchors: dict[str, Any]) -> list[str]:
    lines = [
        "## 거래대금 -> 슬리피지 매핑 함수 (구간중앙값 + 등위회귀, 단조성 강제)",
        "",
        f"원자료 {anchors.get('raw_point_count', 'N/A')}개 실측점을 로그거래대금 구간으로 묶어 "
        "구간중앙값을 구하고, Pool-Adjacent-Violators로 비증가 단조성을 강제한 결과:",
        "",
        "| 구간중심 거래대금 | 구간표본수 | 적합 슬리피지(bp) |",
        "|---:|---:|---:|",
    ]
    anchor_log_v = anchors.get("anchor_log_volume", [])
    anchor_bp = anchors.get("anchor_bp", [])
    counts = anchors.get("bucket_counts", [])
    for log_v, bp, count in zip(anchor_log_v, anchor_bp, counts):
        lines.append(f"| {_fmt_usd(10.0**log_v, 0)} | {count} | {bp:.3f} |")
    lines.append("")
    lines.append(
        "NaN/미보유 거래대금(신규상장 등)은 최악 구간값으로 fail-closed 처리한다 (표의 최상단 행 "
        "= 가장 비싼 구간). BTC/ETH에 대한 별도 하드코딩 예외는 없다 -- 거래대금 자체가 압도적으로 "
        "높아 매핑이 자연스럽게 최저구간에 배치한다 (wave12까지의 '메이저 1bp 고정' 가정을 대체)."
    )
    return lines


# ---------------------------------------------------------------------------
# Section: L1-L5 config table + frontier metrics.
# ---------------------------------------------------------------------------


def _config_table() -> list[str]:
    lines = ["| Candidate | 유니버스 | 정의 |", "|---|---|---|"]
    for candidate_id in CONFIG_IDS:
        config = get_config(candidate_id)
        lines.append(f"| {candidate_id} | {_fmt_universe(config)} | {config.note} |")
    return lines


def _metrics_table(payloads: dict[str, dict]) -> list[str]:
    lines = [
        "| Candidate | 편입심볼(정적) | 편입심볼(일별중앙값) | 고펀딩기 연환산 | MC p05 | 블록MDD p95 | 건당비용($) | 스트레스(x3) 고펀딩기 | 스트레스 MDD p95 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for candidate_id in CONFIG_IDS:
        payload = payloads[candidate_id]
        metadata = payload.get("metadata", {})
        regime = payload.get("regime_breakdown")
        stress_regime = payload.get("stress_regime_breakdown")
        gates = payload.get("gates")
        if regime is None or gates is None:
            lines.append(f"| {candidate_id} | {metadata.get('universe_size_static', 'N/A')} | - | - | - | - | - | - | run --stage gates 필요 |")
            continue
        high_funding = regime.get("high_funding_mean_annualized_return")
        stress_high_funding = stress_regime.get("high_funding_mean_annualized_return") if stress_regime else None
        eligible_median = metadata.get("eligible_count_stats", {}).get("median")
        eligible_str = "N/A" if eligible_median is None else f"{eligible_median:.1f}"
        stress_mdd = gates["gate_s5"].get("stress_block_mdd_p95")
        lines.append(
            f"| {candidate_id} | {metadata.get('universe_size_static', 'N/A')} | {eligible_str} | "
            f"{_fmt_pct(high_funding)} | {_fmt_usd(gates['gate_s2'].get('p05'))} | {_fmt_pct(gates['gate_s3'].get('mdd_p95'))} | "
            f"{_fmt_usd(metadata.get('avg_cost_per_trade_usdt'))} | {_fmt_pct(stress_high_funding)} | {_fmt_pct(stress_mdd)} |"
        )
    return lines


def _gate_table(payloads: dict[str, dict]) -> list[str]:
    lines = [
        "| Candidate | S1 구조 | S2 MC(p05/ruin) | S3 블록MDD p95(<=10%) | S4 실행가능성 | S5 스트레스(부호∧MDD<=15%) | Overall | 승격 | 실패사유 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for candidate_id in CONFIG_IDS:
        payload = payloads[candidate_id]
        gates = payload.get("gates")
        if not gates:
            lines.append(f"| {candidate_id} | - | - | - | - | - | NOT_EVALUATED | - | run --stage gates 필요 |")
            continue
        s1, s2, s3, s4, s5 = gates["gate_s1"], gates["gate_s2"], gates["gate_s3"], gates["gate_s4"], gates["gate_s5"]
        promo = gates["promotion"]
        s1_cell = f"{s1['status']} ({s1['leverage_multiplier_of_active_capital']:.2f}x)"
        s2_cell = f"{s2['status']} (p05={_fmt_usd(s2['p05'])}, ruin={_fmt_pct(s2['ruin_probability'])})"
        s3_cell = f"{s3['status']} ({_fmt_pct(s3['mdd_p95'])})"
        s4_cell = f"{s4['status']} (레그{_fmt_usd(s4['leg_usdt_nominal'])})"
        s5_cell = f"{s5['status']} (부호={'+' if s5['sign_preserved'] else '반전'}, MDD={_fmt_pct(s5.get('stress_block_mdd_p95'))})"
        actually_promoted = gates["overall"] == "PASS" and bool(promo["promoted"])
        promoted_cell = "YES" if actually_promoted else "no"
        reasons = ", ".join(gates.get("failure_reasons", [])) or "-"
        lines.append(f"| {candidate_id} | {s1_cell} | {s2_cell} | {s3_cell} | {s4_cell} | {s5_cell} | {gates['overall']} | {promoted_cell} | {reasons} |")
    return lines


def _wave12_comparison_section(payloads: dict[str, dict]) -> list[str]:
    """SPEC.md: L3="wave12 U0 대응", L4="wave12 정점 U2 대응" -- same universe-membership
    rule (top100/12mo, top200/12mo respectively), only the cost model differs. Reads
    wave12's own U0.json/U2.json off disk (read-only) rather than re-deriving anything."""
    lines = ["## L3/L4 vs wave12 U0/U2 (동일 유니버스 규칙, 비용모델만 실측 교체)", ""]
    pairs = (("L3", "U0"), ("L4", "U2"))
    any_data = False
    for wave13_id, wave12_id in pairs:
        wave12_payload = _load_wave12_baseline(wave12_id)
        wave13_payload = payloads.get(wave13_id, {})
        wave13_regime = wave13_payload.get("regime_breakdown")
        if wave12_payload is None or wave13_regime is None:
            lines.append(f"- {wave13_id} vs {wave12_id}: 비교 불가 (wave12 결과 또는 wave13 게이트 미실행).")
            continue
        any_data = True
        wave12_regime = wave12_payload.get("regime_breakdown", {})
        wave12_gates = wave12_payload.get("gates", {})
        wave12_high = wave12_regime.get("high_funding_mean_annualized_return")
        wave13_high = wave13_regime.get("high_funding_mean_annualized_return")
        wave12_mdd = wave12_gates.get("gate_s3", {}).get("mdd_p95")
        wave13_mdd = wave13_payload.get("gates", {}).get("gate_s3", {}).get("mdd_p95")
        wave12_cost = wave12_payload.get("metadata", {}).get("avg_cost_per_trade_usdt")
        wave13_cost = wave13_payload.get("metadata", {}).get("avg_cost_per_trade_usdt")
        delta = "-" if (wave12_high is None or wave13_high is None) else f"{(wave13_high - wave12_high) * 100.0:+.2f}%p"
        lines.append(
            f"- **{wave13_id}(실측비용) vs {wave12_id}(계층가정비용)**: 고펀딩기 연환산 "
            f"{_fmt_pct(wave13_high)} vs {_fmt_pct(wave12_high)} ({delta}); 블록MDD p95 "
            f"{_fmt_pct(wave13_mdd)} vs {_fmt_pct(wave12_mdd)}; 건당비용 {_fmt_usd(wave13_cost)} vs {_fmt_usd(wave12_cost)}."
        )
    if not any_data:
        lines.append("- wave12_frontier/results/U0.json, U2.json 을 찾을 수 없어 비교를 생략한다.")
    return lines


def _l5_deep_dive(payloads: dict[str, dict]) -> list[str]:
    payload = payloads.get("L5", {})
    metadata = payload.get("metadata", {})
    gates = payload.get("gates")
    config = get_config("L5")
    lines = [
        "## L5 심층 분석 (핵심 가설: 랭크 대신 체결가능성으로 거르기)",
        "",
        f"필터: 진입(매일) 시점 30d 평균 거래대금 >= {_fmt_usd(config.dynamic_volume_floor_usdt, 0)} 이고 "
        f"실측매핑 슬리피지 <= {config.dynamic_slippage_cap_bp:.0f}bp 인 심볼만, 랭크 무관, 매일 재평가.",
        "",
        f"- 모(母)후보군(3mo 히스토리 하한) 정적 크기: {metadata.get('universe_size_static', 'N/A')}종.",
        f"- 실제 매일 두 조건을 모두 통과해 순위매김 대상이 된 심볼 수: 중앙값 "
        f"{metadata.get('eligible_count_stats', {}).get('median', 'N/A')}, "
        f"평균 {metadata.get('eligible_count_stats', {}).get('mean', 'N/A')}, "
        f"최대 {metadata.get('eligible_count_stats', {}).get('max', 'N/A')}.",
    ]
    if gates is not None:
        regime = payload.get("regime_breakdown", {})
        lines.append(
            f"- 고펀딩기 연환산 {_fmt_pct(regime.get('high_funding_mean_annualized_return'))}, "
            f"건당비용 {_fmt_usd(metadata.get('avg_cost_per_trade_usdt'))}, Overall {gates['overall']}."
        )
    else:
        lines.append("- 게이트 미실행 -- `--stage gates` 필요.")
    return lines


def _verdict_section(payloads: dict[str, dict]) -> tuple[str, list[str]]:
    promoted: list[tuple[str, float]] = []
    attempted: list[str] = []
    for candidate_id in CONFIG_IDS:
        gates = payloads[candidate_id].get("gates")
        if not gates:
            continue
        attempted.append(candidate_id)
        promo = gates["promotion"]
        if gates["overall"] == "PASS" and promo["promoted"]:
            high_funding = promo["high_funding_mean_annualized_return"]
            if high_funding is not None:
                promoted.append((candidate_id, high_funding))

    if promoted:
        winner_id, winner_return = max(promoted, key=lambda item: item[1])
        winner_config = get_config(winner_id)
        lines = [
            "## 판정: 통과",
            "",
            f"S1-S5 전부 PASS한 구성: {', '.join(cid for cid, _ in promoted)}.",
            "",
            f"**최종 추천: {winner_id}** ({_fmt_universe(winner_config)}) -- 고펀딩기 연환산 "
            f"{_fmt_pct(winner_return)}, 실측 비용모델 하에서 S1-S5 전부 PASS (스트레스 x3에서도 "
            "고펀딩기 연환산 부호 유지 + 블록MDD p95 <= 15%).",
        ]
        return "PASS", lines

    lines = [
        "## 판정: 실측비용 하에서 이 계열은 리스크예산 밖",
        "",
        "S1-S5를 모두 통과한 구성이 없다 (게이트 완화 없이 정직 보고):",
        "",
        "| Candidate | 고펀딩기 연환산 | Overall | 실패사유 |",
        "|---|---:|---|---|",
    ]
    for candidate_id in attempted:
        payload = payloads[candidate_id]
        gates = payload["gates"]
        regime = payload.get("regime_breakdown", {})
        high_funding = regime.get("high_funding_mean_annualized_return")
        gate_cell = gates["overall"] if gates["overall"] == "PASS" else f"FAIL({', '.join(gates['failure_reasons']) or '?'})"
        lines.append(f"| {candidate_id} | {_fmt_pct(high_funding)} | {gates['overall']} | {gate_cell} |")
    return "FAIL", lines


def _dsr_note() -> str:
    return (
        f"다중검정 보정: 누적 시행 {DSR_CUMULATIVE_TRIALS}회(wave12까지 71회 + 이 wave의 L1-L5 5개) 기준 "
        "DSR(Deflated Sharpe Ratio) 참고치를 각 결과 JSON의 `reference_metrics.dsr`에 기록 "
        "(샤프는 참고 지표이며 승격 판정에는 사용하지 않음, wave10/11/12와 동일 원칙)."
    )


def write_wave13_report(results_dir: Path, report_dir: Path, registry_path: Path, cache_dir: Path) -> None:
    payloads = {candidate_id: _load(results_dir, candidate_id) for candidate_id in CONFIG_IDS}
    spread_payload_path = cache_dir / "measured_spreads.json"
    spread_payload = json.loads(spread_payload_path.read_text(encoding="utf-8")) if spread_payload_path.exists() else {}
    # anchor/mapping table reads from whichever config's results carries cost_mapping
    # (identical across all five -- one shared mapping fit once per --stage run).
    anchors = next((payloads[cid]["cost_mapping"] for cid in CONFIG_IDS if "cost_mapping" in payloads.get(cid, {})), {})
    verdict, verdict_lines = _verdict_section(payloads)

    lines: list[str] = [
        "# Wave-13 리포트 — 실측 스프레드 비용모델 재보정 + 유동성 제약 캐리 (L1-L5)",
        "",
        *_limitations_section(),
        "",
        *_measured_spread_table(spread_payload),
        *_anomaly_callouts(spread_payload, anchors),
        "",
        *_mapping_table(anchors),
        "",
        "## 구성 정의",
        "",
        *_config_table(),
        "",
        "## 구성 x 지표",
        "",
        *_metrics_table(payloads),
        "",
        *_wave12_comparison_section(payloads),
        "",
        *_l5_deep_dive(payloads),
        "",
        *verdict_lines,
        "",
        "## 게이트 (S1-S5)",
        "",
        *_gate_table(payloads),
        "",
        "## 다중검정",
        "",
        _dsr_note(),
        "",
    ]
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "wave13_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    registry_lines = ["# Wave-13 registry", "", "| Candidate | Family | State | Verdict | 승격 | 분류 |", "|---|---|---|---|---|---|"]
    for candidate_id in CONFIG_IDS:
        payload = payloads[candidate_id]
        gates = payload.get("gates")
        if not gates:
            registry_lines.append(f"| {candidate_id} | wave13_liquidity | PENDING | - | - | gates 미실행 |")
            continue
        promo = gates["promotion"]
        result_verdict = gates["overall"]
        actually_promoted = result_verdict == "PASS" and bool(promo["promoted"])
        promoted_cell = "YES" if actually_promoted else "no"
        classification = "승격" if actually_promoted else (f"게이트위반({', '.join(gates['failure_reasons'])})" if result_verdict != "PASS" else "no")
        registry_lines.append(f"| {candidate_id} | wave13_liquidity | EVALUATED | {result_verdict} | {promoted_cell} | {classification} |")
    registry_lines.append("")
    registry_lines.append(f"**판정**: {'통과' if verdict == 'PASS' else '실측비용 하에서 이 계열은 리스크예산 밖'} (자세한 내용은 report/wave13_report.md 참조).")
    registry_path.write_text("\n".join(registry_lines) + "\n", encoding="utf-8")


__all__ = ["write_wave13_report"]
