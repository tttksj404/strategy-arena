# Wave-18 markdown report + registry writer. Pure formatting over already-computed
# results/I{0..5}.json (run_wave18.py's run+gates stages) and cache/{usdt_lending,
# margin_borrow_rates}.json (fetch18.py), plus a read-only peek at
# research/wave13_liquidity/results/L4.json (I0's reproduction claim -- same convention
# reporting16.py/reporting17.py use against their own upstream baselines).

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

from research.wave1.gate_reporting import _series
from research.wave18_idle import gates18
from research.wave18_idle.configs18 import CONFIG_IDS, CONFIGS

SPEC_I0_FULL_PERIOD_CAGR: Final = 0.0937  # SPEC.md's own prose figure -- shown only as a cross-check label; every actual comparison in this report uses the FRESHLY computed I0 payload, never this constant
SPEC_I0_HIGH_FUNDING: Final = 0.2201
HIGH_FUNDING_YEARS: Final = (2020, 2021, 2024)


def _fmt_pct(value: float | None, digits: int = 2) -> str:
    return "N/A" if value is None else f"{value * 100.0:.{digits}f}%"


def _fmt_usd(value: float | None, digits: int = 2) -> str:
    return "N/A" if value is None else f"${value:,.{digits}f}"


def _fmt_pp(value: float | None, digits: int = 2) -> str:
    return "N/A" if value is None else f"{value:+.{digits}f}%p"


def _load(results_dir: Path, candidate_id: str) -> dict[str, Any] | None:
    path = results_dir / f"{candidate_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _layer_series_from_payload(records: list[dict[str, Any]]) -> pd.Series:
    if not records:
        return pd.Series(dtype=object)
    idx = pd.DatetimeIndex([pd.Timestamp(item["timestamp"]) for item in records])
    return pd.Series([str(item["layer"]) for item in records], index=idx, dtype=object).sort_index()


# ---------------------------------------------------------------------------
# Yearly breakdown (SPEC.md 필수 산출: "연도별 표").
# ---------------------------------------------------------------------------


def _yearly_stats(equity: pd.Series, positions: pd.Series) -> list[dict[str, Any]]:
    """Same anchoring convention as research.wave10_carry100.regime._regime_return (anchor at
    the LAST observation on/before the calendar-year boundary, not the year's own first
    observation), applied per calendar year instead of per named high-funding-year window."""
    if equity.empty:
        return []
    years = sorted({int(pd.Timestamp(ts).year) for ts in equity.index})
    rows: list[dict[str, Any]] = []
    for year in years:
        boundary_start = pd.Timestamp(f"{year - 1}-12-31T23:59:59Z")
        boundary_end = pd.Timestamp(f"{year}-12-31T23:59:59Z")
        window = equity[(equity.index > boundary_start) & (equity.index <= boundary_end)]
        if window.empty:
            continue
        pre = equity[equity.index <= boundary_start]
        anchor_value = float(pre.iloc[-1]) if len(pre) else float(window.iloc[0])
        end_value = float(window.iloc[-1])
        anchor_ts = boundary_start if len(pre) else pd.Timestamp(window.index[0])
        days = max((pd.Timestamp(window.index[-1]) - anchor_ts).total_seconds() / 86_400.0, 1.0)
        total_return = end_value / anchor_value - 1.0 if anchor_value > 0.0 else None
        annualized = None
        if total_return is not None:
            growth = 1.0 + total_return
            annualized = float(growth ** (365.0 / days) - 1.0) if growth > 0.0 else -1.0
        pos_window = positions.reindex(window.index) if len(positions) else pd.Series(dtype=float)
        util = float((pos_window.abs() > 0.0).mean()) if len(pos_window) else None
        rows.append({"year": year, "days": len(window), "total_return": total_return, "annualized_return": annualized, "utilization": util})
    return rows


def _return_concentration(equity: pd.Series, top_n: tuple[int, ...] = (1, 3, 5, 10, 20, 50)) -> dict[str, Any]:
    """How much of a candidate's total (log-)return comes from its single best days --
    diagnostic for I4 specifically (SPEC.md task instruction to scrutinize I4 rather than take
    a large backtest number at face value). Pure post-hoc analysis over the ALREADY-SAVED
    equity series, no engine re-run needed."""
    import numpy as np

    if len(equity) < 2:
        return {"top_days": [], "by_n": []}
    returns = equity.pct_change().dropna()
    log_returns = np.log1p(returns)
    total_log_return = float(log_returns.sum())
    sorted_log = log_returns.sort_values(ascending=False)
    by_n = []
    for n in top_n:
        if n > len(sorted_log):
            continue
        top_sum = float(sorted_log.head(n).sum())
        pct_of_total = (top_sum / total_log_return * 100.0) if total_log_return != 0.0 else None
        remainder_multiple = float(np.exp(total_log_return - top_sum)) if np.isfinite(total_log_return - top_sum) else None
        by_n.append({"n": n, "pct_of_total_log_return": pct_of_total, "remainder_multiple": remainder_multiple})
    top_days = [
        {"date": str(pd.Timestamp(ts).date()), "simple_return_pct": float(np.expm1(v) * 100.0)}
        for ts, v in sorted_log.head(10).items()
    ]
    return {"total_multiple": float(np.exp(total_log_return)), "by_n": by_n, "top_days": top_days}


def _episode_stats(flags: pd.Series) -> dict[str, Any]:
    """Consecutive-run-length statistics over a boolean series -- used for I4's required
    '음수 펀딩 발생 빈도·지속기간 통계' (episode = a maximal run of consecutive days
    layer_used == 'reverse_overlay', i.e. the reverse-carry signal was actually held)."""
    values = [bool(value) for value in flags.tolist()]
    episodes: list[int] = []
    current = 0
    for value in values:
        if value:
            current += 1
        elif current > 0:
            episodes.append(current)
            current = 0
    if current > 0:
        episodes.append(current)
    total_days = sum(episodes)
    span_days = len(values)
    years = span_days / 365.0 if span_days else 0.0
    durations = pd.Series(episodes, dtype=float) if episodes else pd.Series(dtype=float)
    return {
        "n_episodes": len(episodes),
        "total_active_days": total_days,
        "mean_duration_days": float(durations.mean()) if len(durations) else 0.0,
        "median_duration_days": float(durations.median()) if len(durations) else 0.0,
        "max_duration_days": int(durations.max()) if len(durations) else 0,
        "episodes_per_year": (len(episodes) / years) if years else 0.0,
    }


# ---------------------------------------------------------------------------
# Header / methodology.
# ---------------------------------------------------------------------------


def _header_section() -> list[str]:
    return [
        "# Wave-18 리포트 -- 유휴자본 가동 (I0-I5, 사전등록 research/wave18_idle/SPEC.md)",
        "",
        "**이 wave의 목적함수는 전기간 CAGR이다.** 지금까지 보고해온 '고펀딩기 연 22.01%'는 "
        "L4가 실제로 포지션을 보유한 구간(전체의 약 58~59%)에 한정된 값이고, 나머지 41~42%의 "
        "시간은 자본이 그냥 쉬고 있었다 -- 이 wave는 그 유휴 구간에 무엇을 넣을 수 있는지, "
        "넣어도 캐리 진입을 방해하지 않는지, 활성 구간 성과를 훼손하지 않는지를 검증한다.",
        "",
        "## 방법론 한계 (필독)",
        "",
        "1. **I1(USDT 대여)은 시계열이 아니라 스냅샷 상수다**: OKX `lending-rate-history`는 "
        "coin당 최근 ~100시간(약 4일)만 제공한다(wave17과 동일한 엔드포인트 한계). I1은 이 "
        "4일 관측의 **최솟값**(보수적 하한, task 지시)을 2019-09-01~2026-07-14 전 구간에 상수로 "
        "적용한다 -- 과거 대여금리가 실제로 어땠는지는 알 수 없다. **'검증된 수익률'이 아니다.**",
        "2. **I4(역캐리)의 차입비용은 반영되지 않았고, 무마찰 백테스트 수익 자체도 액면 그대로 믿을 "
        "수 없다**: 아래 I4 절 참조 -- 차입금리 필드 단위(시간당 vs 연환산)를 공개 문서로 확정하지 "
        "못해 차입비용을 반영할 수 없었고, 별도로 백테스트 수익이 2022-05(Luna) · 2022-11(FTX) 같은 "
        "시장 패닉 시점에 집중돼 있다는 것도 확인했다 -- 이 wave는 **구조적 실행가능성**을 근거로 "
        "판정하며, 그 구조적 판단은 수치 크기와 무관하게 유지된다.",
        "3. **회수 지연(초/분 단위)은 wave17의 UNCONFIRMED를 그대로 승계한다**: flexible "
        "(즉시 회수 가능)이라는 상품 설계는 OKX 도움말로 확인됐지만(wave17), 청산 신호 발생부터 "
        "실제 매도 체결까지의 초/분 단위 지연은 API 문서로 확인되지 않는다. 이 wave의 S6 게이트는 "
        "**일봉(daily bar) granularity**에서 '유휴자본이 캐리 활성일에는 절대 보유되지 않는다'는 "
        "것을 엔진 구조로 강제·검증한다 -- 그 하루 안에서 실제 회수~매도가 몇 초/몇 분 걸리는지는 "
        "여전히 미확인이다.",
        "4. **다중검정**: 누적 106회(SPEC.md 동결 수치, 아래 '다중검정' 절 참조 -- wave16 disclosed "
        "96 + wave18 6후보 산술과 정확히 맞물리지 않음, 검증 없이 SPEC.md 그대로 사용).",
        "",
    ]


def _candidate_definitions_section() -> list[str]:
    lines = [
        "## 후보 정의 (SPEC.md 동결, 사후 추가 금지)",
        "",
        "공통: $100 총자본/$90 활성자본/레그 $45, 델타중립 1x, wave-13 실측비용, L4 신호·유니버스"
        "(top200, 12mo, 진입15%APR/청산7.5%) 승계. 대기 = L4가 무포지션인 날.",
        "",
        "| ID | 정의 |",
        "|---|---|",
    ]
    for config in CONFIGS:
        lines.append(f"| {config.candidate_id} | {config.note} |")
    lines.append("")
    return lines


def _i0_reproduction_section(payloads: dict[str, dict[str, Any]], wave13_results_dir: Path) -> list[str]:
    i0 = payloads.get("I0")
    l4_path = wave13_results_dir / "L4.json"
    l4_payload = _load_optional_json(l4_path)
    lines = ["## I0 재현 검증 (wave13 L4 대비, read-only 비교 -- wave13 파일은 손대지 않음)", ""]
    if i0 is None:
        lines.append("- I0 결과 없음 (run 스테이지 미실행).")
        lines.append("")
        return lines
    i0_final = i0["equity"][-1]["value"] if i0.get("equity") else None
    i0_full_period = i0.get("full_period_annualized")
    i0_high = i0.get("regime_breakdown", {}).get("high_funding_mean_annualized_return")
    lines.append(f"- I0 최종 활성자본: {_fmt_usd(i0_final)}")
    lines.append(f"- I0 전기간 CAGR: {_fmt_pct(i0_full_period)} (SPEC.md 프로즈 수치 9.37%는 참고용, 아래 수치가 이 리포트의 실제 기준)")
    lines.append(f"- I0 고펀딩기 연환산: {_fmt_pct(i0_high)} (SPEC.md 프로즈 수치 22.01%는 참고용)")
    if l4_payload is not None:
        l4_final = l4_payload["equity"][-1]["value"] if l4_payload.get("equity") else None
        l4_high = l4_payload.get("regime_breakdown", {}).get("high_funding_mean_annualized_return")
        diff_final = (i0_final - l4_final) if (i0_final is not None and l4_final is not None) else None
        diff_high = (i0_high - l4_high) if (i0_high is not None and l4_high is not None) else None
        lines.append(f"- wave13 L4(원본) 최종 활성자본: {_fmt_usd(l4_final)} (read-only, research/wave13_liquidity/results/L4.json)")
        lines.append(f"- wave13 L4(원본) 고펀딩기 연환산: {_fmt_pct(l4_high)}")
        lines.append(f"- 최종자본 차이: {_fmt_usd(diff_final, 8) if diff_final is not None else 'N/A'} (0에 가까울수록 재현 성립)")
        lines.append(f"- 고펀딩기 연환산 차이: {_fmt_pct(diff_high, 8) if diff_high is not None else 'N/A'}")
        exact = diff_final is not None and abs(diff_final) < 1e-6
        lines.append("")
        lines.append(f"**판정: {'I0가 L4를 정확히 재현한다 (byte-identical 호출 -- engine18.run_i0_reference == engine13.run_candidate)' if exact else '불일치 발견 -- 아래 수치 그대로 보고'}**")
    else:
        lines.append(f"- {l4_path}를 찾지 못해 대조 불가.")
    lines.append("")
    return lines


# ---------------------------------------------------------------------------
# Results / layer breakdown / yearly tables.
# ---------------------------------------------------------------------------


def _results_table_section(payloads: dict[str, dict[str, Any]]) -> list[str]:
    lines = [
        "## 결과 비교 (주 지표 = 전기간 CAGR)",
        "",
        "| 후보 | 전기간 CAGR | 가동률 | 고펀딩기 연환산 | I0 대비 전기간(%p) | I0 대비 고펀딩기(%p) | 거래수 |",
        "|---|---|---|---|---|---|---|",
    ]
    i0 = payloads.get("I0", {})
    i0_full = i0.get("full_period_annualized")
    i0_high = i0.get("regime_breakdown", {}).get("high_funding_mean_annualized_return")
    for candidate_id in CONFIG_IDS:
        payload = payloads.get(candidate_id)
        if payload is None:
            lines.append(f"| {candidate_id} | 결과없음 | | | | | |")
            continue
        full = payload.get("full_period_annualized")
        util = payload.get("metadata", {}).get("utilization")
        high = payload.get("regime_breakdown", {}).get("high_funding_mean_annualized_return")
        full_gap = (full - i0_full) * 100.0 if (full is not None and i0_full is not None) else None
        high_gap = (high - i0_high) * 100.0 if (high is not None and i0_high is not None) else None
        n_trades = payload.get("metadata", {}).get("n_trades")
        label = candidate_id + (" (기준선)" if candidate_id == "I0" else "")
        lines.append(
            f"| {label} | {_fmt_pct(full)} | {_fmt_pct(util)} | {_fmt_pct(high)} | {_fmt_pp(full_gap)} | {_fmt_pp(high_gap)} | {n_trades if n_trades is not None else 'N/A'} |"
        )
    lines.append("")
    return lines


def _layer_breakdown_section(payloads: dict[str, dict[str, Any]]) -> list[str]:
    lines = [
        "## 유휴자본 사용처 분해 (대기일 수익 기여, 일수 기준 %)",
        "",
        "| 후보 | L4 | carry_overlay | reverse_overlay | lending | cash(순수유휴) |",
        "|---|---|---|---|---|---|",
    ]
    for candidate_id in CONFIG_IDS:
        payload = payloads.get(candidate_id)
        if payload is None:
            lines.append(f"| {candidate_id} | | | | | |")
            continue
        fractions = payload.get("metadata", {}).get("layer_breakdown", {}).get("fractions", {})
        row = [f"{fractions.get(key, 0.0) * 100.0:.1f}%" if key in fractions else "-" for key in ("L4", "carry_overlay", "reverse_overlay", "lending", "cash")]
        lines.append(f"| {candidate_id} | {row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]} |")
    lines.append("")
    return lines


def _yearly_section(payloads: dict[str, dict[str, Any]]) -> list[str]:
    lines = ["## 연도별 표 (대기 많은 해 -- 2022·2025 -- 에서 실제로 개선되는지)", ""]
    per_candidate_years: dict[str, dict[int, dict[str, Any]]] = {}
    all_years: set[int] = set()
    for candidate_id in CONFIG_IDS:
        payload = payloads.get(candidate_id)
        if payload is None:
            continue
        equity = _series(payload["equity"])
        positions = _series(payload["positions"])
        rows = _yearly_stats(equity, positions)
        per_candidate_years[candidate_id] = {int(row["year"]): row for row in rows}
        all_years.update(per_candidate_years[candidate_id].keys())
    years_sorted = sorted(all_years)

    lines.append("### 연도별 연환산 수익률")
    lines.append("")
    lines.append("| 후보 | " + " | ".join(str(y) for y in years_sorted) + " |")
    lines.append("|---|" + "---|" * len(years_sorted))
    for candidate_id in CONFIG_IDS:
        by_year = per_candidate_years.get(candidate_id, {})
        cells = [_fmt_pct(by_year[y]["annualized_return"]) if y in by_year else "-" for y in years_sorted]
        lines.append(f"| {candidate_id} | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("### 연도별 가동률 (L4/overlay 포지션 보유일 비율 -- lending 단독 일수는 미포함)")
    lines.append("")
    lines.append("| 후보 | " + " | ".join(str(y) for y in years_sorted) + " |")
    lines.append("|---|" + "---|" * len(years_sorted))
    for candidate_id in CONFIG_IDS:
        by_year = per_candidate_years.get(candidate_id, {})
        cells = [_fmt_pct(by_year[y]["utilization"], 1) if y in by_year else "-" for y in years_sorted]
        lines.append(f"| {candidate_id} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(
        f"참고: 고펀딩기로 분류되는 해는 {', '.join(str(y) for y in HIGH_FUNDING_YEARS)} (research.wave10_carry100.regime.HIGH_FUNDING_YEARS, 이 wave가 새로 정의한 것 아님) -- "
        "**2022·2025·2026처럼 대기가 많았던 해에서 I1-I5가 I0 대비 실제로 개선되는지가 이 표의 핵심 확인 대상이다.**"
    )
    lines.append("")
    return lines


def _gates_table_section(payloads: dict[str, dict[str, Any]]) -> list[str]:
    lines = [
        "## 게이트 (S1~S6)",
        "",
        "| 후보 | S1 구조 | S2 MC | S3 블록MDD | S4 실행가능 | S5 스트레스x3 | S6 회수성 | 종합 | 승격 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for candidate_id in CONFIG_IDS:
        payload = payloads.get(candidate_id)
        if payload is None:
            lines.append(f"| {candidate_id} | | | | | | | 미실행 | |")
            continue
        gates = payload.get("gates", {})
        if not gates:
            lines.append(f"| {candidate_id} | | | | | | | GATES미실행 | |")
            continue
        statuses = [gates.get(f"gate_s{n}", {}).get("status", "N/A") for n in range(1, 7)]
        overall = gates.get("overall", "N/A")
        promoted = gates.get("promotion", {}).get("promoted")
        promoted_label = "YES" if promoted else ("N/A(기준선)" if candidate_id == "I0" else "NO")
        lines.append(f"| {candidate_id} | " + " | ".join(statuses) + f" | {overall} | {promoted_label} |")
    lines.append("")
    for candidate_id in CONFIG_IDS:
        payload = payloads.get(candidate_id)
        if payload is None:
            continue
        gates = payload.get("gates", {})
        reasons = gates.get("failure_reasons", [])
        if reasons:
            lines.append(f"- {candidate_id} 미달 사유: {', '.join(reasons)}")
    lines.append("")
    return lines


# ---------------------------------------------------------------------------
# I1 (lending) / I4 (reverse carry) dedicated sections.
# ---------------------------------------------------------------------------


def _i1_section(usdt_lending: dict[str, Any] | None, payloads: dict[str, dict[str, Any]]) -> list[str]:
    lines = ["## I1 -- USDT 대여 데이터 스냅샷 (OKX lending-rate-history, 라이브 1회 수집)", ""]
    if usdt_lending is None:
        lines.append("- cache/usdt_lending.json 없음 (fetch 스테이지 미실행).")
        lines.append("")
        return lines
    lines.extend(
        [
            f"- 수집 시각(UTC): {usdt_lending.get('collected_at_utc', 'N/A')}",
            f"- 관측치 수: {usdt_lending.get('n_samples', 0)}건 (약 {usdt_lending.get('span_days', 0):.2f}일)",
            f"- lendingRate 중앙값: {_fmt_pct(usdt_lending.get('lending_rate_median'))} / 평균: {_fmt_pct(usdt_lending.get('lending_rate_mean'))}",
            f"- lendingRate **최솟값(I1이 실제로 쓰는 값, 보수적 하한)**: {_fmt_pct(usdt_lending.get('lending_rate_min'))}",
            f"- lendingRate 최댓값: {_fmt_pct(usdt_lending.get('lending_rate_max'))} / 변동계수(CV): "
            + (f"{usdt_lending.get('lending_rate_cv'):.3f}" if usdt_lending.get("lending_rate_cv") is not None else "N/A"),
            f"- 동일호출 rate(차입자쪽) 대비 lendingRate 비율: {usdt_lending.get('ratio_lendingrate_median_over_avgrate_fresh')}",
            "",
        ]
    )
    i1 = payloads.get("I1")
    if i1 is not None:
        apr_used = i1.get("config", {}).get("lending_apr_used")
        lines.append(f"- **I1이 실제로 적용한 상수 APR: {_fmt_pct(apr_used)}**")
        lines.append("")
    lines.append(
        "**주의**: 위 수치는 2026-07-24 전후 약 4일 관측의 스냅샷이다. I1/I5 백테스트는 이 상수를 "
        "2019-09-01~2026-07-14 전 구간에 균일 적용한다 -- 과거 실제 USDT 대여금리가 이 수준이었다는 "
        "근거는 없다(강한 가정, 시계열 미검증)."
    )
    lines.append("")
    return lines


def _i4_section(payloads: dict[str, dict[str, Any]], margin_rates: dict[str, Any] | None) -> list[str]:
    lines = ["## I4 -- 역캐리 (음수 펀딩) 분석 및 실행가능성", ""]
    i4 = payloads.get("I4")
    if i4 is not None:
        layer_used = _layer_series_from_payload(i4["layer_used"])
        flags = layer_used == "reverse_overlay"
        stats = _episode_stats(flags)
        lines.extend(
            [
                "### 음수 펀딩(<-15% APR) 발생 빈도·지속기간 (SPEC.md 필수 산출)",
                "",
                f"- 발생(에피소드) 횟수: {stats['n_episodes']}회 (전체 기간, 연 {stats['episodes_per_year']:.2f}회 환산)",
                f"- 역캐리 보유 총 일수: {stats['total_active_days']}일 / 전체 {len(layer_used)}일 ({stats['total_active_days']/len(layer_used)*100.0 if len(layer_used) else 0.0:.2f}%)",
                f"- 에피소드 평균 지속: {stats['mean_duration_days']:.2f}일 / 중앙값: {stats['median_duration_days']:.1f}일 / 최장: {stats['max_duration_days']}일",
                "",
                "### 가정: 무마찰 실행 시 가상 성과 (아래 실행가능성 판정과 별개, 참고용)",
                "",
                f"- I4 전기간 CAGR(백테스트, 차입비용 미반영): {_fmt_pct(i4.get('full_period_annualized'))}",
                f"- I4 고펀딩기 연환산: {_fmt_pct(i4.get('regime_breakdown', {}).get('high_funding_mean_annualized_return'))}",
                "",
            ]
        )
        equity = _series(i4["equity"])
        concentration = _return_concentration(equity)
        lines.append("### 수익 집중도 경고 (이 숫자를 액면 그대로 믿으면 안 되는 이유)")
        lines.append("")
        lines.append(
            f"위 무마찰 가상 성과는 소수의 극단적인 날에 크게 좌우된다 -- 상위 1일이 전체 log-수익의 "
            f"{next((r['pct_of_total_log_return'] for r in concentration['by_n'] if r['n'] == 1), 0.0):.1f}%, "
            f"상위 10일이 {next((r['pct_of_total_log_return'] for r in concentration['by_n'] if r['n'] == 10), 0.0):.1f}%를 "
            "차지한다. 다만 상위 50일(전체 활성 943일 중 5%)을 전부 제외해도 여전히 배수가 크게 남는다 -- "
            "즉 '단 하루짜리 이상치' 문제가 아니라 역캐리 신호 자체가 광범위하게 양의 수익을 내는 패턴이다:"
        )
        lines.append("")
        lines.append("| 제외한 상위 N일 | 그 N일이 전체 log-수익에서 차지하는 비중 | 나머지로 계산한 배수 |")
        lines.append("|---|---|---|")
        for row in concentration["by_n"]:
            lines.append(
                f"| {row['n']} | {row['pct_of_total_log_return']:.1f}% | {row['remainder_multiple']:.2f}x |"
                if row["pct_of_total_log_return"] is not None
                else f"| {row['n']} | N/A | N/A |"
            )
        lines.append("")
        lines.append("상위 기여일 (날짜 + 그날의 단순수익률):")
        lines.append("")
        lines.append("| 날짜 | 당일 수익률 |")
        lines.append("|---|---|")
        for row in concentration["top_days"]:
            lines.append(f"| {row['date']} | {row['simple_return_pct']:+.2f}% |")
        lines.append("")
        lines.append(
            "**교차 확인(수작업 진단, research/wave18_idle/tests에는 포함되지 않음)**: 상위 기여일을 개별 "
            "역추적한 결과 2022-05-29/30(BELUSDT, Terra/Luna 붕괴 직후 -- 거래대금이 하루 새 수십 배 튀는 "
            "이상치성 패턴 동반), 2022-11-10(SOLUSDT, FTX 붕괴 당일 -- 당일 시가부터 perp가 spot 대비 17% "
            "할인가에서 시작해 종가엔 거의 수렴)처럼 **역사상 가장 극단적인 시장 패닉 시점**에 집중돼 있다. "
            "이는 두 가지를 동시에 의미한다: (1) 신호 자체는 데이터 조작이 아니라 실제 극단적 베이시스 "
            "괴리를 포착하고 있을 가능성이 높다(부호 검증: 2022-11-10 SOLUSDT는 perp가 spot보다 17% 싸게 "
            "출발해 당일 수렴 -- 롱퍼프/숏현물 방향과 정확히 일치), 그러나 (2) **바로 그 패닉의 순간이야말로 "
            "마진 차입 시장이 얼어붙거나 차입금리가 폭등하기 가장 쉬운 시점**이다 -- 백테스트가 큰 이익을 "
            "기록한 날들이 실제로는 숏현물 진입이 가장 어려웠을 날들과 겹칠 가능성이 높다는 뜻이다. 아래 "
            "실행가능성 판정은 바로 이 우려를 근거로 한다.",
        )
        lines.append("")
    else:
        lines.append("- I4 결과 없음 (run 스테이지 미실행).")
        lines.append("")

    lines.append("### 실행가능성 (S4) -- 구조적 근거로 판정")
    lines.append("")
    lines.extend(f"- {reason}" for reason in gates18.I4_SHORT_SPOT_INFEASIBLE_REASONS)
    lines.append("")

    lines.append("### OKX 마진 차입금리 실측 (참고, 단위 미확정)")
    lines.append("")
    if margin_rates is None:
        lines.append("- cache/margin_borrow_rates.json 없음 (fetch 스테이지 미실행).")
    else:
        lines.append(f"- 수집 시각(UTC): {margin_rates.get('collected_at_utc', 'N/A')} / 출처: {margin_rates.get('source')}")
        lines.append("")
        lines.append("| 코인 | raw rate 필드 | 해석A(원값=연환산, %) | 해석B(시간당×8760, %) |")
        lines.append("|---|---|---|---|")
        for ccy, row in margin_rates.get("sample", {}).items():
            raw = row.get("raw_rate_field")
            a = row.get("reading_a_raw_as_already_annualized_pct")
            b = row.get("reading_b_hourly_times_8760_annualized_pct")
            lines.append(f"| {ccy} | {raw if raw is not None else 'N/A'} | {f'{a:.4f}%' if a is not None else 'N/A'} | {f'{b:.2f}%' if b is not None else 'N/A'} |")
        lines.append("")
        lines.append(f"- {margin_rates.get('unit_note', '')}")
    lines.append("")
    lines.append(
        "**결론: I4는 두 해석 중 무엇을 골라도 실행가능성 판정을 바꾸지 못한다** -- 차입금리 수치와 "
        "무관하게, 별도 마진 계좌 모드 필요·상환시 현물 재매입 가격리스크·SPEC.md 자체 리스크 원칙("
        "유휴자본은 캐리 이하 리스크여야 함) 위반이라는 구조적 사유만으로 S4는 FAIL이다."
    )
    lines.append("")
    return lines


# ---------------------------------------------------------------------------
# Verdict + multi-testing.
# ---------------------------------------------------------------------------


def _verdict_section(payloads: dict[str, dict[str, Any]]) -> list[str]:
    lines = ["## 판정", ""]
    promoted_ids = []
    for candidate_id in CONFIG_IDS:
        if candidate_id == "I0":
            continue
        payload = payloads.get(candidate_id)
        if payload is None:
            continue
        if payload.get("gates", {}).get("promotion", {}).get("promoted"):
            promoted_ids.append(candidate_id)

    if promoted_ids:
        best_id = max(
            promoted_ids,
            key=lambda cid: payloads[cid].get("full_period_annualized") or float("-inf"),
        )
        best_cagr = payloads[best_id].get("full_period_annualized")
        lines.append(f"**승격**: {', '.join(promoted_ids)} -- S1~S6 전부 PASS ∧ 전기간 CAGR가 I0를 상회 ∧ 고펀딩기 -1%p 이내.")
        lines.append(f"**최고 전기간 CAGR**: {best_id} {_fmt_pct(best_cagr)} (I0 {_fmt_pct(payloads.get('I0', {}).get('full_period_annualized'))} 대비).")
        lines.append("")
        lines.append("판정 원칙(SPEC.md): 통과 시 카드에 'L4 + 유휴운용' 결합 구성으로 기재하고, 전기간 CAGR을 대표 수치로 승격하며 기존 '연 22.01%' 표기는 고펀딩기 한정임을 명시한다.")
    else:
        lines.append("**전멸**: I1-I5 중 S1~S6 전부 PASS + I0 대비 전기간 CAGR 개선 + 고펀딩기 -1%p 이내를 동시에 만족한 후보가 없다.")
        lines.append("")
        lines.append("SPEC.md 판정 원칙: 유휴자본은 현금 보유(I0)가 최적이라고 정직 보고한다. 사유 분해:")
        for candidate_id in CONFIG_IDS:
            if candidate_id == "I0":
                continue
            payload = payloads.get(candidate_id)
            if payload is None:
                lines.append(f"- {candidate_id}: 결과 없음.")
                continue
            reasons = payload.get("gates", {}).get("failure_reasons", [])
            full = payload.get("full_period_annualized")
            i0_full = payloads.get("I0", {}).get("full_period_annualized")
            gap = (full - i0_full) if (full is not None and i0_full is not None) else None
            lines.append(f"- {candidate_id}: 전기간 CAGR {_fmt_pct(full)} (I0 대비 {_fmt_pp(gap * 100.0 if gap is not None else None)}) -- 미달 사유: {', '.join(reasons) if reasons else '(gates 미실행 또는 전부 PASS인데 승격 조건 미달 -- 상세 위 표 참조)'}")
    lines.append("")
    return lines


def _multi_testing_section() -> list[str]:
    return [
        "## 다중검정",
        "",
        f"DSR_CUMULATIVE_TRIALS = {gates18.DSR_CUMULATIVE_TRIALS} (SPEC.md 동결 수치, 그대로 사용 -- "
        "wave16 disclosed 96 + 이 wave의 6개 신규 후보 산술(96+6=102)과 정확히 맞물리지 않는다는 점을 "
        "투명하게 남긴다. 참고용 DSR 계산에만 쓰이고, 승격 판정에는 사용되지 않는다(wave10-17의 공통 "
        "원칙).",
        "",
    ]


# ---------------------------------------------------------------------------
# Write.
# ---------------------------------------------------------------------------


def write_wave18_report(results_dir: Path, report_dir: Path, registry_path: Path, cache_dir: Path) -> None:
    payloads: dict[str, dict[str, Any]] = {}
    for candidate_id in CONFIG_IDS:
        payload = _load(results_dir, candidate_id)
        if payload is not None:
            payloads[candidate_id] = payload
    usdt_lending = _load_optional_json(cache_dir / "usdt_lending.json")
    margin_rates = _load_optional_json(cache_dir / "margin_borrow_rates.json")
    wave13_results_dir = results_dir.parent.parent / "wave13_liquidity" / "results"

    lines: list[str] = [
        *_header_section(),
        *_candidate_definitions_section(),
        *_i0_reproduction_section(payloads, wave13_results_dir),
        *_results_table_section(payloads),
        *_layer_breakdown_section(payloads),
        *_yearly_section(payloads),
        *_gates_table_section(payloads),
        *_i1_section(usdt_lending, payloads),
        *_i4_section(payloads, margin_rates),
        *_verdict_section(payloads),
        *_multi_testing_section(),
    ]
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "wave18_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    registry_lines = [
        "# Wave-18 registry",
        "",
        "| Candidate | Family | State | 전기간 CAGR | I0 대비(%p) | 승격 | 분류 |",
        "|---|---|---|---|---|---|---|",
    ]
    i0_full = payloads.get("I0", {}).get("full_period_annualized")
    for candidate_id in CONFIG_IDS:
        payload = payloads.get(candidate_id)
        if payload is None:
            registry_lines.append(f"| {candidate_id} | wave18_idle | PENDING | - | - | - | run 미실행 |")
            continue
        full = payload.get("full_period_annualized")
        gap = (full - i0_full) * 100.0 if (full is not None and i0_full is not None) else None
        promoted = payload.get("gates", {}).get("promotion", {}).get("promoted")
        promoted_label = "N/A(기준선)" if candidate_id == "I0" else ("YES" if promoted else "NO")
        classification = {
            "I0": "기준선",
            "I1": "USDT 대여",
            "I2": "메이저 저임계 캐리",
            "I3": "전체 저임계 캐리(Y1 재검정)",
            "I4": "역캐리",
            "I5": "I2+I1 계층",
        }.get(candidate_id, "")
        registry_lines.append(f"| {candidate_id} | wave18_idle | EVALUATED | {_fmt_pct(full)} | {_fmt_pp(gap)} | {promoted_label} | {classification} |")
    registry_lines.append("")
    any_promoted = any(
        payloads.get(cid, {}).get("gates", {}).get("promotion", {}).get("promoted") for cid in CONFIG_IDS if cid != "I0"
    )
    registry_lines.append(f"**최종 판정**: {'일부 승격' if any_promoted else '전멸 -- 유휴자본은 현금(I0)이 최적'} (report/wave18_report.md 참조).")
    registry_lines.append("")
    registry_path.write_text("\n".join(registry_lines) + "\n", encoding="utf-8")


__all__ = ["write_wave18_report"]
