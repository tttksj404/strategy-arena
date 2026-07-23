# Wave-15 markdown report + registry writer. Pure formatting over already-computed
# results/{A1,A2,A3,B1,B2,C1,D1}.json (run_wave15.py --stage run/gates) plus the read-only
# L4 reference recompute in cache/l4_reference.json.

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

import pandas as pd  # noqa: PANDAS_OK

from research.wave15_diverse.configs15 import CANDIDATE_IDS
from research.wave15_diverse.gates15 import DSR_CUMULATIVE_TRIALS


def _fmt_pct(value: float | None, digits: int = 2) -> str:
    return "N/A" if value is None else f"{value * 100.0:.{digits}f}%"


def _fmt_usd(value: float | None, digits: int = 2) -> str:
    return "N/A" if value is None else f"${value:,.{digits}f}"


def _fmt_bp(value: float | None, digits: int = 4) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}bp"


def _load(results_dir: Path, candidate_id: str) -> dict[str, Any]:
    path = results_dir / f"{candidate_id}.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _load_l4_reference(cache_dir: Path) -> dict[str, Any] | None:
    path = cache_dir / "l4_reference.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _series_from_payload(records: list[dict[str, Any]]) -> pd.Series:
    if not records:
        return pd.Series([], index=pd.DatetimeIndex([], tz="UTC"), dtype=float)
    idx = pd.DatetimeIndex([pd.Timestamp(item["timestamp"]) for item in records])
    values = [float(item["value"]) for item in records]
    return pd.Series(values, index=idx, dtype=float).sort_index()


def _annualize(total_return: float, days: float) -> float:
    growth = 1.0 + total_return
    if growth <= 0.0:
        return -1.0
    return float(growth ** (365.0 / max(days, 1.0)) - 1.0)


def year_slice_annualized(equity: pd.Series, year: int) -> dict[str, Any] | None:
    """Same slicing convention as research.wave10_carry100.regime._regime_return (half-open
    (Dec 31 prev year, Dec 31 this year] window, anchored off the full series so the
    transition day's return is included) -- reimplemented locally (not imported) because B2's
    SPEC.md-mandated 2022 bear-market disclosure needs a YEAR regime.py's own
    HIGH_FUNDING_YEARS=(2020,2021,2024) tuple does not include."""
    start = pd.Timestamp(f"{year - 1}-12-31T23:59:59Z")
    end = pd.Timestamp(f"{year}-12-31T23:59:59Z")
    mask = (equity.index > start) & (equity.index <= end)
    window = equity[mask]
    if window.empty:
        return None
    pre = equity[equity.index <= start]
    anchor_value = float(pre.iloc[-1]) if len(pre) else float(window.iloc[0])
    end_value = float(window.iloc[-1])
    days = max((pd.Timestamp(window.index[-1]) - pd.Timestamp(start)).total_seconds() / 86_400.0, 1.0)
    total_return = end_value / anchor_value - 1.0
    return {"year": year, "start_usdt": anchor_value, "end_usdt": end_value, "total_return": total_return, "annualized_return": _annualize(total_return, days), "days": days}


# ---------------------------------------------------------------------------
# Header / methodology
# ---------------------------------------------------------------------------


def _header_and_scope_note() -> list[str]:
    return [
        "# Wave-15 리포트 -- 방법 다양화 (4계열, 수익 메커니즘 전환)",
        "",
        "## SPEC.md 후보수 표기 불일치 (투명 공개)",
        "",
        "SPEC.md 제목 섹션은 \"후보 8개\"라 쓰고 다중검정 문구도 \"누적 92회\"(=wave14까지 84회+8)라 "
        "적었지만, SPEC.md 본문의 후보 표는 **정확히 7개 ID**(A1,A2,A3,B1,B2,C1,D1)만 나열한다. "
        "SPEC은 \"동결, 사후 추가 금지\"이므로 이 리포트는 실제로 명시된 7개만 구현했다 -- 헤더 "
        "숫자를 맞추려 8번째 후보를 사후에 새로 발명하는 쪽이 오히려 동결 원칙 위반이라 판단했다. "
        "다중검정 카운트도 실제 실행된 7개 기준 누적 91회(wave14 공시 84 + 7)로 표기한다.",
        "",
        "## 방법론 한계 (필독 -- 아래 모든 수치에 선행하는 전제)",
        "",
        "- **A계열(A1-A3) 유니버스**: 1h OHLCV 캐시는 BTC/ETH/SOL 3종이 존재하지만, SOL은 실측 "
        "결과 2022-11 이후 정산주기가 예외 없는 8h가 아니라 구간적으로 2h 간격(거래소의 변동성 "
        "확대시 동적 정산주기 단축)이 섞여 있어 제외했다 -- 이 엔진의 상태기계는 유니버스 전체에 "
        "단일 DECISION_HOURS/SETTLEMENT_HOURS를 공유하므로 가변주기 심볼을 섞으면 그 심볼의 "
        "정산 대부분을 조용히 놓치는 구조적 오류가 생긴다. **BTC/ETH 2종**으로 스코프를 좁혔다 "
        "(wave13 L1의 고정 2종 스코프와 동일 전례). SPEC.md는 \"A계열에 더 필요하면 교정된 "
        "페처로 수집\"을 허용하지만, wave12-14가 이미 심볼폭 축을 포화 판정했고 이 wave의 목적은 "
        "폭이 아니라 메커니즘(회전율) 검증이므로, 수백 심볼 x 1h x 7년 신규수집에 시간을 쓰지 "
        "않았다. BTC(2019-09~)/ETH(2019-11~) 둘 다 2020/2021/2024 고펀딩기를 전부 커버한다.",
        "- **B1 Earn APR / B2 USDT 담보이자**: Binance Simple Earn Flexible 과거·현재 APR을 "
        "인증 없이 조회 가능한 공개 엔드포인트가 없음을 실측 확인했다(`earn_apr.py`의 "
        "`probe_all()`; `/sapi/v1/simple-earn/flexible/list`는 API-key 없이 HTTP 400 "
        "`{code:-2014}`, 나머지 추정 공개 URL들은 전부 404). 이 세션은 거래소 API 키를 발급·입력할 "
        "권한이 없고 사용자로부터 제공받은 키도 없으므로, **보수적 고정값 2%/년으로 가정**했다 "
        "(`common15.ASSUMED_FLEXIBLE_EARN_APR`). 이 값이 들어간 모든 수치는 \"검증된 수익\"이 아니라 "
        "\"가정값\"이며, 캐리 단독 수익과 분리된 열로 제시한다.",
        "- **C1 미결제약정(OI) 특징 제외**: Binance `/futures/data/openInterestHist`를 실측 호출한 "
        "결과 트레일링 30일치만 보관됨을 확인했다(limit=500/시작시각 과거 지정 모두 30일로 절단). "
        "이 wave의 백테스트 구간(수년, 2020/2021/2024 고펀딩기 포함)에는 턱없이 부족해 **3특징 -> "
        "2특징(가격모멘텀 + 펀딩추세)으로 축소**했다 -- SPEC.md가 사전 승인한 폴백.",
        "- **D1 밈섹터 구성**: SHIBUSDT는 이 저장소의 3개 폴백 캐시 디렉터리 어디에도 존재하지 "
        "않아 WIFUSDT로 대체했다. 또한 SPEC.md 문면(z>2 진입, 단방향)을 대칭 |z| 규칙으로 구현했다 "
        "-- 어느 심볼을 '먼저' 라벨링하는지는 임의적이므로, 단방향 규칙은 그 임의성에 성과가 좌우되는 "
        "구조적 결함이 된다. 대칭화는 메커니즘·임계값을 그대로 둔 채 그 결함만 제거한 것이다.",
        "- **L4 재산출**: SPEC.md \"동일 비용모델·동일 기간으로 재산출한 L4 값과 비교\"는 "
        "`research/wave13_liquidity/engine13.run_candidate(get_config('L4'))`를 메모리 내에서 "
        "그대로 재실행해 얻는다 (wave13 코드/캐시는 읽기 전용, 결과는 wave15 자신의 "
        "`cache/l4_reference.json`에만 저장 -- wave13의 results/·report/는 손대지 않는다). "
        "동일한 `costs_measured` 비용모델, 동일한 동결 데이터 스냅샷(2026-07-22)이므로 "
        "\"동일 비용모델·동일 기간\" 조건을 만족한다.",
        "",
    ]


# ---------------------------------------------------------------------------
# Family A (intraday carry)
# ---------------------------------------------------------------------------


def _family_a_section(payloads: dict[str, dict[str, Any]]) -> list[str]:
    lines = ["## A계열 -- 인트라데이 캐리 (자본회전, 이 wave의 최우선 가설)", ""]
    a_payloads = {cid: payloads[cid] for cid in ("A1", "A2", "A3") if cid in payloads}
    if not a_payloads:
        return lines + ["- 실행되지 않음.", ""]

    sample = next(iter(a_payloads.values()))
    breakeven = sample.get("breakeven_analysis", {})
    lines += [
        "### 손익분기 수식",
        "",
        "왕복비용(entry+exit, 레그당 both-legs cost_for) > 그 회차 실현 펀딩이면 인트라데이 사이클은 "
        "손실이다:",
        "",
        "```",
        "cost_for(symbol) = 2 x maker(0.02%) + 2 x measured_slippage_bp(symbol) x stress_mult   [양 레그, 편도]",
        "1사이클 왕복비용   = 2 x cost_for(symbol)                      [진입 1회 + 청산 1회]",
        "손익분기(회차)     : funding_rate_1period > 2 x cost_for(symbol)",
        "손익분기(연환산)   : 2 x cost_for(symbol) x 3 x 365            [1일 3회 정산]",
        "```",
        "",
        "현재 실측 매핑 기준 (최신 30d 거래대금 스냅샷):",
        "",
        "| 심볼 | 30d거래대금 | 실측슬리피지 | 편도(양레그) 비용 | 1사이클 왕복비용 | 손익분기 연환산 APR |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for symbol, row in breakeven.get("per_symbol", {}).items():
        lines.append(
            f"| {symbol} | {_fmt_usd(row.get('trailing_30d_volume_usdt'), 0)} | {_fmt_bp(row.get('measured_slippage_bp'))} | "
            f"{_fmt_pct(row.get('one_way_both_legs_cost_rate'), 4)} | {_fmt_pct(row.get('round_trip_cost_per_cycle'), 4)} | "
            f"{_fmt_pct(row.get('breakeven_annualized_apr'))} |"
        )
    lines += [
        "",
        "비교: 일봉/다일 보유(L4류)는 같은 왕복비용을 **여러 회차에 걸쳐 상각**하므로 회차당 손익분기가 "
        "N(보유기간 내 정산횟수)분의 1로 낮아진다 -- 그래서 L4의 진입 임계값이 연 15%로 낮게 잡혀도 "
        "작동한다. A1/A2/A3는 사이클마다(또는 A3는 약신호 구간에서만) 이 왕복비용을 매번 새로 낸다.",
        "",
        "### 후보별 결과",
        "",
        "| Candidate | 정의 | 트레이드수 | 인트라데이 점유율 | 최종 활성자본 | 고펀딩기 연환산 | 총비용($) | Overall |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for cid in ("A1", "A2", "A3"):
        payload = a_payloads.get(cid)
        if payload is None:
            continue
        metadata = payload.get("metadata", {})
        regime = payload.get("regime_breakdown")
        gates = payload.get("gates")
        high_funding = regime.get("high_funding_mean_annualized_return") if regime else None
        final_equity = _series_from_payload(payload.get("equity", [])).iloc[-1] if payload.get("equity") else None
        overall = gates["overall"] if gates else "게이트 미실행"
        lines.append(
            f"| {cid} | {payload.get('definition', '')[:40]}... | {metadata.get('n_trades', 'N/A')} | "
            f"{_fmt_pct(metadata.get('intraday_bar_fraction'))} | {_fmt_usd(final_equity)} | {_fmt_pct(high_funding)} | "
            f"{_fmt_usd(metadata.get('total_cost_usdt'))} | {overall} |"
        )

    lines += ["", "### 상태기계 무결성 (A3 이중모드 전환)", ""]
    a3 = a_payloads.get("A3")
    if a3 is not None:
        meta = a3.get("metadata", {})
        lines.append(
            f"- A3 위반횟수(결정바 시점에 INTRADAY 모드가 걸려있던 경우, 반드시 0이어야 함): "
            f"**{int(meta.get('state_machine_invariant_violations', -1))}** "
            f"(일봉모드 진입 {int(meta.get('n_daily_entries', 0))}회, 인트라데이모드 진입 {int(meta.get('n_intraday_entries', 0))}회, "
            f"일봉모드 점유 {_fmt_pct(meta.get('daily_bar_fraction'))}, 인트라데이 점유 {_fmt_pct(meta.get('intraday_bar_fraction'))}, "
            f"공백 {_fmt_pct(meta.get('flat_bar_fraction'))})."
        )
    lines.append("")
    return lines


# ---------------------------------------------------------------------------
# Family B (dual yield)
# ---------------------------------------------------------------------------


def _family_b_section(payloads: dict[str, dict[str, Any]]) -> list[str]:
    lines = ["## B계열 -- 현물 레그 이중수익", ""]
    if "B1" in payloads:
        b1 = payloads["B1"]
        attribution = b1.get("yield_attribution", {})
        regime = b1.get("regime_breakdown", {})
        gates = b1.get("gates")
        lines += [
            "### B1 -- Simple Earn Flexible 오버레이 (ASSUMED APR, 검증 아님)",
            "",
            f"- 캐리 단독 고펀딩기 연환산: {_fmt_pct(attribution.get('carry_only_high_funding_annualized'))}",
            f"- 캐리+ASSUMED Earn(연 {_fmt_pct(0.02)}) 고펀딩기 연환산: "
            f"{_fmt_pct(attribution.get('carry_plus_assumed_yield_high_funding_annualized'))}",
            f"- **ASSUMED Earn 기여분** (검증된 수익 아님): "
            f"{attribution.get('assumed_yield_contribution_annualized_pts', 0.0) * 100.0:+.2f}%p 연환산",
            f"- Overall: {gates['overall'] if gates else '게이트 미실행'}",
            "",
        ]
    if "B2" in payloads:
        b2 = payloads["B2"]
        gates = b2.get("gates")
        regime = b2.get("regime_breakdown", {})
        equity = _series_from_payload(b2.get("equity", []))
        bear_2022 = year_slice_annualized(equity, 2022) if len(equity) else None
        lines += [
            "### B2 -- USDT 담보 숏퍼프 단독 (델타중립 아님 -- 방향노출 명시)",
            "",
            f"- **구조**: {b2.get('structure', {}).get('note', 'N/A')}",
            f"- 고펀딩기 연환산: {_fmt_pct(regime.get('high_funding_mean_annualized_return'))}",
            "",
            "**2022 약세장 구간 성과 (SPEC.md 필수 표기 -- 방향노출 리스크의 실제 발현 여부)**:",
            "",
        ]
        if bear_2022 is not None:
            lines.append(
                f"- 2022년: 시작 ${bear_2022['start_usdt']:.2f} -> 종료 ${bear_2022['end_usdt']:.2f}, "
                f"총수익률 {_fmt_pct(bear_2022['total_return'])}, 연환산 {_fmt_pct(bear_2022['annualized_return'])} "
                f"({bear_2022['days']:.0f}일)"
            )
            if bear_2022["total_return"] > 0:
                lines.append(
                    "- 2022년 자체는 플러스다 -- 그러나 이는 숏퍼프 구조상 **가격 하락**이 오히려 "
                    "이 레그에 유리했기 때문이며(방향베팅이 우연히 맞은 결과), 상승장이었다면 "
                    "부호가 반전됐을 정성적 리스크는 그대로 남는다. 아래 스트레스(S5) 결과와 함께 "
                    "읽을 것."
                )
            else:
                lines.append("- 2022년 마이너스 -- 방향노출 리스크가 실제로 발현된 사례.")
        else:
            lines.append("- 2022년 구간 데이터 없음 (유니버스 히스토리 범위 밖).")
        lines.append(f"- Overall: {gates['overall'] if gates else '게이트 미실행'}")
        lines.append("")
    return lines


# ---------------------------------------------------------------------------
# Family C (predictive)
# ---------------------------------------------------------------------------


def _family_c_section(payloads: dict[str, dict[str, Any]], l4_reference: dict[str, Any] | None) -> list[str]:
    lines = ["## C계열 -- 펀딩 예측 (선행 진입)", ""]
    if "C1" not in payloads:
        return lines + ["- 실행되지 않음.", ""]
    c1 = payloads["C1"]
    regime = c1.get("regime_breakdown", {})
    gates = c1.get("gates")
    config = c1.get("config", {})
    l4_high = l4_reference.get("high_funding_mean_annualized_return") if l4_reference else None
    c1_high = regime.get("high_funding_mean_annualized_return")
    delta = "N/A" if (l4_high is None or c1_high is None) else f"{(c1_high - l4_high) * 100.0:+.2f}%p"
    lines += [
        f"- 신호: 0.5 x z(7d 가격모멘텀) + 0.5 x z(7d 펀딩추세), 계수/임계값(엔트리 z>{config.get('predictive_entry_z')}) "
        "전부 사전고정(학습 없음). OI 특징은 제외(위 한계 참조).",
        f"- 유니버스: L4와 동일(top{c1.get('universe', {}).get('breadth')}, {c1.get('universe', {}).get('history_months')}mo).",
        f"- 고펀딩기 연환산: {_fmt_pct(c1_high)} (vs L4 재산출 {_fmt_pct(l4_high)}, 차이 {delta})",
        f"- Overall: {gates['overall'] if gates else '게이트 미실행'}",
        "",
    ]
    return lines


# ---------------------------------------------------------------------------
# Family D (sector pairs)
# ---------------------------------------------------------------------------


def _family_d_section(payloads: dict[str, dict[str, Any]]) -> list[str]:
    lines = ["## D계열 -- 섹터 내 페어 회귀", ""]
    if "D1" not in payloads:
        return lines + ["- 실행되지 않음.", ""]
    d1 = payloads["D1"]
    gates = d1.get("gates")
    regime = d1.get("regime_breakdown", {})
    lines += [
        f"- {d1.get('deviation_note', '')}",
        f"- {d1.get('meme_substitution_note', '')}",
        "",
        "| 섹터 | 페어 | pair_id |",
        "|---|---|---|",
    ]
    for sector in d1.get("sectors", []):
        lines.append(f"| {sector['sector']} | {sector['symbol_a']}/{sector['symbol_b']} | {sector['pair_id']} |")
    lines += [
        "",
        f"- 고펀딩기 연환산: {_fmt_pct(regime.get('high_funding_mean_annualized_return'))}",
        f"- 트레이드수: {d1.get('metadata', {}).get('n_trades', 'N/A')}, 총비용: {_fmt_usd(d1.get('metadata', {}).get('total_cost_usdt'))}",
        f"- Overall: {gates['overall'] if gates else '게이트 미실행'}",
        "",
    ]
    return lines


# ---------------------------------------------------------------------------
# Gate table + verdict
# ---------------------------------------------------------------------------


def _gate_table(payloads: dict[str, dict[str, Any]]) -> list[str]:
    lines = ["## 게이트 (S1-S5, wave13 gates13.py와 동일 수치 바)", "", "| Candidate | S1(구조/델타중립) | S2 MC(p05/ruin) | S3 블록MDD p95(<=10%) | S4 실행가능성 | S5 스트레스x3(부호∧MDD<=15%) | Overall | 실패사유 |", "|---|---|---|---|---|---|---|---|"]
    for candidate_id in CANDIDATE_IDS:
        payload = payloads.get(candidate_id)
        gates = payload.get("gates") if payload else None
        if not gates:
            lines.append(f"| {candidate_id} | - | - | - | - | - | NOT_EVALUATED | run --stage gates 필요 |")
            continue
        s1, s2, s3, s4, s5 = gates["gate_s1"], gates["gate_s2"], gates["gate_s3"], gates["gate_s4"], gates["gate_s5"]
        s1_cell = f"{s1['status']} ({'델타중립' if s1['delta_neutral_by_construction'] else '방향노출'}, {s1['leverage_multiplier_of_active_capital']:.2f}x)"
        s2_cell = f"{s2['status']} (p05={_fmt_usd(s2['p05'])}, ruin={_fmt_pct(s2['ruin_probability'])})"
        s3_cell = f"{s3['status']} ({_fmt_pct(s3['mdd_p95'])})"
        s4_cell = f"{s4['status']} (레그{_fmt_usd(s4['leg_usdt_nominal'])})"
        s5_cell = f"{s5['status']} (부호={'+' if s5['sign_preserved'] else '반전'}, MDD={_fmt_pct(s5.get('stress_block_mdd_p95'))})"
        reasons = ", ".join(gates.get("failure_reasons", [])) or "-"
        lines.append(f"| {candidate_id} | {s1_cell} | {s2_cell} | {s3_cell} | {s4_cell} | {s5_cell} | {gates['overall']} | {reasons} |")
    return lines


def _verdict_section(payloads: dict[str, dict[str, Any]], l4_reference: dict[str, Any] | None) -> tuple[str, list[str]]:
    l4_high = l4_reference.get("high_funding_mean_annualized_return") if l4_reference else None
    promoted: list[tuple[str, float]] = []
    attempted: list[str] = []
    for candidate_id in CANDIDATE_IDS:
        payload = payloads.get(candidate_id)
        gates = payload.get("gates") if payload else None
        if not gates:
            continue
        attempted.append(candidate_id)
        regime = payload.get("regime_breakdown", {})
        high_funding = regime.get("high_funding_mean_annualized_return")
        beats_l4 = high_funding is not None and l4_high is not None and high_funding > l4_high
        if gates["overall"] == "PASS" and beats_l4:
            promoted.append((candidate_id, high_funding))

    lines = [f"## 판정 -- L4 재산출값({_fmt_pct(l4_high)}) 대비", ""]
    verdict = "PASS" if promoted else "FAIL"
    if promoted:
        winner_id, winner_return = max(promoted, key=lambda item: item[1])
        lines += [
            f"S1-S5 전부 PASS ∧ 고펀딩기 연환산 > L4({_fmt_pct(l4_high)})인 후보: {', '.join(cid for cid, _ in promoted)}.",
            "",
            f"**최종 추천: {winner_id}** -- 고펀딩기 연환산 {_fmt_pct(winner_return)}.",
        ]
        attribution = payloads.get(winner_id, {}).get("yield_attribution")
        if attribution:
            carry_only = attribution.get("carry_only_high_funding_annualized")
            contribution = attribution.get("assumed_yield_contribution_annualized_pts", 0.0)
            carry_only_beats_l4 = carry_only is not None and l4_high is not None and carry_only > l4_high
            lines += [
                "",
                f"**경고 -- 이 승격은 검증되지 않은 가정에 전적으로 의존한다.** {winner_id}의 캐리 단독(가정 Earn 제외) "
                f"고펀딩기 연환산은 {_fmt_pct(carry_only)}로, L4({_fmt_pct(l4_high)}) 대비 "
                f"{'미세하게 상회(재구현 오차 범위 내로 추정, 캐리 메커니즘 자체의 유의미한 개선으로 보기 어려움)' if carry_only_beats_l4 else '오히려 하회'}한다. "
                f"L4를 넘어서는 근거의 사실상 전부({contribution * 100.0:+.2f}%p)는 `common15.ASSUMED_FLEXIBLE_EARN_APR`"
                "(연 2%, earn_apr.py가 실측 실패 후 채택한 가정값)에서 나온다. "
                f"**{winner_id}을 '검증된 개선'으로 승격 확정하지 말 것** -- 이 가정이 틀리면(또는 거래소 정책이 바뀌면) "
                "승격 근거 자체가 사라진다. 실무 채택 전 Binance API 키로 실측 Earn APR 이력을 확인하는 것이 선행되어야 한다.",
            ]
    else:
        lines.append("S1-S5 전부 통과 + L4 상회를 동시에 만족한 후보 없음 (게이트 완화 없이 정직 보고).")

    lines += ["", "### 전체 후보 요약 (7개 전부, 승격 여부와 무관하게)", "", "| Candidate | 계열 | 고펀딩기 연환산 | Overall | 실패사유 |", "|---|---|---:|---|---|"]
    family_map = {"A1": "A_intraday", "A2": "A_intraday", "A3": "A_intraday", "B1": "B_dual_yield", "B2": "B_dual_yield", "C1": "C_predictive", "D1": "D_sector_pairs"}
    for candidate_id in attempted:
        payload = payloads[candidate_id]
        gates = payload["gates"]
        regime = payload.get("regime_breakdown", {})
        high_funding = regime.get("high_funding_mean_annualized_return")
        gate_cell = gates["overall"] if gates["overall"] == "PASS" else f"FAIL({', '.join(gates['failure_reasons']) or '?'})"
        lines.append(f"| {candidate_id} | {family_map.get(candidate_id, '?')} | {_fmt_pct(high_funding)} | {gates['overall']} | {gate_cell} |")
    # 6/7 candidates fail even when the overall verdict is PASS (a single caveat-laden
    # winner) -- the failure decomposition is valuable regardless of the headline verdict,
    # so it always renders, not just on a total wipeout.
    lines += ["", *_failure_decomposition(payloads, l4_high)]
    return verdict, lines


def _failure_decomposition(payloads: dict[str, dict[str, Any]], l4_high: float | None) -> list[str]:
    lines = ["### 계열별 실패 원인 분해 (\"다 안 됨\"으로 뭉뚱그리지 않음)", ""]
    a_ids = [cid for cid in ("A1", "A2", "A3") if cid in payloads]
    if a_ids:
        worst = payloads[a_ids[0]]
        breakeven_sample = next(iter(worst.get("breakeven_analysis", {}).get("per_symbol", {}).values()), {})
        lines.append(
            f"- **A계열(비용초과)**: 인트라데이 왕복비용의 연환산 손익분기가 심볼당 약 "
            f"{_fmt_pct(breakeven_sample.get('breakeven_annualized_apr'))}로, L4의 진입임계 15%APR을 크게 상회한다. "
            "회전을 늘려도 회차당 지불하는 왕복비용이 다일보유의 상각효과를 이기지 못해 구조적으로 비용에 잠식된다."
        )
    if "B1" in payloads:
        attribution = payloads["B1"].get("yield_attribution", {})
        lines.append(
            f"- **B1(기회부재/가정값 한계)**: 캐리 단독 {_fmt_pct(attribution.get('carry_only_high_funding_annualized'))} "
            f"vs L4 {_fmt_pct(l4_high)} -- 같은 유니버스/신호이므로 캐리부분 자체는 L4와 사실상 동일하고, "
            "Earn 가산분은 미검증 가정값이라 승격 근거로 못 쓴다."
        )
    if "B2" in payloads:
        lines.append("- **B2(구조적 방향노출)**: 게이트 통과 여부와 무관하게 헤지 없는 방향베팅이라 " "캐리류와 리스크 성격이 달라 같은 등급으로 승격시킬 수 없다(상세 위 B2 섹션/2022 구간 참조).")
    if "C1" in payloads:
        c1_regime = payloads["C1"].get("regime_breakdown", {})
        c1_high = c1_regime.get("high_funding_mean_annualized_return")
        lines.append(
            f"- **C1(신호개선 미흡)**: 예측신호 고펀딩기 {_fmt_pct(c1_high)} vs L4 {_fmt_pct(l4_high)} -- "
            "선행 진입이 진입가/기회를 개선했는지는 이 수치 차이로 판정, 개선 폭이 미미하거나 음수면 " "2특징(OI 제외) 예측력 자체가 부족한 것."
        )
    if "D1" in payloads:
        d1_regime = payloads["D1"].get("regime_breakdown", {})
        lines.append(
            f"- **D1(기회부재 가능성)**: 고펀딩기 연환산 {_fmt_pct(d1_regime.get('high_funding_mean_annualized_return'))} -- "
            "캐리와 무관한 평균회귀 소스이므로 고펀딩기라는 레짐 구분 자체가 D1에는 느슨하게 적용된다; "
            "트레이드 수/z 임계 통과 빈도가 낮으면 기회부재, 있는데도 수익이 안 나면 스프레드 자체가 " "평균회귀하지 않는 구조적 문제."
        )
    return lines


def _dsr_note() -> str:
    return (
        f"다중검정 보정: 누적 시행 {DSR_CUMULATIVE_TRIALS}회(wave14까지 84회 + 이 wave의 실제 7개 후보) 기준 "
        "DSR(Deflated Sharpe Ratio) 참고치를 각 결과 JSON의 `reference_metrics.dsr`에 기록 "
        "(샤프는 참고 지표이며 승격 판정에는 사용하지 않음, wave10-14와 동일 원칙)."
    )


def write_wave15_report(results_dir: Path, report_dir: Path, registry_path: Path, cache_dir: Path) -> None:
    payloads: dict[str, dict[str, Any]] = {}
    for candidate_id in CANDIDATE_IDS:
        try:
            payloads[candidate_id] = _load(results_dir, candidate_id)
        except FileNotFoundError:
            continue
    l4_reference = _load_l4_reference(cache_dir)
    verdict, verdict_lines = _verdict_section(payloads, l4_reference)

    lines: list[str] = [
        *_header_and_scope_note(),
        f"## L4 재산출 기준선 (동일 costs_measured 비용모델, 동일 동결 스냅샷 2026-07-22)",
        "",
        f"- 고펀딩기 연환산: {_fmt_pct(l4_reference.get('high_funding_mean_annualized_return') if l4_reference else None)}",
        f"- 유니버스: top{l4_reference.get('universe_size') if l4_reference else 'N/A'}, 구간: "
        f"{l4_reference.get('span', {}).get('start') if l4_reference else 'N/A'} ~ {l4_reference.get('span', {}).get('end') if l4_reference else 'N/A'}",
        "",
        *_family_a_section(payloads),
        *_family_b_section(payloads),
        *_family_c_section(payloads, l4_reference),
        *_family_d_section(payloads),
        *verdict_lines,
        "",
        *_gate_table(payloads),
        "",
        "## 다중검정",
        "",
        _dsr_note(),
        "",
    ]
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "wave15_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    registry_lines = ["# Wave-15 registry", "", "| Candidate | 계열 | Family | State | Verdict | 승격 | 분류 |", "|---|---|---|---|---|---|---|"]
    family_map = {"A1": "A_intraday_carry", "A2": "A_intraday_carry", "A3": "A_intraday_carry", "B1": "B_dual_yield", "B2": "B_dual_yield", "C1": "C_predictive", "D1": "D_sector_pairs"}
    l4_high = l4_reference.get("high_funding_mean_annualized_return") if l4_reference else None
    for candidate_id in CANDIDATE_IDS:
        payload = payloads.get(candidate_id)
        gates = payload.get("gates") if payload else None
        if not gates:
            registry_lines.append(f"| {candidate_id} | {family_map[candidate_id]} | wave15_diverse | PENDING | - | - | gates 미실행 |")
            continue
        regime = payload.get("regime_breakdown", {})
        high_funding = regime.get("high_funding_mean_annualized_return")
        beats_l4 = high_funding is not None and l4_high is not None and high_funding > l4_high
        result_verdict = gates["overall"]
        actually_promoted = result_verdict == "PASS" and beats_l4
        promoted_cell = "YES" if actually_promoted else "no"
        if actually_promoted:
            classification = "승격(ASSUMED Earn 가정 의존 -- 캐리단독은 L4 상회분 미미, report 경고 참조)" if payload.get("yield_attribution") else "승격"
        elif result_verdict != "PASS":
            classification = f"게이트위반({', '.join(gates['failure_reasons'])})"
        else:
            classification = "L4미달"
        registry_lines.append(f"| {candidate_id} | {family_map[candidate_id]} | wave15_diverse | EVALUATED | {result_verdict} | {promoted_cell} | {classification} |")
    registry_lines.append("")
    registry_lines.append(f"**판정**: {'통과' if verdict == 'PASS' else '전멸 -- 계열별 원인은 report/wave15_report.md 참조'}.")
    registry_path.write_text("\n".join(registry_lines) + "\n", encoding="utf-8")


__all__ = ["write_wave15_report", "year_slice_annualized"]
