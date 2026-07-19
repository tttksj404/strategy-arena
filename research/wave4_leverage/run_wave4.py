from __future__ import annotations

import argparse
import json
from pathlib import Path

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.wave4_leverage.sweep import MC_PATHS, STRUCTURES, VALID_LEVERAGES, run_sweep


def _fmt(value: float, percent: bool = False) -> str:
    return f"{value * 100:.2f}%" if percent else f"{value:.2f}"


def write_report(root: Path, results: tuple[object, ...]) -> None:
    rows = []
    for item in results:
        result = item
        gate = result.mc_p05 > 300.0 and result.bankruptcy_probability < 0.05 and result.mdd <= 0.25
        rows.append((result, gate))
    passed = [result for result, gate in rows if gate]
    max_safe = max((result.leverage for result in passed), default=None)
    report = [
        "# Wave-4 레버리지 스윕 보고서",
        "",
        "## 결론",
        "",
        f"- 사전등록 조합: {len(results)}개 (W2c/F1f × SYM/ASYM × 6 레버리지)",
        f"- 게이트 통과: {len(passed)}개",
        f"- 최대 안전 레버리지: {max_safe:g}x" if max_safe is not None else "- 최대 안전 레버리지: 없음",
        "",
        "## 사전등록 및 모델",
        "",
        "- 레버리지: `{1.0, 1.5, 2, 3, 5, 10}`; 구조: `{SYM, ASYM}`; 실행 후 조합 추가 없음.",
        "- SYM: 두 레그 노셔널을 함께 스케일하고, 현물 차입분에 연 10% 이자 부과.",
        "- ASYM: 현물은 현금, 퍼프만 마진; 퍼프 증거금=노셔널/L, 총 자본효율=`2/(1+1/L)`.",
        "- 청산: 매 보유일 `spot_low/spot_open - perp_high/perp_open`의 비동기 최악 베이시스를 사용. 손실이 초기 퍼프 증거금−노셔널의 0.5% 이상이면 청산하고 노셔널의 0.06% 수수료를 추가.",
        f"- MC: 일별 순수익률 재표본화 {MC_PATHS:,}회, p05는 최종자본, 파산은 최종자본 `< $150`.",
        "- 입력: 기존 엔진의 `research/wave1/cache`만 사용; 네트워크 호출 없음.",
        "",
        "## 조합별 지표",
        "",
        "| 후보 | 구조 | L | CAGR | MDD | MC p05 | 파산확률 | 청산 | 빌림비용 | 게이트 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for result, gate in rows:
        report.append(
            f"| {result.candidate_id} | {result.structure} | {result.leverage:g} | {_fmt(result.cagr, True)} | {_fmt(result.mdd, True)} | ${result.mc_p05:,.2f} | {_fmt(result.bankruptcy_probability, True)} | {result.liquidation_count} | ${result.borrowing_cost_total:,.2f} | {'PASS' if gate else 'FAIL'} |"
        )
    report.extend([
        "",
        "## 판정",
        "",
        "게이트는 `MC p05 > 300 ∧ 파산확률 < 5% ∧ MDD ≤ 25%`를 조합별로 적용했다. 최대 안전 레버리지는 이 조건을 만족한 조합 중 가장 큰 유효 레버리지이며, 통과 조합이 없으면 안전 레버리지 없음으로 판정한다.",
        "",
    ])
    (root / "research" / "wave4_leverage" / "LEVERAGE_REPORT.md").write_text("\n".join(report), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the cache-only Wave-4 leverage sweep")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    root = args.root.resolve()
    results = run_sweep(root)
    write_report(root, results)
    print(f"wrote {len(results)} combinations, report, and JSON results")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
