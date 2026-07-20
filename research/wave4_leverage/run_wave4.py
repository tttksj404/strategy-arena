from __future__ import annotations

import argparse
import os
from pathlib import Path

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.wave4_leverage.sweep import SimulationResult, run_sweep


def _fmt(value: float, percent: bool = False) -> str:
    return f"{value * 100:.2f}%" if percent else f"{value:.2f}"


def write_report(root: Path, results: tuple[SimulationResult, ...]) -> None:
    reconciliation = [result for result in results if result.leverage == 1.0]
    if len(reconciliation) != 4 or any(result.reconciliation_pass is not True for result in reconciliation):
        raise RuntimeError("baseline reconciliation gate failed; report publication blocked")
    rows = []
    for result in results:
        gate = result.mc_p05 > 300.0 and result.bankruptcy_probability < 0.05 and result.mdd <= 0.25
        rows.append((result, gate))
    passed = [result for result, gate in rows if gate]
    max_safe = max((result.leverage for result in passed), default=None)
    report = [
        "# Wave-4 Leverage Sweep Report",
        "",
        "## Publication gate",
        "",
        "- Grid is unchanged: W2c/F1f x SYM/ASYM x {1, 1.5, 2, 3, 5, 10} = 24 combinations.",
        "- Gate criteria are unchanged: MC p05 > $300, final bankruptcy probability < 5%, and MDD <= 25%.",
        "- Report publication is blocked unless all four L=1 reconciliation rows pass at <= 1% relative error for both CAGR and MDD.",
        "",
        "### L=1 reconciliation",
        "",
        "| Candidate | Structure | Sweep CAGR | Engine CAGR | CAGR rel. error | Sweep MDD | Engine MDD | MDD rel. error | Status |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for result in reconciliation:
        report.append(
            f"| {result.candidate_id} | {result.structure} | {_fmt(result.cagr, True)} | {_fmt(result.baseline_cagr or 0.0, True)} | {_fmt(result.cagr_relative_error or 0.0, True)} | {_fmt(result.mdd, True)} | {_fmt(result.baseline_mdd or 0.0, True)} | {_fmt(result.mdd_relative_error or 0.0, True)} | {'PASS' if result.reconciliation_pass else 'FAIL'} |"
        )
    report.extend([
        "",
        f"- Reconciliation gate: {'PASS' if len(reconciliation) == 4 and all(result.reconciliation_pass for result in reconciliation) else 'FAIL'}.",
        "",
        "## Model contract",
        "",
        "- Daily P&L is replayed from the imported wave-1/wave-2 engine path. The sweep adds leverage scaling, borrow interest, and liquidation overlays only.",
        "- Liquidation basis move: `abs(simultaneous_close_basis_change) + max(0, perp_intraday_range_pct - spot_intraday_range_pct) * 0.5`.",
        "- Stress basis move is the same value multiplied by 1.5 and is reported separately.",
        "- Inputs are cache-only from `research/wave1/cache`; no network calls are made.",
        "",
        "## Combination results",
        "",
        "| Candidate | Structure | L | CAGR | MDD | MC p05 | Bankruptcy | Liq. | Stress liq. | Borrowing | Gate |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ])
    for result, gate in rows:
        report.append(
            f"| {result.candidate_id} | {result.structure} | {result.leverage:g} | {_fmt(result.cagr, True)} | {_fmt(result.mdd, True)} | ${result.mc_p05:,.2f} | {_fmt(result.bankruptcy_probability, True)} | {result.liquidation_count} | {result.stress_liquidation_count} | ${result.borrowing_cost_total:,.2f} | {'PASS' if gate else 'FAIL'} |"
        )
    report.extend([
        "",
        "## Conclusion",
        "",
        f"- Combination gates passed: {len(passed)}/{len(results)}.",
        f"- Maximum leverage passing the unchanged risk gate: {max_safe:g}x." if max_safe is not None else "- Maximum leverage passing the unchanged risk gate: none.",
        "- The reconciliation gate passed, so this report is publishable.",
        "",
    ])
    report_path = root / "research" / "wave4_leverage" / "LEVERAGE_REPORT.md"
    staged_path = report_path.with_name(f".{report_path.name}.staged")
    staged_path.write_text("\n".join(report), encoding="utf-8")
    os.replace(staged_path, report_path)

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
