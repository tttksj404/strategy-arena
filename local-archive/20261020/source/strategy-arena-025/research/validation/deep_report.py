from __future__ import annotations

from pathlib import Path
from typing import TypeAlias

JsonValue: TypeAlias = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]

def _cell(result: dict[str, JsonValue], section: str, label: str) -> str:
    criteria = result["criteria"]
    if not isinstance(criteria, dict):
        return "FAIL invalid criteria"
    item = criteria.get(section)
    if item is None:
        return "N/A"
    if not isinstance(item, dict):
        return "FAIL invalid"
    status = str(item.get("status", "FAIL"))
    data = result.get(label)
    if section == "mc" and isinstance(data, dict):
        unit = data.get("unit", {})
        if isinstance(unit, dict):
            return f"{status} p05={float(unit.get('p05', 0.0)):.2f}; ruin={float(unit.get('ruin_probability', 0.0)):.2%}"
    if section == "dsr" and isinstance(data, dict):
        return f"{status} score={float(data.get('score', 0.0)):.3f}"
    if section == "bitget_sign" and isinstance(data, dict):
        return f"{status} sign={float(data.get('sign_agreement', 0.0)):.2%}; entry={float(data.get('entry_agreement', 0.0)):.2%}; coverage={int(data.get('coverage_days', 0))}d"
    if section == "loo" and isinstance(data, list):
        return f"{status} years={len(data)}"
    if section == "regime_blocks" and isinstance(data, dict):
        return f"{status} blocks={int(data.get('block_count', 0))}; MDD p95={float(data.get('mdd_p95', 0.0)):.2%}"
    return status

def write_report(results: list[dict[str, JsonValue]], path: Path) -> None:
    rows: list[str] = [
        "# Deep validation report",
        "",
        "Cache-only full-history validation for W2c, F1e, W3c, and W3d.",
        "",
        "## Candidate x validation",
        "",
        "| Candidate | MC 10k | Leave-one-year-out | DSR (28 trials) | Bitget native | 90d block shuffle | Overall |",
        "|---|---|---|---|---|---|---|",
    ]
    for result in results:
        candidate = str(result["candidate_id"])
        cells = [
            _cell(result, "mc", "bootstrap_mc"),
            _cell(result, "loo", "leave_one_year_out"),
            _cell(result, "dsr", "deflated_sharpe"),
            _cell(result, "bitget_sign", "bitget_native"),
            _cell(result, "regime_blocks", "regime_block_bootstrap"),
        ]
        overall = result.get("overall")
        overall_status = str(overall.get("status", "FAIL")) if isinstance(overall, dict) else "FAIL"
        rows.append(f"| {candidate} | {' | '.join(cells)} | {overall_status} |")
    native_results = [result for result in results if isinstance(result.get("bitget_native"), dict) and result["bitget_native"].get("status") != "N/A"]
    native = native_results[0].get("bitget_native", {}) if native_results else {}
    native_observations = int(native.get("observations", 0)) if isinstance(native, dict) else 0
    native_coverage = int(native.get("coverage_days", 0)) if isinstance(native, dict) else 0
    native_symbols = len(native.get("symbols", [])) if isinstance(native, dict) and isinstance(native.get("symbols"), list) else 0
    integrity = native_results[0].get("cache_integrity", {}) if native_results else {}
    integrity_status = str(integrity.get("status", "UNDETERMINED")) if isinstance(integrity, dict) else "UNDETERMINED"
    rows.extend([
        "",
        "## Year-by-year leave-one-out",
        "",
        "| Candidate | Year | Held-out return | Held-out Sharpe | Remaining return | Remaining Sharpe |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for result in results:
        candidate = str(result["candidate_id"])
        loo = result.get("leave_one_year_out")
        if isinstance(loo, list):
            for row in loo:
                if isinstance(row, dict):
                    rows.append(
                        f"| {candidate} | {int(row['year'])} | {float(row['heldout_return']):.4f} | "
                        f"{float(row['heldout_sharpe']):.3f} | {float(row['remaining_return']):.4f} | "
                        f"{float(row['remaining_sharpe']):.3f} |"
                    )
    rows.extend([
        "",
        "## Decision criteria",
        "",
        "- MC: 10,000 unit-exposure full-period trade bootstrap paths; PASS requires final-capital p05 > 300 and P(capital < 150) < 5%.",
        "- MC input contract: F1e/W2c use closed-trade returns; W3c/W3d expose active-day returns in the same field, so their MC cells remain UNDETERMINED until upstream semantics are separated.",
        "- Kelly: the existing gate-compatible mean(trade_return) / sample_variance estimate; both f* and 0.25f* are recorded.",
        "- DSR: Bailey-Lopez de Prado daily-return deflated Sharpe z-score with trials=28; PASS requires DSR score > 0.",
        "- Bitget native: record 7-day score correlation, entry-signal agreement, and sign agreement. PASS requires sign agreement > 80% and the requested 133-day common coverage.",
        "- Leave-one-year-out and 90-day block permutation are diagnostic checks; PASS means the calculation completed with the minimum sample.",
        "- The 90-day block method shuffles block order without replacement; final capital is therefore an invariant and MDD is the distributional stress output.",
        "",
        "## Data constraints",
        "",
        "- No network calls were made. Bitget reproduction uses only normalized local cache rows in research/wave1/cache.",
        f"- The current common funding-score interval is {native_symbols} symbols, {native_observations} observations, and {native_coverage} days. It is shorter than the requested 133 days, so the native cells and overall verdicts for F1e/W2c are FAIL.",
        f"- The wave2 cache manifest recheck is {integrity_status}; every consumed funding file must have a matching byte count and SHA-256 before native evidence can be accepted.",
        "- Funding rows are normalized to UTC 8-hour buckets and a 7-day score is emitted only for 21 contiguous buckets; gaps reset the rolling window.",
        "- Final candidate status is the intersection of the stated gates; insufficient cache coverage is never interpreted as a PASS.",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
