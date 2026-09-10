from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import work.generate_top_strategy_pdf as pdf


def _apply_replacements(text: str, replacements: dict[str, str]) -> str:
    for source, replacement in replacements.items():
        text = text.replace(source, replacement)
    return text


def test_price_volume_leverage_replacements_do_not_duplicate_numeric_labels(monkeypatch) -> None:
    monkeypatch.setattr(
        pdf,
        "CURRENT_DIAGNOSTIC",
        {
            "strategies": {
                "paper_gate_top_price_volume": {
                    "segments": {
                        "locked_test": {
                            "leverage": {
                                "median": 0.6009029544367737,
                                "mean": 0.6318713574442012,
                                "p95": 1.1745477903370922,
                                "max": 1.2715085271996802,
                            }
                        }
                    }
                }
            }
        },
    )
    monkeypatch.setattr(pdf, "CURRENT_FRONTIER", {})
    monkeypatch.setattr(pdf, "CURRENT_ADAPTIVE", {})
    monkeypatch.setattr(pdf, "CURRENT_ADAPTIVE_5M", {})

    replacements = pdf._diagnostic_replacements()
    page_6 = _apply_replacements(
        "실효배율 median 0.6009x, max 1.2715x",
        replacements,
    )
    page_14 = _apply_replacements(
        "median .6009x / mean .632x / p95 1.175x / max 1.2715x",
        replacements,
    )

    assert page_6 == "실효배율 median 0.6009x, max 1.27151x"
    assert page_14 == "median 0.6009x / mean 0.6319x / p95 1.175x / max 1.27151x"
    assert "00.6009x" not in page_6
    assert "max max" not in page_6
    assert "max max" not in page_14


def test_path_risk_illustration_really_crosses_its_barrier_before_recovery() -> None:
    # Given: the exact price path used by the PDF illustration.
    example = pdf.path_risk_example()
    # When: checking the marked breach, independently of its caption.
    first_breach = next(index for index, price in enumerate(example.prices) if price <= example.barrier)
    # Then: the marked point crosses first, and the later recovery is not pre-breach profit.
    assert first_breach == example.breach_index < example.recovery_index
    assert example.prices[example.recovery_index] > example.prices[0]
