from __future__ import annotations

from run_bitget_5x_leverage_audit import ScreenInterval, ScreenReport, ScreenRow, select_cap_five_screen_passes


def test_select_cap_five_screen_passes_keeps_only_current_five_x_passes() -> None:
    report = ScreenReport(
        mode="public-api-paper-only",
        results=(
            ScreenInterval(
                interval="4H",
                results=(
                    ScreenRow(candidate="five", risk="L=5", max_leverage=5.0, exchange_screen_pass=True),
                    ScreenRow(candidate="one", risk="L=1", max_leverage=1.0, exchange_screen_pass=True),
                    ScreenRow(candidate="failed", risk="L=5", max_leverage=5.0, exchange_screen_pass=False),
                ),
            ),
        ),
    )

    selected = select_cap_five_screen_passes(report)

    assert [(interval, row.candidate) for interval, row in selected] == [("4H", "five")]
