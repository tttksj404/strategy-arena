#!/usr/bin/env -S python
# /// script
# dependencies = ["matplotlib>=3.8", "reportlab>=4.0"]
# ///
"""Create the strategy-specific Korean strategy-arena evidence PDF.

Usage:
    python work/generate_top_strategy_pdf.py
"""

from __future__ import annotations

import gzip
import hashlib
import json
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from top_strategy_path_illustration import path_risk_example

ROOT: Final = Path(__file__).resolve().parents[1]
OUTPUT: Final = ROOT / "output" / "pdf" / "strategy-arena-top-return-candidates-ko.pdf"
REGULAR: Final = Path(r"C:\Windows\Fonts\malgun.ttf")
BOLD: Final = Path(r"C:\Windows\Fonts\malgunbd.ttf")
NAVY: Final = colors.HexColor("#10243A")
INK: Final = colors.HexColor("#1D2939")
MUTED: Final = colors.HexColor("#5A6878")
BLUE: Final = colors.HexColor("#1464B4")
TEAL: Final = colors.HexColor("#008F83")
ORANGE: Final = colors.HexColor("#D66A1F")
RED: Final = colors.HexColor("#B42318")
PALE: Final = colors.HexColor("#F2F6FA")
GRID: Final = colors.HexColor("#D9E2EC")
PAGE_W, PAGE_H = A4
ARCHIVE: Final = ROOT / "work" / "bitget_history_full_daily_20260824_v2"
DIAGNOSTIC_PATH: Final = ROOT / "work" / "bitget_top_strategy_diagnostic_report.json"
FRONTIER_PATH: Final = ROOT / "work" / "bitget_top_leverage_frontier_report.json"
ADAPTIVE_PATH: Final = ROOT / "work" / "bitget_15m_adaptive_max_return_search_report.json"
ADAPTIVE_5M_PATH: Final = ROOT / "work" / "bitget_5m_adaptive_max_return_search_report.json"
SYMBOLS: Final = ("ETH", "BNB", "ADA", "SOL", "DOGE", "XRP", "DOT", "LTC", "LINK", "AVAX", "UNI", "ATOM", "NEAR")
CURRENT_DIAGNOSTIC: dict[str, object] = {}
CURRENT_FRONTIER: dict[str, object] = {}
CURRENT_ADAPTIVE: dict[str, object] = {}
CURRENT_ADAPTIVE_5M: dict[str, object] = {}


def register_fonts() -> None:
    if not REGULAR.exists() or not BOLD.exists():
        raise FileNotFoundError("Malgun Gothic fonts are required")
    pdfmetrics.registerFont(TTFont("Malgun", str(REGULAR)))
    pdfmetrics.registerFont(TTFont("MalgunBold", str(BOLD)))


def styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "cover": ParagraphStyle("cover", parent=base["Title"], fontName="MalgunBold", fontSize=24, leading=34, textColor=colors.white, wordWrap="CJK"),
        "cover_sub": ParagraphStyle("cover_sub", parent=base["Normal"], fontName="Malgun", fontSize=11, leading=17, textColor=colors.HexColor("#C9D8E8"), wordWrap="CJK"),
        "h1": ParagraphStyle("h1", parent=base["Heading1"], fontName="MalgunBold", fontSize=18, leading=25, textColor=NAVY, spaceAfter=9, wordWrap="CJK"),
        "h2": ParagraphStyle("h2", parent=base["Heading2"], fontName="MalgunBold", fontSize=12.5, leading=18, textColor=INK, spaceBefore=7, spaceAfter=5, wordWrap="CJK"),
        "body": ParagraphStyle("body", parent=base["BodyText"], fontName="Malgun", fontSize=9.1, leading=15, textColor=INK, spaceAfter=6, wordWrap="CJK"),
        "small": ParagraphStyle("small", parent=base["BodyText"], fontName="Malgun", fontSize=7.5, leading=11, textColor=MUTED, wordWrap="CJK"),
        "table": ParagraphStyle("table", parent=base["BodyText"], fontName="Malgun", fontSize=7.8, leading=11.2, textColor=INK, wordWrap="CJK"),
        "head": ParagraphStyle("head", parent=base["BodyText"], fontName="MalgunBold", fontSize=8, leading=11, textColor=colors.white, alignment=TA_CENTER, wordWrap="CJK"),
        "callout": ParagraphStyle("callout", parent=base["BodyText"], fontName="MalgunBold", fontSize=10.5, leading=16, textColor=NAVY, wordWrap="CJK"),
        "center": ParagraphStyle("center", parent=base["BodyText"], fontName="Malgun", fontSize=9, leading=14, alignment=TA_CENTER, textColor=INK, wordWrap="CJK"),
    }


def p(text: str, style: ParagraphStyle) -> Paragraph:
    for source, replacement in _diagnostic_replacements().items():
        text = text.replace(source, replacement)
    return Paragraph(text.replace("−", "-"), style)


def _report_provenance(path: Path, report: dict[str, object]) -> str:
    """Return reproducible local-file provenance without claiming a server timestamp."""

    try:
        label = path.relative_to(ROOT).as_posix()
    except ValueError:
        label = path.as_posix()
    if not path.exists():
        return f"{label} · 파일 없음"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    modified = datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat()
    snapshot_ms = report.get("data", {}).get("snapshot_ms")
    snapshot = datetime.fromtimestamp(float(snapshot_ms) / 1000.0, UTC).isoformat() if snapshot_ms else "미상"
    return f"{label} · file_mtime_utc={modified} · sha256={digest} · data_snapshot_utc={snapshot}"


def _adaptive_summary(report: dict[str, object], path: Path) -> dict[str, object]:
    """Extract only report-backed adaptive-search fields used in the PDF."""

    winner = report.get("selected_validation_winner_record", {})
    validation = winner.get("validation_stress_fixed_notional", {})
    locked = report.get("locked_test", {})
    fixed = locked.get("fixed_notional", {})
    stress = fixed.get("stress", {})
    pbo = report.get("selection_bias", {}).get("pbo", {})
    dsr = locked.get("deflated_sharpe_stress", {})
    survival_test = locked.get("survival", {}).get("test", [])
    first_horizon = survival_test[0] if survival_test else {}
    return {
        "interval": str(report.get("data", {}).get("interval", "?")),
        "strategy": str(report.get("strategy", "adaptive search")),
        "family": str(winner.get("family", "?")),
        "trials": int(pbo.get("candidates", report.get("predeclared_search", {}).get("raw_configurations", 0))),
        "models": int(report.get("predeclared_search", {}).get("unique_models", 0)),
        "rows": int(report.get("data", {}).get("common_rows", 0)),
        "horizon_bars": int(first_horizon.get("horizon_bars", 0)),
        "horizon_minutes": int(first_horizon.get("horizon_minutes", 0)),
        "validation_stress": 100 * float(validation.get("net_return", 0.0)),
        "locked_stress": 100 * float(stress.get("net_return", 0.0)),
        "pbo": 100 * float(pbo.get("pbo", 0.0)),
        "dsr": float(dsr.get("probability", 0.0)),
        "provenance": _report_provenance(path, report),
    }


def _diagnostic_replacements() -> dict[str, str]:
    """Keep repeated report facts synchronized when the diagnostic is regenerated."""

    if not CURRENT_DIAGNOSTIC:
        return {}
    strategies = CURRENT_DIAGNOSTIC.get("strategies", {})
    donchian = strategies.get("raw_top_donchian", {}).get("locked_test_daily_bar_diagnostic", {})
    price_volume = strategies.get("paper_gate_top_price_volume", {}).get("locked_test_daily_bar_diagnostic", {})
    donchian_censored = strategies.get("raw_top_donchian", {}).get("locked_test_daily_bar_diagnostic", {}).get("right_censored_event_count", "?")
    price_volume_censored = strategies.get("paper_gate_top_price_volume", {}).get("locked_test_daily_bar_diagnostic", {}).get("right_censored_event_count", "?")
    replacements = {
        "2026-09-04T07:44:17.759637+00:00": str(CURRENT_DIAGNOSTIC.get("observed_at", "")),
        "잠금 일봉 market-proxy diagnostic:": f"잠금 일봉 market-proxy diagnostic (terminal-direction metric; right-censored Donchian={donchian_censored}, Price–Volume={price_volume_censored}):",
    }
    if CURRENT_FRONTIER:
        replacements["work/bitget_top_leverage_frontier_report.json."] = f"work/bitget_top_leverage_frontier_report.json · generated_at {CURRENT_FRONTIER.get('observed_at', '')}."
        frontier_metrics = CURRENT_FRONTIER.get("locked_test", {}).get("metrics", {})
        if isinstance(frontier_metrics, dict):
            frontier_base = frontier_metrics.get("base", {}).get("net_return")
            frontier_stress = frontier_metrics.get("stress", {}).get("net_return")
            frontier_hits = frontier_metrics.get("market_ohlc_proxy_liquidation_hit_count")
            if frontier_base is not None:
                replacements["+232.7489%"] = f"{100 * float(frontier_base):+.4f}%"
            if frontier_stress is not None:
                replacements["+208.2314%"] = f"{100 * float(frontier_stress):+.4f}%"
            if frontier_hits is not None:
                replacements["proxy-hit 0 frontier"] = f"proxy-hit {frontier_hits} frontier"
            leverage = CURRENT_FRONTIER.get("locked_test", {}).get("leverage")
            if leverage is not None:
                replacements["고정 2.5배"] = f"고정 {float(leverage):g}배"
                replacements["2.5배"] = f"{float(leverage):g}배"
    for strategy_name, old_values in (("raw_top_donchian", (("+452.19%", "+452.19%"), ("+373.73%", "+373.73%"))), ("paper_gate_top_price_volume", (("-6.74%", "-6.74%"), ("-33.12%", "-33.12%")))):
        counterfactual = strategies.get(strategy_name, {}).get("fixed_5x_counterfactual", {})
        for field, (old, fallback) in zip(("base", "stress"), old_values):
            value = counterfactual.get(field, {}).get("net_return")
            if value is not None:
                replacements[old] = f"{100 * float(value):+.2f}%"
    selection_source = CURRENT_DIAGNOSTIC.get("selection_audit_source", {})
    if isinstance(selection_source, dict) and selection_source.get("sha256"):
        replacements["선택·PBO 맥락: work/bitget_archive_pbo_search_report.json 및 work/bitget_top_leverage_frontier_report.json"] = (
            "선택·PBO 맥락: work/bitget_archive_pbo_search_report.json · "
            f"selection_audit_source.sha256={selection_source['sha256']} · "
            f"frontier provenance: {_report_provenance(FRONTIER_PATH, CURRENT_FRONTIER)}"
        )
    if CURRENT_DIAGNOSTIC:
        diagnostic_observed = str(CURRENT_DIAGNOSTIC.get("observed_at", ""))
        replacements[f"진단 출처: work/bitget_top_strategy_diagnostic_report.json · generated_at {diagnostic_observed} UTC · mode=public-archive-paper-only."] = (
            f"진단 provenance: {_report_provenance(DIAGNOSTIC_PATH, CURRENT_DIAGNOSTIC)} · mode=public-archive-paper-only."
        )
        replacements[f"diagnostic source: work/bitget_top_strategy_diagnostic_report.json · generated_at {diagnostic_observed} UTC · observed mode=public-archive-paper-only."] = (
            f"diagnostic provenance: {_report_provenance(DIAGNOSTIC_PATH, CURRENT_DIAGNOSTIC)} · observed mode=public-archive-paper-only."
        )
    for strategy_name, old_values in (
        ("raw_top_donchian", (("+55.4650%", "+55.4650%"), ("+52.5866%", "+52.5866%"), ("9.1505%", "9.1505%"))),
        ("paper_gate_top_price_volume", (("+20.4189%", "+20.4189%"), ("+15.3315%", "+15.3315%"), ("13.3705%", "13.3705%"))),
    ):
        locked = strategies.get(strategy_name, {}).get("segments", {}).get("locked_test", {})
        for section, (old, _fallback) in zip(("base", "stress", "mdd"), old_values):
            if section == "mdd":
                value = locked.get("base", {}).get("max_drawdown")
            else:
                value = locked.get(section, {}).get("net_return")
            if value is not None:
                replacements[old] = f"{100 * float(value):+.4f}%" if section != "mdd" else f"{100 * float(value):.4f}%"
    for label, payload in (("Donchian", donchian), ("Price–Volume", price_volume)):
        count = payload.get("event_count")
        if count is not None:
            replacements[f"{label} 111 events" if label == "Donchian" else f"{label} 221 events"] = f"{label} {count} events"
        rate = payload.get("completed_event_terminal_direction_win_rate", payload.get("eventual_direction_win_rate"))
        if rate is not None:
            replacements["eventual win 45.95%" if label == "Donchian" else "eventual win 48.42%"] = f"terminal-direction win {float(rate):.2%}"
        for key, old in (("mae_median", "MAE50 2.334%" if label == "Donchian" else "MAE50 2.865%"), ("mae_p95", "MAE95 7.619%" if label == "Donchian" else "MAE95 21.214%")):
            value = payload.get(key)
            if value is not None:
                replacements[old] = f"{'MAE50' if key == 'mae_median' else 'MAE95'} {float(value):.3%}"
    for strategy_name, old_values in (
        ("raw_top_donchian", ((".289x", ".289x"), (".530x", ".530x"), ("1.526x", "1.526x"), ("5x", "5x"))),
        ("paper_gate_top_price_volume", ((".6009x", ".6009x"), (".632x", ".632x"), ("1.175x", "1.175x"), ("1.2715x", "1.2715x"))),
    ):
        leverage = strategies.get(strategy_name, {}).get("segments", {}).get("locked_test", {}).get("leverage", {})
        for key, (old, _fallback) in zip(("median", "mean", "p95", "max"), old_values):
            value = leverage.get(key)
            if value is not None:
                formatted = f"{float(value):.4g}x" if key != "max" else f"{float(value):g}x"
                source_variants = (old, f"0{old}") if old.startswith(".") else (old,)
                for source in source_variants:
                    replacements[f"{key} {source}"] = f"{key} {formatted}"
    if CURRENT_ADAPTIVE:
        adaptive = _adaptive_summary(CURRENT_ADAPTIVE, ADAPTIVE_PATH)
        adaptive_5m = _adaptive_summary(CURRENT_ADAPTIVE_5M, ADAPTIVE_5M_PATH)
        old_adaptive_sentence = (
            f"별도 {adaptive['interval']} adaptive {adaptive['trials']}개 탐색은 "
            f"validation stress {adaptive['validation_stress']:+.2f}%, locked stress {adaptive['locked_stress']:+.2f}%, "
            f"PBO {adaptive['pbo']:.2f}%, DSR {adaptive['dsr']:.2e}로 후보에서 탈락했다. "
            "이 결과는 fixed-notional 합산이며 계좌 수익률이 아니고, 보고서의 현재 데이터/실행계약 시각을 별도 확인해야 한다."
        )
        replacements[old_adaptive_sentence] = (
            f"별도 strategy={adaptive['interval']} adaptive {adaptive['trials']}개({adaptive['family']}; "
            f"첫 survival {adaptive['horizon_bars']} bars={adaptive['horizon_minutes']}분) 탐색은 "
            f"validation stress {adaptive['validation_stress']:+.2f}%, locked stress {adaptive['locked_stress']:+.2f}%, "
            f"PBO {adaptive['pbo']:.2f}%, DSR {adaptive['dsr']:.4e}로 후보에서 탈락했다. "
            f"별도 strategy={adaptive_5m['interval']} adaptive {adaptive_5m['trials']}개/{adaptive_5m['models']} models "
            f"(rows {adaptive_5m['rows']}; {adaptive_5m['family']})도 "
            f"validation stress {adaptive_5m['validation_stress']:+.4f}%, locked stress {adaptive_5m['locked_stress']:+.4f}%, "
            f"PBO {adaptive_5m['pbo']:.2f}%, DSR {adaptive_5m['dsr']:.4e}로 탈락했다. "
            "두 결과는 fixed-notional 합산이며 계좌 수익률이 아니다. "
            f"15m provenance: {adaptive['provenance']} · 5m provenance: {adaptive_5m['provenance']}"
        )
    return replacements


def table(rows: list[list[str]], st: dict[str, ParagraphStyle], widths: list[float], header: bool = True) -> Table:
    converted = [[p(cell, st["head"] if header and i == 0 else st["table"]) for cell in row] for i, row in enumerate(rows)]
    commands: list[tuple[object, ...]] = [("GRID", (0, 0), (-1, -1), 0.35, GRID), ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5), ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]
    if header:
        commands.extend([("BACKGROUND", (0, 0), (-1, 0), NAVY), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white)])
    for i in range(1 if header else 0, len(rows)):
        if i % 2 == 0:
            commands.append(("BACKGROUND", (0, i), (-1, i), PALE))
    result = Table(converted, colWidths=widths, repeatRows=1 if header else 0)
    result.setStyle(TableStyle(commands))
    return result


def _load_archive_arrays() -> tuple[tuple[datetime, ...], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Read the verified daily archive directly for deterministic explanatory cases."""

    by_symbol: dict[str, dict[int, dict[str, float]]] = {}
    for symbol in SYMBOLS:
        rows: dict[int, dict[str, float]] = {}
        for path in (ARCHIVE / f"{symbol}USDT" / "1D").glob("*.jsonl.gz"):
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                for raw in handle:
                    if raw.strip():
                        item = json.loads(raw)
                        rows[int(item["timestamp_ms"])] = item
        if not rows:
            return None
        by_symbol[symbol] = rows
    common = sorted(set.intersection(*(set(rows) for rows in by_symbol.values())))
    if not common:
        return None
    def matrix(field: str) -> np.ndarray:
        return np.asarray([[by_symbol[symbol][ts][field] for symbol in SYMBOLS] for ts in common], dtype=float)
    dates = tuple(datetime.fromtimestamp(ts / 1000.0, UTC) for ts in common)
    return dates, matrix("high_price"), matrix("low_price"), matrix("close_price"), matrix("quote_volume_usdt"), np.asarray(common, dtype=np.int64)


def _actual_case_diagrams(root: Path) -> dict[str, str] | None:
    """Create post-selection descriptive charts from the locked archive."""

    loaded = _load_archive_arrays()
    if loaded is None:
        return None
    dates, high, low, close, volume, _ = loaded
    locked_start = len(dates) // 2 + (len(dates) - len(dates) // 2) // 2
    # Match the diagnostic episode state machine: a new non-zero sign opens an
    # event, and the event ends on the first subsequent sign exit. MFE/MAE are
    # measured over the complete episode, not only its first bar.
    donchian_positions = np.zeros((len(dates), len(SYMBOLS)), dtype=float)
    donchian_channels: dict[tuple[int, int], tuple[float, float]] = {}
    for index in range(91, len(dates)):
        prior_high = high[index - 91 : index - 1].max(axis=0)
        prior_low = low[index - 91 : index - 1].min(axis=0)
        signs = np.sign((close[index - 1] > prior_high).astype(float) - (close[index - 1] < prior_low).astype(float))
        signs[volume[index - 1] < 1_000_000.0] = 0.0
        for asset in np.flatnonzero(signs != 0.0):
            donchian_channels[index, int(asset)] = (float(prior_high[asset]), float(prior_low[asset]))
        donchian_positions[index] = signs
    donchian_events: list[tuple[float, float, int, int, int, float, float, float]] = []
    for index in range(locked_start, len(dates)):
        for asset in range(len(SYMBOLS)):
            direction = donchian_positions[index, asset]
            previous_direction = donchian_positions[index - 1, asset]
            if direction == 0.0 or direction == previous_direction:
                continue
            exit_index = index + 1
            while exit_index < len(dates) and donchian_positions[exit_index, asset] == direction:
                exit_index += 1
            if exit_index >= len(dates):
                continue
            entry = close[index - 1, asset]
            lowest = low[index:exit_index, asset].min()
            highest = high[index:exit_index, asset].max()
            adverse = max(0.0, (entry - lowest) / entry) if direction > 0 else max(0.0, (highest - entry) / entry)
            favorable = max(0.0, (highest - entry) / entry) if direction > 0 else max(0.0, (entry - lowest) / entry)
            upper, lower = donchian_channels[index, asset]
            donchian_events.append((float(favorable), float(adverse), index, exit_index, int(asset), upper, lower, float(direction)))
    if not donchian_events:
        return None
    mfe_case = max(donchian_events, key=lambda item: (item[0], -item[2], -item[3]))
    mae_case = max(donchian_events, key=lambda item: (item[1], -item[2], -item[3]))
    mfe_start = dates[mfe_case[2] - 1].astimezone(timezone(timedelta(hours=9))).strftime("%Y-%m-%d")
    mfe_end = dates[mfe_case[3] - 1].astimezone(timezone(timedelta(hours=9))).strftime("%Y-%m-%d")
    # Regression guard for the known locked-test archive case.
    assert (SYMBOLS[mfe_case[4]], mfe_start, mfe_end, round(mfe_case[0] * 100, 2)) == ("ADA", "2026-06-01", "2026-06-06", 34.55)
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 5.5), sharex=False)
    for axis, case, label in zip(axes, (mfe_case, mae_case), ("locked-test MFE 최대", "locked-test MAE 최대")):
        favorable, adverse, index, exit_index, asset, upper, lower, direction_value = case
        left = max(0, index - 96)
        right = min(len(dates) - 1, exit_index - 1)
        x = np.arange(left, right + 1)
        axis.plot(x, close[left : right + 1, asset], color="#1464B4", lw=1.8, label="종가")
        axis.axhline(upper, color="#D66A1F", ls="--", lw=1.2, label="신호 직전 90일 상단")
        axis.axhline(lower, color="#008F83", ls="--", lw=1.2, label="신호 직전 90일 하단")
        axis.scatter([index - 1], [close[index - 1, asset]], color="#B42318", s=42, zorder=4, label="돌파 신호 종가")
        axis.axvspan(index, right, color="#D9E2EC", alpha=.35, label="완전 episode 구간")
        start_kst = dates[index - 1].astimezone(timezone(timedelta(hours=9))).strftime("%Y-%m-%d")
        end_kst = dates[exit_index - 1].astimezone(timezone(timedelta(hours=9))).strftime("%Y-%m-%d")
        direction = "long" if direction_value > 0 else "short"
        axis.set_title(f"{label}: {SYMBOLS[asset]}USDT · KST {start_kst}~{end_kst} · {direction} · MFE {favorable:.2%} / MAE {adverse:.2%}", fontsize=9)
        axis.grid(alpha=.2); axis.legend(loc="best", fontsize=6.5, ncol=3)
    axes[-1].set_xlabel("locked-test archive row (신호는 다음 적용 row에 반영)")
    fig.tight_layout(); path = root / "donchian_actual.png"; fig.savefig(path, dpi=170); plt.close(fig)

    scores = np.full(len(SYMBOLS), np.nan)
    pv_index = None
    for index in range(61, len(dates)):
        if index < locked_start or (index - locked_start) % 2:
            continue
        momentum = close[index - 1] / close[index - 61] - 1.0
        mean_volume = volume[index - 60 : index].mean(axis=0)
        scores = momentum * np.divide(volume[index - 1], mean_volume, out=np.zeros_like(mean_volume), where=mean_volume > 0)
        eligible = volume[index - 1] >= 20_000_000.0
        ordered = np.argsort(scores)
        if np.isfinite(scores).all() and eligible[ordered[0]] and eligible[ordered[-1]]:
            pv_index = index
            break
    if pv_index is None:
        return {"donchian_actual": str(path)}
    ordered = np.argsort(scores)
    bottom, top = int(ordered[0]), int(ordered[-1])
    fig, axis = plt.subplots(figsize=(8.5, 3.5))
    colors_ = ["#D66A1F" if i == bottom else "#008F83" if i == top else "#B7C5D3" for i in range(len(SYMBOLS))]
    bars = axis.bar(SYMBOLS, scores, color=colors_)
    axis.axhline(0, color="#344054", lw=.8)
    axis.set_ylabel("60일 momentum × relative quote volume")
    kst = dates[pv_index - 1].astimezone(__import__("datetime").timezone(__import__("datetime").timedelta(hours=9))).strftime("%Y-%m-%d")
    axis.set_title(f"실제 locked-test PV 리밸런싱 · KST {kst} · long {SYMBOLS[top]} / short {SYMBOLS[bottom]}", fontsize=10)
    axis.tick_params(axis="x", labelrotation=35)
    for bar, value in zip(bars, scores):
        axis.text(bar.get_x() + bar.get_width() / 2, value + (0.03 if value >= 0 else -0.03), f"{value:.2f}", ha="center", va="bottom" if value >= 0 else "top", fontsize=6)
    axis.grid(axis="y", alpha=.2); fig.tight_layout(); pv_path = root / "pv_actual.png"; fig.savefig(pv_path, dpi=170); plt.close(fig)
    return {"donchian_actual": str(path), "pv_actual": str(pv_path), "pv_meta": f"{SYMBOLS[top]}USDT long / {SYMBOLS[bottom]}USDT short · KST {kst} · top/bottom score {scores[top]:.4f}/{scores[bottom]:.4f} · momentum {momentum[top]:.2%}/{momentum[bottom]:.2%} · relative volume {volume[pv_index-1,top]/mean_volume[top]:.2f}x/{volume[pv_index-1,bottom]/mean_volume[bottom]:.2f}x"}


@contextmanager
def diagrams() -> Iterator[dict[str, object]]:
    with TemporaryDirectory(prefix="top_strategy_") as tmp:
        root = Path(tmp)
        font_manager.fontManager.addfont(str(REGULAR))
        korean_font = font_manager.FontProperties(fname=str(REGULAR)).get_name()
        plt.rcParams.update({"font.family": korean_font, "axes.unicode_minus": False})
        paths: dict[str, object] = {}
        fig, ax = plt.subplots(figsize=(8.5, 3.2))
        labels = ["Donchian\nraw", "Donchian\nstress", "Price-volume\nraw", "Price-volume\nstress"]
        diagnostic_segments = CURRENT_DIAGNOSTIC.get("strategies", {})
        donchian_locked = diagnostic_segments.get("raw_top_donchian", {}).get("segments", {}).get("locked_test", {})
        price_volume_locked = diagnostic_segments.get("paper_gate_top_price_volume", {}).get("segments", {}).get("locked_test", {})
        values = [
            100 * float(donchian_locked.get("base", {}).get("net_return", 0.55465)),
            100 * float(donchian_locked.get("stress", {}).get("net_return", 0.525866)),
            100 * float(price_volume_locked.get("base", {}).get("net_return", 0.204189)),
            100 * float(price_volume_locked.get("stress", {}).get("net_return", 0.153315)),
        ]
        bars = ax.bar(labels, values, color=["#1464B4", "#7EA6CE", "#008F83", "#79BBB5"], width=.62)
        ax.axhline(0, color="#344054", lw=.8)
        ax.set_ylabel("locked OOS net return (%)")
        ax.set_title("두 후보의 잠금 테스트 수익률: raw vs stress")
        ax.grid(axis="y", alpha=.22)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 1.2, f"{value:.4f}%", ha="center", fontsize=9)
        fig.tight_layout(); paths["returns"] = root / "returns.png"; fig.savefig(paths["returns"], dpi=160); plt.close(fig)

        fig, ax = plt.subplots(figsize=(8.5, 3.1))
        x = np.arange(34); price = 100 + np.cumsum(np.array([.8, .5, -.3, .6, .9, -.2, .4, .2, -.7, .8, 1.0, .4, -.4, -.2, .2, 1.2, 1.0, .7, -.2, .5, .6, -.4, -.8, -.6, .1, .5, 1.1, .6, .4, -.7, .7, 1.0, .8, .4])); upper = np.maximum.accumulate(price); lower = np.minimum.accumulate(price)
        ax.plot(x, price, color="#1464B4", lw=2, label="종가")
        ax.plot(x, np.r_[np.repeat(np.nan, 5), np.maximum.accumulate(price[:-5])], color="#D66A1F", lw=1.3, label="이전 90일 상단 개념")
        ax.fill_between(x, lower - 1, upper + 1, color="#EAF2F8", alpha=.65)
        ax.scatter([15, 26], [price[15], price[26]], color="#008F83", s=48, zorder=3)
        ax.set_title("Donchian-90: 상단 돌파 후 다음 일부터 포지션 반영 (도식)"); ax.legend(loc="upper left", fontsize=8); ax.grid(alpha=.2); fig.tight_layout()
        paths["donchian"] = root / "donchian.png"; fig.savefig(paths["donchian"], dpi=160); plt.close(fig)

        fig, ax = plt.subplots(figsize=(8.5, 3.1))
        names = ["A", "B", "C", "D", "E", "F"]; score = [-.4, .2, 1.7, -.8, 1.1, -.1]; colors_ = ["#D66A1F" if v < 0 else "#008F83" for v in score]
        ax.bar(names, score, color=colors_); ax.axhline(0, color="#344054", lw=.8); ax.set_ylabel("60일 모멘텀 × 상대 거래대금"); ax.set_title("Price–Volume-60: 횡단면 순위로 롱/숏 결정 (도식)"); ax.grid(axis="y", alpha=.2); fig.tight_layout()
        paths["pv"] = root / "pv.png"; fig.savefig(paths["pv"], dpi=160); plt.close(fig)

        fig, ax = plt.subplots(figsize=(8.5, 3.0))
        example = path_risk_example(); x = np.arange(len(example.prices)); path = np.array(example.prices); ax.plot(x, path, color="#1464B4", lw=2.4, marker="o", ms=3.5); ax.axhline(example.barrier, color="#B42318", ls="--", label="설명용 장벽 82 · 실제 청산가 아님"); ax.scatter([example.breach_index], [path[example.breach_index]], color="#B42318", s=64, zorder=3); ax.scatter([example.recovery_index], [path[example.recovery_index]], color="#008F83", s=64, zorder=3); ax.text(4.2, 78, "먼저 장벽 통과", color="#B42318", fontsize=8); ax.text(9.2, 104.5, "나중에 방향 적중", color="#008F83", fontsize=8); ax.set_xticks([]); ax.set_yticks([]); ax.set_title("경로 위험: 최종 방향이 맞아도 먼저 청산되면 회복할 수 없음"); ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=.2); fig.tight_layout()
        paths["path"] = root / "path.png"; fig.savefig(paths["path"], dpi=160); plt.close(fig)
        actual = _actual_case_diagrams(root)
        if actual is not None:
            paths.update({key: Path(value) for key, value in actual.items() if key.endswith("_actual")})
            paths["actual_pv_meta"] = actual.get("pv_meta", "")
        yield paths


def header_footer(canvas: object, doc: SimpleDocTemplate) -> None:
    canvas.saveState()  # type: ignore[attr-defined]
    if doc.page == 1:
        canvas.setFillColor(NAVY); canvas.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
        canvas.setStrokeColor(colors.HexColor("#49657F")); canvas.setFillColor(colors.HexColor("#C9D8E8"))
    else:
        canvas.setStrokeColor(GRID); canvas.setFillColor(MUTED)
    canvas.setLineWidth(.4); canvas.line(16 * mm, 13 * mm, PAGE_W - 16 * mm, 13 * mm)
    canvas.setFont("Malgun", 7); canvas.drawString(16 * mm, 8 * mm, "strategy-arena · 전략 특화 연구자료 · public-api paper-only")
    canvas.drawRightString(PAGE_W - 16 * mm, 8 * mm, f"{doc.page}")
    canvas.restoreState()  # type: ignore[attr-defined]


def build() -> Path:
    global CURRENT_DIAGNOSTIC, CURRENT_FRONTIER, CURRENT_ADAPTIVE, CURRENT_ADAPTIVE_5M
    CURRENT_DIAGNOSTIC = json.loads(DIAGNOSTIC_PATH.read_text(encoding="utf-8")) if DIAGNOSTIC_PATH.exists() else {}
    CURRENT_FRONTIER = json.loads(FRONTIER_PATH.read_text(encoding="utf-8")) if FRONTIER_PATH.exists() else {}
    CURRENT_ADAPTIVE = json.loads(ADAPTIVE_PATH.read_text(encoding="utf-8")) if ADAPTIVE_PATH.exists() else {}
    CURRENT_ADAPTIVE_5M = json.loads(ADAPTIVE_5M_PATH.read_text(encoding="utf-8")) if ADAPTIVE_5M_PATH.exists() else {}
    register_fonts(); OUTPUT.parent.mkdir(parents=True, exist_ok=True); st = styles(); story: list[object] = []
    frontier_locked = CURRENT_FRONTIER.get("locked_test", {}).get("metrics", {})
    frontier_validation = CURRENT_FRONTIER.get("selected_validation", {})
    frontier_development = CURRENT_FRONTIER.get("selected_development", {})
    frontier_leverage = float(CURRENT_FRONTIER.get("locked_test", {}).get("leverage", 0.0))
    frontier_locked_base = 100 * float(frontier_locked.get("base", {}).get("net_return", 0.0))
    frontier_locked_stress = 100 * float(frontier_locked.get("stress", {}).get("net_return", 0.0))
    frontier_locked_stress_mdd = 100 * float(frontier_locked.get("stress", {}).get("max_drawdown", 0.0))
    frontier_validation_stress = 100 * float(frontier_validation.get("stress", {}).get("net_return", 0.0))
    frontier_validation_mdd = 100 * float(frontier_validation.get("stress", {}).get("max_drawdown", 0.0))
    frontier_dev_hits = int(frontier_development.get("market_ohlc_proxy_liquidation_hit_count", -1))
    frontier_test_hits = int(frontier_locked.get("market_ohlc_proxy_liquidation_hit_count", -1))
    higher_hit_leverages = [
        float(candidate.get("leverage", 0.0))
        for candidate in CURRENT_FRONTIER.get("frontier_candidates", [])
        if float(candidate.get("leverage", 0.0)) > frontier_leverage and int(candidate.get("development_liquidation_hits", 0)) > 0
    ]
    first_higher_hit = min(higher_hit_leverages) if higher_hit_leverages else None
    first_higher_hit_label = f"{first_higher_hit:g}" if first_higher_hit is not None else "확인 불가"
    adaptive = _adaptive_summary(CURRENT_ADAPTIVE, ADAPTIVE_PATH)
    # Keep the existing page composition readable while sourcing every value
    # from the current report objects above.
    adaptive_interval = str(adaptive["interval"])
    adaptive_trials = int(adaptive["trials"])
    adaptive_validation_stress = float(adaptive["validation_stress"])
    adaptive_locked_stress = float(adaptive["locked_stress"])
    adaptive_pbo = float(adaptive["pbo"])
    adaptive_dsr = float(adaptive["dsr"])
    with diagrams() as d:
        story.extend([Spacer(1, 13 * mm), p("strategy-arena 최고 수익 후보 해부<br/>— 1D Donchian-90 / Price–Volume-60", st["cover"]), Spacer(1, 9 * mm), p("우리 탐색에서 실제로 살아남은 두 후보의 구조·수익·실패 게이트·차트 실행 규칙을 한 문서로 고정한다.", st["cover_sub"]), Spacer(1, 80 * mm), p("판정일 2026-09-04<br/>공공 Bitget 데이터 기반 · 주문 실행 없음 · 교육 및 연구용", st["cover_sub"]), PageBreak()])

        story += [p("1. 먼저 결론: 수익 후보는 있으나, 실전 승격 전략은 아직 없다", st["h1"]), p("현재 자료에서 가장 높은 잠금 OOS raw 수익 후보는 Donchian-90이다. 그러나 PBO/DSR 게이트를 통과하지 못해 ‘검증된 실전 전략’이라고 부를 수 없다. Price–Volume-60은 paper pass였지만 PBO와 DSR이 모두 실패했다. <b>selection_independent=false</b>이므로 두 수치는 post-selection descriptive reproduction이며 eligible/live 후보가 아니다.", st["body"]), Image(str(d["returns"]), width=171 * mm, height=64 * mm), Spacer(1, 3 * mm), table([["후보", "잠금 OOS raw", "stress", "MDD", "blocks", "판정"], ["1D Donchian-90", "+55.4650%", "+52.5866%", "9.1505%", "2", "paper fail"], ["Price–Volume-60", "+20.4189%", "+15.3315%", "13.3705%", "3", "paper pass; PBO/DSR fail"]], st, [31 * mm, 26 * mm, 25 * mm, 20 * mm, 17 * mm, 43 * mm]), Spacer(1, 4 * mm), p("잠금 일봉 market-proxy diagnostic: Donchian 111 events, eventual win 45.95%, MAE50 2.334%, MAE95 7.619%; Price–Volume 221 events, eventual win 48.42%, MAE50 2.865%, MAE95 21.214%. 두 결과 모두 proxy hit 0이지만 mark/MMR 청산 검증은 아니다. +55.4650%는 고정 5배 수익이 아니라 대부분 1배 미만의 동적 실효배율에서 나온 수치다.", st["small"]), Spacer(1, 3 * mm), p(f"사용자 가설에 대한 직접 반례 — 고정 5배 반사실: Donchian base +452.19%, stress +373.73%처럼 보이지만 market-OHLC proxy hit 1회가 먼저 발생했고, 청산 뒤에도 수익을 계속 합산한 실행불가능 경로다. Price–Volume은 base −6.74%, stress −33.12%, proxy hit 6회다. 현재 frontier는 고정 {frontier_leverage:g}배 Donchian으로, validation stress {frontier_validation_stress:+.2f}% / stress MDD {frontier_validation_mdd:.2f}%, locked base {frontier_locked_base:+.4f}% / stress {frontier_locked_stress:+.4f}% / stress MDD {frontier_locked_stress_mdd:.2f}%다. development/test market-OHLC proxy hit는 {frontier_dev_hits}/{frontier_test_hits}회이며, {first_higher_hit_label}배 이상은 development hit가 발생한다. 이 값은 compounded close-to-close account-equity proxy이지 exact Bitget account return이 아니다.", st["callout"]), Spacer(1, 2 * mm), p("핵심 문장: ‘버티면 결국 방향을 맞힌다’는 레버리지 전략의 안전성 증명이 아니다. 경로 중간의 손실·청산·증거금 규칙이 최종 방향보다 먼저 계좌를 결정한다.", st["callout"]), PageBreak()]

        story += [p("2. 증거의 범위와 탐색 편향", st["h1"]), p("원자료는 <b>bitget_history_full_daily_20260824_v2</b>의 공통 1,705행이다. 학습 852행, 개발 1,278행, 잠금 테스트 427행으로 분리했고, 기존 139,536개 탐색에서 validation으로 후보를 선택한 뒤 locked test를 한 번 읽었다.", st["body"]), table([["항목", "현재 증거"], ["탐색 후보 수", "139,536개"], ["global PBO", "41.4286% (CSCV 70 splits)"], ["DSR 기준", "0.95 이상이어야 통과"], ["live_promotable", "0 (두 후보 모두)"], ["체결/청산 데이터", "시장 OHLC 기반, mark·funding·maintenance margin 미포함"]], st, [52 * mm, 110 * mm]), Spacer(1, 5 * mm), p("selection_independent=false. 따라서 아래의 두 후보 수치는 새 독립 발견이 아니라, validation 선택 이후 locked test를 다시 산출한 post-selection descriptive reproduction이다. 두 후보 모두 eligible=false, live=false다.", st["callout"]), Spacer(1, 4 * mm), p("139,536개 중 좋은 결과만 골라보면 우연히 좋은 곡선이 나올 확률이 커진다. 그래서 raw 수익률만으로는 부족하고, 스트레스 수익률·PBO·DSR·경로 생존 검증이 함께 필요하다.", st["body"]), p("이 문서의 수익률은 고정 명목금액 close-to-close 근사다. 현재 구현은 row i 신호를 close[i-1]→close[i] 수익에 적용하는 방식이므로, 실전의 다음 봉 체결과 동일하지 않다. 손익을 계좌 복리로 재투자한 수익률도 아니며, 실제 Bitget 유지증거금 계산 결과도 아니다.", st["callout"]), p("diagnostic source: work/bitget_top_strategy_diagnostic_report.json · generated_at 2026-09-04T07:44:17.759637+00:00 UTC · observed mode=public-archive-paper-only.", st["small"]), PageBreak()]

        story += [p("3. 후보 A: 1D Donchian-90의 정확한 규칙", st["h1"]), p("각 코인의 직전 90개 일봉에서 상단 최고가와 하단 최저가를 계산한다. 현재 봉의 종가가 직전 상단을 넘으면 long, 직전 하단을 깨면 short 방향을 만든다. 방향은 매일 비노출 상태에서 재산출된다. 활성 신호는 동일가중 ±1/N이고 1m USDT quote-volume 유동성 guard를 코인별로 적용한다. 특정 코인이 guard를 통과하지 못하면 그 코인의 포지션만 0이며, 다른 코인은 계속 계산한다.", st["body"]), table([["구성", "구체적 의미"], ["lookback", "직전 90일 상·하단 (현재 봉 포함 금지)"], ["direction", "상단 돌파 long / 하단 이탈 short"], ["portfolio", "활성 신호 동일가중 ±1/N; 1m USDT quote-volume guard"], ["guard 실패", "개별 illiquid 자산만 해당 row 포지션=0; pair 전체를 0으로 만들지 않음"], ["rebalance", "1일 단위"], ["risk axis", "L=5는 탐색 최대값이며 고정 사용 선언이 아님"], ["vol target", "lagged 40일 변동성, 0.5% 목표"], ["거래비용", "엔진의 base/stress 비용 가정"], ["청산", "market OHLC liquidation proxy일 뿐"]], st, [42 * mm, 120 * mm]), Spacer(1, 5 * mm), Image(str(d.get("donchian_actual", d["donchian"])), width=171 * mm, height=62 * mm), p("실제 archive locked-test의 사후 설명 사례다. 진단과 같은 complete episode 정의로 MFE 최대·MAE 최대 event를 결정론적으로 각각 1개 선택해 종목·KST 기간·직전 90일 채널을 표시했으며, 성과 증명이 아니라 규칙을 읽는 예시다. archive가 없을 때만 위 이미지는 정적 schematic으로 대체된다. 다음 봉 체결은 재현 권고이며 기존 잠금 수익률 계산과 동일하지 않다.", st["small"]), PageBreak()]

        story += [p("4. Donchian-90을 차트에서 읽는 순서", st["h1"]), p("① 일봉으로 고정한다. ② 오늘 봉을 계산에 섞지 않고 직전 90일 상·하단을 표시한다. ③ 종가 기준 새 돌파가 확인되었는지 본다. ④ 매일 신호를 다시 계산한다. ⑤ 그날 새 돌파가 없으면 다음 적용 row의 포지션은 0이다. 기존 포지션을 반대 돌파까지 계속 보유하는 규칙이 아니다.", st["body"]), table([["차트에서 볼 것", "실제 규칙에 맞는 판단"], ["상단·하단이 평평하고 가격이 범위 안", "신호 없음. 다음 적용 row=0"], ["종가가 상단을 넘음", "long 후보. 다음 적용 row에서 활성화"], ["종가가 상단을 넘었지만 긴 윗꼬리·거래량 급감", "규칙상 신호와 별개인 위험 경고. 임의 필터는 연구에 추가해야 함"], ["그날 새 상·하단 돌파 없음", "다음 적용 row에서 0; 이전 포지션을 보유하지 않음"], ["활성 코인 수 N", "동일가중 ±1/N 및 1m USDT quote-volume guard"]], st, [62 * mm, 100 * mm]), Spacer(1, 7 * mm), p("이 후보의 +55.4650%는 ‘신호가 매번 맞았다’는 뜻이 아니다. 실제 자료는 MDD 9.1505%, positive blocks 2, DSR probability 1.468%로, 탐색편향 조정 후 우연을 배제하지 못했다.", st["callout"]), PageBreak()]

        story += [p("5. 후보 B: Price–Volume-60의 정확한 규칙", st["h1"]), p("각 코인의 60일 가격 모멘텀과 상대 quote volume을 곱한 점수를 계산한다. 매 리밸런싱 시 횡단면에서 1위 코인 하나를 long, 최하위 코인 하나를 short로 뽑고 각각 목표 비중 0.5를 둔다. 20m USDT quote-volume guard를 적용하며, 한쪽이라도 실패하면 pair 전체를 0으로 만든다. 리밸런싱 주기는 2일이다.", st["body"]), Image(str(d.get("pv_actual", d["pv"])), width=171 * mm, height=62 * mm), p(f"실제 archive locked-test의 첫 번째 양쪽 guard 통과 리밸런싱 row를 표시한 사후 설명 사례다. {d.get('actual_pv_meta', '실제 사례 메타데이터를 읽지 못해 정적 schematic으로 대체됨')} 성과 증명이 아니라 top1/bottom1 순위와 계산 원리를 읽는 예시다.", st["small"]), table([["구성", "구체적 의미"], ["score", "60일 momentum × relative quote volume"], ["portfolio", "cross-sectional top1 long / bottom1 short"], ["weight", "각 방향 0.5 목표"], ["guard", "20m USDT quote-volume; 한쪽 실패면 전체 pair=0"], ["rebalance", "2일 단위"], ["risk axis", "L=5는 탐색 최대 cap. 잠금 후보 overlay도 L=5이며 실효배율 median 0.6009x, max 1.2715x"], ["vol target", "lagged 40일 변동성, 1% 목표"], ["판정", "paper pass, PBO/DSR fail; live_promotable=0"]], st, [42 * mm, 120 * mm]), PageBreak()]

        story += [p("6. Price–Volume-60을 차트에서 읽는 순서", st["h1"]), p("① 같은 시점의 모든 후보 코인에 대해 60일 가격 변화와 quote volume 상대값을 계산한다. ② 곱셈 점수가 높은 한 종목과 낮은 한 종목만 고른다. ③ 절대 row index가 짝수인 리밸런싱 row에서만 2일 주기로 교체하고, 그 사이 row에서는 직전 포지션을 유지한다. 순위가 바뀌었는지는 리밸런싱 조건이 아니다. ④ 코인 하나의 차트만 보고 ‘좋아 보인다’고 진입하지 말고, 횡단면 순위를 계산한다.", st["body"]), table([["오독", "정확한 해석"], ["거래량이 많으면 무조건 long", "가격 모멘텀과 상대 quote volume의 곱이 필요"], ["가장 많이 오른 코인을 매수", "60일 모멘텀·거래대금으로 전체 코인 중 순위를 비교"], ["순위가 하루 흔들리면 즉시 교체", "절대 row index 기준 2일마다 rebalance; 중간 row는 유지"], ["수익률이 플러스면 성공", "잠금 OOS에서는 플러스지만 PBO 41.4286% 환경에서 선택편향 확인 필요"]], st, [52 * mm, 110 * mm]), Spacer(1, 8 * mm), p("price-volume 후보는 raw +20.4189%, stress +15.3315%, MDD 13.3705%, blocks 3이다. paper gate는 통과했지만 DSR probability는 0.0090678%로 매우 낮고 PBO도 실패했다.", st["callout"]), PageBreak()]

        story += [p("7. 레버리지 5배와 ‘버티기’ 가설의 오류", st["h1"]), p("레버리지는 방향성을 맞혔을 때 손익을 확대하지만, 반대 방향의 일시적 움직임도 같은 비율로 확대한다. 5배에서 단순화한 손실 한계는 약 20% 부근이며, 실제 거래소 청산은 유지증거금·수수료·펀딩·mark price·강제감소 등으로 달라진다.", st["body"]), Image(str(d["path"]), width=171 * mm, height=61 * mm), table([["질문", "이 자료가 답할 수 있는 범위"], ["최종 방향이 맞았나?", "close-to-close 결과만 보여줌"], ["중간에 계좌가 살아남았나?", "정확한 mark-price·margin 검증 없음"], ["언제 TP/SL인가?", "고정 TP/SL 없는 엔진; 답하지 않음"], ["5배로 안전한가?", "안전성 입증 아님. L=5는 탐색 상한"], ["고정 5배 반사실", "Donchian +452.19% / stress +373.73%처럼 보이나 proxy hit 1회 후 계속 합산"], ["선택된 고정 frontier", f"고정 {frontier_leverage:g}배: validation stress {frontier_validation_stress:+.2f}% / MDD {frontier_validation_mdd:.2f}%; locked base {frontier_locked_base:+.4f}% / stress {frontier_locked_stress:+.4f}% / stress MDD {frontier_locked_stress_mdd:.2f}%; proxy hit {frontier_test_hits}"], ["회복까지 버틸 수 있나?", "MAE/MFE·recovery time 새 검증 필요"]], st, [56 * mm, 106 * mm]), PageBreak()]

        story += [p("8. 계좌 관점의 최소 계산", st["h1"]), p("실제 진입 전에 방향보다 먼저 다음을 계산한다. 명목 포지션 크기는 계좌자산 × 허용손실률 ÷ (진입가와 무효화가의 차이)로 정하고, 레버리지는 필요한 증거금의 표시일 뿐 손실 허용액을 늘리는 근거가 아니다. 동적 overlay의 공식은 <b>min(5, target / max(lagged_40d_strategy_vol, target/5))</b>이다.", st["body"]), table([["계산", "식 / 체크"], ["동적 레버리지", "min(5, target / max(lagged_40d_strategy_vol, target/5))"], ["가격 손실률", "long: (진입가−무효화가)/진입가; short는 반대"], ["예상 계좌손실", "명목 포지션 × 가격 손실률 + 수수료 + 펀딩"], ["강제청산 여유", "mark price 기준 유지증거금·수수료·갭 포함"], ["동시 노출", "top1/bottom1 외 다른 포지션과 상관·총 증거금 합산"], ["주문 중지", "mark 데이터·잔고·유지증거금 계산이 없으면 주문 금지"]], st, [50 * mm, 112 * mm]), Spacer(1, 7 * mm), p("우리 현재 결과는 고정 명목금액 연구 결과이므로 계좌 손실률로 변환할 수 없다. ‘5배를 걸고 버티기’는 이 문서의 결론이 아니다.", st["callout"]), PageBreak()]

        story += [p("9. 실전 차트 판독 체크카드 — Donchian", st["h1"]), table([["순서", "체크", "통과 조건"], ["1", "시간봉", "일봉 신호를 5분봉 타점으로 임의 변환하지 않음"], ["2", "창 고정", "직전 90일 상·하단만 사용"], ["3", "돌파", "종가가 상단/하단을 실제로 돌파"], ["4", "체결", "다음 봉 체결·슬리피지 반영 (기존 잠금 수익률과 동일하지 않음)"], ["5", "경로", "MAE가 mark 청산 경계보다 낮은지 확인"], ["6", "실제 종료", "매일 신호 재산출; 새 돌파가 없으면 다음 적용 row=0. 반대 신호까지 계속 보유하지 않음"], ["7", "기록", "신호시각·진입·최대불리·최대유리·종료 이유 저장"]], st, [14 * mm, 39 * mm, 109 * mm]), Spacer(1, 9 * mm), p("돌파가 확인되었어도 5배를 자동 적용하지 않는다. 검증되지 않은 레버리지는 신호의 일부가 아니라 별도의 위험 가설이다.", st["callout"]), PageBreak()]

        story += [p("10. 실전 차트 판독 체크카드 — Price–Volume", st["h1"]), table([["순서", "체크", "통과 조건"], ["1", "모집단", "동일 시점의 알트코인 후보군이 완성됨"], ["2", "모멘텀", "60일 가격 변화가 계산됨"], ["3", "거래대금", "quote volume을 상대화함"], ["4", "순위", "top1 long / bottom1 short가 명확함"], ["5", "유동성", "20m guard를 양쪽 모두 통과; 한쪽 실패면 market-neutral 전체 row=0"], ["6", "리밸런싱", "절대 row index 기준 2일마다 교체; 중간 row는 직전 포지션 유지"], ["7", "경로", "MAE·mark 청산 proxy·비용을 별도로 확인"], ["8", "기록", "순위·점수·노출·종료 이유 저장"]], st, [14 * mm, 39 * mm, 109 * mm]), Spacer(1, 9 * mm), p("한 코인의 차트만 보고 이 전략을 재현할 수 없다. 이 후보의 핵심은 ‘좋은 차트’가 아니라 코인 전체를 같은 시점에 정렬한 순위다.", st["callout"]), PageBreak()]

        story += [p("11. 의사결정 트리: 들어갈 때와 나올 때", st["h1"]), p("신호 → 경로 생존 → 비용·증거금 → 실행 순서로 판단한다. 하나라도 확인되지 않으면 포지션을 열지 않는다.", st["body"]), table([["단계", "질문", "다음 행동"], ["A. 신호", "규칙으로 재현 가능한가?", "아니면 진입하지 않음"], ["B. 순위/돌파", "Donchian 돌파 또는 PV 순위와 guard가 유효한가?", "하나라도 실패하면 관찰; PV 한쪽 실패면 pair=0"], ["C. 경로", "최악 예상 MAE가 청산 경계보다 충분히 작은가?", "아니면 레버리지 축소/스킵"], ["D. 비용", "수수료·펀딩·슬리피지 후 기대값이 남는가?", "아니면 스킵"], ["E. 실행", "다음 봉 체결·주문 상태를 확인했나?", "확인 후에만 진입"], ["F. Donchian 실제 종료", "매일 새 돌파가 있는가?", "없으면 다음 적용 row=0; 반대 신호까지 계속 보유하지 않음"]], st, [30 * mm, 80 * mm, 58 * mm]), Spacer(1, 8 * mm), p("고정 TP/SL은 현재 엔진에 없다. 따라서 ‘TP 몇 %, SL 몇 %’를 이 PDF가 전략 사실처럼 제시하면 안 된다.", st["callout"]), PageBreak()]

        story += [p("12. 아직 검증하지 않은 다음 단계: Path-Survival Triple-Barrier", st["h1"]), p("현재 후보를 실전형으로 바꾸려면 신호 수익률이 아니라 진입 이후의 첫 사건을 라벨링해야 한다. 이는 기존 결과가 아니라 새 검증 오버레이다.", st["body"]), table([["사건", "정의"], ["TP", "진입 후 목표 수익 장벽을 먼저 터치"], ["SL", "손실 장벽을 먼저 터치"], ["LIQ", "mark price 기반 청산 경계에 먼저 도달"], ["HORIZON", "어느 장벽도 안 닿으면 시간 만료"], ["동일 봉", "보수적 순서: LIQ > SL > TP"], ["기록", "MAE, MFE, 각 사건까지 걸린 시간, 회복시간"]], st, [34 * mm, 134 * mm]), Spacer(1, 7 * mm), p("선정 기준도 바뀌어야 한다: (1) mark 청산 생존, (2) TP-first 하한, (3) stress 계좌수익 하한, (4) MAE CVaR·MDD·회복시간 순이다. 이 검증이 끝나기 전에는 ‘버티면 된다’고 말할 근거가 없다.", st["callout"]), PageBreak()]

        story += [p("13. 현 시점 운영 카드 — 연구용으로만", st["h1"]), table([["항목", "Donchian-90", "Price–Volume-60"], ["신호", "직전 90일 상·하단 돌파", "60일 momentum × 상대 quote volume"], ["포트폴리오", "동일가중 ±1/N; 1m guard는 개별 자산 적용; illiquid 자산만 0", "top1 long / bottom1 short 각 .5; 20m guard는 market-neutral row-level; 한쪽 실패면 전체 pair=0"], ["주기", "1일", "절대 row index 기준 2일마다 rebalance; 중간 row는 유지"], ["vol target", "lagged 40d, .5%", "lagged 40d, 1%"], ["실효배율 diagnostic", "median .289x / mean .530x / p95 1.526x / max 5x", "median .6009x / mean .632x / p95 1.175x / max 1.2715x"], ["잠금 raw", "+55.4650%", "+20.4189%"], ["잠금 stress", "+52.5866%", "+15.3315%"], ["게이트", "paper fail", "paper pass; PBO/DSR fail"], ["라이브", "승격 금지", "승격 금지"]], st, [37 * mm, 67 * mm, 64 * mm]), Spacer(1, 8 * mm), p("이 표는 ‘현재 연구에서 무엇을 재현할지’를 고정한 운영 카드다. 매매 추천, 수익 보장, 청산 회피 보증이 아니다.", st["callout"]), PageBreak()]

        story += [p("14. 반드시 남겨야 하는 로그와 재검증 게이트", st["h1"]), p("다음 실험은 동일 신호를 mark/funding/maintenance margin까지 포함한 상태로 다시 실행해야 한다. 결과가 좋아도 한 번의 잠금 테스트로 라이브 승격하지 않는다.", st["body"]), table([["로그 필드", "목적"], ["timestamp / symbol / timeframe", "신호 재현"], ["entry / exit / side / leverage", "실제 경로와 노출 복원"], ["mark high/low / market OHLC", "청산 경로 분리"], ["MAE / MFE / T_TP / T_SL / T_LIQ", "경로 생존 검증"], ["funding / fee / slippage", "비용 후 결과"], ["equity / margin ratio", "계좌 단위 위험"], ["결정 당시 모델 버전", "선택편향·재현성"]], st, [62 * mm, 106 * mm]), Spacer(1, 8 * mm), p("최종 게이트: market-OHLC proxy hit=0은 해당 proxy 범위에서 hit가 없었다는 뜻일 뿐 충분조건이 아니다. PBO·DSR·stress·경로 통계·prospective paper 관찰이 함께 필요하다.", st["callout"]), PageBreak()]

        story += [p("15. 최종 판정", st["h1"]), p("우리의 현재 최고 수익 후보는 <b>1D Donchian-90</b>이다. 잠금 OOS raw +55.4650%, stress +52.5866%, MDD 9.1505%라는 숫자는 분명히 매력적이다. 하지만 139,536개 탐색에서 global PBO 41.4286%, DSR probability 1.468%, paper fail이므로 ‘최고의 검증 전략’이 아니라 ‘가장 먼저 경로 생존 검증을 받아야 하는 후보’다.", st["body"]), p("두 번째 후보 Price–Volume-60은 raw +20.4189%, stress +15.3315%, MDD 13.3705%, paper pass였지만 PBO/DSR fail이다. 두 후보 모두 live_promotable=0이다.", st["body"]), p(f"선택 독립성 판정: <b>selection_independent=false</b>. 두 후보는 prior validation 선택 후 locked test를 한 번 읽은 post-selection descriptive reproduction이며, eligible=false·live=false다. 고정 5배 반사실은 Donchian base +452.19% / stress +373.73%·proxy hit 1회, Price–Volume base −6.74% / stress −33.12%·proxy hit 6회로, 청산 이후 합산되는 실행불가능 경로다. 현재 선택된 frontier는 고정 {frontier_leverage:g}배 Donchian이며 validation stress {frontier_validation_stress:+.2f}% / MDD {frontier_validation_mdd:.2f}%, locked base {frontier_locked_base:+.4f}% / stress {frontier_locked_stress:+.4f}% / stress MDD {frontier_locked_stress_mdd:.2f}%, market-OHLC proxy hit {frontier_test_hits}회다. 이는 compounded close-to-close account-equity proxy이지 exact Bitget account return이 아니며, {first_higher_hit_label}배 이상은 development hit가 발생한다.", st["callout"]), p(f"별도 {adaptive_interval} adaptive {adaptive_trials}개 탐색은 validation stress {adaptive_validation_stress:+.2f}%, locked stress {adaptive_locked_stress:+.2f}%, PBO {adaptive_pbo:.2f}%, DSR {adaptive_dsr:.2e}로 후보에서 탈락했다. 이 결과는 fixed-notional 합산이며 계좌 수익률이 아니고, 보고서의 현재 데이터/실행계약 시각을 별도 확인해야 한다.", st["small"]), Spacer(1, 7 * mm), p("이 PDF가 확정하는 것", st["h2"]), p("규칙·수익 증거·탐색 편향·차트 재현 순서·레버리지의 경로 위험.", st["body"]), p("이 PDF가 확정하지 않는 것", st["h2"]), p("실제 TP/SL 값, 특정 알트코인 추천, 5배 청산 회피, 미래 수익, Bitget 실계정 실행.", st["body"]), Spacer(1, 8 * mm), p("다음 결론은 triple-barrier + mark-price 기반 MAE/MFE 검증이 끝난 뒤에만 갱신한다.", st["callout"]), p("진단 출처: work/bitget_top_strategy_diagnostic_report.json · generated_at 2026-09-04T07:44:17.759637+00:00 UTC · mode=public-archive-paper-only. 선택·PBO 맥락: work/bitget_archive_pbo_search_report.json 및 work/bitget_top_leverage_frontier_report.json.", st["small"]), Spacer(1, 4 * mm), p("거래소 운영 참고(조회일 2026-09-04): <link href='https://www.bitget.com/support/articles/12560603808759' color='#1464B4'>Bitget liquidation·mark price·MMR formula</link> · <link href='https://www.bitget.com/support/articles/360058821652' color='#1464B4'>Bitget leverage/risk management·mark liquidation</link> · <link href='https://www.bitget.com/support/articles/12560603838070' color='#1464B4'>Bitget 2025-09 funding formula update</link>.", st["small"])]

        doc = SimpleDocTemplate(str(OUTPUT), pagesize=A4, rightMargin=16 * mm, leftMargin=16 * mm, topMargin=16 * mm, bottomMargin=18 * mm, title="strategy-arena 최고 수익 후보 해부 — 1D Donchian-90 / Price–Volume-60", author="strategy-arena research")
        doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
    return OUTPUT


if __name__ == "__main__":
    print(build())
