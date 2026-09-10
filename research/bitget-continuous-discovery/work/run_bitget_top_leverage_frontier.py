#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pydantic>=2"]
# ///

# ─── How to run ───
# 1. Install uv (if not installed):
#      curl -LsSf https://astral.sh/uv/install.sh | sh
# 2. Run directly (no venv, no pip install needed):
#      uv run run_bitget_top_leverage_frontier.py [--report PATH]
# 3. Or make executable and run:
#      chmod +x run_bitget_top_leverage_frontier.py && ./run_bitget_top_leverage_frontier.py
# ──────────────────

"""Select one fixed-leverage candidate on validation, then test it once."""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, TypeAlias, cast

import numpy as np
import numpy.typing as npt

_WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))
from work.alt_search_engine import segment_metrics
from work.alt_search_models import (
    BASE_FEE,
    BASE_SLIPPAGE,
    LIQUIDATION_RESERVE,
    STRESS_SLIPPAGE,
    FloatMatrix,
    MarketData,
    PathResult,
)
from work.alt_search_signals import build_positions
from work.bitget_archive_daily import ArchiveDailyError, load_verified_daily_archive
from work.run_bitget_top_strategy_diagnostic import (
    DEFAULT_ARCHIVE,
    EXPECTED_SELECTION_SOURCE_SHA256,
    split_bounds,
    strategy_specs,
)

DEFAULT_REPORT: Final[Path] = Path(__file__).with_name("bitget_top_leverage_frontier_report.json")
PRIOR_PBO_REPORT: Final[Path] = Path(__file__).with_name("bitget_archive_pbo_search_report.json")
PRIOR_GLOBAL_PBO: Final[float] = 0.4142857142857143
PRIOR_DONCHIAN_DSR_PROBABILITY: Final[float] = 0.014680027865062376
JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
Matrix2D: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.float64]]


class ReportShapeError(ValueError):
    """Raised when an internal report value is not the declared JSON shape."""


def json_map(value: JsonValue) -> dict[str, JsonValue]:
    if isinstance(value, dict):
        return value
    raise ReportShapeError("expected JSON object")


def _json_number(value: JsonValue) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReportShapeError("expected JSON number")
    return float(value)


def _json_int(value: JsonValue) -> int:
    return int(_json_number(value))


def json_text(value: JsonValue) -> str:
    if isinstance(value, str):
        return value
    raise ReportShapeError("expected JSON text")


@dataclass(frozen=True, slots=True)
class FixedPath:
    """A fixed-leverage path with all intrabar proxy hits retained."""

    base_returns: np.ndarray
    stress_returns: np.ndarray
    leverage: float
    liquidation_hits: tuple[int, ...]
    continues_after_first_liquidation: bool
    evaluations: int = 1


@dataclass
class EvaluationCounter:  # noqa: RUF100  # MUTABLE_OK
    """Mutable instrumentation for the intentionally bounded simulation plan."""

    validation_evaluations: int = 0
    locked_test_evaluations: int = 0


def leverage_grid() -> tuple[float, ...]:
    """Return the predeclared 20-point 0.25x-spaced frontier from 0.25x to 5x."""

    return tuple(float(value) for value in np.arange(0.25, 5.0001, 0.25))


def simulate_fixed_leverage(data: MarketData, positions: FloatMatrix, leverage: float) -> FixedPath:
    """Apply constant leverage and trace causal prior-close-to-bar OHLC proxies.

    Each asset episode starts at its prior close and keeps that entry and
    cumulative adverse extreme. A same-side increase recalculates the
    weighted-average entry and starts a new extreme at the increase bar;
    a reduction preserves the existing entry and extreme. Flat positions and
    reversals reset the episode before the new direction starts.
    """

    if not np.isfinite(leverage) or leverage <= 0.0 or leverage > 5.0:
        raise ValueError("leverage must be in (0, 5]")
    if positions.shape != data.close_price.shape or not np.isfinite(positions).all():
        raise ValueError("positions must be finite and aligned with market data")
    close: npt.NDArray[np.float64] = np.asarray(data.close_price, dtype=np.float64)
    position_values: npt.NDArray[np.float64] = np.asarray(positions, dtype=np.float64)
    returns = np.zeros_like(close)
    returns[1:] = close[1:] / close[:-1] - 1.0
    active = np.abs(positions).sum(axis=1) > 0.0
    scaled = position_values * leverage
    turnover = np.abs(scaled).sum(axis=1)
    turnover[1:] = np.abs(scaled[1:] - scaled[:-1]).sum(axis=1)
    raw = (scaled * returns).sum(axis=1)
    base = raw - turnover * (BASE_FEE + BASE_SLIPPAGE)
    stress = raw - turnover * (BASE_FEE + STRESS_SLIPPAGE)
    threshold = np.full(len(data.dates), np.inf, dtype=np.float64)
    threshold[active] = 1.0 / leverage - LIQUIDATION_RESERVE
    hits = _episode_proxy_hits(data, position_values, threshold)
    return FixedPath(base, stress, leverage, hits, bool(hits and hits[0] < len(base) - 1))


def _episode_proxy_hits(data: MarketData, positions: npt.NDArray[np.float64], threshold: npt.NDArray[np.float64]) -> tuple[int, ...]:
    """Find cumulative entry-to-extreme hits with weighted same-side entries."""

    hit_indices: set[int] = set()
    close = cast(Matrix2D, np.asarray(data.close_price, dtype=np.float64))
    low = cast(Matrix2D, np.asarray(data.low_price, dtype=np.float64))
    high = cast(Matrix2D, np.asarray(data.high_price, dtype=np.float64))
    position_values = cast(Matrix2D, positions)
    date_count = len(data.dates)
    for asset in range(len(data.symbols)):
        entry_price = 0.0
        exposure = 0.0
        extreme = 0.0
        hit = False
        for index in range(1, date_count):
            current = float(cast(np.float64, position_values[index, asset]))
            previous = float(cast(np.float64, position_values[index - 1, asset]))
            current_sign = 1.0 if current > 0.0 else -1.0 if current < 0.0 else 0.0
            previous_sign = 1.0 if previous > 0.0 else -1.0 if previous < 0.0 else 0.0
            if current_sign == 0.0 or current_sign != previous_sign:
                if current_sign == 0.0:
                    entry_price, exposure, extreme, hit = 0.0, 0.0, 0.0, False
                    continue
                entry_price = float(cast(np.float64, close[index - 1, asset]))
                exposure = abs(current)
                extreme = float(cast(np.float64, low[index, asset] if current_sign > 0.0 else high[index, asset]))
                hit = False
            elif abs(current) > exposure:
                added = abs(current) - exposure
                price = float(cast(np.float64, close[index - 1, asset]))
                entry_price = (entry_price * exposure + price * added) / abs(current)
                exposure = abs(current)
                # An increase creates a new effective risk basis.  Do not
                # retroactively apply the prior exposure's extreme to it.
                extreme = float(cast(np.float64, low[index, asset] if current_sign > 0.0 else high[index, asset]))
            else:
                exposure = abs(current)
            if current_sign > 0.0:
                extreme = min(extreme, float(cast(np.float64, low[index, asset])))
                adverse = (entry_price - extreme) / entry_price
            else:
                extreme = max(extreme, float(cast(np.float64, high[index, asset])))
                adverse = (extreme - entry_price) / entry_price
            if not hit and adverse >= threshold[index]:
                hit_indices.add(index)
                hit = True
    return tuple(sorted(hit_indices))


def _metrics(path: FixedPath, start: int, end: int) -> dict[str, JsonValue]:
    """Serialize compounded close-to-close account-equity proxy metrics."""

    proxy = PathResult(path.base_returns, path.stress_returns, np.full(len(path.base_returns), path.leverage), len(path.liquidation_hits), path.liquidation_hits[0] if path.liquidation_hits else -1)
    base, stress, _ = segment_metrics(proxy, start, end)
    hits = tuple(index for index in path.liquidation_hits if start <= index < end)
    return {"return_basis": "compounded_close_to_close_account_equity_proxy", "exact_bitget_account_return": False, "base": {"net_return": base.net_return, "max_drawdown": base.max_drawdown, "sharpe": base.sharpe, "positive_blocks": base.positive_blocks, "observations": base.observations}, "stress": {"net_return": stress.net_return, "max_drawdown": stress.max_drawdown, "sharpe": stress.sharpe, "positive_blocks": stress.positive_blocks, "observations": stress.observations}, "market_ohlc_proxy_liquidation_hit_count": len(hits), "first_market_ohlc_proxy_liquidation_index": hits[0] if hits else None, "counterfactual_continues_after_first_liquidation": bool(hits and end - hits[0] > 1)}


def select_validation_winner(rows: list[Mapping[str, JsonValue]] | tuple[Mapping[str, JsonValue], ...]) -> Mapping[str, JsonValue]:
    """Select one development-safe row by maximum validation stress return."""

    eligible = [row for row in rows if _json_int(row["development_liquidation_hits"]) == 0]
    if not eligible:
        raise ValueError("no zero-hit validation leverage candidate")
    return max(eligible, key=lambda row: _json_number(row["validation_stress_return"]))


def _sha256_file(path: Path) -> str | None:
    """Return an evidence-file digest, or None when the local source is absent."""

    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verified_selection_source_hash(path: Path) -> str:
    """Require the actual selection report to match the diagnostic's pin."""

    source_hash = _sha256_file(path)
    if source_hash is None:
        raise ArchiveDailyError(f"selection source missing: {path}")
    if source_hash != EXPECTED_SELECTION_SOURCE_SHA256:
        raise ArchiveDailyError(f"selection source hash mismatch for {path}")
    return source_hash


def contract_payload(source_evidence_path: Path = PRIOR_PBO_REPORT) -> dict[str, JsonValue]:
    """Return the fail-closed provenance and validation limits for this frontier."""

    source_hash = _verified_selection_source_hash(source_evidence_path)
    return {
        "frontier_selection_independent_of_locked_test": True,
        "source_strategy_selection_independent": False,
        "source_strategy_search_trials": 139_536,
        "prior_global_pbo": PRIOR_GLOBAL_PBO,
        "prior_global_pbo_threshold": 0.05,
        "prior_global_pbo_pass": False,
        "prior_donchian_dsr_probability": PRIOR_DONCHIAN_DSR_PROBABILITY,
        "prior_donchian_dsr_threshold": 0.95,
        "prior_donchian_dsr_pass": False,
        "prior_donchian_paper_pass": False,
        "prior_donchian_live_promotable": False,
        "selected_strategy_paper_pass": False,
        "selected_strategy_live_promotable": False,
        "selected_strategy_fail_closed_reason": "validation stress MDD exceeds the predeclared 25% paper gate; locked-test stress MDD is below 25% but positive blocks are only 2, so the selected strategy remains fail closed",
        "local_pbo_dsr": {
            "status": "not_calculated",
            "reason": "the 40 trials are a leverage frontier diagnostic, not an independent source-strategy search",
        },
        "prior_evidence_source": {
            "path": str(source_evidence_path),
            "sha256": source_hash,
            "sha256_verified": True,
            "pbo_field": "selection_bias.pbo.pbo",
            "dsr_field": "locked_test_results[*].deflated_sharpe_stress.probability",
        },
        "selection_development_scope": "eligible only when the full train+validation development path has zero market-OHLC proxy hits; locked test is never used for selection",
        "return_accounting": "compounded close-to-close account-equity proxy via cumulative product of per-bar returns; not fixed-notional PnL",
        "execution": "close-to-close approximation: position at row i applies close[i]/close[i-1]",
        "liquidation": "episode state starts at the entry prior close; same-side increases use a weighted-average entry and start a new extreme on the increase bar; reductions preserve entry/extreme; flat or reversal resets; each active bar then checks that causal episode state against the day's high/low with fixed leverage and 0.02 reserve",
        "tp_sl": "none; this is the existing signal's exit behavior",
        "counterfactual": "path returns continue after proxy hits only for diagnosis; no-stop returns are not executable",
        "funding": "not reflected",
        "mark_price": "not reflected",
        "maintenance_margin": "not reflected",
        "bitget_liquidation_exact": False,
        "no_live_order_or_account_test": True,
        "fixed_notional_pnl": False,
        "exact_bitget_account_return": False,
    }


def _prefix_data(data: MarketData, end: int) -> MarketData:
    """Limit validation simulations to the data actually available at its boundary."""

    return MarketData(data.dates[:end], data.symbols, data.open_price[:end], data.high_price[:end], data.low_price[:end], data.close_price[:end], data.quote_volume[:end])


def candidate_rows(data: MarketData, start: int, validation_end: int, counter: EvaluationCounter) -> tuple[dict[str, JsonValue], ...]:
    rows: list[dict[str, JsonValue]] = []
    for spec in strategy_specs():
        validation_data = _prefix_data(data, validation_end)
        positions = build_positions(validation_data, spec.candidate)
        for leverage in leverage_grid():
            counter.validation_evaluations += 1
            path = simulate_fixed_leverage(validation_data, positions, leverage)
            validation = _metrics(path, start, validation_end)
            development = _metrics(path, 0, validation_end)
            validation_map = json_map(validation)
            validation_stress = json_map(validation_map["stress"])
            validation_base = json_map(validation_map["base"])
            development_map = json_map(development)
            rows.append({"strategy": spec.name, "candidate": spec.candidate.identifier, "leverage": leverage, "validation_stress_return": validation_stress["net_return"], "validation_base_return": validation_base["net_return"], "validation_max_drawdown": validation_stress["max_drawdown"], "validation_liquidation_hits": validation_map["market_ohlc_proxy_liquidation_hit_count"], "development_liquidation_hits": development_map["market_ohlc_proxy_liquidation_hit_count"], "validation": validation, "development": development})
    return tuple(rows)


def run(archive_root: Path = DEFAULT_ARCHIVE) -> dict[str, JsonValue]:
    """Run validation-only leverage selection and evaluate the winner once on locked test."""

    assets = ("ETH", "BNB", "ADA", "SOL", "DOGE", "XRP", "DOT", "LTC", "LINK", "AVAX", "UNI", "ATOM", "NEAR")
    data, metadata = load_verified_daily_archive(archive_root, assets)
    _, train_end, validation_end, test_end = split_bounds(len(data.dates))
    counter = EvaluationCounter()
    rows = candidate_rows(data, train_end, validation_end, counter)
    selected = select_validation_winner(rows)
    selected_spec = next(spec for spec in strategy_specs() if spec.name == json_text(selected["strategy"]))
    positions = build_positions(data, selected_spec.candidate)
    locked_path = simulate_fixed_leverage(data, positions, _json_number(selected["leverage"]))
    counter.locked_test_evaluations += locked_path.evaluations
    test = _metrics(locked_path, validation_end, test_end)
    return {"mode": "public-archive-paper-only", "observed_at": datetime.now(UTC).isoformat(), "trial_count": len(rows), "evaluation_counts": {"validation": counter.validation_evaluations, "locked_test": counter.locked_test_evaluations, "total": counter.validation_evaluations + counter.locked_test_evaluations}, "leverage_grid": list(leverage_grid()), "selection": {"criterion": "maximum validation stress compounded close-to-close account-equity proxy net return among rows with zero market-OHLC-proxy hits across full train+validation development path", "selected_strategy": selected["strategy"], "selected_candidate": selected["candidate"], "selected_leverage": selected["leverage"], "candidate_count": len(rows), "selection_independent_of_locked_test": True, "locked_test_evaluations": locked_path.evaluations, "return_basis": "compounded_close_to_close_account_equity_proxy"}, "selected_validation": selected["validation"], "selected_development": selected["development"], "locked_test": {"strategy": selected["strategy"], "candidate": selected["candidate"], "leverage": selected["leverage"], "metrics": test, "return_basis": "compounded_close_to_close_account_equity_proxy", "exact_bitget_account_return": False}, "data": {"archive_root": str(archive_root), "snapshot_ms": metadata.snapshot_ms, "common_rows": metadata.common_rows, "integrity_verified": metadata.integrity_verified, "source_hashes": list(metadata.source_hashes), "source_hash_audit": {"integrity_verified": metadata.integrity_verified, "manifest_hashes_recorded": True, "source_hash_count": len(metadata.source_hashes)}, "split": {"train_rows": train_end, "validation_rows": validation_end - train_end, "locked_test_rows": test_end - validation_end}}, "contract": contract_payload(), "frontier_candidates": list(rows)}


def _cli_paths(argv: list[str]) -> tuple[Path, Path]:
    """Parse the two optional paths without an untyped argparse namespace."""

    archive_root = DEFAULT_ARCHIVE
    report = DEFAULT_REPORT
    index = 0
    while index < len(argv):
        flag = argv[index]
        if flag in {"--archive-root", "--report"} and index + 1 < len(argv):
            path = Path(argv[index + 1])
            if flag == "--archive-root":
                archive_root = path
            else:
                report = path
            index += 2
            continue
        raise ReportShapeError(f"unknown or incomplete argument: {flag}")
    return archive_root, report


def main(argv: list[str] | None = None) -> int:
    """Run the paper-only frontier CLI."""

    archive_root, report_path = _cli_paths(sys.argv[1:] if argv is None else argv)
    try:
        run_report = run(archive_root)
    except ArchiveDailyError as error:
        print(f"FRONTIER_ERROR={error}")
        return 1
    _ = report_path.write_text(json.dumps(run_report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(f"FRONTIER_DONE report={report_path} trials=40 locked_test_evaluations=1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
