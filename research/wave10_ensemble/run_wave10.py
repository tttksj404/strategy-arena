from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Final

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.validation.deep_stats import TimedValue, deflated_sharpe
from research.wave8_alternative import run_wave8 as w8
from research.wave9_methods import run_wave9 as w9

BASE_DIR: Final = Path(__file__).resolve().parent
RESULTS_DIR: Final = BASE_DIR / "results"
REPORT_DIR: Final = BASE_DIR / "report"
INITIAL_CAPITAL: Final = 100.0
MIN_ORDER: Final = 5.0
GROSS_CAP: Final = 0.60
FEE_RATE: Final = 0.0006
BASE_SLIPPAGE: Final = 0.0003
STRESS_SLIPPAGE: Final = 0.0006
MC_PATHS: Final = 10_000
BLOCK_PATHS: Final = 1_000
BLOCK_DAYS: Final = 90
OOS_SPLIT: Final = pd.Timestamp("2025-09-30T23:59:59Z")
IDS: Final = ("E10a", "E10b", "E10c", "E10d", "E10e", "E10f")


@dataclass(frozen=True, slots=True)
class Candidate:
    candidate_id: str
    components: tuple[tuple[str, float], ...]
    throttle: bool = False


CANDIDATES: Final[tuple[Candidate, ...]] = (
    Candidate("E10a", (("D9b", 0.5), ("F8d", 0.5))),
    Candidate("E10b", (("D9b", 0.5), ("M10a", 0.5))),
    Candidate("E10c", (("D9b", 0.5), ("P9b", 0.5))),
    Candidate("E10d", (("D9b", 0.4), ("M10a", 0.3), ("F8d", 0.3))),
    Candidate("E10e", (("D9b", 0.5), ("F8d", 0.5)), True),
    Candidate("E10f", (("D9b", 0.5), ("M10a", 0.5)), True),
)

DEFINITIONS: Final[dict[str, str]] = {
    "E10a": "equal blend of D9b downside-volatility and F8d guarded funding spread",
    "E10b": "equal blend of D9b and M10a capital-aware top-3 20-day cross-sectional momentum",
    "E10c": "equal blend of D9b and P9b residual trend",
    "E10d": "40/30/30 blend of D9b, M10a, and F8d",
    "E10e": "E10a with lagged 10% drawdown half-exposure throttle",
    "E10f": "E10b with lagged 10% drawdown half-exposure throttle",
}


def _load() -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict]:
    frames, metadata = w9._load_data()
    close, _volume, funding, _wave8_meta = w8._load_data()
    if not close.index.equals(frames["close"].index):
        raise ValueError("Wave-8 and Wave-9 component indexes do not align")
    return frames, funding, metadata


def _components(frames: dict[str, pd.DataFrame], funding: pd.DataFrame) -> dict[str, tuple[pd.DataFrame, bool]]:
    close = frames["close"]
    d9b = w9._downside_vol(close, 30)
    m10a = w9._top_bottom(close.pct_change(20).shift(1), 3, long_high=True)
    p9b = w9._residual_trend(close, 30)
    f8d = w8._funding_positions(close, funding, "F8d")
    return {"D9b": (d9b, False), "M10a": (m10a, False), "P9b": (p9b, False), "F8d": (f8d, True)}


def _blend(candidate: Candidate, components: dict[str, tuple[pd.DataFrame, bool]], close: pd.DataFrame, funding: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    positions = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    funding_leg = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    for name, weight in candidate.components:
        leg, has_funding = components[name]
        positions = positions.add(weight * leg, fill_value=0.0)
        if has_funding:
            funding_leg = funding_leg.add(weight * leg, fill_value=0.0)
    if not candidate.throttle:
        return positions, funding_leg
    price_returns = close.pct_change().shift(-1).fillna(0.0)
    next_funding = funding.shift(-1).fillna(0.0)
    scaled_positions = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    scaled_funding_leg = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    previous = pd.Series(0.0, index=close.columns)
    equity = INITIAL_CAPITAL
    peak = INITIAL_CAPITAL
    for timestamp in close.index:
        scale = 0.5 if equity / peak - 1.0 < -0.10 else 1.0
        current = positions.loc[timestamp] * scale
        current_funding = funding_leg.loc[timestamp] * scale
        turnover = float((current - previous).abs().sum())
        gross = float((current * price_returns.loc[timestamp]).sum())
        carry = float((-current_funding * next_funding.loc[timestamp]).sum())
        daily = gross + carry - turnover * (FEE_RATE + BASE_SLIPPAGE)
        scaled_positions.loc[timestamp] = current
        scaled_funding_leg.loc[timestamp] = current_funding
        equity *= 1.0 + max(daily, -0.99)
        peak = max(peak, equity)
        previous = current
    return scaled_positions, scaled_funding_leg


def _returns(positions: pd.DataFrame, funding_leg: pd.DataFrame, close: pd.DataFrame, funding: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    price_returns = close.pct_change().shift(-1).fillna(0.0)
    gross = (positions * price_returns).sum(axis=1)
    carry = (-funding_leg * funding.shift(-1).fillna(0.0)).sum(axis=1)
    turnover = positions.diff().abs().sum(axis=1).fillna(positions.abs().sum(axis=1))
    net = (gross + carry - turnover * (FEE_RATE + BASE_SLIPPAGE)).iloc[:-1]
    stress = (gross + carry - turnover * (FEE_RATE + STRESS_SLIPPAGE)).reindex(net.index)
    active = positions.reindex(net.index).abs().sum(axis=1) > 0.0
    return net, stress, active


def _result(candidate: Candidate, positions: pd.DataFrame, funding_leg: pd.DataFrame, frames: dict[str, pd.DataFrame], funding: pd.DataFrame, metadata: dict, seed: int) -> dict:
    close = frames["close"]
    net, stress, active = _returns(positions, funding_leg, close, funding)
    positions = positions.reindex(net.index)
    equity = w9._equity(net)
    metrics = w9._metrics(equity, net)
    oos = w9._oos(net, stress, active)
    mc = w9._mc(net, seed)
    block = w9._block_mdd(net, seed + 17)
    dsr_curve = tuple(TimedValue(pd.Timestamp(ts).to_pydatetime(), float(value)) for ts, value in equity.items())
    dsr = deflated_sharpe(dsr_curve, trials=len(CANDIDATES))
    capital = equity.shift(1).reindex(positions.index).fillna(INITIAL_CAPITAL)
    notionals = positions.abs().mul(capital, axis=0)
    active_notionals = notionals.where(notionals > 0.0).stack()
    min_order = float(active_notionals.min()) if not active_notionals.empty else 0.0
    max_gross = float(positions.abs().sum(axis=1).max())
    gates = {
        "data_validation": bool(metadata["data_valid"]),
        "fixed_universe": tuple(metadata["symbols"]) == tuple(w9.SYMBOLS),
        "oos_split_present": bool(net.index[net.index > OOS_SPLIT].size > 0),
        "selection_independent": False,
        "capital_contract": bool(max_gross <= GROSS_CAP + 1e-9 and min_order >= MIN_ORDER),
        "oos_activity": int(oos["active_days"]) >= 90,
        "oos_positive": float(oos["return"]) > 0.0,
        "oos_sharpe": float(oos["sharpe"]) >= 1.0,
        "positive_blocks": int(oos["positive_blocks"]) >= 2,
        "stress_positive": float(oos["stress_return"]) > 0.0,
        "mc_p05": float(mc["p05"]) > INITIAL_CAPITAL,
        "ruin_probability": float(mc["ruin_probability"]) < 0.05,
        "historical_mdd": metrics["mdd"] <= 0.25,
        "block_mdd_p95": float(block["p95"]) <= 0.25,
        "deflated_sharpe": float(dsr.probability) >= 0.95,
    }
    return {
        "candidate_id": candidate.candidate_id, "definition": DEFINITIONS[candidate.candidate_id],
        "config": {"components": list(candidate.components), "throttle": candidate.throttle}, "data": metadata,
        "metrics": metrics, "oos": oos, "monte_carlo": mc, "block_shuffle": block,
        "deflated_sharpe": {"probability": float(dsr.probability), "score": float(dsr.score), "trials": len(CANDIDATES)},
        "execution": {"max_gross": max_gross, "min_order_usd": min_order, "active_days": int(active.sum())},
        "gates": {name: {"status": "PASS" if value else "FAIL", "passed": bool(value)} for name, value in gates.items()},
        "all_gates_pass": bool(all(gates.values())),
        "equity": [{"timestamp": pd.Timestamp(ts).isoformat(), "value": float(value)} for ts, value in equity.items()],
        "daily_returns": [{"timestamp": pd.Timestamp(ts).isoformat(), "value": float(value)} for ts, value in net.items()],
    }


def _json_safe(value):
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_report(results: list[dict]) -> None:
    lines = ["# Wave-10 ensemble and risk-overlay report", "", "Exploratory evidence only. Component selection was informed by Wave-8/9 results, so selection-independent is false and no live capital is approved.", "", "| Candidate | OOS return | OOS Sharpe | MDD | Stress OOS | MC p05 | Ruin | Block MDD p95 | Min order | Fails |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for result in sorted(results, key=lambda item: item["oos"]["return"], reverse=True):
        failures = ", ".join(name for name, gate in result["gates"].items() if not gate["passed"])
        lines.append(f"| {result['candidate_id']} | {result['oos']['return'] * 100:.2f}% | {result['oos']['sharpe']:.2f} | {result['metrics']['mdd'] * 100:.2f}% | {result['oos']['stress_return'] * 100:.2f}% | ${result['monte_carlo']['p05']:.2f} | {result['monte_carlo']['ruin_probability'] * 100:.2f}% | {result['block_shuffle']['p95'] * 100:.2f}% | ${result['execution']['min_order_usd']:.2f} | {failures or 'none'} |")
    lines.extend(["", "## Verdict", "", "Eligible candidates: none.", "", "Any near-pass candidate is a prospective paper candidate only after a genuinely unseen forward window.", ""])
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report = REPORT_DIR / "wave10_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    aggregate = RESULTS_DIR / "wave10_results.json"
    manifest = {
        "report_sha256": _hash(report), "results_sha256": _hash(aggregate),
        "spec_sha256": _hash(BASE_DIR / "SPEC.md"), "runner_sha256": _hash(BASE_DIR / "run_wave10.py"),
        "result_ids": IDS, "eligible": [], "selection_independent": False,
    }
    (REPORT_DIR / "wave10_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def run(only: str | None = None) -> list[dict]:
    frames, funding, metadata = _load()
    component_map = _components(frames, funding)
    selected = [candidate for candidate in CANDIDATES if only is None or candidate.candidate_id == only]
    if not selected:
        raise ValueError(f"unknown registered candidate: {only}")
    results: list[dict] = []
    candidate_indices = {candidate.candidate_id: index for index, candidate in enumerate(CANDIDATES)}
    for candidate in selected:
        positions, funding_leg = _blend(candidate, component_map, frames["close"], funding)
        result = _result(candidate, positions, funding_leg, frames, funding, metadata, 20_260_721 + candidate_indices[candidate.candidate_id] * 173)
        _write(RESULTS_DIR / f"{candidate.candidate_id}.json", result)
        results.append(result)
        print(f"completed {candidate.candidate_id}: oos={result['oos']['return']:.4f} gates={sum(g['passed'] for g in result['gates'].values())}/{len(result['gates'])}")
    if only is None:
        _write(RESULTS_DIR / "wave10_results.json", {"candidate_ids": IDS, "results": results})
        _write_report(results)
        print("WAVE10_DONE eligible=[]")
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Wave-10 ensemble and risk-overlay research runner")
    parser.add_argument("--only", choices=IDS)
    args = parser.parse_args(argv)
    try:
        run(args.only)
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
