import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.wave1.costs import slippage_rate
from research.wave3.fetch import verify_manifest
from research.wave9_capital_native.run_wave9 import _block_mdd, _bootstrap, _read_frame


BASE_DIR: Final = Path(__file__).resolve().parent
CACHE_DIR: Final = BASE_DIR.parent / "wave3" / "cache"
RESULTS_DIR: Final = BASE_DIR / "results"
REPORT_DIR: Final = BASE_DIR / "report"
SYMBOLS: Final[tuple[str, ...]] = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
OOS_SPLIT: Final = pd.Timestamp("2025-09-30T23:59:59Z")
INITIAL_TOTAL: Final = 100.0
RESERVE: Final = 10.0
INITIAL_ACTIVE: Final = INITIAL_TOTAL - RESERVE
MAKER_FEE: Final = 0.0002
MIN_ORDER: Final = 5.0
GROSS_CAP: Final = 0.60
BORROW_APR: Final = 0.08
BOOTSTRAP_SEED: Final = 20260722


@dataclass(frozen=True, slots=True)
class Candidate:
    candidate_id: str
    family: str
    window: int
    entry: float
    exit: float
    hold_days: int
    long_short: bool


CANDIDATES: Final[tuple[Candidate, ...]] = (
    Candidate("R11a", "rsi", 2, 10.0, 50.0, 5, False),
    Candidate("R11b", "rsi", 2, 10.0, 50.0, 5, True),
    Candidate("R11c", "rsi", 2, 5.0, 50.0, 3, False),
    Candidate("R11d", "rsi", 2, 5.0, 50.0, 3, True),
    Candidate("B11a", "bollinger", 20, 1.5, 0.0, 10, True),
    Candidate("B11b", "bollinger", 20, 2.0, 0.0, 10, True),
    Candidate("B11c", "bollinger", 40, 1.5, 0.0, 15, True),
    Candidate("B11d", "bollinger", 40, 2.0, 0.0, 15, False),
    Candidate("F11a", "funding_reversal", 30, 1.5, 0.0, 3, True),
    Candidate("F11b", "funding_reversal", 30, 2.0, 0.0, 3, True),
    Candidate("F11c", "funding_reversal", 30, 1.5, 0.0, 5, True),
    Candidate("F11d", "funding_reversal", 30, 2.0, 0.0, 5, True),
    Candidate("V11a", "basis", 10, 1.5, 0.5, 3, True),
    Candidate("V11b", "basis", 20, 1.5, 0.5, 5, True),
    Candidate("V11c", "basis", 30, 2.0, 0.5, 5, True),
    Candidate("V11d", "basis", 20, 2.0, 0.25, 10, True),
)


class Wave11Error(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    manifest = json.loads((CACHE_DIR / "manifest.json").read_text(encoding="utf-8"))
    roster = manifest.get("crypto_symbols") if isinstance(manifest, dict) else None
    if not isinstance(roster, list) or any(symbol not in roster for symbol in SYMBOLS):
        raise Wave11Error("fixed major roster is incomplete")
    perp: dict[str, pd.Series] = {}
    spot: dict[str, pd.Series] = {}
    funding: dict[str, pd.Series] = {}
    for symbol in SYMBOLS:
        perp_frame = _read_frame(CACHE_DIR / f"binance_fapi_{symbol}_1d.csv.gz", ("close",))
        spot_frame = _read_frame(CACHE_DIR / f"binance_spot_{symbol}_1d.csv.gz", ("close",))
        funding_frame = _read_frame(CACHE_DIR / f"binance_funding_{symbol}.csv.gz", ("funding_rate",))
        perp[symbol] = pd.to_numeric(perp_frame["close"], errors="coerce")
        spot[symbol] = pd.to_numeric(spot_frame["close"], errors="coerce")
        funding[symbol] = pd.to_numeric(funding_frame["funding_rate"], errors="coerce").resample("1D").sum(min_count=1)
    perp_frame = pd.concat(perp, axis=1, sort=False).sort_index()
    spot_frame = pd.concat(spot, axis=1, sort=False).reindex(perp_frame.index)
    funding_frame = pd.concat(funding, axis=1, sort=False).reindex(perp_frame.index)
    valid = perp_frame.notna() & funding_frame.notna()
    for symbol in SYMBOLS:
        first = valid[symbol].where(valid[symbol]).first_valid_index()
        if first is None or not valid[symbol].loc[first:].all():
            raise Wave11Error(f"data gap on {symbol}")
        if (perp_frame[symbol].loc[first:] <= 0.0).any() or (spot_frame[symbol].loc[first:] <= 0.0).any():
            raise Wave11Error(f"non-positive price on {symbol}")
    common = valid.all(axis=1)
    if not common.any():
        raise Wave11Error("no common market interval")
    index = common[common].index
    return perp_frame.loc[index], spot_frame.loc[index], funding_frame.loc[index], valid.loc[index]


def _rsi(close: pd.DataFrame, window: int) -> pd.DataFrame:
    delta = close.diff()
    gain = delta.clip(lower=0.0).rolling(window, min_periods=window).mean()
    loss = (-delta.clip(upper=0.0)).rolling(window, min_periods=window).mean()
    rs = gain.div(loss.replace(0.0, np.nan))
    result = 100.0 - 100.0 / (1.0 + rs)
    result = result.where(loss > 0.0, 100.0)
    return result.where(gain > 0.0, 0.0).shift(1)


def _state_positions(candidate: Candidate, perp: pd.DataFrame, spot: pd.DataFrame, funding: pd.DataFrame) -> pd.DataFrame:
    if candidate.family == "rsi":
        feature = _rsi(perp, candidate.window)
    elif candidate.family == "bollinger":
        mid = perp.rolling(candidate.window, min_periods=candidate.window).mean()
        std = perp.rolling(candidate.window, min_periods=candidate.window).std(ddof=1)
        feature = pd.concat({"mid": mid.shift(1), "upper": (mid + candidate.entry * std).shift(1), "lower": (mid - candidate.entry * std).shift(1)}, axis=1)
    elif candidate.family == "funding_reversal":
        mean = funding.rolling(candidate.window, min_periods=candidate.window).mean()
        std = funding.rolling(candidate.window, min_periods=candidate.window).std(ddof=1)
        z = funding.sub(mean).div(std.replace(0.0, np.nan)).shift(1)
        ret = perp.pct_change(3).shift(1)
        feature = pd.concat({"z": z, "ret": ret}, axis=1)
    else:
        basis = perp.div(spot).sub(1.0)
        mean = basis.rolling(candidate.window, min_periods=candidate.window).mean()
        std = basis.rolling(candidate.window, min_periods=candidate.window).std(ddof=1)
        feature = basis.sub(mean).div(std.replace(0.0, np.nan)).shift(1)
    state = {symbol: 0 for symbol in SYMBOLS}
    age = {symbol: 0 for symbol in SYMBOLS}
    rows: list[dict[str, float]] = []
    for day in perp.index:
        next_state = dict(state)
        next_age = dict(age)
        for symbol in SYMBOLS:
            value = feature.loc[day, symbol] if candidate.family not in {"bollinger", "funding_reversal"} else np.nan
            if candidate.family == "bollinger":
                mid = feature.loc[day, ("mid", symbol)]
                upper = feature.loc[day, ("upper", symbol)]
                lower = feature.loc[day, ("lower", symbol)]
                close = perp.loc[day, symbol]
                exit_long = pd.notna(mid) and close <= mid
                exit_short = pd.notna(mid) and close >= mid
                enter_long = pd.notna(upper) and close > upper
                enter_short = pd.notna(lower) and close < lower
            elif candidate.family == "funding_reversal":
                z = feature.loc[day, ("z", symbol)]
                ret = feature.loc[day, ("ret", symbol)]
                exit_long = pd.notna(z) and z >= 0.0
                exit_short = pd.notna(z) and z <= 0.0
                enter_long = pd.notna(z) and pd.notna(ret) and z <= -candidate.entry and ret <= 0.0
                enter_short = pd.notna(z) and pd.notna(ret) and z >= candidate.entry and ret >= 0.0
            elif candidate.family == "basis":
                exit_long = pd.notna(value) and abs(float(value)) <= candidate.exit
                exit_short = exit_long
                enter_long = pd.notna(value) and float(value) <= -candidate.entry
                enter_short = pd.notna(value) and float(value) >= candidate.entry
            else:
                exit_long = pd.notna(value) and float(value) >= candidate.exit
                exit_short = pd.notna(value) and float(value) <= 100.0 - candidate.exit
                enter_long = pd.notna(value) and float(value) <= candidate.entry
                enter_short = pd.notna(value) and float(value) >= 100.0 - candidate.entry
            if state[symbol] != 0:
                next_age[symbol] += 1
                if (state[symbol] == 1 and exit_long) or (state[symbol] == -1 and exit_short) or next_age[symbol] >= candidate.hold_days:
                    next_state[symbol] = 0
                    next_age[symbol] = 0
            if next_state[symbol] == 0:
                if enter_long:
                    next_state[symbol] = 1
                    next_age[symbol] = 0
                elif candidate.long_short and enter_short:
                    next_state[symbol] = -1
                    next_age[symbol] = 0
        rows.append(dict(next_state))
        state = next_state
        age = next_age
    return pd.DataFrame(rows, index=perp.index).astype(float)


def _weights(candidate: Candidate, states: pd.DataFrame) -> pd.DataFrame:
    pair = candidate.family == "basis"
    divisor = 2.0 if pair else 1.0
    result = states.copy()
    for day in result.index:
        active = int((result.loc[day].abs() > 0.0).sum())
        result.loc[day] = result.loc[day] * (GROSS_CAP / divisor / active if active else 0.0)
    return result


def _run_path(candidate: Candidate, weights: pd.DataFrame, returns: pd.DataFrame, stress: bool, pair: bool) -> dict[str, object]:
    equity = INITIAL_ACTIVE
    curve: list[float] = []
    daily: list[float] = []
    gross_values: list[float] = []
    normalized_orders: list[float] = []
    active_days: list[pd.Timestamp] = []
    event_count = 0
    previous = pd.Series(0.0, index=SYMBOLS)
    leg_count = 2.0 if pair else 1.0
    borrow_rate = BORROW_APR / 365.0 if pair else 0.0
    rates = pd.Series({symbol: MAKER_FEE + (2.0 if stress else 1.0) * slippage_rate(symbol) for symbol in SYMBOLS})
    for day in weights.index:
        target = weights.loc[day].reindex(SYMBOLS).fillna(0.0)
        previous_returns = returns.loc[day].reindex(SYMBOLS).fillna(0.0)
        pnl = equity * float((previous * previous_returns).sum())
        borrow = equity * float((previous.clip(lower=0.0)).sum()) * borrow_rate
        turnover = (target - previous).abs()
        cost = equity * float((turnover * rates).sum()) * leg_count
        if float(turnover.sum()) > 1e-12:
            event_count += 1
        equity = max(0.0, equity + pnl - borrow - cost)
        total = RESERVE + equity
        prior_total = curve[-1] if curve else INITIAL_TOTAL
        curve.append(total)
        daily.append(total / prior_total - 1.0 if prior_total > 0.0 else -1.0)
        gross = float(target.abs().sum()) * leg_count
        gross_values.append(gross)
        notional = (target.abs() * equity).replace(0.0, np.nan).dropna()
        normalized = notional * INITIAL_TOTAL / total if total > 0.0 else notional.iloc[0:0]
        normalized_orders.append(float(normalized.min()) if not normalized.empty else float("inf"))
        if gross > 0.0:
            active_days.append(day)
        previous = target
    curve_series = pd.Series(curve, index=weights.index, dtype=float)
    return_series = pd.Series(daily, index=weights.index, dtype=float)
    oos_curve = curve_series[curve_series.index > OOS_SPLIT]
    oos_returns = return_series[return_series.index > OOS_SPLIT]
    anchor_slice = curve_series[curve_series.index <= OOS_SPLIT]
    anchor = float(anchor_slice.iloc[-1]) if not anchor_slice.empty else INITIAL_TOTAL
    oos_return = float(oos_curve.iloc[-1] / anchor - 1.0) if not oos_curve.empty and anchor > 0.0 else 0.0
    clean = oos_returns.replace([np.inf, -np.inf], np.nan).dropna()
    sharpe = float(clean.mean() / clean.std(ddof=1) * math.sqrt(365.0)) if len(clean) > 1 and clean.std(ddof=1) > 0.0 else 0.0
    rolling_max = curve_series.cummax()
    mdd = float((1.0 - curve_series / rolling_max).max()) if not curve_series.empty else 0.0
    oos_orders = [value for day, value in zip(weights.index, normalized_orders, strict=True) if day > OOS_SPLIT and math.isfinite(value)]
    return {"curve": curve_series, "returns": return_series, "oos_returns": oos_returns, "oos_return": oos_return, "oos_sharpe": sharpe, "mdd": mdd, "max_gross": float(max(gross_values, default=0.0)), "min_oos_order": float(min(oos_orders, default=0.0)), "oos_active_days": int(sum(day > OOS_SPLIT for day in active_days)), "event_count": event_count, "data_valid": True}


def _gates(normal: dict[str, object], stress: dict[str, object], bootstrap: dict[str, float], block_mdd: float) -> list[dict[str, object]]:
    returns = normal["oos_returns"]
    blocks = np.array_split(returns.replace([np.inf, -np.inf], np.nan).dropna().to_numpy(), 4)
    positive_blocks = sum(float(np.prod(1.0 + block) - 1.0) > 0.0 for block in blocks if len(block))
    rows = [
        ("data_validation", bool(normal["data_valid"]), "validated=true"),
        ("fixed_major_universe", True, "BTCUSDT,ETHUSDT,SOLUSDT"),
        ("oos_independence", False, "independent=false; prospective paper required"),
        ("capital_contract", float(normal["max_gross"]) <= GROSS_CAP + 1e-12 and float(normal["min_oos_order"]) >= MIN_ORDER, f"gross={float(normal['max_gross']):.4f}; normalized_min_order=${float(normal['min_oos_order']):.2f}"),
        ("oos_activity", int(normal["oos_active_days"]) >= 60, f"active_days={int(normal['oos_active_days'])}"),
        ("oos_after_cost", float(normal["oos_return"]) > 0.0, f"return={float(normal['oos_return']):.4f}"),
        ("oos_sharpe", float(normal["oos_sharpe"]) >= 1.0, f"sharpe={float(normal['oos_sharpe']):.4f}"),
        ("oos_block_robustness", positive_blocks >= 2, f"positive_blocks={positive_blocks}/4"),
        ("historical_mdd", float(normal["mdd"]) <= 0.25, f"mdd={float(normal['mdd']):.4f}"),
        ("double_slippage", float(stress["oos_return"]) > 0.0, f"return={float(stress['oos_return']):.4f}"),
        ("bootstrap_p05", bootstrap["p05_final"] > INITIAL_TOTAL, f"p05=${bootstrap['p05_final']:.2f}"),
        ("bootstrap_ruin", bootstrap["ruin_frequency"] < 0.05, f"ruin={bootstrap['ruin_frequency']:.4f}"),
        ("block_mdd_p95", block_mdd <= 0.25, f"mdd_p95={block_mdd:.4f}"),
    ]
    return [{"name": name, "status": "PASS" if passed else "FAIL", "value": value} for name, passed, value in rows]


def _result(candidate: Candidate, perp: pd.DataFrame, spot: pd.DataFrame, funding: pd.DataFrame, spot_valid: bool) -> dict[str, object]:
    states = _state_positions(candidate, perp, spot, funding)
    weights = _weights(candidate, states)
    pair = candidate.family == "basis"
    perp_return = perp.pct_change().fillna(0.0)
    if pair:
        returns = perp_return - spot.pct_change().fillna(0.0) - funding
    else:
        returns = perp_return - funding
    normal = _run_path(candidate, weights, returns, False, pair)
    normal["data_valid"] = bool(not pair or spot_valid)
    stress = _run_path(candidate, weights, returns, True, pair)
    bootstrap = _bootstrap(normal["oos_returns"])
    block_mdd = _block_mdd(normal["oos_returns"])
    gates = _gates(normal, stress, bootstrap, block_mdd)
    metrics = {"oos_return": normal["oos_return"], "oos_sharpe": normal["oos_sharpe"], "mdd": normal["mdd"], "max_gross": normal["max_gross"], "min_oos_order": normal["min_oos_order"], "oos_active_days": normal["oos_active_days"], "event_count": normal["event_count"], "stress_oos_return": stress["oos_return"], "bootstrap": bootstrap, "block_mdd_p95": block_mdd}
    return {"candidate_id": candidate.candidate_id, "family": candidate.family, "candidate_config": asdict(candidate), "oos_independent": False, "metrics": metrics, "gates": gates, "all_gates_pass": all(item["status"] == "PASS" for item in gates)}


def _json_safe(value: object) -> object:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _write_report(results: list[dict[str, object]], manifest_sha: str, runner_sha: str, spec_sha: str) -> None:
    lines = ["# Wave-11 multi-family report", "", "- Families: RSI mean reversion, Bollinger expansion, funding-price reversal, and spot-perpetual basis relative value.", f"- Source manifest SHA-256: `{manifest_sha}`", f"- Runner SHA-256: `{runner_sha}`", f"- Spec SHA-256: `{spec_sha}`", f"- Registered candidates: {len(results)}; eligible: `[]`", "- Historical screen only; OOS independence is explicitly false.", "- Minimum order is normalized to a fresh $100 OOS start.", "", "| Candidate | Family | OOS return | Sharpe | stress | p05 | ruin | active days | min order | MDD | block MDD p95 | failed gates | verdict |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|"]
    for item in sorted(results, key=lambda value: float(value["metrics"]["oos_return"]), reverse=True):
        metrics = item["metrics"]
        bootstrap = metrics["bootstrap"]
        failures = ", ".join(gate["name"] for gate in item["gates"] if gate["status"] == "FAIL") or "-"
        lines.append(f"| {item['candidate_id']} | {item['family']} | {metrics['oos_return']:.2%} | {metrics['oos_sharpe']:.2f} | {metrics['stress_oos_return']:.2%} | ${bootstrap['p05_final']:.2f} | {bootstrap['ruin_frequency']:.2%} | {metrics['oos_active_days']} | ${metrics['min_oos_order']:.2f} | {metrics['mdd']:.2%} | {metrics['block_mdd_p95']:.2%} | {failures} | {'PASS' if item['all_gates_pass'] else 'FAIL'} |")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "wave11_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(only: str | None = None) -> list[dict[str, object]]:
    runner_path = Path(__file__)
    spec_path = BASE_DIR / "SPEC.md"
    dep_paths = {"research/wave1/costs.py": BASE_DIR.parent / "wave1" / "costs.py", "research/wave3/fetch.py": BASE_DIR.parent / "wave3" / "fetch.py", "research/wave9_capital_native/run_wave9.py": BASE_DIR.parent / "wave9_capital_native" / "run_wave9.py"}
    runner_before = _sha256(runner_path)
    spec_before = _sha256(spec_path)
    deps_before = {name: _sha256(path) for name, path in dep_paths.items()}
    manifest_path = CACHE_DIR / "manifest.json"
    verify_manifest(CACHE_DIR)
    manifest_before = _sha256(manifest_path)
    perp, spot, funding, _ = _load_data()
    spot_valid = bool(spot.loc[spot.index > OOS_SPLIT].notna().all().all())
    candidate_map = {candidate.candidate_id: candidate for candidate in CANDIDATES}
    if only is not None and only not in candidate_map:
        raise Wave11Error(f"unknown registered candidate: {only}")
    selected = CANDIDATES if only is None else (candidate_map[only],)
    results = [_result(candidate, perp, spot, funding, spot_valid) for candidate in selected]
    verify_manifest(CACHE_DIR)
    manifest_after = _sha256(manifest_path)
    if manifest_before != manifest_after or runner_before != _sha256(runner_path) or spec_before != _sha256(spec_path) or deps_before != {name: _sha256(path) for name, path in dep_paths.items()}:
        raise Wave11Error("source, runner, spec, or dependency changed during run")
    if only is None:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        payload = {"source_manifest_sha256": manifest_before, "source_manifest_sha256_before": manifest_before, "source_manifest_sha256_after": manifest_after, "runner_sha256": runner_before, "spec_sha256": spec_before, "dependency_sha256": deps_before, "fixed_universe": list(SYMBOLS), "registered_candidates": [asdict(candidate) for candidate in CANDIDATES], "results": _json_safe(results), "eligible_candidates": []}
        (RESULTS_DIR / "wave11_results.json").write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
        _write_report(results, manifest_before, runner_before, spec_before)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only")
    args = parser.parse_args()
    try:
        results = run(args.only)
    except Wave11Error as exc:
        parser.error(str(exc))
    print("WAVE11_DONE")
    print("eligible=" + json.dumps([item["candidate_id"] for item in results if item["all_gates_pass"]]))


if __name__ == "__main__":
    main()
