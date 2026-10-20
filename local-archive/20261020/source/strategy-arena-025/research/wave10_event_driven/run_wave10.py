from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd  # noqa: PANDAS_OK

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
BOOTSTRAP_PATHS: Final = 10_000
BOOTSTRAP_SEED: Final = 20260721


@dataclass(frozen=True, slots=True)
class Candidate:
    candidate_id: str
    entry_days: int
    exit_days: int
    long_short: bool
    funding_filter: str
    gross_cap: float


CANDIDATES: Final[tuple[Candidate, ...]] = (
    Candidate("E10a", 20, 10, False, "none", 0.90),
    Candidate("E10b", 20, 20, False, "none", 0.90),
    Candidate("E10c", 55, 20, False, "none", 0.90),
    Candidate("E10d", 55, 40, False, "none", 0.90),
    Candidate("E10e", 20, 10, True, "none", 0.90),
    Candidate("E10f", 55, 20, True, "none", 0.90),
    Candidate("E10g", 20, 10, False, "carry_aware", 0.60),
    Candidate("E10h", 55, 20, True, "carry_aware", 0.60),
)


class Wave10Error(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class Market:
    high: pd.Series
    low: pd.Series
    close: pd.Series
    funding: pd.Series


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_markets() -> dict[str, Market]:
    manifest_path = CACHE_DIR / "manifest.json"
    if not manifest_path.exists():
        raise Wave10Error("wave-3 manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    roster = manifest.get("crypto_symbols") if isinstance(manifest, dict) else None
    if not isinstance(roster, list) or any(symbol not in roster for symbol in SYMBOLS):
        raise Wave10Error("fixed major roster is incomplete")
    markets: dict[str, Market] = {}
    for symbol in SYMBOLS:
        perp = _read_frame(CACHE_DIR / f"binance_fapi_{symbol}_1d.csv.gz", ("high", "low", "close"))
        funding = _read_frame(CACHE_DIR / f"binance_funding_{symbol}.csv.gz", ("funding_rate",))["funding_rate"].resample("1D").sum(min_count=1)
        high = pd.to_numeric(perp["high"], errors="coerce")
        low = pd.to_numeric(perp["low"], errors="coerce")
        close = pd.to_numeric(perp["close"], errors="coerce")
        if high.replace([np.inf, -np.inf], np.nan).dropna().empty or low.replace([np.inf, -np.inf], np.nan).dropna().empty or close.replace([np.inf, -np.inf], np.nan).dropna().empty:
            raise Wave10Error(f"empty OHLC series: {symbol}")
        markets[symbol] = Market(high, low, close, funding)
    return markets


def _align_markets(markets: dict[str, Market]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    high = pd.concat({symbol: market.high for symbol, market in markets.items()}, axis=1, sort=False).sort_index().replace([np.inf, -np.inf], np.nan)
    low = pd.concat({symbol: market.low for symbol, market in markets.items()}, axis=1, sort=False).reindex(high.index).replace([np.inf, -np.inf], np.nan)
    close = pd.concat({symbol: market.close for symbol, market in markets.items()}, axis=1, sort=False).reindex(high.index).replace([np.inf, -np.inf], np.nan)
    funding = pd.concat({symbol: market.funding for symbol, market in markets.items()}, axis=1, sort=False).reindex(high.index).replace([np.inf, -np.inf], np.nan)
    valid = close.notna()
    if (valid & high.isna()).any().any() or (valid & low.isna()).any().any():
        raise Wave10Error("OHLC gap on a valid close row")
    if ((valid & (high <= 0.0)).any().any()) or ((valid & (low <= 0.0)).any().any()) or ((valid & (high < low)).any().any()):
        raise Wave10Error("invalid OHLC range")
    for symbol in SYMBOLS:
        first_funding = funding[symbol].first_valid_index()
        if first_funding is not None and (valid[symbol].loc[first_funding:] & funding[symbol].loc[first_funding:].isna()).any():
            raise Wave10Error(f"funding gap on a valid close row: {symbol}")
    return high, low, close, funding.fillna(0.0)


def _positions(candidate: Candidate, high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, funding: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, int]:
    prior_high = high.rolling(candidate.entry_days, min_periods=candidate.entry_days).max().shift(1)
    prior_low = low.rolling(candidate.entry_days, min_periods=candidate.entry_days).min().shift(1)
    exit_high = high.rolling(candidate.exit_days, min_periods=candidate.exit_days).max().shift(1)
    exit_low = low.rolling(candidate.exit_days, min_periods=candidate.exit_days).min().shift(1)
    funding_score = funding.rolling(3, min_periods=3).mean().shift(1)
    state = {symbol: 0 for symbol in SYMBOLS}
    previous_weights = {symbol: 0.0 for symbol in SYMBOLS}
    rows: list[dict[str, float]] = []
    rebalances: list[bool] = []
    event_count = 0
    for day in close.index:
        next_state = dict(state)
        for symbol in SYMBOLS:
            value = close.at[day, symbol]
            if not math.isfinite(float(value)):
                next_state[symbol] = 0
                continue
            if next_state[symbol] == 1 and value < exit_low.at[day, symbol]:
                next_state[symbol] = 0
            elif next_state[symbol] == -1 and value > exit_high.at[day, symbol]:
                next_state[symbol] = 0
            if next_state[symbol] == 0:
                funding_value = funding_score.at[day, symbol]
                long_ok = candidate.funding_filter == "none" or (pd.notna(funding_value) and funding_value <= 0.0)
                short_ok = candidate.funding_filter == "none" or (pd.notna(funding_value) and funding_value >= 0.0)
                if pd.notna(prior_high.at[day, symbol]) and value > prior_high.at[day, symbol] and long_ok:
                    next_state[symbol] = 1
                elif candidate.long_short and pd.notna(prior_low.at[day, symbol]) and value < prior_low.at[day, symbol] and short_ok:
                    next_state[symbol] = -1
        active = [(symbol, direction) for symbol, direction in next_state.items() if direction]
        weights = {symbol: 0.0 for symbol in SYMBOLS}
        if active:
            weight = candidate.gross_cap / len(active)
            weights = {symbol: direction * weight for symbol, direction in active}
        changed = weights != previous_weights
        if changed:
            event_count += 1
        rows.append(weights)
        rebalances.append(changed)
        state = next_state
        previous_weights = weights
    return pd.DataFrame(rows, index=close.index).fillna(0.0), pd.Series(rebalances, index=close.index, dtype=bool), event_count


def _run_path(candidate: Candidate, high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, funding: pd.DataFrame, stress: bool = False) -> dict:
    returns = close.pct_change().fillna(0.0)
    previous_close = close.shift(1)
    positions, rebalances, event_count = _positions(candidate, high, low, close, funding)
    equity = INITIAL_ACTIVE
    units = pd.Series(0.0, index=SYMBOLS)
    total_curve: list[float] = []
    daily_returns: list[float] = []
    gross_values: list[float] = []
    min_order_values: list[float] = []
    normalized_min_order_values: list[float] = []
    active_days: list[pd.Timestamp] = []
    slip_multiplier = 2.0 if stress else 1.0
    for day in close.index:
        price = close.loc[day].reindex(SYMBOLS)
        current_notional = (units * price).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        previous_notional = (units * previous_close.loc[day].reindex(SYMBOLS)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        pnl_dollars = float((previous_notional * (returns.loc[day] - funding.loc[day])).sum())
        equity = max(0.0, equity + pnl_dollars)
        target = positions.loc[day].reindex(SYMBOLS).fillna(0.0)
        desired_units = units.copy()
        if bool(rebalances.loc[day]):
            desired_units = (target * equity).div(price.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        desired_notional = (desired_units * price).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        desired_gross = float(desired_notional.abs().sum() / equity) if equity > 0.0 else 0.0
        if desired_gross > candidate.gross_cap and desired_gross > 0.0:
            desired_units *= candidate.gross_cap / desired_gross
            desired_notional = (desired_units * price).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        turnover_by_symbol = (desired_notional - current_notional).abs()
        cost_dollars = float(sum(turnover_by_symbol[symbol] * (MAKER_FEE + slip_multiplier * slippage_rate(symbol)) for symbol in SYMBOLS))
        if not math.isfinite(cost_dollars):
            raise Wave10Error("non-finite execution cost")
        equity = max(0.0, equity - cost_dollars)
        units = desired_units
        post_notional = (units * price).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        post_gross = float(post_notional.abs().sum() / equity) if equity > 0.0 else 0.0
        if post_gross > candidate.gross_cap and post_gross > 0.0:
            units *= candidate.gross_cap / post_gross
            post_notional = (units * price).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        signed = post_notional.div(equity if equity > 0.0 else 1.0)
        total = RESERVE + equity
        total_curve.append(total)
        daily_returns.append(total / (total_curve[-2] if len(total_curve) > 1 else INITIAL_TOTAL) - 1.0)
        gross = float(signed.abs().sum())
        gross_values.append(gross)
        notionals = (signed.abs() * equity).replace(0.0, np.nan).dropna()
        min_order_values.append(float(notionals.min()) if not notionals.empty else float("inf"))
        normalized_notionals = notionals * INITIAL_TOTAL / total if total > 0.0 else notionals.iloc[0:0]
        normalized_min_order_values.append(float(normalized_notionals.min()) if not normalized_notionals.empty else float("inf"))
        if gross > 0.0:
            active_days.append(day)
    curve = pd.Series(total_curve, index=close.index, dtype=float)
    ret = pd.Series(daily_returns, index=close.index, dtype=float)
    oos = curve[curve.index > OOS_SPLIT]
    oos_ret = ret[ret.index > OOS_SPLIT]
    anchor_curve = curve[curve.index <= OOS_SPLIT]
    anchor = float(anchor_curve.iloc[-1]) if not anchor_curve.empty else INITIAL_TOTAL
    oos_return = float(oos.iloc[-1] / anchor - 1.0) if not oos.empty else 0.0
    clean = oos_ret.replace([np.inf, -np.inf], np.nan).dropna()
    sharpe = float(clean.mean() / clean.std(ddof=1) * math.sqrt(365.0)) if len(clean) > 1 and clean.std(ddof=1) > 0.0 else 0.0
    rolling_max = curve.cummax()
    mdd = float((1.0 - curve / rolling_max).max()) if not curve.empty else 0.0
    oos_orders = [value for day, value in zip(close.index, min_order_values, strict=True) if day > OOS_SPLIT and math.isfinite(value)]
    normalized_oos_orders = [value for day, value in zip(close.index, normalized_min_order_values, strict=True) if day > OOS_SPLIT and math.isfinite(value)]
    return {"curve": curve, "returns": ret, "oos_return": oos_return, "oos_sharpe": sharpe, "mdd": mdd, "max_gross": float(max(gross_values, default=0.0)), "min_oos_order": float(min(normalized_oos_orders, default=0.0)), "min_oos_order_historical": float(min(oos_orders, default=0.0)), "oos_active_days": int(sum(day > OOS_SPLIT for day in active_days)), "stress": stress, "event_count": event_count, "data_valid": True}


def _gates(normal: dict, stress: dict, bootstrap: dict, block_mdd: float) -> list[dict[str, object]]:
    oos_blocks = np.array_split(normal["returns"][normal["returns"].index > OOS_SPLIT].dropna().to_numpy(), 4)
    positive_blocks = sum(float(np.prod(1.0 + block) - 1.0) > 0.0 for block in oos_blocks if len(block))
    rows = [
        ("data_validation", bool(normal["data_valid"]), "validated=true"),
        ("fixed_major_universe", True, "BTCUSDT,ETHUSDT,SOLUSDT"),
        ("oos_independence", False, "independent=false; prospective paper required"),
        ("capital_contract", normal["max_gross"] <= 0.90 + 1e-12 and normal["min_oos_order"] >= MIN_ORDER, f"gross={normal['max_gross']:.4f}; normalized_min_order=${normal['min_oos_order']:.2f}"),
        ("oos_activity", normal["oos_active_days"] >= 60, f"active_days={normal['oos_active_days']}"),
        ("oos_after_cost_and_funding", normal["oos_return"] > 0.0, f"return={normal['oos_return']:.4f}"),
        ("oos_sharpe", normal["oos_sharpe"] >= 1.0, f"sharpe={normal['oos_sharpe']:.4f}"),
        ("oos_block_robustness", positive_blocks >= 2, f"positive_blocks={positive_blocks}/4"),
        ("historical_mdd", normal["mdd"] <= 0.25, f"mdd={normal['mdd']:.4f}"),
        ("double_slippage", stress["oos_return"] > 0.0, f"return={stress['oos_return']:.4f}"),
        ("bootstrap_p05", bootstrap["p05_final"] > INITIAL_TOTAL, f"p05=${bootstrap['p05_final']:.2f}"),
        ("bootstrap_ruin", bootstrap["ruin_frequency"] < 0.05, f"ruin={bootstrap['ruin_frequency']:.4f}"),
        ("block_mdd_p95", block_mdd <= 0.25, f"mdd_p95={block_mdd:.4f}"),
    ]
    return [{"name": name, "status": "PASS" if passed else "FAIL", "value": value} for name, passed, value in rows]


def _result(candidate: Candidate, high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, funding: pd.DataFrame) -> dict:
    normal = _run_path(candidate, high, low, close, funding)
    stress = _run_path(candidate, high, low, close, funding, stress=True)
    bootstrap = _bootstrap(normal["returns"][normal["returns"].index > OOS_SPLIT])
    block_mdd = _block_mdd(normal["returns"][normal["returns"].index > OOS_SPLIT])
    gates = _gates(normal, stress, bootstrap, block_mdd)
    return {"candidate_id": candidate.candidate_id, "family": "fixed_major_donchian_event", "candidate_config": asdict(candidate), "oos_independent": False, "metrics": {"oos_return": normal["oos_return"], "oos_sharpe": normal["oos_sharpe"], "mdd": normal["mdd"], "max_gross": normal["max_gross"], "min_oos_order": normal["min_oos_order"], "min_oos_order_historical": normal["min_oos_order_historical"], "oos_active_days": normal["oos_active_days"], "event_count": normal["event_count"], "stress_oos_return": stress["oos_return"], "bootstrap": bootstrap, "block_mdd_p95": block_mdd}, "gates": gates, "all_gates_pass": all(row["status"] == "PASS" for row in gates)}


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


def _write_report(results: list[dict], manifest_sha: str, runner_sha: str, spec_sha: str, output_name: str) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    eligible = [item["candidate_id"] for item in results if item["all_gates_pass"]]
    lines = ["# Wave-10 fixed-major event-driven report", "", "- Strategy family: fixed BTC/ETH/SOL Donchian breakout with optional funding filter.", f"- Source manifest SHA-256: `{manifest_sha}`", f"- Runner SHA-256: `{runner_sha}`", f"- Spec SHA-256: `{spec_sha}`", f"- Registered candidates: {len(results)}; eligible: `{eligible}`", "- Historical screen only; OOS independence is explicitly false and no live approval is implied.", "- `min order` is normalized to a fresh `$100` OOS start; historical compounded notional is retained in JSON only for diagnostics.", "", "| Candidate | OOS return | Sharpe | stress return | p05 final | ruin | events | max gross | min order ($100 OOS) | historical MDD | block MDD p95 | failed gates | verdict |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|"]
    for item in results:
        metrics = item["metrics"]
        bootstrap = metrics["bootstrap"]
        failures = ", ".join(row["name"] for row in item["gates"] if row["status"] == "FAIL") or "-"
        lines.append(f"| {item['candidate_id']} | {metrics['oos_return']:.2%} | {metrics['oos_sharpe']:.2f} | {metrics['stress_oos_return']:.2%} | ${bootstrap['p05_final']:.2f} | {bootstrap['ruin_frequency']:.2%} | {metrics['event_count']} | {metrics['max_gross']:.3f} | ${metrics['min_oos_order']:.2f} | {metrics['mdd']:.2%} | {metrics['block_mdd_p95']:.2%} | {failures} | {'PASS' if item['all_gates_pass'] else 'FAIL'} |")
    (REPORT_DIR / f"{output_name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(only: str | None = None, output_name: str = "wave10_results") -> list[dict]:
    runner_path = Path(__file__)
    spec_path = BASE_DIR / "SPEC.md"
    dep_paths = {"research/wave1/costs.py": BASE_DIR.parent / "wave1" / "costs.py", "research/wave3/fetch.py": BASE_DIR.parent / "wave3" / "fetch.py", "research/wave9_capital_native/run_wave9.py": BASE_DIR.parent / "wave9_capital_native" / "run_wave9.py"}
    runner_before = _sha256(runner_path)
    spec_before = _sha256(spec_path)
    deps_before = {name: _sha256(path) for name, path in dep_paths.items()}
    manifest_path = CACHE_DIR / "manifest.json"
    verify_manifest(CACHE_DIR)
    manifest_before = _sha256(manifest_path)
    markets = _load_markets()
    high, low, close, funding = _align_markets(markets)
    candidate_map = {item.candidate_id: item for item in CANDIDATES}
    if only is not None and only not in candidate_map:
        raise Wave10Error(f"unknown registered candidate: {only}")
    selected = CANDIDATES if only is None else (candidate_map[only],)
    results = [_result(candidate, high, low, close, funding) for candidate in selected]
    verify_manifest(CACHE_DIR)
    manifest_after = _sha256(manifest_path)
    runner_after = _sha256(runner_path)
    spec_after = _sha256(spec_path)
    deps_after = {name: _sha256(path) for name, path in dep_paths.items()}
    if manifest_before != manifest_after or runner_before != runner_after or spec_before != spec_after or deps_before != deps_after:
        raise Wave10Error("source, runner, spec, or dependency changed during run")
    if only is None:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        payload = {"source_manifest_sha256": manifest_before, "source_manifest_sha256_before": manifest_before, "source_manifest_sha256_after": manifest_after, "runner_sha256": runner_before, "spec_sha256": spec_before, "dependency_sha256": deps_before, "fixed_universe": list(SYMBOLS), "registered_candidates": [asdict(item) for item in CANDIDATES], "results": _json_safe(results), "eligible_candidates": [item["candidate_id"] for item in results if item["all_gates_pass"]]}
        (RESULTS_DIR / f"{output_name}.json").write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
        _write_report(results, manifest_before, runner_before, spec_before, output_name)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only")
    args = parser.parse_args()
    try:
        results = run(args.only)
    except Wave10Error as exc:
        parser.error(str(exc))
    print("WAVE10_DONE")
    print("eligible=" + json.dumps([item["candidate_id"] for item in results if item["all_gates_pass"]]))


if __name__ == "__main__":
    main()
