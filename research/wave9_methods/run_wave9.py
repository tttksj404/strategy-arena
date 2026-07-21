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

BASE_DIR: Final = Path(__file__).resolve().parent
REPO_ROOT: Final = BASE_DIR.parents[1]
CACHE_DIR: Final = REPO_ROOT / "research" / "wave3" / "cache"
RESULTS_DIR: Final = BASE_DIR / "results"
REPORT_DIR: Final = BASE_DIR / "report"
OOS_SPLIT: Final = pd.Timestamp("2025-09-30T23:59:59Z")
INITIAL_CAPITAL: Final = 100.0
GROSS_CAP: Final = 0.60
MIN_ORDER: Final = 5.0
FEE_RATE: Final = 0.0006
BASE_SLIPPAGE: Final = 0.0003
STRESS_SLIPPAGE: Final = 0.0006
MC_PATHS: Final = 10_000
BLOCK_PATHS: Final = 1_000
BLOCK_DAYS: Final = 90
SYMBOLS: Final = ("BTC", "ETH", "BNB", "ADA", "XRP", "DOGE", "SOL", "AVAX", "DOT", "LINK", "LTC", "BCH")
PAIR_SYMBOLS: Final = {"P9c": ("BTC", "ETH"), "P9d": ("SOL", "ETH")}
IDS: Final = (
    "T9a", "T9b", "T9c", "T9d", "P9a", "P9b", "P9c", "P9d",
    "H9a", "H9b", "H9c", "H9d", "D9a", "D9b", "D9c", "D9d",
)


@dataclass(frozen=True, slots=True)
class Candidate:
    candidate_id: str
    family: str
    kind: str
    window: int
    top_k: int = 3


CANDIDATES: Final[tuple[Candidate, ...]] = (
    Candidate("T9a", "T9", "tsmom", 20),
    Candidate("T9b", "T9", "tsmom", 60),
    Candidate("T9c", "T9", "donchian", 20),
    Candidate("T9d", "T9", "ema_cross", 20),
    Candidate("P9a", "P9", "residual_trend", 10),
    Candidate("P9b", "P9", "residual_trend", 30),
    Candidate("P9c", "P9", "pair", 30),
    Candidate("P9d", "P9", "pair", 30),
    Candidate("H9a", "H9", "atr_continuation", 5),
    Candidate("H9b", "H9", "clv", 5),
    Candidate("H9c", "H9", "gap_reversal", 3),
    Candidate("H9d", "H9", "price_volume", 5),
    Candidate("D9a", "D9", "drawdown_recovery", 30),
    Candidate("D9b", "D9", "downside_vol", 30),
    Candidate("D9c", "D9", "guarded_trend", 20),
    Candidate("D9d", "D9", "low_corr_trend", 30),
)

DEFINITIONS: Final[dict[str, str]] = {
    "T9a": "20-day time-series momentum",
    "T9b": "60-day time-series momentum",
    "T9c": "20-day Donchian breakout",
    "T9d": "20/60 EMA crossover with inverse-volatility weights",
    "P9a": "10-day market-beta residual trend",
    "P9b": "30-day market-beta residual trend",
    "P9c": "BTC/ETH 30-day log-spread mean reversion",
    "P9d": "SOL/ETH 30-day log-spread mean reversion",
    "H9a": "5-day ATR-normalized return continuation",
    "H9b": "5-day close-location-value accumulation",
    "H9c": "3-day open-gap reversal",
    "H9d": "5-day price-volume confirmation",
    "D9a": "30-day drawdown recovery",
    "D9b": "30-day downside-volatility spread",
    "D9c": "20-day trend with BTC 200-day regime guard",
    "D9d": "30-day trend weighted by low market correlation",
}


def _read(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, compression="gzip", usecols=columns)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, format="ISO8601")
    frame["day"] = frame["timestamp"].dt.normalize()
    return frame


def _load_data() -> tuple[dict[str, pd.DataFrame], dict]:
    fields: dict[str, dict[str, pd.Series]] = {key: {} for key in ("open", "high", "low", "close", "volume")}
    for symbol in SYMBOLS:
        bars = _read(CACHE_DIR / f"binance_fapi_{symbol}USDT_1d.csv.gz", ["timestamp", "open", "high", "low", "close", "quote_volume", "volume"])
        daily = bars.groupby("day", sort=True).agg(
            open=("open", "first"), high=("high", "max"), low=("low", "min"),
            close=("close", "last"), volume=("quote_volume", "sum"), fallback=("volume", "sum"),
        )
        daily["volume"] = daily["volume"].where(daily["volume"] > 0.0, daily["fallback"])
        for key in fields:
            fields[key][symbol] = daily[key]
    start = max(series.index.min() for series in fields["close"].values())
    end = min(series.index.max() for series in fields["close"].values())
    index = pd.date_range(start, end, freq="D", tz="UTC")
    frames = {key: pd.DataFrame({symbol: series.reindex(index) for symbol, series in values.items()}) for key, values in fields.items()}
    for key, frame in frames.items():
        if frame.isna().any().any() or ~np.isfinite(frame.to_numpy()).all():
            raise ValueError(f"fixed-universe {key} data has gaps or invalid values")
    if (frames["close"] <= 0.0).any().any() or (frames["low"] <= 0.0).any().any() or (frames["volume"] < 0.0).any().any():
        raise ValueError("fixed-universe data contains non-positive prices or negative volume")
    metadata = {
        "symbols": list(SYMBOLS), "start": index[0].isoformat(), "end": index[-1].isoformat(),
        "rows": int(len(index)), "data_valid": True, "oos_split": OOS_SPLIT.isoformat(),
        "selection_independent": False,
    }
    return frames, metadata


def _zscore(frame: pd.DataFrame) -> pd.DataFrame:
    mean = frame.mean(axis=1)
    std = frame.std(axis=1, ddof=0).replace(0.0, np.nan)
    return frame.sub(mean, axis=0).div(std, axis=0)


def _top_bottom(score: pd.DataFrame, top_k: int, long_high: bool = True) -> pd.DataFrame:
    positions = pd.DataFrame(0.0, index=score.index, columns=score.columns)
    per_leg = GROSS_CAP / (2.0 * top_k)
    for timestamp, row in score.iterrows():
        values = row.dropna().sort_values()
        if len(values) < 2 * top_k:
            continue
        low = values.head(top_k).index
        high = values.tail(top_k).index
        if long_high:
            positions.loc[timestamp, high] = per_leg
            positions.loc[timestamp, low] = -per_leg
        else:
            positions.loc[timestamp, low] = per_leg
            positions.loc[timestamp, high] = -per_leg
    return positions


def _normalize(raw: pd.DataFrame) -> pd.DataFrame:
    positions = pd.DataFrame(0.0, index=raw.index, columns=raw.columns)
    for timestamp, row in raw.iterrows():
        values = row.replace([np.inf, -np.inf], np.nan).dropna()
        values = values[values != 0.0]
        if values.empty:
            continue
        positions.loc[timestamp, values.index] = GROSS_CAP * values / values.abs().sum()
    return positions


def _tsmom(close: pd.DataFrame, window: int) -> pd.DataFrame:
    score = np.sign(close.pct_change(window).shift(1))
    return _normalize(score)


def _donchian(close: pd.DataFrame, high: pd.DataFrame, low: pd.DataFrame, window: int) -> pd.DataFrame:
    prior_high = high.rolling(window, min_periods=window).max().shift(2)
    prior_low = low.rolling(window, min_periods=window).min().shift(2)
    prior_close = close.shift(1)
    signal = pd.DataFrame(np.where(prior_close > prior_high, 1.0, np.where(prior_close < prior_low, -1.0, 0.0)), index=close.index, columns=close.columns)
    return _normalize(signal)


def _ema_cross(close: pd.DataFrame) -> pd.DataFrame:
    fast = close.ewm(span=20, adjust=False, min_periods=20).mean().shift(1)
    slow = close.ewm(span=60, adjust=False, min_periods=60).mean().shift(1)
    volatility = close.pct_change().rolling(20, min_periods=20).std().shift(1).replace(0.0, np.nan)
    return _normalize(np.sign(fast - slow).div(volatility))


def _residual_trend(close: pd.DataFrame, window: int) -> pd.DataFrame:
    returns = close.pct_change()
    market = returns.mean(axis=1)
    market_var = market.rolling(60, min_periods=60).var().replace(0.0, np.nan)
    beta = pd.DataFrame({symbol: returns[symbol].rolling(60, min_periods=60).cov(market).div(market_var) for symbol in SYMBOLS})
    residual = returns.sub(beta.mul(market, axis=0), axis=0)
    return _top_bottom(residual.rolling(window, min_periods=window).sum().shift(1), 3, long_high=True)


def _pair(close: pd.DataFrame, candidate_id: str) -> pd.DataFrame:
    base, quote = PAIR_SYMBOLS[candidate_id]
    spread = np.log(close[base]) - np.log(close[quote])
    mean = spread.rolling(30, min_periods=30).mean()
    std = spread.rolling(30, min_periods=30).std().replace(0.0, np.nan)
    signal = (-((spread - mean) / std)).shift(1)
    positions = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    positions[base] = np.sign(signal) * GROSS_CAP / 2.0
    positions[quote] = -np.sign(signal) * GROSS_CAP / 2.0
    return positions


def _atr_continuation(close: pd.DataFrame, high: pd.DataFrame, low: pd.DataFrame, window: int) -> pd.DataFrame:
    true_range = (high - low).div(close.shift(1)).replace([np.inf, -np.inf], np.nan)
    atr = true_range.rolling(14, min_periods=14).mean().shift(1)
    score = close.pct_change(window).shift(1).div(atr)
    return _top_bottom(score, 3, long_high=True)


def _clv(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, window: int) -> pd.DataFrame:
    range_value = (high - low).replace(0.0, np.nan)
    clv = (2.0 * close - high - low).div(range_value)
    return _top_bottom(clv.rolling(window, min_periods=window).mean().shift(1), 3, long_high=True)


def _gap_reversal(open_: pd.DataFrame, close: pd.DataFrame, window: int) -> pd.DataFrame:
    gap = open_.div(close.shift(1)) - 1.0
    return _top_bottom(-gap.rolling(window, min_periods=window).mean().shift(1), 3, long_high=True)


def _price_volume(close: pd.DataFrame, volume: pd.DataFrame, window: int) -> pd.DataFrame:
    price_score = _zscore(close.pct_change(window).shift(1))
    volume_score = _zscore(np.log(volume).diff(window).shift(1))
    return _top_bottom(price_score.add(volume_score).div(2.0), 3, long_high=True)


def _drawdown_recovery(close: pd.DataFrame, window: int) -> pd.DataFrame:
    high_water = close.rolling(window, min_periods=window).max().shift(1)
    drawdown = close.shift(1).div(high_water) - 1.0
    rebound = close.pct_change(5).shift(1)
    return _top_bottom(rebound.add(0.5 * drawdown), 3, long_high=True)


def _downside_vol(close: pd.DataFrame, window: int) -> pd.DataFrame:
    returns = close.pct_change()
    downside = returns.where(returns < 0.0, 0.0).rolling(window, min_periods=window).std().shift(1)
    return _top_bottom(downside, 3, long_high=False)


def _guarded_trend(close: pd.DataFrame, window: int) -> pd.DataFrame:
    btc = close["BTC"]
    guard = btc.shift(1) >= btc.rolling(200, min_periods=200).mean().shift(1)
    score = close.pct_change(window).shift(1).where(guard, 0.0)
    return _normalize(np.sign(score))


def _low_corr_trend(close: pd.DataFrame, window: int) -> pd.DataFrame:
    returns = close.pct_change()
    market = returns.mean(axis=1)
    momentum = returns.rolling(window, min_periods=window).sum().shift(1)
    correlation = pd.DataFrame({symbol: returns[symbol].rolling(30, min_periods=30).corr(market).shift(1) for symbol in SYMBOLS})
    score = momentum.div(1.0 + correlation.abs())
    return _normalize(np.sign(score) * score.abs())


def _positions(candidate: Candidate, frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    close, high, low, open_, volume = (frames[key] for key in ("close", "high", "low", "open", "volume"))
    if candidate.kind == "tsmom":
        return _tsmom(close, candidate.window)
    if candidate.kind == "donchian":
        return _donchian(close, high, low, candidate.window)
    if candidate.kind == "ema_cross":
        return _ema_cross(close)
    if candidate.kind == "residual_trend":
        return _residual_trend(close, candidate.window)
    if candidate.kind == "pair":
        return _pair(close, candidate.candidate_id)
    if candidate.kind == "atr_continuation":
        return _atr_continuation(close, high, low, candidate.window)
    if candidate.kind == "clv":
        return _clv(high, low, close, candidate.window)
    if candidate.kind == "gap_reversal":
        return _gap_reversal(open_, close, candidate.window)
    if candidate.kind == "price_volume":
        return _price_volume(close, volume, candidate.window)
    if candidate.kind == "drawdown_recovery":
        return _drawdown_recovery(close, candidate.window)
    if candidate.kind == "downside_vol":
        return _downside_vol(close, candidate.window)
    if candidate.kind == "guarded_trend":
        return _guarded_trend(close, candidate.window)
    if candidate.kind == "low_corr_trend":
        return _low_corr_trend(close, candidate.window)
    raise ValueError(f"unknown candidate kind: {candidate.kind}")


def _equity(returns: pd.Series) -> pd.Series:
    clean = returns.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=-0.99)
    anchor = pd.Series([INITIAL_CAPITAL], index=[clean.index[0] - pd.Timedelta(days=1)])
    curve = INITIAL_CAPITAL * (1.0 + clean).cumprod()
    return pd.concat([anchor, curve])


def _metrics(equity: pd.Series, returns: pd.Series) -> dict[str, float]:
    daily = returns.astype(float)
    total = float(equity.iloc[-1] / INITIAL_CAPITAL - 1.0)
    days = max((equity.index[-1] - equity.index[0]).total_seconds() / 86_400.0, 1.0)
    cagr = float((equity.iloc[-1] / INITIAL_CAPITAL) ** (365.0 / days) - 1.0) if equity.iloc[-1] > 0 else -1.0
    vol = float(daily.std(ddof=1)) if len(daily) > 1 else 0.0
    sharpe = float(daily.mean() / vol * np.sqrt(365.0)) if vol > 0.0 else 0.0
    drawdown = equity / equity.cummax() - 1.0
    return {"total_return": total, "cagr": cagr, "sharpe": sharpe, "mdd": abs(float(drawdown.min()))}


def _oos(returns: pd.Series, stress: pd.Series, active: pd.Series) -> dict[str, float | int | list[float]]:
    mask = returns.index > OOS_SPLIT
    oos = returns[mask]
    stress_oos = stress[mask]
    active_oos = active[mask]
    total = float((1.0 + oos).prod() - 1.0) if not oos.empty else 0.0
    vol = float(oos.std(ddof=1)) if len(oos) > 1 else 0.0
    sharpe = float(oos.mean() / vol * np.sqrt(365.0)) if vol > 0.0 else 0.0
    blocks = [float((1.0 + oos.iloc[indexes]).prod() - 1.0) for indexes in np.array_split(np.arange(len(oos)), 4) if len(indexes)] if not oos.empty else []
    stress_total = float((1.0 + stress_oos).prod() - 1.0) if not stress_oos.empty else 0.0
    return {"return": total, "sharpe": sharpe, "stress_return": stress_total, "active_days": int(active_oos.sum()), "positive_blocks": int(sum(value > 0.0 for value in blocks)), "block_returns": blocks}


def _mc(returns: pd.Series, seed: int) -> dict[str, float | int]:
    values = returns.to_numpy(dtype=float)
    if len(values) == 0:
        return {"p05": INITIAL_CAPITAL, "ruin_probability": 1.0, "paths": 0}
    rng = np.random.default_rng(seed)
    finals = np.empty(MC_PATHS, dtype=float)
    for start in range(0, MC_PATHS, 500):
        stop = min(start + 500, MC_PATHS)
        sampled = rng.choice(values, size=(stop - start, len(values)), replace=True)
        finals[start:stop] = INITIAL_CAPITAL * np.prod(1.0 + np.clip(sampled, -0.99, None), axis=1)
    return {"p05": float(np.quantile(finals, 0.05)), "ruin_probability": float(np.mean(finals < INITIAL_CAPITAL / 2.0)), "paths": MC_PATHS}


def _block_mdd(returns: pd.Series, seed: int) -> dict[str, float | int]:
    values = returns.to_numpy(dtype=float)
    if len(values) == 0:
        return {"p95": 1.0, "paths": 0, "blocks": 0}
    blocks = [values[index:index + BLOCK_DAYS] for index in range(0, len(values), BLOCK_DAYS)]
    rng = np.random.default_rng(seed)
    mdds = np.empty(BLOCK_PATHS, dtype=float)
    for path_index in range(BLOCK_PATHS):
        sample = np.concatenate([blocks[index] for index in rng.permutation(len(blocks))])
        curve = INITIAL_CAPITAL * np.cumprod(1.0 + np.clip(sample, -0.99, None))
        peaks = np.maximum.accumulate(np.concatenate(([INITIAL_CAPITAL], curve)))[1:]
        mdds[path_index] = float(np.max(1.0 - curve / peaks))
    return {"p95": float(np.quantile(mdds, 0.95)), "paths": BLOCK_PATHS, "blocks": len(blocks)}


def _result(candidate: Candidate, positions: pd.DataFrame, frames: dict[str, pd.DataFrame], data_meta: dict, seed: int) -> dict:
    close = frames["close"]
    price_returns = close.pct_change().shift(-1).fillna(0.0)
    gross = (positions * price_returns).sum(axis=1)
    turnover = positions.diff().abs().sum(axis=1).fillna(positions.abs().sum(axis=1))
    net = (gross - turnover * (FEE_RATE + BASE_SLIPPAGE)).iloc[:-1]
    stress = (gross - turnover * (FEE_RATE + STRESS_SLIPPAGE)).reindex(net.index)
    positions = positions.reindex(net.index)
    active = positions.abs().sum(axis=1) > 0.0
    equity = _equity(net)
    metrics = _metrics(equity, net)
    oos = _oos(net, stress, active)
    mc = _mc(net, seed)
    block = _block_mdd(net, seed + 17)
    dsr_curve = tuple(TimedValue(pd.Timestamp(ts).to_pydatetime(), float(value)) for ts, value in equity.items())
    dsr = deflated_sharpe(dsr_curve, trials=len(CANDIDATES))
    capital = equity.shift(1).reindex(positions.index).fillna(INITIAL_CAPITAL)
    notionals = positions.abs().mul(capital, axis=0)
    active_notionals = notionals.where(notionals > 0.0).stack()
    min_order = float(active_notionals.min()) if not active_notionals.empty else 0.0
    max_gross = float(positions.abs().sum(axis=1).max())
    gates = {
        "data_validation": bool(data_meta["data_valid"]),
        "fixed_universe": tuple(data_meta["symbols"]) == SYMBOLS,
        "oos_split_present": bool(net.index[net.index > OOS_SPLIT].size > 0),
        "selection_independent": bool(data_meta["selection_independent"]),
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
        "candidate_id": candidate.candidate_id, "family": candidate.family, "definition": DEFINITIONS[candidate.candidate_id],
        "config": {"kind": candidate.kind, "window": candidate.window, "top_k": candidate.top_k}, "data": data_meta,
        "metrics": metrics, "oos": oos, "monte_carlo": mc, "block_shuffle": block,
        "deflated_sharpe": {"probability": float(dsr.probability), "score": float(dsr.score), "trials": len(CANDIDATES)},
        "execution": {"max_gross": max_gross, "min_order_usd": min_order, "active_days": int(active.sum()), "turnover_sum": float(turnover.reindex(net.index).sum())},
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


def _write_report(results: list[dict], data_meta: dict) -> None:
    lines = [
        "# Wave-9 method expansion report", "",
        "This report is exploratory simulation evidence. The historical split is not selection-independent because prior waves were inspected before this expansion; no live-capital recommendation is made.", "",
        f"Data: {data_meta['start']} to {data_meta['end']} ({data_meta['rows']} common daily rows); OOS begins 2025-10-01.", "",
        "| Candidate | Family | OOS return | OOS Sharpe | MDD | Stress OOS | MC p05 | Ruin | Block MDD p95 | Min order | Fails |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for result in sorted(results, key=lambda item: item["oos"]["return"], reverse=True):
        failures = ", ".join(name for name, gate in result["gates"].items() if not gate["passed"])
        lines.append(f"| {result['candidate_id']} | {result['family']} | {result['oos']['return'] * 100:.2f}% | {result['oos']['sharpe']:.2f} | {result['metrics']['mdd'] * 100:.2f}% | {result['oos']['stress_return'] * 100:.2f}% | ${result['monte_carlo']['p05']:.2f} | {result['monte_carlo']['ruin_probability'] * 100:.2f}% | {result['block_shuffle']['p95'] * 100:.2f}% | ${result['execution']['min_order_usd']:.2f} | {failures or 'none'} |")
    survivors = [result["candidate_id"] for result in results if result["all_gates_pass"]]
    lines.extend(["", "## Verdict", "", f"Eligible candidates: {survivors or 'none'}.", "", "Selection-independent: false. Any positive result is a candidate for a fresh prospective paper window only after new unseen data exists.", ""])
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report = REPORT_DIR / "wave9_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    aggregate = RESULTS_DIR / "wave9_results.json"
    manifest = {
        "report_sha256": _hash(report), "results_sha256": _hash(aggregate),
        "spec_sha256": _hash(BASE_DIR / "SPEC.md"), "runner_sha256": _hash(BASE_DIR / "run_wave9.py"),
        "result_ids": IDS, "eligible": survivors, "selection_independent": False,
    }
    (REPORT_DIR / "wave9_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def run(only: str | None = None) -> list[dict]:
    frames, data_meta = _load_data()
    selected = [candidate for candidate in CANDIDATES if only is None or candidate.candidate_id == only]
    if not selected:
        raise ValueError(f"unknown registered candidate: {only}")
    results: list[dict] = []
    candidate_indices = {candidate.candidate_id: index for index, candidate in enumerate(CANDIDATES)}
    for candidate in selected:
        result = _result(candidate, _positions(candidate, frames), frames, data_meta, 20_260_721 + candidate_indices[candidate.candidate_id] * 131)
        _write(RESULTS_DIR / f"{candidate.candidate_id}.json", result)
        results.append(result)
        print(f"completed {candidate.candidate_id}: oos={result['oos']['return']:.4f} gates={sum(g['passed'] for g in result['gates'].values())}/{len(result['gates'])}")
    if only is None:
        _write(RESULTS_DIR / "wave9_results.json", {"candidate_ids": IDS, "results": results})
        _write_report(results, data_meta)
        print(f"WAVE9_DONE eligible={[r['candidate_id'] for r in results if r['all_gates_pass']]}")
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Wave-9 method expansion crypto research runner")
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
