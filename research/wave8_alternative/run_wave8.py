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
RESERVE: Final = 10.0
GROSS_CAP: Final = 0.60
MIN_ORDER: Final = 5.0
FEE_RATE: Final = 0.0006
BASE_SLIPPAGE: Final = 0.0003
STRESS_SLIPPAGE: Final = 0.0006
MC_PATHS: Final = 10_000
BLOCK_PATHS: Final = 1_000
BLOCK_DAYS: Final = 90
SYMBOLS: Final = ("BTC", "ETH", "BNB", "ADA", "XRP", "DOGE", "SOL", "AVAX", "DOT", "LINK", "LTC", "BCH")
MAJORS: Final = ("BTC", "ETH", "SOL")
IDS: Final = (
    "R8a", "R8b", "R8c", "R8d", "V8a", "V8b", "V8c", "V8d",
    "Q8a", "Q8b", "Q8c", "Q8d", "F8a", "F8b", "F8c", "F8d",
)


@dataclass(frozen=True, slots=True)
class Candidate:
    candidate_id: str
    family: str
    kind: str
    window: int
    top_k: int = 3
    funding: bool = False


CANDIDATES: Final[tuple[Candidate, ...]] = (
    Candidate("R8a", "R8", "reversal", 3),
    Candidate("R8b", "R8", "reversal", 5),
    Candidate("R8c", "R8", "reversal", 1),
    Candidate("R8d", "R8", "residual_reversal", 5),
    Candidate("V8a", "V8", "invvol_trend", 14),
    Candidate("V8b", "V8", "invvol_trend", 30),
    Candidate("V8c", "V8", "filtered_trend", 14),
    Candidate("V8d", "V8", "lowvol_spread", 20),
    Candidate("Q8a", "Q8", "volume_continuation", 1),
    Candidate("Q8b", "Q8", "volume_reversal", 1),
    Candidate("Q8c", "Q8", "volume_continuation", 5),
    Candidate("Q8d", "Q8", "volume_reversal", 5),
    Candidate("F8a", "F8", "funding_spread", 3, funding=True),
    Candidate("F8b", "F8", "funding_change", 3, funding=True),
    Candidate("F8c", "funding_price_divergence", "funding_price_divergence", 3, funding=True),
    Candidate("F8d", "F8", "funding_guarded_spread", 7, funding=True),
)

DEFINITIONS: Final[dict[str, str]] = {
    "R8a": "3-day cross-sectional reversal",
    "R8b": "5-day cross-sectional reversal",
    "R8c": "1-day cross-sectional reversal",
    "R8d": "5-day residual cross-sectional reversal",
    "V8a": "14-day inverse-volatility trend",
    "V8b": "30-day inverse-volatility trend",
    "V8c": "14-day trend with lower-volatility half filter",
    "V8d": "20-day low-volatility spread",
    "Q8a": "volume-shock continuation, 1-day return",
    "Q8b": "volume-shock reversal, 1-day return",
    "Q8c": "volume-shock continuation, 5-day return",
    "Q8d": "low-volume 5-day reversal",
    "F8a": "3-day funding spread",
    "F8b": "funding-change spread",
    "F8c": "funding-price divergence",
    "F8d": "7-day funding spread with BTC MA200 guard",
}


def _read(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, compression="gzip", usecols=columns)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, format="ISO8601")
    frame["day"] = frame["timestamp"].dt.normalize()
    return frame


def _load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    closes: dict[str, pd.Series] = {}
    volumes: dict[str, pd.Series] = {}
    funding: dict[str, pd.Series] = {}
    for symbol in SYMBOLS:
        bars = _read(CACHE_DIR / f"binance_fapi_{symbol}USDT_1d.csv.gz", ["timestamp", "close", "volume", "quote_volume"])
        daily = bars.groupby("day", sort=True).agg(close=("close", "last"), volume=("volume", "sum"), quote_volume=("quote_volume", "sum"))
        closes[symbol] = daily["close"]
        volumes[symbol] = daily["quote_volume"].where(daily["quote_volume"] > 0.0, daily["volume"])
        rates = _read(CACHE_DIR / f"binance_funding_{symbol}USDT.csv.gz", ["timestamp", "funding_rate"])
        funding[symbol] = rates.groupby("day", sort=True)["funding_rate"].sum()
    start = max(series.index.min() for series in closes.values())
    end = min(series.index.max() for series in closes.values())
    index = pd.date_range(start, end, freq="D", tz="UTC")
    close = pd.DataFrame({symbol: series.reindex(index) for symbol, series in closes.items()})
    volume = pd.DataFrame({symbol: series.reindex(index) for symbol, series in volumes.items()})
    funding_frame = pd.DataFrame({symbol: series.reindex(index) for symbol, series in funding.items()})
    if close.isna().any().any() or volume.isna().any().any() or funding_frame.isna().any().any():
        raise ValueError("fixed-universe data has gaps in the common interval")
    if (close <= 0.0).any().any() or (volume < 0.0).any().any() or ~np.isfinite(close.to_numpy()).all():
        raise ValueError("fixed-universe data contains invalid values")
    metadata = {
        "symbols": list(SYMBOLS),
        "start": index[0].isoformat(),
        "end": index[-1].isoformat(),
        "rows": int(len(index)),
        "data_valid": True,
        "oos_split": OOS_SPLIT.isoformat(),
        "selection_independent": False,
    }
    return close, volume, funding_frame, metadata


def _zscore(frame: pd.DataFrame) -> pd.DataFrame:
    mean = frame.mean(axis=1)
    std = frame.std(axis=1, ddof=0).replace(0.0, np.nan)
    return frame.sub(mean, axis=0).div(std, axis=0)


def _top_bottom(score: pd.DataFrame, top_k: int, long_high: bool = False) -> pd.DataFrame:
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


def _invvol_trend(close: pd.DataFrame, window: int, filtered: bool = False, major_only: bool = False) -> pd.DataFrame:
    symbols = list(MAJORS) if major_only else list(close.columns)
    frame = close[symbols]
    returns = frame.pct_change()
    momentum = frame.pct_change(window).shift(1)
    vol = returns.rolling(20 if window <= 14 else 30, min_periods=20 if window <= 14 else 30).std().shift(1) * np.sqrt(365.0)
    positions = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    for timestamp in close.index:
        m = momentum.loc[timestamp]
        v = vol.loc[timestamp]
        valid = m.notna() & v.notna() & (v > 0.0)
        if filtered:
            rank = v.rank(pct=True)
            valid &= rank <= 0.5
        if not valid.any():
            continue
        raw = np.sign(m[valid]) / v[valid]
        raw = raw[raw != 0.0]
        if raw.empty:
            continue
        positions.loc[timestamp, raw.index] = GROSS_CAP * raw / raw.abs().sum()
    return positions


def _lowvol(close: pd.DataFrame, window: int) -> pd.DataFrame:
    vol = close.pct_change().rolling(window, min_periods=window).std().shift(1)
    return _top_bottom(vol, 3, long_high=False)


def _volume_signal(close: pd.DataFrame, volume: pd.DataFrame, window: int, reversal: bool, low_volume: bool = False) -> pd.DataFrame:
    returns = close.pct_change()
    feature_return = returns.rolling(window, min_periods=window).sum().shift(1)
    prior_volume = volume.shift(1)
    volume_mean = prior_volume.rolling(20, min_periods=20).mean().shift(1)
    volume_std = prior_volume.rolling(20, min_periods=20).std().shift(1).replace(0.0, np.nan)
    z = prior_volume.sub(volume_mean).div(volume_std)
    positions = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    for timestamp in close.index:
        r = feature_return.loc[timestamp]
        vz = z.loc[timestamp]
        if low_volume:
            active = vz < -1.0
        else:
            active = vz > (1.5 if window > 1 else 2.0)
        active &= r.notna() & vz.notna()
        if not active.any():
            continue
        sign = np.sign(r[active])
        if reversal:
            sign = -sign
        sign = sign[sign != 0.0]
        if sign.empty:
            continue
        positions.loc[timestamp, sign.index] = GROSS_CAP * sign / sign.abs().sum()
    return positions


def _funding_positions(close: pd.DataFrame, funding: pd.DataFrame, candidate_id: str) -> pd.DataFrame:
    price_return = close.pct_change(3)
    score = funding.rolling(3, min_periods=3).mean().shift(1)
    if candidate_id == "F8b":
        score = funding.rolling(3, min_periods=3).mean().sub(funding.rolling(15, min_periods=15).mean()).shift(1)
    if candidate_id == "F8c":
        score = _zscore(-score) + _zscore(-price_return.shift(1))
        return _top_bottom(score, 3, long_high=True)
    if candidate_id == "F8d":
        score = funding.rolling(7, min_periods=7).mean().shift(1)
        btc = close["BTC"]
        ma = btc.rolling(200, min_periods=200).mean().shift(1)
        score = score.where(btc.shift(1) >= ma)
    return _top_bottom(score, 3, long_high=False)


def _positions(candidate: Candidate, close: pd.DataFrame, volume: pd.DataFrame, funding: pd.DataFrame) -> pd.DataFrame:
    returns = close.pct_change()
    if candidate.kind == "reversal":
        return _top_bottom(returns.rolling(candidate.window, min_periods=candidate.window).sum().shift(1), candidate.top_k)
    if candidate.kind == "residual_reversal":
        residual = returns.sub(returns.mean(axis=1), axis=0)
        return _top_bottom(residual.rolling(candidate.window, min_periods=candidate.window).sum().shift(1), candidate.top_k)
    if candidate.kind == "invvol_trend":
        return _invvol_trend(close, candidate.window, major_only=candidate.candidate_id in {"V8a", "V8b"})
    if candidate.kind == "filtered_trend":
        return _invvol_trend(close, candidate.window, filtered=True)
    if candidate.kind == "lowvol_spread":
        return _lowvol(close, candidate.window)
    if candidate.kind == "volume_continuation":
        return _volume_signal(close, volume, candidate.window, reversal=False)
    if candidate.kind == "volume_reversal":
        return _volume_signal(close, volume, candidate.window, reversal=True, low_volume=candidate.candidate_id == "Q8d")
    if candidate.kind in {"funding_spread", "funding_change", "funding_price_divergence", "funding_guarded_spread"}:
        return _funding_positions(close, funding, candidate.candidate_id)
    raise ValueError(f"unknown candidate kind: {candidate.kind}")


def _equity(returns: pd.Series) -> pd.Series:
    clean = returns.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=-0.99)
    anchor = pd.Series([INITIAL_CAPITAL], index=[clean.index[0] - pd.Timedelta(days=1)])
    curve = INITIAL_CAPITAL * (1.0 + clean).cumprod()
    return pd.concat([anchor, curve])


def _metrics(equity: pd.Series, returns: pd.Series) -> dict[str, float]:
    daily = returns.dropna().astype(float)
    total = float(equity.iloc[-1] / INITIAL_CAPITAL - 1.0)
    days = max((equity.index[-1] - equity.index[0]).total_seconds() / 86_400.0, 1.0)
    cagr = float((equity.iloc[-1] / INITIAL_CAPITAL) ** (365.0 / days) - 1.0)
    vol = float(daily.std(ddof=1)) if len(daily) > 1 else 0.0
    sharpe = float(daily.mean() / vol * np.sqrt(365.0)) if vol > 0.0 else 0.0
    drawdown = equity / equity.cummax() - 1.0
    return {"total_return": total, "cagr": cagr, "sharpe": sharpe, "mdd": abs(float(drawdown.min()))}


def _oos(returns: pd.Series, stress: pd.Series) -> dict[str, float | int]:
    oos = returns[returns.index > OOS_SPLIT]
    stress_oos = stress[stress.index > OOS_SPLIT]
    equity = (1.0 + oos).cumprod() if not oos.empty else pd.Series(dtype=float)
    total = float(equity.iloc[-1] - 1.0) if not equity.empty else 0.0
    vol = float(oos.std(ddof=1)) if len(oos) > 1 else 0.0
    sharpe = float(oos.mean() / vol * np.sqrt(365.0)) if vol > 0.0 else 0.0
    blocks = [float((1.0 + oos.iloc[indexes]).prod() - 1.0) for indexes in np.array_split(np.arange(len(oos)), 4) if len(indexes)] if not oos.empty else []
    stress_total = float((1.0 + stress_oos).prod() - 1.0) if not stress_oos.empty else 0.0
    return {"return": total, "sharpe": sharpe, "stress_return": stress_total, "active_days": int((oos.abs() > 0.0).sum()), "positive_blocks": int(sum(value > 0.0 for value in blocks)), "block_returns": blocks}


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


def _gate(value: bool) -> str:
    return "PASS" if value else "FAIL"


def _funding_cash(positions: pd.DataFrame, funding: pd.DataFrame) -> pd.DataFrame:
    return -(positions * funding.shift(-1).fillna(0.0))


def _result(candidate: Candidate, positions: pd.DataFrame, close: pd.DataFrame, funding: pd.DataFrame, data_meta: dict, seed: int) -> dict:
    price_returns = close.pct_change().shift(-1).fillna(0.0)
    funding_cash = _funding_cash(positions, funding)
    gross = (positions * price_returns).sum(axis=1)
    carry = funding_cash.sum(axis=1) if candidate.funding else pd.Series(0.0, index=close.index)
    turnover = positions.diff().abs().sum(axis=1).fillna(positions.abs().sum(axis=1))
    net = gross + carry - turnover * (FEE_RATE + BASE_SLIPPAGE)
    stress = gross + carry - turnover * (FEE_RATE + STRESS_SLIPPAGE)
    net = net.iloc[:-1]
    stress = stress.reindex(net.index)
    positions = positions.reindex(net.index)
    equity = _equity(net)
    metrics = _metrics(equity, net)
    oos = _oos(net, stress)
    mc = _mc(net, seed)
    block = _block_mdd(net, seed + 17)
    dsr_curve = tuple(
        TimedValue(pd.Timestamp(ts).to_pydatetime(), float(value))
        for ts, value in equity.items()
    )
    dsr = deflated_sharpe(dsr_curve, trials=len(CANDIDATES))
    capital = equity.shift(1).reindex(positions.index).fillna(INITIAL_CAPITAL)
    notionals = positions.abs().mul(capital, axis=0)
    active_notionals = notionals.where(notionals > 0.0).stack()
    min_order = float(active_notionals.min()) if not active_notionals.empty else 0.0
    max_gross = float(positions.abs().sum(axis=1).max())
    data_valid = bool(data_meta["data_valid"])
    capital_ok = max_gross <= 0.9 and min_order >= MIN_ORDER and max_gross <= GROSS_CAP + 1e-9
    gates = {
        "data_validation": data_valid,
        "fixed_universe": tuple(data_meta["symbols"]) == SYMBOLS,
        "oos_independent": bool(data_meta["selection_independent"]),
        "capital_contract": capital_ok,
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
        "candidate_id": candidate.candidate_id,
        "family": candidate.family,
        "definition": DEFINITIONS[candidate.candidate_id],
        "config": {"kind": candidate.kind, "window": candidate.window, "top_k": candidate.top_k, "funding": candidate.funding},
        "data": data_meta,
        "metrics": metrics,
        "oos": oos,
        "monte_carlo": mc,
        "block_shuffle": block,
        "deflated_sharpe": {"probability": float(dsr.probability), "score": float(dsr.score), "trials": len(CANDIDATES)},
        "execution": {"max_gross": max_gross, "min_order_usd": min_order, "active_days": int((positions.abs().sum(axis=1) > 0.0).sum()), "turnover_sum": float(turnover.reindex(net.index).sum())},
        "gates": {name: {"status": _gate(value), "passed": bool(value)} for name, value in gates.items()},
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
        "# Wave-8 alternative alpha report",
        "",
        "This report is generated from the fixed-universe cache and the preregistered SPEC.md. It is simulation evidence only; prior waves were inspected before this expansion, so selection-independent is false and no live-capital recommendation is made.",
        "",
        f"Data: {data_meta['start']} to {data_meta['end']} ({data_meta['rows']} common daily rows); OOS begins 2025-10-01.",
        "",
        "| Candidate | Family | OOS return | OOS Sharpe | MDD | Stress OOS | MC p05 | Ruin | Block MDD p95 | Min order | Gates |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for result in sorted(results, key=lambda item: item["oos"]["return"], reverse=True):
        lines.append(
            f"| {result['candidate_id']} | {result['family']} | {result['oos']['return'] * 100:.2f}% | {result['oos']['sharpe']:.2f} | {result['metrics']['mdd'] * 100:.2f}% | {result['oos']['stress_return'] * 100:.2f}% | ${result['monte_carlo']['p05']:.2f} | {result['monte_carlo']['ruin_probability'] * 100:.2f}% | {result['block_shuffle']['p95'] * 100:.2f}% | ${result['execution']['min_order_usd']:.2f} | {sum(g['passed'] for g in result['gates'].values())}/{len(result['gates'])} |"
        )
    survivors = [result["candidate_id"] for result in results if result["all_gates_pass"]]
    lines.extend(["", f"## Verdict", "", f"Eligible candidates: {survivors or 'none'}.", "", "The gate is fail-closed: a positive OOS return is not enough when capital, drawdown, cost stress, Monte Carlo, block-shuffle, or deflated-Sharpe evidence fails.", "", "Basis and live-order research are outside this wave; no exchange credentials or execution code were added."])
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report = REPORT_DIR / "wave8_report.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    aggregate = RESULTS_DIR / "wave8_results.json"
    manifest = {
        "report_sha256": _hash(report), "results_sha256": _hash(aggregate),
        "spec_sha256": _hash(BASE_DIR / "SPEC.md"), "runner_sha256": _hash(BASE_DIR / "run_wave8.py"),
        "result_ids": IDS, "eligible": survivors, "selection_independent": False,
    }
    (REPORT_DIR / "wave8_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def run(only: str | None = None) -> list[dict]:
    close, volume, funding, data_meta = _load_data()
    selected = [candidate for candidate in CANDIDATES if only is None or candidate.candidate_id == only]
    if not selected:
        raise ValueError(f"unknown registered candidate: {only}")
    results: list[dict] = []
    candidate_indices = {candidate.candidate_id: index for index, candidate in enumerate(CANDIDATES)}
    for candidate in selected:
        positions = _positions(candidate, close, volume, funding)
        result = _result(candidate, positions, close, funding, data_meta, 20_260_721 + candidate_indices[candidate.candidate_id] * 101)
        _write(RESULTS_DIR / f"{candidate.candidate_id}.json", result)
        results.append(result)
        print(f"completed {candidate.candidate_id}: oos={result['oos']['return']:.4f} gates={sum(g['passed'] for g in result['gates'].values())}/{len(result['gates'])}")
    if only is None:
        _write(RESULTS_DIR / "wave8_results.json", {"candidate_ids": IDS, "results": results})
        _write_report(results, data_meta)
        print(f"WAVE8_DONE eligible={[r['candidate_id'] for r in results if r['all_gates_pass']]}")
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Wave-8 alternative crypto research runner")
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
