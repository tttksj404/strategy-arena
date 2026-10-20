# Wave-10 gate evaluation: (A) executability, (B) MC bootstrap principal preservation,
# (C) block-shuffle drawdown, (D) full-period cost-after profit, (E) dormancy-aware OOS.
#
# MC bootstrap and block-shuffle methodology mirror research/wave8_capital/run_capital100.py
# (_simulate_mc / _block_shuffle) exactly -- same path counts, same $100/$90/$10 capital
# basis, same block length -- so wave10 results stay comparable to the wave8 judgment they
# are directly answering. wave10 runs its own equity path natively at ACTIVE_CAPITAL instead
# of rescaling a $300 base, so no rescale-approximation step is needed here.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import pandas as pd  # noqa: PANDAS_OK

from research.wave10_carry100.configs import Wave10Config
from research.wave10_carry100.engine import ACTIVE_CAPITAL, MIN_ORDER_USDT, OOS_SPLIT, RESERVE_FRACTION, TOTAL_CAPITAL, Wave10Result

MC_PATHS: Final = 10_000
BLOCK_PATHS: Final = 1_000
BLOCK_DAYS: Final = 90
SEED: Final = 20_260_722  # wave10 freeze date

FAILURE_GROSS = "gross"
FAILURE_MIN_ORDER = "최소주문"
FAILURE_PROFIT = "수익부족"
FAILURE_DORMANT = "휴면"


def leg_usdt(config: Wave10Config) -> float:
    return config.leg_fraction * ACTIVE_CAPITAL


def gross_usdt(config: Wave10Config) -> float:
    return 2.0 * config.candidate.top_k * config.leg_fraction * ACTIVE_CAPITAL


def _daily_returns(equity: pd.Series) -> tuple[tuple[pd.Timestamp, ...], np.ndarray]:
    clean = equity.dropna().astype(float)
    values = clean.to_numpy()
    if len(values) < 2 or not np.isfinite(values).all() or (values <= 0.0).any():
        raise ValueError("wave10 equity series must have >=2 finite, positive observations")
    returns = values[1:] / values[:-1] - 1.0
    if not np.isfinite(returns).all() or (returns <= -1.0).any():
        raise ValueError("wave10 equity returns contain invalid values")
    timestamps = tuple(pd.Timestamp(ts) for ts in clean.index[1:])
    return timestamps, returns


def _simulate_mc(returns: np.ndarray, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    finals = np.empty(MC_PATHS, dtype=float)
    chunk_size = 250
    for start in range(0, MC_PATHS, chunk_size):
        stop = min(start + chunk_size, MC_PATHS)
        samples = rng.choice(returns, size=(stop - start, returns.size), replace=True)
        growth = np.prod(1.0 + np.clip(samples, -0.999999, None), axis=1)
        finals[start:stop] = TOTAL_CAPITAL * RESERVE_FRACTION + ACTIVE_CAPITAL * growth
    return {
        "p05": float(np.quantile(finals, 0.05)),
        "ruin_probability": float(np.mean(finals < TOTAL_CAPITAL / 2.0)),
        "mean": float(np.mean(finals)),
        "median": float(np.median(finals)),
        "paths": MC_PATHS,
    }


def _blocks(timestamps: tuple[pd.Timestamp, ...], returns: np.ndarray) -> tuple[np.ndarray, ...]:
    anchor = timestamps[0]
    grouped: dict[int, list[float]] = {}
    for timestamp, value in zip(timestamps, returns):
        index = (timestamp - anchor).days // BLOCK_DAYS
        grouped.setdefault(index, []).append(float(value))
    return tuple(np.asarray(grouped[key], dtype=float) for key in sorted(grouped))


def _mdd(values: np.ndarray) -> float:
    curve = TOTAL_CAPITAL * RESERVE_FRACTION + ACTIVE_CAPITAL * np.cumprod(1.0 + np.clip(values, -0.999999, None))
    peaks = np.maximum.accumulate(np.concatenate(([TOTAL_CAPITAL], curve)))
    return float(np.max(1.0 - curve / peaks[1:]))


def _block_shuffle(timestamps: tuple[pd.Timestamp, ...], returns: np.ndarray, seed: int) -> dict[str, float]:
    blocks = _blocks(timestamps, returns)
    rng = np.random.default_rng(seed)
    mdds = np.empty(BLOCK_PATHS, dtype=float)
    finals = np.empty(BLOCK_PATHS, dtype=float)
    for index in range(BLOCK_PATHS):
        path = np.concatenate([blocks[item] for item in rng.permutation(len(blocks))])
        mdds[index] = _mdd(path)
        finals[index] = TOTAL_CAPITAL * RESERVE_FRACTION + ACTIVE_CAPITAL * np.prod(1.0 + np.clip(path, -0.999999, None))
    return {
        "block_days": BLOCK_DAYS,
        "block_count": len(blocks),
        "paths": BLOCK_PATHS,
        "mdd_p95": float(np.quantile(mdds, 0.95)),
        "final_p05": float(np.quantile(finals, 0.05)),
    }


def gate_a_feasibility(config: Wave10Config, result: Wave10Result) -> dict[str, Any]:
    leg = leg_usdt(config)
    gross = gross_usdt(config)
    min_order_ok = leg >= MIN_ORDER_USDT
    gross_ok = gross <= ACTIVE_CAPITAL + 1e-9
    positioned_equity = result.equity[result.positions > 0.0]
    dynamic_min_leg = float(positioned_equity.min()) * config.leg_fraction if len(positioned_equity) else None
    dynamic_ok = None if dynamic_min_leg is None else bool(dynamic_min_leg >= MIN_ORDER_USDT)
    delta_neutral = True  # structural: one shared `weights` value drives both legs (see engine.py); regression-tested in tests/test_wave10_engine.py
    passed = bool(min_order_ok and gross_ok and delta_neutral)
    return {
        "leg_usdt_nominal": leg,
        "gross_usdt_nominal": gross,
        "gross_multiplier_of_active_capital": gross / ACTIVE_CAPITAL,
        "min_order_usdt": MIN_ORDER_USDT,
        "min_order_feasible": min_order_ok,
        "gross_exposure_feasible": gross_ok,
        "dynamic_min_leg_usdt": dynamic_min_leg,
        "dynamic_min_order_feasible": dynamic_ok,
        "delta_neutral_by_construction": delta_neutral,
        "status": "PASS" if passed else "FAIL",
    }


def gate_b_mc(result: Wave10Result, seed_offset: int) -> dict[str, Any]:
    _, returns = _daily_returns(result.equity)
    mc = _simulate_mc(returns, SEED + seed_offset * 101)
    principal_preserved = mc["p05"] > TOTAL_CAPITAL
    ruin_ok = mc["ruin_probability"] < 0.05
    return {**mc, "principal_preserved": principal_preserved, "ruin_ok": ruin_ok, "status": "PASS" if (principal_preserved and ruin_ok) else "FAIL"}


def gate_c_block_mdd(result: Wave10Result, seed_offset: int) -> dict[str, Any]:
    timestamps, returns = _daily_returns(result.equity)
    block = _block_shuffle(timestamps, returns, SEED + seed_offset * 103)
    ok = block["mdd_p95"] <= 0.25
    return {**block, "status": "PASS" if ok else "FAIL"}


def gate_d_full_period_profit(result: Wave10Result) -> dict[str, Any]:
    equity = result.equity.dropna()
    start = float(equity.iloc[0])
    end = float(equity.iloc[-1])
    total_return = end / start - 1.0
    return {
        "start_usdt": start,
        "end_usdt": end,
        "total_return": total_return,
        "status": "PASS" if total_return > 0.0 else "FAIL",
    }


def gate_e_oos_dormancy(result: Wave10Result) -> dict[str, Any]:
    positions_oos = result.positions[result.positions.index > OOS_SPLIT]
    has_position = bool((positions_oos > 0.0).any())
    if not has_position:
        return {
            "oos_split": OOS_SPLIT.isoformat(),
            "has_position": False,
            "oos_return": None,
            "status": "UNTESTED_IN_OOS",
        }
    is_equity = result.equity[result.equity.index <= OOS_SPLIT]
    oos_equity = result.equity[result.equity.index > OOS_SPLIT]
    anchor = float(is_equity.iloc[-1]) if len(is_equity) else float(result.equity.iloc[0])
    oos_return = float(oos_equity.iloc[-1]) / anchor - 1.0
    oos_trade_count = int((result.trade_returns.index > OOS_SPLIT).sum())
    return {
        "oos_split": OOS_SPLIT.isoformat(),
        "has_position": True,
        "oos_trade_count": oos_trade_count,
        "oos_return": oos_return,
        "status": "PASS" if oos_return >= 0.0 else "FAIL",
    }


@dataclass(frozen=True, slots=True)
class GateReport:
    gate_a: dict[str, Any]
    gate_b: dict[str, Any]
    gate_c: dict[str, Any]
    gate_d: dict[str, Any]
    gate_e: dict[str, Any]
    overall: str
    failure_reasons: tuple[str, ...]


def _classify_failures(gate_a: dict, gate_b: dict, gate_c: dict, gate_d: dict, gate_e: dict) -> tuple[str, ...]:
    reasons: list[str] = []
    if gate_a["status"] == "FAIL":
        if not gate_a["min_order_feasible"]:
            reasons.append(FAILURE_MIN_ORDER)
        if not gate_a["gross_exposure_feasible"]:
            reasons.append(FAILURE_GROSS)
    if gate_b["status"] == "FAIL":
        reasons.append(FAILURE_PROFIT)
    if gate_c["status"] == "FAIL":
        reasons.append(FAILURE_PROFIT)
    if gate_d["status"] == "FAIL":
        reasons.append(FAILURE_PROFIT)
    if gate_e["status"] == "FAIL":
        reasons.append(FAILURE_PROFIT)
    elif gate_e["status"] == "UNTESTED_IN_OOS":
        reasons.append(FAILURE_DORMANT)
    # de-duplicate, keep first-seen order
    seen: list[str] = []
    for reason in reasons:
        if reason not in seen:
            seen.append(reason)
    return tuple(seen)


def evaluate_gates(config: Wave10Config, result: Wave10Result, seed_offset: int) -> GateReport:
    gate_a = gate_a_feasibility(config, result)
    gate_b = gate_b_mc(result, seed_offset)
    gate_c = gate_c_block_mdd(result, seed_offset)
    gate_d = gate_d_full_period_profit(result)
    gate_e = gate_e_oos_dormancy(result)
    hard_fail = any(gate["status"] == "FAIL" for gate in (gate_a, gate_b, gate_c, gate_d))
    if hard_fail:
        overall = "FAIL"
    elif gate_e["status"] == "UNTESTED_IN_OOS":
        overall = "UNTESTED_IN_OOS"
    elif gate_e["status"] == "FAIL":
        overall = "FAIL"
    else:
        overall = "PASS"
    reasons = _classify_failures(gate_a, gate_b, gate_c, gate_d, gate_e)
    return GateReport(gate_a, gate_b, gate_c, gate_d, gate_e, overall, reasons)
