# Wave-7 carry+momentum combination engine.
#
# Self-contained by design: the only cross-wave dependency this package takes is the
# explicitly sanctioned import of research.validation.deep_stats (see deepval_w7.py).
# Everything else -- loading W2c/W3c results, loading wave-1 cache CSVs, computing the
# funding-APR regime score and the BTC<MA200 crash guard -- is reimplemented locally
# from the cached data files so that wave7 never depends on (and never has to modify)
# code outside research/wave7/.
#
# Combination contract (see SPEC.md):
#   - Inputs are the *daily return* series derived from W2c's and W3c's own equity
#     curves (equity.pct_change()), aligned on the intersection of both calendars.
#   - W7a/W7b are static daily-rebalanced blends.
#   - W7c switches between "carry regime active" (100% carry) and a 60/40 blend using
#     the mean of BTC and ETH 7-day funding APR (the codebase's own "majors" carry
#     proxy -- see F1e/W2b/W2e's majors_only convention) compared against the same
#     0.15 APR threshold W2c itself uses (see research/wave2/results/W2c.json
#     metadata.candidate_config.threshold_apr). The signal is shifted by one day so
#     today's weights only ever use information known as of yesterday's close.
#   - W7d layers a momentum crash guard on top of W7c: whenever BTC's close is below
#     its (shifted) 200-day moving average -- the same regime rule W3d used -- the
#     momentum sleeve's realized return that day is zeroed (sits in cash). The carry
#     weight is left unchanged; freed capital is not reallocated to carry.

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Final

import pandas as pd  # noqa: PANDAS_OK


INITIAL_CAPITAL: Final = 300.0
FUNDING_WINDOW_DAYS: Final = 7
FUNDING_OBSERVATIONS: Final = FUNDING_WINDOW_DAYS * 3  # 3 fundings/day on Binance USDT-M
REGIME_THRESHOLD_APR: Final = 0.15  # matches W2c's own candidate_config.threshold_apr
MA200_WINDOW: Final = 200
CAPITAL_BUFFER_FRACTION: Final = 0.9  # $300 * 0.9 = $270 simultaneous-margin ceiling

W7_CANDIDATE_IDS: Final = ("W7a", "W7b", "W7c", "W7d")
CANDIDATE_DEFINITIONS: Final = {
    "W7a": "정적 70/30 (캐리/모멘텀), 일 리밸런스",
    "W7b": "정적 60/40 (캐리/모멘텀), 일 리밸런스",
    "W7c": "레짐 스위치: BTC/ETH 7d 펀딩 APR(평균)>15% -> 캐리100/모멘텀0, 아니면 캐리60/모멘텀40",
    "W7d": "W7c + 모멘텀 크래시가드: BTC<MA200(시프트) 시 모멘텀 슬리브 현금(0)",
}

STATIC_WEIGHTS: Final = {"W7a": (0.7, 0.3), "W7b": (0.6, 0.4)}


@dataclass(frozen=True, slots=True)
class Wave7Error(Exception):
    """Raised for wave-7 pipeline input/contract violations."""

    message: str

    def __str__(self) -> str:
        return self.message


def _read_gz_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise Wave7Error(f"missing cache file: {path}")
    frame = pd.read_csv(path, compression="gzip", encoding="utf-8")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, format="ISO8601")
    return frame.set_index("timestamp").sort_index()


def load_equity_series(path: Path) -> pd.Series:
    """Load the {timestamp,value} equity list of a wave2/wave3-style result JSON."""
    if not path.exists():
        raise Wave7Error(f"missing result file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    equity = payload.get("equity") if isinstance(payload, dict) else None
    if not isinstance(equity, list) or not equity:
        raise Wave7Error(f"{path}: 'equity' field missing or empty")
    data = {pd.Timestamp(item["timestamp"]): float(item["value"]) for item in equity}
    series = pd.Series(data, dtype=float).sort_index()
    series.index.name = "timestamp"
    if series.le(0.0).any():
        raise Wave7Error(f"{path}: equity must remain positive")
    return series


def load_metadata(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metadata = payload.get("metadata") if isinstance(payload, dict) else None
    return metadata if isinstance(metadata, dict) else {}


def load_component_returns(w2c_path: Path, w3c_path: Path) -> tuple[pd.Series, pd.Series]:
    """Return (carry_returns, momentum_returns) daily simple returns aligned on the
    intersection of W2c's and W3c's own calendars. Each series is computed via
    pct_change() on its *own* full-history equity curve first (so the first common
    day's return still reflects the immediately preceding day within that series),
    and only intersected afterwards.
    """
    equity_c = load_equity_series(w2c_path)
    equity_m = load_equity_series(w3c_path)
    returns_c_full = equity_c.pct_change().dropna()
    returns_m_full = equity_m.pct_change().dropna()
    common_index = returns_c_full.index.intersection(returns_m_full.index).sort_values()
    if len(common_index) < 30:
        raise Wave7Error("insufficient overlap between W2c and W3c daily return series")
    carry_returns = returns_c_full.reindex(common_index)
    momentum_returns = returns_m_full.reindex(common_index)
    if carry_returns.isna().any() or momentum_returns.isna().any():
        raise Wave7Error("aligned component returns contain gaps")
    return carry_returns, momentum_returns


def funding_score(funding: pd.Series, window_days: int = FUNDING_WINDOW_DAYS) -> pd.Series:
    """Trailing rolling-mean funding rate annualized to APR (x3 fundings/day x365).

    Identical formula to research.wave1.fam_funding.funding_score (reimplemented here
    to keep wave7 free of cross-wave code imports); observations are counted as raw
    rows (3/day nominal Binance cadence), not calendar-time, matching the rest of the
    codebase's convention.
    """
    observations = window_days * 3
    return funding.rolling(observations, min_periods=observations).mean() * 3.0 * 365.0


def load_carry_regime_signal(cache_dir: Path, common_index: pd.DatetimeIndex) -> pd.Series:
    """BTC/ETH majors funding-APR regime flag, shifted +1 day (no lookahead).

    Regime APR is the mean of BTC's and ETH's own 7-day funding APR score (skipping
    whichever symbol has no score yet, e.g. ETH funding cache starts later than BTC's);
    NaN (neither symbol scored yet) compares as False, matching this codebase's
    existing "insufficient data -> conservative default" convention (see
    research.wave3.engine.btc_above_ma200).
    """
    btc = _read_gz_csv(cache_dir / "binance_funding_BTCUSDT.csv.gz")["funding_rate"]
    eth = _read_gz_csv(cache_dir / "binance_funding_ETHUSDT.csv.gz")["funding_rate"]
    btc_daily = funding_score(btc).resample("1D").last()
    eth_daily = funding_score(eth).resample("1D").last()
    regime_apr = pd.concat([btc_daily.rename("btc"), eth_daily.rename("eth")], axis=1).mean(axis=1, skipna=True)
    active = regime_apr > REGIME_THRESHOLD_APR
    active_shifted = active.shift(1).fillna(False).astype(bool)
    return active_shifted.reindex(common_index).fillna(False).astype(bool)


def load_momentum_crash_guard(cache_dir: Path, common_index: pd.DatetimeIndex) -> pd.Series:
    """True when BTC's close is below its (shifted) 200-day moving average.

    Reimplements research.wave3.engine.btc_above_ma200's exact semantics (MA200
    undefined during warmup -> treated as "not above", i.e. crash-guard True) so W7d's
    guard matches the precedent W3d already established for this exact regime rule.
    """
    close = _read_gz_csv(cache_dir / "binance_fapi_BTCUSDT_1d.csv.gz")["close"]
    ma200 = close.rolling(MA200_WINDOW, min_periods=MA200_WINDOW).mean()
    above_200 = (close >= ma200).where(ma200.notna(), False)
    below_200 = ~above_200
    below_shifted = below_200.shift(1).fillna(True).astype(bool)
    return below_shifted.reindex(common_index).fillna(True).astype(bool)


@dataclass(frozen=True, slots=True)
class CombinedResult:
    candidate_id: str
    definition: str
    equity: pd.Series
    combined_returns: pd.Series
    carry_weight: pd.Series
    momentum_weight: pd.Series
    carry_contribution: pd.Series
    momentum_contribution: pd.Series
    carry_regime_active_ratio: float
    crash_guard_active_ratio: float


def equity_from_returns(returns: pd.Series) -> pd.Series:
    """Compound a daily-return series into a $300-anchored equity curve.

    Public so run_wave7.py's gates stage can build a carry-alone equity curve over
    the exact same aligned dates as the combined candidates, for an apples-to-apples
    Sharpe/OOS comparison (SPEC gate: "결합 Sharpe > 캐리 단독").
    """
    growth = (1.0 + returns).cumprod() * INITIAL_CAPITAL
    anchor = returns.index[0] - pd.Timedelta(days=1)
    equity = pd.concat([pd.Series([INITIAL_CAPITAL], index=[anchor]), growth])
    equity.index.name = "timestamp"
    return equity


def build_candidate(
    candidate_id: str,
    carry_returns: pd.Series,
    momentum_returns: pd.Series,
    carry_regime_active: pd.Series,
    momentum_crash_guard: pd.Series,
) -> CombinedResult:
    if candidate_id not in W7_CANDIDATE_IDS:
        raise Wave7Error(f"unknown wave7 candidate: {candidate_id}")
    index = carry_returns.index
    if candidate_id in STATIC_WEIGHTS:
        carry_static, momentum_static = STATIC_WEIGHTS[candidate_id]
        carry_weight = pd.Series(carry_static, index=index)
        momentum_weight = pd.Series(momentum_static, index=index)
    else:
        active = carry_regime_active.reindex(index).fillna(False).astype(bool)
        carry_weight = active.map({True: 1.0, False: 0.6}).astype(float)
        momentum_weight = active.map({True: 0.0, False: 0.4}).astype(float)
        if candidate_id == "W7d":
            guard = momentum_crash_guard.reindex(index).fillna(True).astype(bool)
            momentum_weight = momentum_weight.where(~guard, 0.0)
    carry_contribution = carry_weight * carry_returns
    momentum_contribution = momentum_weight * momentum_returns
    combined_returns = carry_contribution + momentum_contribution
    equity = equity_from_returns(combined_returns)
    return CombinedResult(
        candidate_id=candidate_id,
        definition=CANDIDATE_DEFINITIONS[candidate_id],
        equity=equity,
        combined_returns=combined_returns,
        carry_weight=carry_weight,
        momentum_weight=momentum_weight,
        carry_contribution=carry_contribution,
        momentum_contribution=momentum_contribution,
        carry_regime_active_ratio=float(carry_regime_active.reindex(index).fillna(False).astype(bool).mean()),
        crash_guard_active_ratio=float(momentum_crash_guard.reindex(index).fillna(True).astype(bool).mean()),
    )


def capital_reality_check(
    candidate_id: str,
    carry_weight: pd.Series,
    momentum_weight: pd.Series,
    carry_meta: dict,
    momentum_meta: dict,
) -> dict:
    """SPEC section 5 arithmetic sanity check (no new data, existing metadata only).

    Two independent sub-checks, both using only max/min_position_weight and
    min_order_usdt already recorded on W2c's and W3c's own result metadata:
      1. buffer_ok: the combined blend weight (carry+momentum) never exceeds the
         $300 x 0.9 = $270 simultaneous-margin ceiling, i.e. combined weight <= 0.9.
      2. order_ok: at the smallest nonzero blend weight each sleeve actually uses,
         that sleeve's own smallest per-leg position (min_position_weight fraction of
         its slice of the $300 pool) still clears the $5 minimum order size.
    """
    combined = carry_weight + momentum_weight
    max_combined = float(combined.max())
    mean_combined = float(combined.mean())
    buffer_ok = max_combined <= CAPITAL_BUFFER_FRACTION

    min_order = max(float(carry_meta.get("min_order_usdt", 5.0)), float(momentum_meta.get("min_order_usdt", 5.0)))
    carry_min_pw = float(carry_meta.get("min_position_weight", 0.0))
    momentum_min_pw = float(momentum_meta.get("min_position_weight", 0.0))

    carry_nonzero = carry_weight[carry_weight > 0.0]
    momentum_nonzero = momentum_weight[momentum_weight > 0.0]
    carry_min_blend_weight = float(carry_nonzero.min()) if not carry_nonzero.empty else 0.0
    momentum_min_blend_weight = float(momentum_nonzero.min()) if not momentum_nonzero.empty else 0.0

    carry_min_leg_usd = carry_min_pw * carry_min_blend_weight * INITIAL_CAPITAL
    momentum_min_leg_usd = momentum_min_pw * momentum_min_blend_weight * INITIAL_CAPITAL

    carry_order_ok = carry_min_blend_weight == 0.0 or carry_min_leg_usd >= min_order
    momentum_order_ok = momentum_min_blend_weight == 0.0 or momentum_min_leg_usd >= min_order

    status = "PASS" if (buffer_ok and carry_order_ok and momentum_order_ok) else "FAIL"
    return {
        "candidate_id": candidate_id,
        "max_combined_weight": max_combined,
        "mean_combined_weight": mean_combined,
        "buffer_threshold": CAPITAL_BUFFER_FRACTION,
        "buffer_ok": buffer_ok,
        "min_order_usdt": min_order,
        "carry_min_position_weight": carry_min_pw,
        "carry_min_blend_weight": carry_min_blend_weight,
        "carry_min_leg_usd": carry_min_leg_usd,
        "carry_order_ok": carry_order_ok,
        "momentum_min_position_weight": momentum_min_pw,
        "momentum_min_blend_weight": momentum_min_blend_weight,
        "momentum_min_leg_usd": momentum_min_leg_usd,
        "momentum_order_ok": momentum_order_ok,
        "status": status,
        "note": (
            f"combined weight max={max_combined:.2f} (buffer<=0.9: {buffer_ok}); "
            f"carry min leg=${carry_min_leg_usd:.2f} (>=${min_order:.2f}: {carry_order_ok}); "
            f"momentum min leg=${momentum_min_leg_usd:.2f} (>=${min_order:.2f}: {momentum_order_ok})"
        ),
    }


def _series_payload(series: pd.Series) -> list[dict]:
    return [{"timestamp": pd.Timestamp(idx).isoformat(), "value": float(value)} for idx, value in series.items()]


def series_from_payload(items: list[dict]) -> pd.Series:
    """Inverse of the equity/trade_returns list-of-{timestamp,value} JSON encoding."""
    data = {pd.Timestamp(item["timestamp"]): float(item["value"]) for item in items}
    series = pd.Series(data, dtype=float).sort_index()
    series.index.name = "timestamp"
    return series


def to_result_payload(result: CombinedResult, reality: dict, carry_meta: dict, momentum_meta: dict) -> dict:
    return {
        "candidate_id": result.candidate_id,
        "family": "F7",
        "definition": result.definition,
        "equity": _series_payload(result.equity),
        "trade_returns": _series_payload(result.combined_returns),
        "component_contribution": {
            "carry": _series_payload(result.carry_contribution),
            "momentum": _series_payload(result.momentum_contribution),
        },
        "weights": {
            "carry": _series_payload(result.carry_weight),
            "momentum": _series_payload(result.momentum_weight),
        },
        "capital_reality": reality,
        "metadata": {
            "intended_factor": "carry_momentum_blend",
            "exploratory_only": False,
            "cost_model_valid": bool(carry_meta.get("cost_model_valid", True) and momentum_meta.get("cost_model_valid", True)),
            "data_valid": bool(carry_meta.get("data_valid", True) and momentum_meta.get("data_valid", True)),
            "carry_source": "research/wave2/results/W2c.json",
            "momentum_source": "research/wave3/results/W3c.json",
            "regime_threshold_apr": REGIME_THRESHOLD_APR if result.candidate_id in ("W7c", "W7d") else None,
            "regime_signal": (
                "mean(BTC,ETH) 7d funding APR > 0.15, shifted +1 day" if result.candidate_id in ("W7c", "W7d") else None
            ),
            "crash_guard": (
                "BTC close < 200d MA (shifted +1 day) -> momentum sleeve cash" if result.candidate_id == "W7d" else None
            ),
            "carry_regime_active_ratio": result.carry_regime_active_ratio if result.candidate_id in ("W7c", "W7d") else None,
            "crash_guard_active_ratio": result.crash_guard_active_ratio if result.candidate_id == "W7d" else None,
            "component_metadata": {
                "carry": {
                    "max_concurrent_positions": carry_meta.get("max_concurrent_positions"),
                    "max_position_weight": carry_meta.get("max_position_weight"),
                    "min_position_weight": carry_meta.get("min_position_weight"),
                    "min_order_usdt": carry_meta.get("min_order_usdt"),
                },
                "momentum": {
                    "max_concurrent_positions": momentum_meta.get("max_concurrent_positions"),
                    "max_position_weight": momentum_meta.get("max_position_weight"),
                    "min_position_weight": momentum_meta.get("min_position_weight"),
                    "min_order_usdt": momentum_meta.get("min_order_usdt"),
                },
            },
        },
    }


def run_all(w2c_path: Path, w3c_path: Path, cache_dir: Path) -> dict[str, dict]:
    """Build all four wave-7 candidates and return {candidate_id: json_payload}."""
    carry_returns, momentum_returns = load_component_returns(w2c_path, w3c_path)
    carry_active = load_carry_regime_signal(cache_dir, carry_returns.index)
    crash_guard = load_momentum_crash_guard(cache_dir, carry_returns.index)
    carry_meta = load_metadata(w2c_path)
    momentum_meta = load_metadata(w3c_path)
    payloads: dict[str, dict] = {}
    for candidate_id in W7_CANDIDATE_IDS:
        result = build_candidate(candidate_id, carry_returns, momentum_returns, carry_active, crash_guard)
        reality = capital_reality_check(candidate_id, result.carry_weight, result.momentum_weight, carry_meta, momentum_meta)
        payloads[candidate_id] = to_result_payload(result, reality, carry_meta, momentum_meta)
    return payloads


__all__ = [
    "CANDIDATE_DEFINITIONS",
    "CAPITAL_BUFFER_FRACTION",
    "CombinedResult",
    "INITIAL_CAPITAL",
    "REGIME_THRESHOLD_APR",
    "STATIC_WEIGHTS",
    "W7_CANDIDATE_IDS",
    "Wave7Error",
    "build_candidate",
    "capital_reality_check",
    "equity_from_returns",
    "funding_score",
    "load_carry_regime_signal",
    "load_component_returns",
    "load_equity_series",
    "load_metadata",
    "load_momentum_crash_guard",
    "run_all",
    "series_from_payload",
    "to_result_payload",
]
