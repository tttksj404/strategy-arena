#!/usr/bin/env python
"""Wave-26 (frequency-control tournament, C0-C7) pipeline CLI. `fetch` verifies the required
caches are present (no network). `run`/`gates`/`report` are cache-only and deterministic,
mirroring research/wave25_gamble/run_wave25.py's own stage split exactly. `live` is the one
stage that DOES touch the network (task brief: "라이브 신호 확인엔 네트워크 허용"): it fetches the
incremental 1H bars since each symbol's own wave6-cache tail, replays every candidate against
that freshened data (full history, so cooldown/gate state at the live edge is authoritative, not
approximated), and reports whether each is a live actionable signal today -- entirely separate
from (never overwrites) the frozen results/C*.json the Q1-Q5 gates are computed against. See
research/wave26_freq/SPEC.md for the frozen pre-registration this pipeline implements.
"""

from __future__ import annotations

import argparse
from enum import StrEnum
import json
import math
from pathlib import Path
import sys
from typing import Any, Final, assert_never

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

import numpy as np
import pandas as pd  # noqa: PANDAS_OK

from research.wave13_liquidity import costs_measured
from research.wave20_convex import dataio20
from research.wave25_gamble import indicators25
from research.wave26_freq import engine26, gates26
from research.wave26_freq.configs26 import (
    ADX_REGIME_THRESHOLD,
    ADX_REGIME_WINDOW,
    CANDIDATE_IDS,
    CONTROL_SPECS,
    GAMBLE_CAPITAL,
    I5_RESULTS_PATH,
    LIVE_CACHE_DIR,
    P5_STRESS_MULTIPLIER,
    REGISTRY_PATH,
    REPORT_DIR,
    RESULTS_DIR,
    STABLE_CAPITAL,
    SYMBOLS,
    TOTAL_CAPITAL,
    WAVE1_CACHE_DIR,
    WAVE6_CACHE_DIR,
    Z_SCORE_THRESHOLD,
    Z_SCORE_WINDOW_BARS,
)

CANDIDATE_DEFINITIONS: Final[dict[str, str]] = {
    "C0": "V1 재현(±2xATR 양방향 돌파, BTC 단일) -- 통제 없음, 기준선 (wave25 B0와 동일해야 함)",
    "C1": "MACD(12,26,9) 히스토그램 부호전환 + 쿨다운 5일",
    "C2": "MACD(12,26,9) + 쿨다운 5일 + ADX(14)>20",
    "C3": "MACD(12,26,9) + 쿨다운 5일 + ADX(14)>20 + 신호값 20일 z-score>1.0 (3중 통제)",
    "C4": "슈퍼트렌드(10,3.0) 방향전환 + 쿨다운 5일 + ADX(14)>20",
    "C5": "앙상블(B1~B6 중 3+) + 쿨다운 5일 + ADX(14)>20",
    "C6": "앙상블(B1~B6 중 3+) + 쿨다운 10일 + ADX(14)>20 + z-score>1.0 (최강 억제)",
    "C7": "V1 재현 + 쿨다운 5일 + ADX(14)>20 (기준선에도 통제)",
}


class Stage(StrEnum):
    FETCH = "fetch"
    RUN = "run"
    GATES = "gates"
    REPORT = "report"
    LIVE = "live"
    ALL = "all"


class Wave26CliError(Exception):
    pass


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _series_payload(series: pd.Series) -> list[dict[str, Any]]:
    clean = series.dropna()
    return [{"timestamp": str(timestamp), "value": float(value)} for timestamp, value in clean.items()]


def _series_from_payload(records: list[dict[str, Any]]) -> pd.Series:
    if not records:
        return pd.Series(dtype=float)
    index = pd.DatetimeIndex([pd.Timestamp(item["timestamp"]) for item in records])
    values = [float(item["value"]) for item in records]
    return pd.Series(values, index=index, dtype=float).sort_index()


def _trade_payload(trade: engine26.Trade) -> dict[str, Any]:
    return {
        "symbol": trade.symbol,
        "direction": trade.direction,
        "entry_time": str(trade.entry_time) if trade.entry_time is not None else None,
        "exit_time": str(trade.exit_time) if trade.exit_time is not None else None,
        "entry_price": trade.entry_price,
        "exit_price": trade.exit_price,
        "entry_equity_usdt": trade.entry_equity_usdt,
        "pnl_usdt": trade.pnl_usdt,
        "pnl_fraction": trade.pnl_fraction,
        "exit_reason": trade.exit_reason,
        "cost_usdt": trade.cost_usdt,
    }


def _final_from_series(records: list[dict[str, Any]], default: float) -> float:
    clean = [r for r in records if r.get("value") is not None]
    return float(clean[-1]["value"]) if clean else default


# ---------------------------------------------------------------------------
# fetch
# ---------------------------------------------------------------------------


def _stage_fetch() -> None:
    missing: list[str] = []
    for symbol in SYMBOLS:
        hourly_path = WAVE6_CACHE_DIR / f"binance_fapi_{symbol}_1h.csv.gz"
        daily_path = WAVE1_CACHE_DIR / f"binance_fapi_{symbol}_1d.csv.gz"
        if not hourly_path.exists():
            missing.append(str(hourly_path))
        if not daily_path.exists():
            missing.append(str(daily_path))
    from research.wave13_liquidity.collect_spreads import CACHE_DIR as W13_CACHE_DIR

    if not (W13_CACHE_DIR / "measured_spreads.json").exists():
        missing.append(str(W13_CACHE_DIR / "measured_spreads.json"))
    if not I5_RESULTS_PATH.exists():
        missing.append(str(I5_RESULTS_PATH))
    if missing:
        raise Wave26CliError("fetch: missing required cache files:\n  " + "\n  ".join(missing))
    print("fetch: all required caches present (wave1/wave6/wave13_liquidity/wave18_idle) -- no network fetch needed for the run/gates/report stages")


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def _stage_run(only: str | None) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    mapping = costs_measured.fit_mapping()
    print(f"run: measured-cost mapping fitted from {mapping.raw_point_count} Bitget points -> {len(mapping.anchor_bp)} isotonic buckets (worst={mapping.worst_bp:.4f}bp, best={mapping.best_bp:.4f}bp)")
    stable_equity, stable_info = engine26.load_stable_leg(STABLE_CAPITAL)
    print(f"run: stable leg loaded from {stable_info['source']} (I5 full_period_annualized={stable_info['source_full_period_annualized']:.4f})")

    for candidate_id in CANDIDATE_IDS:
        if only is not None and candidate_id != only:
            continue
        result = engine26.run_candidate(candidate_id, mapping=mapping, stress_multiplier=1.0)
        stressed_result = engine26.run_candidate(candidate_id, mapping=mapping, stress_multiplier=P5_STRESS_MULTIPLIER)
        stressed_final = float(stressed_result.equity.dropna().iloc[-1]) if len(stressed_result.equity.dropna()) else GAMBLE_CAPITAL
        combined = engine26.combine_portfolio(stable_equity, result.equity, GAMBLE_CAPITAL)
        payload = {
            "candidate_id": candidate_id,
            "family": "wave26_freq",
            "definition": CANDIDATE_DEFINITIONS[candidate_id],
            "control": result.metadata.get("control", {}),
            "base_family": result.metadata.get("base_family"),
            "config": result.metadata.get("config", {}),
            "capital_contract": {"total_capital_usdt": TOTAL_CAPITAL, "gamble_capital_usdt": GAMBLE_CAPITAL, "stable_capital_usdt": STABLE_CAPITAL},
            "cost_model": "single-leg (maker 0.02% + wave13_liquidity.costs_measured measured-slippage mapping, unmodified) per entry/exit transition -- reused from research.wave20_convex.engine20 via wave25_gamble.engine25",
            "stable_leg_source": stable_info,
            "gamble_equity": _series_payload(result.equity),
            "combined_equity": _series_payload(combined["total"]),
            "trades": [_trade_payload(t) for t in result.trades],
            "symbols_used": list(result.symbols_used),
            "metadata": {k: v for k, v in result.metadata.items() if k not in ("config",)},
            "stress_test": {"stress_multiplier": P5_STRESS_MULTIPLIER, "final_gamble_equity_usdt": stressed_final, "n_trades": stressed_result.metadata.get("n_trades")},
        }
        _save_json(RESULTS_DIR / f"{candidate_id}.json", payload)
        final_gamble = float(result.equity.dropna().iloc[-1]) if len(result.equity.dropna()) else GAMBLE_CAPITAL
        final_combined = float(combined["total"].dropna().iloc[-1]) if len(combined["total"].dropna()) else TOTAL_CAPITAL
        total_cost = result.metadata.get("total_cost_usdt", 0.0)
        print(f"run: {candidate_id} done (gamble_final=${final_gamble:.4f}, combined_final=${final_combined:.2f}, n_trades={len(result.trades)}, total_cost=${total_cost:.4f}, stressed_final=${stressed_final:.4f})")


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------


def _stage_gates(only: str | None) -> None:
    baseline_path = RESULTS_DIR / "C0.json"
    baseline_payload = _load_json(baseline_path)
    c0_gamble_equity = _series_from_payload(baseline_payload["gamble_equity"])
    c0_final_usdt = _final_from_series(baseline_payload["gamble_equity"], GAMBLE_CAPITAL)
    print(f"gates: C0 baseline sleeve final=${c0_final_usdt:.4f} (n_trades={len(baseline_payload['trades'])})")

    for seed_offset, candidate_id in enumerate(CANDIDATE_IDS):
        if only is not None and candidate_id != only:
            continue
        path = RESULTS_DIR / f"{candidate_id}.json"
        payload = _load_json(path)
        gamble_equity = _series_from_payload(payload["gamble_equity"])
        combined_equity = _series_from_payload(payload["combined_equity"])
        trades_payload = payload["trades"]
        stressed_final = float(payload["stress_test"]["final_gamble_equity_usdt"])
        total_cost = float(payload["metadata"].get("total_cost_usdt", 0.0))
        report = gates26.evaluate_candidate(
            candidate_id, gamble_equity, trades_payload, combined_equity, c0_gamble_equity, c0_final_usdt, stressed_final, total_cost, seed_offset
        )
        payload["gates_report"] = report
        payload["full_period_cagr_combined"] = gates26.full_period_annualized(combined_equity)
        payload["full_period_cagr_gamble_only"] = gates26.full_period_annualized(gamble_equity)
        _save_json(path, payload)
        statuses = {g["gate_id"]: g["status"] for g in report["gates"]}
        print(f"gates: {candidate_id} -> {statuses} overall={report['overall']['status']} promoted={report['overall']['promoted']}")


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def _stage_report() -> None:
    from research.wave26_freq.reporting26 import write_wave26_report

    live_path = RESULTS_DIR / "live_signals.json"
    live_payload = _load_json(live_path) if live_path.exists() else None
    write_wave26_report(RESULTS_DIR, REPORT_DIR, REGISTRY_PATH, live_payload)
    print(f"report: wrote {REPORT_DIR / 'wave26_report.md'} and {REGISTRY_PATH}")


# ---------------------------------------------------------------------------
# live -- the one network-touching stage.
# ---------------------------------------------------------------------------


def _fetch_incremental_hourly(symbol: str, cached: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """Appends any 1H bars newer than `cached`'s own tail via a live Binance fetch. Returns
    (frame, network_used). Saved under LIVE_CACHE_DIR (this wave's OWN cache), never written
    back into research/wave6/cache."""
    import requests

    from research.wave1.fetch_binance import BinanceKlineRequest, fetch_klines

    last_ts = cached.index.max()
    start_ms = int((pd.Timestamp(last_ts) + pd.Timedelta(hours=1)).timestamp() * 1000)
    end_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
    if start_ms > end_ms:
        return cached, False
    with requests.Session() as session:
        increment = fetch_klines(BinanceKlineRequest(symbol, "1h", start_ms, end_ms), session)
    if increment.empty:
        return cached, True
    combined = pd.concat([cached, increment[list(cached.columns)]]).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]
    return combined, True


def _v1_family_live_status(hourly_btc: pd.DataFrame, result: engine26.GambleResult, adx_gate: bool) -> dict[str, Any]:
    """C0/C7 live status. If the last authoritative trade is still open at data end, report the
    held position directly (identical convention to wave25's own _b0_live_status). Otherwise
    replay V1's OWN flat-state anchor-arming rule from the bar the last trade exited (same
    replay wave25 already does), and additionally report whether a would-be breakout right now
    is actually admissible given the CURRENT cooldown/ADX state (engine26's own additions). The
    cooldown state itself is read directly from `result.metadata` (authoritative, produced by
    the simulation that just ran) rather than re-derived from a days count here."""
    from research.wave20_convex import engine20 as wave20_engine
    from research.wave20_convex.configs20 import V1_CONFIG

    trades = result.trades
    daily = dataio20.resample_hourly_to_daily(hourly_btc)
    vol20 = wave20_engine.realized_vol(daily["close"], V1_CONFIG.vol_window_days)
    vol_pct_rank = wave20_engine.trailing_percentile_rank(vol20, V1_CONFIG.vol_percentile_lookback_days)
    armable_daily = (vol_pct_rank < V1_CONFIG.vol_percentile_threshold).shift(1).fillna(False)
    armable_hourly = armable_daily.reindex(hourly_btc.index, method="ffill").fillna(False).astype(bool).to_numpy()
    atr_daily_lagged = wave20_engine.atr(daily, V1_CONFIG.atr_window_days).shift(1)
    atr_hourly = atr_daily_lagged.reindex(hourly_btc.index, method="ffill").to_numpy(dtype=float)
    close = hourly_btc["close"].to_numpy(dtype=float)

    if trades and trades[-1].exit_reason == "end_of_data":
        last = trades[-1]
        current_price = float(close[-1])
        unrealized_pct = (current_price / last.entry_price - 1.0) * last.direction if last.entry_price else None
        return {"status": "HOLDING", "direction": last.direction, "entry_price": last.entry_price, "entry_time": str(last.entry_time), "current_price": current_price, "unrealized_pct": unrealized_pct}

    start_idx = 0
    if trades:
        try:
            start_idx = hourly_btc.index.get_loc(trades[-1].exit_time)
        except KeyError:
            start_idx = 0
    anchor = float("nan")
    for i in range(start_idx, len(close)):
        if armable_hourly[i] and np.isnan(anchor):
            anchor = float(close[i])
    if np.isnan(anchor):
        return {"status": "FLAT_NOT_ARMED", "reason": "BTC currently outside the low-realized-vol regime (20d vol percentile >= 30th) -- V1 is not watching for a breakout right now", "current_price": float(close[-1])}

    atr_now = float(atr_hourly[-1]) if len(atr_hourly) else float("nan")
    threshold = V1_CONFIG.atr_multiplier * atr_now if not np.isnan(atr_now) else None
    cooldown_remaining_bars = int(result.metadata.get("entry_admission", {}).get("final_cooldown_remaining_bars", 0))
    adx_ok = True
    if adx_gate:
        adx_now = indicators25.adx_dmi(hourly_btc, ADX_REGIME_WINDOW)["adx"].iloc[-1]
        adx_ok = bool(adx_now > ADX_REGIME_THRESHOLD) if not pd.isna(adx_now) else False
    gate_note = "admissible now" if (cooldown_remaining_bars <= 0 and adx_ok) else f"BLOCKED (cooldown_bars_left={cooldown_remaining_bars}, adx_ok={adx_ok})"
    return {
        "status": "FLAT_ARMED",
        "anchor": anchor,
        "current_price": float(close[-1]),
        "atr_now": atr_now,
        "threshold": threshold,
        "distance_to_long_trigger": (anchor + threshold) - float(close[-1]) if threshold is not None else None,
        "distance_to_short_trigger": float(close[-1]) - (anchor - threshold) if threshold is not None else None,
        "admission_if_triggered": gate_note,
    }


def _controlled_candidate_live_status(candidate_id: str, symbol: str, hourly: pd.DataFrame, daily: pd.DataFrame, result: engine26.GambleResult) -> dict[str, Any]:
    spec = CONTROL_SPECS[candidate_id]
    symbol_trades = [t for t in result.trades if t.symbol == symbol]
    close = hourly["close"].to_numpy(dtype=float)
    if symbol_trades and symbol_trades[-1].exit_time == hourly.index[-1] and symbol_trades[-1].exit_reason == "end_of_data":
        last = symbol_trades[-1]
        current_price = float(close[-1])
        unrealized_pct = (current_price / last.entry_price - 1.0) * last.direction if last.entry_price else None
        return {"status": "HOLDING", "direction": last.direction, "entry_price": last.entry_price, "entry_time": str(last.entry_time), "current_price": current_price, "unrealized_pct": unrealized_pct}

    raw_signal_series = engine26.BASE_SIGNAL_FNS[spec.base_family](hourly, daily)
    last_signal = float(raw_signal_series.iloc[-1]) if len(raw_signal_series) else 0.0
    if last_signal == 0.0:
        return {"status": "FLAT_NO_SIGNAL", "current_price": float(close[-1])}

    cooldown_remaining_bars = int(result.metadata.get("entry_admission", {}).get("final_cooldown_remaining_bars", 0))
    adx_ok = True
    if spec.adx_gate:
        adx_now = indicators25.adx_dmi(hourly, ADX_REGIME_WINDOW)["adx"].iloc[-1]
        adx_ok = bool(adx_now > ADX_REGIME_THRESHOLD) if not pd.isna(adx_now) else False
    z_ok = True
    if spec.z_gate:
        strength_fn = engine26.STRENGTH_FNS[spec.base_family]
        z_mask = engine26.zscore_gate_mask(strength_fn(hourly, daily), Z_SCORE_WINDOW_BARS, Z_SCORE_THRESHOLD)
        z_ok = bool(z_mask.iloc[-1]) if len(z_mask) else False

    direction_word = "LONG" if last_signal > 0.0 else "SHORT"
    if cooldown_remaining_bars > 0:
        return {"status": f"SIGNAL_BLOCKED_COOLDOWN_{direction_word}", "current_price": float(close[-1]), "cooldown_bars_left": cooldown_remaining_bars}
    if not (adx_ok and z_ok):
        return {"status": f"SIGNAL_BLOCKED_GATE_{direction_word}", "current_price": float(close[-1]), "adx_ok": adx_ok, "z_ok": z_ok}
    status = "FRESH_LONG_SIGNAL" if last_signal > 0.0 else "FRESH_SHORT_SIGNAL"
    return {"status": status, "current_price": float(close[-1]), "note": "fires at the next available bar's open (t->t+1 fill discipline)"}


def _stage_live() -> None:
    LIVE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    mapping = costs_measured.fit_mapping()
    network_used = False
    live_hourly: dict[str, pd.DataFrame] = {}
    for symbol in SYMBOLS:
        cached = engine26.load_hourly(symbol)
        try:
            extended, used = _fetch_incremental_hourly(symbol, cached)
        except Exception as error:  # noqa: BLE001 -- network can fail in many ways; degrade to cache rather than abort the whole stage
            print(f"live: WARNING -- incremental fetch for {symbol} failed ({error!r}); falling back to cached data only", file=sys.stderr)
            extended, used = cached, False
        network_used = network_used or used
        live_hourly[symbol] = extended
        from research.wave1.common import save_frame

        save_frame(LIVE_CACHE_DIR / f"binance_fapi_{symbol}_1h_live.csv.gz", extended)
        print(f"live: {symbol} 1H rows={len(extended)} (cache tail was {cached.index.max()}, now through {extended.index.max()})")

    live_daily = {symbol: dataio20.resample_hourly_to_daily(live_hourly[symbol]) for symbol in SYMBOLS}

    signals_payload: dict[str, Any] = {"generated_at_utc": str(pd.Timestamp.now(tz="UTC")), "network_used": network_used, "candidates": {}}

    c0_result = engine26.run_c0_live(live_hourly["BTCUSDT"], mapping, stress_multiplier=1.0)
    signals_payload["candidates"]["C0"] = {"BTCUSDT": _v1_family_live_status(live_hourly["BTCUSDT"], c0_result, adx_gate=False)}

    for candidate_id in ("C1", "C2", "C3", "C4", "C5", "C6"):
        runner = engine26.RUNNERS[candidate_id]
        result = runner(mapping=mapping, stress_multiplier=1.0, hourly_frames=live_hourly, daily_frames=live_daily)
        per_symbol = {symbol: _controlled_candidate_live_status(candidate_id, symbol, live_hourly[symbol], live_daily[symbol], result) for symbol in SYMBOLS}
        signals_payload["candidates"][candidate_id] = per_symbol
        actionable = [f"{s}:{v['status']}" for s, v in per_symbol.items() if v["status"] != "FLAT_NO_SIGNAL"]
        print(f"live: {candidate_id} -> " + (", ".join(actionable) if actionable else "no symbol actionable right now"))

    c7_result = engine26.run_c7(mapping=mapping, stress_multiplier=1.0, hourly_frames=live_hourly, daily_frames=live_daily)
    signals_payload["candidates"]["C7"] = {"BTCUSDT": _v1_family_live_status(live_hourly["BTCUSDT"], c7_result, adx_gate=True)}

    _save_json(RESULTS_DIR / "live_signals.json", signals_payload)
    print(f"live: wrote {RESULTS_DIR / 'live_signals.json'} (network_used={network_used})")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Wave-26 frequency-control tournament (C0-C7) pipeline")
    parser.add_argument("--stage", required=True, type=Stage, choices=tuple(Stage))
    parser.add_argument("--only", choices=CANDIDATE_IDS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        match args.stage:
            case Stage.FETCH:
                _stage_fetch()
            case Stage.RUN:
                _stage_run(args.only)
            case Stage.GATES:
                _stage_gates(args.only)
            case Stage.REPORT:
                _stage_report()
            case Stage.LIVE:
                _stage_live()
            case Stage.ALL:
                _stage_fetch()
                _stage_run(args.only)
                _stage_gates(args.only)
                _stage_live()
                _stage_report()
            case unreachable:
                assert_never(unreachable)
    except (FileNotFoundError, Wave26CliError, engine26.Wave26Error, RuntimeError, ValueError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
