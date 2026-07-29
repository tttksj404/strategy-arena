#!/usr/bin/env python
"""Wave-25 (short-horizon gamble tournament, B0-B7) pipeline CLI. `fetch` verifies the
required caches are present (no network). `run`/`gates`/`report` are cache-only and
deterministic, mirroring research/wave20_convex/run_wave20.py's own stage split exactly.
`live` is the one stage that DOES touch the network (task brief: "라이브 신호 확인엔 네트워크
허용"): it fetches the incremental 1H bars since each symbol's own wave6-cache tail, replays
every candidate against that freshened data, and reports whether each is a live actionable
signal today -- entirely separate from (never overwrites) the frozen results/B*.json the P1-P5
gates are computed against. See research/wave25_gamble/SPEC.md for the frozen pre-registration
this pipeline implements.
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
from research.wave25_gamble import engine25, gates25
from research.wave25_gamble.configs25 import (
    CANDIDATE_IDS,
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
)

CANDIDATE_DEFINITIONS: Final[dict[str, str]] = {
    "B0": "V1 재현(±2xATR 양방향 돌파, BTC 단일) -- 기준선, 엔진 재사용(재구현 아님)",
    "B1": "MACD(12,26,9) 히스토그램 부호 전환 진입, ATR 트레일링",
    "B2": "ADX(14)>25 AND +DI/-DI 교차 진입",
    "B3": "슈퍼트렌드(10, 3.0) 방향 전환",
    "B4": "켈트너 채널(20, 2xATR) 상/하단 이탈",
    "B5": "MTF 정합: 1D MA50 기울기 방향 AND 1H ATR모멘텀 돌파(고정랙, 돈치안 아님)",
    "B6": "스토캐스틱(14,3) 과매도/과매수 이탈 + MA50 기울기 추세필터",
    "B7": "앙상블: B1~B6 중 3개 이상 동일방향 동시 발화",
}


class Stage(StrEnum):
    FETCH = "fetch"
    RUN = "run"
    GATES = "gates"
    REPORT = "report"
    LIVE = "live"
    ALL = "all"


class Wave25CliError(Exception):
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


def _trade_payload(trade: engine25.Trade) -> dict[str, Any]:
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
        raise Wave25CliError("fetch: missing required cache files:\n  " + "\n  ".join(missing))
    print("fetch: all required caches present (wave1/wave6/wave13_liquidity/wave18_idle) -- no network fetch needed for the run/gates/report stages")


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def _stage_run(only: str | None) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    mapping = costs_measured.fit_mapping()
    print(f"run: measured-cost mapping fitted from {mapping.raw_point_count} Bitget points -> {len(mapping.anchor_bp)} isotonic buckets (worst={mapping.worst_bp:.4f}bp, best={mapping.best_bp:.4f}bp)")
    stable_equity, stable_info = engine25.load_stable_leg(STABLE_CAPITAL)
    print(f"run: stable leg loaded from {stable_info['source']} (I5 full_period_annualized={stable_info['source_full_period_annualized']:.4f})")

    for candidate_id in CANDIDATE_IDS:
        if only is not None and candidate_id != only:
            continue
        result = engine25.run_candidate(candidate_id, mapping=mapping, stress_multiplier=1.0)
        stressed_result = engine25.run_candidate(candidate_id, mapping=mapping, stress_multiplier=P5_STRESS_MULTIPLIER)
        stressed_final = float(stressed_result.equity.dropna().iloc[-1]) if len(stressed_result.equity.dropna()) else GAMBLE_CAPITAL
        combined = engine25.combine_portfolio(stable_equity, result.equity, GAMBLE_CAPITAL)
        payload = {
            "candidate_id": candidate_id,
            "family": "wave25_gamble",
            "definition": CANDIDATE_DEFINITIONS[candidate_id],
            "config": result.metadata.get("config", {}),
            "capital_contract": {"total_capital_usdt": TOTAL_CAPITAL, "gamble_capital_usdt": GAMBLE_CAPITAL, "stable_capital_usdt": STABLE_CAPITAL},
            "cost_model": "single-leg (maker 0.02% + wave13_liquidity.costs_measured measured-slippage mapping, unmodified) per entry/exit transition -- reused from research.wave20_convex.engine20",
            "stable_leg_source": stable_info,
            "gamble_equity": _series_payload(result.equity),
            "combined_equity": _series_payload(combined["total"]),
            "trades": [_trade_payload(t) for t in result.trades],
            "symbols_used": list(result.symbols_used),
            "metadata": {k: v for k, v in result.metadata.items() if k != "config"},
            "stress_test": {"stress_multiplier": P5_STRESS_MULTIPLIER, "final_gamble_equity_usdt": stressed_final, "n_trades": stressed_result.metadata.get("n_trades")},
        }
        _save_json(RESULTS_DIR / f"{candidate_id}.json", payload)
        final_gamble = float(result.equity.dropna().iloc[-1]) if len(result.equity.dropna()) else GAMBLE_CAPITAL
        final_combined = float(combined["total"].dropna().iloc[-1]) if len(combined["total"].dropna()) else TOTAL_CAPITAL
        print(f"run: {candidate_id} done (gamble_final=${final_gamble:.4f}, combined_final=${final_combined:.2f}, n_trades={len(result.trades)}, stressed_final=${stressed_final:.4f})")


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------


def _stage_gates(only: str | None) -> None:
    baseline_path = RESULTS_DIR / "B0.json"
    baseline_payload = _load_json(baseline_path)
    baseline_gamble_equity = _series_from_payload(baseline_payload["gamble_equity"])
    baseline_final_usdt = _final_from_series(baseline_payload["gamble_equity"], GAMBLE_CAPITAL)
    print(f"gates: B0 baseline sleeve final=${baseline_final_usdt:.4f} (n_trades={len(baseline_payload['trades'])})")

    for seed_offset, candidate_id in enumerate(CANDIDATE_IDS):
        if only is not None and candidate_id != only:
            continue
        path = RESULTS_DIR / f"{candidate_id}.json"
        payload = _load_json(path)
        gamble_equity = _series_from_payload(payload["gamble_equity"])
        combined_equity = _series_from_payload(payload["combined_equity"])
        trades_payload = payload["trades"]
        stressed_final = float(payload["stress_test"]["final_gamble_equity_usdt"])
        report = gates25.evaluate_candidate(
            candidate_id, gamble_equity, trades_payload, combined_equity, baseline_gamble_equity, baseline_final_usdt, stressed_final, seed_offset
        )
        payload["gates_report"] = report
        payload["full_period_cagr_combined"] = gates25.full_period_annualized(combined_equity)
        payload["full_period_cagr_gamble_only"] = gates25.full_period_annualized(gamble_equity)
        _save_json(path, payload)
        statuses = {g["gate_id"]: g["status"] for g in report["gates"]}
        print(f"gates: {candidate_id} -> {statuses} overall={report['overall']['status']} promoted={report['overall']['promoted']}")


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def _stage_report() -> None:
    from research.wave25_gamble.reporting25 import write_wave25_report

    live_path = RESULTS_DIR / "live_signals.json"
    live_payload = _load_json(live_path) if live_path.exists() else None
    write_wave25_report(RESULTS_DIR, REPORT_DIR, REGISTRY_PATH, live_payload)
    print(f"report: wrote {REPORT_DIR / 'wave25_report.md'} and {REGISTRY_PATH}")


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


def _b0_live_status(hourly_btc: pd.DataFrame, trades: tuple[engine25.Trade, ...], mapping, stress_multiplier: float = 1.0) -> dict[str, Any]:
    """See engine25._run_b0_with_hourly's own docstring for the resimulation this consumes.
    If the authoritative trade list's last trade is still open at the data's own last bar
    ("end_of_data"), report the held position directly; otherwise replay ONLY V1's flat-state
    anchor-arming rule (engine20.simulate_breakout_reversal's `direction==0` branch) starting
    from the bar the last trade exited -- correct by construction because a real breakout
    after that bar would already appear as a further trade in `trades` (see this function's
    call site for why no further replay of the in-position branch is needed)."""
    from research.wave20_convex import engine20 as wave20_engine
    from research.wave20_convex.configs20 import V1_CONFIG

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
        return {
            "status": "HOLDING",
            "direction": last.direction,
            "entry_price": last.entry_price,
            "entry_time": str(last.entry_time),
            "current_price": current_price,
            "unrealized_pct": unrealized_pct,
        }

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
    return {
        "status": "FLAT_ARMED",
        "anchor": anchor,
        "current_price": float(close[-1]),
        "atr_now": atr_now,
        "threshold": threshold,
        "distance_to_long_trigger": (anchor + threshold) - float(close[-1]) if threshold is not None else None,
        "distance_to_short_trigger": float(close[-1]) - (anchor - threshold) if threshold is not None else None,
    }


def _signal_candidate_live_status(candidate_id: str, symbol: str, hourly: pd.DataFrame, daily: pd.DataFrame, result: engine25.GambleResult) -> dict[str, Any]:
    symbol_trades = [t for t in result.trades if t.symbol == symbol]
    close = hourly["close"].to_numpy(dtype=float)
    if symbol_trades and symbol_trades[-1].exit_time == hourly.index[-1] and symbol_trades[-1].exit_reason == "end_of_data":
        last = symbol_trades[-1]
        current_price = float(close[-1])
        unrealized_pct = (current_price / last.entry_price - 1.0) * last.direction if last.entry_price else None
        return {"status": "HOLDING", "direction": last.direction, "entry_price": last.entry_price, "entry_time": str(last.entry_time), "current_price": current_price, "unrealized_pct": unrealized_pct}

    signal_fns = {
        "B1": lambda: engine25.macd_signal(hourly),
        "B2": lambda: engine25.adx_dmi_signal(hourly),
        "B3": lambda: engine25.supertrend_signal(hourly),
        "B4": lambda: engine25.keltner_signal(hourly),
        "B5": lambda: engine25.mtf_confluence_signal(hourly, daily),
        "B6": lambda: engine25.stochastic_signal(hourly),
        "B7": lambda: engine25.ensemble_signal_for_symbol(hourly, daily),
    }
    raw_signal = signal_fns[candidate_id]()
    last_signal = float(raw_signal.iloc[-1]) if len(raw_signal) else 0.0
    if last_signal > 0.0:
        return {"status": "FRESH_LONG_SIGNAL", "current_price": float(close[-1]), "note": "fires at the next available bar's open (t->t+1 fill discipline)"}
    if last_signal < 0.0:
        return {"status": "FRESH_SHORT_SIGNAL", "current_price": float(close[-1]), "note": "fires at the next available bar's open (t->t+1 fill discipline)"}
    return {"status": "FLAT_NO_SIGNAL", "current_price": float(close[-1])}


def _stage_live() -> None:
    LIVE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    mapping = costs_measured.fit_mapping()
    network_used = False
    live_hourly: dict[str, pd.DataFrame] = {}
    for symbol in SYMBOLS:
        cached = engine25.load_hourly(symbol)
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

    b0_result = engine25._run_b0_with_hourly(live_hourly["BTCUSDT"], mapping, stress_multiplier=1.0)
    signals_payload["candidates"]["B0"] = {"BTCUSDT": _b0_live_status(live_hourly["BTCUSDT"], b0_result.trades, mapping)}

    for candidate_id in ("B1", "B2", "B3", "B4", "B5", "B6", "B7"):
        runner = engine25.RUNNERS[candidate_id]
        result = runner(mapping=mapping, stress_multiplier=1.0, hourly_frames=live_hourly, daily_frames=live_daily)
        per_symbol = {symbol: _signal_candidate_live_status(candidate_id, symbol, live_hourly[symbol], live_daily[symbol], result) for symbol in SYMBOLS}
        signals_payload["candidates"][candidate_id] = per_symbol
        actionable = [f"{s}:{v['status']}" for s, v in per_symbol.items() if v["status"] != "FLAT_NO_SIGNAL"]
        print(f"live: {candidate_id} -> " + (", ".join(actionable) if actionable else "no symbol actionable right now"))

    _save_json(RESULTS_DIR / "live_signals.json", signals_payload)
    print(f"live: wrote {RESULTS_DIR / 'live_signals.json'} (network_used={network_used})")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Wave-25 gamble tournament (B0-B7) pipeline")
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
    except (FileNotFoundError, Wave25CliError, engine25.Wave25Error, RuntimeError, ValueError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
