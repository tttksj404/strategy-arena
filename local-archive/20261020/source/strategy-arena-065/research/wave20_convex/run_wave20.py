#!/usr/bin/env python
"""Wave-20 (convex-gamble, V1-V5) pipeline CLI. Mirrors research/wave18_idle/run_wave18.py's
own --stage convention: `fetch` here is a cache-completeness VERIFICATION only (network is
permitted for this task, but every symbol/timeframe V1-V5 need is already cached -- see
configs20.py's module docstring for the inventory); `run`/`gates`/`report` are cache-only and
deterministic. See research/wave20_convex/SPEC.md for the frozen pre-registration this
pipeline implements.
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

import pandas as pd  # noqa: PANDAS_OK

from research.wave13_liquidity import costs_measured
from research.wave20_convex import dataio20, engine20, gates20
from research.wave20_convex.configs20 import (
    CANDIDATE_IDS,
    GAMBLE_CAPITAL,
    REGISTRY_PATH,
    REPORT_DIR,
    RESULTS_DIR,
    STABLE_CAPITAL,
    TOTAL_CAPITAL,
    V1_CONFIG,
    V2_CONFIG,
    V3_CONFIG,
    V4_CONFIG,
    V5_CONFIG,
    WAVE1_CACHE_DIR,
    WAVE3_CACHE_DIR,
    WAVE6_CACHE_DIR,
)

CANDIDATE_DEFINITIONS: Final[dict[str, str]] = {
    "V1": "양방향 돌파 추격(롱 변동성 스트래들 근사) -- BTC perp, 저변동 20d<30분위 구간 진입, ±2xATR(일봉) 돌파/반전",
    "V2": "꼬리 사냥 -- 펀딩 극단(>연100%) 코인 방향 롱퍼프(캐리 아님), top1 로테이션",
    "V3": "신규상장 첫 7일 -- 상장 D+0 롱 진입, 확장 ATR ±2x 반전, D+7 청산, 단일슬롯 비중첩",
    "V4": "청산 캐스케이드 반등(대칭 대조군) -- BTC/ETH/SOL 1H -8% 급락 후 롱, +3%/-3%, 24h 최대보유",
    "V5": "복권 바스켓 -- 저가·고변동 알트 5종 point-in-time 매월 재선정, 균등비중, 30일 보유",
}


class Stage(StrEnum):
    FETCH = "fetch"
    RUN = "run"
    GATES = "gates"
    REPORT = "report"
    ALL = "all"


class Wave20Error(Exception):
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


def _trade_payload(trade: engine20.Trade) -> dict[str, Any]:
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


def _candidate_config(candidate_id: str) -> Any:
    return {"V1": V1_CONFIG, "V2": V2_CONFIG, "V3": V3_CONFIG, "V4": V4_CONFIG, "V5": V5_CONFIG}[candidate_id]


def _stage_fetch() -> None:
    """No network calls: every candidate reads from caches other waves already populated.
    This stage only verifies those caches are actually present, so a clean checkout fails
    loudly and early instead of deep inside a multi-minute `run`."""
    missing: list[str] = []
    if not (WAVE6_CACHE_DIR / "binance_fapi_BTCUSDT_1h.csv.gz").exists():
        missing.append(str(WAVE6_CACHE_DIR / "binance_fapi_BTCUSDT_1h.csv.gz"))
    for symbol in V4_CONFIG.symbols:
        path = WAVE6_CACHE_DIR / f"binance_fapi_{symbol}_1h.csv.gz"
        if not path.exists():
            missing.append(str(path))
    if not dataio20.wave1_symbols_with_funding():
        missing.append(f"{WAVE1_CACHE_DIR} (no symbol has both price+funding)")
    if len(dataio20.wave3_symbols()) < 50:
        missing.append(f"{WAVE3_CACHE_DIR} (fewer than 50 symbols)")
    from research.wave13_liquidity.collect_spreads import CACHE_DIR as W13_CACHE_DIR

    if not (W13_CACHE_DIR / "measured_spreads.json").exists():
        missing.append(str(W13_CACHE_DIR / "measured_spreads.json"))
    if missing:
        raise Wave20Error("fetch: missing required cache files:\n  " + "\n  ".join(missing))
    print("fetch: all required caches present (wave1/wave3/wave6/wave13_liquidity) -- no network fetch needed")


def _stage_run(only: str | None) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    mapping = costs_measured.fit_mapping()
    print(
        f"run: measured-cost mapping fitted from {mapping.raw_point_count} Bitget points "
        f"-> {len(mapping.anchor_bp)} isotonic buckets (worst={mapping.worst_bp:.4f}bp, best={mapping.best_bp:.4f}bp)"
    )
    stable_equity, stable_info = engine20.load_stable_leg(STABLE_CAPITAL)
    print(f"run: stable leg loaded from {stable_info['source']} (I5 full_period_annualized={stable_info['source_full_period_annualized']:.4f})")

    for candidate_id in CANDIDATE_IDS:
        if only is not None and candidate_id != only:
            continue
        result = engine20.run_candidate(candidate_id, mapping=mapping)
        combined = engine20.combine_portfolio(stable_equity, result.equity, GAMBLE_CAPITAL)
        payload = {
            "candidate_id": candidate_id,
            "family": "wave20_convex",
            "definition": CANDIDATE_DEFINITIONS[candidate_id],
            "config": result.metadata.get("config", {}),
            "capital_contract": {
                "total_capital_usdt": TOTAL_CAPITAL,
                "gamble_capital_usdt": GAMBLE_CAPITAL,
                "stable_capital_usdt": STABLE_CAPITAL,
            },
            "cost_model": (
                "single-leg (maker 0.02% + wave13_liquidity.costs_measured measured-slippage mapping, unmodified) "
                "per entry/exit transition -- see engine20.py module docstring for why single-leg, not wave13's 2-leg carry-pair formula"
            ),
            "stable_leg_source": stable_info,
            "gamble_equity": _series_payload(result.equity),
            "combined_equity": _series_payload(combined["total"]),
            "trades": [_trade_payload(t) for t in result.trades],
            "symbols_used": list(result.symbols_used),
            "metadata": {k: v for k, v in result.metadata.items() if k != "config"},
        }
        _save_json(RESULTS_DIR / f"{candidate_id}.json", payload)
        final_gamble = float(result.equity.dropna().iloc[-1]) if len(result.equity.dropna()) else GAMBLE_CAPITAL
        final_combined = float(combined["total"].dropna().iloc[-1]) if len(combined["total"].dropna()) else TOTAL_CAPITAL
        print(
            f"run: {candidate_id} done (gamble_final=${final_gamble:.4f}, combined_final=${final_combined:.2f}, "
            f"n_trades={len(result.trades)}, total_cost=${result.metadata.get('total_cost_usdt', 0.0):.4f})"
        )


def _stage_gates(only: str | None) -> None:
    stable_equity, stable_info = engine20.load_stable_leg(STABLE_CAPITAL)
    stable_solo_cagr = gates20.full_period_annualized(stable_equity)
    if stable_solo_cagr is None:
        raise Wave20Error("gates: could not compute I5-solo CAGR from the stable leg")
    print(f"gates: I5-solo reference full_period_cagr={stable_solo_cagr:.4f} (source: {stable_info['source']})")

    for seed_offset, candidate_id in enumerate(CANDIDATE_IDS):
        if only is not None and candidate_id != only:
            continue
        path = RESULTS_DIR / f"{candidate_id}.json"
        payload = _load_json(path)
        gamble_equity = _series_from_payload(payload["gamble_equity"])
        combined_equity = _series_from_payload(payload["combined_equity"])
        trades_payload = payload["trades"]
        report = gates20.evaluate_candidate(
            candidate_id, gamble_equity, trades_payload, combined_equity, stable_equity, stable_solo_cagr, seed_offset
        )
        payload["gates_report"] = report
        payload["full_period_cagr_combined"] = gates20.full_period_annualized(combined_equity)
        payload["full_period_cagr_gamble_only"] = gates20.full_period_annualized(gamble_equity)
        _save_json(path, payload)
        statuses = {g["gate_id"]: g["status"] for g in report["gates"]}
        print(f"gates: {candidate_id} -> {statuses} overall={report['overall']['status']} promoted={report['overall']['promoted']}")


def _stage_report() -> None:
    from research.wave20_convex.reporting20 import write_wave20_report

    write_wave20_report(RESULTS_DIR, REPORT_DIR, REGISTRY_PATH)
    print(f"report: wrote {REPORT_DIR / 'wave20_report.md'} and {REGISTRY_PATH}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Wave-20 convex-gamble (V1-V5) pipeline")
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
            case Stage.ALL:
                _stage_fetch()
                _stage_run(args.only)
                _stage_gates(args.only)
                _stage_report()
            case unreachable:
                assert_never(unreachable)
    except (FileNotFoundError, Wave20Error, engine20.Wave20Error, RuntimeError, ValueError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
