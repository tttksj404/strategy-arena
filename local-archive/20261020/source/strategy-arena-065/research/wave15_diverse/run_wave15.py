#!/usr/bin/env python
"""Wave-15 (method diversification: 4 profit-mechanism families, 7 pre-registered
candidates) pipeline CLI. Mirrors research/wave13_liquidity/run_wave13.py's --stage
convention. No `collect` stage: every daily candidate's OHLCV/funding is already cached
(borrowed read-only from wave12_frontier/wave11_yield/wave1, exactly like wave13 does), and
A1-A3's 1h OHLCV is already cached in wave6/wave11_yield -- the only new "collection" this
wave attempted (B1's Simple Earn Flexible historical APR) is a probe, not a fetch (see
earn_apr.py); it is recorded, not re-run, on every `run`.
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

from research.wave1.gate_reporting import _series
from research.wave10_carry100.engine import ACTIVE_CAPITAL, MIN_ORDER_USDT, RESERVE_FRACTION, TOTAL_CAPITAL, Wave10Result
from research.wave10_carry100.regime import regime_breakdown
from research.wave15_diverse import common15, configs15, engine_daily, engine_intraday, engine_pairs, gates15, signals15
from research.wave15_diverse.earn_apr import resolve_flexible_earn_apr
from research.wave15_diverse.reporting15 import write_wave15_report

BASE_DIR: Final = Path(__file__).resolve().parent
RESULTS_DIR: Final = BASE_DIR / "results"
REPORT_DIR: Final = BASE_DIR / "report"
CACHE_DIR: Final = BASE_DIR / "cache"
REGISTRY_PATH: Final = BASE_DIR / "REGISTRY.md"

A_IDS: Final = ("A1", "A2", "A3")
DAILY_MECHANISM_IDS: Final = ("B1", "B2", "C1")
D_IDS: Final = ("D1",)


class Stage(StrEnum):
    RUN = "run"
    GATES = "gates"
    REPORT = "report"
    ALL = "all"


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
    return [{"timestamp": str(timestamp), "value": float(value)} for timestamp, value in series.items()]


def _result_payload_block(prefix: str, result: Wave10Result) -> dict[str, Any]:
    return {
        f"{prefix}equity": _series_payload(result.equity),
        f"{prefix}positions": _series_payload(result.positions),
        f"{prefix}turnover": _series_payload(result.turnover),
        f"{prefix}trade_returns": _series_payload(result.trade_returns),
    }


def _result_from_payload(payload: dict[str, Any], prefix: str, symbols_used: tuple[str, ...] = ()) -> Wave10Result:
    return Wave10Result(
        equity=_series(payload[f"{prefix}equity"]),
        positions=_series(payload[f"{prefix}positions"]),
        turnover=_series(payload[f"{prefix}turnover"]),
        trade_returns=_series(payload[f"{prefix}trade_returns"]),
        max_concurrent_positions=int(payload.get("metadata", {}).get("max_concurrent_positions", 0)),
        symbols_used=symbols_used,
    )


# ---------------------------------------------------------------------------
# L4 reference recompute -- READS research/wave13_liquidity (code + cache) only, WRITES only
# into research/wave15_diverse/cache/l4_reference.json. Never touches wave13's own results/
# or report/ directories (SPEC.md: "research/wave15_diverse/ 밖 수정 금지").
# ---------------------------------------------------------------------------


def _run_l4_reference() -> dict[str, Any]:
    from research.wave13_liquidity import costs_measured as cm13
    from research.wave13_liquidity import engine13
    from research.wave13_liquidity.configs13 import get_config

    print("run: recomputing L4 reference (in-memory, read-only wave13 engine13.run_candidate)...")
    mapping = cm13.fit_mapping()
    config = get_config("L4")
    result, total_cost, eligible = engine13.run_candidate(config, mapping, stress_multiplier=engine13.DEFAULT_STRESS_MULTIPLIER)
    regime = regime_breakdown(result)
    payload = {
        "source": "in-memory recompute of research.wave13_liquidity.engine13.run_candidate(get_config('L4')), read-only",
        "cost_model": "identical costs_measured mapping wave15's own candidates use (fit fresh from the same cache/measured_spreads.json)",
        "regime_breakdown": regime,
        "high_funding_mean_annualized_return": regime.get("high_funding_mean_annualized_return"),
        "total_cost_usdt": total_cost,
        "n_trades": len(result.trade_returns),
        "universe_size": len(result.symbols_used),
        "span": {"start": str(result.equity.index.min()), "end": str(result.equity.index.max())},
    }
    _save_json(CACHE_DIR / "l4_reference.json", payload)
    return payload


# ---------------------------------------------------------------------------
# A1-A3
# ---------------------------------------------------------------------------


def _a_series_breakeven_table(quote_volume_frame: pd.DataFrame, mapping) -> dict[str, Any]:
    """SPEC.md: '손익분기를 리포트에 수식으로 명시'. Breakeven condition per intraday cycle:
    funding_rate_realized_this_period > 2 x cost_for(symbol) (round trip = open + close, each
    a both-legs cost_for() charge) -- see report/wave15_report.md's derivation. Uses each
    symbol's MOST RECENT point-in-time-known trailing-30d volume (i.e. "what the mapping
    would price today"), for a concrete, current numeric illustration -- the actual backtest
    itself uses the full historical, point-in-time-varying cost_rate_frame, not this snapshot."""
    from research.wave13_liquidity import costs_measured

    known_avg = costs_measured.rolling_trailing_avg_volume(quote_volume_frame).ffill()
    per_symbol: dict[str, Any] = {}
    for symbol in quote_volume_frame.columns:
        volume = float(known_avg[symbol].iloc[-1]) if len(known_avg) and pd.notna(known_avg[symbol].iloc[-1]) else None
        bp = costs_measured.slippage_bp_for_volume(volume, mapping)
        one_way_both_legs = float(common15.two_leg_cost_rate_from_bp(bp))
        round_trip_cost = 2.0 * one_way_both_legs
        breakeven_apr = round_trip_cost * 3.0 * 365.0
        per_symbol[symbol] = {
            "trailing_30d_volume_usdt": volume,
            "measured_slippage_bp": bp,
            "one_way_both_legs_cost_rate": one_way_both_legs,
            "round_trip_cost_per_cycle": round_trip_cost,
            "breakeven_period_funding_rate": round_trip_cost,
            "breakeven_annualized_apr": breakeven_apr,
        }
    return {
        "formula": "breakeven: funding_rate_per_8h_period > 2 * cost_for(symbol) ; annualized: 2*cost_for(symbol)*3*365",
        "per_symbol": per_symbol,
    }


def _run_a_series(only: str | None) -> None:
    ids = tuple(cid for cid in A_IDS if only is None or cid == only)
    if not ids:
        return
    print(f"run: building A-series hourly frames ({', '.join(engine_intraday.A_SERIES_SYMBOLS)}, 1h)...")
    frames = engine_intraday.build_hourly_frames()
    quote_volume = common15.load_quote_volume_frame(engine_intraday.A_SERIES_SYMBOLS)
    mapping = common15.fit_measured_cost_mapping()
    cost_base = engine_intraday.build_hourly_cost_rate_frame(quote_volume, engine_intraday.A_SERIES_SYMBOLS, frames.index, mapping, common15.BASE_STRESS_MULTIPLIER)
    cost_stress = engine_intraday.build_hourly_cost_rate_frame(quote_volume, engine_intraday.A_SERIES_SYMBOLS, frames.index, mapping, common15.STRESS_MULTIPLIER)
    print(f"run: A-series frames ready ({len(frames.index)} hourly bars, {frames.index.min()} -> {frames.index.max()})")

    for candidate_id in ids:
        config = configs15.A_SERIES_CONFIGS[candidate_id]
        print(f"run: {candidate_id} starting (fast_entry={config.fast_entry_threshold}, daily_entry={config.daily_entry_threshold})...")
        result, total_cost, diag = engine_intraday.run_intraday_carry(frames, config, cost_base)
        stress_result, stress_total_cost, stress_diag = engine_intraday.run_intraday_carry(frames, config, cost_stress)

        breakeven = _a_series_breakeven_table(quote_volume, mapping)
        payload: dict[str, Any] = {
            "candidate_id": candidate_id,
            "family": "wave15_diverse",
            "mechanism_family": "A_intraday_carry",
            "definition": {
                "A1": "정산 T-1h 진입 -> T+1h 청산, 대상 직전 8h 펀딩 실현값 상위(임계 없음), 1쌍 $45/$45",
                "A2": "A1 + 임계필터(직전 펀딩>0.03%=연33%만 진입)",
                "A3": "하이브리드: 7d APR>15%면 일봉보유 유지, 아니면 A2 인트라데이 (두 모드 전환)",
            }[candidate_id],
            "universe": list(engine_intraday.A_SERIES_SYMBOLS),
            "universe_note": (
                "1h OHLCV 캐시가 존재하는 메이저 중 정산주기가 전체 히스토리에서 예외 없이 8h로 "
                "고정된 심볼만 사용(BTC/ETH). SOLUSDT는 1h 캐시가 있으나 실측 결과 2022-11 이후 "
                "구간에 2h 간격 정산이 섞여 있어(변동성 확대 시 거래소의 동적 정산주기 단축) 제외 "
                "-- wave13 L1과 동일하게 BTC/ETH 고정 2종 스코프."
            ),
            "structure": {"delta_neutral_by_construction": True, "note": "spot_long+perp_short 2레그, 기존 캐리와 동일 구조, 보유 타이밍만 변경"},
            "config": {
                "fast_entry_threshold": config.fast_entry_threshold,
                "daily_entry_threshold": config.daily_entry_threshold,
                "daily_exit_threshold": config.daily_exit_threshold,
                "leg_fraction_of_active_capital": common15.LEG_FRACTION,
                "settlement_hours_utc": sorted(engine_intraday.SETTLEMENT_HOURS),
                "decision_hours_utc": sorted(engine_intraday.DECISION_HOURS),
            },
            "capital_contract": {
                "total_capital_usdt": TOTAL_CAPITAL,
                "reserve_fraction": RESERVE_FRACTION,
                "active_capital_usdt": ACTIVE_CAPITAL,
                "min_order_usdt": MIN_ORDER_USDT,
                "leg_usdt_nominal": common15.leg_usdt(),
                "gross_usdt_nominal": common15.gross_usdt(),
            },
            "cost_model": "bitget_measured_volume_mapping(wave13 costs_measured, isotonic fit)+maker_0.02pct_per_leg, both legs -- broadcast from daily to hourly (see engine_intraday.build_hourly_cost_rate_frame)",
            **_result_payload_block("", result),
            **_result_payload_block("stress_", stress_result),
            "metadata": {
                "symbols_used": list(result.symbols_used),
                "max_concurrent_positions": result.max_concurrent_positions,
                "n_trades": len(result.trade_returns),
                "total_cost_usdt": total_cost,
                "avg_cost_per_trade_usdt": (total_cost / len(result.trade_returns)) if len(result.trade_returns) else 0.0,
                "stress_total_cost_usdt": stress_total_cost,
                "stress_n_trades": len(stress_result.trade_returns),
                "n_daily_entries": diag["n_daily_entries"],
                "n_intraday_entries": diag["n_intraday_entries"],
                "daily_bar_fraction": diag["daily_bar_fraction"],
                "intraday_bar_fraction": diag["intraday_bar_fraction"],
                "flat_bar_fraction": diag["flat_bar_fraction"],
                "state_machine_invariant_violations": diag["state_machine_invariant_violations"],
                "annualized_round_trips": gates15.annualized_round_trips(result),
                "utilization": gates15.utilization(result),
                "bar_frequency": "1h",
                "source_engine": "research.wave15_diverse.engine_intraday.run_intraday_carry",
            },
            "breakeven_analysis": breakeven,
        }
        _save_json(RESULTS_DIR / f"{candidate_id}.json", payload)
        final_equity = float(result.equity.iloc[-1]) if len(result.equity) else float("nan")
        print(
            f"run: {candidate_id} done (trades={len(result.trade_returns)}, final_active_equity=${final_equity:.4f}, "
            f"total_cost=${total_cost:.2f}, intraday_bar_frac={diag['intraday_bar_fraction']:.3f})"
        )


# ---------------------------------------------------------------------------
# B1/B2/C1 -- share ONE universe (L4's own breadth=200/12mo rule).
# ---------------------------------------------------------------------------


def _shared_daily_universe() -> tuple[tuple[str, ...], dict, Any]:
    from research.wave12_frontier import universe_frontier as uf12

    pool = common15.load_candidate_pool()
    symbols = uf12.symbols_for_breadth_history(pool, configs15.SHARED_DAILY_BREADTH, configs15.SHARED_DAILY_HISTORY_MONTHS)
    return symbols, pool, uf12


def _run_daily_mechanism_candidates(only: str | None) -> None:
    ids = tuple(cid for cid in DAILY_MECHANISM_IDS if only is None or cid == only)
    if not ids:
        return
    print("run: resolving shared B1/B2/C1 universe (== L4's own top200/12mo rule)...")
    symbols, pool, uf12 = _shared_daily_universe()
    markets = common15.load_markets(symbols)
    price_frames = common15.build_price_frames(markets)
    index = price_frames["spot_open"].index
    quote_volume = common15.load_quote_volume_frame(symbols)
    mapping = common15.fit_measured_cost_mapping()
    # SAME data-availability mask research.wave13_liquidity.costs_measured.build_data_availability_mask
    # applies to L1-L4 (True iff a point-in-time trailing-30d volume is even KNOWN yet) --
    # required for B1/B2/C1's "carry-only" path to be a genuine apples-to-apples reproduction
    # of L4 under an identical cost model, not merely the same signal on a looser eligibility
    # set. Omitting this was caught by comparing an early B1 carry-only run (20.97% high-
    # funding annualized) against the L4 reference (22.01%) on paper: SAME universe/signal
    # should have landed much closer than a full percentage point apart.
    liquidity_mask = common15.build_liquidity_mask(quote_volume, symbols).reindex(index=index, columns=list(symbols)).fillna(False)
    print(f"run: shared universe ready ({len(symbols)} symbols, {len(index)} daily bars)")

    earn_apr, earn_verified, _probe = resolve_flexible_earn_apr(force_probe=False)

    for candidate_id in ids:
        config = configs15.DAILY_CANDIDATE_CONFIGS[candidate_id]
        single_leg = config.structure == "perp_only_short"
        cost_base = common15.build_cost_rate_frame(quote_volume, symbols, mapping, common15.BASE_STRESS_MULTIPLIER, single_leg=single_leg, index=index)
        cost_stress = common15.build_cost_rate_frame(quote_volume, symbols, mapping, common15.STRESS_MULTIPLIER, single_leg=single_leg, index=index)

        if candidate_id == "C1":
            active_frame, score_frame = signals15.build_predictive_signal_frames(markets, index)
        else:
            active_frame, score_frame = signals15.build_realized_funding_signal(markets, index)
        active_frame = active_frame.where(liquidity_mask, 0.0)

        extra_yield = common15.ASSUMED_FLEXIBLE_EARN_APR if candidate_id in ("B1", "B2") else 0.0
        print(f"run: {candidate_id} starting (structure={config.structure}, extra_yield={extra_yield})...")

        result, total_cost = engine_daily.run_generic_carry(price_frames, active_frame, score_frame, common15.TOP_K, common15.LEG_FRACTION, cost_base, config.structure, extra_yield)
        stress_result, stress_total_cost = engine_daily.run_generic_carry(price_frames, active_frame, score_frame, common15.TOP_K, common15.LEG_FRACTION, cost_stress, config.structure, extra_yield)

        payload: dict[str, Any] = {
            "candidate_id": candidate_id,
            "family": "wave15_diverse",
            "mechanism_family": {"B1": "B_dual_yield", "B2": "B_dual_yield", "C1": "C_predictive"}[candidate_id],
            "definition": config.note,
            "universe": {"kind": "breadth", "breadth": config.breadth, "history_months": config.history_months, "size": len(symbols)},
            "structure": {
                "delta_neutral_by_construction": config.structure == "spot_perp",
                "note": "spot_long+perp_short 2레그, 델타중립" if config.structure == "spot_perp" else "현물 레그 없음 -- USDT담보+숏퍼프 단독, 방향노출 있음(헤지 없음)",
            },
            "config": {
                "window_days": signals15.BASELINE_WINDOW_DAYS,
                "entry_threshold_apr": common15.ENTRY_THRESHOLD_APR if candidate_id != "C1" else None,
                "exit_threshold_apr": common15.EXIT_THRESHOLD_APR,
                "predictive_entry_z": signals15.C1_ENTRY_Z if candidate_id == "C1" else None,
                "predictive_weights": {"momentum": signals15.C1_WEIGHT_MOMENTUM, "funding_trend": signals15.C1_WEIGHT_FUNDING_TREND} if candidate_id == "C1" else None,
                "top_k_pairs": common15.TOP_K,
                "leg_fraction_of_active_capital": common15.LEG_FRACTION,
                "assumed_extra_annual_yield": extra_yield if extra_yield else None,
                "extra_yield_verified": earn_verified if extra_yield else None,
            },
            "capital_contract": {
                "total_capital_usdt": TOTAL_CAPITAL,
                "reserve_fraction": RESERVE_FRACTION,
                "active_capital_usdt": ACTIVE_CAPITAL,
                "min_order_usdt": MIN_ORDER_USDT,
                "leg_usdt_nominal": common15.leg_usdt(),
                "gross_usdt_nominal": common15.gross_usdt(),
            },
            "cost_model": (
                "bitget_measured_volume_mapping(wave13 costs_measured)+maker_0.02pct, SINGLE leg only (no spot leg)"
                if single_leg
                else "bitget_measured_volume_mapping(wave13 costs_measured)+maker_0.02pct_per_leg, both legs"
            ),
            **_result_payload_block("", result),
            **_result_payload_block("stress_", stress_result),
            "metadata": {
                "symbols_used": list(result.symbols_used),
                "max_concurrent_positions": result.max_concurrent_positions,
                "n_trades": len(result.trade_returns),
                "total_cost_usdt": total_cost,
                "avg_cost_per_trade_usdt": (total_cost / len(result.trade_returns)) if len(result.trade_returns) else 0.0,
                "stress_total_cost_usdt": stress_total_cost,
                "stress_n_trades": len(stress_result.trade_returns),
                "utilization": gates15.utilization(result),
                "annualized_round_trips": gates15.annualized_round_trips(result),
                "source_engine": "research.wave15_diverse.engine_daily.run_generic_carry",
            },
        }

        if candidate_id in ("B1", "B2"):
            carry_only_result, carry_only_cost = engine_daily.run_generic_carry(price_frames, active_frame, score_frame, common15.TOP_K, common15.LEG_FRACTION, cost_base, config.structure, 0.0)
            payload["carry_only"] = {
                **_result_payload_block("", carry_only_result),
                "regime_breakdown": regime_breakdown(carry_only_result),
                "total_cost_usdt": carry_only_cost,
            }
            combined_regime = regime_breakdown(result)
            carry_only_regime = payload["carry_only"]["regime_breakdown"]
            payload["yield_attribution"] = {
                "carry_only_high_funding_annualized": carry_only_regime.get("high_funding_mean_annualized_return"),
                "carry_plus_assumed_yield_high_funding_annualized": combined_regime.get("high_funding_mean_annualized_return"),
                "assumed_yield_contribution_annualized_pts": (
                    (combined_regime.get("high_funding_mean_annualized_return") or 0.0) - (carry_only_regime.get("high_funding_mean_annualized_return") or 0.0)
                ),
                "note": "assumed_yield_contribution은 검증된 수익이 아니라 ASSUMED_FLEXIBLE_EARN_APR 가정에서 나온 값 -- earn_apr.py 참조.",
            }

        _save_json(RESULTS_DIR / f"{candidate_id}.json", payload)
        final_equity = float(result.equity.iloc[-1]) if len(result.equity) else float("nan")
        print(f"run: {candidate_id} done (trades={len(result.trade_returns)}, final_active_equity=${final_equity:.2f}, total_cost=${total_cost:.2f})")


# ---------------------------------------------------------------------------
# D1
# ---------------------------------------------------------------------------


def _run_d1(only: str | None) -> None:
    if only is not None and only != "D1":
        return
    print("run: D1 resolving sector pairs...")
    pool = common15.load_candidate_pool()
    pairs = engine_pairs.select_sector_pairs(pool)
    all_symbols = tuple(sorted({symbol for pair in pairs for symbol in (pair.symbol_a, pair.symbol_b)}))
    markets = common15.load_markets(all_symbols)
    quote_volume = common15.load_quote_volume_frame(all_symbols)
    mapping = common15.fit_measured_cost_mapping()

    frames_base = {pair.pair_id: engine_pairs.build_pair_frames(pair, markets, quote_volume, mapping, common15.BASE_STRESS_MULTIPLIER) for pair in pairs}
    frames_stress = {pair.pair_id: engine_pairs.build_pair_frames(pair, markets, quote_volume, mapping, common15.STRESS_MULTIPLIER) for pair in pairs}

    result, total_cost = engine_pairs.run_sector_pairs(pairs, frames_base)
    stress_result, stress_total_cost = engine_pairs.run_sector_pairs(pairs, frames_stress)

    payload: dict[str, Any] = {
        "candidate_id": "D1",
        "family": "wave15_diverse",
        "mechanism_family": "D_sector_pairs",
        "definition": "섹터 내 페어 회귀: 동일 섹터 상위 2종의 30d 로그스프레드 |z|>2 진입, |z|<0.5 청산, 방향은 진입시점 z부호로 고정. 양쪽 퍼프, 델타중립(달러중립 페어).",
        "sectors": [
            {"sector": pair.sector, "symbol_a": pair.symbol_a, "symbol_b": pair.symbol_b, "pair_id": pair.pair_id}
            for pair in pairs
        ],
        "meme_substitution_note": engine_pairs.MEME_SUBSTITUTION_NOTE,
        "deviation_note": (
            "SPEC.md 문면은 'z>2 진입/z<0.5 청산'(단방향)이나, 어느 심볼을 '먼저' 라벨링하느냐는 임의적이므로 "
            "대칭 |z| 규칙(및 진입시점 z부호로 고정한 방향)으로 구현 -- 메커니즘/임계값 동일, 부호 자의성만 제거."
        ),
        "structure": {
            "delta_neutral_by_construction": True,
            "note": "양쪽 퍼프 달러중립 페어 -- spot-perp 베이시스 델타중립과는 다른 리스크(두 심볼 간 잔차 상관/베이시스 리스크).",
        },
        "config": {
            "zscore_window_days": engine_pairs.ZSCORE_WINDOW_DAYS,
            "entry_z": engine_pairs.ENTRY_Z,
            "exit_z": engine_pairs.EXIT_Z,
            "top_k_pairs": common15.TOP_K,
            "leg_fraction_of_active_capital": common15.LEG_FRACTION,
        },
        "capital_contract": {
            "total_capital_usdt": TOTAL_CAPITAL,
            "reserve_fraction": RESERVE_FRACTION,
            "active_capital_usdt": ACTIVE_CAPITAL,
            "min_order_usdt": MIN_ORDER_USDT,
            "leg_usdt_nominal": common15.leg_usdt(),
            "gross_usdt_nominal": common15.gross_usdt(),
        },
        "cost_model": "bitget_measured_volume_mapping(wave13 costs_measured)+maker_0.02pct_per_leg, both legs (each leg's OWN measured bp, not assumed-equal)",
        **_result_payload_block("", result),
        **_result_payload_block("stress_", stress_result),
        "metadata": {
            "symbols_used": list(result.symbols_used),
            "max_concurrent_positions": result.max_concurrent_positions,
            "n_trades": len(result.trade_returns),
            "total_cost_usdt": total_cost,
            "avg_cost_per_trade_usdt": (total_cost / len(result.trade_returns)) if len(result.trade_returns) else 0.0,
            "stress_total_cost_usdt": stress_total_cost,
            "stress_n_trades": len(stress_result.trade_returns),
            "utilization": gates15.utilization(result),
            "annualized_round_trips": gates15.annualized_round_trips(result),
            "source_engine": "research.wave15_diverse.engine_pairs.run_sector_pairs",
        },
    }
    _save_json(RESULTS_DIR / "D1.json", payload)
    final_equity = float(result.equity.iloc[-1]) if len(result.equity) else float("nan")
    print(f"run: D1 done (trades={len(result.trade_returns)}, final_active_equity=${final_equity:.2f}, total_cost=${total_cost:.2f})")


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------


def _evaluate_and_save(candidate_id: str, seed_offset: int) -> gates15.GateReport:
    path = RESULTS_DIR / f"{candidate_id}.json"
    payload = _load_json(path)
    symbols_used = tuple(payload.get("metadata", {}).get("symbols_used", ()))
    result = _result_from_payload(payload, "", symbols_used)
    stress_result = _result_from_payload(payload, "stress_", symbols_used)

    leg_usdt = payload["capital_contract"]["leg_usdt_nominal"]
    gross_usdt = payload["capital_contract"]["gross_usdt_nominal"]
    delta_neutral = bool(payload["structure"]["delta_neutral_by_construction"])
    structure_note = str(payload["structure"]["note"])
    resample_daily = payload.get("metadata", {}).get("bar_frequency") == "1h"

    report = gates15.evaluate_gates(result, stress_result, leg_usdt, gross_usdt, delta_neutral, structure_note, seed_offset, resample_daily)
    payload["gates"] = gates15.gate_report_payload(report)
    payload["regime_breakdown"] = regime_breakdown(result)
    payload["stress_regime_breakdown"] = regime_breakdown(stress_result)
    payload["reference_metrics"] = {
        "dsr": gates15.deflated_sharpe_reference(result),
        "total_trials_disclosed": gates15.DSR_CUMULATIVE_TRIALS,
    }
    _save_json(path, payload)
    return report


def _stage_gates(only: str | None) -> None:
    for seed_offset, candidate_id in enumerate(configs15.CANDIDATE_IDS):
        if only is not None and candidate_id != only:
            continue
        report = _evaluate_and_save(candidate_id, seed_offset)
        print(
            f"gates: {candidate_id} -> {report.overall} "
            f"(high_funding_annualized={report.promotion.high_funding_mean_annualized_return}, reasons={list(report.failure_reasons)})"
        )


def _stage_run(only: str | None) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _run_l4_reference()
    _run_a_series(only)
    _run_daily_mechanism_candidates(only)
    _run_d1(only)


def _stage_report() -> None:
    write_wave15_report(RESULTS_DIR, REPORT_DIR, REGISTRY_PATH, CACHE_DIR)
    print(f"report: wrote {REPORT_DIR / 'wave15_report.md'} and {REGISTRY_PATH}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Wave-15 method-diversification pipeline")
    parser.add_argument("--stage", required=True, type=Stage, choices=tuple(Stage))
    parser.add_argument("--only", choices=configs15.CANDIDATE_IDS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        match args.stage:
            case Stage.RUN:
                _stage_run(args.only)
            case Stage.GATES:
                _stage_gates(args.only)
            case Stage.REPORT:
                _stage_report()
            case Stage.ALL:
                _stage_run(args.only)
                _stage_gates(args.only)
                _stage_report()
            case unreachable:
                assert_never(unreachable)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
