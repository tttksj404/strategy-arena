# Wave-17 work item 2: F0-F3 (+ reference-only F_min) recompute. Reuses
# research.wave16_duallayer.engine16's `DualLayerRunner` / `VariantRun` / discount mechanism
# (imported, NOT reimplemented -- no new backtest loop, no new frame-construction, no new
# cost/liquidity model anywhere in this module). The only new logic here is (a) building a
# `lending_apr_by_symbol` mapping from cache/lending_realized.json's REALIZED `lendingRate`
# stats instead of wave16's `avgRate`, via wave16's OWN symbol<->ccy join
# (`by_symbol[symbol]['base_ccy_matched']`, read-only), and (b) calling
# `DualLayerRunner.run_variant(ranking_lending_discount=0.0, pnl_lending_discount=X)` at
# X in {0.0, 1.0, 0.5, 0.0} for F0/F1/F2/F3 -- `ranking_lending_discount` is 0.0 for every
# candidate in this wave BY DESIGN (SPEC.md 범위: this wave only re-derives E1-style candidates,
# never E2/E3/E4's combined-ranking variants, which wave16 already evaluated and rejected on
# their own terms independent of the avgRate/lendingRate issue).
#
# Because ranking_lending_discount=0.0 for F0-F3, EVERY candidate's "funding-only companion"
# (SPEC.md 방법 4's own convention, inherited from wave16) is the (0.0, 0.0) variant -- i.e.
# F0 itself -- for every one of F0/F1/F2/F3, not just some of them (contrast wave16, where only
# E1's companion was E0; E2/E4 shared a different companion; E3 was unique). This also means
# F0 and F3 (both (0.0, 0.0)) are, structurally, THE SAME RUN -- not just expected to match
# empirically, but literally the same memoized DualLayerRunner cache entry -- which is exactly
# SPEC.md's own "F3 = 회귀/무결성 검증" requirement made true by construction, not by a
# post-hoc tolerance check (tests/test_wave17.py pins this identity).

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Final

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

from research.wave10_carry100.regime import regime_breakdown
from research.wave13_liquidity import costs_measured
from research.wave13_liquidity import universe_liquidity as ul
from research.wave16_duallayer import engine16
from research.wave16_duallayer.configs16 import L4_CONFIG
from research.wave16_duallayer.run_wave16 import _json_safe, _save_json, _variant_payload
from research.wave17_lending_verified import fetch17

BASE_DIR: Final = Path(__file__).resolve().parent
RESULTS_DIR: Final = BASE_DIR / "results"

MEDIAN_FIELD: Final = "lending_rate_median"
MIN_FIELD: Final = "lending_rate_min"


@dataclass(frozen=True, slots=True)
class Candidate17:
    candidate_id: str
    pnl_lending_discount: float
    lending_apr_source: str  # "realized_lending_rate_median_4day" | "realized_lending_rate_min_4day" | "none (pnl_discount=0.0)"
    note: str


CANDIDATES17: Final[tuple[Candidate17, ...]] = (
    Candidate17(
        "F0",
        0.0,
        "none (pnl_discount=0.0)",
        "L4/E0 재현 -- 기준선. 랭킹=펀딩만, 대여수익 0. wave16 E0와 동일 (ranking=0.0, pnl=0.0이면 "
        "어떤 lending_apr_by_symbol을 넣어도 결과가 그 매핑과 무관해진다 -- 곱해지는 계수가 0이므로).",
    ),
    Candidate17(
        "F1",
        1.0,
        "realized_lending_rate_median_4day",
        "wave16 E1의 정정판 -- 랭킹=펀딩만(F0과 동일 트레이드 선택), 대여수익은 OKX "
        "lending-rate-history의 실측 lendingRate 4일 중앙값(코인별)을 가산. avgRate(wave16이 실제로 "
        "쓴 값)이 아니다.",
    ),
    Candidate17(
        "F2",
        0.5,
        "realized_lending_rate_median_4day",
        "F1 + 50% 보수 할인 -- 변동성/미체결 리스크 대비 (F1과 동일한 realized median 값에 "
        "pnl_lending_discount=0.5만 적용, 값 자체를 바꾸지 않음).",
    ),
    Candidate17(
        "F3",
        0.0,
        "none (pnl_discount=0.0)",
        "대여수익 0% 가정(실패 시나리오) -- 정의상 F0과 동일한 (ranking=0.0, pnl=0.0) 변형이므로 "
        "F0의 회귀/무결성 검증 역할을 겸한다.",
    ),
)
CANDIDATE_IDS17: Final[tuple[str, ...]] = tuple(c.candidate_id for c in CANDIDATES17)

F_MIN = Candidate17(
    "F_min",
    1.0,
    "realized_lending_rate_min_4day",
    "참고용(게이트 미적용) -- 코인별 4일 관측 구간 중 최저 lendingRate를 대여수익으로 가정한 "
    "최악 시나리오. 이미 최저값이므로 추가 할인은 적용하지 않는다.",
)


# ---------------------------------------------------------------------------
# symbol <-> ccy join (read-only reuse of wave16's OWN mapping -- never recomputed here).
# ---------------------------------------------------------------------------


def load_symbol_to_ccy_map() -> dict[str, str | None]:
    import json

    wave16_snapshot = json.loads(fetch17.WAVE16_SNAPSHOT_PATH.read_text(encoding="utf-8"))
    return {symbol: info.get("base_ccy_matched") for symbol, info in wave16_snapshot["by_symbol"].items()}


def build_realized_lending_apr_by_symbol(lending_realized: dict[str, Any], stat_field: str) -> dict[str, float]:
    """symbol -> realized lending APR (`stat_field` in {lending_rate_median, lending_rate_min}),
    joined via wave16's OWN symbol<->ccy map. Fail-closed: a symbol whose ccy has no
    `history_available` realized data (or whose ccy is unmatched entirely) is simply OMITTED --
    engine16's own `lending_apr_by_symbol.get(symbol, 0.0)` already treats a missing key as 0.0
    (see engine16.py's `_build_aligned_frames_dual` / `_lending_daily_rate_series`), so this never
    invents a rate for a symbol wave16 itself couldn't match."""
    symbol_to_ccy = load_symbol_to_ccy_map()
    by_ccy = lending_realized["by_ccy"]
    result: dict[str, float] = {}
    for symbol, ccy in symbol_to_ccy.items():
        if not ccy:
            continue
        row = by_ccy.get(ccy)
        if row is None or not row.get("history_available"):
            continue
        value = row.get(stat_field)
        if value is None:
            continue
        result[symbol] = float(value)
    return result


# ---------------------------------------------------------------------------
# Runner construction -- shared markets/mapping (loaded ONCE), two lending_apr_by_symbol
# mappings (median-based for F0-F3, min-based for F_min only) each get their own
# DualLayerRunner instance (constructed directly, not via engine16.build_runner, purely to
# avoid loading the SAME markets/cost-mapping twice -- DualLayerRunner itself is still the
# exact class engine16.build_runner would have handed back).
# ---------------------------------------------------------------------------


def build_runners(lending_realized: dict[str, Any]) -> tuple[engine16.DualLayerRunner, engine16.DualLayerRunner, dict[str, float], dict[str, float]]:
    symbols = ul.verify_cache_and_load_symbols(L4_CONFIG)
    markets = ul.load_markets_for_symbols(symbols)
    mapping = costs_measured.fit_mapping()
    median_apr = build_realized_lending_apr_by_symbol(lending_realized, MEDIAN_FIELD)
    min_apr = build_realized_lending_apr_by_symbol(lending_realized, MIN_FIELD)
    runner_median = engine16.DualLayerRunner(L4_CONFIG, mapping, markets, median_apr)
    runner_min = engine16.DualLayerRunner(L4_CONFIG, mapping, markets, min_apr)
    return runner_median, runner_min, median_apr, min_apr


def _variant(runner: engine16.DualLayerRunner, ranking_discount: float, pnl_discount: float) -> engine16.VariantRun:
    """Identical body to engine16._variant_run -- restated (not imported) only because that
    helper is module-private; the actual computation (`runner.run_variant`, which owns the
    frame-build + backtest-loop + memoization) is 100% engine16's own, not reimplemented here."""
    result, total_cost, eligible = runner.run_variant(ranking_discount, pnl_discount, engine16.DEFAULT_STRESS_MULTIPLIER)
    stress_result, stress_total_cost, stress_eligible = runner.run_variant(ranking_discount, pnl_discount, engine16.STRESS_MULTIPLIER)
    return engine16.VariantRun(result, total_cost, eligible, stress_result, stress_total_cost, stress_eligible)


def run_all_candidates() -> dict[str, Any]:
    """Returns {'variants': {candidate_id: VariantRun}, 'runner_median':..., 'runner_min':...,
    'median_apr_by_symbol':..., 'min_apr_by_symbol':...}."""
    lending_realized = fetch17.load_lending_realized()
    runner_median, runner_min, median_apr, min_apr = build_runners(lending_realized)

    variants: dict[str, engine16.VariantRun] = {}
    for candidate in CANDIDATES17:
        variants[candidate.candidate_id] = _variant(runner_median, 0.0, candidate.pnl_lending_discount)
    variants[F_MIN.candidate_id] = _variant(runner_min, 0.0, F_MIN.pnl_lending_discount)

    return {
        "variants": variants,
        "runner_median": runner_median,
        "runner_min": runner_min,
        "median_apr_by_symbol": median_apr,
        "min_apr_by_symbol": min_apr,
        "lending_realized": lending_realized,
    }


# ---------------------------------------------------------------------------
# JSON payload (mirrors research.wave16_duallayer.run_wave16's own `_base_payload` shape --
# `_variant_payload`/`_json_safe`/`_save_json` are reused directly, imported).
# ---------------------------------------------------------------------------


def _base_payload(candidate: Candidate17, variant: engine16.VariantRun, funding_only_variant: engine16.VariantRun, lending_realized: dict[str, Any], n_symbols_with_data: int) -> dict[str, Any]:
    fam_candidate = L4_CONFIG.candidate
    payload: dict[str, Any] = {
        "candidate_id": candidate.candidate_id,
        "family": "wave17_lending_verified",
        "definition": candidate.note,
        "ranking_lending_discount": 0.0,
        "pnl_lending_discount": candidate.pnl_lending_discount,
        "lending_apr_source": candidate.lending_apr_source,
        "config": {
            "window_days": fam_candidate.window_days,
            "threshold_apr": fam_candidate.threshold_apr,
            "top_k_pairs": fam_candidate.top_k,
            "leg_fraction_of_active_capital": L4_CONFIG.leg_fraction,
            "universe_kind": L4_CONFIG.universe_kind,
            "breadth": L4_CONFIG.breadth,
            "history_months": L4_CONFIG.history_months,
        },
        "capital_contract": {
            "total_capital_usdt": engine16.TOTAL_CAPITAL,
            "reserve_fraction": engine16.RESERVE_FRACTION,
            "active_capital_usdt": engine16.ACTIVE_CAPITAL,
            "min_order_usdt": engine16.MIN_ORDER_USDT,
        },
        "cost_model": "bitget_measured_volume_mapping(wave13_liquidity.costs_measured, unmodified)+maker_0.02pct_per_leg -- L4/wave16 승계, wave17이 재도출하지 않음",
        "metadata": {
            "symbols_used": list(variant.result.symbols_used),
            "universe_size_static": len(variant.result.symbols_used),
            "n_symbols_with_realized_lending_data": n_symbols_with_data,
            "n_symbols_with_realized_lending_data_pct": (n_symbols_with_data / len(variant.result.symbols_used) * 100.0) if variant.result.symbols_used else 0.0,
            "lending_realized_collected_at_utc": lending_realized.get("collected_at_utc"),
            "lending_realized_history_span_days_median": lending_realized.get("span_days_median"),
            "max_concurrent_positions_combined": variant.result.max_concurrent_positions,
            "max_concurrent_positions_funding_only": funding_only_variant.result.max_concurrent_positions,
        },
        "regime_breakdown_combined": regime_breakdown(variant.result),
        "regime_breakdown_funding_only": regime_breakdown(funding_only_variant.result),
    }
    payload.update(_variant_payload("combined_", variant))
    payload.update(_variant_payload("funding_only_", funding_only_variant))
    return payload


def run_and_save() -> dict[str, dict[str, Any]]:
    run = run_all_candidates()
    variants: dict[str, engine16.VariantRun] = run["variants"]
    lending_realized = run["lending_realized"]
    median_apr = run["median_apr_by_symbol"]
    min_apr = run["min_apr_by_symbol"]
    f0_variant = variants["F0"]  # every F0-F3 candidate's funding-only companion, by construction

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payloads: dict[str, dict[str, Any]] = {}
    for candidate in CANDIDATES17:
        n_with_data = sum(1 for symbol in f0_variant.result.symbols_used if symbol in median_apr)
        payload = _base_payload(candidate, variants[candidate.candidate_id], f0_variant, lending_realized, n_with_data)
        payloads[candidate.candidate_id] = payload
        _save_json(RESULTS_DIR / f"{candidate.candidate_id}.json", payload)

    n_with_min_data = sum(1 for symbol in f0_variant.result.symbols_used if symbol in min_apr)
    f_min_funding_only = variants["F0"]  # F_min uses runner_min, but its OWN (0.0,0.0) companion is numerically identical to F0 (see module docstring); reuse F0's object directly rather than re-deriving
    f_min_payload = _base_payload(F_MIN, variants["F_min"], f_min_funding_only, lending_realized, n_with_min_data)
    payloads["F_min"] = f_min_payload
    _save_json(RESULTS_DIR / "F_min.json", f_min_payload)

    for candidate_id, payload in payloads.items():
        combined_final = payload["combined_equity"][-1]["value"] if payload["combined_equity"] else float("nan")
        high = payload["regime_breakdown_combined"].get("high_funding_mean_annualized_return")
        print(f"recompute17: {candidate_id} done (combined_final=${combined_final:.2f}, high_funding_annualized={high})")
    return payloads


__all__ = [
    "CANDIDATE_IDS17",
    "CANDIDATES17",
    "F_MIN",
    "Candidate17",
    "build_realized_lending_apr_by_symbol",
    "build_runners",
    "load_symbol_to_ccy_map",
    "run_all_candidates",
    "run_and_save",
]


if __name__ == "__main__":
    run_and_save()
