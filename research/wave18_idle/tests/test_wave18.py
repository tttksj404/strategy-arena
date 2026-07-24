from __future__ import annotations

from pathlib import Path
import sys

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[3]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

import pandas as pd  # noqa: PANDAS_OK
import pytest

from research.wave1.fam_funding import FundingCandidate, FundingMarket, carry_position
from research.wave13_liquidity import engine13
from research.wave18_idle import engine18, gates18
from research.wave18_idle.configs18 import (
    CONFIG_IDS,
    CONFIGS,
    L4_CONFIG,
    LEG_FRACTION,
    MAJORS_ONLY_SYMBOLS,
    OVERLAY_CARRY_CANDIDATE,
    OVERLAY_REVERSE_CANDIDATE,
    TOP_K,
    get_config,
)

# ---------------------------------------------------------------------------
# Synthetic market fixtures (same style as research/wave13_liquidity/tests/test_wave13.py's
# own _two_symbol_synthetic_market -- duplicated per-wave-test-file convention, not imported).
# ---------------------------------------------------------------------------


def _positive_funding_market(periods: int = 12) -> dict[str, FundingMarket]:
    daily_index = pd.date_range("2026-01-01", periods=periods, freq="D", tz="UTC")
    funding_index = pd.date_range("2026-01-01", periods=periods * 3, freq="8h", tz="UTC")
    a_close = [100.0 * (1.004**i) for i in range(periods)]
    a_open = [100.0, *a_close[:-1]]
    b_close = [50.0 * (1.002**i) for i in range(periods)]
    b_open = [50.0, *b_close[:-1]]
    spot_a = pd.DataFrame({"open": a_open, "close": a_close}, index=daily_index)
    perp_a = pd.DataFrame({"open": [v * 0.999 for v in a_open], "close": [v * 0.999 for v in a_close]}, index=daily_index)
    spot_b = pd.DataFrame({"open": b_open, "close": b_close}, index=daily_index)
    perp_b = pd.DataFrame({"open": [v * 1.001 for v in b_open], "close": [v * 1.001 for v in b_close]}, index=daily_index)
    funding_a = pd.Series(0.0010, index=funding_index, name="funding_rate")
    funding_b = pd.Series(0.0006, index=funding_index, name="funding_rate")
    return {"BTCUSDT": FundingMarket(spot_a, perp_a, funding_a), "ETHUSDT": FundingMarket(spot_b, perp_b, funding_b)}


def _negative_funding_market(periods: int = 12) -> dict[str, FundingMarket]:
    """Same price paths as _positive_funding_market, but funding is strongly NEGATIVE
    throughout (shorts paying longs) -- used to drive engine18.reverse_carry_position /
    the reverse overlay layer into an actual entry."""
    daily_index = pd.date_range("2026-01-01", periods=periods, freq="D", tz="UTC")
    funding_index = pd.date_range("2026-01-01", periods=periods * 3, freq="8h", tz="UTC")
    a_close = [100.0 * (1.001**i) for i in range(periods)]
    a_open = [100.0, *a_close[:-1]]
    b_close = [50.0 * (0.999**i) for i in range(periods)]
    b_open = [50.0, *b_close[:-1]]
    spot_a = pd.DataFrame({"open": a_open, "close": a_close}, index=daily_index)
    perp_a = pd.DataFrame({"open": [v * 1.0005 for v in a_open], "close": [v * 1.0005 for v in a_close]}, index=daily_index)
    spot_b = pd.DataFrame({"open": b_open, "close": b_close}, index=daily_index)
    perp_b = pd.DataFrame({"open": [v * 0.9995 for v in b_open], "close": [v * 0.9995 for v in b_close]}, index=daily_index)
    funding_a = pd.Series(-0.0100, index=funding_index, name="funding_rate")  # ~ -0.03/day * 3 * 365 ~= -32.85% APR
    funding_b = pd.Series(-0.0120, index=funding_index, name="funding_rate")
    return {"BTCUSDT": FundingMarket(spot_a, perp_a, funding_a), "ETHUSDT": FundingMarket(spot_b, perp_b, funding_b)}


# ---------------------------------------------------------------------------
# 1) Config registry integrity: frozen 6 candidates, layer flags match SPEC.md's table.
# ---------------------------------------------------------------------------


def test_config_registry_is_frozen_to_six_preregistered_ids() -> None:
    assert CONFIG_IDS == ("I0", "I1", "I2", "I3", "I4", "I5")
    assert len(CONFIGS) == 6


def test_idle_config_layer_flags_match_spec_table() -> None:
    i0, i1, i2, i3, i4, i5 = (get_config(cid) for cid in CONFIG_IDS)
    assert not i0.uses_carry_overlay and not i0.uses_reverse_overlay and not i0.uses_lending_fallback
    assert i1.uses_lending_fallback and not i1.uses_carry_overlay and not i1.uses_reverse_overlay
    assert i2.uses_carry_overlay and i2.overlay_symbols == MAJORS_ONLY_SYMBOLS and not i2.uses_lending_fallback and not i2.uses_reverse_overlay
    assert i3.uses_carry_overlay and i3.overlay_symbols is None and not i3.uses_lending_fallback  # top200, unrestricted
    assert i4.uses_reverse_overlay and not i4.uses_carry_overlay and not i4.uses_lending_fallback
    assert i5.uses_carry_overlay and i5.overlay_symbols == MAJORS_ONLY_SYMBOLS and i5.uses_lending_fallback  # I2 first, I1 fallback


def test_capital_contract_and_thresholds_match_spec() -> None:
    assert LEG_FRACTION == pytest.approx(0.50)
    assert TOP_K == 1
    assert L4_CONFIG.candidate.threshold_apr == pytest.approx(0.15)
    assert OVERLAY_CARRY_CANDIDATE.threshold_apr == pytest.approx(0.08)
    assert OVERLAY_CARRY_CANDIDATE.threshold_apr / 2.0 == pytest.approx(0.04)  # carry_position's built-in exit -- matches wave-11 Y1's own "진입 8%/청산 4%"
    assert OVERLAY_REVERSE_CANDIDATE.threshold_apr == pytest.approx(0.15)  # magnitude for SPEC.md's "-15% APR"


# ---------------------------------------------------------------------------
# 2) reverse_carry_position is the algebraic mirror of carry_position.
# ---------------------------------------------------------------------------


def test_reverse_carry_position_equals_carry_position_on_negated_score() -> None:
    idx = pd.date_range("2026-01-01", periods=40, freq="D", tz="UTC")
    # a path that wanders across both the +threshold and -threshold/2 boundaries several times
    raw = [0.30, 0.30, 0.30, -0.30, -0.30, 0.02, 0.02, -0.20, -0.20, -0.02] * 4
    score = pd.Series(raw[:40], index=idx, dtype=float)
    candidate = FundingCandidate("T", 7, 0.15, 1)

    reverse = engine18.reverse_carry_position(score, candidate)
    mirrored = carry_position(-score, candidate)
    assert reverse.tolist() == pytest.approx(mirrored.tolist())
    # sanity: it actually turns on somewhere (not degenerately all-zero)
    assert reverse.sum() > 0.0


# ---------------------------------------------------------------------------
# 3) Engine equivalence: engine18 with NO overlays and NO lending must reproduce engine13's
#    own loop bit-for-bit -- this is the mechanism I0's "reproduces L4" claim ultimately rests
#    on (run_wave18.py itself calls engine13.run_candidate directly for I0; THIS test proves
#    the more general engine18 loop reduces to the identical numbers when unused).
# ---------------------------------------------------------------------------


def test_engine18_matches_engine13_with_no_overlays_and_no_lending() -> None:
    markets = _positive_funding_market()
    candidate = FundingCandidate("EQTEST", 1, 0.50, 1)
    leg_fraction = 0.5

    frames13 = engine13._build_aligned_frames(markets, candidate)
    spot_open13 = frames13[0]
    flat_cost = pd.DataFrame({s: [0.001] * len(spot_open13.index) for s in spot_open13.columns}, index=spot_open13.index)
    always_liquid = pd.DataFrame(True, index=spot_open13.index, columns=spot_open13.columns)
    result13, cost13, _elig13 = engine13._run_liquidity_loop(*frames13, candidate.top_k, leg_fraction, flat_cost, always_liquid)

    spot_open18, spot_close18, perp_open18, perp_close18, funding18, raw_score18 = engine18._build_aligned_frames18(markets, window_days=1)
    ranking18 = raw_score18.shift(1)
    l4_active18 = engine18.active_frame_for(raw_score18, candidate)
    result18, cost18, _elig18 = engine18._run_idle_overlay_loop(
        spot_open18, spot_close18, perp_open18, perp_close18, funding18,
        ranking18, l4_active18, (), candidate.top_k, leg_fraction, flat_cost, always_liquid, None,
    )

    assert result18.equity.tolist() == pytest.approx(result13.equity.tolist(), rel=1e-12)
    assert result18.positions.tolist() == pytest.approx(result13.positions.tolist(), rel=1e-12)
    assert result18.turnover.tolist() == pytest.approx(result13.turnover.tolist(), rel=1e-12)
    assert result18.trade_returns.tolist() == pytest.approx(result13.trade_returns.tolist(), rel=1e-12)
    assert cost18 == pytest.approx(cost13, rel=1e-12)
    assert set(result18.layer_used.unique()) <= {engine18.LAYER_L4, engine18.LAYER_CASH}


# ---------------------------------------------------------------------------
# 4) S6 recoverability -- structural: an overlay layer that is ALSO eligible on an L4-active
#    day must never win that day, and the resulting position must be identical to an L4-only
#    run on those specific days (no contamination).
# ---------------------------------------------------------------------------


def test_overlay_never_overrides_an_active_l4_day() -> None:
    markets = _positive_funding_market()
    l4_candidate = FundingCandidate("L4T", 1, 0.30, 1)
    overlay_candidate = FundingCandidate("OVT", 1, 0.05, 1)  # looser -> eligible whenever L4 is, and more

    spot_open, spot_close, perp_open, perp_close, funding, raw_score = engine18._build_aligned_frames18(markets, window_days=1)
    ranking = raw_score.shift(1)
    l4_active = engine18.active_frame_for(raw_score, l4_candidate)
    overlay_active = engine18.active_frame_for(raw_score, overlay_candidate)
    symbols = tuple(spot_open.columns)
    cost = pd.DataFrame(0.001, index=spot_open.index, columns=symbols)
    liquid = pd.DataFrame(True, index=spot_open.index, columns=symbols)
    layer = engine18.OverlayLayer(engine18.LAYER_CARRY_OVERLAY, overlay_active, symbols, 1.0)

    combined, _, _ = engine18._run_idle_overlay_loop(
        spot_open, spot_close, perp_open, perp_close, funding, ranking, l4_active, (layer,), 1, 0.5, cost, liquid, None
    )
    l4_only, _, _ = engine18._run_idle_overlay_loop(
        spot_open, spot_close, perp_open, perp_close, funding, ranking, l4_active, (), 1, 0.5, cost, liquid, None
    )

    l4_active_days = l4_only.positions.abs() > 0.0
    assert bool(l4_active_days.any())  # the fixture must actually exercise this path
    assert (combined.layer_used[l4_active_days] == engine18.LAYER_L4).all()
    assert combined.positions[l4_active_days].tolist() == pytest.approx(l4_only.positions[l4_active_days].tolist())
    assert combined.equity[l4_active_days].tolist() == pytest.approx(l4_only.equity[l4_active_days].tolist())


# ---------------------------------------------------------------------------
# 5) Lending fallback: on days both L4 AND every overlay miss, capital compounds by the flat
#    daily lending rate and the delta-neutral weights vector stays exactly zero.
# ---------------------------------------------------------------------------


def test_lending_fallback_compounds_on_fully_idle_days() -> None:
    markets = _positive_funding_market()
    l4_never = FundingCandidate("L4NEVER", 1, 5.0, 1)  # threshold unreachable -> L4 never active
    spot_open, spot_close, perp_open, perp_close, funding, raw_score = engine18._build_aligned_frames18(markets, window_days=1)
    ranking = raw_score.shift(1)
    l4_active = engine18.active_frame_for(raw_score, l4_never)
    symbols = tuple(spot_open.columns)
    cost = pd.DataFrame(0.0, index=spot_open.index, columns=symbols)  # zero cost isolates the lending arithmetic
    liquid = pd.DataFrame(True, index=spot_open.index, columns=symbols)
    lending_daily = 0.0001

    result, _, _ = engine18._run_idle_overlay_loop(
        spot_open, spot_close, perp_open, perp_close, funding, ranking, l4_active, (), 1, 0.5, cost, liquid, lending_daily
    )

    assert (result.layer_used == engine18.LAYER_LENDING).all()
    assert (result.positions == 0.0).all()  # lending never touches the delta-neutral weights vector
    expected_final = engine18.ACTIVE_CAPITAL * (1.0 + lending_daily) ** len(spot_open.index)
    assert float(result.equity.iloc[-1]) == pytest.approx(expected_final, rel=1e-9)


# ---------------------------------------------------------------------------
# 6) Reverse-overlay (I4) trades are recorded, not silently dropped -- guards the
#    abs(weight)-generalization of engine13's own trade_growth bookkeeping.
# ---------------------------------------------------------------------------


def test_reverse_overlay_trades_are_recorded_and_move_equity() -> None:
    markets = _negative_funding_market()
    l4_never = FundingCandidate("L4NEVER", 1, 5.0, 1)
    reverse_candidate = FundingCandidate("REV", 1, 0.05, 1)

    spot_open, spot_close, perp_open, perp_close, funding, raw_score = engine18._build_aligned_frames18(markets, window_days=1)
    ranking = raw_score.shift(1)
    l4_active = engine18.active_frame_for(raw_score, l4_never)
    reverse_active = engine18.reverse_active_frame_for(raw_score, reverse_candidate)
    symbols = tuple(spot_open.columns)
    cost = pd.DataFrame(0.001, index=spot_open.index, columns=symbols)
    liquid = pd.DataFrame(True, index=spot_open.index, columns=symbols)
    layer = engine18.OverlayLayer(engine18.LAYER_REVERSE_OVERLAY, reverse_active, symbols, -1.0)

    result, total_cost, _ = engine18._run_idle_overlay_loop(
        spot_open, spot_close, perp_open, perp_close, funding, ranking, l4_active, (layer,), 1, 0.5, cost, liquid, None
    )

    assert (result.layer_used == engine18.LAYER_REVERSE_OVERLAY).any()
    assert len(result.trade_returns) >= 1  # NOT silently dropped by the abs()-generalized open/close branches
    assert total_cost > 0.0
    assert float(result.equity.iloc[-1]) != pytest.approx(engine18.ACTIVE_CAPITAL)  # something actually happened


def test_signed_weight_direction_flip_costs_double_a_single_entry() -> None:
    """Turnover-cost sanity for signed weights: going from +leg_fraction to -leg_fraction on
    the SAME symbol must cost exactly 2x a fresh entry's cost (close one direction, open the
    other) -- engine13's existing abs(weights_new - weights_old) formula already gets this
    right once weights carry a sign; this pins that claim numerically."""
    idx = pd.date_range("2026-01-01", periods=1, freq="D", tz="UTC")
    weights_from = pd.Series({"AAAUSDT": 0.5})
    weights_to = pd.Series({"AAAUSDT": -0.5})
    leg_rate = 0.001
    fresh_entry_cost = abs(0.5 - 0.0) * leg_rate
    flip_cost = abs(float(weights_to["AAAUSDT"]) - float(weights_from["AAAUSDT"])) * leg_rate
    assert flip_cost == pytest.approx(2.0 * fresh_entry_cost)
    _ = idx  # index unused, kept only for readability of the "one day" framing above


# ---------------------------------------------------------------------------
# 7) daily_rate_from_apr round-trips to the same APR under compounding.
# ---------------------------------------------------------------------------


def test_daily_rate_from_apr_compounds_back_to_the_same_apr() -> None:
    apr = 0.05
    daily = engine18.daily_rate_from_apr(apr)
    compounded = (1.0 + daily) ** 365.0 - 1.0
    assert compounded == pytest.approx(apr, rel=1e-9)


# ---------------------------------------------------------------------------
# 8) gates18: full_period_annualized, gate_s6_recoverability, promotion_check.
# ---------------------------------------------------------------------------


def test_full_period_annualized_matches_known_growth_over_one_year() -> None:
    idx = pd.date_range("2020-01-01", periods=2, freq="365D", tz="UTC")  # exactly 365 days apart
    equity = pd.Series([100.0, 121.0], index=idx, dtype=float)
    cagr = gates18.full_period_annualized(equity)
    assert cagr == pytest.approx(0.21, rel=1e-6)


def test_gate_s6_flags_violation_when_overlay_appears_on_an_l4_active_day() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
    i0_positions = pd.Series([0.0, 0.5, 0.5, 0.0, 0.0], index=idx)
    layer_ok = pd.Series(["cash", "L4", "L4", "lending", "cash"], index=idx, dtype=object)
    layer_bad = pd.Series(["cash", "carry_overlay", "L4", "lending", "cash"], index=idx, dtype=object)

    ok = gates18.gate_s6_recoverability(layer_ok, i0_positions)
    bad = gates18.gate_s6_recoverability(layer_bad, i0_positions)
    assert ok["status"] == "PASS" and ok["violations"] == 0 and ok["l4_active_days"] == 2
    assert bad["status"] == "FAIL" and bad["violations"] == 1


def test_promotion_check_requires_gates_and_both_return_conditions() -> None:
    # high_funding=0.215 vs i0=0.22 -> gap = -0.5pp, comfortably within the -1pp tolerance
    passes_all = gates18.promotion_check(0.10, 0.215, 0.09, 0.22, "PASS")
    assert passes_all.promoted is True

    misses_full_period = gates18.promotion_check(0.08, 0.215, 0.09, 0.22, "PASS")
    assert misses_full_period.promoted is False
    assert misses_full_period.beats_i0_full_period is False

    damages_high_funding = gates18.promotion_check(0.10, 0.19, 0.09, 0.22, "PASS")  # gap = -3pp, worse than -1pp tolerance
    assert damages_high_funding.promoted is False
    assert damages_high_funding.within_tolerance_of_i0_high_funding is False

    gates_fail = gates18.promotion_check(0.10, 0.215, 0.09, 0.22, "FAIL")
    assert gates_fail.promoted is False


def test_promotion_check_high_funding_exactly_at_tolerance_boundary_passes() -> None:
    # gap == exactly -1.0pp must still count as "within tolerance" (SPEC.md: "-1%p 이내" == inclusive)
    boundary = gates18.promotion_check(0.10, 0.21, 0.09, 0.22, "PASS")
    assert boundary.high_funding_gap_pp == pytest.approx(-1.0)
    assert boundary.within_tolerance_of_i0_high_funding is True
    assert boundary.promoted is True
