# Wave-22 test suite. Every test uses either pure in-memory data (no cache) or the fast
# synthetic-market cache (fitness.market_cache_from_markets, no disk I/O) EXCEPT
# test_g1_reproduces_strategy_card_reference_numbers_on_real_cache, which is this suite's one
# slower integration check against the real repo cache -- mirrors
# research/wave21_ga/tests/test_wave21.py's own convention (self-skips if the real cache is
# unavailable).

from __future__ import annotations

from pathlib import Path
import sys

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[3]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

import numpy as np
import pandas as pd  # noqa: PANDAS_OK
import pytest

from research.wave1.fam_funding import FundingMarket
from research.wave21_ga import fitness
from research.wave21_ga.genome import GENE_NAMES, Genome, I5_BASELINE_GENOME
from research.wave22_overfit import attribution, dsr, perturb, regime, reporting22, rolling, shuffle_control, verdict
from research.wave22_overfit.evaluate import MetricsCache
from research.wave22_overfit.genomes import G1_GENOME, G1_REFERENCE_METRICS, I5_GENOME

# ---------------------------------------------------------------------------
# Synthetic market fixture (mirrors tests/test_wave21.py's own _two_symbol_market).
# ---------------------------------------------------------------------------


def _synthetic_market(start: str = "2024-01-01", periods: int = 400) -> dict[str, FundingMarket]:
    idx = pd.date_range(start, periods=periods, freq="D", tz="UTC")
    funding_idx = pd.date_range(start, periods=periods * 3, freq="8h", tz="UTC")
    rng = np.random.default_rng(42)
    a_close = 100.0 * np.cumprod(1.0 + rng.normal(0.0004, 0.01, periods))
    a_open = np.concatenate([[100.0], a_close[:-1]])
    b_close = 50.0 * np.cumprod(1.0 + rng.normal(0.0002, 0.008, periods))
    b_open = np.concatenate([[50.0], b_close[:-1]])
    spot_a = pd.DataFrame({"open": a_open, "close": a_close}, index=idx)
    perp_a = pd.DataFrame({"open": a_open * 0.999, "close": a_close * 0.999}, index=idx)
    spot_b = pd.DataFrame({"open": b_open, "close": b_close}, index=idx)
    perp_b = pd.DataFrame({"open": b_open * 1.001, "close": b_close * 1.001}, index=idx)
    funding_a = pd.Series(0.0009, index=funding_idx, name="funding_rate")
    funding_b = pd.Series(0.0004, index=funding_idx, name="funding_rate")
    return {"AAAUSDT": FundingMarket(spot_a, perp_a, funding_a), "BBBUSDT": FundingMarket(spot_b, perp_b, funding_b)}


@pytest.fixture(scope="module")
def synthetic_cache() -> fitness.MarketCache:
    return fitness.market_cache_from_markets(_synthetic_market(), majors=("AAAUSDT",), flat_cost_rate=0.0008, always_liquid=True)


SYNTH_I5 = Genome(entry_threshold_apr=0.15, exit_threshold_ratio=0.5, window_days=7, top_k_pairs=1, leg_fraction=0.30, universe_breadth=300, idle_mode="none")
SYNTH_G1 = Genome(entry_threshold_apr=0.10, exit_threshold_ratio=0.4, window_days=10, top_k_pairs=1, leg_fraction=0.30, universe_breadth=200, idle_mode="usdt_lend")


# ---------------------------------------------------------------------------
# 1) genomes.py -- frozen values.
# ---------------------------------------------------------------------------


def test_g1_genome_matches_frozen_task_spec_values() -> None:
    assert G1_GENOME.entry_threshold_apr == pytest.approx(0.11818955034178509)
    assert G1_GENOME.exit_threshold_ratio == pytest.approx(0.3873937165748336)
    assert G1_GENOME.window_days == 14
    assert G1_GENOME.top_k_pairs == 1
    assert G1_GENOME.leg_fraction == pytest.approx(0.5)
    assert G1_GENOME.universe_breadth == 100
    assert G1_GENOME.idle_mode == "usdt_lend"


def test_i5_genome_is_the_shared_wave21_baseline_not_a_redeclared_copy() -> None:
    assert I5_GENOME == I5_BASELINE_GENOME


# ---------------------------------------------------------------------------
# 2) perturb.py -- continuous axis tiers.
# ---------------------------------------------------------------------------


def test_continuous_axis_perturbation_applies_signed_percent_and_stays_in_bounds() -> None:
    for tier, expected_delta in ((-2, -0.20), (-1, -0.10), (1, 0.10), (2, 0.20)):
        point = perturb.perturb_axis(G1_GENOME, "entry_threshold_apr", tier)
        assert point.available
        expected_value = G1_GENOME.entry_threshold_apr * (1.0 + expected_delta)
        assert point.gene_value == pytest.approx(expected_value)
        # constructing the perturbed Genome must not raise (i.e. stays within genome.py's registry bounds)
        assert point.genome is not None
        assert point.genome.entry_threshold_apr == pytest.approx(expected_value)


def test_continuous_axis_perturbation_clips_and_flags_boundary_collapse() -> None:
    # leg_fraction=0.5 is AT the registered max (LEG_FRACTION_MAX=0.50); any +tier must clip
    # back onto 0.5 itself and therefore be reported unavailable (no real perturbation exists).
    plus_10 = perturb.perturb_axis(G1_GENOME, "leg_fraction", 1)
    plus_20 = perturb.perturb_axis(G1_GENOME, "leg_fraction", 2)
    assert plus_10.available is False
    assert plus_20.available is False
    assert "baseline" in plus_10.note or "bound" in plus_10.note
    # -10%/-20% move strictly away from the bound and must be available.
    minus_10 = perturb.perturb_axis(G1_GENOME, "leg_fraction", -1)
    assert minus_10.available is True
    assert minus_10.gene_value == pytest.approx(0.45)


# ---------------------------------------------------------------------------
# 3) perturb.py -- ordered-categorical boundary handling.
# ---------------------------------------------------------------------------


def test_window_days_at_top_of_range_has_no_upward_neighbor() -> None:
    # G1.window_days == 14 == max(WINDOW_DAYS_CHOICES) -- +1/+2 step must not exist.
    plus_1 = perturb.perturb_axis(G1_GENOME, "window_days", 1)
    plus_2 = perturb.perturb_axis(G1_GENOME, "window_days", 2)
    assert plus_1.available is False
    assert plus_2.available is False
    minus_1 = perturb.perturb_axis(G1_GENOME, "window_days", -1)
    minus_2 = perturb.perturb_axis(G1_GENOME, "window_days", -2)
    assert minus_1.available and minus_1.gene_value == 10
    assert minus_2.available and minus_2.gene_value == 7


def test_top_k_pairs_at_bottom_of_range_has_no_downward_neighbor() -> None:
    # G1.top_k_pairs == 1 == min(TOP_K_PAIRS_CHOICES) -- -1/-2 step must not exist.
    minus_1 = perturb.perturb_axis(G1_GENOME, "top_k_pairs", -1)
    assert minus_1.available is False
    plus_1 = perturb.perturb_axis(G1_GENOME, "top_k_pairs", 1)
    plus_2 = perturb.perturb_axis(G1_GENOME, "top_k_pairs", 2)
    assert plus_1.available and plus_1.gene_value == 2
    assert plus_2.available and plus_2.gene_value == 3


def test_idle_mode_alternatives_are_the_three_other_choices() -> None:
    alternatives = perturb.axis_perturbations(G1_GENOME, "idle_mode")
    assert len(alternatives) == 3
    values = {point.gene_value for point in alternatives}
    assert values == {"none", "majors_low_thr", "tiered"}  # every IDLE_MODE_CHOICES entry except G1's own "usdt_lend"
    assert all(point.available for point in alternatives)


def test_grid_perturbations_rejects_idle_mode_and_covers_full_cartesian_product() -> None:
    with pytest.raises(ValueError):
        perturb.grid_perturbations(G1_GENOME, "idle_mode", "leg_fraction")
    cells = perturb.grid_perturbations(G1_GENOME, "entry_threshold_apr", "exit_threshold_ratio", tiers=(-1, 0, 1))
    assert len(cells) == 9  # 3x3 cartesian product
    center = [cell for cell in cells if cell.tier_a == 0 and cell.tier_b == 0][0]
    assert center.available and center.genome == G1_GENOME


# ---------------------------------------------------------------------------
# 4) evaluate.py -- metrics cache dedup.
# ---------------------------------------------------------------------------


def test_metrics_cache_dedups_identical_genomes(synthetic_cache: fitness.MarketCache) -> None:
    cache = MetricsCache()
    cache.get(SYNTH_I5, synthetic_cache)
    cache.get(SYNTH_I5, synthetic_cache)  # exact repeat -> must hit, not recompute
    slightly_different = Genome(**{**SYNTH_I5.to_dict(), "entry_threshold_apr": SYNTH_I5.entry_threshold_apr + 1e-12})
    cache.get(slightly_different, synthetic_cache)  # float-noise duplicate (genome_key rounds to 9dp) -> still a hit
    assert cache.misses == 1
    assert cache.hits == 2
    assert len(cache) == 1


# ---------------------------------------------------------------------------
# 5) rolling.py -- window boundaries and win-rate math.
# ---------------------------------------------------------------------------


def test_six_month_windows_are_consecutive_and_non_overlapping() -> None:
    start, end = pd.Timestamp("2020-01-01", tz="UTC"), pd.Timestamp("2021-01-15", tz="UTC")
    windows = rolling.six_month_windows(start, end)
    assert windows[0] == (pd.Timestamp("2020-01-01", tz="UTC"), pd.Timestamp("2020-07-01", tz="UTC"))
    for (_, end_a), (start_b, _) in zip(windows, windows[1:]):
        assert end_a == start_b  # half-open (start, end] windows tile the timeline with no gap/overlap
    assert windows[-1][1] >= end  # last boundary covers the requested end


def test_rolling_win_rate_on_synthetic_series_where_g1_always_wins() -> None:
    idx = pd.date_range("2020-01-01", periods=730, freq="D", tz="UTC")
    g1_equity = pd.Series(90.0 * np.cumprod(1.0 + np.full(730, 0.002)), index=idx)  # strong steady growth
    i5_equity = pd.Series(90.0 * np.cumprod(1.0 + np.full(730, 0.0002)), index=idx)  # weak steady growth
    result = rolling.run(g1_equity, i5_equity)
    assert result["g1_win_rate"] == pytest.approx(1.0)
    assert result["n_windows_counted"] >= 3


# ---------------------------------------------------------------------------
# 6) regime.py -- bucket assignment.
# ---------------------------------------------------------------------------


def test_regime_bucket_assignment_matches_task_spec() -> None:
    for year in (2020, 2021, 2024):
        assert regime._bucket_for(year) == "high_funding"
    for year in (2022, 2023, 2025, 2026):
        assert regime._bucket_for(year) == "low_funding"
    for year in (2019, 2027):
        assert regime._bucket_for(year) == "unclassified"


def test_regime_run_computes_bucket_means_from_yearly_returns() -> None:
    idx = pd.date_range("2020-01-01", periods=365 * 3, freq="D", tz="UTC")
    # G1 grows faster than I5 every year -> every G1-I5 delta is positive in every bucket present (2020, 2021 = high funding here).
    g1_equity = pd.Series(90.0 * np.cumprod(1.0 + np.full(len(idx), 0.0015)), index=idx)
    i5_equity = pd.Series(90.0 * np.cumprod(1.0 + np.full(len(idx), 0.0005)), index=idx)
    result = regime.run(g1_equity, i5_equity)
    assert result["high_funding"]["n_years"] >= 1
    assert result["high_funding"]["mean_g1_minus_i5_pp"] > 0.0


# ---------------------------------------------------------------------------
# 7) dsr.py -- trial-count bookkeeping and monotonic deflation.
# ---------------------------------------------------------------------------


def test_dsr_cumulative_trials_equals_121_plus_7500() -> None:
    from research.wave21_ga import gates21

    assert gates21.CUMULATIVE_TRIALS_WITH_GA == 121 + 7_500 == 7_621


def test_dsr_score_is_monotonically_non_increasing_in_trials() -> None:
    idx = pd.date_range("2020-01-01", periods=500, freq="D", tz="UTC")
    rng = np.random.default_rng(3)
    equity = pd.Series(90.0 * np.cumprod(1.0 + rng.normal(0.0008, 0.01, len(idx))), index=idx)
    low_trials = fitness.deflated_sharpe_for_trials(equity, 10)
    high_trials = fitness.deflated_sharpe_for_trials(equity, 7_621)
    assert low_trials is not None and high_trials is not None
    # more candidate trials -> a harder benchmark to beat -> DSR score can only fall or stay flat
    assert high_trials["score"] <= low_trials["score"] + 1e-9


# ---------------------------------------------------------------------------
# 8) attribution.py -- concentration rule on hand-built contribution dicts.
# ---------------------------------------------------------------------------


def test_concentration_flags_single_axis_dominance() -> None:
    result = attribution.concentration_from_contributions({"a": 9.0, "b": 0.5, "c": 0.5, "d": -3.0})
    assert result["concentrated_single_axis"] is True
    assert result["spread_across_2plus"] is False
    assert result["top_axis"] == "a"


def test_concentration_flags_even_spread() -> None:
    result = attribution.concentration_from_contributions({"a": 2.0, "b": 1.8, "c": 1.7, "d": -1.0})
    assert result["concentrated_single_axis"] is False
    assert result["spread_across_2plus"] is True


def test_concentration_handles_no_positive_contributions() -> None:
    result = attribution.concentration_from_contributions({"a": -1.0, "b": -2.0})
    assert result["top_axis"] is None
    assert result["concentrated_single_axis"] is None


def test_attribution_run_on_synthetic_cache_identifies_unchanged_axes(synthetic_cache: fitness.MarketCache) -> None:
    result = attribution.run(synthetic_cache, None, SYNTH_I5, SYNTH_G1)
    # SYNTH_I5 and SYNTH_G1 share leg_fraction=0.30 and top_k_pairs=1 by construction.
    assert set(result["unchanged_axes"]) == {"leg_fraction", "top_k_pairs"}
    assert set(result["changed_axes"]) == set(GENE_NAMES) - {"leg_fraction", "top_k_pairs"}


# ---------------------------------------------------------------------------
# 9) shuffle_control.py -- gross-feasibility rejection sampling.
# ---------------------------------------------------------------------------


def test_gross_feasible_draws_all_satisfy_1x_constraint_and_force_top_k_1() -> None:
    from research.wave10_carry100.engine import ACTIVE_CAPITAL
    from research.wave21_ga import gates21

    genomes, attempts = shuffle_control.draw_gross_feasible_genomes(30, seed=123)
    assert len(genomes) == 30
    assert attempts >= 30
    for genome in genomes:
        assert gates21.gross_usdt(genome) <= ACTIVE_CAPITAL + 1e-9
        assert genome.top_k_pairs == 1  # forced consequence of leg_fraction's own registered floor -- see module docstring


# ---------------------------------------------------------------------------
# 10) verdict.py -- FAIL-first precedence.
# ---------------------------------------------------------------------------


def _make_inputs(stability: float, rolling_rate: float, spread: bool, concentrated: bool, dsr_score: float, shuffle_top5: bool) -> tuple[dict, dict, dict, dict, dict, dict]:
    sensitivity_result = {"overall": {"primary_value": stability, "worst_axis": "entry_threshold_apr"}}
    rolling_result = {"g1_win_rate": rolling_rate}
    regime_result = {"dominant_regime_by_magnitude": "low_funding"}
    dsr_result = {"g1_dsr_score_cumulative": dsr_score, "g1_dsr_positive_at_cumulative_trials": dsr_score > 0.0}
    attribution_result = {"forward_concentration": {"spread_across_2plus": spread, "concentrated_single_axis": concentrated, "top_axis": "window_days", "top_share": 0.9 if concentrated else 0.3}, "methodology": {"concentration_threshold": 0.6}}
    shuffle_result = {"g1_in_top_5pct_full_cagr": shuffle_top5, "rank_by_full_cagr": {"g1_top_pct_of_pooled_31": 3.0 if shuffle_top5 else 40.0}}
    return sensitivity_result, rolling_result, regime_result, dsr_result, attribution_result, shuffle_result


def test_verdict_all_pass_conditions_met_yields_pass() -> None:
    inputs = _make_inputs(stability=0.9, rolling_rate=0.60, spread=True, concentrated=False, dsr_score=0.5, shuffle_top5=True)
    result = verdict.combine(*inputs)
    assert result["overall"] == "PASS"
    assert result["fail_reasons"] == []


def test_verdict_low_stability_forces_fail_even_if_other_checks_pass() -> None:
    inputs = _make_inputs(stability=0.4, rolling_rate=0.60, spread=True, concentrated=False, dsr_score=0.5, shuffle_top5=True)
    result = verdict.combine(*inputs)
    assert result["overall"] == "FAIL"
    assert any("stability_ratio" in reason for reason in result["fail_reasons"])


def test_verdict_single_axis_concentration_forces_fail() -> None:
    inputs = _make_inputs(stability=0.9, rolling_rate=0.60, spread=False, concentrated=True, dsr_score=0.5, shuffle_top5=True)
    result = verdict.combine(*inputs)
    assert result["overall"] == "FAIL"


def test_verdict_low_rolling_win_rate_forces_fail() -> None:
    inputs = _make_inputs(stability=0.9, rolling_rate=0.40, spread=True, concentrated=False, dsr_score=0.5, shuffle_top5=True)
    result = verdict.combine(*inputs)
    assert result["overall"] == "FAIL"


def test_verdict_partial_credit_yields_conditional_not_pass_or_fail() -> None:
    # stability sits in [0.6, 0.8) -- not FAIL-triggering, but also not PASS-qualifying.
    inputs = _make_inputs(stability=0.7, rolling_rate=0.60, spread=True, concentrated=False, dsr_score=0.5, shuffle_top5=True)
    result = verdict.combine(*inputs)
    assert result["overall"] == "CONDITIONAL"
    assert "stability_ratio_ge_0_8" in result["unmet_pass_checks"]


# ---------------------------------------------------------------------------
# 11) reporting22.py -- report writer runs end-to-end on minimal valid JSON without KeyErrors.
# ---------------------------------------------------------------------------


def test_write_wave22_report_smoke(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    sensitivity_axis = {
        "kind": "continuous", "baseline_value": 0.1, "points": [
            {"tier": -1, "tier_label": "-10%", "available": True, "gene_value": 0.09, "note": "", "full_cagr": 0.10, "oos_cagr": 0.03, "full_cagr_ratio": 0.9, "oos_cagr_ratio": 0.9, "gross_feasible_1x": True},
            {"tier": 1, "tier_label": "+10%", "available": False, "gene_value": None, "note": "boundary", "full_cagr": None, "oos_cagr": None, "full_cagr_ratio": None, "oos_cagr_ratio": None, "gross_feasible_1x": None},
        ],
        "n_available": 1, "n_total_tiers": 2, "at_range_boundary": True,
        "neighbor_avg_full_cagr": 0.10, "neighbor_avg_full_cagr_ratio": 0.9, "neighbor_min_full_cagr_ratio": 0.9,
        "neighbor_avg_oos_cagr_ratio": 0.9, "stable_mean_ratio_ge_0_8": True,
    }
    (results_dir / "sensitivity.json").write_text(__import__("json").dumps({
        "methodology": {}, "g1_full_cagr": 0.1235, "g1_oos_cagr_self_contained": 0.0404,
        "per_axis": {axis: sensitivity_axis for axis in GENE_NAMES},
        "overall": {"stability_ratio_min_of_axis_means": 0.85, "stability_ratio_mean_of_axis_means": 0.9, "worst_axis": "entry_threshold_apr", "n_axes_below_0_8": 1, "n_axes_below_0_6": 0, "primary_metric": "x", "primary_value": 0.85, "stable": True, "fail_threshold_0_6_breached": False},
        "grid": {"pairs": [{"axis_a": "entry_threshold_apr", "axis_b": "exit_threshold_ratio", "tiers": [-1, 0, 1], "cells": [{"tier_a": 0, "tier_b": 0, "available": True, "note": "", "full_cagr": 0.12, "full_cagr_ratio": 1.0, "gross_feasible_1x": True}], "n_available_cells": 1, "n_gross_infeasible_cells": 0, "min_full_cagr_ratio": 1.0, "mean_full_cagr_ratio": 1.0}]},
    }), encoding="utf-8")
    (results_dir / "rolling.json").write_text(__import__("json").dumps({
        "methodology": {}, "windows": [{"window_start": "2020-01-01", "window_end": "2020-07-01", "contains_oos": False, "fully_oos": False, "low_confidence": False, "g1_cagr": 0.1, "i5_cagr": 0.08, "g1_minus_i5_pp": 2.0, "g1_wins": True, "note": ""}],
        "n_windows_total": 1, "n_windows_counted": 1, "n_windows_low_confidence": 0, "g1_win_rate": 1.0, "g1_win_rate_pct": 100.0, "n_g1_wins": 1,
        "win_rate_pre_oos_windows": 1.0, "win_rate_oos_touching_windows": None, "n_pre_oos_windows": 1, "n_oos_touching_windows": 0,
        "win_rate_first_half_chronological": 1.0, "win_rate_second_half_chronological": 1.0,
        "streaks": {"longest_g1_win_streak": 1, "longest_i5_win_streak": 0}, "limitations": ["n small"],
    }), encoding="utf-8")
    (results_dir / "regime.json").write_text(__import__("json").dumps({
        "methodology": {"high_funding_years": [2020], "low_funding_years": [2022]},
        "by_year": [{"year": 2020, "bucket": "high_funding", "g1_cagr": 0.15, "i5_cagr": 0.10, "g1_minus_i5_pp": 5.0, "g1_wins": True, "is_partial_year": False, "straddles_oos_split": False}],
        "high_funding": {"years_expected": [2020], "years_present": [2020], "years_missing": [], "n_years": 1, "mean_g1_minus_i5_pp": 5.0, "median_g1_minus_i5_pp": 5.0, "g1_win_count": 1, "g1_win_rate": 1.0},
        "low_funding": {"years_expected": [2022], "years_present": [], "years_missing": [2022], "n_years": 0, "mean_g1_minus_i5_pp": None, "median_g1_minus_i5_pp": None, "g1_win_count": 0, "g1_win_rate": None},
        "dominant_regime_by_magnitude": "high_funding", "improvement_only_in_one_regime": "high_funding", "improvement_positive_in_both_regimes": False,
        "limitations": ["few years"],
    }), encoding="utf-8")
    (results_dir / "dsr.json").write_text(__import__("json").dumps({
        "methodology": {}, "g1_dsr_at_trials_this_wave_only": {"score": 0.3, "probability": 0.6, "trials": 7500, "observed_sharpe": 1.0},
        "g1_dsr_at_trials_cumulative": {"score": 0.25, "probability": 0.55, "trials": 7621, "observed_sharpe": 1.0},
        "g1_dsr_score_cumulative": 0.25, "g1_dsr_positive_at_cumulative_trials": True,
        "wave21_report_reference_ga_final_top_k3": {"trials_this_wave_only": 0.23594, "trials_cumulative_121_plus_7500": 0.23196},
        "ga_final_top_k3_cross_check_this_wave": None, "limitations": ["dsr caveat"],
    }), encoding="utf-8")
    attribution_axis_changed = {"i5_value": 0.15, "g1_value": 0.10, "unchanged": False, "forward_full_cagr": 0.11, "forward_contribution_pp": 1.0, "forward_oos_contribution_pp": 0.5, "backward_full_cagr": 0.12, "backward_contribution_pp": 0.8, "backward_oos_contribution_pp": 0.4}
    attribution_axis_unchanged = {"i5_value": 0.5, "g1_value": 0.5, "unchanged": True, "forward_full_cagr": 0.1, "forward_contribution_pp": 0.0, "forward_oos_contribution_pp": 0.0, "backward_full_cagr": 0.1, "backward_contribution_pp": 0.0, "backward_oos_contribution_pp": 0.0}
    (results_dir / "attribution.json").write_text(__import__("json").dumps({
        "methodology": {}, "i5_full_cagr": 0.1027, "g1_full_cagr": 0.1235, "total_gap_full_cagr_pp": 2.08, "total_gap_oos_cagr_pp": 0.98,
        "changed_axes": ["entry_threshold_apr", "exit_threshold_ratio", "window_days", "universe_breadth", "idle_mode"], "unchanged_axes": ["top_k_pairs", "leg_fraction"],
        "per_axis": {axis: (attribution_axis_unchanged if axis in ("top_k_pairs", "leg_fraction") else attribution_axis_changed) for axis in GENE_NAMES},
        "forward_concentration": {"positive_axes": ["window_days"], "shares": {"window_days": 1.0}, "top_axis": "window_days", "top_share": 0.5, "concentrated_single_axis": False, "spread_across_2plus": True, "n_axes_with_meaningful_share": 3},
        "backward_concentration": {"positive_axes": ["window_days"], "shares": {"window_days": 1.0}, "top_axis": "window_days", "top_share": 0.5, "concentrated_single_axis": False, "spread_across_2plus": True, "n_axes_with_meaningful_share": 3},
        "forward_and_backward_agree_on_concentration": True, "sum_of_forward_contributions_pp": 2.0, "interaction_residual_pp": 0.08,
        "limitations": ["one at a time caveat"],
    }), encoding="utf-8")
    (results_dir / "shuffle_control.json").write_text(__import__("json").dumps({
        "methodology": {"n_draws_attempted": 90, "n_draws_requested": 30}, "g1_full_cagr": 0.1235, "g1_oos_cagr_self_contained": 0.0404,
        "draws": [{"index": i, "genome": {}, "full_cagr": 0.05 + i * 0.001, "oos_cagr_self_contained": 0.01, "mdd_full": 0.05, "gross_usdt": 90.0} for i in range(30)],
        "rank_by_full_cagr": {"n_pool": 30, "n_random_below_g1": 29, "g1_beats_pct_of_random_pool": 96.7, "g1_rank_within_pooled_31": 1, "g1_top_pct_of_pooled_31": 3.2, "g1_in_top_5pct": True},
        "rank_by_oos_cagr": {"n_pool": 30, "n_random_below_g1": 20, "g1_beats_pct_of_random_pool": 66.7, "g1_rank_within_pooled_31": 8, "g1_top_pct_of_pooled_31": 25.8, "g1_in_top_5pct": False},
        "g1_in_top_5pct_full_cagr": True, "g1_in_top_5pct_oos_cagr": False, "limitations": ["n=30 is small"],
    }), encoding="utf-8")
    (results_dir / "verdict.json").write_text(__import__("json").dumps({
        "methodology": {}, "inputs": {"stability_ratio": 0.85, "rolling_win_rate": 1.0, "spread_across_2plus_axes": True, "concentrated_single_axis": False, "concentration_top_axis": "window_days", "dsr_score_cumulative": 0.25, "dsr_positive": True, "shuffle_top5pct_full_cagr": True, "shuffle_g1_top_pct": 3.2, "dominant_regime": "high_funding"},
        "fail_reasons": [], "pass_checks": {"stability_ratio_ge_0_8": True, "rolling_win_rate_ge_55pct": True, "spread_across_2plus_axes": True, "dsr_positive": True, "shuffle_control_top5pct": True},
        "unmet_pass_checks": [], "overall": "PASS", "recommendation": "test recommendation text",
    }), encoding="utf-8")

    report_dir = tmp_path / "report"
    registry_path = tmp_path / "REGISTRY.md"
    reporting22.write_wave22_report(results_dir, report_dir, registry_path)

    report_text = (report_dir / "wave22_report.md").read_text(encoding="utf-8")
    assert "종합판정: PASS" in report_text
    assert registry_path.exists()
    assert "PASS" in registry_path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 12) Real-cache integration check (slow; self-skips if the repo cache is unavailable).
# ---------------------------------------------------------------------------


def test_g1_reproduces_strategy_card_reference_numbers_on_real_cache() -> None:
    universe_cache_marker = Path(__file__).resolve().parents[3] / "research" / "wave12_frontier"
    if not universe_cache_marker.exists():
        pytest.skip("research/wave12_frontier cache not present -- cannot build the real market cache")
    try:
        cache = fitness.build_market_cache()
    except Exception as error:  # pragma: no cover - environment-dependent
        pytest.skip(f"real market cache unavailable: {error}")

    g1_equity = fitness.run_backtest(G1_GENOME, cache, fitness.MODE_OOS_FINAL)
    full_cagr = fitness.cagr(g1_equity)
    oos_cagr = fitness.cagr(fitness.oos_slice(g1_equity, fitness.MODE_OOS_FINAL))

    assert full_cagr == pytest.approx(G1_REFERENCE_METRICS["full_period_cagr"], abs=0.002)
    assert oos_cagr == pytest.approx(G1_REFERENCE_METRICS["oos_cagr_self_contained"], abs=0.002)
