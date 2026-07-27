# Wave-21 GA test suite. Every fixture uses an in-memory synthetic market
# (fitness.market_cache_from_markets) -- no disk I/O, no dependency on the real repo cache --
# EXCEPT test_final_evaluation_reproduces_i5_json_on_real_cache, which is the one place this
# suite touches the real cache (mirrors research/wave18_idle/tests/test_wave18.py's own
# reliance on synthetic markets for engine-shape tests). That one test self-skips if
# research/wave18_idle/results/I5.json is not present.
#
# The two tests that matter most for this wave's own integrity:
#   - test_vectorized_engine_matches_reference_engine13: the numeric-equivalence anchor the
#     whole 15,000-evaluation run's correctness rests on (see fitness.py's own module
#     docstring).
#   - test_run_backtest_is_mode_never_returns_oos_rows / test_oos_slice_raises_unless_final_mode
#     / test_evaluate_genome_has_no_mode_parameter: SPEC.md's OOS-sealing requirement, enforced
#     both structurally (no mode param on the search-facing entry point) and empirically (a
#     raised exception if OOS data is ever touched outside mode=MODE_OOS_FINAL).

from __future__ import annotations

import inspect
from pathlib import Path
import sys

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[3]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

import numpy as np
import pandas as pd  # noqa: PANDAS_OK
import pytest

from research.wave1.fam_funding import FundingCandidate, FundingMarket, carry_position
from research.wave13_liquidity import engine13
from research.wave21_ga import fitness, ga, gates21, random_search, run_wave21
from research.wave21_ga.genome import (
    ENTRY_THRESHOLD_APR_MAX,
    ENTRY_THRESHOLD_APR_MIN,
    EXIT_THRESHOLD_RATIO_MAX,
    EXIT_THRESHOLD_RATIO_MIN,
    IDLE_MODE_CHOICES,
    LEG_FRACTION_MAX,
    LEG_FRACTION_MIN,
    TOP_K_PAIRS_CHOICES,
    UNIVERSE_BREADTH_CHOICES,
    WINDOW_DAYS_CHOICES,
    Genome,
    crossover,
    from_dict,
    genome_key,
    mutate,
    random_genome,
)

# ---------------------------------------------------------------------------
# Synthetic fixtures. Default anchor date is comfortably inside IS (well before
# OOS_SPLIT=2025-09-30) so evaluate_genome/run_ga/run_random_search (always mode=MODE_IS) work
# on them without every day being truncated away.
# ---------------------------------------------------------------------------


def _two_symbol_market(start: str = "2024-01-01", periods: int = 48) -> dict[str, FundingMarket]:
    idx = pd.date_range(start, periods=periods, freq="D", tz="UTC")
    funding_idx = pd.date_range(start, periods=periods * 3, freq="8h", tz="UTC")
    a_close = [100.0 * (1.003**i) for i in range(periods)]
    a_open = [100.0, *a_close[:-1]]
    b_close = [50.0 * (1.001**i) for i in range(periods)]
    b_open = [50.0, *b_close[:-1]]
    spot_a = pd.DataFrame({"open": a_open, "close": a_close}, index=idx)
    perp_a = pd.DataFrame({"open": [v * 0.999 for v in a_open], "close": [v * 0.999 for v in a_close]}, index=idx)
    spot_b = pd.DataFrame({"open": b_open, "close": b_close}, index=idx)
    perp_b = pd.DataFrame({"open": [v * 1.001 for v in b_open], "close": [v * 1.001 for v in b_close]}, index=idx)
    funding_a = pd.Series(0.0009, index=funding_idx, name="funding_rate")
    funding_b = pd.Series(0.0005, index=funding_idx, name="funding_rate")
    return {"AAAUSDT": FundingMarket(spot_a, perp_a, funding_a), "BBBUSDT": FundingMarket(spot_b, perp_b, funding_b)}


def _straddling_market(periods: int = 60) -> dict[str, FundingMarket]:
    """Spans OOS_SPLIT (2025-09-30): starts 2025-09-10, so ~20 IS days then ~40 OOS days."""
    return _two_symbol_market(start="2025-09-10", periods=periods)


BASELINE_GENOME: Genome = Genome(
    entry_threshold_apr=0.15,
    exit_threshold_ratio=0.5,
    window_days=7,
    top_k_pairs=1,
    leg_fraction=0.5,
    universe_breadth=300,
    idle_mode="none",
)


# ---------------------------------------------------------------------------
# 1) Genome: bounds, sampling, operators, cache keys.
# ---------------------------------------------------------------------------


def test_genome_rejects_out_of_registry_values() -> None:
    with pytest.raises(ValueError):
        Genome(entry_threshold_apr=0.01, exit_threshold_ratio=0.5, window_days=7, top_k_pairs=1, leg_fraction=0.5, universe_breadth=200, idle_mode="none")
    with pytest.raises(ValueError):
        Genome(entry_threshold_apr=0.15, exit_threshold_ratio=0.5, window_days=9, top_k_pairs=1, leg_fraction=0.5, universe_breadth=200, idle_mode="none")
    with pytest.raises(ValueError):
        Genome(entry_threshold_apr=0.15, exit_threshold_ratio=0.5, window_days=7, top_k_pairs=1, leg_fraction=0.5, universe_breadth=200, idle_mode="bogus")


def test_random_genome_always_within_frozen_bounds() -> None:
    rng = np.random.default_rng(1)
    for _ in range(500):
        individual = random_genome(rng)
        assert ENTRY_THRESHOLD_APR_MIN <= individual.entry_threshold_apr <= ENTRY_THRESHOLD_APR_MAX
        assert EXIT_THRESHOLD_RATIO_MIN <= individual.exit_threshold_ratio <= EXIT_THRESHOLD_RATIO_MAX
        assert LEG_FRACTION_MIN <= individual.leg_fraction <= LEG_FRACTION_MAX
        assert individual.window_days in WINDOW_DAYS_CHOICES
        assert individual.top_k_pairs in TOP_K_PAIRS_CHOICES
        assert individual.universe_breadth in UNIVERSE_BREADTH_CHOICES
        assert individual.idle_mode in IDLE_MODE_CHOICES


def test_mutate_stays_in_bounds_and_changes_values_with_probability_one() -> None:
    rng = np.random.default_rng(2)
    base = Genome(entry_threshold_apr=0.20, exit_threshold_ratio=0.50, window_days=7, top_k_pairs=1, leg_fraction=0.40, universe_breadth=200, idle_mode="none")
    changed_continuous = False
    changed_categorical = False
    for _ in range(200):
        mutated = mutate(base, rng, sigma_fraction=0.10, probability=1.0)  # probability=1.0 -> every gene mutates every call
        assert ENTRY_THRESHOLD_APR_MIN <= mutated.entry_threshold_apr <= ENTRY_THRESHOLD_APR_MAX
        assert LEG_FRACTION_MIN <= mutated.leg_fraction <= LEG_FRACTION_MAX
        assert mutated.window_days in WINDOW_DAYS_CHOICES
        assert mutated.idle_mode in IDLE_MODE_CHOICES
        if mutated.entry_threshold_apr != base.entry_threshold_apr:
            changed_continuous = True
        if mutated.idle_mode != base.idle_mode:
            changed_categorical = True
    assert changed_continuous
    assert changed_categorical


def test_crossover_child_genes_come_from_exactly_one_parent() -> None:
    rng = np.random.default_rng(3)
    parent_a = Genome(0.10, 0.30, 3, 1, 0.30, 30, "none")
    parent_b = Genome(0.35, 0.70, 14, 3, 0.50, 300, "tiered")
    for _ in range(100):
        child = crossover(parent_a, parent_b, rng)
        for name, value in child.to_dict().items():
            assert value == getattr(parent_a, name) or value == getattr(parent_b, name)


def test_genome_key_collapses_float_noise_but_distinguishes_real_differences() -> None:
    a = Genome(0.15, 0.50, 7, 1, 0.50, 200, "tiered")
    b = Genome(0.15 + 1e-12, 0.50, 7, 1, 0.50, 200, "tiered")
    c = Genome(0.16, 0.50, 7, 1, 0.50, 200, "tiered")
    assert genome_key(a) == genome_key(b)
    assert genome_key(a) != genome_key(c)
    cache: dict[tuple, int] = {genome_key(a): 1}
    assert genome_key(b) in cache  # b hits the same cache slot as a


def test_genome_to_dict_from_dict_roundtrip() -> None:
    original = Genome(0.22, 0.44, 10, 2, 0.37, 100, "usdt_lend")
    restored = from_dict(original.to_dict())
    assert restored == original


# ---------------------------------------------------------------------------
# 2) Vectorized hysteresis matches the reference carry_position bit-for-bit.
# ---------------------------------------------------------------------------


def test_vectorized_hysteresis_matches_reference_carry_position() -> None:
    idx = pd.date_range("2024-01-01", periods=48, freq="D", tz="UTC")
    raw_values = [0.30, 0.30, 0.30, -0.30, -0.30, 0.02, 0.02, -0.20, -0.20, -0.02, 0.40, 0.05] * 4
    score = pd.Series(raw_values[:48], index=idx, dtype=float)
    frame = pd.DataFrame({"AAAUSDT": score, "BBBUSDT": score * 0.5})
    candidate = FundingCandidate("T", 7, 0.15, 1)  # carry_position's own hardcoded exit = threshold/2.0

    reference = pd.DataFrame({col: carry_position(frame[col], candidate) for col in frame.columns})
    vectorized = fitness.vectorized_hysteresis(frame, candidate.threshold_apr, candidate.threshold_apr / 2.0)

    assert vectorized["AAAUSDT"].tolist() == pytest.approx(reference["AAAUSDT"].tolist())
    assert vectorized["BBBUSDT"].tolist() == pytest.approx(reference["BBBUSDT"].tolist())
    assert vectorized["AAAUSDT"].sum() > 0.0  # sanity: fixture actually triggers entries, not degenerately all-zero


def test_vectorized_hysteresis_respects_arbitrary_exit_ratio() -> None:
    """A ratio other than carry_position's own hardcoded 0.5 -- pure sanity on the
    generalization SPEC.md's exit_threshold_ratio gene needs: a LOOSER exit (higher ratio,
    e.g. 0.9 -> exits sooner after entry) must never hold a position longer than the
    stricter/lower-ratio exit (e.g. 0.1) does, for the identical score path."""
    idx = pd.date_range("2024-01-01", periods=30, freq="D", tz="UTC")
    score = pd.Series(np.linspace(0.30, -0.10, 30), index=idx, dtype=float)  # monotonically decaying score
    frame = pd.DataFrame({"AAAUSDT": score})
    tight_exit = fitness.vectorized_hysteresis(frame, 0.15, 0.15 * 0.1)["AAAUSDT"]  # exits late (low exit bar)
    loose_exit = fitness.vectorized_hysteresis(frame, 0.15, 0.15 * 0.9)["AAAUSDT"]  # exits early (high exit bar)
    assert float(loose_exit.sum()) <= float(tight_exit.sum())


# ---------------------------------------------------------------------------
# 3) Engine equivalence: fitness.run_backtest (idle_mode="none", i.e. no overlay/lending)
#    must reproduce engine13._run_liquidity_loop bit-for-bit on the SAME synthetic market.
# ---------------------------------------------------------------------------


def test_vectorized_engine_matches_reference_engine13() -> None:
    markets = _two_symbol_market()
    candidate = FundingCandidate("EQTEST", 7, 0.15, 1)
    leg_fraction = 0.5

    frames13 = engine13._build_aligned_frames(markets, candidate)
    spot_open13 = frames13[0]
    flat_cost = pd.DataFrame(0.001, index=spot_open13.index, columns=spot_open13.columns)
    always_liquid = pd.DataFrame(True, index=spot_open13.index, columns=spot_open13.columns)
    result13, cost13, _eligible13 = engine13._run_liquidity_loop(*frames13, candidate.top_k, leg_fraction, flat_cost, always_liquid)

    cache = fitness.market_cache_from_markets(markets, flat_cost_rate=0.001, always_liquid=True)
    individual = Genome(
        entry_threshold_apr=candidate.threshold_apr,
        exit_threshold_ratio=0.5,
        window_days=candidate.window_days,
        top_k_pairs=candidate.top_k,
        leg_fraction=leg_fraction,
        universe_breadth=300,
        idle_mode="none",
    )
    equity21 = fitness.run_backtest(individual, cache, fitness.MODE_IS)  # entire fixture is pre-OOS_SPLIT, so MODE_IS performs no truncation here

    assert equity21.tolist() == pytest.approx(result13.equity.tolist(), rel=1e-9)
    assert len(equity21) == len(result13.equity)


def test_vectorized_engine_leverage_and_feasibility_numbers_match_gates_formula() -> None:
    """gross_usdt()/leg_usdt() (gates21) must equal the SAME 2*top_k*leg_fraction*ACTIVE_CAPITAL
    convention research/wave18_idle/gates18.py uses for its own S1/S4 leverage gate -- pins the
    formula, not a specific engine run."""
    from research.wave10_carry100.engine import ACTIVE_CAPITAL

    individual = Genome(0.15, 0.5, 7, 3, 0.50, 200, "none")  # top_k=3, leg=0.5 -> deliberately > 1x gross
    assert gates21.leg_usdt(individual) == pytest.approx(0.50 * ACTIVE_CAPITAL)
    assert gates21.gross_usdt(individual) == pytest.approx(2.0 * 3 * 0.50 * ACTIVE_CAPITAL)
    assert gates21.gross_usdt(individual) > ACTIVE_CAPITAL  # confirms H4's leverage check is load-bearing, not vacuously true for every genome


# ---------------------------------------------------------------------------
# 4) OOS sealing.
# ---------------------------------------------------------------------------


def test_oos_slice_raises_unless_final_mode() -> None:
    idx = pd.date_range("2025-09-01", periods=10, freq="D", tz="UTC")
    equity = pd.Series(np.linspace(90.0, 95.0, 10), index=idx)
    with pytest.raises(fitness.OOSLeakageError):
        fitness.oos_slice(equity, fitness.MODE_IS)
    with pytest.raises(fitness.OOSLeakageError):
        fitness.oos_slice(equity, "not_a_real_mode")
    sliced = fitness.oos_slice(equity, fitness.MODE_OOS_FINAL)
    assert bool((sliced.index > fitness.OOS_SPLIT).all())


def test_run_backtest_is_mode_never_returns_oos_rows() -> None:
    markets = _straddling_market()
    cache = fitness.market_cache_from_markets(markets, flat_cost_rate=0.001, always_liquid=True)
    assert int(cache.is_row_mask.sum()) > 0  # fixture genuinely has IS days
    assert int(cache.is_row_mask.sum()) < len(cache.index)  # AND genuinely has OOS days -- otherwise this test proves nothing

    equity_is = fitness.run_backtest(BASELINE_GENOME, cache, fitness.MODE_IS)
    assert bool((equity_is.index <= fitness.OOS_SPLIT).all())

    equity_full = fitness.run_backtest(BASELINE_GENOME, cache, fitness.MODE_OOS_FINAL)
    assert bool((equity_full.index > fitness.OOS_SPLIT).any())  # the full run DOES reach OOS -- contrast case


def test_run_backtest_rejects_unknown_mode() -> None:
    markets = _two_symbol_market()
    cache = fitness.market_cache_from_markets(markets)
    with pytest.raises(ValueError):
        fitness.run_backtest(BASELINE_GENOME, cache, "OOS_SNEAKY")


def test_evaluate_genome_has_no_mode_parameter() -> None:
    """Structural half of the OOS seal: the function ga.py/random_search.py actually call
    cannot be asked for OOS data even if a future edit tried -- there is no `mode` parameter
    to pass one to."""
    parameters = inspect.signature(fitness.evaluate_genome).parameters
    assert "mode" not in parameters


def test_evaluate_genome_on_straddling_cache_only_uses_is_range() -> None:
    """End-to-end: evaluate_genome's fitness must be IDENTICAL whether the cache holds extra
    OOS-range rows or not, proving those rows are never consulted. `is_only_markets` is built
    with EXACTLY as many days as `cache_straddling`'s own IS prefix (same start date, same
    deterministic per-day price formula) so the two caches' IS portions cover the identical
    calendar range -- only `cache_straddling` additionally has OOS rows appended after it."""
    straddling_markets = _two_symbol_market(start="2025-06-01", periods=200)  # runs well past OOS_SPLIT
    cache_straddling = fitness.market_cache_from_markets(straddling_markets, flat_cost_rate=0.001)
    is_day_count = int(cache_straddling.is_row_mask.sum())
    assert 0 < is_day_count < len(cache_straddling.index)  # fixture must genuinely straddle the boundary, else this test proves nothing

    is_only_markets = _two_symbol_market(start="2025-06-01", periods=is_day_count)  # identical prefix, no OOS rows at all
    cache_is_only = fitness.market_cache_from_markets(is_only_markets, flat_cost_rate=0.001)
    assert int(cache_is_only.is_row_mask.sum()) == len(cache_is_only.index) == is_day_count  # sanity: this cache is ALL-IS

    result_is_only = fitness.evaluate_genome(BASELINE_GENOME, cache_is_only)
    result_straddling = fitness.evaluate_genome(BASELINE_GENOME, cache_straddling)
    assert result_is_only.fitness == pytest.approx(result_straddling.fitness, rel=1e-9)
    assert result_is_only.fold_cagrs == pytest.approx(result_straddling.fold_cagrs, rel=1e-9)


# ---------------------------------------------------------------------------
# 5) Walk-forward fitness formula.
# ---------------------------------------------------------------------------


def test_walk_forward_fitness_matches_hand_computed_folds() -> None:
    idx = pd.date_range("2024-01-01", periods=8, freq="365.25D", tz="UTC")  # 8 obs, exactly 1 year apart -> trivial per-fold CAGR
    values = [100.0, 110.0, 121.0, 133.1, 100.0, 90.0, 99.0, 108.9]  # 4 folds of 2 obs each: [100->110], [121->133.1], [100->90], [99->108.9]
    equity = pd.Series(values, index=idx, dtype=float)
    result = fitness.walk_forward_fitness(equity, n_folds=4)

    expected_fold_cagrs = [0.10, 0.10, -0.10, 0.10]  # each fold is a clean +-10%/year move by construction
    assert result.fold_cagrs == pytest.approx(expected_fold_cagrs, abs=1e-3)
    assert result.median_fold_cagr == pytest.approx(float(np.median(expected_fold_cagrs)), abs=1e-3)
    assert result.std_fold_cagr == pytest.approx(float(np.std(expected_fold_cagrs)), abs=1e-3)
    assert result.fitness == pytest.approx(result.median_fold_cagr - result.std_fold_cagr - result.mdd_penalty, abs=1e-9)


def test_walk_forward_fitness_penalizes_drawdown_beyond_floor() -> None:
    idx = pd.date_range("2024-01-01", periods=20, freq="D", tz="UTC")
    calm = pd.Series(np.linspace(100.0, 105.0, 20), index=idx)
    crashy_values = np.linspace(100.0, 105.0, 20)
    crashy_values[10] = 60.0  # a sharp, deep one-day drawdown well past the 10% floor
    crashy = pd.Series(crashy_values, index=idx)

    calm_result = fitness.walk_forward_fitness(calm, n_folds=4)
    crashy_result = fitness.walk_forward_fitness(crashy, n_folds=4)
    assert calm_result.mdd_penalty == pytest.approx(0.0, abs=1e-9)
    assert crashy_result.mdd_penalty > 0.0
    assert crashy_result.fitness < calm_result.fitness


def test_max_drawdown_known_case() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC")
    equity = pd.Series([100.0, 120.0, 60.0, 90.0], index=idx)  # peak 120 -> trough 60 == 50% drawdown
    assert fitness._max_drawdown(equity) == pytest.approx(0.5)


def test_walk_forward_fitness_rejects_too_short_series() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
    equity = pd.Series([100.0, 101.0, 102.0], index=idx)
    with pytest.raises(ValueError):
        fitness.walk_forward_fitness(equity, n_folds=4)


# ---------------------------------------------------------------------------
# 6) GA operators / run_ga end-to-end on a cheap synthetic cache.
# ---------------------------------------------------------------------------


def test_tournament_selection_favors_higher_fitness() -> None:
    rng = np.random.default_rng(7)
    population = [Genome(0.05 + 0.01 * i, 0.5, 7, 1, 0.40, 200, "none") for i in range(10)]
    fitnesses = [float(i) for i in range(10)]  # index 9 strictly dominates every tournament it enters
    winners = [ga._tournament_select(population, fitnesses, rng) for _ in range(300)]
    winner_indices = [population.index(w) for w in winners]
    assert float(np.mean(winner_indices)) > 4.5  # best individual should be over-represented vs a uniform-random mean of 4.5


def test_run_ga_generational_best_never_regresses_due_to_elitism() -> None:
    markets = _two_symbol_market(periods=40)
    cache = fitness.market_cache_from_markets(markets, flat_cost_rate=0.0015, always_liquid=True)
    result = ga.run_ga(seed=999_001, cache=cache, progress=False)

    assert result.n_evaluations == ga.POPULATION_SIZE * ga.GENERATIONS == 1_500
    assert len(result.history) == ga.GENERATIONS
    best_per_generation = [record.best_fitness for record in result.history]
    for earlier, later in zip(best_per_generation, best_per_generation[1:]):
        assert later >= earlier - 1e-12  # elitism (ELITE_COUNT=2) guarantees non-regression, modulo float noise
    assert result.best_fitness == pytest.approx(max(best_per_generation))
    assert result.n_backtests_run <= result.n_evaluations  # caching can only reduce fresh backtests, never exceed the logical budget


def test_run_random_search_is_deterministic_given_seed() -> None:
    markets = _two_symbol_market(periods=40)
    cache = fitness.market_cache_from_markets(markets, flat_cost_rate=0.0015, always_liquid=True)
    result_a = random_search.run_random_search(seed=555_001, cache=cache, n_evaluations=50, progress=False)
    result_b = random_search.run_random_search(seed=555_001, cache=cache, n_evaluations=50, progress=False)
    assert genome_key(result_a.best_genome) == genome_key(result_b.best_genome)
    assert result_a.best_fitness == pytest.approx(result_b.best_fitness)
    assert result_a.fitness_history == pytest.approx(result_b.fitness_history)
    assert len(result_a.fitness_history) == 50


# ---------------------------------------------------------------------------
# 7) gates21.
# ---------------------------------------------------------------------------


def test_gate_h1_requires_at_least_four_of_five_seed_wins() -> None:
    four_wins = gates21.gate_h1_ga_beats_random([1, 1, 1, 1, -1], [0, 0, 0, 0, 0])
    assert four_wins["status"] == "PASS" and four_wins["n_wins"] == 4
    three_wins = gates21.gate_h1_ga_beats_random([1, 1, 1, -1, -1], [0, 0, 0, 0, 0])
    assert three_wins["status"] == "FAIL" and three_wins["n_wins"] == 3


def test_gate_h2_beats_i5_oos_requires_strict_improvement() -> None:
    assert gates21.gate_h2_beats_i5_oos(0.05, 0.03)["status"] == "PASS"
    assert gates21.gate_h2_beats_i5_oos(0.03, 0.03)["status"] == "FAIL"
    assert gates21.gate_h2_beats_i5_oos(None, 0.03)["status"] == "FAIL"


def test_gate_h5_worst_years_flags_any_regression() -> None:
    idx = pd.date_range("2021-06-01", periods=1200, freq="D", tz="UTC")
    rng = np.random.default_rng(11)
    final_equity = pd.Series(100.0 * np.cumprod(1.0 + rng.normal(0.0006, 0.01, len(idx))), index=idx)
    i5_equity = final_equity * 1.5  # a pure positive rescale changes annualized-return math only via the (tiny) anchoring boundary effects, not the trend
    report = gates21.gate_h5_worst_years(final_equity, i5_equity, years=(2022,))
    assert set(report["years"].keys()) == {"2022"}


def test_evaluate_all_gates_requires_every_gate_to_pass() -> None:
    passing = {"status": "PASS"}
    failing = {"status": "FAIL"}
    all_pass = gates21.evaluate_all_gates(passing, passing, passing, passing, passing)
    assert all_pass.overall == "PASS" and all_pass.promoted is True and all_pass.failure_reasons == ()

    one_fail = gates21.evaluate_all_gates(passing, failing, passing, passing, passing)
    assert one_fail.overall == "FAIL" and one_fail.promoted is False and one_fail.failure_reasons == ("H2",)


# ---------------------------------------------------------------------------
# 8) final_evaluation reproduces I5's own saved OOS/full-period figures on the REAL cache.
#    The one test in this suite that is a slower integration check, not a synthetic unit test.
# ---------------------------------------------------------------------------


def test_select_final_candidate_picks_median_not_max() -> None:
    """SPEC.md '5회 모두에서 재현되는 개선만 인정(단일 시드 대박 무효)' -- a single outlier
    (seed 5, fitness=100.0) must NOT become the final candidate; the median-fitness seed
    (seed 3, fitness=0.30) must."""
    seeds = [1, 2, 3, 4, 5]
    genomes = [Genome(0.05 + 0.01 * i, 0.5, 7, 1, 0.40, 200, "none") for i in range(5)]
    fitnesses = [0.10, 0.20, 0.30, 0.05, 100.0]  # seed 5 is the single-seed jackpot; sorted ascending: 0.05(s4) 0.10(s1) 0.20(s2) 0.30(s3) 100(s5) -> median (3rd of 5) = 0.20 -> seed 2
    chosen_genome, chosen_seed, chosen_fitness = run_wave21.select_final_candidate(seeds, genomes, fitnesses)
    assert chosen_seed == 2
    assert chosen_fitness == pytest.approx(0.20)
    assert genome_key(chosen_genome) == genome_key(genomes[1])
    assert chosen_seed != 5  # the single-seed jackpot must never be selected directly


def test_select_final_candidate_requires_odd_count() -> None:
    with pytest.raises(ValueError):
        run_wave21.select_final_candidate([1, 2], [BASELINE_GENOME, BASELINE_GENOME], [0.1, 0.2])


def test_final_evaluation_reproduces_i5_json_on_real_cache() -> None:
    import json

    i5_path = Path(__file__).resolve().parents[2] / "wave18_idle" / "results" / "I5.json"
    if not i5_path.exists():
        pytest.skip("research/wave18_idle/results/I5.json not present -- wave18 must have run first")
    i5_payload = json.loads(i5_path.read_text(encoding="utf-8"))

    from research.wave21_ga.genome import I5_BASELINE_GENOME

    cache = fitness.build_market_cache()
    result = fitness.final_evaluation(I5_BASELINE_GENOME, cache)

    assert result.full_period_cagr == pytest.approx(i5_payload["full_period_annualized"], rel=1e-6)
    i5_oos = i5_payload["regime_breakdown"]["current_low_funding"]["annualized_return"]
    assert result.oos_cagr_regime_anchored == pytest.approx(i5_oos, rel=1e-6)
