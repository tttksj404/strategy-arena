# Wave-23 GA test suite. Every fixture uses an in-memory synthetic market
# (engine23.market_cache_from_markets) -- no disk I/O -- EXCEPT
# test_build_market_cache_real_cache_smoke, which touches the real repo cache and self-skips if
# it is not present (mirrors research/wave21_ga/tests/test_wave21.py's own convention).
#
# The tests that matter most for this wave's own integrity:
#   - test_holding_days_gives_exact_hold_length / test_no_same_day_rollover_reentry: pins the
#     lifecycle simulator's core promise (SPEC.md "보유기간 <=14일 강제") against the exact
#     off-by-one and same-day-rollover bugs this module's own development caught and fixed --
#     see engine23._simulate_lifecycle's docstring.
#   - test_normalized_weight_never_exceeds_1x_gross: SPEC.md "레버리지는 1x 고정", enforced
#     structurally (genome23.Genome.normalized_weight), exhaustively over the whole gene grid.
#   - test_run_backtest_is_mode_never_returns_oos_rows / test_oos_slice_raises_unless_final_mode
#     / test_evaluate_genome_has_no_mode_parameter: the OOS-sealing requirement.
#   - test_gate_k3_dsr_uses_the_passed_equity_series_only: pins that K3 always scores the
#     genome/equity pair it is GIVEN -- the exact class of mistake wave21_ga made (reporting a
#     different individual's DSR than the one actually gated) is structurally impossible here
#     because gate_k3_dsr takes no genome selection logic of its own, only a single equity
#     series argument.

from __future__ import annotations

from pathlib import Path
import sys

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[3]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

import inspect

import numpy as np
import pandas as pd  # noqa: PANDAS_OK
import pytest

from research.wave1.fam_funding import FundingMarket
from research.wave23_ga_short import engine23, fitness23, ga23, gates23, genome23, random_search23, run_wave23
from research.wave23_ga_short.genome23 import (
    ENTRY_Z_MAX,
    ENTRY_Z_MIN,
    HOLDING_DAYS_CHOICES,
    MAX_CONCURRENT_CHOICES,
    POSITION_FRACTION_MAX,
    POSITION_FRACTION_MIN,
    STOP_LOSS_PCT_CHOICES,
    STRATEGY_KIND_CHOICES,
    TAKE_PROFIT_PCT_CHOICES,
    UNIVERSE_BREADTH_CHOICES,
    Genome,
    crossover,
    from_dict,
    genome_key,
    mutate,
    random_genome,
)

# ---------------------------------------------------------------------------
# Synthetic fixtures.
# ---------------------------------------------------------------------------


def _two_symbol_market(start: str = "2024-01-01", periods: int = 400, seed: int = 0) -> dict[str, FundingMarket]:
    idx = pd.date_range(start, periods=periods, freq="D", tz="UTC")
    funding_idx = pd.date_range(start, periods=periods * 3, freq="8h", tz="UTC")
    rng = np.random.default_rng(seed)
    a_close = 100.0 * np.cumprod(1.0 + rng.normal(0.0015, 0.02, periods))
    b_close = 50.0 * np.cumprod(1.0 + rng.normal(0.0005, 0.015, periods))
    a_open = np.concatenate([[100.0], a_close[:-1]])
    b_open = np.concatenate([[50.0], b_close[:-1]])
    a_high, a_low = np.maximum(a_open, a_close) * 1.01, np.minimum(a_open, a_close) * 0.99
    b_high, b_low = np.maximum(b_open, b_close) * 1.01, np.minimum(b_open, b_close) * 0.99
    spot_a = pd.DataFrame({"open": a_open, "close": a_close}, index=idx)
    perp_a = pd.DataFrame({"open": a_open * 0.999, "high": a_high, "low": a_low, "close": a_close * 0.999}, index=idx)
    spot_b = pd.DataFrame({"open": b_open, "close": b_close}, index=idx)
    perp_b = pd.DataFrame({"open": b_open * 1.001, "high": b_high, "low": b_low, "close": b_close * 1.001}, index=idx)
    funding_a = pd.Series(rng.normal(0.0001, 0.0003, periods * 3), index=funding_idx, name="funding_rate")
    funding_b = pd.Series(rng.normal(0.00005, 0.0002, periods * 3), index=funding_idx, name="funding_rate")
    return {"AAAUSDT": FundingMarket(spot_a, perp_a, funding_a), "BBBUSDT": FundingMarket(spot_b, perp_b, funding_b)}


def _straddling_market(periods: int = 200) -> dict[str, FundingMarket]:
    """Spans OOS_SPLIT (2025-09-30) with >= engine23._MIN_CUTOFF_DAYS (120) days on the IS
    side -- starting date is chosen so ~122 days fall on-or-before OOS_SPLIT, then `periods`
    extends comfortably past it too."""
    return _two_symbol_market(start="2025-06-01", periods=periods)


BASELINE_GENOME: Genome = Genome(
    strategy_kind="momentum", entry_z=1.0, holding_days=5, position_fraction=0.5,
    stop_loss_pct=None, take_profit_pct=None, universe_breadth=100, max_concurrent=1,
)


# ---------------------------------------------------------------------------
# 1) Genome: bounds, sampling, operators, cache keys, leverage cap.
# ---------------------------------------------------------------------------


def test_genome_rejects_out_of_registry_values() -> None:
    with pytest.raises(ValueError):
        Genome("bogus_kind", 1.0, 5, 0.5, None, None, 100, 1)
    with pytest.raises(ValueError):
        Genome("carry", 0.01, 5, 0.5, None, None, 100, 1)
    with pytest.raises(ValueError):
        Genome("carry", 1.0, 9, 0.5, None, None, 100, 1)
    with pytest.raises(ValueError):
        Genome("carry", 1.0, 5, 0.5, 0.07, None, 100, 1)


def test_random_genome_always_within_frozen_bounds() -> None:
    rng = np.random.default_rng(1)
    for _ in range(500):
        individual = random_genome(rng)
        assert ENTRY_Z_MIN <= individual.entry_z <= ENTRY_Z_MAX
        assert POSITION_FRACTION_MIN <= individual.position_fraction <= POSITION_FRACTION_MAX
        assert individual.strategy_kind in STRATEGY_KIND_CHOICES
        assert individual.holding_days in HOLDING_DAYS_CHOICES
        assert individual.stop_loss_pct in STOP_LOSS_PCT_CHOICES
        assert individual.take_profit_pct in TAKE_PROFIT_PCT_CHOICES
        assert individual.universe_breadth in UNIVERSE_BREADTH_CHOICES
        assert individual.max_concurrent in MAX_CONCURRENT_CHOICES


def test_mutate_stays_in_bounds_and_changes_values_with_probability_one() -> None:
    rng = np.random.default_rng(2)
    base = Genome("momentum", 2.0, 5, 0.40, None, None, 100, 1)
    changed_continuous = False
    changed_categorical = False
    for _ in range(200):
        mutated = mutate(base, rng, sigma_fraction=0.10, probability=1.0)
        assert ENTRY_Z_MIN <= mutated.entry_z <= ENTRY_Z_MAX
        assert POSITION_FRACTION_MIN <= mutated.position_fraction <= POSITION_FRACTION_MAX
        assert mutated.holding_days in HOLDING_DAYS_CHOICES
        assert mutated.strategy_kind in STRATEGY_KIND_CHOICES
        if mutated.entry_z != base.entry_z:
            changed_continuous = True
        if mutated.strategy_kind != base.strategy_kind:
            changed_categorical = True
    assert changed_continuous
    assert changed_categorical


def test_crossover_child_genes_come_from_exactly_one_parent() -> None:
    rng = np.random.default_rng(3)
    parent_a = Genome("carry", 0.5, 1, 0.10, None, None, 30, 1)
    parent_b = Genome("convex_dual", 4.0, 14, 1.00, 0.20, 0.50, 200, 3)
    for _ in range(100):
        child = crossover(parent_a, parent_b, rng)
        for name, value in child.to_dict().items():
            assert value == getattr(parent_a, name) or value == getattr(parent_b, name)


def test_genome_key_collapses_float_noise_but_distinguishes_real_differences() -> None:
    a = Genome("momentum", 1.5, 5, 0.50, 0.10, None, 100, 2)
    b = Genome("momentum", 1.5 + 1e-12, 5, 0.50, 0.10, None, 100, 2)
    c = Genome("momentum", 1.6, 5, 0.50, 0.10, None, 100, 2)
    assert genome_key(a) == genome_key(b)
    assert genome_key(a) != genome_key(c)


def test_genome_to_dict_from_dict_roundtrip_including_none_genes() -> None:
    original = Genome("breakout", 2.2, 3, 0.37, None, 0.25, 100, 2)
    restored = from_dict(original.to_dict())
    assert restored == original


def test_normalized_weight_never_exceeds_1x_gross_exhaustive() -> None:
    """SPEC.md '레버리지는 1x 고정' -- exhaustive over the entire discrete gene grid x a fine
    continuous sweep of position_fraction, proving the construction (not a post-hoc gate)
    caps gross exposure. This is the property wave21_ga's H4 gate had to catch AFTER the fact
    (top_k_pairs x leg_fraction could exceed 1x); this wave's own genome forbids it by
    construction instead -- see genome23.py's module docstring."""
    worst_gross = 0.0
    for kind in STRATEGY_KIND_CHOICES:
        for max_concurrent in MAX_CONCURRENT_CHOICES:
            for position_fraction in np.linspace(POSITION_FRACTION_MIN, POSITION_FRACTION_MAX, 50):
                genome = Genome(kind, 1.0, 7, float(position_fraction), None, None, 100, max_concurrent)
                gross = genome.normalized_weight * max_concurrent
                worst_gross = max(worst_gross, gross)
                assert gross <= 1.0 + 1e-9
    assert worst_gross > 0.99  # sanity: the cap is actually load-bearing (some genome gets close to/at 1.0), not vacuously slack


# ---------------------------------------------------------------------------
# 2) Position lifecycle: exact holding_days length, no same-day rollover, stop/take-profit,
#    max_concurrent capacity.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("holding_days", [1, 2, 5, 14])
def test_holding_days_gives_exact_hold_length(holding_days: int) -> None:
    n_days, n_symbols = 30, 1
    entry_ok = np.zeros((n_days, n_symbols), dtype=bool)
    entry_ok[2, 0] = True  # a single ISOLATED entry opportunity -- unambiguous hold-length reading
    direction = np.ones((n_days, n_symbols), dtype=np.int8)
    rank_strength = np.ones((n_days, n_symbols))
    long_gap = np.zeros((n_days, n_symbols))
    long_intraday = np.zeros((n_days, n_symbols))

    magnitude, _signed = engine23._simulate_lifecycle(
        entry_ok, direction, rank_strength, long_gap, long_intraday,
        holding_days=holding_days, stop_loss_pct=None, take_profit_pct=None, max_concurrent=1, normalized_weight=1.0,
    )
    open_days = np.where(magnitude[:, 0] != 0)[0]
    assert open_days.tolist() == list(range(2, 2 + holding_days))


def test_no_same_day_rollover_reentry() -> None:
    """A symbol that exits on day t (holding_days limit) cannot re-enter until day t+1 -- even
    when the entry signal fires again on the VERY FIRST day it becomes eligible, that entry is
    exactly one day after the exit, never the same day."""
    holding_days = 5
    n_days = 30
    entry_ok = np.zeros((n_days, 1), dtype=bool)
    entry_ok[2, 0] = True
    entry_ok[2 + holding_days, 0] = True  # the first day the symbol is flat again
    direction = np.ones((n_days, 1), dtype=np.int8)
    rank_strength = np.ones((n_days, 1))
    long_gap = np.zeros((n_days, 1))
    long_intraday = np.zeros((n_days, 1))

    magnitude, _signed = engine23._simulate_lifecycle(
        entry_ok, direction, rank_strength, long_gap, long_intraday,
        holding_days=holding_days, stop_loss_pct=None, take_profit_pct=None, max_concurrent=1, normalized_weight=1.0,
    )
    open_days = np.where(magnitude[:, 0] != 0)[0]
    assert open_days.tolist() == [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]  # two back-to-back 5-day holds, no gap, no overlap


def test_stop_loss_exits_before_holding_days_limit() -> None:
    n_days = 10
    entry_ok = np.zeros((n_days, 1), dtype=bool)
    entry_ok[1, 0] = True
    direction = np.ones((n_days, 1), dtype=np.int8)
    rank_strength = np.ones((n_days, 1))
    long_gap = np.zeros((n_days, 1))
    long_intraday = np.zeros((n_days, 1))
    long_intraday[2, 0] = -0.50  # a -50% day, day 2 of the hold

    magnitude, _signed = engine23._simulate_lifecycle(
        entry_ok, direction, rank_strength, long_gap, long_intraday,
        holding_days=14, stop_loss_pct=0.10, take_profit_pct=None, max_concurrent=1, normalized_weight=1.0,
    )
    open_days = np.where(magnitude[:, 0] != 0)[0]
    assert open_days.tolist() == [1, 2]  # exits right after the stop-loss-triggering day, well before the 14-day cap


def test_take_profit_exits_before_holding_days_limit() -> None:
    n_days = 10
    entry_ok = np.zeros((n_days, 1), dtype=bool)
    entry_ok[1, 0] = True
    direction = np.ones((n_days, 1), dtype=np.int8)
    rank_strength = np.ones((n_days, 1))
    long_gap = np.zeros((n_days, 1))
    long_intraday = np.zeros((n_days, 1))
    long_intraday[2, 0] = 0.50

    magnitude, _signed = engine23._simulate_lifecycle(
        entry_ok, direction, rank_strength, long_gap, long_intraday,
        holding_days=14, stop_loss_pct=None, take_profit_pct=0.20, max_concurrent=1, normalized_weight=1.0,
    )
    open_days = np.where(magnitude[:, 0] != 0)[0]
    assert open_days.tolist() == [1, 2]


def test_max_concurrent_capacity_never_exceeded_and_ranks_by_strength() -> None:
    n_days, n_symbols = 10, 3
    entry_ok = np.ones((n_days, n_symbols), dtype=bool)
    direction = np.ones((n_days, n_symbols), dtype=np.int8)
    rank_strength = np.tile(np.array([3.0, 2.0, 1.0]), (n_days, 1))  # symbol 0 always strongest, symbol 2 always weakest
    long_gap = np.zeros((n_days, n_symbols))
    long_intraday = np.zeros((n_days, n_symbols))

    magnitude, _signed = engine23._simulate_lifecycle(
        entry_ok, direction, rank_strength, long_gap, long_intraday,
        holding_days=3, stop_loss_pct=None, take_profit_pct=None, max_concurrent=2, normalized_weight=0.5,
    )
    concurrent = (magnitude != 0).sum(axis=1)
    assert int(concurrent.max()) <= 2
    # the weakest-ranked symbol (2) should never be selected while 0/1 are always eligible
    assert bool((magnitude[:, 2] != 0).any()) is False


# ---------------------------------------------------------------------------
# 3) OOS sealing.
# ---------------------------------------------------------------------------


def test_oos_slice_raises_unless_final_mode() -> None:
    idx = pd.date_range("2025-09-01", periods=10, freq="D", tz="UTC")
    equity = pd.Series(np.linspace(90.0, 95.0, 10), index=idx)
    with pytest.raises(engine23.OOSLeakageError):
        engine23.oos_slice(equity, engine23.MODE_IS)
    with pytest.raises(engine23.OOSLeakageError):
        engine23.oos_slice(equity, "not_a_real_mode")
    sliced = engine23.oos_slice(equity, engine23.MODE_OOS_FINAL)
    assert bool((sliced.index > engine23.OOS_SPLIT).all())


def test_run_backtest_is_mode_never_returns_oos_rows() -> None:
    markets = _straddling_market()
    cache = engine23.market_cache_from_markets(markets, flat_cost_rate=0.001, always_liquid=True)
    assert 0 < int(cache.is_row_mask.sum()) < len(cache.index)  # fixture genuinely straddles the boundary

    equity_is = engine23.run_backtest(BASELINE_GENOME, cache, engine23.MODE_IS)
    assert bool((equity_is.index <= engine23.OOS_SPLIT).all())

    equity_full = engine23.run_backtest(BASELINE_GENOME, cache, engine23.MODE_OOS_FINAL)
    assert bool((equity_full.index > engine23.OOS_SPLIT).any())


def test_run_backtest_rejects_unknown_mode() -> None:
    markets = _two_symbol_market()
    cache = engine23.market_cache_from_markets(markets)
    with pytest.raises(ValueError):
        engine23.run_backtest(BASELINE_GENOME, cache, "OOS_SNEAKY")


def test_evaluate_genome_has_no_mode_parameter() -> None:
    parameters = inspect.signature(fitness23.evaluate_genome).parameters
    assert "mode" not in parameters


# ---------------------------------------------------------------------------
# 4) Fitness formula (SPEC.md 60일 롤링창 상위 25% 평균 - 3xP(<-20%)).
# ---------------------------------------------------------------------------


def test_rolling_window_returns_matches_hand_computed_values() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
    equity = pd.Series([100.0, 110.0, 121.0, 133.1, 146.41], index=idx)  # +10%/day compounding
    windows = fitness23.rolling_window_returns(equity, window_days=2)
    # pairs: (100->121)=0.21, (110->133.1)=0.21, (121->146.41)=0.21
    assert windows == pytest.approx([0.21, 0.21, 0.21], abs=1e-9)


def test_compute_fitness_matches_hand_computed_formula() -> None:
    # A 300-day series (comfortably longer than ROLLING_WINDOW_DAYS=60) with an engineered mix
    # of big rallies and a couple of sharp crashes, so both the top-25% bucket and the
    # P(window < -20%) bucket are non-degenerate. rolling_window_returns is called with the
    # SAME (default, real) window_days both times -- no monkeypatching of module constants
    # (which does not work here anyway: compute_fitness's own default `window_days` argument
    # is bound at function-definition time, not looked up dynamically per call).
    rng = np.random.default_rng(99)
    idx = pd.date_range("2022-01-01", periods=300, freq="D", tz="UTC")
    daily_returns = rng.normal(0.002, 0.03, 300)
    daily_returns[100:106] = -0.08  # an engineered sharp drawdown block -- guarantees some 60d windows breach -20%
    daily_returns[200:210] = 0.05  # an engineered rally block -- guarantees some 60d windows land in the top 25%
    equity = pd.Series(90.0 * np.cumprod(1.0 + daily_returns), index=idx)

    window_returns = fitness23.rolling_window_returns(equity)  # real default window_days=ROLLING_WINDOW_DAYS=60
    assert window_returns.size > 0
    cutoff = np.quantile(window_returns, 1.0 - fitness23.TOP_QUANTILE_FRACTION)
    expected_top_mean = float(np.mean(window_returns[window_returns >= cutoff]))
    expected_p_ruin = float(np.mean(window_returns < fitness23.RUIN_WINDOW_RETURN_THRESHOLD))
    expected_fitness = expected_top_mean - fitness23.RUIN_PENALTY_WEIGHT * expected_p_ruin
    assert expected_p_ruin > 0.0  # sanity: the engineered crash block actually produced >=1 ruin window

    result = fitness23.compute_fitness(equity)
    assert result.top_quantile_mean_return == pytest.approx(expected_top_mean)
    assert result.p_window_ruin == pytest.approx(expected_p_ruin)
    assert result.fitness == pytest.approx(expected_fitness)


def test_compute_fitness_too_short_series_fails_closed() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
    equity = pd.Series([100.0, 101.0, 102.0], index=idx)  # far shorter than ROLLING_WINDOW_DAYS=60
    result = fitness23.compute_fitness(equity)
    assert result.n_windows == 0
    assert result.fitness == -1.0  # fails closed to the worst possible score, not a lenient 0.0


# ---------------------------------------------------------------------------
# 5) GA / random-search mechanics.
# ---------------------------------------------------------------------------


def test_tournament_selection_favors_higher_fitness() -> None:
    rng = np.random.default_rng(7)
    population = [Genome("momentum", 1.0 + 0.01 * i, 5, 0.40, None, None, 100, 1) for i in range(10)]
    fitnesses = [float(i) for i in range(10)]
    winners = [ga23._tournament_select(population, fitnesses, rng) for _ in range(300)]
    winner_indices = [population.index(w) for w in winners]
    assert float(np.mean(winner_indices)) > 4.5


def test_run_ga_generational_best_never_regresses_due_to_elitism() -> None:
    markets = _two_symbol_market(periods=200)
    cache = engine23.market_cache_from_markets(markets, flat_cost_rate=0.0015, always_liquid=True)
    import research.wave23_ga_short.ga23 as ga23_mod
    original = (ga23_mod.POPULATION_SIZE, ga23_mod.GENERATIONS, ga23_mod.EVALUATIONS_PER_SEED, ga23_mod.ELITE_COUNT)
    try:
        ga23_mod.POPULATION_SIZE = 8
        ga23_mod.GENERATIONS = 4
        ga23_mod.EVALUATIONS_PER_SEED = 32
        ga23_mod.ELITE_COUNT = 2
        result = ga23_mod.run_ga(seed=999_001, cache=cache, progress=False)
    finally:
        ga23_mod.POPULATION_SIZE, ga23_mod.GENERATIONS, ga23_mod.EVALUATIONS_PER_SEED, ga23_mod.ELITE_COUNT = original

    assert len(result.history) == 4
    best_per_generation = [record.best_fitness for record in result.history]
    for earlier, later in zip(best_per_generation, best_per_generation[1:]):
        assert later >= earlier - 1e-12
    assert result.best_fitness == pytest.approx(max(best_per_generation))
    assert result.n_backtests_run <= result.n_evaluations
    assert sum(result.final_population_kind_counts.values()) == 8  # POPULATION_SIZE used for this run


def test_run_random_search_is_deterministic_given_seed() -> None:
    markets = _two_symbol_market(periods=200)
    cache = engine23.market_cache_from_markets(markets, flat_cost_rate=0.0015, always_liquid=True)
    result_a = random_search23.run_random_search(seed=555_001, cache=cache, n_evaluations=30, progress=False)
    result_b = random_search23.run_random_search(seed=555_001, cache=cache, n_evaluations=30, progress=False)
    assert genome_key(result_a.best_genome) == genome_key(result_b.best_genome)
    assert result_a.best_fitness == pytest.approx(result_b.best_fitness)
    assert result_a.fitness_history == pytest.approx(result_b.fitness_history)


# ---------------------------------------------------------------------------
# 6) gates23 (K1-K6).
# ---------------------------------------------------------------------------


def test_gate_k1_requires_at_least_four_of_five_seed_wins() -> None:
    four_wins = gates23.gate_k1_ga_beats_random([1, 1, 1, 1, -1], [0, 0, 0, 0, 0])
    assert four_wins["status"] == "PASS" and four_wins["n_wins"] == 4
    three_wins = gates23.gate_k1_ga_beats_random([1, 1, 1, -1, -1], [0, 0, 0, 0, 0])
    assert three_wins["status"] == "FAIL" and three_wins["n_wins"] == 3


def test_gate_k2_beats_i5_oos_requires_strict_improvement() -> None:
    assert gates23.gate_k2_beats_i5_oos(0.05, 0.03)["status"] == "PASS"
    assert gates23.gate_k2_beats_i5_oos(0.03, 0.03)["status"] == "FAIL"
    assert gates23.gate_k2_beats_i5_oos(None, 0.03)["status"] == "FAIL"


def test_gate_k3_dsr_uses_the_passed_equity_series_only() -> None:
    """K3 has no genome-selection logic of its own -- it can only ever score whatever equity
    series its CALLER passes in. Passing two DIFFERENT equity series (same shape, different
    values) must be able to produce different DSR scores, proving there is no hidden global
    state / cached 'other candidate' the gate could silently substitute -- the exact bug class
    this wave's SPEC.md calls out from wave21_ga."""
    idx = pd.date_range("2021-01-01", periods=500, freq="D", tz="UTC")
    rng = np.random.default_rng(13)
    good = pd.Series(90.0 * np.cumprod(1.0 + rng.normal(0.002, 0.01, len(idx))), index=idx)
    bad = pd.Series(90.0 * np.cumprod(1.0 + rng.normal(-0.001, 0.03, len(idx))), index=idx)
    result_good = gates23.gate_k3_dsr(good, trials=1000)
    result_bad = gates23.gate_k3_dsr(bad, trials=1000)
    assert result_good["score"] != pytest.approx(result_bad["score"])


def test_gate_k4_ruin_defense_flags_a_crashing_series() -> None:
    idx = pd.date_range("2021-01-01", periods=400, freq="D", tz="UTC")
    crashing = pd.Series(90.0 * np.linspace(1.0, 0.05, 400), index=idx)  # steady ~95% collapse
    result = gates23.gate_k4_ruin_defense(crashing)
    assert result["status"] == "FAIL"
    assert result["single_loss_ok"] is False


def test_gate_k5_executability_min_order_is_never_binding_at_gene_floor() -> None:
    """POSITION_FRACTION_MIN (0.10) x ACTIVE_CAPITAL ($90) = $9, already >= MIN_ORDER_USDT
    ($5) -- this wave's gene bounds happen to make the min-order constraint non-binding for
    ANY valid genome (unlike wave21_ga, where leg_fraction combined with top_k_pairs could
    create a gross>1x failure -- covered separately by the leverage-cap test above). Checked
    here at the gene-space floor to document that the boundary is genuinely non-binding, not
    untested."""
    floor_genome = Genome("momentum", 1.0, 5, genome23.POSITION_FRACTION_MIN, None, None, 100, max(genome23.MAX_CONCURRENT_CHOICES))
    idx = pd.date_range("2021-01-01", periods=200, freq="D", tz="UTC")
    equity = pd.Series(np.linspace(90.0, 95.0, 200), index=idx)
    fake_weights = np.zeros((200, 1))
    result = gates23.gate_k5_executability(floor_genome, fake_weights, equity, equity)
    assert result["leg_usdt"] == pytest.approx(genome23.POSITION_FRACTION_MIN * gates23.ACTIVE_CAPITAL)
    assert result["min_order_ok"] is True


def test_gate_k5_executability_flags_stress_sign_flip() -> None:
    idx = pd.date_range("2021-01-01", periods=300, freq="D", tz="UTC")
    rng = np.random.default_rng(5)
    base = pd.Series(90.0 * np.cumprod(1.0 + rng.normal(0.004, 0.02, 300)), index=idx)  # healthy uptrend -> positive top-25%
    stress = pd.Series(90.0 * np.cumprod(1.0 + rng.normal(-0.01, 0.02, 300)), index=idx)  # steady decline -> non-positive top-25%
    genome = Genome("momentum", 1.0, 5, 0.5, None, None, 100, 1)
    fake_weights = np.zeros((300, 1))
    result = gates23.gate_k5_executability(genome, fake_weights, base, stress)
    assert result["base_top25_mean_return"] > 0.0
    assert result["stress_top25_mean_return"] <= 0.0
    assert result["stress_sign_preserved"] is False
    assert result["status"] == "FAIL"


def test_gate_k6_flags_unsupported_kind_universe_breadth_and_exit_mechanics() -> None:
    breakout_genome = Genome("breakout", 1.0, 5, 0.5, None, None, 30, 1)
    result = breakout_genome and gates23.gate_k6_paper_reproducibility(breakout_genome)
    assert result["status"] == "FAIL"
    assert result["strategy_kind_supported"] is False

    wide_carry = Genome("carry", 1.0, 5, 0.5, None, None, 100, 1)  # breadth=100 > paper's ~42-symbol carry cap
    result_wide = gates23.gate_k6_paper_reproducibility(wide_carry)
    assert result_wide["universe_breadth_ok"] is False
    assert result_wide["status"] == "FAIL"

    stop_loss_carry = Genome("carry", 1.0, 5, 0.5, 0.10, None, 30, 1)  # paper has no stop-loss mechanics
    result_sl = gates23.gate_k6_paper_reproducibility(stop_loss_carry)
    assert result_sl["exit_mechanics_reproducible"] is False
    assert result_sl["status"] == "FAIL"

    clean_carry = Genome("carry", 1.0, 5, 0.5, None, None, 30, 1)
    result_clean = gates23.gate_k6_paper_reproducibility(clean_carry)
    assert result_clean["status"] == "PASS"


def test_evaluate_all_gates_requires_every_gate_to_pass() -> None:
    passing = {"status": "PASS"}
    failing = {"status": "FAIL"}
    all_pass = gates23.evaluate_all_gates(passing, passing, passing, passing, passing, passing)
    assert all_pass.overall == "PASS" and all_pass.promoted is True and all_pass.failure_reasons == ()

    one_fail = gates23.evaluate_all_gates(passing, failing, passing, passing, passing, passing)
    assert one_fail.overall == "FAIL" and one_fail.promoted is False and one_fail.failure_reasons == ("K2",)


# ---------------------------------------------------------------------------
# 7) select_final_candidate (median, not max).
# ---------------------------------------------------------------------------


def test_select_final_candidate_picks_median_not_max() -> None:
    seeds = [1, 2, 3, 4, 5]
    genomes = [Genome("momentum", 1.0 + 0.1 * i, 5, 0.40, None, None, 100, 1) for i in range(5)]
    fitnesses = [0.10, 0.20, 0.30, 0.05, 100.0]  # seed 5 is a single-seed jackpot; median (3rd of 5 sorted) = 0.20 -> seed 2
    chosen_genome, chosen_seed, chosen_fitness = run_wave23.select_final_candidate(seeds, genomes, fitnesses)
    assert chosen_seed == 2
    assert chosen_fitness == pytest.approx(0.20)
    assert genome_key(chosen_genome) == genome_key(genomes[1])
    assert chosen_seed != 5


def test_select_final_candidate_requires_odd_count() -> None:
    with pytest.raises(ValueError):
        run_wave23.select_final_candidate([1, 2], [BASELINE_GENOME, BASELINE_GENOME], [0.1, 0.2])


# ---------------------------------------------------------------------------
# 8) Real-cache integration smoke test (self-skips if the repo cache is not present).
# ---------------------------------------------------------------------------


def test_build_market_cache_real_cache_smoke() -> None:
    try:
        cache = engine23.build_market_cache()
    except (RuntimeError, FileNotFoundError) as error:
        pytest.skip(f"real repo cache not available: {error}")
    assert len(cache.symbols) > 0
    assert 0 < int(cache.is_row_mask.sum()) < len(cache.index)
    for kind in STRATEGY_KIND_CHOICES:
        assert kind in cache.signals
        genome = Genome(kind, 1.5, 5, 0.5, 0.10, 0.25, 30, 1)
        equity = engine23.run_backtest(genome, cache, engine23.MODE_IS)
        assert len(equity) == int(cache.is_row_mask.sum())
        assert bool((equity > 0).all())  # equity floor never breached in a sane sample genome
