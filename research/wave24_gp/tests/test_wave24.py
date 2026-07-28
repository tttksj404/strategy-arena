# Wave-24 GP test suite. Every fixture uses an in-memory synthetic market
# (fitness24.market_cache_from_markets) -- no disk I/O -- EXCEPT
# test_build_market_cache_real_cache_smoke, which touches the real repo cache and self-skips if
# it is not present (mirrors research/wave21_ga/tests/test_wave21.py and
# research/wave23_ga_short/tests/test_wave23.py's own convention).
#
# The tests that matter most for this wave's own integrity:
#   - test_safe_div_preserves_nan / test_safe_log_preserves_nan: pinned by name in tree.py's own
#     module docstring -- the NaN-vs-protected-floor distinction (a NaN input must stay NaN, only
#     a genuinely-finite-but-degenerate input gets floored) is load-bearing for correctness.
#   - test_run_backtest_is_mode_never_returns_oos_rows / test_oos_slice_raises_unless_final_mode
#     / test_evaluate_tree_has_no_mode_parameter: the OOS-sealing requirement.
#   - test_gate_l3_dsr_uses_the_passed_equity_series_only: pins that L3 always scores the
#     tree/equity pair it is GIVEN -- the exact class of mistake wave21_ga made (reporting a
#     different individual's DSR than the one actually gated) is structurally impossible here.
#   - test_gate_l6_universe_breadth_structurally_always_fails_given_frozen_breadth: documents
#     (does not "fix") a real, disclosed property of this wave's own frozen position structure --
#     see that test's own docstring.
#   - test_select_final_candidate_picks_median_not_max: SPEC.md "5시드 재현성" -- a single lucky
#     seed's jackpot can never become the final candidate.

from __future__ import annotations

import inspect
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[3]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

_REPO_ROOT = Path(__file__).resolve().parents[3]
_I5_RESULTS_PATH = _REPO_ROOT / "research" / "wave18_idle" / "results" / "I5.json"

import numpy as np
import pandas as pd  # noqa: PANDAS_OK
import pytest

from research.wave1.fam_funding import FundingMarket
from research.wave24_gp import fitness24, gates24, gp, random_trees, run_wave24
from research.wave24_gp import tree as tree_module
from research.wave24_gp.reporting24 import write_wave24_report
from research.wave24_gp.tree import (
    MAX_DEPTH,
    TERMINAL_VARS,
    Node,
    depth,
    depth_at_index,
    from_dict,
    grow,
    node_count,
    ramped_half_and_half,
    replace_subtree,
    subtree_depth_budget_ok,
    terminal_kinds_used,
    to_dict,
    to_formula_string,
    validate_tree,
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
    """Spans OOS_SPLIT (2025-09-30) with >= fitness24._MIN_CUTOFF_DAYS (120) days on the IS
    side -- start date chosen so ~122 days fall on-or-before OOS_SPLIT, then `periods` extends
    comfortably past it too (mirrors research.wave23_ga_short.tests.test_wave23's own
    _straddling_market)."""
    return _two_symbol_market(start="2025-06-01", periods=periods)


def _full_add_tree(depth_remaining: int) -> Node:
    """A perfectly balanced binary tree of `add` nodes over a single terminal -- node_count =
    2**(depth_remaining+1) - 1, depth = depth_remaining. Used to build trees that deliberately
    breach L7's node-count budget while still respecting MAX_DEPTH."""
    if depth_remaining <= 0:
        return Node("funding_7d")
    return Node("add", (_full_add_tree(depth_remaining - 1), _full_add_tree(depth_remaining - 1)))


def _add_chain(nodes: list[Node]) -> Node:
    """Right-folds >=1 nodes into a chain of binary `add` ops -- avoids manually nested parens
    for tests that need a tree touching many distinct terminal kinds without also inflating
    node_count much (chain of n leaves has node_count=2n-1, not the 2**depth-1 a balanced tree
    would need for the same leaf count)."""
    result = nodes[-1]
    for node in reversed(nodes[:-1]):
        result = Node("add", (node, result))
    return result


# ---------------------------------------------------------------------------
# 1) tree.py: validation, structural measurements, formula strings, (de)serialization.
# ---------------------------------------------------------------------------


def test_node_validate_rejects_malformed_trees() -> None:
    with pytest.raises(ValueError):
        tree_module._validate(Node("const", value=3.0))  # 3.0 not in CONST_VALUES
    with pytest.raises(ValueError):
        tree_module._validate(Node("funding_7d", value=1.0))  # terminal var must carry no value
    with pytest.raises(ValueError):
        tree_module._validate(Node("add", (Node("funding_7d"),)))  # arity mismatch (add needs 2)
    with pytest.raises(ValueError):
        tree_module._validate(Node("ma", (Node("atr_14"),), value=7))  # 7 not in MA_WINDOWS={5,10,20}
    with pytest.raises(ValueError):
        tree_module._validate(Node("bogus_op"))


def test_validate_tree_rejects_excess_depth() -> None:
    too_deep = _full_add_tree(MAX_DEPTH + 1)
    with pytest.raises(ValueError):
        validate_tree(too_deep)


def test_node_count_depth_terminal_kinds_used() -> None:
    tree = Node("div", (Node("funding_7d"), Node("const", value=2.0)))
    assert node_count(tree) == 3
    assert depth(tree) == 1
    assert terminal_kinds_used(tree) == frozenset({"funding_7d", "const"})


def test_ramped_half_and_half_produces_valid_trees_within_bounds() -> None:
    """`grow()` may legitimately terminate at a bare terminal (depth 0) at ANY depth budget --
    that early-termination is grow()'s own defining property (irregular shapes), so min_depth is
    only the max_depth BUDGET handed to each generator call, not a floor on the resulting tree's
    actual depth. Only the upper bound (MAX_DEPTH) and general structural validity are
    guaranteed for every individual; `full()`-generated individuals are checked separately below
    for the stronger "always reaches its budget exactly" guarantee."""
    rng = np.random.default_rng(21)
    population = ramped_half_and_half(rng, 40, min_depth=2, max_depth=MAX_DEPTH)
    assert len(population) == 40
    for individual in population:
        validate_tree(individual)  # raises on malformed/over-depth trees
        assert 0 <= depth(individual) <= MAX_DEPTH

    rng_full = np.random.default_rng(22)
    for target_depth in range(2, MAX_DEPTH + 1):
        full_tree = tree_module.full(rng_full, target_depth)
        validate_tree(full_tree)
        assert depth(full_tree) == target_depth  # full() always reaches its budget exactly


def test_to_dict_from_dict_roundtrip() -> None:
    original = Node("div", (Node("funding_7d"), Node("ma", (Node("atr_14"),), value=10)))
    restored = from_dict(to_dict(original))
    assert restored == original


def test_to_formula_string_matches_expected_shape() -> None:
    tree = Node("div", (Node("funding_7d"), Node("ma", (Node("atr_14"),), value=10)))
    assert to_formula_string(tree) == "(funding_7d / ma(atr_14, 10d))"
    assert to_formula_string(Node("zscore", (Node("basis"),))) == "zscore(basis, 20d)"
    assert to_formula_string(Node("const", value=0.5)) == "0.5"


def test_replace_subtree_and_depth_at_index() -> None:
    tree = Node("add", (Node("funding_7d"), Node("atr_14")))  # pre-order: 0=root, 1=funding_7d, 2=atr_14
    replaced = replace_subtree(tree, 1, Node("basis"))
    assert replaced == Node("add", (Node("basis"), Node("atr_14")))
    assert depth_at_index(tree, 0) == 0
    assert depth_at_index(tree, 1) == 1
    with pytest.raises(IndexError):
        replace_subtree(tree, 99, Node("basis"))


def test_subtree_depth_budget_ok_respects_max_depth() -> None:
    root = Node("funding_7d")  # index 0, depth 0
    assert subtree_depth_budget_ok(root, Node("atr_14"), 0, max_depth=0) is True
    deep_replacement = _full_add_tree(3)  # depth 3
    assert subtree_depth_budget_ok(root, deep_replacement, 0, max_depth=2) is False
    assert subtree_depth_budget_ok(root, deep_replacement, 0, max_depth=3) is True


def test_safe_div_preserves_nan() -> None:
    a = pd.DataFrame({"x": [1.0, np.nan, 2.0]})
    b = pd.DataFrame({"x": [np.nan, 1.0, 0.0]})
    result = tree_module._safe_div(a, b)
    assert np.isnan(result["x"].iloc[0])  # b is NaN -> NaN propagates, never fabricated to 0
    assert np.isnan(result["x"].iloc[1])  # a is NaN -> NaN propagates
    assert result["x"].iloc[2] == pytest.approx(0.0)  # b genuinely (non-NaN) near-zero -> protected floor


def test_safe_log_preserves_nan() -> None:
    a = pd.DataFrame({"x": [np.nan, 0.0, np.e]})
    result = tree_module._safe_log(a)
    assert np.isnan(result["x"].iloc[0])  # NaN input -> NaN output, never fabricated
    assert np.isfinite(result["x"].iloc[1])  # genuinely-zero (non-NaN) input -> protected floor, finite output
    assert result["x"].iloc[2] == pytest.approx(1.0)


def test_evaluate_add_and_terminal_lookup() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
    terminals = {
        "funding_7d": pd.DataFrame({"AAA": [0.01, 0.02, 0.03]}, index=idx),
        "atr_14": pd.DataFrame({"AAA": [1.0, 2.0, 3.0]}, index=idx),
    }
    tree = Node("add", (Node("funding_7d"), Node("const", value=1.0)))
    result = tree_module.evaluate(tree, terminals)
    assert result["AAA"].tolist() == pytest.approx([1.01, 1.02, 1.03])


# ---------------------------------------------------------------------------
# 2) gp.py: reproduction operators always respect MAX_DEPTH, tournament selection, run_gp.
# ---------------------------------------------------------------------------


def test_crossover_always_respects_max_depth_and_validity() -> None:
    rng = np.random.default_rng(11)
    for _ in range(200):
        parent_a, parent_b = grow(rng, MAX_DEPTH), grow(rng, MAX_DEPTH)
        child = gp.crossover(parent_a, parent_b, rng)
        assert depth(child) <= MAX_DEPTH
        validate_tree(child)


def test_mutate_subtree_always_respects_max_depth_and_validity() -> None:
    rng = np.random.default_rng(12)
    for _ in range(200):
        parent = grow(rng, MAX_DEPTH)
        child = gp.mutate_subtree(parent, rng)
        assert depth(child) <= MAX_DEPTH
        validate_tree(child)


def test_mutate_point_preserves_node_count() -> None:
    rng = np.random.default_rng(13)
    for _ in range(200):
        parent = grow(rng, MAX_DEPTH)
        child = gp.mutate_point(parent, rng)
        assert node_count(child) == node_count(parent)
        validate_tree(child)


def test_tournament_selection_favors_higher_fitness() -> None:
    rng = np.random.default_rng(7)
    population = [Node(op=name) for name in TERMINAL_VARS[:10]]  # 10 distinct, valid terminal nodes
    fitnesses = [float(i) for i in range(10)]
    winners = [gp._tournament_select(population, fitnesses, rng) for _ in range(300)]
    winner_indices = [population.index(winner) for winner in winners]
    assert float(np.mean(winner_indices)) > 4.5


def test_run_gp_generational_best_never_regresses_due_to_elitism() -> None:
    markets = _two_symbol_market(periods=200)
    cache = fitness24.market_cache_from_markets(markets, flat_cost_rate=0.0015, always_liquid=True)
    import research.wave24_gp.gp as gp_mod

    original = (gp_mod.POPULATION_SIZE, gp_mod.GENERATIONS, gp_mod.EVALUATIONS_PER_SEED, gp_mod.ELITE_COUNT)
    try:
        gp_mod.POPULATION_SIZE = 8
        gp_mod.GENERATIONS = 4
        gp_mod.EVALUATIONS_PER_SEED = 32
        gp_mod.ELITE_COUNT = 2
        result = gp_mod.run_gp(seed=999_001, cache=cache, progress=False)
    finally:
        gp_mod.POPULATION_SIZE, gp_mod.GENERATIONS, gp_mod.EVALUATIONS_PER_SEED, gp_mod.ELITE_COUNT = original

    assert len(result.history) == 4
    best_per_generation = [record.best_fitness for record in result.history]
    for earlier, later in zip(best_per_generation, best_per_generation[1:]):
        assert later >= earlier - 1e-12
    assert result.best_fitness == pytest.approx(max(best_per_generation))
    assert result.n_backtests_run <= result.n_evaluations


# ---------------------------------------------------------------------------
# 3) OOS sealing + walk-forward fitness formula.
# ---------------------------------------------------------------------------


def test_oos_slice_raises_unless_final_mode() -> None:
    idx = pd.date_range("2025-09-01", periods=10, freq="D", tz="UTC")
    equity = pd.Series(np.linspace(90.0, 95.0, 10), index=idx)
    with pytest.raises(fitness24.OOSLeakageError):
        fitness24.oos_slice(equity, fitness24.MODE_IS)
    with pytest.raises(fitness24.OOSLeakageError):
        fitness24.oos_slice(equity, "not_a_real_mode")
    sliced = fitness24.oos_slice(equity, fitness24.MODE_OOS_FINAL)
    assert bool((sliced.index > fitness24.OOS_SPLIT).all())


def test_run_backtest_is_mode_never_returns_oos_rows() -> None:
    markets = _straddling_market()
    cache = fitness24.market_cache_from_markets(markets, flat_cost_rate=0.001, always_liquid=True)
    assert 0 < int(cache.is_row_mask.sum()) < len(cache.index)  # fixture genuinely straddles the boundary

    tree = Node("zscore", (Node("funding_7d"),))
    equity_is = fitness24.run_backtest(tree, cache, fitness24.MODE_IS)
    assert bool((equity_is.index <= fitness24.OOS_SPLIT).all())

    equity_full = fitness24.run_backtest(tree, cache, fitness24.MODE_OOS_FINAL)
    assert bool((equity_full.index > fitness24.OOS_SPLIT).any())


def test_run_backtest_rejects_unknown_mode() -> None:
    markets = _two_symbol_market()
    cache = fitness24.market_cache_from_markets(markets)
    with pytest.raises(ValueError):
        fitness24.run_backtest(Node("funding_7d"), cache, "OOS_SNEAKY")


def test_evaluate_tree_has_no_mode_parameter() -> None:
    assert "mode" not in inspect.signature(fitness24.evaluate_tree).parameters
    assert "mode" not in inspect.signature(fitness24.evaluate_tree_cached).parameters


def test_evaluate_tree_cached_reuses_result_for_structurally_identical_tree() -> None:
    markets = _two_symbol_market(periods=200)
    cache = fitness24.market_cache_from_markets(markets, flat_cost_rate=0.001, always_liquid=True)
    tree_a = Node("zscore", (Node("funding_7d"),))
    tree_b = Node("zscore", (Node("funding_7d"),))  # structurally identical, different object
    fitness_cache: dict = {}
    result_a, hit_a = fitness24.evaluate_tree_cached(tree_a, cache, fitness_cache)
    result_b, hit_b = fitness24.evaluate_tree_cached(tree_b, cache, fitness_cache)
    assert hit_a is False
    assert hit_b is True
    assert result_a is result_b
    assert len(fitness_cache) == 1


def test_assemble_terminals_never_produces_object_dtype_even_with_empty_source_symbol() -> None:
    """Regression test for a real bug found running the full pipeline against the actual repo
    cache (2026-07-28): AEROUSDT has futures/perp data but ZERO rows of spot data. An empty
    source DataFrame makes pandas infer dtype=object for the resulting all-NaN column, which
    poisons that ONE column's dtype all the way through to the `basis` terminal (perp/spot
    division) -- any numpy ufunc a GP tree then applies (log/abs/zscore/...) raises `TypeError:
    loop of ufunc does not support argument 0 of type float which has no callable log method`
    the moment it touches that column (numpy's object-dtype per-element method-call fallback).
    Fixed in fitness24._assemble_terminals by casting every terminal panel to float64 before
    returning -- a pure storage-representation fix, NaN stays NaN."""
    markets = _two_symbol_market(periods=200)
    empty_index = pd.DatetimeIndex([], tz="UTC")
    dead_spot = pd.DataFrame(columns=["open", "close"], index=empty_index)
    dead_perp = pd.DataFrame(columns=["open", "high", "low", "close"], index=empty_index)
    dead_funding = pd.Series(dtype=float, index=empty_index, name="funding_rate")
    markets["DEADUSDT"] = FundingMarket(dead_spot, dead_perp, dead_funding)

    cache = fitness24.market_cache_from_markets(markets, flat_cost_rate=0.001, always_liquid=True)
    for name, frame in cache.terminals.items():
        assert set(frame.dtypes.tolist()) == {np.dtype("float64")}, f"terminal {name!r} has non-float64 column(s): {frame.dtypes.to_dict()}"

    dead_index = cache.symbols.index("DEADUSDT")
    assert not cache.available[:, dead_index].any()  # confirms DEADUSDT is genuinely untradeable, as intended -- the fix does not make it tradeable

    tree = Node("log", (Node("basis"),))  # would previously raise TypeError the moment it touched the object-dtype column
    result = fitness24.evaluate_tree(tree, cache)
    assert np.isfinite(result.fitness)


def test_walk_forward_fitness_matches_hand_computed_formula() -> None:
    idx = pd.date_range("2024-01-01", periods=40, freq="D", tz="UTC")  # 40/4folds=10 -> exact fold edges, no rounding ambiguity
    rng = np.random.default_rng(42)
    equity = pd.Series(90.0 * np.cumprod(1.0 + rng.normal(0.001, 0.01, 40)), index=idx)
    node_count_value = 7

    result = fitness24.walk_forward_fitness(equity, node_count_value, n_folds=4)

    fold_cagrs = [fitness24.cagr(equity.iloc[lo:hi]) for lo, hi in [(0, 10), (10, 20), (20, 30), (30, 40)]]
    expected_median = float(np.median(fold_cagrs))
    expected_std = float(np.std(fold_cagrs, ddof=0))
    expected_mdd = fitness24._max_drawdown(equity)
    expected_mdd_penalty = fitness24.MDD_PENALTY_WEIGHT * max(0.0, expected_mdd - fitness24.MDD_FOLD_FLOOR)
    expected_node_penalty = fitness24.NODE_COUNT_PENALTY_WEIGHT * node_count_value
    expected_fitness = expected_median - expected_std - expected_mdd_penalty - expected_node_penalty

    assert result.fold_cagrs == pytest.approx(tuple(fold_cagrs))
    assert result.median_fold_cagr == pytest.approx(expected_median)
    assert result.std_fold_cagr == pytest.approx(expected_std)
    assert result.mdd == pytest.approx(expected_mdd)
    assert result.node_count == node_count_value
    assert result.fitness == pytest.approx(expected_fitness)


def test_walk_forward_fitness_rejects_too_short_series() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
    equity = pd.Series([90.0, 91.0, 92.0, 91.0, 93.0], index=idx)
    with pytest.raises(ValueError):
        fitness24.walk_forward_fitness(equity, node_count_value=3, n_folds=4)


# ---------------------------------------------------------------------------
# 4) random_trees.py.
# ---------------------------------------------------------------------------


def test_run_random_search_is_deterministic_given_seed() -> None:
    markets = _two_symbol_market(periods=200)
    cache = fitness24.market_cache_from_markets(markets, flat_cost_rate=0.0015, always_liquid=True)
    result_a = random_trees.run_random_search(seed=555_001, cache=cache, n_evaluations=30, progress=False)
    result_b = random_trees.run_random_search(seed=555_001, cache=cache, n_evaluations=30, progress=False)
    assert result_a.best_tree == result_b.best_tree
    assert result_a.best_fitness == pytest.approx(result_b.best_fitness)
    assert result_a.fitness_history == pytest.approx(result_b.fitness_history)
    assert result_a.n_evaluations == 30
    assert len(result_a.fitness_history) == 30


# ---------------------------------------------------------------------------
# 5) gates24 (L1-L7).
# ---------------------------------------------------------------------------


def test_gate_l1_requires_at_least_four_of_five_seed_wins() -> None:
    four_wins = gates24.gate_l1_gp_beats_random([1, 1, 1, 1, -1], [0, 0, 0, 0, 0])
    assert four_wins["status"] == "PASS" and four_wins["n_wins"] == 4
    three_wins = gates24.gate_l1_gp_beats_random([1, 1, 1, -1, -1], [0, 0, 0, 0, 0])
    assert three_wins["status"] == "FAIL" and three_wins["n_wins"] == 3


def test_gate_l2_beats_i5_oos_requires_strict_improvement() -> None:
    assert gates24.gate_l2_beats_i5_oos(0.05, 0.03)["status"] == "PASS"
    assert gates24.gate_l2_beats_i5_oos(0.03, 0.03)["status"] == "FAIL"
    assert gates24.gate_l2_beats_i5_oos(None, 0.03)["status"] == "FAIL"


def test_gate_l3_dsr_uses_the_passed_equity_series_only() -> None:
    """L3 has no tree-selection logic of its own -- it can only ever score whatever equity series
    its CALLER passes in. Two different equity series must produce different DSR scores, proving
    no hidden global state could silently substitute a different candidate's curve (the exact bug
    class wave21_ga's own H3 hit -- see gates24.py's module docstring)."""
    idx = pd.date_range("2021-01-01", periods=500, freq="D", tz="UTC")
    rng = np.random.default_rng(13)
    good = pd.Series(90.0 * np.cumprod(1.0 + rng.normal(0.002, 0.01, len(idx))), index=idx)
    bad = pd.Series(90.0 * np.cumprod(1.0 + rng.normal(-0.001, 0.03, len(idx))), index=idx)
    result_good = gates24.gate_l3_dsr(good, trials=1000)
    result_bad = gates24.gate_l3_dsr(bad, trials=1000)
    assert result_good["score"] != pytest.approx(result_bad["score"])


def test_gate_l4_ruin_defense_flags_a_crashing_series() -> None:
    idx = pd.date_range("2021-01-01", periods=400, freq="D", tz="UTC")
    crashing = pd.Series(90.0 * np.linspace(1.0, 0.05, 400), index=idx)  # steady ~95% collapse
    result = gates24.gate_l4_mc_and_block(crashing)
    assert result["status"] == "FAIL"


def test_gate_l5_executability_reflects_fixed_position_structure() -> None:
    idx = pd.date_range("2021-01-01", periods=300, freq="D", tz="UTC")
    equity = pd.Series(np.linspace(90.0, 95.0, 300), index=idx)
    result = gates24.gate_l5_executability(equity, equity)
    assert result["leg_usdt_nominal"] == pytest.approx(gates24.LEG_USDT)
    assert result["min_order_feasible"] is True
    assert result["gross_leverage_1x_ok"] is True
    assert result["status"] == "PASS"


def test_gate_l5_executability_flags_stress_sign_flip() -> None:
    idx = pd.date_range("2021-01-01", periods=300, freq="D", tz="UTC")
    rng = np.random.default_rng(5)
    base = pd.Series(90.0 * np.cumprod(1.0 + rng.normal(0.004, 0.02, 300)), index=idx)  # healthy uptrend
    stress = pd.Series(90.0 * np.cumprod(1.0 + rng.normal(-0.01, 0.02, 300)), index=idx)  # steady decline
    result = gates24.gate_l5_executability(base, stress)
    assert result["stress_sign_preserved"] is False
    assert result["status"] == "FAIL"


def test_gate_l6_universe_breadth_structurally_always_fails_given_frozen_breadth() -> None:
    """wave24 freezes universe_breadth=200 (FIXED_UNIVERSE_BREADTH, inherited unchanged from
    L4/I5's own position-structure baseline) as a hardcoded constant -- it is NOT a GP-evolvable
    gene (task brief: GP evolves the signal only). PAPER_CARRY_UNIVERSE_CAP=42 (research.paper.
    market_data's own live-fetched carry universe) is far narrower, so L6's breadth check is FAIL
    for every conceivable tree, unconditionally. This is a KNOWN, disclosed structural property
    of this wave (see gates24.py's own gate_l6_paper_reproducibility docstring: "a
    POSITION-STRUCTURE inheritance gap... not something this wave's GP formula search could have
    avoided"), not a code bug and not something any individual formula's quality can change --
    this test documents/pins that fact rather than trying to "fix" it (fixing it would require
    touching fitness24.FIXED_UNIVERSE_BREADTH or research/paper/, both out of this wave's scope)."""
    assert gates24.FIXED_UNIVERSE_BREADTH > gates24.PAPER_CARRY_UNIVERSE_CAP
    for tree in (Node("funding_7d"), Node("zscore", (Node("basis"),)), Node("const", value=1.0)):
        result = gates24.gate_l6_paper_reproducibility(tree)
        assert result["data_ok"] is True  # every SPEC.md terminal DOES map to a known live data family
        assert result["universe_breadth_ok"] is False
        assert result["status"] == "FAIL"


def test_gate_l7_simplicity_thresholds() -> None:
    small = Node("funding_7d")
    result_small = gates24.gate_l7_simplicity(small)
    assert result_small["status"] == "PASS"
    assert result_small["node_count"] == 1
    assert result_small["n_terminal_kinds"] == 1

    big = _full_add_tree(4)  # depth 4 <= MAX_DEPTH(5), node_count = 2**5-1 = 31 > L7_MAX_NODE_COUNT(15)
    result_big = gates24.gate_l7_simplicity(big)
    assert result_big["node_count"] == 31
    assert result_big["node_count_ok"] is False
    assert result_big["status"] == "FAIL"

    # 6 distinct market terminal kinds (> L7_MAX_TERMINAL_KINDS=5) chained via a right-leaning
    # `add` fold -- node_count stays small (11) so only the terminal-KINDS check trips, not the
    # node-count one (kept as a separate, isolated assertion from the `big` case above).
    many_kinds = _add_chain([Node("funding_1d"), Node("funding_7d"), Node("price_ret_1d"), Node("atr_14"), Node("basis"), Node("quote_volume_30d")])
    result_many_kinds = gates24.gate_l7_simplicity(many_kinds)
    assert result_many_kinds["n_terminal_kinds"] == 6
    assert result_many_kinds["terminal_kinds_ok"] is False
    assert result_many_kinds["status"] == "FAIL"


def test_evaluate_all_gates_requires_every_gate_to_pass() -> None:
    passing = {"status": "PASS"}
    failing = {"status": "FAIL"}
    all_pass = gates24.evaluate_all_gates(passing, passing, passing, passing, passing, passing, passing)
    assert all_pass.overall == "PASS" and all_pass.promoted is True and all_pass.failure_reasons == ()

    one_fail = gates24.evaluate_all_gates(passing, failing, passing, passing, passing, passing, passing)
    assert one_fail.overall == "FAIL" and one_fail.promoted is False and one_fail.failure_reasons == ("L2",)


# ---------------------------------------------------------------------------
# 6) select_final_candidate (median, not max).
# ---------------------------------------------------------------------------


def test_select_final_candidate_picks_median_not_max() -> None:
    seeds = [1, 2, 3, 4, 5]
    trees = [Node(op=name) for name in TERMINAL_VARS[:5]]
    fitnesses = [0.10, 0.20, 0.30, 0.05, 100.0]  # seed 5 is a single-seed jackpot; median (3rd of 5 sorted) = 0.20 -> seed 2
    chosen_tree, chosen_seed, chosen_fitness = run_wave24.select_final_candidate(seeds, trees, fitnesses)
    assert chosen_seed == 2
    assert chosen_fitness == pytest.approx(0.20)
    assert chosen_tree == trees[1]
    assert chosen_seed != 5


def test_select_final_candidate_requires_odd_count() -> None:
    with pytest.raises(ValueError):
        run_wave24.select_final_candidate([1, 2], [Node("funding_7d"), Node("basis")], [0.1, 0.2])


# ---------------------------------------------------------------------------
# 7) reporting24 (formula readability, 5-seed comparison, economic interpretability).
# ---------------------------------------------------------------------------


def _tree_payload(node: Node) -> dict:
    return {"tree": to_dict(node), "formula": to_formula_string(node), "node_count": node_count(node)}


def test_write_wave24_report_promoted_case_covers_required_sections(tmp_path: Path) -> None:
    results_dir, report_dir, registry_path = tmp_path / "results", tmp_path / "report", tmp_path / "REGISTRY.md"
    results_dir.mkdir()

    seed_trees = {
        gp.SEEDS[0]: Node("zscore", (Node("funding_7d"),)),  # pure funding -> interpretable
        gp.SEEDS[1]: Node("div", (Node("funding_7d"), Node("atr_14"))),  # funding/risk -> interpretable
        gp.SEEDS[2]: Node("sub", (Node("price_ret_7d"), Node("const", value=0.5))),  # momentum -> not interpretable
        gp.SEEDS[3]: Node("mul", (Node("funding_14d"), Node("realized_vol_20d"))),  # funding+risk -> interpretable
        gp.SEEDS[4]: Node("log", (Node("quote_volume_30d"),)),  # liquidity only, no funding -> not interpretable
    }
    fitnesses = {gp.SEEDS[0]: 0.05, gp.SEEDS[1]: 0.12, gp.SEEDS[2]: -0.02, gp.SEEDS[3]: 0.20, gp.SEEDS[4]: 0.01}
    for seed in gp.SEEDS:
        payload = {
            "seed": seed,
            "best_tree": _tree_payload(seed_trees[seed]),
            "best_fitness": fitnesses[seed],
            "n_evaluations": 24,
            "n_backtests_run": 20,
            "wall_seconds": 12.3,
            "history": [
                {"generation": g, "best_fitness": fitnesses[seed], "mean_fitness": fitnesses[seed] - 0.05, "worst_fitness": fitnesses[seed] - 0.1, "mean_node_count": 5.0, "best_tree": _tree_payload(seed_trees[seed])}
                for g in range(3)
            ],
        }
        (results_dir / f"gp_seed{seed}.json").write_text(json.dumps(payload), encoding="utf-8")
    for seed in random_trees.SEEDS:
        payload = {"seed": seed, "best_tree": _tree_payload(Node("basis")), "best_fitness": -0.03, "n_evaluations": 24, "n_backtests_run": 24, "wall_seconds": 5.5, "fitness_history": [-0.03] * 5}
        (results_dir / f"random_seed{seed}.json").write_text(json.dumps(payload), encoding="utf-8")

    final_tree = seed_trees[gp.SEEDS[1]]  # median seed by construction of `fitnesses` above
    final_payload = {
        "final_tree": _tree_payload(final_tree),
        "source_seed": gp.SEEDS[1],
        "source_is_fitness": 0.12,
        "gp_best_by_seed": {str(s): fitnesses[s] for s in gp.SEEDS},
        "random_best_by_seed": {str(s): -0.03 for s in random_trees.SEEDS},
        "full_period_cagr": 0.09, "is_cagr": 0.15, "oos_cagr_self_contained": 0.02, "oos_cagr_regime_anchored": 0.04, "mdd_full": 0.11,
        "regime_breakdown": {},
        "i5_reference": {"oos_cagr": 0.0306, "is_cagr": 0.08, "is_oos_gap_pp": 5.0},
        "gates": {
            "l1": {"status": "PASS", "n_wins": 4, "n_seeds": 5, "threshold": 4},
            "l2": {"status": "PASS", "final_oos_cagr": 0.04, "i5_oos_cagr": 0.0306, "gap_pp": 0.94},
            "l3": {"status": "PASS", "score": 0.5, "probability": 0.69, "trials": 82621, "observed_sharpe": 1.1},
            "l4": {"status": "PASS", "mc": {"p05": 150.0, "ruin_probability": 0.01}, "block_mdd_p95": 0.10},
            "l5": {"status": "PASS", "leg_usdt_nominal": 45.0, "gross_usdt_nominal": 90.0, "stress_start_usdt": 90.0, "stress_end_usdt": 95.0},
            "l6": {"status": "PASS", "data_ok": True, "universe_breadth": 200, "paper_carry_universe_cap": 42, "universe_breadth_ok": True, "reasons": []},
            "l7": {"status": "PASS", "node_count": 3, "n_terminal_kinds": 2, "depth": 1},
            "overall": "PASS", "promoted": True, "failure_reasons": [],
        },
        "dsr_reference_cumulative": {"score": 0.5, "probability": 0.69, "trials": 82621, "observed_sharpe": 1.1},
    }
    (results_dir / "final_candidate.json").write_text(json.dumps(final_payload), encoding="utf-8")

    write_wave24_report(results_dir, report_dir, registry_path, _I5_RESULTS_PATH)
    report_text = (report_dir / "wave24_report.md").read_text(encoding="utf-8")
    registry_text = registry_path.read_text(encoding="utf-8")

    assert "(funding_7d / atr_14)" in report_text  # human-readable final formula, verbatim
    assert "해석 가능" in report_text  # at least one interpretable seed formula
    assert "해석 불가" in report_text  # at least one uninterpretable seed formula (momentum/liquidity-only)
    assert "자카드 유사도" in report_text  # 5-seed comparison ran
    assert "판정: 시드마다 상이" in report_text or "판정: 구조 재현성 있음" in report_text
    assert "| YES |" in registry_text
    assert "(funding_7d / atr_14)" in registry_text


def test_write_wave24_report_declares_methodology_exhausted_when_all_three_waves_fail(tmp_path: Path) -> None:
    """wave21_ga/REGISTRY.md and wave23_ga_short/REGISTRY.md (real repo files, frozen historical
    results) both already record 승격=NO. If wave24 ALSO fails to promote, the verdict section's
    read-only cross-check should fire SPEC.md's own pre-registered '탐색 방법론 소진' declaration."""
    assert "| NO |" in _REPO_ROOT.joinpath("research", "wave21_ga", "REGISTRY.md").read_text(encoding="utf-8")
    assert "| NO |" in _REPO_ROOT.joinpath("research", "wave23_ga_short", "REGISTRY.md").read_text(encoding="utf-8")

    results_dir, report_dir, registry_path = tmp_path / "results", tmp_path / "report", tmp_path / "REGISTRY.md"
    results_dir.mkdir()
    final_payload = {
        "final_tree": _tree_payload(Node("price_ret_7d")),
        "source_seed": gp.SEEDS[0], "source_is_fitness": 0.01,
        "gp_best_by_seed": {str(s): 0.01 for s in gp.SEEDS},
        "random_best_by_seed": {str(s): 0.02 for s in random_trees.SEEDS},
        "full_period_cagr": 0.01, "is_cagr": 0.05, "oos_cagr_self_contained": -0.01, "oos_cagr_regime_anchored": 0.01, "mdd_full": 0.30,
        "regime_breakdown": {},
        "i5_reference": {"oos_cagr": 0.0306, "is_cagr": 0.08, "is_oos_gap_pp": 5.0},
        "gates": {
            "l1": {"status": "FAIL", "n_wins": 2, "n_seeds": 5, "threshold": 4},
            "l2": {"status": "FAIL", "final_oos_cagr": 0.01, "i5_oos_cagr": 0.0306, "gap_pp": -2.06},
            "l3": {"status": "FAIL", "score": -0.3, "probability": 0.38, "trials": 82621, "observed_sharpe": 0.1},
            "l4": {"status": "FAIL", "mc": {"p05": 40.0, "ruin_probability": 0.20}, "block_mdd_p95": 0.30},
            "l5": {"status": "PASS", "leg_usdt_nominal": 45.0, "gross_usdt_nominal": 90.0, "stress_start_usdt": 90.0, "stress_end_usdt": 91.0},
            "l6": {"status": "FAIL", "data_ok": True, "universe_breadth": 200, "paper_carry_universe_cap": 42, "universe_breadth_ok": False, "reasons": ["universe too wide"]},
            "l7": {"status": "PASS", "node_count": 1, "n_terminal_kinds": 1, "depth": 0},
            "overall": "FAIL", "promoted": False, "failure_reasons": ["L1", "L2", "L3", "L4", "L6"],
        },
        "dsr_reference_cumulative": {"score": -0.3, "probability": 0.38, "trials": 82621, "observed_sharpe": 0.1},
    }
    (results_dir / "final_candidate.json").write_text(json.dumps(final_payload), encoding="utf-8")

    write_wave24_report(results_dir, report_dir, registry_path, _I5_RESULTS_PATH)
    report_text = (report_dir / "wave24_report.md").read_text(encoding="utf-8")

    assert "미승격" in report_text
    assert "탐색 방법론 소진 선언" in report_text
    assert "GP 무의미" in report_text
    assert "| NO |" in registry_path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 8) Real-cache integration smoke test (self-skips if the repo cache is not present).
# ---------------------------------------------------------------------------


def test_build_market_cache_real_cache_smoke() -> None:
    try:
        cache = fitness24.build_market_cache()
    except (RuntimeError, FileNotFoundError) as error:
        pytest.skip(f"real repo cache not available: {error}")
    assert len(cache.symbols) > 0
    assert 0 < int(cache.is_row_mask.sum()) < len(cache.index)
    tree = Node("zscore", (Node("funding_7d"),))
    equity = fitness24.run_backtest(tree, cache, fitness24.MODE_IS)
    assert len(equity) == int(cache.is_row_mask.sum())
    assert bool((equity > 0).all())  # equity floor never breached for a sane sample tree
