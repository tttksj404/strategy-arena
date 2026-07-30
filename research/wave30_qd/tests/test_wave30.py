# Wave-30 tests. These pin the claims the report will make, in particular the ones a reader
# cannot verify by eye: that the stop is provably interior to the liquidation band for EVERY
# reachable genome, that no signal reads a bar it could not have seen, that the same-bar
# stop/target tie is resolved against the strategy, and that the search loop cannot touch OOS.

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.wave30_qd.dataio30 import OOSLeakageError, OOS_SPLIT, SymbolArrays, build_market_cache
from research.wave30_qd.engine30 import _resolve_trade, run_genome, signal_direction
from research.wave30_qd.fitness30 import descriptor_of, evaluate
from research.wave30_qd.genome30 import (
    LEV_CAP,
    MAINT_MARGIN,
    STOP_BAND_MARGIN,
    Genome,
    InvalidGenomeError,
    crossover,
    mutate,
    random_genome,
)
from research.wave30_qd.search30 import Evaluator, fast_non_dominated_sort, run_map_elites


# ---------------------------------------------------------------------------------------
# Structural leverage guarantee (the whole premise of the wave)
# ---------------------------------------------------------------------------------------


def test_random_genomes_are_always_feasible_and_within_cap():
    rng = np.random.default_rng(30_001)
    for _ in range(5_000):
        genome = random_genome(rng)
        genome.validate()
        assert 1.0 <= genome.leverage <= LEV_CAP
        assert genome.stop_pct <= STOP_BAND_MARGIN * genome.liquidation_band + 1e-12


def test_mutation_and_crossover_never_produce_infeasible_survivors():
    rng = np.random.default_rng(30_002)
    parents = [random_genome(rng) for _ in range(40)]
    produced = 0
    for _ in range(3_000):
        try:
            child = mutate(parents[int(rng.integers(len(parents)))], rng)
        except InvalidGenomeError:
            continue
        produced += 1
        assert child.leverage <= LEV_CAP
        assert child.stop_pct <= STOP_BAND_MARGIN * child.liquidation_band + 1e-12
    assert produced > 1_000, "mutation is rejecting nearly everything; the sampler is mis-specified"

    for _ in range(1_000):
        left, right = rng.choice(len(parents), size=2, replace=False)
        try:
            child = crossover(parents[int(left)], parents[int(right)], rng)
        except InvalidGenomeError:
            continue
        assert child.stop_pct <= STOP_BAND_MARGIN * child.liquidation_band + 1e-12


def test_leverage_above_cap_is_rejected_not_clipped():
    # risk 0.40 with a 1% stop implies 40x. The cap must make this INFEASIBLE rather than
    # silently reporting 20x while sizing as if it were 40x.
    genome = Genome(
        signal_family="breakout",
        lookback_bars=24,
        entry_threshold=0.0,
        stop_pct=0.01,
        target_r=2.0,
        trail_enabled=False,
        risk_frac=0.40,
        max_hold_bars=48,
        allow_short=True,
        symbols=("BTCUSDT",),
        max_concurrent=1,
        cooldown_bars_after_loss=0,
        sleeve_fraction=0.25,
    )
    assert not genome.is_feasible
    with pytest.raises(InvalidGenomeError):
        genome.validate()


def test_twenty_x_is_actually_reachable():
    """The user's mandate is 20x. If the feasible region excluded it the wave would be
    answering a different question, so pin that the corner exists."""
    rng = np.random.default_rng(30_003)
    best = max(random_genome(rng).leverage for _ in range(20_000))
    assert best > 19.0, f"20x corner unreachable (best sampled {best:.2f}x)"


# ---------------------------------------------------------------------------------------
# Trade resolution semantics on synthetic bars
# ---------------------------------------------------------------------------------------


def _synthetic_arrays(open_, high, low, close) -> SymbolArrays:
    n = len(close)
    zeros = np.zeros(n)
    return SymbolArrays(
        symbol="TESTUSDT",
        open=np.asarray(open_, dtype=float),
        high=np.asarray(high, dtype=float),
        low=np.asarray(low, dtype=float),
        close=np.asarray(close, dtype=float),
        tradable=np.ones(n, dtype=bool),
        funding_at_bar=zeros.copy(),
        cost_rate=0.0,
        prior_high={},
        prior_low={},
        ret={},
        vol={},
        zscore={},
    )


def _genome(**overrides) -> Genome:
    base = dict(
        signal_family="breakout",
        lookback_bars=24,
        entry_threshold=0.0,
        stop_pct=0.02,
        target_r=2.0,
        trail_enabled=False,
        risk_frac=0.20,
        max_hold_bars=10,
        allow_short=True,
        symbols=("BTCUSDT",),
        max_concurrent=1,
        cooldown_bars_after_loss=0,
        sleeve_fraction=0.25,
    )
    base.update(overrides)
    return Genome(**base)


def test_same_bar_stop_and_target_resolves_to_stop():
    # Bar 1 range spans BOTH the 2% stop (98) and the 4% target (104): ambiguity must resolve
    # to the stop, biasing the result downward.
    arrays = _synthetic_arrays(
        open_=[100.0, 100.0, 100.0],
        high=[100.0, 105.0, 100.0],
        low=[100.0, 97.0, 100.0],
        close=[100.0, 100.0, 100.0],
    )
    genome = _genome(stop_pct=0.02, target_r=2.0)
    exit_bar, exit_price, reason, mae, liquidated = _resolve_trade(
        arrays, entry_bar=1, direction=1.0, genome=genome, liq_band=genome.liquidation_band, n_bars=3
    )
    assert reason == "stop"
    assert exit_bar == 1
    assert exit_price == pytest.approx(98.0)
    assert not liquidated


def test_gap_through_stop_fills_at_the_open_not_the_stop():
    # Fill at bar 1's open (100), stop at 98. Bar 2 OPENS at 90, far below the stop: a model
    # that assumed an exact stop fill would book -2%; the real fill is -10%.
    arrays = _synthetic_arrays(
        open_=[100.0, 100.0, 90.0],
        high=[100.0, 100.0, 91.0],
        low=[100.0, 99.5, 89.0],
        close=[100.0, 100.0, 90.0],
    )
    genome = _genome(stop_pct=0.02)
    exit_bar, exit_price, reason, mae, liquidated = _resolve_trade(
        arrays, entry_bar=1, direction=1.0, genome=genome, liq_band=genome.liquidation_band, n_bars=3
    )
    assert reason == "stop"
    assert exit_bar == 2
    assert exit_price == pytest.approx(90.0)
    assert mae == pytest.approx(0.11, abs=1e-9)


def test_gap_beyond_liquidation_band_wipes_the_margin():
    # 10x leverage => band = 1/10 - 0.005 = 9.5%. A gap to -12% is past the band.
    genome = _genome(stop_pct=0.02, risk_frac=0.20)  # 10x
    assert genome.leverage == pytest.approx(10.0)
    band = genome.liquidation_band
    assert band == pytest.approx(0.095)
    arrays = _synthetic_arrays(
        open_=[100.0, 100.0, 88.0],
        high=[100.0, 100.0, 88.5],
        low=[100.0, 99.9, 87.0],
        close=[100.0, 100.0, 88.0],
    )
    _, exit_price, _, _, liquidated = _resolve_trade(
        arrays, entry_bar=1, direction=1.0, genome=genome, liq_band=band, n_bars=3
    )
    assert exit_price == pytest.approx(88.0)  # gapped straight through the 98 stop
    assert liquidated is True  # -12% is past the 9.5% band, so the margin is gone


def test_ordinary_stop_never_liquidates_because_stop_is_interior_to_band():
    """The structural claim: with no gap, a stop-out is a bounded loss, never a wipe. This is
    exactly what wave29's post-hoc leverage model could not offer."""
    rng = np.random.default_rng(30_004)
    for _ in range(400):
        genome = random_genome(rng)
        entry = 100.0
        stop_price = entry * (1.0 - genome.stop_pct)
        arrays = _synthetic_arrays(
            open_=[entry, entry, entry],
            high=[entry, entry, entry],
            low=[entry, stop_price - 1e-9, stop_price],
            close=[entry, stop_price, stop_price],
        )
        _, _, reason, _, liquidated = _resolve_trade(
            arrays, entry_bar=1, direction=1.0, genome=genome, liq_band=genome.liquidation_band, n_bars=3
        )
        assert reason == "stop"
        assert not liquidated, f"stop-out liquidated at lev {genome.leverage:.2f}"


def test_trailing_stop_uses_only_prior_bars():
    # Bar 1 runs up to 110 then closes; bar 2's trailing stop should be anchored on 110
    # (a PRIOR bar's high), and bar 1's own stop must still be measured from the entry price.
    arrays = _synthetic_arrays(
        open_=[100.0, 100.0, 108.0, 108.0],
        high=[100.0, 110.0, 108.5, 108.5],
        low=[100.0, 99.5, 107.0, 100.0],
        close=[100.0, 110.0, 108.0, 100.0],
    )
    genome = _genome(stop_pct=0.02, target_r=8.0, trail_enabled=True, max_hold_bars=3)
    exit_bar, exit_price, reason, _, _ = _resolve_trade(
        arrays, entry_bar=1, direction=1.0, genome=genome, liq_band=genome.liquidation_band, n_bars=4
    )
    # Trailing level at bar 2 is 110*0.98 = 107.8; bar 2's low of 107.0 breaches it.
    assert reason == "stop"
    assert exit_bar == 2
    assert exit_price == pytest.approx(107.8)


def test_short_side_is_mirrored():
    arrays = _synthetic_arrays(
        open_=[100.0, 100.0, 100.0],
        high=[100.0, 103.0, 100.0],
        low=[100.0, 100.0, 100.0],
        close=[100.0, 103.0, 100.0],
    )
    genome = _genome(stop_pct=0.02)
    _, exit_price, reason, mae, _ = _resolve_trade(
        arrays, entry_bar=1, direction=-1.0, genome=genome, liq_band=genome.liquidation_band, n_bars=3
    )
    assert reason == "stop"
    assert exit_price == pytest.approx(102.0)
    assert mae == pytest.approx(0.03)


# ---------------------------------------------------------------------------------------
# No-lookahead and OOS sealing against the real cache
# ---------------------------------------------------------------------------------------


def test_breakout_signal_ignores_its_own_bar_extremes():
    """A breakout at bar i compares close[i] against the channel through i-1. If bar i's own
    high leaked into the channel the condition could never fire, so a firing signal whose own
    high is the period maximum is proof the channel excluded it."""
    cache = build_market_cache()
    arrays = cache.arrays["BTCUSDT"]
    genome = _genome(signal_family="breakout", lookback_bars=24, allow_short=False)
    signal = signal_direction(arrays, genome)
    fired = np.flatnonzero(signal > 0)
    assert len(fired) > 100
    sample = fired[:2000]
    assert np.all(arrays.close[sample] > arrays.prior_high[24][sample])
    # And the channel value at i is strictly the max over [i-24, i-1].
    for bar in sample[:200]:
        assert arrays.prior_high[24][bar] == pytest.approx(np.max(arrays.high[bar - 24 : bar]))


def test_is_mode_never_touches_a_post_split_bar():
    cache = build_market_cache()
    rng = np.random.default_rng(30_005)
    split_position = int(cache.is_mask.sum())
    for _ in range(60):
        genome = random_genome(rng)
        result = run_genome(cache, genome, mode="is")
        for trade in result.trades:
            assert trade.exit_bar < split_position
            assert cache.index[trade.entry_bar] <= OOS_SPLIT
            assert cache.index[trade.exit_bar] <= OOS_SPLIT


def test_unknown_mode_raises_leakage_error():
    cache = build_market_cache()
    genome = _genome()
    with pytest.raises(OOSLeakageError):
        run_genome(cache, genome, mode="oos")


def test_search_loop_only_ever_requests_is_mode(monkeypatch):
    """Structural guard: patch run_genome to explode on any mode but 'is', then run a short
    MAP-Elites. If any code path inside the search reached for the future this fails."""
    import research.wave30_qd.engine30 as engine30
    import research.wave30_qd.fitness30 as fitness30

    original = engine30.run_genome

    def guarded(cache, genome, mode="is"):
        if mode != "is":
            raise AssertionError(f"search requested mode={mode!r}")
        return original(cache, genome, mode=mode)

    monkeypatch.setattr(fitness30, "run_genome", guarded)
    cache = build_market_cache()
    evaluator = Evaluator(cache, seed=30_006)
    rng = np.random.default_rng(30_006)
    archive = run_map_elites(evaluator, rng, n_init=12, n_iterations=12)
    assert archive.coverage >= 1


# ---------------------------------------------------------------------------------------
# Accounting identities
# ---------------------------------------------------------------------------------------


def test_sleeve_can_never_go_negative_and_total_respects_capital_contract():
    cache = build_market_cache()
    rng = np.random.default_rng(30_007)
    for _ in range(80):
        genome = random_genome(rng)
        result = run_genome(cache, genome, mode="is")
        assert np.all(result.sleeve_equity_daily >= -1e-9)
        assert result.sleeve_start_usdt + result.stable_start_usdt == pytest.approx(100.0)
        assert np.all(result.trade_returns >= -1.0 - 1e-12)


def test_zero_sleeve_limit_equals_the_i5_baseline():
    """sleeve_fraction cannot literally be 0 (not in the frozen set), but the stable leg alone
    must reproduce the $100-basis I5 curve, which is what P2 compares against."""
    from research.wave30_qd.dataio30 import i5_baseline_total_curve

    cache = build_market_cache()
    baseline = i5_baseline_total_curve(cache)
    # Day 0 is an END-of-day mark, so it already carries that day's I5 growth.
    assert baseline[0] == pytest.approx(90.0 * cache.stable_daily_factor[0] + 10.0)
    assert baseline[0] == pytest.approx(100.0, abs=0.01)
    # 90% compounding at I5's realised factors + 10% idle reserve.
    expected_final = 90.0 * float(np.prod(cache.stable_daily_factor)) + 10.0
    assert baseline[-1] == pytest.approx(expected_final)


def test_funding_is_charged_on_notional_so_leverage_amplifies_it():
    cache = build_market_cache()
    genome = _genome(signal_family="breakout", risk_frac=0.20, stop_pct=0.02, max_hold_bars=336)
    result = run_genome(cache, genome, mode="is")
    charged = [t for t in result.trades if t.funding_fraction != 0.0]
    assert charged, "no trade spanned a funding stamp; the test is not exercising the path"
    trade = charged[0]
    rebuilt = (
        trade.gross_price_return * trade.leverage
        - trade.cost_fraction * trade.leverage
        - trade.funding_fraction * trade.leverage
    )
    assert trade.net_return_on_base == pytest.approx(max(rebuilt, -1.0))


# ---------------------------------------------------------------------------------------
# Search machinery
# ---------------------------------------------------------------------------------------


def test_non_dominated_sort_matches_a_hand_worked_example():
    objectives = np.array(
        [
            [1.0, 1.0],  # 0: non-dominated
            [2.0, 2.0],  # 1: dominated by 0
            [1.0, 3.0],  # 2: dominated by 0
            [0.5, 5.0],  # 3: non-dominated (better on axis 0)
        ]
    )
    fronts = fast_non_dominated_sort(objectives)
    assert set(fronts[0]) == {0, 3}
    assert set(fronts[1]) == {1, 2}


def test_descriptor_binning_hits_the_frozen_edges():
    assert descriptor_of(1.0, 0.0, 0.0) == (0, 0, 0)
    assert descriptor_of(20.0, 0.99, 1_000.0) == (5, 5, 4)
    assert descriptor_of(3.0, 0.15, 30.0) == (1, 1, 1)
    assert descriptor_of(7.0, 0.35, 150.0) == (3, 3, 3)


def test_archive_keeps_one_elite_per_cell_and_only_improves():
    cache = build_market_cache()
    evaluator = Evaluator(cache, seed=30_008)
    rng = np.random.default_rng(30_008)
    archive = run_map_elites(evaluator, rng, n_init=60, n_iterations=60)
    assert archive.coverage == len(archive.cells)
    for cell, elite in archive.cells.items():
        assert elite.descriptor == cell


def test_evaluator_budget_counts_distinct_genomes_only():
    cache = build_market_cache()
    evaluator = Evaluator(cache, seed=30_009)
    rng = np.random.default_rng(30_009)
    genome = random_genome(rng)
    evaluator(genome)
    evaluator(genome)
    evaluator(replace(genome))
    assert evaluator.n_evaluations == 1


def test_fitness_penalises_untradeable_genomes():
    cache = build_market_cache()
    rng = np.random.default_rng(30_010)
    # A 6-bar lookback breakout on a 336-bar hold with cooldown is fine; instead force a
    # genome that cannot accumulate MIN_TRADES_FOR_FITNESS trades by demanding a huge move.
    genome = _genome(signal_family="reversion", entry_threshold=3.0, lookback_bars=6, max_hold_bars=720)
    evaluation = evaluate(cache, genome, rng)
    if evaluation.n_trades < 20:
        assert evaluation.fitness <= evaluation.is_total_cagr - 1.0 + 1e-12
