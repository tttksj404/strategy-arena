# Wave-30 independent cross-check of the judged candidate.
#
# engine30 resolves each trade with vectorised numpy (cumulative extrema, argmax on boolean
# masks). A vectorisation error there would not raise -- it would quietly produce a better
# backtest, which is precisely the failure mode that matters when the headline is "$100 became
# $230,504". This module re-simulates the SAME genome with a deliberately naive, explicit
# bar-by-bar Python loop written from the SPEC's rules rather than from engine30's code, and
# asserts the two agree trade-for-trade.
#
# Deliberate differences in style (so a shared mistake is unlikely): no cumulative arrays, no
# argmax, stop/target checked with plain `if` statements in bar order, trailing extreme carried
# in a scalar.

from __future__ import annotations

import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

import numpy as np

from research.wave30_qd.dataio30 import build_market_cache
from research.wave30_qd.engine30 import run_genome, signal_direction
from research.wave30_qd.genome30 import Genome
from research.wave30_qd.run_wave30 import RESULTS_DIR, _genome_from_dict


def naive_trade_list(cache, genome: Genome, horizon: int) -> list[dict]:
    """Explicit bar-by-bar re-implementation. Single symbol, single concurrent position only --
    which is what the judged candidate is; this function refuses anything else rather than
    pretending to cover cases it does not."""
    if len(genome.symbols) != 1 or genome.max_concurrent != 1:
        raise NotImplementedError("naive checker covers the single-symbol / single-slot case only")
    symbol = genome.symbols[0]
    arrays = cache.arrays[symbol]
    signal = signal_direction(arrays, genome)
    band = genome.liquidation_band

    trades: list[dict] = []
    bar = 1
    cooldown_until = -1
    while bar < horizon:
        if signal[bar - 1] == 0 or bar <= cooldown_until:
            bar += 1
            continue
        direction = float(signal[bar - 1])
        entry = float(arrays.open[bar])
        if not np.isfinite(entry) or entry <= 0.0:
            bar += 1
            continue

        target = entry * (1.0 + direction * genome.stop_pct * genome.target_r)
        best_favourable = entry  # trailing anchor, carried as a plain scalar
        exit_bar = None
        exit_price = None
        reason = None
        worst = 0.0
        last = min(horizon - 1, bar + genome.max_hold_bars)
        for j in range(bar, last + 1):
            high = float(arrays.high[j])
            low = float(arrays.low[j])
            open_j = float(arrays.open[j])
            if direction > 0:
                worst = max(worst, (entry - low) / entry)
                stop_price = (best_favourable if genome.trail_enabled else entry) * (1.0 - genome.stop_pct)
                hit_stop = low <= stop_price
                hit_target = high >= target
            else:
                worst = max(worst, (high - entry) / entry)
                stop_price = (best_favourable if genome.trail_enabled else entry) * (1.0 + genome.stop_pct)
                hit_stop = high >= stop_price
                hit_target = low <= target

            if hit_stop:  # stop wins ties, checked first on purpose
                if direction > 0:
                    exit_price = stop_price if open_j > stop_price else open_j
                else:
                    exit_price = stop_price if open_j < stop_price else open_j
                exit_bar, reason = j, "stop"
                break
            if hit_target:
                exit_price = target
                if direction > 0 and open_j > target:
                    exit_price = open_j
                elif direction < 0 and open_j < target:
                    exit_price = open_j
                exit_bar, reason = j, "target"
                break
            # Trailing anchor advances only AFTER this bar has been tested, so bar j's own
            # extreme can never tighten bar j's own stop.
            if direction > 0:
                best_favourable = max(best_favourable, high)
            else:
                best_favourable = min(best_favourable, low)
        if exit_bar is None:
            exit_bar, reason = last, "max_hold"
            exit_price = float(arrays.close[last])

        gross = direction * (exit_price / entry - 1.0)
        liquidated = gross <= -band
        funding = direction * float(arrays.funding_at_bar[bar : exit_bar + 1].sum())
        cost = 2.0 * arrays.cost_rate
        net = -1.0 if liquidated else max(gross * genome.leverage - cost * genome.leverage - funding * genome.leverage, -1.0)
        trades.append(
            {
                "entry_bar": bar,
                "exit_bar": exit_bar,
                "entry_price": entry,
                "exit_price": float(exit_price),
                "reason": "liquidation" if liquidated else reason,
                "net": net,
                "mae": worst,
            }
        )
        cooldown_until = exit_bar + (genome.cooldown_bars_after_loss if net < 0.0 else 0)
        bar = max(exit_bar + 1, cooldown_until + 1)
    return trades


def main() -> int:
    cache = build_market_cache()
    final = json.loads((RESULTS_DIR / "final.json").read_text(encoding="utf-8"))
    genome = _genome_from_dict(final["candidate"]["genome"])

    for mode, horizon in (("is", int(cache.is_mask.sum())), ("full", cache.n_bars)):
        engine = run_genome(cache, genome, mode=mode)
        naive = naive_trade_list(cache, genome, horizon)
        print(f"[{mode}] engine trades {len(engine.trades)} | naive trades {len(naive)}")
        assert len(engine.trades) == len(naive), "trade counts disagree"
        worst_diff = 0.0
        for a, b in zip(engine.trades, naive):
            assert a.entry_bar == b["entry_bar"], f"entry bar {a.entry_bar} vs {b['entry_bar']}"
            assert a.exit_bar == b["exit_bar"], f"exit bar {a.exit_bar} vs {b['exit_bar']}"
            assert a.exit_reason == b["reason"], f"reason {a.exit_reason} vs {b['reason']}"
            assert abs(a.exit_price - b["exit_price"]) < 1e-9 * max(1.0, abs(b["exit_price"]))
            worst_diff = max(worst_diff, abs(a.net_return_on_base - b["net"]))
        print(f"        max |net return| disagreement = {worst_diff:.3e}")
        assert worst_diff < 1e-12

        wins = sum(1 for t in engine.trades if t.net_return_on_base > 0)
        equity = 100.0 * genome.sleeve_fraction
        for t in engine.trades:
            equity = max(0.0, equity * (1.0 + t.net_return_on_base))
        print(f"        wins {wins}/{len(engine.trades)} ({wins/len(engine.trades)*100:.1f}%) "
              f"| sleeve ${100.0*genome.sleeve_fraction:.2f} -> ${equity:,.2f}")
        reasons: dict[str, int] = {}
        for t in engine.trades:
            reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
        print(f"        exit reasons {reasons} | max MAE {max(t.mae for t in engine.trades)*100:.2f}% "
              f"vs liquidation band {genome.liquidation_band*100:.2f}%")

    print()
    print("Independent naive re-simulation agrees with engine30 trade-for-trade.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
