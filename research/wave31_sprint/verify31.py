# Wave-31 independent cross-check of the judged candidate.
#
# verify30.py covered the single-symbol / single-slot path. This candidate trades THREE symbols
# with max_concurrent=1, which exercises engine30's cross-symbol candidate selection and the
# searchsorted pointer-jumping that the single-symbol test never touched. Since the headline is
# "$100 -> $1,338 with the OOS holding up", that path gets its own from-scratch check.
#
# Written deliberately differently from engine30: no numpy cumulative arrays, no argmax, no
# pointer arithmetic -- a plain scan over bars with a single `open_until` scalar. Tie-breaking
# on simultaneous signals reproduces engine30's rule (first symbol in genome.symbols order
# wins, because next_candidate() only replaces the incumbent on a STRICTLY earlier bar).

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
from research.wave30_qd.run_wave30 import _genome_from_dict

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def naive_multi_symbol_trades(cache, genome: Genome, horizon: int) -> list[dict]:
    if genome.max_concurrent != 1:
        raise NotImplementedError("this checker covers the single-slot case only")
    signals = {s: signal_direction(cache.arrays[s], genome) for s in genome.symbols}
    band = genome.liquidation_band
    trades: list[dict] = []
    cooldown_until = {s: -1 for s in genome.symbols}
    bar = 1
    while bar < horizon:
        picked = None
        for symbol in genome.symbols:  # order matters: first symbol wins a tie
            if signals[symbol][bar - 1] != 0 and bar > cooldown_until[symbol]:
                picked = (symbol, float(signals[symbol][bar - 1]))
                break
        if picked is None:
            bar += 1
            continue
        symbol, direction = picked
        arrays = cache.arrays[symbol]
        entry = float(arrays.open[bar])
        if not np.isfinite(entry) or entry <= 0.0 or not arrays.tradable[bar]:
            bar += 1
            continue

        stop_price = entry * (1.0 - direction * genome.stop_pct)
        target = entry * (1.0 + direction * genome.stop_pct * genome.target_r)
        last = min(horizon - 1, bar + genome.max_hold_bars)
        exit_bar = exit_price = reason = None
        worst = 0.0
        for j in range(bar, last + 1):
            high, low, open_j = float(arrays.high[j]), float(arrays.low[j]), float(arrays.open[j])
            if direction > 0:
                worst = max(worst, (entry - low) / entry)
                hit_stop, hit_target = low <= stop_price, high >= target
            else:
                worst = max(worst, (high - entry) / entry)
                hit_stop, hit_target = high >= stop_price, low <= target
            if hit_stop:
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
        if exit_bar is None:
            exit_bar, reason, exit_price = last, "max_hold", float(arrays.close[last])

        gross = direction * (exit_price / entry - 1.0)
        liquidated = gross <= -band
        funding = direction * float(arrays.funding_at_bar[bar : exit_bar + 1].sum())
        cost = 2.0 * arrays.cost_rate
        net = -1.0 if liquidated else max(
            gross * genome.leverage - cost * genome.leverage - funding * genome.leverage, -1.0
        )
        trades.append(
            {
                "symbol": symbol,
                "entry_bar": bar,
                "exit_bar": exit_bar,
                "exit_price": float(exit_price),
                "reason": "liquidation" if liquidated else reason,
                "net": net,
                "mae": worst,
            }
        )
        if net < 0.0:
            cooldown_until[symbol] = exit_bar + genome.cooldown_bars_after_loss
        bar = exit_bar + 1  # single slot: nothing can open until this one closes
    return trades


def main() -> int:
    cache = build_market_cache()
    final = json.loads((RESULTS_DIR / "final.json").read_text(encoding="utf-8"))
    genome = _genome_from_dict(final["candidate"]["genome"])
    print(f"candidate: {genome.signal_family} lev {genome.leverage:.4f}x symbols {genome.symbols} "
          f"concurrent {genome.max_concurrent}")

    for mode, horizon in (("is", int(cache.is_mask.sum())), ("full", cache.n_bars)):
        engine = run_genome(cache, genome, mode=mode)
        naive = naive_multi_symbol_trades(cache, genome, horizon)
        print(f"\n[{mode}] engine {len(engine.trades)} trades | naive {len(naive)} trades")
        assert len(engine.trades) == len(naive), "trade counts disagree"
        worst = 0.0
        for a, b in zip(engine.trades, naive):
            assert a.symbol == b["symbol"], f"symbol {a.symbol} vs {b['symbol']} at bar {a.entry_bar}"
            assert a.entry_bar == b["entry_bar"] and a.exit_bar == b["exit_bar"]
            assert a.exit_reason == b["reason"]
            assert abs(a.exit_price - b["exit_price"]) < 1e-9 * max(1.0, abs(b["exit_price"]))
            worst = max(worst, abs(a.net_return_on_base - b["net"]))
        print(f"        max |net return| disagreement = {worst:.3e}")
        assert worst < 1e-12

        returns = np.array([t.net_return_on_base for t in engine.trades])
        equity = 100.0 * genome.sleeve_fraction
        trough = equity
        peak = equity
        for r in returns:
            equity = max(0.0, equity * (1.0 + r))
            peak = max(peak, equity)
            trough = min(trough, equity)
        reasons: dict[str, int] = {}
        for t in engine.trades:
            reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
        by_symbol: dict[str, int] = {}
        for t in engine.trades:
            by_symbol[t.symbol] = by_symbol.get(t.symbol, 0) + 1
        print(f"        wins {(returns>0).sum()}/{len(returns)} ({(returns>0).mean()*100:.1f}%) "
              f"| mean {returns.mean()*100:+.3f}% | best {returns.max()*100:+.1f}% | worst {returns.min()*100:+.1f}%")
        print(f"        realised sleeve ${100.0*genome.sleeve_fraction:.2f} -> ${equity:,.2f} "
              f"(peak ${peak:,.2f}, lowest ${trough:,.2f})")
        print(f"        exits {reasons} | per symbol {by_symbol}")
        print(f"        max MAE {max(t.mae for t in engine.trades)*100:.2f}% vs band {genome.liquidation_band*100:.2f}%")

    print("\nIndependent naive re-simulation agrees with engine30 trade-for-trade "
          "(3 symbols, single slot).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
