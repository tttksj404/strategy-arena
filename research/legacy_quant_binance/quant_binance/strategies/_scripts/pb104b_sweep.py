"""PB104b sensitivity sweep — does any parameter combo show positive edge?"""
from __future__ import annotations

import itertools
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from pb104b_alt_liquidation import (  # type: ignore
    load_liq, fetch_klines_5m, klines_to_dict, compute_atr,
    detect_signals, simulate_trade,
)

SYMBOLS = ["ETHUSDT", "SOLUSDT", "DOGEUSDT", "WIFUSDT", "1000PEPEUSDT"]

# Pre-load all symbol data once
print("Loading all symbol data ...")
DATA: dict = {}
for s in SYMBOLS:
    liq = load_liq(s)
    ts0, ts1 = liq[0]["timestamp"], liq[-1]["timestamp"] + 30 * 60 * 1000
    kl = klines_to_dict(fetch_klines_5m(s, ts0, ts1))
    sorted_ts = sorted(kl.keys())
    atr = compute_atr(sorted_ts, kl, 14)
    DATA[s] = (liq, kl, atr)
    print(f"  {s}: {len(liq)} liq rows, {len(kl)} klines, {len(atr)} atr")
print()


def run_combo(drop_th, sell_dom, timeout_bars, tp_atr, sl_atr) -> dict:
    all_nets = []
    all_wins = 0
    n = 0
    for s in SYMBOLS:
        liq, kl, atr = DATA[s]
        sigs = detect_signals(liq, lookback_bars=3, drop_th=drop_th, sell_dom=sell_dom, cooldown_bars=3)
        for sig in sigs:
            tr = simulate_trade(sig, kl, atr, tp_atr_mult=tp_atr, sl_atr_mult=sl_atr,
                                timeout_bars=timeout_bars)
            if tr is None:
                continue
            n += 1
            all_nets.append(tr.net_ret)
            if tr.net_ret > 0:
                all_wins += 1
    if n == 0:
        return {"n": 0}
    return {
        "n": n,
        "wr": all_wins / n,
        "avg_net_bps": statistics.mean(all_nets) * 10_000,
        "median_bps": statistics.median(all_nets) * 10_000,
    }


GRID = list(itertools.product(
    [0.005, 0.01, 0.02, 0.05],     # drop_th
    [1.1, 1.3, 1.5, 2.0],          # sell_dom
    [1, 2, 3, 4, 6],               # timeout_bars (5, 10, 15, 20, 30 min)
    [1.0, 1.5, 2.0, 3.0],          # tp_atr
    [0.5, 0.8, 1.5, 3.0],          # sl_atr
))

print(f"Sweeping {len(GRID)} combos...")
results = []
for combo in GRID:
    r = run_combo(*combo)
    if r["n"] >= 30:
        results.append((combo, r))

# sort by avg_net_bps
results.sort(key=lambda x: x[1]["avg_net_bps"], reverse=True)
print("\nTop 15 combos by avg_net_bps:")
print(f"{'drop':>6} {'dom':>5} {'to':>3} {'tp':>4} {'sl':>4} {'n':>5} {'WR':>7} {'net_bps':>9}")
for combo, r in results[:15]:
    drop, dom, to, tp, sl = combo
    print(f"{drop:>6.3f} {dom:>5.1f} {to:>3} {tp:>4.1f} {sl:>4.1f} "
          f"{r['n']:>5} {r['wr']:>7.2%} {r['avg_net_bps']:>9.2f}")

positive = [r for r in results if r[1]["avg_net_bps"] > 0]
strong = [r for r in results if r[1]["avg_net_bps"] >= 50]
print(f"\nCombos with avg_net > 0: {len(positive)} / {len(results)}")
print(f"Combos with avg_net >= 50 bps: {len(strong)} / {len(results)}")
