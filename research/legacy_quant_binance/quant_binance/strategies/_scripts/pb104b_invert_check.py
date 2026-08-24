"""Sanity check: invert PB104b direction (trade WITH cascade instead of reversal).
If reversal is -32 bps, with-cascade should be ~+32 bps minus 2*cost = ~0 to slightly positive.
This isolates whether the cost is what's killing it vs the signal being random noise."""
from __future__ import annotations

import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from pb104b_alt_liquidation import (  # type: ignore
    load_liq, fetch_klines_5m, klines_to_dict, compute_atr,
    detect_signals, simulate_trade, Signal,
)

SYMBOLS = ["ETHUSDT", "SOLUSDT", "DOGEUSDT", "WIFUSDT", "1000PEPEUSDT"]


def invert(sig: Signal) -> Signal:
    return Signal(
        ts=sig.ts, symbol=sig.symbol,
        direction="SHORT" if sig.direction == "LONG" else "LONG",
        cascade=sig.cascade + "_INV", drop=sig.drop, sell_dom=sig.sell_dom,
    )


for s in SYMBOLS:
    liq = load_liq(s)
    ts0, ts1 = liq[0]["timestamp"], liq[-1]["timestamp"] + 30 * 60 * 1000
    kl = klines_to_dict(fetch_klines_5m(s, ts0, ts1))
    atr = compute_atr(sorted(kl.keys()), kl, 14)
    sigs = detect_signals(liq, lookback_bars=3, drop_th=0.01, sell_dom=1.3, cooldown_bars=3)

    rev_nets, inv_nets = [], []
    rev_raws, inv_raws = [], []
    for sig in sigs:
        tr_rev = simulate_trade(sig, kl, atr, tp_atr_mult=1.5, sl_atr_mult=0.8, timeout_bars=2)
        tr_inv = simulate_trade(invert(sig), kl, atr, tp_atr_mult=1.5, sl_atr_mult=0.8, timeout_bars=2)
        if tr_rev:
            rev_nets.append(tr_rev.net_ret); rev_raws.append(tr_rev.raw_ret)
        if tr_inv:
            inv_nets.append(tr_inv.net_ret); inv_raws.append(tr_inv.raw_ret)

    if rev_nets and inv_nets:
        print(f"{s}: n={len(rev_nets)}")
        print(f"   reversal: avg_raw={statistics.mean(rev_raws)*1e4:+7.2f} bps  avg_net={statistics.mean(rev_nets)*1e4:+7.2f} bps")
        print(f"   with-csc: avg_raw={statistics.mean(inv_raws)*1e4:+7.2f} bps  avg_net={statistics.mean(inv_nets)*1e4:+7.2f} bps")
