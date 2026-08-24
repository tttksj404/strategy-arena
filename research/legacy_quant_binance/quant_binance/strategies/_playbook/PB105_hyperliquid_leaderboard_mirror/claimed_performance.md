# PB105 — Hyperliquid Mirror: Claimed Performance + 9-Point Verification

**Status**: NOT a production candidate. PoC only.
**Date**: 2026-04-28
**Verifier**: pb105_mirror_backtest.py (deterministic, seed=42)

## TL;DR

**4 / 9 gates passed for the real mirror, 4 / 9 for the structural null
(random-entry baseline). Alpha vs null = +7.12 bps (statistically negligible
given n=142). Production deployment is not justified.**

---

## Data Reality

| Item | Result |
|---|---|
| Hyperliquid `/info` leaderboard endpoint | **Does not exist** (HTTP 422 on `leaderboard`, `leaderBoard`, `topPositions`) |
| Probed candidate addresses | 15 |
| Non-zero equity addresses | 2 |
| Of those, directional (non-MM) | **0** by automated heuristic; whale_a flagged MM-like (922 fph) but is the only fillable directional-style sample |
| `userFillsByTime` per-call cap | 2000 fills |
| Effective backtest depth | 2.17 hours (whale_a) / 0.28 hours (large MM) — **NOT 30 days** |
| Bitget alt overlap of HL fills | 142 / 755 opens (18.8 %) |

**Implication**: The 30-day mirror simulation called for in the spec is
**infeasible with public data alone**. We ran the largest sim possible:
142 evaluable real-mirror entries + 200 structural-null random entries.

---

## Configuration

```
capital            $55
position_pct       30 %
leverage           5x  (notional $82.50 / trade)
fee_roundtrip      4 bps
slippage           5 bps
latency            5 s  (penalty = 0.5 × bar_range × latency_share)
hold_window        4 h
forward klines     Bitget 1h candles, ending 2026-04-28 08:00 UTC
n samples          REAL=142   NULL=200
```

---

## 9-Point Gate Table

| # | Gate | Real Mirror | Null (random) |
|---|---|:--:|:--:|
| 1 | Quarter-by-quarter avg > 0 | ✗ (1.84, −17.5, −15.9, −11.9) | ✗ |
| 2 | Portfolio sim PnL > 0 | ✗ (−$12.75) | ✗ (−$29.72) |
| 3 | Avg net ≥ +50 bps | ✗ (−10.89 bps) | ✗ (−18.01 bps) |
| 4 | WR ≥ 65 % | ✗ (21.1 %) | ✗ (46.5 %) |
| 5 | 6-axis user fit | ✓ (6/6 in source.md) | ✓ |
| 6 | Cost-sensitivity (2× cost still > 0) | ✗ (−19.89 bps) | ✗ |
| 7 | 5x liquidation rate < 10 % | ✓ (0 %) | ✓ (0 %) |
| 8 | n ≥ 50 | ✓ (142) | ✓ (200) |
| 9 | No warmup | ✓ | ✓ |
| | **Total** | **4 / 9** | **4 / 9** |

**Critical: real mirror's win-rate (21.1 %) is BELOW random null (46.5 %).**
Within this 2-hour data window, the whale's actual entries on overlapping
Bitget alts were systematically wrong-side over the 4-hour forward horizon.
This is plausibly because:

- The 2-hour micro-window is dominated by a single INJ regime move.
- Whale opened directional positions that he himself likely closed inside 1 h
  (we held for 4 h, missing his exit).
- 142 opens on INJ alone (single-coin concentration) destroys diversification.

**The structural null beat the real mirror, which means within this dataset
the whale's signal is anti-alpha when held for 4 h.**

---

## Alpha Decay Risk Assessment

| Risk | Severity | Evidence |
|---|---|---|
| Single-trader concentration wipe-out | **High** | Pool size = 1 directional trader (whale_a). One blow-up = 100 % mirror loss. |
| Latency arbitrage erosion | Medium | 5 s lag adds ~0.3 bps avg cost (cheap) but real-world cross-exchange may push 10 s+ |
| Coin universe mismatch | **High** | 81 % of HL fills are illiquid alts (XPL, BERA, MORPHO, etc.) not on Bitget |
| Top trader strategy drift | Medium | Untestable without longitudinal data |
| Insider-flow / regulatory | Low | Mirror is legal but alpha may be informationally driven (untestable) |

---

## Synergy with G135

| Mode | Verdict |
|---|---|
| **Standalone mirror** | Not viable: 4/9 gates, no alpha vs null |
| **G135 + mirror confirm gate** | Not yet validated — would need both signals, but mirror false-positive rate is ~50 %. Could add noise rather than signal. |
| **Mirror as G135 veto** | If whale opens opposite side of G135, defer entry — would need test |

**Recommendation**: Do not deploy PB105 in any form. Re-evaluate only if:
1. A working leaderboard data source (paid Coinglass API or HypurrScan API) is integrated.
2. Tracked trader pool ≥ 5 verified directional accounts with ≥ 30-day P&L > +20 %.
3. A new backtest with ≥ 500 trades over ≥ 14 days shows alpha vs null > +30 bps net.

---

## What this PoC actually validated

- ✓ Hyperliquid Info API mechanics (`clearinghouseState`, `userFillsByTime`)
- ✓ Mirror-engine plumbing (latency penalty, slippage model, forward-return computation)
- ✓ 9-point gate harness (deterministic, reusable)
- ✗ Top-trader directional alpha (no evidence within available data)

## What this PoC did NOT validate

- ✗ 30-day historical performance (data unavailable)
- ✗ Multi-trader ensemble effect (only 1 directional trader found)
- ✗ Real-world Bitget execution latency (paper-only)
- ✗ Hyperliquid leaderboard ranking (endpoint does not exist)

---

## Production language compliance

Per CLAUDE.md verification meta-rule and quant-strategy 9-point checklist:
**this strategy fails 5 of 9 gates and is therefore NOT a production candidate.**
The term "production-ready" is not used in any artifact. PB105 stays in `_playbook/`.
