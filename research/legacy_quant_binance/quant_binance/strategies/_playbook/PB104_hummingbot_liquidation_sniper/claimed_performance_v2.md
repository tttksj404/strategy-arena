# PB104b - 9-point validation (v2: alt + short hold)

**Status: DEAD (4/9 — same score as PB104 v1; sweep confirms no positive edge)**

_2026-04-28 16:59:32 / 30-day backtest_

## Overall

| metric | value |
|---|---:|
| n | 473 |
| win_rate | 0.0951 |
| avg_net_bps | -32.4598 |
| median_net_bps | -33.2210 |
| stdev_net_bps | 25.4141 |
| tp_rate | 0.0825 |
| sl_rate | 0.2008 |
| to_rate | 0.7167 |
| liq_rate | 0.0000 |
| avg_bars_held | 1.8710 |
| total_pnl_usd | -126.6662 |
| longs | 249 |
| shorts | 224 |
| avg_tp_pct | 0.0048 |
| avg_sl_pct | 0.0032 |

## Per-Symbol

| Symbol | Signals | Trades | WR | avg_net_bps | tp_rate | sl_rate | to_rate | avg_tp% | avg_sl% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ETHUSDT | 134 | 134 | 8.21% | -30.4 | 5.97% | 11.94% | 82.09% | 0.40 | 0.31 |
| SOLUSDT | 84 | 84 | 4.76% | -34.7 | 9.52% | 22.62% | 67.86% | 0.44 | 0.32 |
| DOGEUSDT | 68 | 68 | 4.41% | -33.4 | 7.35% | 17.65% | 75.00% | 0.37 | 0.30 |
| WIFUSDT | 94 | 94 | 13.83% | -34.2 | 6.38% | 26.60% | 67.02% | 0.62 | 0.36 |
| 1000PEPEUSDT | 93 | 93 | 15.05% | -30.9 | 12.90% | 24.73% | 62.37% | 0.54 | 0.34 |

## Cost sweep (round-trip bps)
| cost | avg_net_bps |
|---:|---:|
| 16 | -32.46 |
| 20 | -40.46 |
| 25 | -50.46 |
| 30 | -60.46 |
| 40 | -80.46 |

## Sub-periods (split into 3)
| Period | n | avg_net_bps | ann_on_margin |
|---|---:|---:|---:|
| P1 | 158 | -28.43 | -8616.22% |
| P2 | 158 | -35.0 | -12379.02% |
| P3 | 157 | -33.96 | -8230.11% |

## 9-point checks
| # | Check | Result |
|---:|---|---|
| 1 | subperiod trade-avg net > 0 | FAIL |
| 2 | subperiod annualized portfolio > 0 | FAIL |
| 3 | avg_net >= +50 bps | FAIL |
| 4 | WR >= 65% | FAIL |
| 5 | 6-axis fit (>=3/day + both sides) | PASS |
| 6 | cost up to 30 bps positive | FAIL |
| 7 | 5x liq < 10% | PASS |
| 8 | n >= 50 | PASS |
| 9 | no warmup leakage | PASS |

**Pass: 4/9**
**Trades/day: 16.09**


## Caveats / risk

- Proxy only — no real liquidation feed; maintenance on allForceOrders
- 30-day single regime; no out-of-sample / cross-regime check
- Bitget execution latency / slippage untested
- TP/SL same-bar tie -> SL (conservative)
- ATR computed on signal bar; entry on next bar close — no look-ahead
- Cooldown 15 min may overlap concurrent positions; runner uses max-3 cap

## Parameter sweep verdict (1280 grid)

Swept (drop_th, sell_dom, timeout_bars, tp_atr, sl_atr) over reasonable ranges:
- drop_th in {0.005, 0.01, 0.02, 0.05}
- sell_dom in {1.1, 1.3, 1.5, 2.0}
- timeout_bars in {1, 2, 3, 4, 6} (5..30 min)
- tp_atr in {1.0, 1.5, 2.0, 3.0}
- sl_atr in {0.5, 0.8, 1.5, 3.0}

After excluding combos with n < 30: **1040 valid combos.**

| metric | result |
|---|---:|
| combos with avg_net > 0 | **0 / 1040** |
| combos with avg_net >= 50 bps | **0 / 1040** |
| best combo avg_net_bps | -17.22 (drop=0.01, dom=2.0, to=6, tp=1.5, sl=3.0) |

**Conclusion: directional cascade-reversal thesis is dead on alts.** The proxy
signal (top L/S ratio drop + taker sell dominance) is **not predictive of mean
reversion** at any tested holding period (5..30 min). No parameter tuning rescues
it; the issue is the signal itself, not TP/SL/hold.

## vs PB104 v1

| | PB104 v1 (BTC/ETH/SOL, 30m, flat 1%/3%) | PB104b (alts, 10m, ATR-adapt) |
|---|---:|---:|
| 9-point | 4/9 | 4/9 |
| n | ~50 | 473 |
| WR | 9.9% | 9.5% |
| avg_net_bps | -35.6 | -32.5 |
| timeout rate | 97.8% | 71.7% |
| TP rate | <5% | 8.3% |

Adding alt vol + ATR-adaptive TP raises hit rate (97.8% to 71.7% timeout) but
the underlying directional edge is negative either way. The proxy "L/S ratio
drop + taker imbalance" appears to be a momentum confirmation signal (price
already moved in direction; reversal does NOT follow on 5-30 min horizon).

## Direction-flip diagnostic (zero-edge probe)

Ran the same 473 signals once as **reversal** and once as **with-cascade**
(opposite direction). Pre-cost avg_raw by symbol:

| Symbol | n | reversal raw_bps | with-cascade raw_bps |
|---|---:|---:|---:|
| ETHUSDT | 134 | +1.61 | -2.40 |
| SOLUSDT | 84 | -2.75 | +1.52 |
| DOGEUSDT | 68 | -1.39 | +0.35 |
| WIFUSDT | 94 | -2.25 | -2.62 |
| 1000PEPEUSDT | 93 | +1.11 | -0.02 |

**All raw edges are in the +/-3 bps range.** This is statistical noise
relative to the 32 bps round-trip cost. The proxy signal (top_LS_ratio drop +
taker_sell_dominance over 15 min) carries **essentially zero directional info**
at the 10-min horizon, in either direction. The 4/9 result isn't bad
parameter tuning — **the signal itself is not predictive**.

## Recommendation

- **Park PB104 / PB104b. Do not promote to paper-live.**
- The cascade-reversal idea is fundamentally a real-liquidation play. The
  L/S-ratio + taker-imbalance proxy does NOT reproduce the signal at 5-min
  resolution on either majors (PB104 v1) or alts (PB104b).
- To resurrect: **switch to a true liquidation feed** (Coinglass paid /
  Bybit liquidation WS / direct CEX firehose). Without that, more parameter
  tuning is wasted effort.
- Alternatively: pivot the cascade idea to a different proxy (e.g. spot-perp
  basis spike, OI delta, funding flip). Different signal, different playbook.