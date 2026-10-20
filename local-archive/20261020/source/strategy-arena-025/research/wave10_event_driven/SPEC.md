# Wave-10 fixed-major event-driven search

## Mission

Wave-9 tested cross-sectional momentum, reversal, and regime variants over a
survivorship-prone altcoin roster. Wave-10 changes both dimensions: it uses a
fixed long-lived major universe (`BTCUSDT`, `ETHUSDT`, `SOLUSDT`) and a
price-breakout event signal. This is a new exploratory family, not a retune of
Wave-9.

## Frozen source and timing

- Source: the verified offline Wave-3 Binance perpetual, daily OHLC, and funding
  cache. No network calls are permitted.
- IS: through `2025-09-30T23:59:59Z`.
- Historical evaluation window: `2025-10-01` through the frozen cache end
  `2026-07-14`. It is not an independent promotion OOS because prior research
  has already observed this window.
- At close `t`, a breakout is evaluated only against highs/lows through `t-1`.
  The position is filled at close `t` after the `t-1` to `t` return and earns
  its first return and funding cashflow on `t+1`.

## Capital and execution contract

- Total capital `$100`; `$10` reserve; `$90` active sleeve.
- One perpetual leg per active major; no spot hedge and no leverage.
- Registered gross caps are `0.90` or `0.60`; fixed quantities are held between
  entry/exit events. A mark-to-market gross circuit reduces quantities when
  drift would exceed the candidate cap.
- Maker fee `0.02%` plus `0.01%` slippage for BTC/ETH/SOL on every notional
  change. Stress doubles slippage. Funding is charged as
  `-position * daily_funding_sum`.
- Every open leg must be at least `$5` in OOS after normalizing each OOS order
  to a fresh `$100` starting equity; historical compounded notionals are
  diagnostic only.

## Signal family

For each symbol, enter long when the close breaks the previous `entry_days`
high; long/short candidates also enter short below the previous `entry_days`
low. Exit long below the previous `exit_days` low and exit short above the
previous `exit_days` high. `carry_aware` candidates only enter long when the
prior three-day mean funding is non-positive and short when it is non-negative.

## Preregistered candidates

| ID | Entry | Exit | Direction | Funding filter | Gross cap |
|---|---:|---:|---|---|---:|
| E10a | 20d | 10d | long-only | none | 0.90 |
| E10b | 20d | 20d | long-only | none | 0.90 |
| E10c | 55d | 20d | long-only | none | 0.90 |
| E10d | 55d | 40d | long-only | none | 0.90 |
| E10e | 20d | 10d | long/short | none | 0.90 |
| E10f | 55d | 20d | long/short | none | 0.90 |
| E10g | 20d | 10d | long-only | carry-aware | 0.60 |
| E10h | 55d | 20d | long/short | carry-aware | 0.60 |

No candidate is selected by OOS return. Every candidate is evaluated and the
historical window is a paper-shortlist screen only. Promotion requires a new
untouched prospective OOS and independent audit.

## Gates

All of the following must pass for a provisional candidate: data validation,
fixed-universe validation, capital/minimum-order contract, at least 60 active
OOS days, positive after-cost-and-funding OOS return, Sharpe `>=1`, at least two
of four compound-positive OOS blocks, positive doubled-slippage return,
10,000-path bootstrap p05 final equity `>$100`, bootstrap ruin `<5%`,
historical MDD `<=25%`, and block-bootstrap MDD p95 `<=25%`. The independent-OOS
gate is intentionally false for this historical screen, so no Wave-10 result
is a live promotion claim.
