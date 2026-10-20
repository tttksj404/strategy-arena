# Wave-11 multi-family strategy screen

Wave-11 is a new, fixed-universe search after Wave-10's Donchian event family
produced no promotion candidate. It tests four different return sources on
BTCUSDT, ETHUSDT, and SOLUSDT: close-only RSI mean reversion, Bollinger band
volatility expansion, funding/price crowding reversal, and spot-perpetual basis
relative value.

## Frozen data and timing

- Source is the verified offline Wave-3 Binance cache: daily perpetual OHLC,
  daily spot close, and 8-hour funding aggregated to daily sums.
- Universe is fixed to `BTCUSDT`, `ETHUSDT`, `SOLUSDT`; there is no roster
  optimization or symbol selection.
- Historical screen ends `2025-09-30`; evaluation is `2025-10-01` through the
  latest cached date. This interval has been observed by prior research, so
  `oos_independence=false` and no live promotion is possible.
- Every signal is shifted one day. A signal observed at close `t` affects the
  next return at `t+1`; funding and spot/perp spread returns use the same lag.

## $100 capital and execution contract

- Total capital `$100`, `$10` reserve, `$90` active sleeve, no leverage.
- Directional candidates use one perpetual leg with gross cap `0.60`.
- Basis candidates use two equal-notional legs with total gross cap `0.60`.
- Maker fee is `0.02%` per changed leg plus cached symbol slippage; stress
  doubles slippage. Basis trades include an `8%` annualized spot-borrow charge
  only when the spot leg is short.
- Minimum order is `$5` per leg after normalizing each OOS order to a fresh
  `$100` start. Historical compounded notionals are diagnostic only.
- No live API, wallet, key, or order is used.

## Registered candidates

| ID | Family | Frozen rule |
|---|---|---|
| R11a | RSI | RSI(2) <=10 long, exit >=50, max hold 5d |
| R11b | RSI | RSI(2) <=10 long / >=90 short, exit at 50, max hold 5d |
| R11c | RSI | RSI(2) <=5 long, exit >=50, max hold 3d |
| R11d | RSI | RSI(2) <=5 long / >=95 short, exit at 50, max hold 3d |
| B11a | Bollinger | 20d band, 1.5 sigma breakout, long/short, exit at mid, max hold 10d |
| B11b | Bollinger | 20d band, 2.0 sigma breakout, long/short, exit at mid, max hold 10d |
| B11c | Bollinger | 40d band, 1.5 sigma breakout, long/short, exit at mid, max hold 15d |
| B11d | Bollinger | 40d band, 2.0 sigma breakout, long only, exit at mid, max hold 15d |
| F11a | Funding reversal | 30d funding z <=-1.5 with negative 3d return long / mirror short, hold 3d |
| F11b | Funding reversal | same, z threshold 2.0, hold 3d |
| F11c | Funding reversal | threshold 1.5, hold 5d |
| F11d | Funding reversal | threshold 2.0, hold 5d |
| V11a | Basis RV | basis z(10d), entry 1.5, exit 0.5, max hold 3d |
| V11b | Basis RV | basis z(20d), entry 1.5, exit 0.5, max hold 5d |
| V11c | Basis RV | basis z(30d), entry 2.0, exit 0.5, max hold 5d |
| V11d | Basis RV | basis z(20d), entry 2.0, exit 0.25, max hold 10d |

## Promotion gates

Every candidate must pass data validation, fixed universe, independent OOS,
capital/minimum-order feasibility, at least 60 active OOS days, positive
after-cost OOS return, Sharpe `>=1`, at least two of four OOS compound blocks
positive, positive double-slippage return, bootstrap p05 terminal equity `>$100`,
bootstrap ruin `<5%`, historical MDD `<=25%`, and block-shuffle MDD p95 `<=25%`.

The acceptance output is fail-closed: because the OOS interval is not
independent, the eligible set must remain empty even if a historical candidate
passes numerical gates.
