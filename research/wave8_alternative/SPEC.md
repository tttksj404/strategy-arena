# Wave-8 alternative alpha preregistration

Status: preregistered 2026-07-21; research only; no live order path.

Wave-1 through Wave-7 already covered funding carry, weekly cross-sectional momentum, fixed-window events, weekend/open spillovers, token-underlying deviation, and carry/momentum blends. This wave registers four different signal sources before execution:

| Family | IDs | Signal source |
|---|---|---|
| R8 | R8a-R8d | short-horizon cross-sectional reversal and market-residual reversal |
| V8 | V8a-V8d | inverse-volatility trend, volatility-conditioned trend, low-volatility spread |
| Q8 | Q8a-Q8d | point-in-time quote-volume shock continuation/reversal |
| F8 | F8a-F8d | cross-sectional funding spread, funding change, and funding-price divergence |

## Fixed data contract

- Binance USDT-M perpetual daily bars and funding history already cached in `research/wave3/cache/`.
- Fixed universe: BTC, ETH, BNB, ADA, XRP, DOGE, SOL, AVAX, DOT, LINK, LTC, BCH.
- The common interval is 2020-09-23 through the last cached bar. Missing or non-contiguous bars fail the data gate; no forward fill is allowed.
- A position on day `t` only uses features shifted from completed data through day `t-1`; the return is the close-to-close return from `t` to `t+1`.
- OOS split is 2025-09-30 23:59:59 UTC; OOS begins 2025-10-01.
- This expansion was initiated after earlier waves were inspected, so `selection_independent=false`; the historical split is an internal validation split and cannot by itself authorize live capital.

## Capital and costs

- Initial capital: $100; $10 reserve; gross exposure cap: 0.60.
- Each selected leg is sized from a 0.60 total gross cap. The minimum order gate is $5 per leg.
- Base round-trip charge is modeled as `0.06% taker fee + 0.03% slippage` per unit turnover. Stress doubles slippage to `0.06%`.
- Funding events are summed by UTC day and assigned to the same close-to-close holding interval as price PnL: `funding_cash[t] = -position[t] * funding[t+1]`; the terminal row has no following interval and is excluded.
- No leverage, no borrow, no exchange credentials, and no live execution are introduced.

## Registered candidates

| ID | Definition |
|---|---|
| R8a | 3-day return: long bottom-3 / short top-3, 0.60 gross |
| R8b | 5-day return: long bottom-3 / short top-3, 0.60 gross |
| R8c | 1-day return: long bottom-3 / short top-3, 0.60 gross |
| R8d | 5-day cross-sectional residual return: long bottom-3 / short top-3 |
| V8a | 14-day sign trend across fixed majors, inverse 20-day-vol weighting |
| V8b | 30-day sign trend across fixed majors, inverse 30-day-vol weighting |
| V8c | 14-day trend with only the lower-volatility half of the universe |
| V8d | 20-day low-volatility long/short spread, bottom-3 vs top-3 |
| Q8a | volume z-score >2 and prior return sign continuation |
| Q8b | volume z-score >2 and prior return sign reversal |
| Q8c | volume z-score >1.5 with 5-day return continuation |
| Q8d | volume z-score <-1 with 5-day return reversal |
| F8a | 3-day funding spread: long lowest funding / short highest funding |
| F8b | 3-day minus 15-day funding change spread |
| F8c | funding-price divergence: long low funding plus negative 3-day return |
| F8d | 7-day funding spread with a 200-day BTC trend (MA200) guard |

## Promotion gates

All of the following must pass: data validation; fixed-universe contract; the selection-independent flag; gross cap and minimum order; at least 90 active OOS days; positive after-cost OOS return; OOS Sharpe >= 1.0; at least two positive OOS calendar blocks; positive doubled-slippage OOS return; Monte Carlo p05 final capital > $100; ruin probability < 5%; historical MDD <= 25%; 90-day block-shuffle MDD p95 <= 25%; and deflated-Sharpe probability >= 95% using 16 registered trials.

Zero survivors is a valid result. No candidate may be recommended for live capital from this wave alone.
