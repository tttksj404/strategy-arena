# Wave-9 method expansion specification

Preregistered: 2026-07-21 (Asia/Seoul)

Wave-8 produced no capital-safe candidate. This wave expands the *method* set rather than tuning Wave-8 windows. It includes both genuinely additional families (range/candle structure and drawdown/correlation filters) and controlled baselines (trend, relative value) so overlap is measured rather than hidden. It is still research evidence, not a live-capital recommendation. The fixed universe, daily bars, cost model, capital contract, and gate thresholds remain unchanged so the result is comparable.

## Data and timing contract

- Source: local Binance USD-M perpetual daily caches already present in `research/wave3/cache`.
- Fixed symbols: BTC, ETH, BNB, ADA, XRP, DOGE, SOL, AVAX, DOT, LINK, LTC, BCH.
- Common continuous interval is discovered from all 12 symbols and must have no missing OHLCV rows.
- Feature at day `t` uses information available through the close/open of day `t-1`; the next close-to-close return is applied at `t`. Every signal is shifted before PnL.
- OOS split: 2025-09-30 23:59:59 UTC; OOS begins 2025-10-01.
- Because this wave was initiated after earlier waves were inspected, `selection_independent` is recorded as false. The historical split is an internal validation split, not proof of a fresh prospective edge.

## Capital and execution contract

- Initial capital: $100; $10 reserve is never notionally allocated.
- Maximum gross exposure: 0.60 of equity; no leverage.
- Minimum order: $5 per leg; equal-weight or inverse-vol weights are clipped by this contract.
- Base cost: 0.06% taker fee + 0.03% slippage per unit turnover. Stress doubles slippage to 0.06%.
- No exchange credentials, order code, or live execution is added.

## Registered method families

### T9: time-series trend and breakout

- T9a: 20-day time-series momentum, sign per asset, gross-normalized.
- T9b: 60-day time-series momentum, sign per asset, gross-normalized.
- T9c: 20-day Donchian breakout using prior rolling high/low, gross-normalized.
- T9d: 20/60 EMA crossover with inverse-volatility weights.

### P9: residual and pair relative value

- P9a: 10-day residual trend after rolling market beta removal.
- P9b: 30-day residual trend after rolling market beta removal.
- P9c: BTC/ETH log-price spread z-score, mean-reverting pair.
- P9d: SOL/ETH log-price spread z-score, mean-reverting pair.

### H9: range, candle, and price-volume structure

- H9a: 5-day ATR-normalized return continuation, cross-sectional.
- H9b: 5-day close-location-value accumulation, cross-sectional.
- H9c: 3-day open-gap reversal, cross-sectional.
- H9d: 5-day price/volume confirmation, cross-sectional.

### D9: drawdown and market-structure filters

- D9a: 30-day drawdown recovery, long the strongest recovery and short the weakest.
- D9b: 30-day downside-volatility spread, long low downside-volatility assets and short high downside-volatility assets.
- D9c: 20-day trend with BTC 200-day regime guard.
- D9d: 30-day trend weighted toward assets with low rolling market correlation.

## Gates

All 16 candidates are evaluated without changing definitions after execution. A candidate is eligible only if all are true: data validation, fixed universe, historical split present, selection-independent flag, capital contract, at least 90 active OOS days, positive OOS return, OOS Sharpe >= 1, at least two positive compounded OOS blocks, positive stress OOS return, MC p05 > $100, ruin probability < 5%, historical MDD <= 25%, 90-day block-shuffle MDD p95 <= 25%, and deflated-Sharpe probability >= 95% using the 16 Wave-9 trials. Since `selection_independent=false` in this exploratory wave, no candidate can be promoted to live capital from this report alone.

Zero survivors is a valid result. A positive OOS return without the capital and drawdown gates is not a strategy recommendation.
