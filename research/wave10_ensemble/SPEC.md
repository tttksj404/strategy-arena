# Wave-10 ensemble and risk-overlay specification

Preregistered: 2026-07-21 (Asia/Seoul)

Wave-9 found no single method that passed the $100 risk gates. Its strongest complementary signals were downside-volatility spread (D9b), capital-aware cross-sectional time-series momentum, residual trend (P9b), and the earlier funding-guarded spread (F8d). This wave tests fixed combinations and one fixed drawdown throttle. It is an exploratory follow-up, not a live recommendation: the component waves were already inspected, so `selection_independent=false` is a hard gate.

## Registered candidates

- E10a: 50% D9b + 50% F8d.
- E10b: 50% D9b + 50% M10a capital-aware top-3 20-day cross-sectional momentum.
- E10c: 50% D9b + 50% P9b residual trend.
- E10d: 40% D9b + 30% M10a + 30% F8d.
- E10e: E10a with a 50% gross throttle after a prior equity drawdown greater than 10%.
- E10f: E10b with the same fixed drawdown throttle.

Weights are applied to component positions before PnL. All components use the existing fixed universe and daily data. Funding cashflow is included only for the F8d leg and is assigned to the following close-to-close holding interval. The throttle is lagged one day, uses the realized scaled equity including scaled funding, and never uses the current return.

M10a is registered in this wave as `long top-3 / short bottom-3 by prior 20-day cross-sectional return`, with 0.60 gross before its ensemble weight. It is not an unregistered Wave-9 candidate.

## Common contract and gates

The $100 initial capital, $10 reserve, 0.60 maximum gross, $5 minimum order, 0.06% fee + 0.03% slippage base cost, doubled-slippage stress, 2025-10-01 internal OOS split, MC 10,000 paths, 90-day block MDD, historical MDD, ruin, stress, positive-block, Sharpe, and DSR gates match Wave-8/9. All six candidates must also pass `selection_independent`, which is false in this retrospective ensemble experiment. No exchange credentials or live order path is included.

The purpose is to identify a candidate for a genuinely fresh prospective paper window. A positive blended backtest is not permission to deploy $100.
