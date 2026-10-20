# Wave-10 fixed-major event-driven report

- Strategy family: fixed BTC/ETH/SOL Donchian breakout with optional funding filter.
- Source manifest SHA-256: `2d1aa4772739cc0bab209a27a5684900db802c55c27c63a4a7979adf26e4c0e0`
- Runner SHA-256: `f570d34f7d79fd29139aa1fefdafcc0f47b24784556eb5210b48449852e3f733`
- Spec SHA-256: `783193d79307b1da8e671db69a890e043a68404fe08c76428cc16ea7a9da87e4`
- Registered candidates: 8; eligible: `[]`
- Historical screen only; OOS independence is explicitly false and no live approval is implied.
- `min order` is normalized to a fresh `$100` OOS start; historical compounded notional is retained in JSON only for diagnostics.

| Candidate | OOS return | Sharpe | stress return | p05 final | ruin | events | max gross | min order ($100 OOS) | historical MDD | block MDD p95 | failed gates | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| E10a | -40.43% | -2.20 | -40.54% | $39.11 | 28.41% | 155 | 0.900 | $27.93 | 52.86% | 59.81% | oos_independence, oos_after_cost_and_funding, oos_sharpe, oos_block_robustness, historical_mdd, double_slippage, bootstrap_p05, bootstrap_ruin, block_mdd_p95 | FAIL |
| E10b | -36.65% | -1.66 | -36.75% | $39.56 | 25.38% | 113 | 0.900 | $27.98 | 67.09% | 60.51% | oos_independence, oos_after_cost_and_funding, oos_sharpe, oos_block_robustness, historical_mdd, double_slippage, bootstrap_p05, bootstrap_ruin, block_mdd_p95 | FAIL |
| E10c | -24.69% | -2.36 | -24.73% | $59.69 | 0.39% | 80 | 0.900 | $28.72 | 62.26% | 44.24% | oos_independence, oos_activity, oos_after_cost_and_funding, oos_sharpe, oos_block_robustness, historical_mdd, double_slippage, bootstrap_p05, block_mdd_p95 | FAIL |
| E10d | -33.19% | -1.94 | -33.23% | $45.03 | 12.55% | 67 | 0.900 | $28.07 | 74.18% | 55.33% | oos_independence, oos_after_cost_and_funding, oos_sharpe, oos_block_robustness, historical_mdd, double_slippage, bootstrap_p05, bootstrap_ruin, block_mdd_p95 | FAIL |
| E10e | -15.13% | -0.35 | -15.38% | $48.53 | 9.57% | 250 | 0.900 | $13.79 | 66.40% | 58.34% | oos_independence, oos_after_cost_and_funding, oos_sharpe, historical_mdd, double_slippage, bootstrap_p05, bootstrap_ruin, block_mdd_p95 | FAIL |
| E10f | -11.83% | -0.19 | -11.98% | $47.89 | 10.17% | 133 | 0.900 | $18.20 | 65.97% | 54.45% | oos_independence, oos_after_cost_and_funding, oos_sharpe, historical_mdd, double_slippage, bootstrap_p05, bootstrap_ruin, block_mdd_p95 | FAIL |
| E10g | -8.27% | -1.12 | -8.30% | $79.87 | 0.00% | 33 | 0.600 | $27.48 | 28.76% | 21.36% | oos_independence, oos_activity, oos_after_cost_and_funding, oos_sharpe, oos_block_robustness, historical_mdd, double_slippage, bootstrap_p05 | FAIL |
| E10h | 7.59% | 0.48 | 7.50% | $72.64 | 0.08% | 69 | 0.600 | $15.36 | 33.21% | 34.19% | oos_independence, oos_sharpe, historical_mdd, bootstrap_p05, block_mdd_p95 | FAIL |
