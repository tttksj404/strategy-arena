# Wave-11 multi-family report

- Families: RSI mean reversion, Bollinger expansion, funding-price reversal, and spot-perpetual basis relative value.
- Source manifest SHA-256: `2d1aa4772739cc0bab209a27a5684900db802c55c27c63a4a7979adf26e4c0e0`
- Runner SHA-256: `a13f699c2b246191cbcd2a006a43d8e00850fd7836b58528252ecaa977d953ec`
- Spec SHA-256: `157cb2bbf2b80e4a2e3dbb56e62c09f3f1bd527ee08835c43039d94170e74aa6`
- Registered candidates: 16; eligible: `[]`
- Historical screen only; OOS independence is explicitly false.
- Minimum order is normalized to a fresh $100 OOS start.

| Candidate | Family | OOS return | Sharpe | stress | p05 | ruin | active days | min order | MDD | block MDD p95 | failed gates | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| R11b | rsi | 19.86% | 1.22 | 18.17% | $88.53 | 0.00% | 268 | $12.19 | 81.30% | 28.48% | oos_independence, historical_mdd, bootstrap_p05, block_mdd_p95 | FAIL |
| R11d | rsi | 17.15% | 1.00 | 15.56% | $84.00 | 0.00% | 254 | $13.22 | 77.89% | 30.24% | oos_independence, historical_mdd, bootstrap_p05, block_mdd_p95 | FAIL |
| B11c | bollinger | 7.56% | 0.45 | 7.32% | $68.50 | 0.54% | 218 | $19.62 | 38.72% | 37.48% | oos_independence, oos_sharpe, historical_mdd, bootstrap_p05, block_mdd_p95 | FAIL |
| B11b | bollinger | 7.04% | 0.45 | 6.70% | $71.19 | 0.24% | 190 | $19.31 | 49.23% | 39.06% | oos_independence, oos_sharpe, historical_mdd, bootstrap_p05, block_mdd_p95 | FAIL |
| F11a | funding_reversal | 6.34% | 0.47 | 5.78% | $77.47 | 0.01% | 109 | $18.22 | 49.52% | 28.14% | oos_independence, oos_sharpe, historical_mdd, bootstrap_p05, block_mdd_p95 | FAIL |
| F11c | funding_reversal | 5.36% | 0.40 | 4.83% | $75.14 | 0.07% | 127 | $17.87 | 49.49% | 31.70% | oos_independence, oos_sharpe, historical_mdd, bootstrap_p05, block_mdd_p95 | FAIL |
| R11a | rsi | 0.98% | 0.17 | 0.27% | $71.09 | 0.05% | 152 | $18.78 | 51.54% | 35.66% | oos_independence, oos_sharpe, historical_mdd, bootstrap_p05, block_mdd_p95 | FAIL |
| V11a | basis | 0.00% | 0.00 | 0.00% | $100.00 | 0.00% | 0 | $0.00 | 3.99% | 0.00% | data_validation, oos_independence, capital_contract, oos_activity, oos_after_cost, oos_sharpe, oos_block_robustness, double_slippage, bootstrap_p05 | FAIL |
| V11b | basis | 0.00% | 0.00 | 0.00% | $100.00 | 0.00% | 0 | $0.00 | 3.99% | 0.00% | data_validation, oos_independence, capital_contract, oos_activity, oos_after_cost, oos_sharpe, oos_block_robustness, double_slippage, bootstrap_p05 | FAIL |
| V11c | basis | 0.00% | 0.00 | 0.00% | $100.00 | 0.00% | 0 | $0.00 | 1.75% | 0.00% | data_validation, oos_independence, capital_contract, oos_activity, oos_after_cost, oos_sharpe, oos_block_robustness, double_slippage, bootstrap_p05 | FAIL |
| V11d | basis | 0.00% | 0.00 | 0.00% | $100.00 | 0.00% | 0 | $0.00 | 2.15% | 0.00% | data_validation, oos_independence, capital_contract, oos_activity, oos_after_cost, oos_sharpe, oos_block_robustness, double_slippage, bootstrap_p05 | FAIL |
| F11d | funding_reversal | -1.30% | 0.00 | -1.55% | $74.97 | 0.00% | 69 | $18.71 | 30.94% | 29.28% | oos_independence, oos_after_cost, oos_sharpe, historical_mdd, double_slippage, bootstrap_p05, block_mdd_p95 | FAIL |
| F11b | funding_reversal | -3.40% | -0.16 | -3.65% | $74.39 | 0.01% | 54 | $19.02 | 27.62% | 28.23% | oos_independence, oos_activity, oos_after_cost, oos_sharpe, historical_mdd, double_slippage, bootstrap_p05, block_mdd_p95 | FAIL |
| R11c | rsi | -9.50% | -0.37 | -10.10% | $61.86 | 0.71% | 144 | $18.90 | 48.25% | 45.09% | oos_independence, oos_after_cost, oos_sharpe, historical_mdd, double_slippage, bootstrap_p05, block_mdd_p95 | FAIL |
| B11a | bollinger | -10.20% | -0.28 | -10.51% | $57.06 | 2.59% | 235 | $18.92 | 52.38% | 47.39% | oos_independence, oos_after_cost, oos_sharpe, historical_mdd, double_slippage, bootstrap_p05, block_mdd_p95 | FAIL |
| B11d | bollinger | -13.92% | -1.26 | -14.00% | $69.55 | 0.00% | 62 | $19.75 | 29.51% | 28.62% | oos_independence, oos_after_cost, oos_sharpe, oos_block_robustness, historical_mdd, double_slippage, bootstrap_p05, block_mdd_p95 | FAIL |
