# Wave-9 method expansion report

This report is exploratory simulation evidence. The historical split is not selection-independent because prior waves were inspected before this expansion; no live-capital recommendation is made.

Data: 2020-09-23T00:00:00+00:00 to 2026-07-14T00:00:00+00:00 (2121 common daily rows); OOS begins 2025-10-01.

| Candidate | Family | OOS return | OOS Sharpe | MDD | Stress OOS | MC p05 | Ruin | Block MDD p95 | Min order | Fails |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| D9c | D9 | 16.69% | 1.65 | 25.48% | 16.43% | $94.47 | 0.26% | 32.02% | $5.00 | selection_independent, oos_activity, positive_blocks, mc_p05, historical_mdd, block_mdd_p95 |
| D9b | D9 | 16.34% | 1.79 | 37.18% | 15.70% | $77.80 | 0.44% | 38.38% | $10.00 | selection_independent, mc_p05, historical_mdd, block_mdd_p95 |
| T9d | T9 | 14.28% | 0.68 | 55.78% | 13.98% | $68.57 | 2.22% | 60.31% | $0.78 | selection_independent, capital_contract, oos_sharpe, mc_p05, historical_mdd, block_mdd_p95 |
| T9a | T9 | 13.81% | 0.69 | 39.23% | 12.57% | $57.64 | 3.52% | 57.80% | $4.73 | selection_independent, capital_contract, oos_sharpe, mc_p05, historical_mdd, block_mdd_p95 |
| T9b | T9 | 9.18% | 0.51 | 46.63% | 8.53% | $94.70 | 0.97% | 56.64% | $4.75 | selection_independent, capital_contract, oos_sharpe, mc_p05, historical_mdd, block_mdd_p95 |
| H9a | H9 | 4.95% | 0.65 | 55.12% | 1.13% | $18.85 | 38.56% | 64.20% | $5.09 | selection_independent, oos_sharpe, mc_p05, ruin_probability, historical_mdd, block_mdd_p95, deflated_sharpe |
| P9a | P9 | 3.07% | 0.44 | 51.28% | 0.45% | $25.69 | 20.95% | 62.18% | $5.82 | selection_independent, oos_sharpe, mc_p05, ruin_probability, historical_mdd, block_mdd_p95, deflated_sharpe |
| D9a | D9 | 2.39% | 0.34 | 48.54% | -0.41% | $26.64 | 20.72% | 58.99% | $6.30 | selection_independent, oos_sharpe, stress_positive, mc_p05, ruin_probability, historical_mdd, block_mdd_p95, deflated_sharpe |
| P9d | P9 | -0.64% | -0.04 | 85.82% | -2.02% | $5.73 | 96.45% | 88.19% | $4.73 | selection_independent, capital_contract, oos_positive, oos_sharpe, positive_blocks, stress_positive, mc_p05, ruin_probability, historical_mdd, block_mdd_p95, deflated_sharpe |
| H9b | H9 | -4.69% | -0.55 | 71.35% | -7.91% | $10.82 | 72.35% | 75.98% | $3.22 | selection_independent, capital_contract, oos_positive, oos_sharpe, stress_positive, mc_p05, ruin_probability, historical_mdd, block_mdd_p95, deflated_sharpe |
| P9c | P9 | -5.07% | -0.69 | 33.18% | -6.26% | $48.19 | 6.27% | 45.89% | $21.69 | selection_independent, oos_positive, oos_sharpe, positive_blocks, stress_positive, mc_p05, ruin_probability, historical_mdd, block_mdd_p95, deflated_sharpe |
| H9c | H9 | -7.69% | -1.23 | 83.47% | -11.62% | $9.35 | 99.36% | 83.59% | $1.72 | selection_independent, capital_contract, oos_positive, oos_sharpe, positive_blocks, stress_positive, mc_p05, ruin_probability, historical_mdd, block_mdd_p95, deflated_sharpe |
| D9d | D9 | -8.37% | -0.15 | 40.59% | -9.79% | $163.78 | 0.26% | 61.60% | $0.00 | selection_independent, capital_contract, oos_positive, oos_sharpe, positive_blocks, stress_positive, historical_mdd, block_mdd_p95 |
| P9b | P9 | -10.26% | -1.40 | 22.79% | -11.76% | $87.92 | 0.21% | 34.04% | $9.05 | selection_independent, oos_positive, oos_sharpe, positive_blocks, stress_positive, mc_p05, block_mdd_p95 |
| H9d | H9 | -14.73% | -2.28 | 67.72% | -18.72% | $18.19 | 73.66% | 72.39% | $3.66 | selection_independent, capital_contract, oos_positive, oos_sharpe, positive_blocks, stress_positive, mc_p05, ruin_probability, historical_mdd, block_mdd_p95, deflated_sharpe |
| T9c | T9 | -27.61% | -1.73 | 59.22% | -29.47% | $61.80 | 3.35% | 64.45% | $5.36 | selection_independent, oos_activity, oos_positive, oos_sharpe, positive_blocks, stress_positive, mc_p05, historical_mdd, block_mdd_p95 |

## Verdict

Eligible candidates: none.

Selection-independent: false. Any positive result is a candidate for a fresh prospective paper window only after new unseen data exists.
