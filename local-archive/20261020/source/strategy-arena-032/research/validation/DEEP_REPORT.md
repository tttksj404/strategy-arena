# Deep validation report

Cache-only full-history validation for W2c, F1e, W3c, and W3d.

## Candidate x validation

| Candidate | MC 10k | Leave-one-year-out | DSR (28 trials) | Bitget native | 90d block shuffle | Overall |
|---|---|---|---|---|---|---|
| W2c | PASS p05=720.68; ruin=0.00% | PASS years=8 | PASS score=1.820 | FAIL sign=76.42%; entry=100.00%; coverage=82d | PASS blocks=18; MDD p95=1.48% | FAIL |
| F1e | PASS p05=397.44; ruin=0.00% | PASS years=8 | PASS score=7.585 | FAIL sign=76.42%; entry=92.14%; coverage=82d | PASS blocks=10; MDD p95=0.83% | FAIL |
| F1f | PASS p05=458.66; ruin=0.00% | PASS years=8 | PASS score=1.931 | N/A | PASS blocks=18; MDD p95=3.83% | PASS |
| W3c | UNDETERMINED p05=227.99; ruin=0.31% | PASS years=8 | PASS score=7.708 | N/A | PASS blocks=28; MDD p95=42.71% | FAIL |
| W3d | UNDETERMINED p05=8.29; ruin=36.53% | PASS years=8 | PASS score=17.308 | N/A | PASS blocks=20; MDD p95=95.92% | FAIL |

## Year-by-year leave-one-out

| Candidate | Year | Held-out return | Held-out Sharpe | Remaining return | Remaining Sharpe |
|---|---:|---:|---:|---:|---:|
| W2c | 2019 | 0.0104 | 5.379 | 1.7383 | 4.258 |
| W2c | 2020 | 0.3224 | 9.244 | 1.0923 | 3.493 |
| W2c | 2021 | 0.6202 | 12.036 | 0.7078 | 2.718 |
| W2c | 2022 | 0.0087 | 2.785 | 1.7429 | 4.514 |
| W2c | 2023 | 0.0649 | 7.038 | 1.5982 | 4.280 |
| W2c | 2024 | 0.1972 | 2.657 | 1.3110 | 5.502 |
| W2c | 2025 | -0.0057 | -0.186 | 1.7827 | 4.809 |
| W2c | 2026 | -0.0005 | -1.368 | 1.7683 | 4.375 |
| F1e | 2019 | 0.0137 | 0.957 | 0.6657 | 5.094 |
| F1e | 2020 | 0.2214 | 8.073 | 0.3825 | 3.444 |
| F1e | 2021 | 0.3724 | 12.694 | 0.2304 | 2.321 |
| F1e | 2022 | -0.0063 | -1.170 | 0.6992 | 4.756 |
| F1e | 2023 | 0.0000 | 0.000 | 0.6886 | 4.665 |
| F1e | 2024 | 0.0000 | 0.000 | 0.6886 | 4.666 |
| F1e | 2025 | 0.0000 | 0.000 | 0.6886 | 4.665 |
| F1e | 2026 | 0.0000 | 0.000 | 0.6886 | 4.480 |
| F1f | 2019 | 0.0076 | 3.574 | 0.7149 | 2.295 |
| F1f | 2020 | 0.1534 | 4.483 | 0.4981 | 1.936 |
| F1f | 2021 | 0.3712 | 7.633 | 0.2602 | 1.183 |
| F1f | 2022 | -0.0056 | -0.967 | 0.7377 | 2.487 |
| F1f | 2023 | 0.0078 | 0.638 | 0.7146 | 2.443 |
| F1f | 2024 | 0.1052 | 1.503 | 0.5635 | 2.935 |
| F1f | 2025 | -0.0186 | -0.646 | 0.7608 | 2.670 |
| F1f | 2026 | -0.0023 | -1.368 | 0.7320 | 2.374 |
| W3c | 2019 | -0.1165 | -1.617 | 0.5313 | 0.523 |
| W3c | 2020 | 0.2062 | 0.951 | 0.1216 | 0.214 |
| W3c | 2021 | -0.0897 | -0.611 | 0.4862 | 0.523 |
| W3c | 2022 | 0.0791 | 0.704 | 0.2537 | 0.328 |
| W3c | 2023 | -0.0741 | -0.614 | 0.4612 | 0.496 |
| W3c | 2024 | -0.0469 | -0.292 | 0.4195 | 0.470 |
| W3c | 2025 | 0.1898 | 1.599 | 0.1371 | 0.220 |
| W3c | 2026 | 0.2309 | 2.852 | 0.0991 | 0.176 |
| W3d | 2019 | 0.0000 | 0.000 | 0.0093 | 0.440 |
| W3d | 2020 | 3.5699 | 2.200 | -0.7791 | 0.133 |
| W3d | 2021 | 1.0886 | 1.195 | -0.5168 | 0.234 |
| W3d | 2022 | 0.0000 | 0.000 | 0.0093 | 0.465 |
| W3d | 2023 | -0.2510 | 0.145 | 0.3475 | 0.483 |
| W3d | 2024 | -0.4630 | -0.093 | 0.8796 | 0.542 |
| W3d | 2025 | -0.7371 | -0.855 | 2.8389 | 0.687 |
| W3d | 2026 | 0.0000 | 0.000 | 0.0093 | 0.448 |

## Decision criteria

- MC: 10,000 unit-exposure full-period trade bootstrap paths; PASS requires final-capital p05 > 300 and P(capital < 150) < 5%.
- MC input contract: F1e/W2c use closed-trade returns; W3c/W3d expose active-day returns in the same field, so their MC cells remain UNDETERMINED until upstream semantics are separated.
- Kelly: the existing gate-compatible mean(trade_return) / sample_variance estimate; both f* and 0.25f* are recorded.
- DSR: Bailey-Lopez de Prado daily-return deflated Sharpe z-score with trials=28; PASS requires DSR score > 0.
- Bitget native: record 7-day score correlation, entry-signal agreement, and sign agreement. PASS requires sign agreement > 80% and the requested 133-day common coverage.
- Leave-one-year-out and 90-day block permutation are diagnostic checks; PASS means the calculation completed with the minimum sample.
- The 90-day block method shuffles block order without replacement; final capital is therefore an invariant and MDD is the distributional stress output.

## Data constraints

- No network calls were made. Bitget reproduction uses only normalized local cache rows in research/wave1/cache.
- The current common funding-score interval is 3 symbols, 738 observations, and 82 days. It is shorter than the requested 133 days, so the native cells and overall verdicts for F1e/W2c are FAIL.
- The wave2 cache manifest recheck is FAIL; every consumed funding file must have a matching byte count and SHA-256 before native evidence can be accepted.
- Funding rows are normalized to UTC 8-hour buckets and a 7-day score is emitted only for 21 contiguous buckets; gaps reset the rolling window.
- Final candidate status is the intersection of the stated gates; insufficient cache coverage is never interpreted as a PASS.
