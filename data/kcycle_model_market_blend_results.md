# KCYCLE model-market blend

generated_at: 2026-07-12T02:26:20+00:00
records: 13900
joined_records: 12593
join_coverage: 0.9060
selection: beta selected on val only; test reported once for selected beta per top_k

## Selected beta results
| scope | top_k | beta | val lift | test lift | val exact | test exact |
|---|---:|---:|---:|---:|---:|---:|
| joined_subset | 10 | 0.95 | -0.198pp | -0.334pp | 0.1482 | 0.1619 |
| all_with_market_fallback | 10 | 0.95 | -0.198pp | -0.289pp | 0.1482 | 0.1691 |
| joined_subset | 20 | 0.95 | -0.198pp | -0.334pp | 0.1482 | 0.1619 |
| all_with_market_fallback | 20 | 0.95 | -0.198pp | -0.289pp | 0.1482 | 0.1691 |
| joined_subset | 40 | 0.95 | -0.198pp | -0.334pp | 0.1482 | 0.1619 |
| all_with_market_fallback | 40 | 0.95 | -0.198pp | -0.289pp | 0.1482 | 0.1691 |

## Baselines
| scope | model | top_k | val exact | test exact |
|---|---|---:|---:|---:|
| joined_subset | current_axis | 10 | 0.1502 | 0.1653 |
| joined_subset | gen2_mut_436 | 10 | 0.1514 | 0.1669 |
| joined_subset | current_axis | 20 | 0.1502 | 0.1653 |
| joined_subset | gen2_mut_436 | 20 | 0.1518 | 0.1786 |
| joined_subset | current_axis | 40 | 0.1502 | 0.1653 |
| joined_subset | gen2_mut_436 | 40 | 0.1431 | 0.1611 |
| all_records | current_axis | 10 | 0.1502 | 0.1720 |
| all_records | gen2_mut_436 | 10 | 0.1514 | 0.1770 |
| all_records | current_axis | 20 | 0.1502 | 0.1720 |
| all_records | gen2_mut_436 | 20 | 0.1518 | 0.1879 |
| all_records | current_axis | 40 | 0.1502 | 0.1720 |
| all_records | gen2_mut_436 | 40 | 0.1431 | 0.1727 |
