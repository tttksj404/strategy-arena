# Wave-4 Leverage Sweep Report

## Publication gate

- Grid is unchanged: W2c/F1f x SYM/ASYM x {1, 1.5, 2, 3, 5, 10} = 24 combinations.
- Gate criteria are unchanged: MC p05 > $300, final bankruptcy probability < 5%, and MDD <= 25%.
- Report publication is blocked unless all four L=1 reconciliation rows pass at <= 1% relative error for both CAGR and MDD.

### L=1 reconciliation

| Candidate | Structure | Sweep CAGR | Engine CAGR | CAGR rel. error | Sweep MDD | Engine MDD | MDD rel. error | Status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| W2c | SYM | 16.26% | 16.26% | 0.00% | 1.78% | 1.78% | 0.00% | PASS |
| W2c | ASYM | 16.26% | 16.26% | 0.00% | 1.78% | 1.78% | 0.00% | PASS |
| F1f | SYM | 8.29% | 8.29% | 0.00% | 3.55% | 3.55% | 0.00% | PASS |
| F1f | ASYM | 8.29% | 8.29% | 0.00% | 3.55% | 3.55% | 0.00% | PASS |

- Reconciliation gate: PASS.

## Model contract

- Daily P&L is replayed from the imported wave-1/wave-2 engine path. The sweep adds leverage scaling, borrow interest, and liquidation overlays only.
- Liquidation basis move: `abs(simultaneous_close_basis_change) + max(0, perp_intraday_range_pct - spot_intraday_range_pct) * 0.5`.
- Stress basis move is the same value multiplied by 1.5 and is reported separately.
- Inputs are cache-only from `research/wave1/cache`; no network calls are made.

## Combination results

| Candidate | Structure | L | CAGR | MDD | MC p05 | Bankruptcy | Liq. | Stress liq. | Borrowing | Gate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| W2c | SYM | 1 | 16.26% | 1.78% | $746.32 | 0.00% | 0 | 1 | $0.00 | PASS |
| W2c | SYM | 1.5 | 16.93% | 33.98% | $1,165.60 | 0.00% | 1 | 1 | $39.50 | FAIL |
| W2c | SYM | 2 | 21.35% | 45.29% | $1,858.63 | 0.00% | 1 | 1 | $91.39 | FAIL |
| W2c | SYM | 3 | 25.70% | 67.86% | $4,513.38 | 0.00% | 2 | 1 | $196.20 | FAIL |
| W2c | SYM | 5 | -100.00% | 100.00% | $26,421.74 | 0.00% | 3 | 5 | $16.11 | FAIL |
| W2c | SYM | 10 | -100.00% | 100.00% | $1,892,428.78 | 0.00% | 22 | 44 | $33.41 | FAIL |
| W2c | ASYM | 1 | 16.26% | 1.78% | $746.32 | 0.00% | 0 | 1 | $0.00 | PASS |
| W2c | ASYM | 1.5 | 14.59% | 27.18% | $899.22 | 0.00% | 1 | 1 | $0.00 | FAIL |
| W2c | ASYM | 2 | 16.21% | 30.20% | $1,014.58 | 0.00% | 1 | 1 | $0.00 | FAIL |
| W2c | ASYM | 3 | 18.21% | 33.97% | $1,178.83 | 0.00% | 1 | 1 | $0.00 | FAIL |
| W2c | ASYM | 5 | 20.18% | 37.73% | $1,367.92 | 0.00% | 1 | 1 | $0.00 | FAIL |
| W2c | ASYM | 10 | 21.94% | 41.16% | $1,571.37 | 0.00% | 1 | 1 | $0.00 | FAIL |
| F1f | SYM | 1 | 8.29% | 3.55% | $456.08 | 0.00% | 0 | 1 | $0.00 | PASS |
| F1f | SYM | 1.5 | 5.10% | 34.99% | $562.42 | 0.00% | 1 | 1 | $25.68 | FAIL |
| F1f | SYM | 2 | 5.25% | 46.51% | $689.85 | 0.00% | 1 | 1 | $50.42 | FAIL |
| F1f | SYM | 3 | 1.45% | 69.06% | $1,044.29 | 0.00% | 2 | 1 | $77.19 | FAIL |
| F1f | SYM | 5 | -100.00% | 100.00% | $2,276.49 | 0.00% | 3 | 5 | $15.04 | FAIL |
| F1f | SYM | 10 | -100.00% | 100.00% | $14,768.03 | 0.00% | 22 | 44 | $28.40 | FAIL |
| F1f | ASYM | 1 | 8.29% | 3.55% | $456.08 | 0.00% | 0 | 1 | $0.00 | PASS |
| F1f | ASYM | 1.5 | 5.22% | 27.92% | $497.86 | 0.00% | 1 | 1 | $0.00 | FAIL |
| F1f | ASYM | 2 | 5.70% | 30.99% | $524.56 | 0.00% | 1 | 1 | $0.00 | FAIL |
| F1f | ASYM | 3 | 6.25% | 34.81% | $562.97 | 0.00% | 1 | 1 | $0.00 | FAIL |
| F1f | ASYM | 5 | 6.75% | 38.62% | $603.23 | 0.00% | 1 | 1 | $0.00 | FAIL |
| F1f | ASYM | 10 | 7.15% | 42.07% | $641.79 | 0.00% | 1 | 1 | $0.00 | FAIL |

## Conclusion

- Combination gates passed: 4/24.
- Maximum leverage passing the unchanged risk gate: 1x.
- The reconciliation gate passed, so this report is publishable.
