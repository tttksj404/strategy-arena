# FABLE5 AUDIT PACKET wave8-capital100-20260721

- Task: determine whether any preregistered carry/momentum candidate is executable and promotion-eligible with a total capital of US$100.
- Date: 2026-07-21 (Asia/Seoul)
- Domain: crypto research `strategy-arena`; do not mix with the separate KRA/KCYCLE racing repository.
- Direct audit: pending; this packet records the implementation, independent checks, and fail-closed result.
- Audit amendment: after an independent finance review found W2c creates spot+perp gross notional at 2x equity, a separate-leg 1x `gross_exposure` gate was added and rerun. This is explicitly a post-registration conservative amendment, not a new signal or a favorable retune.

## 1. Goal and acceptance criteria

The attachment's wave-5/6/7 evidence was treated as immutable input. Wave-8 compares only W2c and W7a-W7d; it adds no signal, leverage, network, or live-order experiment.

- total capital: `$100`
- reserve: `10%` (`$10`), active capital `$90`
- leverage: `1x`
- minimum order: `$5`
- Monte Carlo: `10,000` seeded paths; ruin means final equity below `$50`
- 90-day block shuffle: `1,000` seeded paths; MDD p95 must be `<=25%`
- source gates: no `FAIL` or `UNDETERMINED` in the upstream candidate gate set
- gross exposure: separate-leg 1x gross notional must not exceed active capital; absent unified-margin proof, fail closed
- no candidate is a live recommendation unless every gate passes

## 2. Implementation

- `research/wave8_capital/SPEC.md`: frozen scope, gates, and capital assumptions.
- `research/wave8_capital/run_capital100.py`: deterministic scaling, order-size feasibility, MC, block-shuffle MDD, and upstream source-gate checks.
- `research/wave8_capital/results/capital100.json`: machine-readable evidence, including SHA-256 hashes for all five input result JSONs plus the SPEC and runner hashes.
- `research/wave8_capital/report/capital100_report.md`: human-readable decision report.
- `research/STRATEGY_CARD.md`: corrected the prior `$1,000+` arithmetic; W7a needs about `$6,561.64`, W7b/c/d about `$4,921.23` for minimum-order sizing only. The separate-leg gross gate still blocks 1x execution.

## 3. Evidence and independent recomputation

- `python -m compileall -q research/wave8_capital` -> `COMPILE_PASS`.
- `python -m pytest -q research/wave8_capital/tests research/wave5/tests research/wave7/tests research/validation/tests` -> `25 passed`.
- `python research/wave8_capital/run_capital100.py` -> `WAVE8_CAPITAL100_DONE`, eligible set `[]`, raw-return winner `W7c`.
- Independent JSON/formula check (separate from the runner's functions) -> `INDEPENDENT_FORMULA_CHECK_PASS`.
  - W2c minimum leg: `$90 * 0.25 = $22.50`; required total capital: `$5 / (0.9 * 0.25) = $22.2222`; order-feasible.
  - W2c separate-leg gross: `$90 * 2.0 = $180`, greater than active `$90`; gross-infeasible without verified unified margin.
  - W7c momentum minimum leg: `$90 * 0.4 * 0.002822237758123185 = $0.101601`; required total capital: `$4,921.2328`; order-infeasible.
  - W2c upstream gate set contains `FAIL`/`UNDETERMINED`; W7c upstream deep-validation gates contain `FAIL` (Sharpe gate); both are source-ineligible.
- Input provenance check -> `PROVENANCE_HASH_CHECK_PASS`; all five input hashes are 64-character SHA-256 values recorded in the output JSON.
- SPEC/runner provenance check -> both `spec_sha256` and `runner_sha256` are 64-character SHA-256 values recorded in the output JSON.

## 4. Candidate decision

| Candidate | $100 scaled final | MC p05 | Block MDD p95 | Order gate | Gross gate | Source gate | Overall |
|---|---:|---:|---:|---|---|---|---|
| W7c | $282.2810 | $223.6231 | 7.8973% | FAIL | FAIL | FAIL | FAIL |
| W2c | $263.3835 | $229.8998 | 1.7419% | PASS | FAIL | FAIL | FAIL |
| W7d | $259.7697 | $218.0888 | 4.2270% | FAIL | FAIL | FAIL | FAIL |
| W7a | $216.9478 | $176.9961 | 8.9394% | FAIL | FAIL | FAIL | FAIL |
| W7b | $202.8091 | $157.8324 | 12.4826% | FAIL | FAIL | FAIL | FAIL |

Risk paths use daily equity returns; `trade_returns` are excluded because overlapping closed-trade compounding overstated W2c risk. Raw historical return is not the promotion criterion. There is no all-gates-pass candidate at `$100`; W2c passes only the per-leg order check and remains research-only because gross and source gates fail.

## 5. Three orthogonal audit hypotheses

1. **Capital threshold bug** — W7's old `$1,000+` claim may be caused by a hard-coded `$300` check or arithmetic error. Distinguishing evidence: recalculate `min_order / (active_fraction * component_weight * min_position_weight)` from source metadata. Result: refuted; W7c is `$4,921.2328`, not `$1,000+`.
2. **W2c may fail the `$5` order minimum at `$100`** — distinguish with the minimum-leg formula against the actual W2c metadata. Result: refuted for order sizing only; `$22.50` minimum leg is feasible, but the separate-leg gross gate (`$180 > $90`) and upstream source gates fail closed.
3. **W7c raw return may justify promotion despite execution/source failures** — distinguish by requiring order, gross, and upstream gates. Result: refuted; W7c is the raw-return winner but has a `$0.101601` momentum leg, failed gross gate, and failed upstream Sharpe gate.

## 6. Verification limits and residual risk

- No live exchange API, wallet, key, or order was used.
- W2c's current OOS evidence is sparse (`n_trades=2`) and its upstream gate set includes `FAIL` and `UNDETERMINED`; maker `0.02%` per leg with zero slippage is the stored cost route. The result is research-only, not an executable paper recommendation.
- Historical MC/block-shuffle values are not a guarantee of future returns.
- Existing dirty wave-4/paper files and permission-locked evidence directories were preserved; this task did not reset, overwrite, commit, or push them.
- Direct Fable5 audit is pending. Until an independent auditor accepts this packet, the status is `PENDING_EXTERNAL_FABLE5`, not “verified for live capital.”

## 7. Reproduction commands

```text
python research/wave8_capital/run_capital100.py
python -m compileall -q research/wave8_capital
python -m pytest -q research/wave8_capital/tests research/wave5/tests research/wave7/tests research/validation/tests
```
