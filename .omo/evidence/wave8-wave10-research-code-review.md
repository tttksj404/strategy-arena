# Wave 8/9/10 current-state code review

- Review date: 2026-07-21 (Asia/Seoul)
- Scope: latest source, specifications, tests, generated results, reports, and manifests under Wave-8/9/10 only. Prior review evidence was not used as source truth.
- `codeQualityStatus`: **WATCH**
- `recommendation`: **APPROVE**
- `blockers`: **None**

## Current snapshot and verification

- Wave-8 runner: `616d8ee357d46b7abeba2bef95982eaaa182a92ecced2e9baebc3afd328c1642`
- Wave-9 runner: `7d04e4b4a9656384005c7f4d85729db18f2b32efe79ae8102194c2d446df4691`
- Wave-10 runner: `0371d5bf9ecb113bf4ea9a74b8b657b2e69e658d5f32b312254db2c9d57ef82f`
- Scoped tests: **13 passed in 0.48s**. Raw artifact: `C:\Users\SSAFY\.codex\artifacts\tool-output\20260721-172041878-076cf06e.log`, SHA-256 `de028d7f8767b84fb0b1771d29f2bb3e858b8452386df313bfc7aa2657aec6b9`.
- All three validators exited 0. Their result/report hashes match the current manifests:
  - Wave-8 results `324bc33d1a0880a066622cef00522e5cb18b1c49bf297ad465ce8adfd41487a0`; report `dfbc059e9db56db03f396cba6d70e64cbd3d007a3899441e7bae02485749cdb2`.
  - Wave-9 results `30730da9a7fade13739790b3ebb51dff3ea18c0167efb71ed51f6b42420c5450`; report `0eecdcf05957f139963c64cee76eb84b0fb4083845522f5c81f7f2a2c101634f`.
  - Wave-10 results `9a7cb61f41d45d258e570a9cfda3169d05298b96c0ee52a7a02527016b8886b4`; report `bea5f4d6231bc436c36cbb27c1e33500e98089eedcc0a685f6f39cb194eb8886`.
- Independent read-only semantic checks passed:
  - F8d output exactly equals a direct `funding.rolling(7).mean().shift(1)` signal with the lagged BTC MA200 guard.
  - E10e and E10f positions exactly follow the throttle state reconstructed from their final `_returns()` equity path on every row.

## Findings

### CRITICAL

None.

### HIGH

None.

### MEDIUM

1. **Wave-9 specification wording still exceeds the implementation.** The spec says equal/inverse-vol weights are clipped by the `$5` minimum-order contract (`research/wave9_methods/SPEC.md:16-20`), while the runner retains the weights in simulated PnL and fails the capital gate when a leg is below `$5` (`research/wave9_methods/run_wave9.py:346-365`). This is fail-closed and does not create an eligible candidate, so it is not an approval blocker, but the spec should say “gate” or the execution model should actually clip.

2. **Regression tests remain narrower than the fixed behavior.** The Wave-8 funding test checks the helper values but not a complete price-plus-funding return interval (`research/wave8_alternative/tests/test_wave8.py:46-51`). The Wave-10 throttle test includes nonzero funding but its candidate contains only D9b, so no funding leg is exercised (`research/wave10_ensemble/tests/test_wave10.py:20-37`). A direct funding-ensemble path test would protect the repaired invariant.

3. **Wave-8 and Wave-9 runners remain oversized and duplicate financial boundary logic.** At 463 and 472 lines, their repeated PnL, OOS, simulation, gating, and serialization paths increase maintenance risk. No remaining correctness divergence was found in the reviewed patch.

### LOW

1. `test_funding_cash_uses_position_held_before_funding_event` now checks next-interval funding but retains the old event-time name (`research/wave8_alternative/tests/test_wave8.py:46`). Rename it to state the `t -> t+1` interval contract.

## Skill-perspective checks

- `omo:remove-ai-slops`: **ran**. No HIGH slop/overfit issue remains. The helper-level and non-funding throttle tests are incomplete boundary coverage and are recorded as MEDIUM rather than treated as proof of the fixed end-to-end behavior.
- `omo:programming`: **ran**. The repaired production calculations now share a consistent interval and equity contract. Oversized duplicated runners and the spec/gate wording mismatch remain maintainability concerns, not blockers.

## Resolved prior blockers

- Funding cash now uses the same `t -> t+1` interval as price PnL (`research/wave8_alternative/run_wave8.py:315-325`, `research/wave10_ensemble/run_wave10.py:113-118`).
- F8d now directly implements the registered 7-day funding spread before applying the MA200 guard (`research/wave8_alternative/run_wave8.py:215-228`).
- Wave-10 throttle and final returns both use the current scaled funding leg times next-day funding (`research/wave10_ensemble/run_wave10.py:90-107`, `research/wave10_ensemble/run_wave10.py:113-120`).
- Wave-10 reindexes positions to `net.index` before minimum-order calculation (`research/wave10_ensemble/run_wave10.py:124-138`).

## Approval

There are no CRITICAL or HIGH findings and no remaining approval blockers. The current patch is **APPROVED with WATCH items** above.
