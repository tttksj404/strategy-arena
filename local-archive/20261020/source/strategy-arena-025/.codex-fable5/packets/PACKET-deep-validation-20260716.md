# FABLE5 AUDIT PACKET deep-validation-20260716

- Task: implement cache-only deep validation for W2c/F1e/W3c/W3d
- Date: 2026-07-16
- Direct audit: pending; this packet records implementation and independent evidence

## 1. Goal and acceptance criteria

Implement five validations from wave result JSON and local wave1 funding cache. Required gates: MC p05 > 300, ruin probability < 5%, DSR score > 0, native sign agreement > 80%, and native common coverage of 133 days. No network calls.

## 2. Implementation

- `research/validation/deep_stats.py`: deterministic bootstrap, LOYO, Bailey-Lopez de Prado DSR, contiguous 8-hour funding windows, and 90-day block permutation statistics.
- `research/validation/deep_validate.py`: result/cache loading, input-semantic routing, manifest integrity check, JSON artifact generation.
- `research/validation/deep_report.py`: Markdown projection.
- `research/validation/tests/test_deep_validate.py`: five synthetic regression tests.

## 3. Evidence

- CLI completed with exit code 0 and wrote four candidate JSON files plus `research/validation/DEEP_REPORT.md`.
- Synthetic tests: 5 passed.
- No-excuse checker: no violations in 4 files.
- Network guard: runner completed while `socket.socket` was replaced with a raising stub.
- Schema assertions: four candidates, MC paths 10000, block paths 1000, DSR trials 28.
- Repeat run: all five artifact SHA-256 values were identical.

## 4. Current results

- F1e: MC PASS (p05 397.44, ruin 0), DSR PASS (7.585), native FAIL (sign 76.42%, coverage 82d), overall FAIL.
- W2c: MC PASS (p05 746.32, ruin 0), DSR PASS (1.819), native FAIL (sign 76.42%, coverage 82d), overall FAIL.
- W3c/W3d: trade bootstrap values are retained, but MC status is UNDETERMINED because their `trade_returns` are active-day returns rather than closed-trade returns; overall FAIL.

## 5. Failures and residual risks

- Native cache has only 82 common score days, below the requested 133.
- Wave2 manifest does not contain every current funding file, so native evidence is fail-closed until the manifest/cache pair is refreshed and rehashed.
- 90-day order permutation preserves final capital mathematically; MDD is the meaningful distributional output and final-capital quantiles are explicitly marked invariant.
- Full wave1/wave2 related test collection remains blocked in the ai_env Python 3.8 environment by pre-existing `enum.StrEnum` import errors; the new validation suite is green.

## 6. Audit questions

- Confirm whether W3 upstream should expose closed-trade returns or whether its MC should be renamed to daily-return bootstrap.
- Confirm whether the existing manifest should be regenerated for the current funding cache before native reproduction is accepted.
- Confirm whether 90-day sampling should remain permutation-only or add a separate with-replacement block bootstrap for final-capital uncertainty.

