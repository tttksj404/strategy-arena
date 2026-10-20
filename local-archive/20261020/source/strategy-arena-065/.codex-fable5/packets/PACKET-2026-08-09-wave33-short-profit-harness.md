# Fable5 audit packet: Wave 33 short-profit graph harness

## Audit status

Implementation and self-verification are complete. Luna Max closure review passed. Direct Fable5 audit has not been executed; this packet is the handoff boundary.

The financial result is `rejected`. No live order, private exchange endpoint, API key, deployment, or production strategy switch is in scope.

## Repository boundary

- Implementation worktree: `C:\Users\SSAFY\Desktop\2학기\strategy-arena-profit-harness`
- Detached base revision: `0e0f09b70671725de41afe0d95913fc52e1da526`
- User's dirty main checkout was not edited.
- TradingAgents reference clone: `C:\Users\SSAFY\Desktop\2학기\TradingAgents-review`
- TradingAgents revision after `git pull --ff-only`: `a33fd4c0f134485a43553a2c23a63cb14adbd88f`
- Its pre-existing untracked `uv.lock` was preserved.

## Implemented contract

`SEED -> EVALUATE -> GATE -> SELECT -> PROPOSE -> ... -> STOP -> TEST -> REPORT`

- Search uses train and validation only.
- Candidate budget is capped at 24; financial policy floors cannot be relaxed by CLI or artifact config.
- `TEST` is emitted before fidelity, data-provenance, and sealed-test calculations.
- The sealed test is one frozen bundle: champion, long baseline, short baseline, and 1.5x cost stress.
- Cost stress scales mapped entry/exit costs and stop-exit taker premium.
- Perpetual funding is not modeled and therefore forces rejection.
- Source and data fingerprints bind the complete result; the validator independently reruns the graph and compares every persisted field.
- Run/report/validation paths are required to be distinct.

## Measured result

- Candidate: `atr3.00-hold3-lev2.00-stopnone`
- Train: `3.883513x`, CAGR `36.9759%`, max drawdown `-40.8593%`, 31 trades
- Validation: `1.908169x`, CAGR `44.76%`, max drawdown `-16.51%`, 13 trades
- Sealed test: `1.302268x`, CAGR `39.15%`, max drawdown `-8.50%`, 6 trades
- Test baselines: short `1.0986x`, long `0.8624x`
- 1.5x cost stress: `1.2992x`
- Validation stability ratio: `1.2065`
- Validation DSR probability: `0.000000`
- Cumulative trial count: `439`
- Decision reasons: `deflated_sharpe`, `sealed_test_trades`, `unmodeled_funding`

The 1.302268x historical test multiple is not approval evidence: six trades fail the minimum of eight, DSR fails 0.95, and funding is absent.

## Provenance and fidelity

- Bars: 60,175 hourly observations
- Bounds: `2019-09-08T17:00:00+00:00` through `2026-07-20T23:00:00+00:00`
- Monotonic / unique / irregular gaps: `true / true / 0`
- Data fingerprint: `715fdfe02886c185b19870193d2177a4f11623f672687d62f94b5ccd8c32b8bd`
- Code fingerprint: `6455bc8bdf7dd083e9c83a6056bcbaf040d4d5957009d054ad0a191d90150dad`
- Run ID: `f45fdfd8fc8ff23e`
- Run artifact SHA-256: `67b3c89469acb562bf47f44b94ceea78635b4dd3b675885b4f9c6dd44773d4e7`
- Validation evidence SHA-256: `198bc53fcc4b325296473816e86055420af781963438d149da0e4a80d8a0c09e`
- Legacy parity: 153 vs 153 trades; `$138.479597` vs `$138.479597`; difference `0.00e+00`
- Fixed-model analytic checks: long difference `2.84e-14`; short difference `0.00e+00`

## Verification evidence

```text
uvx ruff check research\wave33_graph_harness research\wave30_riskcap\sim30.py
All checks passed!

uv run --with pytest --with numpy --with pandas --with requests python -m pytest research\wave33_graph_harness\tests -q
36 passed

uv run research\wave33_graph_harness\run_wave33.py
pipeline_validation.passed=true
reproducible=true
decision=rejected

uv run research\wave33_graph_harness\validate_wave33.py --input research\wave33_graph_harness\results\run.json --output .omx\specs\autoresearch-short-profit\result.json
external_replay_verified=true
findings=[]
passed=true
```

Luna Max initially returned BLOCK on weak policy floors, incomplete fidelity, taker-premium stress, replay provenance, gap handling, and path aliasing. Those paths received regression tests and were re-reviewed. Final closure artifact: `research/wave33_graph_harness/reviews/luna-path-closure-review.md`, verdict `FINAL_VERDICT: PASS`.

## Residual audit questions

1. Confirm that rejection is the only defensible financial conclusion until funding is modeled and sealed-test trade count grows without reusing this test interval for tuning.
2. Confirm that trial-adjusted Sharpe based on trade returns is sufficiently conservative for rejection, while not being represented as a generally calibrated acceptance statistic.
3. Confirm that adapting TradingAgents' explicit graph and risk-manager pattern, without importing its news/fundamental LLM agents, preserves deterministic quant-test boundaries.
