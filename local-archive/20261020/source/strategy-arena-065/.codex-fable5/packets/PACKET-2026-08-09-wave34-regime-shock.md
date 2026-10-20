# Fable5 audit packet: Wave 34 regime-shock graph harness

## Audit status

Implementation and self-verification are complete. The independent Luna Max
review returned `LUNA_VERDICT: PASS`. The installed Fable5 adapter exposes local
status/goals/findings gates but no direct independent audit command, so direct
Fable5 audit remains pending at this handoff boundary.

The financial decision is `research_only`, not accepted. No API key, exchange
credential, live order, private endpoint, deployment, or capital allocation was
used.

## Repository and domain boundary

- Implementation worktree:
  `C:\Users\SSAFY\Desktop\2학기\strategy-arena-profit-harness`
- Detached base revision:
  `0e0f09b70671725de41afe0d95913fc52e1da526`
- The user's dirty `strategy-arena` checkout was not edited.
- No isolated changed path contains `KRA` or `KCYCLE`; the work is only the
  deterministic quant research harness.
- TradingAgents was used only as an architectural reference for explicit graph
  and risk-manager separation. News/fundamental LLM agents were not inserted
  into backtest selection.

## Implemented contract

`SEED -> TEMPORAL_CV -> SELECT -> PROPOSE -> ... -> STRESS -> FREEZE -> AUDIT -> REPORT`

- Search and proposal read only timestamps through
  `2025-09-30T23:59:59Z` across six chronological folds.
- The already opened post-cutoff interval is evaluated only after one `FREEZE`
  and one `AUDIT`; it cannot trigger proposal, reselection, or acceptance.
- The default search spends 720 unique candidates. Stress evaluates 72
  candidates selected before stress results: 36 by primary objective and 36 by
  pre-cutoff path safety.
- Actual BTCUSDT perpetual funding is applied with signed direction and
  fixed-position notional. Funding source coverage and SHA-256 are persisted.
- Stress includes cost 2x/3x, delay 1h/2h, adverse funding 10 bp, entry shock
  12%/20%/40%, volatility 2x plus cost 2x, and a combined extreme.
- Intrabar risk has a separate drawdown path. Synthetic gap stops fill at the
  gap price, and a gap through liquidation liquidates before a stop.
- Walk-forward uses prior folds only and admits candidates only when a 40%
  analytic shock implies at most 50% equity loss and observed past risk remains
  inside the same bound.
- Result bytes bind frozen config, current Python source, full market/funding
  arrays, and deterministic run id. The external validator reruns all 8,651
  simulations and requires byte-identical output.

## Iteration evidence

The loop preserved each architecture failure before correction:

1. v1 did not retain recovered entry-shock drawdown.
   SHA-256 `f575683da6f2902dbf33af15de6786e2afb5d5954130da4f001c8b41b4e132f4`.
2. v2 measured the 40% shock but a top-12 stress funnel omitted lower-risk
   candidates.
   SHA-256 `55ab54bd61b6f6226630ea2d751e024e46b8e7d08cab926b8b6f2b63a9d4ecee`.
3. v3 balanced the stress pool and selected a stress-passing 1x candidate, but
   its unfiltered walk-forward selector chose 3x historical candidates and
   chained only `0.635065x`.
   SHA-256 `4d2d3acab5b0b0581db54354d9641569cdb772d92f9145fa761e6734a57d1d72`.
4. v4 added gap-fill/liquidation precedence and past-only shock admission. Its
   walk-forward chain rose to `1.449090x` without using the evaluation fold.

All prior Wave 34 searches and one meta-policy revision are carried into the
v4 trial count rather than reset.

## Measured v4 result

- Candidate: `atr3.50-hold2-lev1.00-stopnone`
- Six pre-cutoff folds: all 6 profitable; worst fold `1.017421x`; median fold
  `1.113181x`; 37 trades; base max drawdown `-21.28%`.
- Base six-fold chained multiple: `2.069833x`.
- Past-only walk-forward: `1.449090x`, 34 evaluation trades.
- Stress gate: passed; worst scenario `combined-extreme` at `1.064838x`.
- Worst stress drawdown: `-40.0363%`; worst stress trade `-12.7568%`;
  liquidations `0`.
- Cumulative trial count: `4,988`.
- Deflated Sharpe probability: `0.00000000`.
- Architecture findings: none.
- Robustness rejection reason: `deflated_sharpe_probability`.
- Final decision reasons additionally include
  `fresh_unseen_holdout_absent` and
  `contaminated_audit_not_acceptance_evidence`.

These historical multiples are not expected-return claims and do not authorize
live trading.

## Provenance

- Schema: `wave34-regime-shock/v4`
- Run ID: `9f58588a00014ad1`
- Code fingerprint:
  `2d4690913d3c09fa216f0d9f6f49951f1425be275aa08877cbb89a63a7391fc9`
- Data fingerprint:
  `2ff8233b547a140d45c2ec41f11bb51f70d38af37e48f896e8065fa0716934ac`
- Funding source SHA-256:
  `3066e710e23490257cfa4697d527515d336c369b0531f5c58ec186eb0932fc91`
- Funding coverage: `2019-09-10T08:00:00Z` through
  `2026-07-14T16:00:00Z`, 7,499 normalized unique events, zero missing price
  matches.
- Run artifact SHA-256:
  `e494372b95b634bde3f991bb080fab47592f8bd8664012534760bdb10b858c22`
- Validation evidence SHA-256:
  `26f344ea20f2c01889824f9526c977b182b5881a519d4188bdaebb8e181089e8`
- Legacy simulator fidelity: 153 vs 153 trades; `$138.479597` vs
  `$138.479597`; difference `0.00e+00`; fixed analytic long/short differences
  `2.84e-14` / `0.00e+00`.

## Verification evidence

```text
python -m pytest research/wave33_graph_harness/tests research/wave34_regime_shock/tests -q
60 passed in 29.48s
capsule=20260809-154818810-fb15e8fd
artifact_sha256=31284a7775136442453bd6129b638fdd4c6fbe243b48ec8ca521f40825975433

uvx ruff check --ignore RUF100 [Wave 30 simulator + Wave 34]
All checks passed!

check-no-excuse-rules.py [Wave 30 simulator + Wave 34]
no violations in 31 file(s)

python -m research.wave34_regime_shock.run_wave34
candidate=atr3.50-hold2-lev1.00-stopnone
evaluated_candidates=720
simulation_count=8651
decision=research_only
capsule=20260809-152641706-dd45fbc9

python -m research.wave34_regime_shock.validate_wave34
external_replay_verified=true
findings=[]
passed=true
capsule=20260809-153023070-879174fa
```

Luna Max (`gpt-5.6-luna`, `xhigh`, read-only) found no unresolved HIGH/MEDIUM
correctness issue and returned `LUNA_VERDICT: PASS`. Its concise review is
`research/wave34_regime_shock/reviews/luna-max-review.md`; the full captured
artifact SHA-256 is
`1a4e3bd171a3993b49e9c60c34d02a38c6c3c230d4170a0c5dbeda916ea3a9e4`.

## Residual audit questions

1. Confirm that DSR zero and the absence of a fresh unseen holdout make
   `research_only` the only defensible verdict despite stress and walk-forward
   profitability.
2. Confirm that the 40% synthetic shock and perfect hourly market data still do
   not cover exchange outage, insurance-fund behavior, order-book depth, or
   multi-hour disconnected gaps.
3. Confirm that a future acceptance run must wait for a genuinely new sealed
   interval and must not tune thresholds or orchestration using it.
