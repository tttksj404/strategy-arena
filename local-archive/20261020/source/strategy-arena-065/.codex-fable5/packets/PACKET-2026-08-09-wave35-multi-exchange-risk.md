# Fable audit packet: Wave 35 multi-exchange risk harness

## Objective and authority

Evaluate whether a bounded, shared crypto-perpetual strategy can improve
short-horizon paper returns across the current Binance USD-M and Bitget USDT
contract sets. Preserve Strategy Arena as a quant-only scope; do not touch KRA,
KCYCLE, Kraev, credentials, accounts, positions, or order paths. TradingAgents
may inform orchestration, but must remain outside alpha/backtest computation.

Work was isolated at
`C:\Users\SSAFY\Desktop\2학기\strategy-arena-profit-harness` from base commit
`0e0f09b70671725de41afe0d95913fc52e1da526`. The user's main dirty checkout was
not modified by this work.

## Superseding post-holdout disposition

The initial deterministic validator reproduced the historical final byte for
byte, but the later Luna Max post-holdout audit returned `FAIL` with three
blockers:

1. the one-shot guard could be bypassed by choosing another final output path;
2. canonical `passed=true` represented replay correctness while contradicting
   the evaluator's failed strategy gates;
3. execution tradability referenced next-bar availability.

All three were independently reproduced. The next-bar path had zero numerical
effect on the selected policy's validation, stress, and holdout metrics, but it
was still corrected as a structural temporal defect. The implementation now:

- accepts only canonical `final-run.json` in the search artifact namespace;
- uses only current-bar availability for target tradability and fails closed if
  a held asset later lacks a realizable next open;
- requires both artifact/replay correctness and strategy gates for evaluator
  `passed=true`, while separately exposing both statuses and
  `strategy_approved=false`.

The corrected source fingerprint is
`a7d6b99c4151db9faef94284a0cdabe70152b1ae1b487ca415cdf4457ba69aad`.
Only pre-holdout search/risk were regenerated:

- corrected search SHA-256:
  `042e2db1c3e835d002007b37b4af310afbacef724b77c4126318efc93061c551`;
- corrected risk SHA-256:
  `c25ccdf7cd793e25668473bd88c44da9f2b9af335e1c5b1535288a02beba775b`.

Policy, validation metrics, DSR, and all 14 risk metrics remained unchanged.
The old holdout was not rerun. The exact validator invalidated every canonical
positive surface and stopped on `holdout_attempt_contract_mismatch` before
holdout simulation. Original search, risk, final, attempt, validation, report,
and OMX result files are hash-preserved under
`results/historical/20260809-postholdout-invalidated/`.

The canonical `final-run.json` is now an invalidation marker, not the stale
financial result. Luna Max targeted remediation review found no HIGH/MEDIUM
blocker and ended `LUNA_REMEDIATION_VERDICT: PASS`. This approves structural
remediation only. The current profitability experiment remains invalidated and
rejected, and the holdout remains permanently consumed.

## Frozen evidence

- Public universe ledger: 1,595 terminal rows (854 Binance, 741 Bitget).
- Terminal dispositions: 96 history candidates, 432 liquidity-budget rank,
  464 below minimum turnover, 127 inactive, 275 non-crypto, and 201 non-USDT
  perpetual.
- History window: 1,100 UTC days, 2023-08-05 through 2026-08-08.
- Public requests: 1,117. Candidate rows: 96/96 terminal; 46 accepted and 50
  quality-rejected for fewer than 730 bars; zero acquisition errors.
- Accepted venue series deduplicate to 27 underlyings; 19 same-underlying venue
  duplicates were removed using only pre-return liquidity and fee inputs.
- Current-universe survivorship and current-liquidity conditioning are known
  contamination and are carried into the final decision.

## Graph search and risk result

- Frozen candidate budget: 480 training policies and 48 fixed validation
  comparisons; 1,009 train/validation simulations.
- Trial provenance: 4,988 prior Wave 34 comparisons plus 480 training policies
  plus 48 validation comparisons = 5,516 cumulative unique comparisons.
- Frozen policy:
  `lb30-rb7-l2-s4-tv0.35-aw0.40-vc0.60-cc0.80-dd0.20-cd10`.
- Validation multiple: `1.0490923582637484`; MDD:
  `-0.11037998676221994`; DSR: `0.0`.
- Fourteen pre-holdout abrupt-change scenarios were executed. The corrected
  daily cooldown semantics exposed `combined-extreme` at
  `0.15280804251044405x` and MDD `-0.8471919574895559`; it remains an explicit
  `catastrophic_path` failure. No stress-driven reselection was allowed.
- Search SHA-256:
  `d204579a8dfb2089cf8c7124e7d52f18374ad1c2fa03ea1d8c8c9fde3eae8485`.
- Risk SHA-256:
  `8fe4bf5311565c1dbd4d5bf9b8a8268c1e3ed39e1bec3efa0cdba997b6540e83`.
- Source fingerprint:
  `53dbde6ed27608e81e02ac562a31fbfdd1b68e0ebd822a0e0e61579c528e78c4`.

The completed G003/G004 ledger checkpoints contain intermediate pre-fix SHAs
and metrics. They are retained as historical checkpoints, not current evidence.
The hashes and failure metrics in this packet and the final report supersede
those intermediate checkpoint values.

## TradingAgents boundary

- Pinned clone commit:
  `a33fd4c0f134485a43553a2c23a63cb14adbd88f`, version 0.3.1, Apache-2.0.
- Focused deterministic tests: 42 passed; JUnit SHA-256:
  `d5ad8552868c7dff9e5ab444ff83ef76c97dd82cc421ebbf43fef42fafe11c4f`.
- Exact sorted test-identity digest:
  `c30b79c4a23f9201c6984da8f96314b78ffabe0a0c6a8adbba1b6c57adf9d8b9`.
- Adopted only role separation, adversarial debate/risk review, checkpointed
  graph events, and final-validator routing.
- Excluded from alpha/backtest because historical dates do not freeze live
  news/social inputs point-in-time, LLM decisions are nondeterministic, and the
  upstream data path does not model both exchanges' perpetual funding and
  execution costs.

## One-shot holdout

Pre-holdout Luna Max correctness review ended `LUNA_VERDICT: PASS`. The source,
search, risk, manifest, and panel fingerprints were then bound into an atomic
`O_EXCL` attempt marker before primary evaluation.

- Attempt count: 1.
- Attempt SHA-256:
  `77263412a09578c4aea3a4303effc2050410a78f6f9f0d1fd6f0c1827142eae9`.
- Holdout period: 2026 YTD through the frozen 2026-08-08 UTC boundary.
- Holdout multiple: `0.9312963706857629` (loss, not profit).
- Holdout MDD: `-0.2070113034227129`.
- Holdout DSR: `0.0`.
- Primary holdout evaluations: 1; policy changed after validation: false.
- Final SHA-256:
  `01e84e00859ad89709826cd4f395c79166551b9aa5cdf3b6d13c7dc4cc462a12`.
- Final decision: `research_only`; robustness gates: false.

Decision reasons are
`combined-extreme:catastrophic_path`, `validation_dsr_below_0.95`,
`preholdout_path_gate_failed`, `holdout_dsr_below_0.95`, and
`current_universe_survivorship_bias`.

## Verification evidence

```text
uv run --with pytest --with numpy --with pydantic --with httpx2 --with typer \
  python -m pytest research/wave35_multi_exchange/tests -q
79 passed in 21.73s
capsule=20260809-202103885-d70c88c2
artifact_sha256=0deaba348b1bc9070fd8ea4808a81eeef8cca4a59fb1a22d99da25757a222edb

uvx --with pytest --with numpy --with pydantic --with httpx2 --with typer \
  pyright research/wave35_multi_exchange
0 errors, 0 warnings, 0 informations
capsule=20260809-202207593-ae5d0bf5
artifact_sha256=0c5760626dc3cd960abf15044d8f7f7374f60520138bb5c041f5a384a703b769

check-no-excuse-rules.py research/wave35_multi_exchange
no violations in 75 file(s)
maximum pure LOC=239; functions with more than three non-self parameters=0

uv run research/wave35_multi_exchange/run_wave35.py finalize
decision=research_only; primary holdout evaluations=1
capsule=20260809-202553616-635434d9
artifact_sha256=382d23de75907ef16d7e09cb250aeebd7fa05652aca8291a7e3506c146deaf30

uv run research/wave35_multi_exchange/run_wave35.py validate
passed=true; independent_replay_verified=true; findings=[]
final_sha256 == deterministic_replay_sha256 ==
01e84e00859ad89709826cd4f395c79166551b9aa5cdf3b6d13c7dc4cc462a12
raw_responses_verified=1117; history_candidates=96; universe_contracts=1595
capsule=20260809-202638516-59328981
artifact_sha256=07a3d3085559f0a4b26a3c138f3fded97378e5b4d0bc5ab3794135921cba0d45

# Superseding post-audit verification
uv run --with pytest --with numpy --with pydantic --with httpx2 --with typer \
  python -m pytest research/wave35_multi_exchange/tests -q
86 passed in 17.55s
capsule=20260809-210340627-e436f8d9
artifact_sha256=dd7997e190e0522c9119c5e8afc35ba7b531e83a738af79057ebed96a8791804

uv run research/wave35_multi_exchange/run_wave35.py validate
expected fail-closed: holdout_attempt_contract_mismatch before simulation
capsule=20260809-210409087-ea526fdd
artifact_sha256=037afddb9ef2deb1618cd2a84e4a7580407f874c3099a4a041bbce177da5c457

Luna post-holdout audit: FAIL
review_sha256=52d9906ae869a58fcb563eea0023db3c816612b0410475497b122928d8be362c

Luna targeted remediation audit: PASS
review_sha256=854837c00c3b8a9f735a4defe9ff6f936621bad0cdbaf9f38adee9c2028b72ee
```

Ruff reports zero lint violations. Its current release warns that the
no-excuse checker's required custom suppression token
`# noqa: GENERIC_ERR_OK` is not a native Ruff rule code; removing it would
break the independent no-excuse contract, so the warning is retained and
disclosed.

## Audit questions and disposition

1. Does one-shot reservation remain binding under crash and path-alias cases?
2. Does raw replay cover every normalized cache reference and preserve UTC
   boundaries without accepting NaN, infinity, or negative turnover/fees?
3. Are trial provenance, graph replay, risk replay, vendor evidence, and final
   replay independently recomputed rather than trusted from summaries?
4. Are survivorship/current-liquidity contamination, sparse funding fallback,
   DSR failure, catastrophic stress, and negative holdout all visible in every
   human and machine completion surface?
5. Is `passed=true` on the validator understood strictly as artifact/replay
   correctness while strategy approval remains false?

Disposition: the historical final had deterministic replay evidence before the
post-holdout audit, but it is no longer a valid current experiment. Structural
remediation is test-verified and independently Luna-reviewed for future unseen
data. The strategy and current experiment are rejected for live use and remain
paper-research-only. A direct Fable auditor verdict is still pending; this
packet is the handoff surface and must not be interpreted as self-approval.
