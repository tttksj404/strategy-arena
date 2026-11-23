# FABLE5 AUDIT PACKET wave10-event-driven-20260721

- Task: search for a genuinely different crypto strategy family for a total capital of US$100 after the prior cross-sectional Wave-9 family produced no promotion candidate.
- Date: 2026-07-21 (Asia/Seoul).
- Domain: crypto research `strategy-arena`; keep separate from the KRA/KCYCLE racing repository.
- Direct independent Fable5 audit: pending. This packet records implementation, reproducible checks, and a fail-closed decision.

## 1. Registered scope and gates

- Family: fixed-major event-driven Donchian breakout with optional funding-aware direction filter.
- Fixed universe: `BTCUSDT`, `ETHUSDT`, `SOLUSDT`; no roster optimization or leverage.
- Capital contract: `$100` total, `$10` reserve, `$90` active, gross cap `0.9` or `0.6`, maker `0.02%`, slippage `0.01%` per side, funding cashflow included. Minimum-order feasibility is normalized to a fresh `$100` OOS start; historical compounded notionals are diagnostic only.
- Timing: every feature is shifted one bar; entry/exit and the first target cashflow cannot see the target bar.
- Screen window: historical screen through `2025-09-30`; evaluation `2025-10-01` to `2026-07-14`. This OOS window was already observed by prior research, so `oos_independence=false` and no live promotion is allowed.
- Eight candidates were frozen in `research/wave10_event_driven/SPEC.md` (`E10a`-`E10h`), before the run.
- Promotion gates: validated data, fixed universe, independent OOS, order/capital feasibility, >=60 active OOS days, positive after-cost/funding return, Sharpe >=1, >=2/4 positive compound blocks, double-slippage positive, bootstrap p05 terminal >$100, ruin <5%, historical MDD <=25%, and block MDD p95 <=25%.

## 2. Implementation and artifacts

- `research/wave10_event_driven/SPEC.md`: frozen family, timing, capital contract, candidates, and fail-closed gates.
- `research/wave10_event_driven/run_wave10.py`: deterministic cache loader, gap/range validation, shifted event positions, costs/funding, bootstrap, block MDD, provenance hashes, and report writer.
- `research/wave10_event_driven/validate_wave10.py`: mission validator requiring exactly 8 results, no eligible candidates, and explicit non-independent OOS.
- `research/wave10_event_driven/results/wave10_results.json`: machine-readable result and provenance evidence.
- `research/wave10_event_driven/report/wave10_results.md`: human-readable report.
- `research/wave10_event_driven/results/validation.json`: validator result.

## 3. Reproducible evidence

- Full runner: `python -W ignore -m research.wave10_event_driven.run_wave10` -> `WAVE10_DONE`, `eligible=[]`.
- Mission validator: `python research/wave10_event_driven/validate_wave10.py` -> `{"status":"passed","passed":true}`.
- Regression suite: `python -W ignore -m pytest -q research/wave10_event_driven/tests research/wave9_capital_native/tests research/wave8_capital/tests research/wave5/tests research/wave7/tests research/validation/tests` -> `35 passed`.
- Compile check: `python -W ignore -m compileall -q research/wave10_event_driven` -> pass.
- CLI happy path: `python -W ignore research/wave10_event_driven/run_wave10.py --only E10h` -> `WAVE10_DONE`, `eligible=[]`.
- CLI bad candidate: `--only BAD` is rejected with `unknown registered candidate: BAD`.
- Independent structure check: exactly 8 unique IDs, empty eligible set, unchanged manifest hash, and validator `passed=true` -> `INDEPENDENT_WAVE10_STRUCTURE_PASS`.
- Determinism: result JSON SHA-256 before and after a second full run was identical: `61F0304363B596691962F98A9339F32A1C38BDAAE9CB2B048E081AF14D8FA499`.
- Recorded source manifest SHA-256: `2d1aa4772739cc0bab209a27a5684900db802c55c27c63a4a7979adf26e4c0e0` (before == after).
- Recorded runner SHA-256: `f570d34f7d79fd29139aa1fefdafcc0f47b24784556eb5210b48449852e3f733`.
- Recorded spec SHA-256: `783193d79307b1da8e671db69a890e043a68404fe08c76428cc16ea7a9da87e4`.
- The runner also records direct dependency hashes for Wave-1 costs, Wave-3 fetch, and the frozen Wave-9 runner.

## 4. Candidate decision

| Candidate | OOS return | Sharpe | stress return | bootstrap p05 | ruin | normalized min order | historical MDD | block MDD p95 | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| E10a | -40.43% | -2.20 | -40.54% | $39.11 | 28.41% | $27.93 | 52.86% | 59.81% | FAIL |
| E10b | -36.65% | -1.66 | -36.75% | $39.56 | 25.38% | $27.98 | 67.09% | 60.51% | FAIL |
| E10c | -24.69% | -2.36 | -24.73% | $59.69 | 0.39% | $28.72 | 62.26% | 44.24% | FAIL |
| E10d | -33.19% | -1.94 | -33.23% | $45.03 | 12.55% | $28.07 | 74.18% | 55.33% | FAIL |
| E10e | -15.13% | -0.35 | -15.38% | $48.53 | 9.57% | $13.79 | 66.40% | 58.34% | FAIL |
| E10f | -11.83% | -0.19 | -11.98% | $47.89 | 10.17% | $18.20 | 65.97% | 54.45% | FAIL |
| E10g | -8.27% | -1.12 | -8.30% | $79.87 | 0.00% | $27.48 | 28.76% | 21.36% | FAIL |
| E10h | **7.59%** | 0.48 | **7.50%** | $72.64 | 0.08% | $15.36 | 33.21% | 34.19% | FAIL |

E10h is the only positive OOS return, but it fails independent-OOS, Sharpe, historical MDD, bootstrap p05, and block-MDD gates. Therefore the new direction does not produce a $100 promotion candidate.

## 5. Residual risk and next action

- This is a historical screen, not a live or paper-trading approval. The positive E10h result is a shortlist signal only, not evidence of deployable edge.
- The observed OOS interval was not independent and the sample is regime-specific; prospective paper observation would be required before any fresh OOS claim.
- No exchange API, wallet, key, or order was used.
- Existing dirty worktree changes were preserved; this Wave-10 lane is currently uncommitted and unpushed.
- Status: `PENDING_EXTERNAL_FABLE5`; promotion/live-capital approval: `FAIL`.

## 6. Reproduction commands

```text
python -W ignore -m research.wave10_event_driven.run_wave10
python research/wave10_event_driven/validate_wave10.py
python -W ignore -m compileall -q research/wave10_event_driven
python -W ignore -m pytest -q research/wave10_event_driven/tests research/wave9_capital_native/tests research/wave8_capital/tests research/wave5/tests research/wave7/tests research/validation/tests
```
