# Wave-9 capital-native manual QA matrix

목표: Wave-9 round A/B/C runner와 wave9 전용 테스트를 실제로 재실행하고, 각 round의 JSON/report UTF-8, 결정성, `eligible=[]`, source-manifest 검증을 확인한다.

## artifactRefs

| id | kind | description | path |
|---|---|---|---|
| A-W9-EXEC | command-log | pytest, compileall, exact isolated round A/B/C runs, UTF-8 readback, deterministic hashes, malformed input, provenance, cleanup | `.omo/evidence/wave9-review/a1/wave9-qa-execution-evidence.md` |

## surfaceEvidence

| scenario id | criterion reference | surface | exact invocation | verdict | artifactRefs |
|---|---|---|---|---|---|
| W9-SE-001 | wave9 tests | pytest | `python -m pytest -q research/wave9_capital_native/tests` | PASS (3 passed, exit 0) | A-W9-EXEC |
| W9-SE-002 | test rerun stability | pytest | same command, second run | PASS (3 passed, exit 0) | A-W9-EXEC |
| W9-SE-003 | wave9 compile | Python compileall | `python -m compileall -q research/wave9_capital_native` | PASS (exit 0) | A-W9-EXEC |
| W9-SE-004 | full repository compile | Python compileall | `python -m compileall -q research` | FAIL (exit 1; pre-existing wave1-rwa booster_live.py syntax error, outside wave9) | A-W9-EXEC |
| W9-SE-005 | round A execution | Python module CLI | `python -m research.wave9_capital_native.run_wave9 --round a` | PASS in isolated copy (exit 0, WAVE9_DONE, eligible=[]) | A-W9-EXEC |
| W9-SE-006 | round B execution | Python module CLI | `python -m research.wave9_capital_native.run_wave9 --round b` | PASS in isolated copy (exit 0, WAVE9_DONE, eligible=[]) | A-W9-EXEC |
| W9-SE-007 | round C execution | Python module CLI | `python -m research.wave9_capital_native.run_wave9 --round c` | PASS in isolated copy (exit 0, WAVE9_DONE, eligible=[]) | A-W9-EXEC |
| W9-SE-008 | JSON schema/readback | output JSON | Python `json.loads` for wave9/wave9b/wave9c results | PASS (10/6/5 candidates, all eligible lists empty, all candidates fail at least one gate, manifest SHA matches) | A-W9-EXEC |
| W9-SE-009 | report/JSON UTF-8 | output JSON and Markdown | `read_text(encoding='utf-8')`, replacement count | PASS (isolated and existing source reports decode; replacement count 0) | A-W9-EXEC |
| W9-SE-010 | deterministic rerun | isolated JSON/report outputs | run each round twice and compare bytes/hashes | PASS (all three JSON and reports byte-identical) | A-W9-EXEC |
| W9-SE-011 | report provenance | existing source Markdown reports | compare `Runner SHA-256` lines with current `run_wave9.py` SHA | PARTIAL (wave9b matches; wave9a and wave9c carry stale/different SHA) | A-W9-EXEC |

## adversarialCases

| scenario id | criterion reference | adversarial class | expected behavior | verdict | artifactRefs |
|---|---|---|---|---|---|
| W9-ADV-001 | invalid round | malformed CLI flag/value | Argparse rejects unknown round with non-zero exit and no completion marker | PASS (exit 2, no WAVE9_DONE) | A-W9-EXEC |
| W9-ADV-002 | unknown candidate | invalid `--only` | Runner rejects candidate before data processing and emits no completion marker | PASS (exit 2, no WAVE9_DONE) | A-W9-EXEC |
| W9-ADV-003 | cache integrity | corrupted manifest | `verify_manifest` fails closed before simulation and original manifest can be restored | PASS (exit 1, no WAVE9_DONE, SHA restored) | A-W9-EXEC |
| W9-ADV-004 | repeated execution | flaky/determinism | Same source cache and seed produce byte-identical JSON/report for A/B/C | PASS | A-W9-EXEC |
| W9-ADV-005 | misleading success | empty eligibility with successful process marker | Successful run must still report `eligible=[]`; no candidate is silently promoted | PASS | A-W9-EXEC |
| W9-ADV-006 | dirty worktree | pre-existing dirty/untracked files | QA must not reset or overwrite source files or existing reports | PASS (isolated runner; temp removed; source tracked state untouched) | A-W9-EXEC |
| W9-ADV-007 | prompt injection | prompt/user text | not_applicable: local argparse/data pipeline has no prompt or generated-text input | not_applicable | A-W9-EXEC |
| W9-ADV-008 | cancel/resume | state lifecycle | not_applicable: batch runner has no persistent cancel/resume state | not_applicable | A-W9-EXEC |
| W9-ADV-009 | stale state | stale session metadata | not_applicable: runner reads only frozen cache and registered candidate constants | not_applicable | A-W9-EXEC |

## QA verdict

`PARTIAL`: Wave-9 code/tests, all A/B/C reruns, UTF-8 outputs, deterministic hashes, empty eligibility, and fail-closed cache/CLI validation PASS. Full repository compileall remains blocked by the pre-existing Wave-1-RWA syntax error. Additionally, the existing A/C Markdown reports contain a runner SHA different from the current source runner; their metrics and JSON remain consistent, but provenance should be refreshed by the owning implementation lane.
