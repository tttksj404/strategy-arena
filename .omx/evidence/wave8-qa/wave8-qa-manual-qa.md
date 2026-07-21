# manualQa

수동 CLI QA는 `C:\Users\SSAFY\Desktop\2학기\strategy-arena-crypto`에서 실행했다. `omo ulw-loop status --json`는 이 호스트의 OMO 런타임 부재/구문 오류로 attempt 경로를 반환하지 않아 caller evidence 디렉터리인 `.omx/evidence/wave8-qa`를 사용했다.

## surfaceEvidence

| scenario id | criterion reference | surface | exact invocation | verdict | artifactRefs |
|---|---|---|---|---|---|
| SE-W8-HELP | CLI-H1 | Wave-8 runner CLI | `python research/wave8_alternative/run_wave8.py --help` | PASS (exit 0, all registered choices shown) | A-W8-HELP |
| SE-W9-HELP | CLI-H1 | Wave-9 runner CLI | `python research/wave9_methods/run_wave9.py --help` | PASS (exit 0, all registered choices shown) | A-W9-HELP |
| SE-W10-HELP | CLI-H1 | Wave-10 runner CLI | `python research/wave10_ensemble/run_wave10.py --help` | PASS (exit 0, all registered choices shown) | A-W10-HELP |
| SE-W8-VALID | CLI-V1 | Wave-8 single-candidate run | `python research/wave8_alternative/run_wave8.py --only R8a` | PASS (exit 0; result emitted) | A-W8-VALID |
| SE-W9-VALID | CLI-V1 | Wave-9 single-candidate run | `python research/wave9_methods/run_wave9.py --only T9a` | PASS (exit 0; result emitted) | A-W9-VALID |
| SE-W10-VALID | CLI-V1 | Wave-10 single-candidate run | `python research/wave10_ensemble/run_wave10.py --only E10a` | PASS (exit 0; result emitted) | A-W10-VALID |
| SE-W8-INVALID | CLI-I1 | Wave-8 argument parser | `python research/wave8_alternative/run_wave8.py --only DOES_NOT_EXIST` | PASS (exit 2; invalid choice rejected) | A-W8-INVALID |
| SE-W9-INVALID | CLI-I1 | Wave-9 argument parser | `python research/wave9_methods/run_wave9.py --only DOES_NOT_EXIST` | PASS (exit 2; invalid choice rejected) | A-W9-INVALID |
| SE-W10-INVALID | CLI-I1 | Wave-10 argument parser | `python research/wave10_ensemble/run_wave10.py --only DOES_NOT_EXIST` | PASS (exit 2; invalid choice rejected) | A-W10-INVALID |
| SE-W8-FULL | RUN-F1 | Wave-8 full runner and report writer | `python research/wave8_alternative/run_wave8.py` | PASS (exit 0; 16 candidates; `WAVE8_DONE eligible=[]`) | A-W8-FULL |
| SE-W9-FULL | RUN-F1 | Wave-9 full runner and report writer | `python research/wave9_methods/run_wave9.py` | PASS (exit 0; 16 candidates; `WAVE9_DONE eligible=[]`) | A-W9-FULL |
| SE-W10-FULL | RUN-F1 | Wave-10 full runner and report writer | `python research/wave10_ensemble/run_wave10.py` | PASS (exit 0; 6 candidates; `WAVE10_DONE eligible=[]`) | A-W10-FULL |
| SE-W8-VALIDATOR | VAL-1 | Wave-8 validator | `python research/wave8_alternative/validate_wave8.py` | PASS (exit 0; `WAVE8_VALIDATION_PASS candidates=16 eligible=[]`) | A-W8-VALPOST |
| SE-W9-VALIDATOR | VAL-1 | Wave-9 validator | `python research/wave9_methods/validate_wave9.py` | PASS (exit 0; `WAVE9_VALIDATION_PASS candidates=16 eligible=[] selection_independent=false`) | A-W9-VALPOST |
| SE-W10-VALIDATOR | VAL-1 | Wave-10 validator | `python research/wave10_ensemble/validate_wave10.py` | PASS (exit 0; `WAVE10_VALIDATION_PASS candidates=6 eligible=[] selection_independent=false`) | A-W10-VALPOST |
| SE-W8-TESTS | TEST-1 | Wave-8 targeted pytest | `python -m pytest -q research/wave8_alternative/tests` | PASS (7 passed) | A-W8-TESTS |
| SE-W9-TESTS | TEST-1 | Wave-9 targeted pytest | `python -m pytest -q research/wave9_methods/tests` | PASS (4 passed) | A-W9-TESTS |
| SE-W10-TESTS | TEST-1 | Wave-10 targeted pytest | `python -m pytest -q research/wave10_ensemble/tests` | PASS (2 passed) | A-W10-TESTS |
| SE-GENERATED | ART-1 | Aggregate/individual result and report artifacts | Parse all wave aggregates/manifests; compare IDs, counts, max gross, non-empty reports, and manifest SHA-256 | PASS (16/16/6 IDs and files; data valid; max gross within cap; zero eligible; hashes match) | A-GENERATED |
| SE-HASHES | ART-1 | Final aggregate/report fingerprints | `Get-FileHash -Algorithm SHA256` on all six aggregate/report files after latest full runners | PASS (non-empty final hashes recorded) | A-AGGREGATE-HASHES |
| SE-REPORT-SEMANTICS | ART-2 | Generated research reports | `Select-String` for verdict/selection markers in all three reports | PASS (all report no-live/none-eligible and selection caveats present) | A-REPORT-SEMANTICS |
| SE-W8-INDEPENDENCE | GATE-3 | Wave-8 independence gate and provenance markers | Parse `wave8_results.json`; compare `oos_independent` statuses with report/SPEC markers | PASS (16/16 independence gates FAIL; report says selection-independent=false; registration/OOS dates are disclosed) | A-W8-INDEPENDENCE |
| SE-W8-NOLOOKAHEAD | GATE-4 | Wave-8 volume-signal timing | `python .omx/evidence/wave8-qa/wave8_no_lookahead_probe.py` mutates probe-day BTC volume and compares `_volume_signal` positions | PASS (current-day mutation leaves positions invariant; row L1 diff 0) | A-W8-NOLOOKAHEAD |
| SE-CACHE | DATA-1 | Cache prerequisite check | `Test-Path research/wave1/cache; Test-Path research/wave3/cache; Get-ChildItem research/wave3/cache -File | Measure-Object` | PASS for new waves (wave3 cache present, 1043 files); wave1 cache absent is pre-existing/out-of-scope | A-CACHE |
| SE-REPO-ALL | REG-1 | Whole-repository pytest collection | `python -m pytest -q` | FAIL (3 pre-existing `research/wave1-rwa` collection errors: `ModuleNotFoundError: No module named 'src'`; no wave8/9/10 failure) | A-REPO-ALL |
| SE-REPO-NOWAVE1RWA | REG-2 | Regression suite excluding unrelated wave1-rwa package | `python -m pytest -q --ignore research/wave1-rwa` | FAIL (87 passed, 1 skipped, 2 pre-existing Wave-4 failures because `research/wave1/cache/universe.json` is absent; no wave8/9/10 failure) | A-REPO-NOWAVE1RWA |

## adversarialCases

| scenario id | criterion reference | adversarial class | expected behavior | verdict | artifactRefs |
|---|---|---|---|---|---|
| AC-W8-UNKNOWN | CLI-I1 | unknown candidate selector | Reject unregistered `--only` with argparse error and non-zero status | PASS (exit 2) | A-W8-INVALID |
| AC-W9-UNKNOWN | CLI-I1 | unknown candidate selector | Reject unregistered `--only` with argparse error and non-zero status | PASS (exit 2) | A-W9-INVALID |
| AC-W10-UNKNOWN | CLI-I1 | unknown candidate selector | Reject unregistered `--only` with argparse error and non-zero status | PASS (exit 2) | A-W10-INVALID |
| AC-CAPITAL | GATE-1 | gross-cap violation probe | Generated candidates must remain at or below the 0.60 gross cap | PASS (zero `bad_max_gross` across all 38 results) | A-GENERATED |
| AC-FAIL-CLOSED | GATE-2 | apparent positive/backtest winner | No candidate should be eligible/live-approved solely from positive OOS evidence | PASS (all_gates_pass_count=0; reports say eligible none/no live capital) | A-GENERATED, A-REPORT-SEMANTICS |
| AC-W8-OOS-DEPENDENCE | GATE-3 | historical split presented as independent OOS | Wave-8 must fail its independence gate when prior waves were inspected and report the dependency | PASS (0/16 `oos_independent` passes; validator asserts false) | A-W8-INDEPENDENCE, A-W8-VALPOST |
| AC-W8-NOLOOKAHEAD | GATE-4 | future-data timing perturbation | Changing volume at time *t* must not change the signal/position at the same time *t* | PASS (current implementation shifts volume to prior-day before z-score; probe invariant=true) | A-W8-NOLOOKAHEAD |
| AC-SELECTION-LEAKAGE | GATE-3 | selection dependence disclosure | Wave-9 and Wave-10 must disclose selection-independent=false | PASS (validators and reports explicitly disclose false) | A-W9-VALPOST, A-W10-VALPOST, A-REPORT-SEMANTICS |
| AC-MISSING-WAVE1-CACHE | DATA-1 | legacy cache absence | Mark not applicable to new runners when they use the existing Wave-3 cache and expose no Wave-1 cache override | NOT_APPLICABLE — Wave-8/9/10 load `research/wave3/cache`; `research/wave1/cache` is a pre-existing legacy dependency outside these runners | A-CACHE |
| AC-REPO-COLLECTION | REG-1 | unrelated suite contamination | Whole suite should collect without unrelated import errors | FAIL (pre-existing `wave1-rwa` `src` import errors; targeted new-wave suites remain green) | A-REPO-ALL |
| AC-WAVE1-CACHE | DATA-2 | legacy Wave-1 cache dependency | Existing Wave-4 reconciliation tests should have `research/wave1/cache/universe.json` available | FAIL (pre-existing checkout prerequisite missing; two Wave-4 tests fail, while new waves use Wave-3 cache and pass) | A-REPO-NOWAVE1RWA, A-CACHE |

## artifactRefs

| id | kind | description | path |
|---|---|---|---|
| A-W8-HELP | cli-log | Wave-8 help output and exit code | `C:\Users\SSAFY\Desktop\2학기\strategy-arena-crypto\.omx\evidence\wave8-qa\w8_help.txt` |
| A-W9-HELP | cli-log | Wave-9 help output and exit code | `C:\Users\SSAFY\Desktop\2학기\strategy-arena-crypto\.omx\evidence\wave8-qa\w9_help.txt` |
| A-W10-HELP | cli-log | Wave-10 help output and exit code | `C:\Users\SSAFY\Desktop\2학기\strategy-arena-crypto\.omx\evidence\wave8-qa\w10_help.txt` |
| A-W8-VALID | cli-log | Wave-8 valid `--only R8a` run | `C:\Users\SSAFY\Desktop\2학기\strategy-arena-crypto\.omx\evidence\wave8-qa\w8_valid_only_R8a.txt` |
| A-W9-VALID | cli-log | Wave-9 valid `--only T9a` run | `C:\Users\SSAFY\Desktop\2학기\strategy-arena-crypto\.omx\evidence\wave8-qa\w9_valid_only_T9a.txt` |
| A-W10-VALID | cli-log | Wave-10 valid `--only E10a` run | `C:\Users\SSAFY\Desktop\2학기\strategy-arena-crypto\.omx\evidence\wave8-qa\w10_valid_only_E10a.txt` |
| A-W8-INVALID | cli-log | Wave-8 invalid selector rejection | `C:\Users\SSAFY\Desktop\2학기\strategy-arena-crypto\.omx\evidence\wave8-qa\w8_invalid_only.txt` |
| A-W9-INVALID | cli-log | Wave-9 invalid selector rejection | `C:\Users\SSAFY\Desktop\2학기\strategy-arena-crypto\.omx\evidence\wave8-qa\w9_invalid_only.txt` |
| A-W10-INVALID | cli-log | Wave-10 invalid selector rejection | `C:\Users\SSAFY\Desktop\2학기\strategy-arena-crypto\.omx\evidence\wave8-qa\w10_invalid_only.txt` |
| A-W8-FULL | cli-log | Wave-8 full run output | `C:\Users\SSAFY\Desktop\2학기\strategy-arena-crypto\.omx\evidence\wave8-qa\w8_full_runner.txt` |
| A-W9-FULL | cli-log | Wave-9 full run output | `C:\Users\SSAFY\Desktop\2학기\strategy-arena-crypto\.omx\evidence\wave8-qa\w9_full_runner.txt` |
| A-W10-FULL | cli-log | Wave-10 full run output | `C:\Users\SSAFY\Desktop\2학기\strategy-arena-crypto\.omx\evidence\wave8-qa\w10_full_runner.txt` |
| A-W8-VALPOST | validator-log | Wave-8 post-full validator | `C:\Users\SSAFY\Desktop\2학기\strategy-arena-crypto\.omx\evidence\wave8-qa\w8_validator_post_full.txt` |
| A-W9-VALPOST | validator-log | Wave-9 post-full validator | `C:\Users\SSAFY\Desktop\2학기\strategy-arena-crypto\.omx\evidence\wave8-qa\w9_validator_post_full.txt` |
| A-W10-VALPOST | validator-log | Wave-10 post-full validator | `C:\Users\SSAFY\Desktop\2학기\strategy-arena-crypto\.omx\evidence\wave8-qa\w10_validator_post_full.txt` |
| A-W8-TESTS | test-log | Wave-8 targeted tests | `C:\Users\SSAFY\Desktop\2학기\strategy-arena-crypto\.omx\evidence\wave8-qa\w8_tests.txt` |
| A-W9-TESTS | test-log | Wave-9 targeted tests | `C:\Users\SSAFY\Desktop\2학기\strategy-arena-crypto\.omx\evidence\wave8-qa\w9_tests.txt` |
| A-W10-TESTS | test-log | Wave-10 targeted tests | `C:\Users\SSAFY\Desktop\2학기\strategy-arena-crypto\.omx\evidence\wave8-qa\w10_tests.txt` |
| A-GENERATED | artifact-check | Generated aggregate/individual result and report/hash checks | `C:\Users\SSAFY\Desktop\2학기\strategy-arena-crypto\.omx\evidence\wave8-qa\generated_files_check.txt` |
| A-AGGREGATE-HASHES | hash-log | Final aggregate/report SHA-256 fingerprints | `C:\Users\SSAFY\Desktop\2학기\strategy-arena-crypto\.omx\evidence\wave8-qa\aggregate_hashes.txt` |
| A-REPORT-SEMANTICS | artifact-check | Report verdict and selection disclosure checks | `C:\Users\SSAFY\Desktop\2학기\strategy-arena-crypto\.omx\evidence\wave8-qa\report_semantics_check.txt` |
| A-W8-INDEPENDENCE | artifact-check | Wave-8 independence gate and date/report consistency | `C:\Users\SSAFY\Desktop\2학기\strategy-arena-crypto\.omx\evidence\wave8-qa\wave8_independence_check.txt` |
| A-W8-NOLOOKAHEAD | adversarial-probe | Current-day volume mutation no-lookahead probe | `C:\Users\SSAFY\Desktop\2학기\strategy-arena-crypto\.omx\evidence\wave8-qa\wave8_no_lookahead_probe.txt` |
| A-CACHE | prerequisite-check | Wave-1/Wave-3 cache presence | `C:\Users\SSAFY\Desktop\2학기\strategy-arena-crypto\.omx\evidence\wave8-qa\cache_presence.txt` |
| A-REPO-ALL | test-log | Whole-repository pytest failure evidence | `C:\Users\SSAFY\Desktop\2학기\strategy-arena-crypto\.omx\evidence\wave8-qa\repo_tests_all.txt` |
| A-REPO-NOWAVE1RWA | test-log | Whole-repository pytest excluding wave1-rwa; isolates missing Wave-1 cache failures | `C:\Users\SSAFY\Desktop\2학기\strategy-arena-crypto\.omx\evidence\wave8-qa\repo_tests_no_wave1rwa.txt` |

## QA judgement

CLI/runtime behavior and the three new-wave validators are reproducibly green, with all candidates fail-closed and no live-capital eligibility. The repository-wide regression gate remains red only for pre-existing wave1-rwa import errors and missing Wave-1 cache prerequisites; those are not new Wave-8/9/10 failures. This manual QA record does not override separate code-review findings about economic/timing contract correctness.
