# Wave-8 manual QA matrix

목표: `wave8-capital100-20260721`의 CLI, runner, UTF-8 산출물, 재현성, fail-closed 결과를 실제 실행으로 검증한다.

판정 규칙: PASS는 반드시 아래 execution evidence artifact를 가리킨다. 전체 `research` compileall은 wave8 외부의 기존 문법 오류로 FAIL이며, wave8 관련 범위 compileall과 모든 지정 pytest는 PASS다.

## artifactRefs

| id | kind | description | path |
|---|---|---|---|
| A-W8-EXEC | command-log | 실제 명령, exit code, output, readback, UTF-8/재현성, malformed probe, worktree 보존 증거 | `.omo/evidence/wave8-review/a1/qa-w8-execution-evidence.md` |

## surfaceEvidence

| scenario id | criterion reference | surface | exact invocation | verdict | artifactRefs |
|---|---|---|---|---|---|
| W8-SE-001 | CLI help | Python CLI | `python research/wave8_capital/run_capital100.py --help` | PASS (exit 0, usage와 description 표시) | A-W8-EXEC |
| W8-SE-002 | invalid flag rejection | Python CLI | `python research/wave8_capital/run_capital100.py --unknown` | PASS (exit 2, unrecognized arguments, success marker 없음) | A-W8-EXEC |
| W8-SE-003 | wave8 compile | Python bytecode compile | `python -m compileall -q research/wave5 research/wave7 research/validation research/wave8_capital` | PASS (exit 0) | A-W8-EXEC |
| W8-SE-004 | repository compile | Python bytecode compile | `python -m compileall -q research` | FAIL (exit 1; 기존 `research/wave1-rwa/live/booster_live.py:2` unterminated triple-quoted string) | A-W8-EXEC |
| W8-SE-005 | wave5/wave7/validation tests | pytest | `python -m pytest -q research/wave5/tests research/wave7/tests research/validation/tests` | PASS (21 passed, exit 0) | A-W8-EXEC |
| W8-SE-006 | flaky-test rerun | pytest | same combined pytest invocation, second run | PASS (21 passed, exit 0; first/second green) | A-W8-EXEC |
| W8-SE-007 | wave8 decision runner | isolated temp copy of runner and declared JSON inputs | `python research/wave8_capital/run_capital100.py` | PASS (exit 0, `WAVE8_CAPITAL100_DONE`, eligible `[]`, raw winner `W7c`) | A-W8-EXEC |
| W8-SE-008 | decision readback | generated JSON | Python JSON parse/readback of `research/wave8_capital/results/capital100.json` | PASS (5 candidates; every `all_gates_pass` false; W2c source gate false) | A-W8-EXEC |
| W8-SE-009 | UTF-8 report/JSON readability | output files | Python `read_text(encoding='utf-8')`, `json.loads`, replacement count | PASS (decode succeeds, replacement count 0, schema keys present, title codepoints valid) | A-W8-EXEC |
| W8-SE-010 | deterministic rerun markers | isolated runner outputs | run twice; compare JSON after removing `generated_at`, report after removing generated timestamp | PASS (`normalized_json_equal=True`, `normalized_report_equal=True`) | A-W8-EXEC |

## adversarialCases

| scenario id | criterion reference | adversarial class | expected behavior | verdict | artifactRefs |
|---|---|---|---|---|---|
| W8-ADV-001 | malformed input | truncated JSON fixture | Runner exits non-zero, emits no completion marker, does not claim eligibility | PASS (exit 1, `success_marker_present=False`, JSONDecodeError; isolated copy restored) | A-W8-EXEC |
| W8-ADV-002 | invalid input | unknown CLI flag | Argparse rejects the flag with exit 2 and no success marker | PASS | A-W8-EXEC |
| W8-ADV-003 | misleading success output | success marker/decision contradiction | Exit 0 only on valid input; valid result explicitly has `eligible=[]` and all five candidates fail at least one gate | PASS | A-W8-EXEC |
| W8-ADV-004 | repeated run | deterministic rerun | Repeated valid runs preserve parsed metrics and report content apart from generated timestamp | PASS | A-W8-EXEC |
| W8-ADV-005 | dirty worktree | pre-existing modifications/untracked wave8 files | QA must not reset, overwrite, or hide unrelated dirty files | PASS (post checkpoint status count/hash recorded; only isolated temp outputs changed) | A-W8-EXEC |
| W8-ADV-006 | prompt injection | user-supplied prompt text | Not applicable: runner is a local argparse/JSON batch process with no prompt or text-generation surface | not_applicable | A-W8-EXEC |
| W8-ADV-007 | cancel/resume | interrupted state lifecycle | Not applicable: runner has no persistent state, resume, cancel, or session store | not_applicable | A-W8-EXEC |
| W8-ADV-008 | stale state | stale/mismatched state files | Not applicable: runner reads only the five declared input JSON files and SPEC path | not_applicable | A-W8-EXEC |
| W8-ADV-009 | hung command | bounded long-running process | Not applicable: fixed 10,000 MC and 1,000 block paths complete as a finite local batch; no service/child-process lifecycle | not_applicable | A-W8-EXEC |

## QA verdict

`PARTIAL`: wave8 behavior, fail-closed output, UTF-8 files, deterministic metrics, malformed-input handling, and designated pytest suites all PASS. A repository-wide compileall remains FAIL because of the pre-existing truncated `research/wave1-rwa/live/booster_live.py`; this file is outside wave8 and was not modified.
