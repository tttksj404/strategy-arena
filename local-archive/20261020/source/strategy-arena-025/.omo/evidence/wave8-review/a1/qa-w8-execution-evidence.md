# Wave-8 manual QA execution evidence

대상 저장소: `C:\Users\SSAFY\Desktop\2학기\.domain-separation-20260715\strategy-arena`
실행일: 2026-07-21 (Asia/Seoul)
검증자는 제품 파일을 수정하지 않았고, wave8 runner는 임시 복사본에서만 실행했다.

## CLI help

Invocation: `python research/wave8_capital/run_capital100.py --help`

```text
exit=0
usage: run_capital100.py [-h]

Run the frozen $100 capital-feasibility check.

options:
  -h, --help  show this help message and exit
```

## Invalid flag

Invocation: `python research/wave8_capital/run_capital100.py --unknown`

```text
exit=2
python.exe : usage: run_capital100.py [-h]
run_capital100.py: error: unrecognized arguments: --unknown
```

## Compileall

Targeted invocation: `python -m compileall -q research/wave5 research/wave7 research/validation research/wave8_capital`

```text
exit=0
```

Full invocation: `python -m compileall -q research`

```text
exit=1
*** Error compiling 'research\\wave1-rwa\\live\\booster_live.py'...
  File "research\\wave1-rwa\\live\\booster_live.py", line 2
    """Booster live executor \\u2014 regime-switched cross-sectional momentum on Bitget USDT-M.
    ^
SyntaxError: unterminated triple-quoted string literal (detected at line 12)
```

The failure is outside wave8 and the file is unchanged in the worktree.

## Pytest

Combined invocation: `python -m pytest -q research/wave5/tests research/wave7/tests research/validation/tests`

```text
.....................                                                    [100%]
21 passed in 3.70s
exit=0
```

Per-suite invocations:

```text
research/wave5/tests: 7 passed in 0.73s; exit=0
research/wave7/tests: 9 passed in 3.51s; exit=0
research/validation/tests: 5 passed in 0.21s; exit=0
```

Rerun of combined invocation:

```text
.....................                                                    [100%]
21 passed in 3.42s
exit=0
```

## Wave-8 runner (isolated copy)

The runner was copied with `SPEC.md` and the five declared JSON inputs to:
`C:\Users\SSAFY\AppData\Local\Temp\wave8-qa-5e7ee6030bfd4897809cb1bc820f4261`.

Invocation: `python research/wave8_capital/run_capital100.py`

```text
WAVE8_CAPITAL100_DONE
{"eligible": [], "raw_return_winner": "W7c"}
exit=0
```

Independent readback from the generated JSON:

```text
candidate_count=5
eligible=[]
winner=W7c
all_gates={"W7c": false, "W2c": false, "W7d": false, "W7a": false, "W7b": false}
gate_matrix={"W2c": {"block_mdd": true, "capital_feasibility": true, "mc_p05": true, "ruin": true, "source_gates": false}, "W7a": {"block_mdd": true, "capital_feasibility": false, "mc_p05": true, "ruin": true, "source_gates": false}, "W7b": {"block_mdd": true, "capital_feasibility": false, "mc_p05": true, "ruin": true, "source_gates": false}, "W7c": {"block_mdd": true, "capital_feasibility": false, "mc_p05": true, "ruin": true, "source_gates": false}, "W7d": {"block_mdd": true, "capital_feasibility": false, "mc_p05": true, "ruin": true, "source_gates": false}}
```

## UTF-8 and deterministic rerun checks

Both reruns used the same isolated inputs and fixed simulation seed. The JSON `generated_at` differed as expected, but after removing that field the parsed JSON objects were equal. The report comparison removed only its generated timestamp line.

```text
normalized_json_equal=True
normalized_report_equal=True
report_utf8_replacement_first=0
report_utf8_replacement_second=0
report_first_title_escape=# Wave-8 $100 \\uc790\\ubcf8 \\ud310\\uc815 \\ubcf4\\uace0\\uc11c
report_line_count_first=22 second=22
```

## Malformed JSON adversarial probe

In the isolated copy only, `research/wave2/results/W2c.json` was temporarily replaced with the truncated bytes `{"candidate_id":`, then restored.

```text
exit=1
success_marker_present=False
Traceback (most recent call last):
  ...
json.decoder.JSONDecodeError
```

## Worktree preservation

The dirty worktree was inspected before and after QA commands. At the post-QA checkpoint the same top-level dirty entries remained (`?? .omo/` and `?? research/wave8_capital/`); no tracked product entry was introduced by QA:

```text
status_count=32
status_sha=d6d5e698fe74fe688e6f745b9dde37395a65c2d1c4547a7817c5dde0ef633a86
wave8_status:
?? research/wave8_capital/
```

No product source, tracked result, or pre-existing dirty file was reset or overwritten by this QA pass.

## Cleanup

The isolated temporary directory `C:\Users\SSAFY\AppData\Local\Temp\wave8-qa-5e7ee6030bfd4897809cb1bc820f4261` was removed after the final successful rerun. The two evidence markdown files in this attempt directory are intentional QA deliverables.
