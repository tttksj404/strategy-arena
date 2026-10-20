# Wave-3 CLI/library manual QA transcript

Repository: `C:\Users\SSAFY\Desktop\2학기\.domain-separation-20260715\strategy-arena`

Python: `C:\Users\SSAFY\Desktop\2학기\strategy-arena\.venv\Scripts\python.exe` (Python 3.12.12)

Network policy: `research/wave3/run_wave3.py --stage fetch` was not invoked; no external network call was authorized.

## SUR-HELP

Surface: terminal CLI.

Exact invocation:

```text
C:\Users\SSAFY\Desktop\2학기\strategy-arena\.venv\Scripts\python.exe research/wave3/run_wave3.py --help
```

```text
EXIT_CODE=0
STDOUT_BEGIN
usage: run_wave3.py [-h] --stage {fetch,run,gates,report}
                    [--only {W3a,W3b,W3c,W3d,W3e,W3f}] [--force]

Wave-3 preregistered research pipeline

options:
  -h, --help            show this help message and exit
  --stage {fetch,run,gates,report}
  --only {W3a,W3b,W3c,W3d,W3e,W3f}
  --force
STDOUT_END
STDERR_BEGIN
STDERR_END
```

## ADV-WRONG-ORDER-GATES-BEFORE-RUN

Surface: terminal CLI; gates before any run result exists.

Exact invocation:

```text
C:\Users\SSAFY\Desktop\2학기\strategy-arena\.venv\Scripts\python.exe research/wave3/run_wave3.py --stage gates --only W3e
```

```text
EXIT_CODE=1
STDOUT_BEGIN
STDOUT_END
STDERR_BEGIN
Traceback (most recent call last):
  ...
FileNotFoundError: C:\Users\SSAFY\Desktop\2학기\.domain-separation-20260715\strategy-arena\research\wave3\results\W3e.json
STDERR_END
```

## ADV-MISSING-MANIFEST-RUN-BEFORE-FETCH

Surface: terminal CLI; run before fetch manifest exists.

Exact invocation:

```text
C:\Users\SSAFY\Desktop\2학기\strategy-arena\.venv\Scripts\python.exe research/wave3/run_wave3.py --stage run --only W3e
```

```text
EXIT_CODE=1
STDOUT_BEGIN
STDOUT_END
STDERR_BEGIN
Traceback (most recent call last):
  ...
research.wave1.common.PipelineError: wave-3 fetch stage has not produced cache/manifest.json
STDERR_END
```

Postcondition: `research/wave3/cache/manifest.json` and `research/wave3/results/W3e.json` were absent.

## SUR-TEST-WAVE3

Surface: terminal pytest CLI.

Exact invocation:

```text
C:\Users\SSAFY\Desktop\2학기\strategy-arena\.venv\Scripts\python.exe -m pytest -p no:cacheprovider research/wave3/tests/test_wave3.py
```

```text
EXIT_CODE=0
STDOUT_BEGIN
============================= test session starts =============================
platform win32 -- Python 3.12.12, pytest-9.1.1, pluggy-1.6.0
rootdir: C:\Users\SSAFY\Desktop\2학기\.domain-separation-20260715\strategy-arena
collected 6 items

research\\wave3\\tests\\test_wave3.py ......                                [100%]

============================== 6 passed in 1.04s ==============================
STDOUT_END
STDERR_BEGIN
STDERR_END
```

## SUR-TEST-ALL

Surface: terminal pytest CLI.

Exact invocation:

```text
C:\Users\SSAFY\Desktop\2학기\strategy-arena\.venv\Scripts\python.exe -m pytest -p no:cacheprovider
```

```text
EXIT_CODE=0
STDOUT_BEGIN
============================= test session starts =============================
platform win32 -- Python 3.12.12, pytest-9.1.1, pluggy-1.6.0
rootdir: C:\Users\SSAFY\Desktop\2학기\.domain-separation-20260715\strategy-arena
collected 25 items

research\\wave1\\tests\\test_costs.py ..                                    [  8%]
research\\wave1\\tests\\test_donchian.py .                                  [ 12%]
research\\wave1\\tests\\test_fetch_bitget.py .                              [ 16%]
research\\wave1\\tests\\test_fix_packet_r1.py ...                           [ 28%]
research\\wave1\\tests\\test_funding_sign.py ..                             [ 36%]
research\\wave1\\tests\\test_gates.py .                                     [ 40%]
research\\wave1\\tests\\test_integrity.py .                                 [ 44%]
research\\wave1\\tests\\test_no_lookahead.py ....                           [ 60%]
research\\wave1\\tests\\test_voltarget.py .                                 [ 64%]
research\\wave2\\tests\\test_wave2.py ...                                   [ 76%]
research\\wave3\\tests\\test_wave3.py ......                                [100%]

============================= 25 passed in 2.46s ==============================
STDOUT_END
STDERR_BEGIN
STDERR_END
```

## SUR-SYNTHETIC-PUBLIC-RUN-CANDIDATE

Surface: terminal Python library driver; direct public `research.wave3.engine.run_candidate` call.

Exact invocation:

```text
C:\Users\SSAFY\Desktop\2학기\strategy-arena\.venv\Scripts\python.exe .omo/evidence/r1-python-library-fix/a1/wave3-synthetic-run-candidate.py
```

```text
EXIT_CODE=0
STDOUT_BEGIN
PUBLIC_RUN_CANDIDATE=PASS
candidate_id=W3e
equity_len=70
equity_all_non_null=True
STDOUT_END
STDERR_BEGIN
STDERR_END
```

The first attempt of this evidence-only harness stopped before product execution with `ModuleNotFoundError: No module named 'research'`; the harness was corrected to insert the verified repository root, then the exact invocation above passed. No product file was changed.

## ADV-MALFORMED-CLI

Surface: terminal CLI.

Exact invocations:

```text
C:\Users\SSAFY\Desktop\2학기\strategy-arena\.venv\Scripts\python.exe research/wave3/run_wave3.py --stage invalid
C:\Users\SSAFY\Desktop\2학기\strategy-arena\.venv\Scripts\python.exe research/wave3/run_wave3.py --stage run --only NOPE
```

Both returned `EXIT_CODE=2`, empty stdout, and argparse validation errors on stderr (`invalid Stage value` / `invalid choice`).

## ADV-FLAKY-REPEAT-WAVE3

The focused Wave-3 pytest invocation was repeated. It returned `EXIT_CODE=0`, collected 6 items, and reported `6 passed in 1.29s`; stderr was empty.

## Cleanup/state

- No fetch command was run.
- No Python or pytest process remained at final check.
- Existing dirty files were preserved; no product-file cleanup or reset was performed.
