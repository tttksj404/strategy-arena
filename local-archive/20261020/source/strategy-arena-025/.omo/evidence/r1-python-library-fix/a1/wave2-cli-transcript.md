# Wave-2 manual QA terminal transcript

Repository: `C:\Users\SSAFY\Desktop\2학기\.domain-separation-20260715\strategy-arena`
Python: `C:\Users\SSAFY\Desktop\2학기\strategy-arena\.venv\Scripts\python.exe`
Network: not used; the fetch guard also made `socket.connect` fail closed.

## Baseline and focused tests

Command:

```text
C:\Users\SSAFY\Desktop\2학기\strategy-arena\.venv\Scripts\python.exe -m pytest -p no:cacheprovider research/wave2/tests/test_wave2.py
```

Result: exit `0`; `3 passed in 0.53s`.

The same command was repeated twice: `REPEAT_1_EXIT=0`, `REPEAT_2_EXIT=0`, with `3 passed` each time.

Command:

```text
C:\Users\SSAFY\Desktop\2학기\strategy-arena\.venv\Scripts\python.exe -m pytest -p no:cacheprovider
```

Result: exit `0`; `19 passed in 1.27s`.

## CLI smoke

Help:

```text
C:\Users\SSAFY\Desktop\2학기\strategy-arena\.venv\Scripts\python.exe -m research.wave2.run_wave2 --help
```

Result: exit `0`; stages `{fetch,run,gates,report}` and candidates `{W2a,W2b,W2c,W2d,W2e,W2f,W2g}` were listed.

Current fetch:

```text
C:\Users\SSAFY\Desktop\2학기\strategy-arena\.venv\Scripts\python.exe -m research.wave2.run_wave2 --stage fetch --force
```

Result: exit `0`; `fetch: verified wave-1 cache only (131 files)`. `research/wave2/cache_manifest.json` contains `network_calls: false`, `force_ignored: true`, and `required_files: 131`.

Current W2a run:

```text
C:\Users\SSAFY\Desktop\2학기\strategy-arena\.venv\Scripts\python.exe -m research.wave2.run_wave2 --stage run --only W2a
```

Result: exit `0`; `candidate_id=W2a`, `family=F1`, `equity_count=1000`, `trade_count=22`, `cost_route=maker_0.02pct_per_leg_zero_slippage`, `exploratory_only=false`.

Current W2a gates and report:

```text
C:\Users\SSAFY\Desktop\2학기\strategy-arena\.venv\Scripts\python.exe -m research.wave2.run_wave2 --stage gates --only W2a
C:\Users\SSAFY\Desktop\2학기\strategy-arena\.venv\Scripts\python.exe -m research.wave2.run_wave2 --stage report
```

Result: `GATES_EXIT=0`, `GATE_COUNT=19`, `OOS_TRADE_COUNT=0`, `OOS_LABEL=UNTESTED_IN_OOS`; `REPORT_EXIT=0`, report 349 bytes, registry 310 bytes.

W2g run and gates:

```text
C:\Users\SSAFY\Desktop\2학기\strategy-arena\.venv\Scripts\python.exe -m research.wave2.run_wave2 --stage run --only W2g
C:\Users\SSAFY\Desktop\2학기\strategy-arena\.venv\Scripts\python.exe -m research.wave2.run_wave2 --stage gates --only W2g
```

Result: both exit `0`; W2g has `equity_count=8351`, `family=F3`, `exploratory_only=true`, `basis_mark_to_market_included=true`, `cost_route=maker_0.02pct_per_leg_zero_slippage`, `GATE_COUNT=19`, and `OOS_TRADE_COUNT=6888`.

W2d run and gates:

```text
C:\Users\SSAFY\Desktop\2학기\strategy-arena\.venv\Scripts\python.exe -m research.wave2.run_wave2 --stage run --only W2d
C:\Users\SSAFY\Desktop\2학기\strategy-arena\.venv\Scripts\python.exe -m research.wave2.run_wave2 --stage gates --only W2d
```

Result: run and gates exit `0`; W2d output has 2,509 equity samples, exactly the top-10 symbols, `entry_rate=0.0005`, `exit_apr=0.025`, maker cost route, and 19 gates.

Malformed CLI:

```text
C:\Users\SSAFY\Desktop\2학기\strategy-arena\.venv\Scripts\python.exe -m research.wave2.run_wave2 --stage invalid
C:\Users\SSAFY\Desktop\2학기\strategy-arena\.venv\Scripts\python.exe -m research.wave2.run_wave2 --stage run --only NOPE
```

Result: exits `2` and `2`; argparse rejected both values with usage/error output.

## Adversarial probes

Spike timing probe used synthetic 8-hour funding with the spike at `2026-01-01T16:00Z`:

```text
C:\Users\SSAFY\Desktop\2학기\strategy-arena\.venv\Scripts\python.exe -c "... spike_position(...) ..."
```

Observed: `FIRST_ACTIVE=2026-01-01T08:00:00+00:00`, before the actual 16:00 spike. Exit `0`, but the behavior is a product failure against the next-daily-bar/no-lookahead expectation.

W2g loader trace:

```text
C:\Users\SSAFY\Desktop\2학기\strategy-arena\.venv\Scripts\python.exe -c "... monkeypatch session.load_frame and call _symbol_returns(Path('synthetic-cache'),'SPY') ..."
```

Observed: `LOADED=bitget_SPYUSDT_1H.csv.gz,bitget_funding_SPYUSDT.csv.gz` and `SPOT_LOAD=False`. Exit `0`; the claimed basis flag is metadata-only in the current implementation.

Network guard:

```text
C:\Users\SSAFY\Desktop\2학기\strategy-arena\.venv\Scripts\python.exe -c "... monkeypatch socket.socket.connect to raise NETWORK_CALL; main(['--stage','fetch']) ..."
```

Result: exit `0`, `MAIN_RC=0`, `fetch: verified wave-1 cache only (131 files)`; no network assertion fired.

## Worktree and cleanup

Initial status was recorded before QA and contained the user-owned `.debug-journal.md` and `debug.log`, the listed Wave-1 edits, and the Wave-2 implementation/output paths. It did not contain `research/wave3/` or `research/scanner/`.

Final status still contains `.debug-journal.md` length 1,433 and `debug.log` length 150; SHA-256 values were captured without printing file contents. No Python process remained after QA. Final status additionally contains unexpected untracked `research/wave3/SPEC.md` and `research/scanner/scan_bitget.py`; these were not removed or reverted.

One early W2g command used the wrong checkout and failed before product import with `ModuleNotFoundError`; the corrected current-checkout invocation above passed. This was classified as harness debris, not a product result.
