# manualQa

결론: **FAIL**. 정상 pytest와 staged CLI smoke는 통과했지만, W2d spike 진입이 실제 spike보다 먼저 활성화되고 W2g가 현물 데이터를 읽지 않아 SPEC의 basis mark-to-market 요구를 충족하지 않습니다. 또한 QA 중 초기 status에 없던 `research/wave3/SPEC.md`와 `research/scanner/scan_bitget.py`가 shared worktree에 나타났으며 삭제·복구하지 않았습니다.

## 실행 조건

- Repository: `C:\Users\SSAFY\Desktop\2학기\.domain-separation-20260715\strategy-arena`
- Venv: `C:\Users\SSAFY\Desktop\2학기\strategy-arena\.venv\Scripts\python.exe`
- Network: 사용하지 않음. fetch는 `socket.connect` fail-closed guard로도 재실행함.
- Attempt resolution: `omo ulw-loop status --json`가 설치된 Windows wrapper에서 syntax error를 반환했고 `.omo/ulw-loop` plan state가 없어 caller evidence directory인 `C:\Users\SSAFY\Desktop\2학기\.domain-separation-20260715\strategy-arena\.omo\evidence\r1-python-library-fix\a1`을 사용함.

## surfaceEvidence

| scenario id | criterion reference | surface | exact invocation | verdict | artifactRefs |
|---|---|---|---|---|---|
| SUR-HELP | SPEC CLI staged run | terminal CLI | `C:\Users\SSAFY\Desktop\2학기\strategy-arena\.venv\Scripts\python.exe -m research.wave2.run_wave2 --help` | PASS | A-TRANSCRIPT |
| SUR-FETCH | SPEC lines 5-7, no network | terminal CLI | `C:\Users\SSAFY\Desktop\2학기\strategy-arena\.venv\Scripts\python.exe -m research.wave2.run_wave2 --stage fetch --force` | PASS | A-TRANSCRIPT, A-MANIFEST |
| SUR-TEST-WAVE2 | requested synthetic pytest | terminal CLI | `C:\Users\SSAFY\Desktop\2학기\strategy-arena\.venv\Scripts\python.exe -m pytest -p no:cacheprovider research/wave2/tests/test_wave2.py` | PASS | A-TRANSCRIPT |
| SUR-TEST-ALL | wave1 reuse regression | terminal CLI | `C:\Users\SSAFY\Desktop\2학기\strategy-arena\.venv\Scripts\python.exe -m pytest -p no:cacheprovider` | PASS | A-TRANSCRIPT |
| SUR-RUN-W2A | SPEC W2a, lines 12 and 21 | terminal CLI | `C:\Users\SSAFY\Desktop\2학기\strategy-arena\.venv\Scripts\python.exe -m research.wave2.run_wave2 --stage run --only W2a` | PASS | A-TRANSCRIPT, A-W2A |
| SUR-GATES-W2A | SPEC line 21 | terminal CLI | `C:\Users\SSAFY\Desktop\2학기\strategy-arena\.venv\Scripts\python.exe -m research.wave2.run_wave2 --stage gates --only W2a` | PASS | A-TRANSCRIPT, A-W2A |
| SUR-REPORT | same staged CLI | terminal CLI | `C:\Users\SSAFY\Desktop\2학기\strategy-arena\.venv\Scripts\python.exe -m research.wave2.run_wave2 --stage report` | PASS | A-TRANSCRIPT, A-REPORT |
| SUR-RUN-W2D | SPEC line 15 | terminal CLI | `C:\Users\SSAFY\Desktop\2학기\strategy-arena\.venv\Scripts\python.exe -m research.wave2.run_wave2 --stage run --only W2d` | PASS | A-TRANSCRIPT, A-W2D |
| SUR-RUN-W2G | SPEC lines 18 and 22 | terminal CLI | `C:\Users\SSAFY\Desktop\2학기\strategy-arena\.venv\Scripts\python.exe -m research.wave2.run_wave2 --stage run --only W2g` then `--stage gates --only W2g` | FAIL | A-TRANSCRIPT, A-W2G, A-SOURCE-SESSION |
| SUR-NETWORK-GUARD | user no-network constraint | auxiliary CLI | `C:\Users\SSAFY\Desktop\2학기\strategy-arena\.venv\Scripts\python.exe -c "monkeypatch socket.socket.connect; run_wave2.main(['--stage','fetch'])"` | PASS | A-TRANSCRIPT, A-MANIFEST |
| SUR-MALFORMED-CLI | parser boundary | terminal CLI | `...run_wave2 --stage invalid`; `...run_wave2 --stage run --only NOPE` | PASS | A-TRANSCRIPT |

## adversarialCases

| scenario id | criterion reference | adversarial class | expected behavior | verdict | artifactRefs |
|---|---|---|---|---|---|
| ADV-MAKER-COST | SPEC line 6 | malformed/edge synthetic market | Two-leg maker cost at 0.02% per leg and zero slippage must improve the flat synthetic result over the inherited taker route. | PASS | A-TRANSCRIPT, A-SOURCE-FUNDING |
| ADV-SPIKE-TIMING | SPEC line 15 and wave1 SPEC line 28 | lookahead / timing | A funding spike at 2026-01-01 16:00Z must not activate a position at 08:00Z; activation belongs on the next daily bar. | FAIL: observed first active at 08:00Z | A-TRANSCRIPT, A-SOURCE-SPIKE |
| ADV-UNTESTED-OOS | SPEC line 21 | empty OOS | No OOS trades and no OOS position must label OOS-dependent gates `UNTESTED_IN_OOS`, not FAIL. | PASS | A-TRANSCRIPT, A-W2A |
| ADV-W2G-BASIS | SPEC line 18 | missing/ignored input | W2g must load mark and spot and calculate the basis mark-to-market contribution; missing spot must not be represented only by a boolean. | FAIL: loader read only Bitget mark and funding; `SPOT_LOAD=False` | A-TRANSCRIPT, A-W2G, A-SOURCE-SESSION |
| ADV-W2G-REPORT | SPEC line 22 | misleading/partial success | W2g report must expose effect statistics and the post-cost/basis return sign while retaining exploratory/no-deployment state. | FAIL: report has only state/verdict/OOS label columns | A-REPORT, A-SOURCE-REPORT |
| ADV-MALFORMED-CLI | CLI boundary | malformed input | Invalid stage and candidate must exit non-zero and show parser errors. | PASS: exits 2 and 2 | A-TRANSCRIPT |
| ADV-NETWORK-GUARD | no-network constraint | external side effect | Any attempted socket connection must fail the scenario; fetch must still complete from local cache. | PASS: guard did not fire; exit 0 | A-TRANSCRIPT, A-MANIFEST |
| ADV-FLAKY-REPEAT | test reliability | flaky tests | Repeating focused pytest twice must produce the same 3/3 result. | PASS: both exit 0, 3 passed | A-TRANSCRIPT |
| ADV-DIRTY-WORKTREE | user preservation | dirty worktree | QA must preserve `.debug-journal.md` and `debug.log` and must not introduce out-of-scope files. | FAIL: protected files preserved, but new `research/wave3/` and `research/scanner/` appeared during shared QA | A-TRANSCRIPT, A-UNEXPECTED-WAVE3, A-UNEXPECTED-SCANNER |
| ADV-HUNG-LONG | CLI runtime | long-running command | W2d's longer top-10 run may take minutes but must finish with exit 0 and leave no Python process. | PASS: exit 0; final process check empty | A-TRANSCRIPT |
| ADV-MISLEADING-OUTPUT | CLI failure signaling | misleading success output | Parser failures must not emit a fetch/run success marker or exit 0. | PASS: both invalid invocations exited 2 with argparse errors | A-TRANSCRIPT |
| ADV-PROMPT-INJECTION | CLI surface | prompt injection | Not applicable: this product surface accepts only fixed argparse enums and no free-form prompt text. | not_applicable | A-SOURCE-RUNNER |
| ADV-CANCEL-RESUME | CLI surface | cancel/resume | Not applicable: the staged research CLI has no cancel, resume, or session-state control surface. | not_applicable | A-SOURCE-RUNNER |
| ADV-STALE-STATE | CLI surface | stale state | Not applicable: no `.omx`/session state is read by the Wave-2 runner; result JSON is an input artifact, not a resumable state machine. | not_applicable | A-SOURCE-RUNNER |
| ADV-REPEATED-INTERRUPTION | CLI surface | repeated interruption | Not applicable: no interactive continuation protocol exists; W2d was allowed to complete and was checked for orphan processes. | not_applicable | A-TRANSCRIPT |

## artifactRefs

| id | kind | description | path |
|---|---|---|---|
| A-TRANSCRIPT | terminal-transcript | Exact venv pytest, CLI, malformed, network, timing, and cleanup observations. | `C:\Users\SSAFY\Desktop\2학기\.domain-separation-20260715\strategy-arena\.omo\evidence\r1-python-library-fix\a1\wave2-cli-transcript.md` |
| A-MANIFEST | pipeline-output | Current fetch manifest: local wave1 cache, no network, 131 files, force ignored. | `C:\Users\SSAFY\Desktop\2학기\.domain-separation-20260715\strategy-arena\research\wave2\cache_manifest.json` |
| A-W2A | pipeline-output | Current W2a result and gates, including OOS label. | `C:\Users\SSAFY\Desktop\2학기\.domain-separation-20260715\strategy-arena\research\wave2\results\W2a.json` |
| A-W2D | pipeline-output | Current W2d result with top-10 symbols and spike config. | `C:\Users\SSAFY\Desktop\2학기\.domain-separation-20260715\strategy-arena\research\wave2\results\W2d.json` |
| A-W2G | pipeline-output | Current W2g result/gates with exploratory and basis flags. | `C:\Users\SSAFY\Desktop\2학기\.domain-separation-20260715\strategy-arena\research\wave2\results\W2g.json` |
| A-REPORT | pipeline-output | Current generated report showing missing candidates and no W2g effect-stat columns. | `C:\Users\SSAFY\Desktop\2학기\.domain-separation-20260715\strategy-arena\research\wave2\report\wave2_report.md` |
| A-SOURCE-FUNDING | source | Maker cost path and wave1 portfolio reuse, lines 50-64. | `C:\Users\SSAFY\Desktop\2학기\.domain-separation-20260715\strategy-arena\research\wave2\funding.py` |
| A-SOURCE-SPIKE | source | Daily max alignment and one-row shift that produced pre-spike activation, lines 24-38. | `C:\Users\SSAFY\Desktop\2학기\.domain-separation-20260715\strategy-arena\research\wave2\spike.py` |
| A-SOURCE-SESSION | source | W2g reads mark/funding and only checks spot path existence, lines 16-30. | `C:\Users\SSAFY\Desktop\2학기\.domain-separation-20260715\strategy-arena\research\wave2\session.py` |
| A-SOURCE-REPORT | source | Report construction omits effect statistics and return sign, lines 19-40. | `C:\Users\SSAFY\Desktop\2학기\.domain-separation-20260715\strategy-arena\research\wave2\reporting.py` |
| A-SOURCE-RUNNER | source | Fixed staged enum CLI and no state/prompt surface, lines 103-124. | `C:\Users\SSAFY\Desktop\2학기\.domain-separation-20260715\strategy-arena\research\wave2\run_wave2.py` |
| A-SPEC | specification | Frozen Wave-2 requirements used for verdicts, lines 5-23. | `C:\Users\SSAFY\Desktop\2학기\.domain-separation-20260715\strategy-arena\research\wave2\SPEC.md` |
| A-UNEXPECTED-WAVE3 | unexpected-worktree | New out-of-scope file observed at final status; preserved. | `C:\Users\SSAFY\Desktop\2학기\.domain-separation-20260715\strategy-arena\research\wave3\SPEC.md` |
| A-UNEXPECTED-SCANNER | unexpected-worktree | New out-of-scope file observed at final status; preserved. | `C:\Users\SSAFY\Desktop\2학기\.domain-separation-20260715\strategy-arena\research\scanner\scan_bitget.py` |

## Remediation

1. Fix `research/wave2/spike.py:24-38` so an intraday spike cannot be forward-filled into earlier 8h rows; add a failing synthetic assertion that 08:00 remains flat when the spike occurs at 16:00, then require activation at the next daily bar.
2. Fix `research/wave2/session.py:16-30` to load the actual spot series and calculate basis mark-to-market. Do not claim `basis_mark_to_market_included` from a boolean flag when the spot series was not used; fail closed for missing spot data.
3. Extend `research/wave2/reporting.py:19-40` to report W2g effect statistics and post-cost/basis return sign while preserving exploratory/no-deployment labeling.
4. Resolve ownership of the unexpected `research/wave3/` and `research/scanner/` additions and the concurrent `run_wave2.py` drift before any merge. QA made no product-file fix and did not delete these files.

## Cleanup receipt

- No temporary product/test files were created by QA.
- The one synthetic pytest fixture self-deleted in its `finally` block.
- No Python process remained at final check.
- `.debug-journal.md` and `debug.log` remained present at their initial observed lengths 1,433 and 150 bytes; their contents were not printed.
