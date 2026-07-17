# Evidence Packet: KCYCLE Auto Search Continuation

## Claim under audit
- The prediction search no longer waits for user prompting to try fresh global rerank candidates.
- The loop consumes new seed ranges via `data/kcycle_global_search_state.json`, merges prior candidates, and emits an engine-readable champion JSON.
- The engine dynamically loads the first deployable global rerank champion from `data/kcycle_global_breakthrough_results.json` when feature stats are present, falling back to `gen2_mut_579` if the file is missing or invalid.
- No claim is made that a +10pp breakthrough or 50% full-race 삼쌍 exact has been found.

## Changed surfaces
- `scripts/search_kcycle_global_breakthrough.py`: stateful auto seeds, merge existing candidates, feature stats by top_k, state writeback.
- `scripts/run_prediction_search_loop.sh`: passes state json, auto seed count, merge flag to global breakthrough search.
- `engine.py`: loads deployable champion from global breakthrough JSON and applies dynamic top_k/weights/mu/sigma.
- `tests/test_live_decision.py`: adds regression proving dynamic champion JSON is loaded and reflected in signal output.

## Fresh verification evidence
- Syntax: `bash -n scripts/run_prediction_search_loop.sh scripts/racelens_prediction_watchdog.sh` passed.
- Syntax: `.venv/bin/python -m py_compile engine.py scripts/search_kcycle_global_breakthrough.py` passed.
- Targeted dynamic champion tests: 4 passed.
- Broader relevant tests: `55 passed in 0.35s`; snapshot lines stayed 14040 before/after.
- Corpus audit: ok true, duplicate_keys 0, hash_mismatch 0, board_count_mismatch 0, stored_signal_mismatch 0, actual_joined_live_like_risk 0.
- Manual global run: cycle 4 consumed seeds `[20260709, 20260710]`.
- Watchdog restart runtime run: cycle 5 consumed seeds `[20260711, 20260712]`, next_seed became `20260713`.
- Watchdog is running with `SEARCH_LOOP_INTERVAL_SEC => 0`; stderr was cleared after restart and remained size 0 after 10s.

## Current measured result
- `deployable_count`: 227.
- `breakthrough_10pp_count`: 0.
- Best champion: `gen2_mut_1299`, top_k 20, seed 20260710.
- Best test_exact: 0.18063583970069885.
- Lift vs current axis: +2.001232224898758pp.
- Lift vs same-slice board: +0.8670521781623697pp.

## Audit focus requested
1. Check whether any completion claim overstates the result.
2. Check whether stateful seed search can get stuck or repeatedly re-evaluate the same seeds.
3. Check whether engine dynamic champion loading could silently apply invalid or overfit data.
4. Check whether watchdog runtime evidence proves autonomous continuation rather than manual-only search.
