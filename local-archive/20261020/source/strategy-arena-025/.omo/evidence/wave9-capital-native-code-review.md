# Wave-9 capital-native code quality review

- Review date: 2026-07-21 (Asia/Seoul)
- Reviewed snapshot: `run_wave9.py` SHA-256 `c731922db602d35c157f962e6effed05c45c9a389cc57b32f8c696280ca36a3c`
- Scope: the whole untracked `research/wave9_capital_native/` tree, its imported Wave-1 cost and Wave-3 manifest-verification dependencies, and the final A/B/C result/report artifacts
- ULW routing: `omo ulw-loop status --json` did not return a plan (`The syntax of the command is incorrect.`), so the required fallback evidence path is used.

## Final decision

- `codeQualityStatus`: **WATCH**
- `recommendation`: **APPROVE**
- `blockers`: none

The final snapshot is safe for the claimed outcome: all 21 candidates are ineligible, the missing point-in-time universe is a hard FAIL gate for every round, and the adaptively reused OOS is an additional hard FAIL gate for B/C. This approval is for fail-closed exploratory evidence only, not for promotion or live capital.

## Findings

### CRITICAL

None.

### HIGH

None. The two scientifically high-risk limitations are now enforced rather than merely documented:

- The end-snapshot/survivorship-biased universe is disclosed in `SPEC.md:22-25` and produces `point_in_time_universe=FAIL` in `run_wave9.py:323-329,362` for every candidate.
- Wave-9B/C reuse an already observed OOS window (`ADDENDUM_B.md:24-26`, `ADDENDUM_C.md:21-23`) and produce `oos_independence=FAIL`; only A has that gate PASS.

### MEDIUM

1. **The four-block positivity gate uses an arithmetic sum, not the block's compounded return.** `run_wave9.py:324-325` declares a block positive when `block.sum() > 0`. For daily portfolio returns, the exact block return is `prod(1+r)-1`; for example, `+100%, -50%` has a positive arithmetic sum but zero compounded return. This can change `oos_block_robustness` near the boundary. The permanent point-in-time FAIL prevents a false promotion in this snapshot, but the metric should be corrected before this gate is reusable.

2. **Numeric validation is incomplete and overflow is detected late.** `_finite` is unused (`run_wave9.py:93-96`), close/volume conversion can silently turn corrupt numeric text or infinity into missing observations (`run_wave9.py:103-155`), and price PnL/equity are not checked immediately for finiteness (`run_wave9.py:218-224`). An adversarial `1e308` bootstrap returns `NaN`; `allow_nan=False` then prevents JSON emission at `run_wave9.py:449`, so this is fail-closed for full artifacts, but `--only` can still return non-finite metrics and the error occurs after expensive work. Current frozen inputs were independently checked: 344,882 valid close observations were finite/positive, their volumes were finite/nonnegative, and the aligned funding matrix was finite.

3. **Exact cross-machine reproducibility is not pinned.** The payload hashes the runner, specs, manifest, and two direct dependencies (`run_wave9.py:403-449`), which is good, but it does not record Python/NumPy/pandas versions. Root `requirements.txt:1-4` leaves packages unpinned and omits pandas. The verified environment was Python 3.12.10, NumPy 2.5.1, pandas 3.0.3. Exact same-workspace reproduction passed for N9a, but another environment is not provenance-equivalent by construction.

4. **Result and report replacement is not atomic.** The report and JSON are overwritten directly at `run_wave9.py:400,449`. A killed process can leave a partial file, and a failed regeneration leaves an older artifact at the canonical path. Hash fields made stale artifacts detectable during this review, but atomic temp-file plus `os.replace` would make the evidence boundary safer.

5. **The runner carries avoidable performance and dead-code cost.** `close.shift(1)` is recomputed inside every daily iteration (`run_wave9.py:218-222`), `_result` recomputes identical signals for normal and stress runs (`run_wave9.py:356-361`), and the full-frame `realized` rolling calculation is unused (`run_wave9.py:163`). `CANDIDATE_MAP` and `_finite` are also unused (`run_wave9.py:64,93`). These are remove-ai-slops violations and materially lengthen the 21-candidate regeneration.

6. **The programming-skill quality gate reports seven maintainability violations.** The checker found raw `dict` returns (`run_wave9.py:207,356`), `object` annotations (`run_wave9.py:323,344`), a 426-pure-LOC module over the 250-LOC limit, and an unannotated pandas opt-out in `tests/test_wave9.py:1`. Typed result/gate models and a split between data loading, simulation, statistics, and artifact writing would reduce schema drift risk.

7. **One test mirrors implementation constants, and critical integration invariants remain outside the committed suite.** `test_registered_rosters_are_distinct_and_frozen` (`tests/test_wave9.py:17-20`) duplicates A/B IDs from production and does not verify Round C or derive the roster from the specs. Under the remove-ai-slops perspective this is a constant-mirroring test with limited regression value. The seven-test suite also lacks permanent no-lookahead perturbation, CLI, hash/TOCTOU, point-in-time/OOS-independence, compounded-block, and final-artifact provenance tests. The other tests are relevant behavioral tests; no deletion-only or tautological removal test was found.

### LOW

1. **The Markdown report does not surface the point-in-time-universe gate explicitly.** The JSON includes the gate and every report says `eligible: []`, but `run_wave9.py:385-390` prints only the OOS-independence summary. In Round A this can make the report look independent unless the reader also opens the spec/JSON. Add a point-in-time-universe line to make the fail-closed reason self-contained.

## Independent verification

- Final artifact/source binding: PASS for all A/B/C JSON and Markdown reports.
  - Manifest SHA-256: `2d1aa4772739cc0bab209a27a5684900db802c55c27c63a4a7979adf26e4c0e0`
  - Result SHA-256: A `45d9bcae3d4bc55af070dc15f86a56d2ca9774ef8a886bed39988ff41c482ae4`; B `631dc6b4b691d4f9be8f47e6260be8345f17886e25b2fc70b7290b01aec28ddb`; C `53d6539e22708ea0d4d68af3c7d0311e8e8c9b2c9633fb9d4bf67af7b4762508`
  - Report SHA-256: A `292efedcfe96238aa6c36fb849e4a3efd02a2d883c1b28ccaa2dfbaa673d37ca`; B `6e72ee5fe46b999b206e1b242853f940b00eca33af12fa557d9b06431f575ce0`; C `f7c3a0ec7d8fbb729b69b7f440f0dfd6c0cb9df28c9aa6b66969c4ebfc740286`
  - Every JSON parsed with non-finite constants rejected, contained the current runner/spec/dependency/manifest hashes, had a complete unique roster, and re-derived to `eligible_candidates=[]`.
- Gate re-derivation: PASS for 10 A, 6 B, and 5 C candidates. Every stored gate status that can be derived from saved metrics matched its threshold. Every candidate contained exactly one historical-MDD gate, and its status matched `mdd <= 0.25`.
- Same-snapshot reproduction: PASS. A fresh read-only `run(only="N9a")` result matched the stored N9a object exactly and remained ineligible.
- Direct feature no-lookahead perturbation: PASS. Changing close and volume from a cutoff onward left all positions through the cutoff byte-equivalent and changed later positions.
- CLI: PASS. Direct-file `--help` works; an unknown candidate exits 2 before running.
- Tests: `7 passed in 1.34s`; `compileall` for `research/wave9_capital_native` passed.
- Overflow boundary: the bootstrap can produce NaN for extreme synthetic returns, but strict JSON serialization rejects it; no malformed final artifact was produced.

## Skill-perspective check

- `omo:remove-ai-slops`: **ran**. Violations remain: unused production calculations/symbols and the A/B constant-mirroring roster test. No deletion-only, removal-only, tautological, or clearly useless behavioral test was found after the final fixes.
- `omo:programming`: **ran**, including Python and data-processing guidance and the supplied no-excuse checker. The final diff violates its raw-dict, broad-`object`, module-size, and pandas-opt-out rules. These are MEDIUM maintainability findings, not correctness blockers for the present fail-closed result.

