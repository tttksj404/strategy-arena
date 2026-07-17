# Time-safe Accuracy Uplift Implementation Plan

> For agentic workers: required sub-skill: use task-by-task execution. Steps use checkbox syntax for tracking.

**Goal:** Build a reproducible, leakage-safe tournament for diverse KRA/KCYCLE prediction candidates, and promote nothing until an independent chronological holdout proves a lane-specific gain.

**Architecture:** The tournament is offline-only. It wraps existing model-search outputs in an immutable candidate manifest, evaluates each lane with walk-forward splits, and emits a report that separates selection metrics from locked-test metrics. Production keeps its present market-anchor and V1 ensemble unless a later, separately reviewed promotion passes the stored gates.

**Tech Stack:** Python 3, existing NumPy/scikit-learn research stack, pytest, JSON reports.

---

### Task 1: Freeze lane contracts in a manifest

**Files:**

- Create: tools/prediction_candidate_tournament.py
- Create: tests/test_prediction_candidate_tournament.py

- [x] Step 1: Write a failing test that a candidate selected using its final holdout is rejected with selection_scope_must_be_train_val.
- [x] Step 2: Run python3 -m pytest tests/test_prediction_candidate_tournament.py -q and confirm failure because the module is absent.
- [x] Step 3: Implement immutable candidate and lane contracts. Reject test-selected, post-start, stale, and cross-lane candidates before calculating metrics.
- [x] Step 4: Re-run the focused test and confirm pass.

### Task 2: Add chronological evaluation and selective-calibration accounting

**Files:**

- Modify: tools/prediction_candidate_tournament.py
- Modify: tests/test_prediction_candidate_tournament.py

- [x] Step 1: Write failing tests that selection dates end before locked-test dates and that abstention reports coverage while preserving a baseline metric.
- [x] Step 2: Run the focused tests and confirm failure because chronological evaluation and coverage are absent.
- [x] Step 3: Implement the minimum evaluator. Use dates as the only split boundary; select candidates from train/validation summaries and score the frozen choice once on later rows. Report baseline, candidate, lift, calibration loss, coverage, and rejection reason.
- [x] Step 4: Re-run focused tests and confirm pass.

### Task 3: Wire existing diverse candidates without production mutation

**Files:**

- Modify: tools/prediction_candidate_tournament.py
- Create: tools/prediction_candidate_manifest.py
- Modify: tests/test_prediction_candidate_tournament.py
- Create: tests/test_prediction_candidate_manifest.py

- [x] Step 1: Write a failing test that the known test-selected KCYCLE mutation gen2_mut_436 is rejected.
- [x] Step 2: Run the focused test and confirm failure.
- [x] Step 3: Add a six-candidate manifest for retained KCYCLE V1, test-selected mutations, audit-pending and timed lanes, and two KRA evidence reports. Mark unavailable evidence and test-selected mutations rather than silently scoring them.
- [x] Step 4: Run python3 tools/prediction_candidate_manifest.py --report /tmp/prediction_candidate_manifest.json and confirm every candidate is retained, rejected, review-required, or data-blocked while production artifacts remain untouched.

### Task 4: Verify and commit the research guardrail

**Files:**

- Modify: docs/superpowers/specs/2026-07-17-accuracy-uplift-design.md
- Modify: docs/superpowers/plans/2026-07-17-time-safe-accuracy-uplift.md

- [ ] Step 1: Run python3 -m pytest tests/test_prediction_candidate_tournament.py tests/test_prediction_candidate_manifest.py tests/test_validate_kra_autoresearch.py tests/test_kra_fresh_holdout_guard.py tests/test_kcycle_ensemble.py tests/test_market_timing_policy.py -q.
- [ ] Step 2: Run python3 -m py_compile tools/prediction_candidate_tournament.py tools/prediction_candidate_manifest.py and git diff --check.
- [ ] Step 3: Commit only the tournament, tests, and documents with message feat: add time-safe prediction candidate tournament.
