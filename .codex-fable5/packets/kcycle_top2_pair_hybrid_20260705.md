# KCYCLE Top2 Pair Hybrid Evidence Packet

## Objective

Find and validate another method to improve KCYCLE prediction accuracy without claiming unverified 50%+ trifecta breakthroughs.

## Data

- Snapshot corpus: `data/kcycle_trifecta_snapshots.jsonl`
- Race count: 13,900
- Train years: 2018-2023
- Holdout years: 2024-2026

## Experiments

1. `scripts/experiment_kcycle_surprise_top2.py`
   - Gate-only surprise rescue was rejected.
   - Snapshot proxy best lift was about +0.029pp only, not deployable.

2. `scripts/experiment_kcycle_top2_pair.py`
   - Baseline: lowest-odds trifecta combo top2.
   - Tested slot mass, ordered pair mass, unordered pair mass, entropy/gap routers, and feature-threshold hybrids.
   - Best nonnegative-holdout-year candidate:
     - `hybrid_slot_mass_top2__slot_top1_le_0.799966`
     - Holdout top2 slot hit: 0.6794215511
     - Baseline holdout top2 slot hit: 0.6694394676
     - Holdout lift: +0.998208344pp
     - Worst holdout-year lift: +0.519480519pp
     - Holdout unordered pair lift: +0.307141029pp
     - Holdout ordered pair lift: -0.179165600pp

## Product Application

- Added `_market_top2_hybrid_signal()` to `engine.py`.
- Attached the signal under `trifecta_axis_signal["top2_hybrid"]`.
- Applied it only to mobile `QNL` top2/pair display.
- Did not overwrite exact trifecta order because ordered-pair lift is negative.

## Verification

- `python -m pytest tests/test_kcycle_top2_pair_experiment.py -q`
  - 4 passed
- `python -m pytest tests/test_live_decision.py -k 'top2_hybrid or mobile_picks_use_top2' -q`
  - 2 passed
- `python -m pytest tests/test_kcycle_top2_pair_experiment.py tests/test_live_decision.py tests/test_trifecta_snapshot_audit.py -q`
  - 55 passed
- `python -m pytest tests/test_surprise_top2_experiment.py tests/test_prediction_feedback.py tests/test_trifecta_rule_search.py tests/test_kcycle_top2_pair_experiment.py tests/test_live_decision.py tests/test_trifecta_snapshot_audit.py -q`
  - 73 passed
- `python -m compileall engine.py scripts/experiment_kcycle_top2_pair.py`
  - exit 0

## Audit Questions

- Is applying the candidate only to QNL/top2 display the correct safety boundary given ordered-pair lift is negative?
- Should exact trifecta generation consume this signal only as candidate-set widening, not as order replacement?
- Are the train/holdout gates strict enough, or should an untouched newer live-log split be required before deployment beyond supplemental display?
