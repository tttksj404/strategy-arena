# RaceLens grade routing candidate evidence

## Question
- Can the user's 특선/우수/선발 idea improve prediction accuracy, instead of only filtering confidence?

## Experiment
- Added `scripts/experiment_kcycle_grade_routing.py`.
- Historical snapshots do not include exact grade labels, so this first pass uses race number proxy:
  - 1-5R: 선발
  - 6-10R: 우수
  - 11R+: 특선
- Compared baseline `board_min` against grade-routed algorithms over 13,900 full-board snapshots.
- Splits:
  - train: 2018-2023
  - val: 2024-2025
  - test: 2026

## Result
- Grade-routed exact 삼쌍 replacement selected by exact does not pass validation:
  - selected_policy: baseline
  - deployable: false
- A top1-focused grade routing candidate did emerge:
  - selected_top1_policy: `grade_route_train_top1_best`
  - route:
    - 선발: `xdom_clonal_selection_amplify`
    - 우수: `first_pair_chain`
    - 특선: `first_pair_chain`
  - test races: 1,384
  - test exact: 0.1763 vs board 0.1720, lift +0.4335pp
  - test top1: 0.6358 vs board 0.6344, lift +0.1445pp
  - val top1 lift also positive, but val exact lift negative.

## Interpretation
- Do not replace the full 삼쌍 engine with grade routing yet.
- This is a valid candidate for a separate 1착 direction/leader signal lane.
- Next promotion gate should require true grade labels when available, not only race number proxy.

## Verification
- `python scripts/experiment_kcycle_grade_routing.py`: completed and wrote JSON/MD results.
- `pytest tests/test_grade_routing_experiment.py tests/test_trifecta_rule_search.py`: 7 passed.
- Full suite: 126 passed.
- `py_compile` passed for changed Python files.
- `git diff --check` passed.
