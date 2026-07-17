# RaceLens grade-policy validation gate evidence

## User intent
- Do not directly apply 특선/우수/선발 heuristics.
- Compare existing baseline against grade-aware thresholds on outcome-linked history.
- Only switch runtime behavior when the grade-aware policy shows higher predictive precision.

## Implemented gate
- `scripts/update_prediction_feedback.py` now evaluates:
  - baseline confidence/final-candidate policy
  - grade-aware confidence/final-candidate policy
- Runtime `engine._keirin_grade_context()` uses grade-aware thresholds only when:
  - `participant_learning_priors.json.grade_policy_validation.deployable == true`
  - `selected_policy == "grade_context"`
- Operators can force QA with `KEIRIN_GRADE_POLICY_MODE=force`, but default runtime is `validated`.

## Current measured result
- `python scripts/update_prediction_feedback.py`
- outcome_rows: 20008
- matched_races: 6
- keirin grade-policy comparable races: 5
- baseline: recommended 4 / top1 hits 2 / precision 0.5
- grade_context: recommended 4 / top1 hits 2 / precision 0.5
- selected_policy: baseline
- status: insufficient_matched_races
- deployable: false

## Verification
- `pytest tests/test_confidence_display.py tests/test_prediction_feedback.py`: 18 passed
- `pytest tests/test_live_decision.py tests/test_app_data_layer.py tests/test_confidence_display.py tests/test_prediction_feedback.py`: 79 passed
- `python -m py_compile engine.py scripts/update_prediction_feedback.py tests/test_confidence_display.py tests/test_prediction_feedback.py`: pass
- `git diff --check -- engine.py scripts/update_prediction_feedback.py tests/test_confidence_display.py tests/test_prediction_feedback.py`: pass
- Local preview restarted:
  - backend http://127.0.0.1:8010
  - proxy http://127.0.0.1:4173
- API smoke:
  - `/api/live-decision?...race_no=15`
  - `top_conf.grade_context == ""`
  - first pick probability around 0.12, matching first row pwin
- Browser smoke:
  - opened `http://127.0.0.1:4173/`
  - clicked `모델 신호 보기`
  - prediction screen rendered
  - `1착 후보 신뢰도` displayed 11%, no stale 83% overconfidence reproduced
  - no browser console errors

## Residual audit item
- Fable5 external audit still pending by global rule for betting/prediction changes.
