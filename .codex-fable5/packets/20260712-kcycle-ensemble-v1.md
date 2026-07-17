# Kcycle Ensemble V1 Evidence Packet

## Scope
- Round4 server-side productization only.
- Mobile UI unchanged.
- No git commit.

## Files
- `static/models/kcycle_trifecta_ensemble_v1.json`
- `engine.py`
- `scripts/kcycle_eval_common.py`
- `tests/test_kcycle_ensemble.py`
- `runs/prediction_uplift_progress.md`

## Claims
- Ensemble artifact freezes 20 val-selected deployable formulas from `data/kcycle_global_breakthrough_results.json`.
- `kcycle_ensemble_trifecta_rank(board)` rank-averages the frozen formulas with campaign feature definitions.
- `kcycle_trifecta_confidence_tier(board)` reports `T0_base`, `T1_strong`, or `T2_top16` with historical exact rates only.
- `/api/live-decision` adds `trifecta_ensemble` when a trifecta board exists.
- Board-missing path keeps existing Harville picks.

## Verification
- `python -m pytest tests/test_kcycle_ensemble.py -q`
  - `5 passed`
- `python -m pytest tests -q`
  - `274 passed, 1 warning, 4 subtests passed`
- `python -m py_compile engine.py scripts/kcycle_eval_common.py tests/test_kcycle_ensemble.py`
  - exit 0
- Flask port bind smoke:
  - `PORT=5073 ... python app.py`
  - blocked by sandbox: `Operation not permitted`
- Flask test-client fallback smoke:
  - `/api/live-decision` without date returned `400 hold False`
- Engine live-decision ensemble smoke:
  - patched full board returned `ensemble_v1 4-6-7 5 T0_base`

## Audit Notes
- User requested no completion claim before Fable5 audit.
- Remaining gate: external Fable5 PASS/NEEDS_FIX/FAIL review.
