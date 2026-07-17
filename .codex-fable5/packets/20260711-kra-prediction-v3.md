# Fable5 audit packet — KRA prediction v3

Requested verdict: `PASS`, `NEEDS_FIX`, or `FAIL`.

## Claim under review

The production KRA ranking path is safer and more accurate after separating pre-race intrinsic inference from complete-board live-odds inference. This is an accuracy claim only, never a positive-return claim.

## Root cause

The prior artifact included win/place odds and implied-odds features during training. The same artifact received zeros when odds were absent at serving time, creating a train/serve distribution shift. The old no-odds win pick scored 19.34% top-1 on the locked 2026 holdout.

## Data and selection protocol

- Source: `/Users/tttksj/kra/data/kra.db`, 63,205 valid starters, 6,115 races, 2024-01-05 through 2026-06-21.
- Development: before 2025-01-01.
- Candidate and live-weight selection: 2025 only.
- Locked holdout: 1,189 races from 2026-01-01.
- Candidates: three HGB depth/rate/regularization configurations.
- Live weights: 0.00 through 1.00 in 0.05 increments, selected on 2025 only.
- Odds excluded from the pre-race feature set.
- A live phase requires complete positive win odds for every starter and explicit `odds_snapshot_fresh=true` provenance, preventing settled-result odds from entering an alleged pre-race prediction.
- Horse participant-learning overlay removed because its own OOS record says `deployable=false`, with only 14 matched races and zero lift.

## Locked holdout

| policy | top-1 | top-3 | race log-loss | coverage |
|---|---:|---:|---:|---:|
| old odds-trained model served with zero odds | 19.34% | 46.85% | 2.2807 | 100% |
| v3 intrinsic pre-race | 24.31% | 56.94% | 2.0637 | 100% |
| v3 intrinsic selective | 32.08% | not claimed | not claimed | 31.20% |
| de-vigged live market | 38.10% | 70.98% | 1.7702 | 100% |
| v3 live selected policy | 38.10% | 70.98% | 1.7702 | 100% |
| v3 live selective | 51.88% | not claimed | not claimed | 29.02% |

Pre-race top-1 gain over the actual previous no-odds win path is +4.96 percentage points. The 2025 selector chose HGB depth 3 and live market weight 1.0; v3 therefore does not pretend the intrinsic model beats the KRA live market.

## Files

- `engine.py`
- `static/models/kra_model.joblib`
- `tools/kra_dual_phase_experiment.py`
- `tests/test_kra_prediction_phase.py`
- `runs/kra_dual_phase_results.json`
- `docs/kra_prediction_v3.md`

## Verification evidence

- Red test: `python3 -m unittest tests.test_kra_prediction_phase -v` initially failed because `top` returned the place leader instead of the win leader.
- Targeted green: 12 tests passed before the final test addition.
- Full backend: `python3 -m unittest discover -s tests -v` reported 195 tests, all passed in the final checkpoint.
- Mobile TypeScript: `npm run typecheck` exit 0.
- Mobile security surface: `npm run lint:security` reported `security surface check passed`.
- Runtime smoke observed both `pre_race` and `live_odds` from `data/demo_kra_race.json` with artifact kind `kra_dual_phase_v3`.

## Audit questions

1. Is the 2025 selection and untouched 2026 holdout protocol sufficient to support the stated accuracy claim?
2. Does any feature or prior leak future outcome information into pre-race inference?
3. Is market-weight 1.0 routed only when the odds board is complete and positive?
4. Are the confidence thresholds represented with coverage and without claiming universal 32% or 52% accuracy?
5. Does any code path reintroduce the non-deployable participant-learning overlay for horse predictions?
6. Are there API compatibility, probability normalization, stale-artifact, or UI-semantics regressions?
7. Is the boundary against ROI or profit claims sufficiently explicit?
