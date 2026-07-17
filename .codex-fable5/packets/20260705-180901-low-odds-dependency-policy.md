# RaceLens low odds dependency policy evidence

## User intent
- Keep the algorithm aligned with the explored low-odds-dependency result.
- Do not let odds-only signals dominate or freely override the model.

## Exploration evidence checked
- `docs/kcycle_market_blend_experiment_results.md`
  - robust result: `blend_w0.30`
  - rule: `(1-w)*model+w*market, w=0.30`
  - test top1: `0.6217`
  - market flip rate: `0.050`
- `data/kcycle_market_blend_experiment_results.json`
  - `robust_test.name`: `blend_w0.30`
  - `robust_test.market_flip_rate`: `0.050102249488752554`
  - `chosen` was `blend_w0.70`, but robust policy intentionally rejects high flip rate.

## Code policy after fix
- Keep capped low-dependency blend:
  - early: market weight 0.05
  - mid: market weight 0.15
  - late: market weight 0.30
  - post-start: market weight 0.0
- Remove odds-only probability promotion:
  - strong win-market signal no longer raises `pwin` to `expected_top1`
  - trifecta-axis signal no longer overrides the model leader when first pick conflicts
- Conflicting market-only signals are exposed as blocked/advisory:
  - `blocked_by_order_conflict: true`

## Fresh verification
- `.venv/bin/python -m pytest tests/test_live_decision.py tests/test_app_data_layer.py -q`
  - Result: `61 passed in 0.56s`
- `git diff --check -- engine.py tests/test_live_decision.py`
  - Result: exit 0
- Restart:
  - backend `37037`
  - proxy `37124`
  - tunnel `37334`

## Runtime evidence
- `GET /api/live-decision?sport=keirin&meet=광명&date=2026-07-05&race_no=14`
  - `message`: `저의존 배당 블렌드 반영 (시장 0.30 + 모델 0.70) ...`
  - `market_timing.policy`: `low_odds_dependency_blend_w0.30`
  - `market_timing.policy_flip_rate`: `0.050102249488752554`
  - `trifecta_axis_signal.applied`: `false`
  - `trifecta_axis_signal.blocked_by_order_conflict`: `true`
  - `trifecta_global_signal.applied`: `false`
  - `trifecta_global_signal.blocked_by_order_conflict`: `true`
- Browser smoke on `http://192.168.0.5:4173/`
  - live page loads current race
  - live odds visible as input data
  - RaceLens analysis still shows model conclusion and live odds separately

## Residual audit question
- Fable5 should verify whether keeping up to 30% market blend is the intended definition of "low odds dependency" or whether the user wants `model only` with odds purely informational.
