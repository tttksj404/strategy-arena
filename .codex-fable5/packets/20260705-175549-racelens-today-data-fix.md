# RaceLens today-data fix evidence packet

## Scope
- Fix wrong same-day RaceLens keirin data display on `http://192.168.0.5:4173/`.
- Prevent KCYCLE result popup from injecting previous-round or premature settled data into today's live/pre-race card.
- Prevent browser cache from keeping stale preview bundles.
- Correct same-day market source labeling from historical to live.

## Changed areas
- `app.py`: default race count and current/upcoming race selection.
- `engine.py`: same-day KCYCLE result-outcome acceptance gate.
- `mobile/src/services/raceApi.ts`: KST date-only market source classification.
- `mobile/scripts/preview-proxy-server.cjs`: `cache-control: no-store` for static preview files.
- `mobile/App.tsx`, `mobile/src/components/RaceSelector.tsx`, `mobile/src/screens/HomeScreen.tsx`, `mobile/src/services/raceSchedule.ts`: dynamic race count/default race wiring.
- `tests/test_app_data_layer.py`, `tests/test_live_decision.py`: regression coverage.

## Fresh verification
- `.venv/bin/python -m pytest tests/test_live_decision.py tests/test_app_data_layer.py -q`
  - Result: `61 passed in 0.66s`
- `npm run typecheck --prefix mobile`
  - Result: exit 0, `tsc --noEmit`
- `npm run build --prefix mobile`
  - Result: exit 0, generated `_expo/static/js/web/AppEntry-164ee3af4fa71e2e7a3627a91d158183.js`
- `git diff --check -- ...`
  - Result: exit 0, no whitespace/patch errors
- `RACELENS_PREVIEW_USE_LAUNCHD=1 mobile/scripts/live-preview-control.sh restart`
  - Result: backend `20801`, proxy `20893`, tunnel `20987`

## Runtime evidence
- `curl -sI http://127.0.0.1:4173/`
  - `cache-control: no-store`
- `curl -sI http://127.0.0.1:4173/_expo/static/js/web/AppEntry-164ee3af4fa71e2e7a3627a91d158183.js`
  - `cache-control: no-store`
- `curl /api/live-decision?sport=keirin&meet=광명&date=2026-07-05&race_no=14`
  - `status: odds_live`
  - `actual_result: null`
  - `market_used: true`
  - rows by bno:
    - `1 김주한`
    - `2 권혁진`
    - `3 박용범`
    - `4 김정우`
    - `5 김민준`
    - `6 김태완`
    - `7 유태복`
- Playwright browser smoke on `http://192.168.0.5:4173/`
  - `광명 14R 분석`
  - `분석일 2026-07-05`
  - `LIVE`
  - no stale historical helper text
  - all seven names present: `김주한`, `권혁진`, `박용범`, `김정우`, `김민준`, `김태완`, `유태복`

## Residual audit question
- Fable5 should independently review whether the same-day settlement buffer and market-source classification are sufficient for all KCYCLE timing edge cases.
