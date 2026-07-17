# RaceLens Live E2E Report

- Target: https://168-107-2-218.sslip.io
- Export API: https://168-107-2-218.sslip.io
- Browser harness: Chromium web security disabled to emulate native mobile networking against the live API
- Analysis calls: 3/3
- Past date: 2026-07-05

| Journey | Result | Evidence |
|---|---|---|
| 01 /recent keirin source and home chip match | PASS | 6 days, selected 2026-07-10: 2026-07-03, 2026-07-04, 2026-07-05, 2026-07-10, 2026-07-11, 2026-07-12 |
| 02 direct mobile alias /recent?sport=kra resolves horse Seoul | EXPECTED_FAIL_UNTIL_DEPLOY | sport keirin |
| 03 horse Seoul schedule loads in UI | PASS | 6 horse days, selected 2026-07-07 |
| 04 analysis journey renders honest result state | PASS | live status 200, state roster_waiting |
| 05 tab roundtrip preserves selected race context | PASS | home -> pro -> analyze kept 광명 1R |
| 06 legal links open live server URLs | EXPECTED_FAIL_UNTIL_DEPLOY | 지원 문의 https://168-107-2-218.sslip.io/legal/support returned 404 |
| 07 analysis refresh stays within quota | PASS | second analysis request completed |
| 08 ux-events posts to live server with 202 | PASS | ux statuses 202,202,202,202,202,202,202,202,202,202,202,202,202,202,202,202,202,202,202,202,202,202,202,202,202 |
| 09 past race day lookup renders settled or honest unavailable state | PASS | 2026-07-05: settled |
| 10 browser stability gates | PASS | console error 0, overflow <= 1px |

## DEPLOY_REQUIRED

- `/recent` CORS and sport aliases `kra -> horse`, `kcycle -> keirin` must be deployed to the live server.
- `/legal/support` and `EXPO_PUBLIC_RACELENS_SUPPORT_URL` must be deployed before the fourth Pro information link can return HTTP 200 on the live server.
- Past keirin result settlement fix must be deployed for old races to return `status: settled` whenever official results are available.
