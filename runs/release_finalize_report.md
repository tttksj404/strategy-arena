# RaceLens Release Finalize Report

- Result: PASS
- Generated: 2026-07-07T13:52:14.258Z
- release.env: /Users/tttksj/github-portfolio-docs-work/strategy-arena/mobile/release.env

| Status | Check                       | Detail                                                                                                                                          |
| ------ | --------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| PASS   | release.env                 | wrote /Users/tttksj/github-portfolio-docs-work/strategy-arena/mobile/release.env                                                                |
| PASS   | healthz                     | https://168-107-2-218.sslip.io/healthz returned HTTP 200                                                                                        |
| PASS   | privacy page                | https://168-107-2-218.sslip.io/legal/privacy returned HTTP 200 with Korean body text                                                            |
| PASS   | terms page                  | https://168-107-2-218.sslip.io/legal/terms returned HTTP 200 with Korean body text                                                              |
| PASS   | account deletion page       | https://168-107-2-218.sslip.io/legal/account-deletion returned HTTP 200 with Korean body text                                                   |
| PASS   | live decision JSON          | https://168-107-2-218.sslip.io/api/live-decision?sport=kcycle&date=2026-07-07&meet=%EA%B4%91%EB%AA%85&race_no=1 returned 200 JSON with ok/error |
| PASS   | ux events live_odds_refresh | https://168-107-2-218.sslip.io/api/ux-events accepted event with HTTP 202                                                                       |
| PASS   | check-store-release-env     | store release environment gate passed                                                                                                           |
