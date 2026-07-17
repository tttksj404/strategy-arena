# RaceLens Local Preview Formalization Report

Date: 2026-07-10

## Files

- `.gitignore`: ignores `mobile/dist-preview/`.
- `mobile/PREVIEW.md`: documents local preview usage, 4173 same-origin requirement, live upstream behavior, and production AAB separation.
- `mobile/package.json`: adds `preview:build`, `preview:serve`, and `preview`.
- `mobile/scripts/preview-build.mjs`: cross-platform Expo web export wrapper with preview-only environment and no injected API base URL.
- `mobile/scripts/preview-server.mjs`: static SPA server plus streaming live API proxy to `https://168-107-2-218.sslip.io`.
- `mobile/scripts/preview.mjs`: sequential build-then-serve wrapper for `npm run preview`.

Production files intentionally unchanged:

- `mobile/eas.json`
- `mobile/app.json`
- `mobile/app.config.js`

## Verification

- `node --check scripts/preview-build.mjs`, `node --check scripts/preview-server.mjs`, `node --check scripts/preview.mjs`: pass.
- `node -e "JSON.parse(...package.json...)"`: pass.
- `npm run preview:build`: pass; `dist-preview/index.html` created.
  - Evidence capsule: `20260710-161229343-6dc2cb47`
  - Build warnings inspected: only Expo cache rebuild and repeated `NO_COLOR`/`FORCE_COLOR` notices.
- `node scripts/preview-server.mjs` on `0.0.0.0:4173`: pass.
  - `GET http://127.0.0.1:4173/`: 200.
  - `GET http://127.0.0.1:4173/recent?sport=kcycle&meet=광명`: 200; live JSON contained `days` with 6 entries.
- `npm run typecheck`: pass.
- `npm run lint:design`: pass.
- `npm run lint:security`: pass.
  - Evidence capsule: `20260710-161401868-d97f7570`

## Result

`npm run preview` in `mobile/` now rebuilds the local web export and starts a reproducible local preview server at `http://localhost:4173`, with same-origin API requests proxied to the live sslip.io server.
