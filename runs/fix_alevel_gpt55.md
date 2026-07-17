# RACELENS UI/UX A- Gate Fix

## Scope
- Rebuilt app icon, adaptive icon, favicon, splash, and 48px preview from `mobile/scripts/generate-brand-assets.mjs`.
- Added centered lens geometry, short gold gauge arc, flat inner lens, larger aperture/pupil, adaptive safe-zone scaling, and primary mass centering assert.
- Added `BrandMark` SVG component to the Home hero lockup and aligned RaceLens wordmark spacing.
- Split roster block waiting vs mismatch copy and added `다시 시도` / `다른 경주 선택` actions.
- Documented brand mark, CTA, red/rose, and roster-block rules in `mobile/DESIGN.md`.

## Verification
- PASS: `node mobile\scripts\generate-brand-assets.mjs`
- PASS: `cd mobile && npm run typecheck`
- PASS: `cd mobile && npm run lint:design`
- PASS: `cd mobile && npm run lint:security`
- PASS: `cd mobile && npm run qa:mobile` (`66 passed`)
- PASS: `cd mobile && npm run qa:api-failure`
- PASS: `cd mobile && npm run qa:analytics`
- PASS: `cd mobile && npm run qa:release-visual`
- PASS: red grep found no non-error raw red usage in `mobile/src`; only design docs, script channel variables, and non-color text matches remain.

## Visual Evidence
- Reviewed `mobile/assets/icon.png`, `mobile/assets/adaptive-icon.png`, `mobile/assets/splash.png`, and `runs/icon_preview_48.png`.
- Reviewed release visual QA captures in `C:\tmp\racelens-release-readiness`.

## Stdout Last Line
`brand assets generated and verified: icon 1024x1024 centered, adaptive 1024x1024 safe-zone transparent, splash 1284x2778, favicon 192x192, preview 48x48`
