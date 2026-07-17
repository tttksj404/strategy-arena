# RACELENS Brand Asset Fix - GPT-5.5

## Changes

- Regenerated `mobile/assets/icon.png`, `adaptive-icon.png`, `splash.png`, and `favicon.png` from `mobile/scripts/generate-brand-assets.mjs`.
- Replaced the old red primary CTA with sport-scoped CTA colors: keirin lime and horse gold.
- Reframed `accentPrimary` as a non-red data/action accent; red/rose is reserved for blocked, unsafe, error, or destructive states.
- Removed the decorative home hero radio icon and tightened the RaceLens wordmark with heavier weight and letter spacing.
- Updated `mobile/DESIGN.md` and mobile QA assertions to match the new brand rules.

## Validation

- `node scripts/generate-brand-assets.mjs` PASS
  - stdout last line: `brand assets generated and verified: icon 1024x1024, adaptive 1024x1024 safe-zone transparent, splash 1284x2778, favicon 192x192`
- `npm.cmd run typecheck` PASS
- `npm.cmd run lint:design` PASS
- `npm.cmd run lint:security` PASS
- `npm.cmd run qa:mobile` PASS: 66 passed
- `npm.cmd run qa:api-failure` PASS
- `npm.cmd run qa:analytics` PASS: 11 events
- `npm.cmd run qa:release-visual` PASS
  - 320 home keirin light, 375 home horse light, 390 analyze horse dark, 320 pro dark safety, 768 analysis keirin light all reported `badText: 0`, `overflow: 0`, `smallTargets: []`, and no errors.

## Self-Check

- Splash has no Korean text and no recursive icon nesting.
- App icon is full-bleed dark green-black with lime lens ring, gold gauge arc, and central aperture point.
- Adaptive icon is transparent outside the central 66% safe-zone circle; script asserts this invariant.
- CTA color no longer uses red/tomato for the main analysis action.
