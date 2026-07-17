# RaceLens Store Policy Guardrails

## Allowed Product Language

- Race data analysis
- Model signal
- Confidence and uncertainty
- Historical verification
- Race recap
- Market data availability

## Blocked Product Language

- Guaranteed win
- Guaranteed profit
- Betting recommendation
- Stake amount
- Buy this ticket
- Risk-free
- Sure hit

## Blocked Features Before Legal Review

- Links to betting, ticket purchase, wagering, or official online sales.
- In-app workflows that calculate purchase amount or staking strategy.
- User-entered bankroll, profit target, or loss-chasing tools.
- Gambling ads or affiliate links.
- Push notifications that create urgency to bet.

## Required In-App Notices

- The app is for information and analysis only.
- It does not provide financial advice, betting instructions, or guaranteed results.
- Race outcomes are uncertain and model performance can degrade.
- The app is intended for adults.

## Subscription Rules

- Paid features can remove ads and unlock unlimited analysis, historical comparisons, model lab visibility, and notification convenience.
- Paid features must not imply better betting outcomes.
- Payment activation waits until Play Store and App Store policy review wording is finalized.

## Release Blockers To Clear

- Apple Developer and Google Play Developer accounts are created and app records exist.
- Privacy policy, terms, and account deletion pages are live before reviewer access at `/legal/privacy`, `/legal/terms`, and `/legal/account-deletion` on the production HTTPS API domain.
- EAS production submit credentials are configured outside git.
- Production API uses HTTPS and points to the Korea-hosted collector/backend, not Render, a local URL, QA tunnel, or placeholder domain.
- Store billing is either disabled for review or wired to server-side receipt validation before paid access is enabled.
- TestFlight and Google closed testing evidence is collected before public rollout.
- Privacy-safe UX analytics endpoint is configured so beta retention, analysis-start rate, detail-view rate, session length, and Pro-screen drop-off can be measured before public rollout.
- Store listing copy keeps the product framed as data analysis, with no betting CTA, purchase link, or profit guarantee.
- `npm run qa:store-readiness` passes with the real release environment loaded.
- `npm run qa:submission` passes with the real release environment loaded before TestFlight, Google closed testing, or public submission.
- AdMob's rewarded unit uses `https://<production-domain>/api/rewarded-ad/ssv`, production contains no Google test IDs, and Play Data safety discloses Google Mobile Ads processing.
