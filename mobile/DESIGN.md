# RaceLens Design System

## 1. Atmosphere & Identity

RaceLens is a race-intelligence cockpit, not a gambling product. It should feel closer to a commercial finance/sports app than a soft wellness dashboard: graphite control surfaces, clean neutral cards, one kinetic lime signal, horse-race gold, and red reserved for errors or destructive states only. The signature is the "race board": near-black track panels, bright signal selections, neutral data cards, teal verification states, and plain uncertainty cues that make model confidence inspectable instead of magical. Home is only for choosing race conditions and starting analysis. The analysis screen must first answer the user's core question: who is predicted 1st, 2nd, and 3rd. Only then should detailed rider/horse data, evidence, odds, and risk context follow.

Reference synthesis:
- Commercial color benchmark: finance apps such as Robinhood lean on black/white/neutrals with a memorable neon signal; Toss-style mobile foundations show broad neutral ramps before brand color; sports brands such as DraftKings use strong green/orange energy on disciplined black/white/grey systems. RaceLens borrows the structure, not the exact brand assets.
- curated.design: category-level scan for data apps, AI products, finance tools, and mobile app structure.
- godly.website: recent design taste checks across web interface, branding, product, typography, motion, and app icons.
- awwwards.com: high-craft interaction benchmark; borrow restraint and polish, not heavy web spectacle.
- landing.love: motion and launch-page references for any RaceLens web marketing surface.
- saaspo.com: SaaS dashboard hierarchy, subscription framing, and compact operational UI density.
- onepagelove.com: concise one-page launch/onboarding structure.
- navbar.gallery and cta.gallery: navigation and call-to-action hierarchy, especially for safe non-betting conversion.
- collectui.com and mobbin.com: practical mobile flows, paywall patterns, onboarding, and recurring app screen structure.
- 60fps.design: micro-interactions should be small, tactile, and useful: rail fill, chip press, tab transition, loading shimmer, long-press/board-like feedback.
- 21st.dev: component quality benchmark for React primitives and templates.
- open-design.ai and getdesign.md: every app screen derives from this DESIGN.md, so the visual language is reusable instead of one-off.
- component.gallery: use standard interface patterns with names users already understand: tabs, carousel-like chips, popovers/sheets later, accordions for legal details, pagination only for long archives.
- rebrand.gallery, logofolio.com, svgl.app, coolors.co, fontpair.co, hugeicons.com: brand, logo, SVG, palette, font, and icon research inputs; final app tokens and assets must remain license-safe.
- Dezign Heroes: use curated icon/motion resources as taste signals, but keep implementation native and store-safe.

## 2. Color

### Palette

| Role | Token | Light | Dark | Usage |
|------|-------|-------|------|-------|
| Surface/base | surfaceBase | #F6F8F1 | #080A08 | Neutral app wash / night-track background |
| Surface/raised | surfaceRaised | #FFFFFF | #111611 | Commercial card surface / dark instrument panels |
| Surface/inset | surfaceInset | #EBF0E8 | #1C241C | Inputs, chips, metric wells |
| Surface/glass | surfaceGlass | #FFFFFFEB | #131A14EB | Overlay panels |
| Surface/overlay | overlayScrim | #11151266 | #000000B8 | Free ad gate and blocking overlays |
| Surface/board | surfaceBoard | #101512 | #030604 | Signature race board hero |
| Text/on-board | textOnBoard | #FFFFFF | #FFFFFF | Hero board headline and stat value |
| Text/board-muted | textBoardMuted | #DDE7D8 | #DDE7D8 | Hero board supporting copy |
| Text/board-quiet | textBoardQuiet | #A8B6A4 | #A8B6A4 | Hero board labels |
| Text/primary | textPrimary | #111512 | #F7FAF2 | Headlines, body |
| Text/secondary | textSecondary | #455047 | #C9D5C3 | Captions, helper text |
| Text/muted | textMuted | #687168 | #98A590 | Disabled, quiet metadata |
| Border/subtle | borderSubtle | #D2DBCF | #2F3A2F | Dividers, cards |
| Accent/primary | accentPrimary | #1E6B4F | #4FC08D | General non-error action and data accent; sport CTAs use sportPalette |
| Accent/signal | accentSignal | #C9F24A | #D2FF5A | Selected state, hero confidence, active tab |
| Accent/teal | accentTeal | #007F72 | #58DEC8 | Positive verified states |
| Accent/gold | accentGold | #A86500 | #F4B83F | Historical/sample odds marker; horse text/border on light surfaces |
| Accent/gold-surface | accentGoldSurface | #F0B429 | #F4B83F | Bright horse fill for chips, tabs, CTA, and podium hero (dark text on top) |
| Accent/turf | accentTurf | #1E6B4F | #4FC08D | Horse-racing secondary turf identity |
| Accent/amber | accentAmber | #A86500 | #F4B83F | Caution, market missing |
| Accent/rose | accentRose | #B4233F | #FF6F8A | Error, restriction |
| Accent/violet | accentViolet | #5A45C8 | #AA98FF | Pro and advanced review |
| Brand/background | brandBackground | #0B0D0C | #0B0D0C | App icon and splash full-bleed brand field |
| Brand/lens-inner | brandLensInner | #10150F | #10150F | Flat inner lens fill inside the lime ring |
| Brand/white | brandWhite | #F7FAF2 | #F7FAF2 | Aperture and splash wordmark |
| Shadow/tint | shadowTint | #152018 | #020402 | Tinted card elevation |
| Rail/base | railBase | #D6E0D3 | #2A342A | Probability rail track |

### Rules
- The signature memory is kinetic lime on an obsidian race board with horse-race gold as the paired sport accent. Red is not a primary action color; keep red/rose for blocked, unsafe, error, or destructive states only.
- Brand mark geometry is fixed at a 1024 canvas: full-bleed #0B0D0C field, centered lime lens ring r300/stroke88, flat #10150F inner lens r256, white aperture r96, black pupil r40, and a short gold gauge arc r424/stroke56 from -50deg to +50deg. Do not reintroduce radial glow or long offset double rings.
- Sport-scoped accent is resolved through `sportPalette(mode, sport)`: keirin keeps lime signal; horse uses gold with turf-green header tint. Header tint, selected chips, pick highlights, and active tabs must use this helper.
- Selected chips, active tabs, and primary submit actions use the current sport accent via `sportPalette(mode, sport)`: keirin uses lime with dark text, horse uses gold with contrast-safe text. Do not blur them into one generic accent.
- Number identity is domain-critical: keirin uses official 1-7 number colors; horse uses dark gate badges with a gold border.
- Teal means verified or healthy data path. Brass/amber means sample, uncertain, or missing data. Rose means blocked or unsafe.
- Surfaces use neutral commercial whites, graphite boards, and soft green-grey insets so the app feels like a race-intelligence instrument rather than a generic SaaS screen. Light mode must not become a flat hospital-mint screen.
- Keep accent colors purposeful. No casino neon, no purple-blue AI gradient, and no raw colors outside `src/theme/tokens.ts`.
- Red/rose tokens (`accentRose` and legacy raw reds such as #C7361A/#FF623D) are restricted to error, blocked, unsafe, or destructive states. Sport CTAs, pills, tabs, and ordinary progress must use sportPalette, teal, amber, lime, or gold.
- Any legal/safety notice uses amber or rose with plain language.

## 3. Typography

### Scale

| Level | Size | Weight | Line Height | Tracking | Usage |
|-------|------|--------|-------------|----------|-------|
| Display | 34px | 700 | 1.08 | 0 | App title, hero metric |
| H1 | 28px | 700 | 1.14 | 0 | Screen heading |
| H2 | 22px | 700 | 1.22 | 0 | Section heading |
| H3 | 18px | 700 | 1.30 | 0 | Card title |
| Body | 15px | 500 | 1.45 | 0 | Default text |
| Body/strong | 15px | 700 | 1.35 | 0 | Values and labels |
| Body/sm | 13px | 500 | 1.40 | 0 | Secondary copy |
| Caption | 11px | 700 | 1.25 | 0.08em | Pills, overlines |
| Mono | 12px | 700 | 1.35 | 0.03em | Status IDs and percentages |

### Font Stack
- Primary: system San Francisco / Apple SD Gothic Neo / Noto Sans KR fallback.
- Mono: SF Mono / ui-monospace fallback.

### Rules
- No display text inside compact cards unless it is the single hero metric.
- Korean labels must fit within small controls without truncating meaning.
- Do not use emoji as icons; use vector line icons or text labels.

## 4. Spacing & Layout

### Base Unit
All spacing derives from 4px.

| Token | Value | Usage |
|-------|-------|-------|
| space1 | 4px | Tight inline spacing |
| space2 | 8px | Chip gap |
| space3 | 12px | Compact padding |
| space4 | 16px | Default card padding |
| space5 | 20px | Screen horizontal padding |
| space6 | 24px | Large card padding |
| space8 | 32px | Section separation |
| space10 | 40px | Hero spacing |

### Grid
- Mobile-first single column.
- Screen padding is 20px.
- Cards use 16px radius unless they are pills. `radius.large` at 22px is reserved for hero/primary control panels that need stronger separation.
- Bottom navigation height reserves safe-area inset.

### Rules
- No nested cards. Use inset wells inside cards when necessary.
- Lists must remain scan-friendly: label left, value right, explanation below.

## 5. Components

### LensCard
- **Structure**: outer rounded panel with optional header, body, footer.
- **Variants**: base, glass, warning, danger, verified, pro.
- **Spacing**: space4 default, space6 for hero panels.
- **States**: default, pressed, loading, empty.
- **Accessibility**: card actions must expose a role and label.
- **Motion**: fade-up on mount, press scale 0.985.

### StatusPill
- **Structure**: compact pill with optional dot and uppercase/caption label.
- **Variants**: verified, caution, blocked, pro, neutral.
- **Spacing**: horizontal space3, vertical space2.
- **States**: default, focus.
- **Accessibility**: text never color-only; label includes semantic state.
- **Motion**: opacity + transform only.

### BrandMark
- **Structure**: react-native-svg mark used in icon, splash-derived hero lockup, and small in-app brand surfaces.
- **Geometry**: simplified version of the app icon: lime ring, flat inner lens, white aperture, black pupil, and short gold gauge arc.
- **Size**: 24-28px in navigation or hero lockups; do not shrink below 24px because the aperture/pupil must remain legible.
- **Color**: use `accentSignal`, `accentGold`, `brandLensInner`, `brandWhite`, and `brandBackground`; no raw hex in components.
- **Accessibility**: decorative when paired with visible RaceLens text, labelled only when it stands alone.

### NumberBadge
- **Structure**: circular numeric badge for entrants and selections.
- **Keirin**: official number colors: 1 white, 2 black, 3 red, 4 blue, 5 yellow, 6 green, 7 pink, with contrast-safe text.
- **Horse**: neutral dark gate badge with a gold border so horse racing does not inherit keirin colors.
- **Usage**: participant cards, prediction podium, pick rows, market odds board, and horse gate board.

### DataStatusStrip
- **Structure**: one-line three-slot strip: roster verification, odds freshness, race phase.
- **Color**: teal for healthy, amber for caution/missing, rose for blocked.
- **Usage**: analysis screen top, replacing scattered roster/freshness/phase status text.

### ProbabilityRail
- **Structure**: label row, fixed-height rail, percentage.
- **Variants**: primary, teal, amber, rose, violet.
- **Spacing**: space2 between label and rail.
- **States**: default, loading with shimmer, empty.
- **Accessibility**: exposes percentage in text.
- **Motion**: fill animates from 0 to its target width on mount and on value change (220ms, ease-out); final width matches the underlying percentage exactly.

### RaceSelector
- **Structure**: segmented sport control, schedule-only race date chip row, venue/race picker.
- **Variants**: compact, loading, disabled.
- **Spacing**: space3 between controls, space4 around group.
- **States**: default, selected, disabled, loading, error.
- **Accessibility**: each control is a button with selected state.
- **Touch target**: interactive chips and buttons must be at least 44px tall on mobile.
- **Date rule**: users cannot type arbitrary dates; only dates present in the sport/venue race schedule are selectable. Previous/next controls move to the previous/next race day, not calendar day.
- **Primary action**: `모델 신호 보기` uses the current sport CTA color from `sportPalette(mode, sport)` with an analytics icon and explicit accessibility label. It must not use red or rose unless the action is destructive or blocked.
- **CTA rule**: normal sport actions never use red. Keirin CTA is lime signal; horse CTA is gold. Red/rose only appears when a request is blocked, unsafe, failed, or destructive.

### PredictionSummary
- **Structure**: analysis-first result card with race context, a podium metaphor, one large centered 1st-candidate metric, lower 2nd/3rd candidates, and a separate compact TRI row.
- **Variants**: verified, caution, blocked, neutral through the parent `LensCard` risk variant.
- **Spacing**: ranked rows use space3 gaps and inset surfaces; the title must stay above all evidence modules.
- **States**: default, fallback when participant details are missing, empty/error card when official API data or 1·2·3착 prediction order is unavailable, blocked/risk state through status card nearby.
- **Accessibility**: every ranked row must expose visible text for rank, participant number, name, and main supporting stat. Color is decorative only.
- **Rules**: the only large number on the screen is the 1st-candidate model estimate; TRI probability is small and explicitly labeled `모델 추정`; empty/error states never show candidate badges, podium slots, or 0% model values.
- **Purpose**: make the app immediately answer the user's primary question before asking them to inspect detailed evidence.

### ParticipantBoard
- **Structure**: accordion list for each rider or horse. Collapsed state shows only number badge, name, and three core metrics.
- **Core metrics**: keirin uses score, 200m, placing rate; horse uses quinella rate, gate, assigned weight.
- **Expanded state**: profile, recent flow, tactics, note, and Pro reasons.
- **Variants**: collapsed default, expanded on tap.
- **Spacing**: rows use space3 gap and inset surface; no nested cards.
- **States**: default, empty later, live data later.
- **Accessibility**: every row includes a combined label so the source data is readable without relying on model picks.
- **Purpose**: keep RaceLens from forcing a judgement by showing the underlying provided race materials next to the model signal.

### EvidenceGuide
- **Structure**: beginner-oriented evidence sequence with three ranked evidence rows and optional glossary chips.
- **Variants**: full for analysis; do not show on home because home should not look like an example prediction page.
- **Keirin content**: center candidate, 200m/average score/placing rate, tactics, and market-source variable.
- **Horse content**: horse condition, jockey/assigned weight, gate or running-style variable, and market-source variable.
- **Accessibility**: every row uses visible text, not color-only meaning; numbers are sequence aids, not betting priority commands.
- **Purpose**: let first-time users compare concrete grounds before reading the model pick, reducing the feeling that the app is forcing a judgment.

### MarketOddsBoard
- **Structure**: market-source board for win, quinella/exacta, and trifecta odds: code/label left, number badges plus odds right, change text below in detailed mode.
- **Variants**: compact for home, detailed for analysis, live, fallback, empty.
- **Spacing**: rows use space3 gap and inset surface; the live/fallback pill stays 32px+ high.
- **States**: live data, fallback/waiting, empty.
- **Accessibility**: every row exposes label, selection, odds, and change text in one readable label.
- **Purpose**: show real-time odds as neutral market material, separate from model recommendations, so users can compare source signals without being pushed toward an action.

### BottomTab
- **Structure**: three fixed tabs with line icons and labels.
- **Variants**: home, analyze, pro.
- **Spacing**: safe-area aware.
- **States**: active, inactive, pressed, focus.
- **Accessibility**: role tab, selected state.
- Active tab color follows `sportPalette(mode, sport)`.

### StoreSafeNotice
- **Structure**: legal card with short headline and body.
- **Variants**: compact, full, blocked.
- **Spacing**: space4.
- **States**: default.
- **Accessibility**: plain language; no small text below 13px for critical warnings.
- **Purpose**: state that hit rate is not profit, racing is average-loss after takeout, and RaceLens is an information analysis tool with no purchase, betting, or profit guarantee.

### RosterBlock
- **Structure**: blocking LensCard with status icon, state-specific headline/copy, and two bottom actions.
- **States**: unverified uses caution/amber with headline `공식 출주표를 아직 확인하지 못했습니다`; mismatch uses blocked/rose with headline `공식 출주표와 달라 예측을 중단했습니다`.
- **Actions**: primary `다시 시도` re-requests analysis; secondary `다른 경주 선택` returns to Home. Both actions are 44px+ touch targets.
- **Rule**: do not mix unverified and mismatch language in one sentence. No prediction podium, candidate badge, or 0% model value is shown while this card is active.

### FreeAdGate
- **Structure**: full-screen dim overlay with a raised advertising confirmation panel, policy-safe label, sponsor placeholder copy, primary confirm button, and quiet cancel button.
- **Variants**: default free-analysis gate; no Pro variant because Pro bypasses it.
- **Spacing**: panel uses space5 padding, space3 internal gaps, and 44px+ buttons.
- **States**: visible before a free analysis request, dismissed, confirming.
- **Accessibility**: visible title says this is an advertising gate, not a betting recommendation; confirm and cancel are buttons with explicit labels.
- **Purpose**: make the free plan feel materially different by requiring an ad confirmation before checking another race while keeping model output and betting actions separate.

## 6. Motion & Interaction

### Timing

| Type | Duration | Easing | Usage |
|------|----------|--------|-------|
| Tap | 120ms | cubic-bezier(0.2, 0.8, 0.2, 1) | Press feedback |
| Standard | 240ms | cubic-bezier(0.32, 0.72, 0, 1) | Tab and chip changes |
| Emphasis | 520ms | cubic-bezier(0.16, 1, 0.3, 1) | Screen entry |

### Rules
- Animate only opacity and transform in shared components.
- Respect reduced motion by keeping animations short and non-essential.
- Use 60fps-style microdetails sparingly: pill press and bottom-tab spring.

## 7. Depth & Surface

### Strategy
Mixed: tonal-shift plus restrained glass and hairline borders.

| Level | Treatment | Usage |
|-------|-----------|-------|
| Base | surfaceBase | Screen background |
| Raised | surfaceRaised + borderSubtle | Cards |
| Inset | surfaceInset | Metric wells, selectors |
| Glass | surfaceGlass + blur where native supports it | Hero lens panels |

Depth must not look like casino lighting. Use restrained brass/teal status accents, vermillion action points, mint data panels, lacquer board surfaces, and compact motion instead of heavy drop shadows or generic blue glow.
