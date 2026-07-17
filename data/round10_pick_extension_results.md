# Round 10 pick extension results

generated_at: `2026-07-12T13:17:45+00:00`

Pure measurement only. `engine.py`, `app.py`, and tests were not modified.

## A. KCYCLE remaining ticket types

Market columns are **삼쌍보드 근사**: normalized trifecta board mass marginalized to each ticket type, not native pool odds.

- snapshots deduped valid: 17312
- entries: 12593
- scored races: 12593

### 복승 (quinella)

1-2착 unordered pair; 삼쌍보드 1-2 슬롯 주변화

| subset | n | model | market proxy | market-model |
| --- | --- | --- | --- | --- |
| all | 12593 | 37.7% | 45.7% | +7.96pp |
| agree | 7612 | 49.7% | 49.7% | +0.00pp |
| disagree | 4981 | 19.3% | 39.4% | +20.14pp |
| 2026_all | 1198 | 31.8% | 43.1% | +11.27pp |
| 2026_agree | 603 | 44.4% | 44.4% | +0.00pp |
| 2026_disagree | 595 | 19.0% | 41.7% | +22.69pp |

Year split:

| year | n | model | market proxy | market-model |
| --- | --- | --- | --- | --- |
| 2018 | 2259 | 40.6% | 46.1% | +5.49pp |
| 2019 | 1387 | 47.1% | 52.5% | +5.41pp |
| 2020 | 321 | 48.0% | 55.5% | +7.48pp |
| 2021 | 209 | 37.3% | 53.1% | +15.79pp |
| 2022 | 2346 | 39.3% | 46.8% | +7.54pp |
| 2023 | 2350 | 34.4% | 42.9% | +8.51pp |
| 2024 | 770 | 33.4% | 38.4% | +5.06pp |
| 2025 | 1753 | 32.8% | 44.0% | +11.18pp |
| 2026 | 1198 | 31.8% | 43.1% | +11.27pp |

Tier matrix:

| tier | n | model | market proxy | market-model |
| --- | --- | --- | --- | --- |
| strong_pull | 2055 | 58.3% | 67.7% | +9.39pp |
| non_strong_pull | 10538 | 33.7% | 41.4% | +7.69pp |
| 2026_non_strong_pull | 1011 | 28.0% | 39.2% | +11.18pp |

2026 P3-refined: P0 31.8%, P1 43.1% (+11.27pp), P3 43.1% (+11.27pp), switch_n=0.
Decision: **candidate** (market_candidate=True, hybrid_candidate=True).

### 쌍승 (exacta)

1-2착 ordered pair; 삼쌍보드 1-2 슬롯 주변화

| subset | n | model | market proxy | market-model |
| --- | --- | --- | --- | --- |
| all | 12593 | 27.4% | 35.2% | +7.84pp |
| agree | 6211 | 40.0% | 40.0% | +0.00pp |
| disagree | 6382 | 15.1% | 30.6% | +15.47pp |
| 2026_all | 1198 | 23.0% | 33.1% | +10.18pp |
| 2026_agree | 472 | 35.8% | 35.8% | +0.00pp |
| 2026_disagree | 726 | 14.6% | 31.4% | +16.80pp |

Year split:

| year | n | model | market proxy | market-model |
| --- | --- | --- | --- | --- |
| 2018 | 2259 | 28.6% | 33.8% | +5.14pp |
| 2019 | 1387 | 34.7% | 40.7% | +5.98pp |
| 2020 | 321 | 36.1% | 39.9% | +3.74pp |
| 2021 | 209 | 27.8% | 43.1% | +15.31pp |
| 2022 | 2346 | 29.1% | 36.9% | +7.80pp |
| 2023 | 2350 | 25.7% | 33.7% | +8.00pp |
| 2024 | 770 | 23.1% | 30.9% | +7.79pp |
| 2025 | 1753 | 23.2% | 34.1% | +10.90pp |
| 2026 | 1198 | 23.0% | 33.1% | +10.18pp |

Tier matrix:

| tier | n | model | market proxy | market-model |
| --- | --- | --- | --- | --- |
| strong_pull | 2055 | 49.5% | 59.3% | +9.78pp |
| non_strong_pull | 10538 | 23.1% | 30.5% | +7.46pp |
| 2026_non_strong_pull | 1011 | 19.0% | 29.0% | +9.99pp |

2026 P3-refined: P0 23.0%, P1 33.1% (+10.18pp), P3 33.1% (+10.18pp), switch_n=0.
Decision: **candidate** (market_candidate=True, hybrid_candidate=True).

### 삼복승 (trio)

1-2-3착 unordered trio; 삼쌍보드 3슬롯 주변화

| subset | n | model | market proxy | market-model |
| --- | --- | --- | --- | --- |
| all | 12593 | 28.6% | 34.6% | +6.02pp |
| agree | 7526 | 38.5% | 38.5% | +0.00pp |
| disagree | 5067 | 13.8% | 28.8% | +14.96pp |
| 2026_all | 1198 | 26.8% | 33.6% | +6.84pp |
| 2026_agree | 734 | 35.3% | 35.3% | +0.00pp |
| 2026_disagree | 464 | 13.4% | 31.0% | +17.67pp |

Year split:

| year | n | model | market proxy | market-model |
| --- | --- | --- | --- | --- |
| 2018 | 2259 | 30.3% | 33.6% | +3.36pp |
| 2019 | 1387 | 32.8% | 37.8% | +4.97pp |
| 2020 | 321 | 32.7% | 36.1% | +3.43pp |
| 2021 | 209 | 29.7% | 39.7% | +10.05pp |
| 2022 | 2346 | 28.4% | 37.0% | +8.53pp |
| 2023 | 2350 | 28.0% | 34.1% | +6.04pp |
| 2024 | 770 | 25.8% | 30.0% | +4.16pp |
| 2025 | 1753 | 25.6% | 32.7% | +7.13pp |
| 2026 | 1198 | 26.8% | 33.6% | +6.84pp |

Tier matrix:

| tier | n | model | market proxy | market-model |
| --- | --- | --- | --- | --- |
| strong_pull | 2055 | 47.3% | 55.4% | +8.08pp |
| non_strong_pull | 10538 | 24.9% | 30.5% | +5.62pp |
| 2026_non_strong_pull | 1011 | 23.6% | 30.4% | +6.73pp |

2026 P3-refined: P0 26.8%, P1 33.6% (+6.84pp), P3 33.6% (+6.84pp), switch_n=0.
Decision: **candidate** (market_candidate=True, hybrid_candidate=True).

### 쌍복승 (quinella_place)

사용자 지정: 1-2착 unordered pair; 복승과 동일한 삼쌍보드 근사

| subset | n | model | market proxy | market-model |
| --- | --- | --- | --- | --- |
| all | 12593 | 37.7% | 45.7% | +7.96pp |
| agree | 7612 | 49.7% | 49.7% | +0.00pp |
| disagree | 4981 | 19.3% | 39.4% | +20.14pp |
| 2026_all | 1198 | 31.8% | 43.1% | +11.27pp |
| 2026_agree | 603 | 44.4% | 44.4% | +0.00pp |
| 2026_disagree | 595 | 19.0% | 41.7% | +22.69pp |

Year split:

| year | n | model | market proxy | market-model |
| --- | --- | --- | --- | --- |
| 2018 | 2259 | 40.6% | 46.1% | +5.49pp |
| 2019 | 1387 | 47.1% | 52.5% | +5.41pp |
| 2020 | 321 | 48.0% | 55.5% | +7.48pp |
| 2021 | 209 | 37.3% | 53.1% | +15.79pp |
| 2022 | 2346 | 39.3% | 46.8% | +7.54pp |
| 2023 | 2350 | 34.4% | 42.9% | +8.51pp |
| 2024 | 770 | 33.4% | 38.4% | +5.06pp |
| 2025 | 1753 | 32.8% | 44.0% | +11.18pp |
| 2026 | 1198 | 31.8% | 43.1% | +11.27pp |

Tier matrix:

| tier | n | model | market proxy | market-model |
| --- | --- | --- | --- | --- |
| strong_pull | 2055 | 58.3% | 67.7% | +9.39pp |
| non_strong_pull | 10538 | 33.7% | 41.4% | +7.69pp |
| 2026_non_strong_pull | 1011 | 28.0% | 39.2% | +11.18pp |

2026 P3-refined: P0 31.8%, P1 43.1% (+11.27pp), P3 43.1% (+11.27pp), switch_n=0.
Decision: **candidate** (market_candidate=True, hybrid_candidate=True).

## B. KRA place pick

- corpus races: 6249
- fresh split: rcDate >= 20260622

| subset | n | v4 pplc | plcOdds market | market-model |
| --- | --- | --- | --- | --- |
| all | 6249 | 60.8% | 67.9% | +7.07pp |
| agree | 3254 | 71.5% | 71.5% | +0.00pp |
| disagree | 2995 | 49.3% | 64.0% | +14.76pp |
| 2026_all | 1323 | 61.1% | 68.0% | +6.80pp |
| 2026_agree | 717 | 71.3% | 71.3% | +0.00pp |
| 2026_disagree | 606 | 49.2% | 64.0% | +14.85pp |
| fresh_all | 134 | 61.9% | 68.7% | +6.72pp |
| fresh_agree | 65 | 75.4% | 75.4% | +0.00pp |
| fresh_disagree | 69 | 49.3% | 62.3% | +13.04pp |

Year split:

| year | n | v4 pplc | plcOdds market | market-model |
| --- | --- | --- | --- | --- |
| 2024 | 2469 | 59.7% | 67.3% | +7.57pp |
| 2025 | 2457 | 61.8% | 68.5% | +6.72pp |
| 2026 | 1323 | 61.1% | 68.0% | +6.80pp |

## C. Weak-signal hybrid micro-optimization

KRA tier matrix:

| tier | n | v4 pplc | plcOdds market | market-model |
| --- | --- | --- | --- | --- |
| all | 6249 | 60.8% | 67.9% | +7.07pp |
| very_strong_pull | 1597 | 72.8% | 81.7% | +8.95pp |
| strong_pull | 1881 | 60.9% | 68.8% | +7.92pp |
| price_short | 23 | 78.3% | 87.0% | +8.70pp |
| gap_wide | 537 | 48.6% | 55.7% | +7.08pp |
| weak_or_open | 735 | 53.9% | 58.0% | +4.08pp |
- KRA all P3-refined: P0 60.8%, P1 67.9% (+7.07pp), P3 67.9% (+7.07pp), switch_n=0.
- KRA 2026 P3-refined: P0 61.1%, P1 68.0% (+6.80pp), P3 68.0% (+6.80pp), switch_n=0.
- KRA fresh P3-refined: P0 61.9%, P1 68.7% (+6.72pp), P3 68.7% (+6.72pp), switch_n=0.

KRA decision: **candidate** (market_candidate=True, hybrid_candidate=True).

