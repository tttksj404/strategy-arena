# KRA market corpus results

generated_at: 2026-07-12T03:49:29+00:00
source_db: /Users/tttksj/kra/data/kra.db
corpus: /Users/tttksj/github-portfolio-docs-work/strategy-arena/data/kra_market_corpus.jsonl

## Probe
- `RaceDetailResult_1` has win/place odds and finish order in the local official DB.
- `API160_1/integratedInfo_1` is the confirmed-dividend route; local shell DNS blocked a fresh live probe, so measurement uses already collected official rows.

## Corpus
- races: 6249
- entries: 64548
- races_with_dividend: 3535

## Win market tiers
| tier | races | coverage | top1 | top3 |
|---|---:|---:|---:|---:|
| all | 6249 | 1.000 | 0.367 | 0.692 |
| very_strong_pull | 1597 | 0.256 | 0.542 | 0.773 |
| strong_pull | 3478 | 0.557 | 0.445 | 0.736 |
| price_short | 2230 | 0.357 | 0.507 | 0.762 |
| gap_wide | 3729 | 0.597 | 0.424 | 0.714 |
| weak_or_open | 735 | 0.118 | 0.244 | 0.614 |

## Win market by field size
| bucket | tier | races | coverage | top1 | top3 |
|---|---|---:|---:|---:|---:|
| field_le_7 | all | 73 | 1.000 | 0.329 | 0.712 |
| field_le_7 | very_strong_pull | 24 | 0.329 | 0.458 | 0.708 |
| field_le_7 | strong_pull | 46 | 0.630 | 0.370 | 0.696 |
| field_le_7 | price_short | 34 | 0.466 | 0.441 | 0.647 |
| field_le_7 | gap_wide | 44 | 0.603 | 0.386 | 0.727 |
| field_le_7 | weak_or_open | 8 | 0.110 | 0.125 | 0.500 |
| field_8_10 | all | 3429 | 1.000 | 0.382 | 0.708 |
| field_8_10 | very_strong_pull | 933 | 0.272 | 0.573 | 0.790 |
| field_8_10 | strong_pull | 1974 | 0.576 | 0.467 | 0.758 |
| field_8_10 | price_short | 1291 | 0.376 | 0.536 | 0.779 |
| field_8_10 | gap_wide | 2069 | 0.603 | 0.448 | 0.732 |
| field_8_10 | weak_or_open | 396 | 0.115 | 0.227 | 0.634 |
| field_11_plus | all | 2747 | 1.000 | 0.350 | 0.671 |
| field_11_plus | very_strong_pull | 640 | 0.233 | 0.500 | 0.750 |
| field_11_plus | strong_pull | 1458 | 0.531 | 0.418 | 0.709 |
| field_11_plus | price_short | 905 | 0.329 | 0.467 | 0.743 |
| field_11_plus | gap_wide | 1616 | 0.588 | 0.394 | 0.690 |
| field_11_plus | weak_or_open | 331 | 0.120 | 0.266 | 0.592 |

## Exotic market baseline
| pool | tier | races | coverage | top1 | top3 |
|---|---|---:|---:|---:|---:|
| 단승식 | all | 3519 | 1.000 | 0.374 | 0.702 |
| 단승식 | very_strong_pull | 2168 | 0.616 | 0.426 | 0.723 |
| 단승식 | strong_pull | 2586 | 0.735 | 0.413 | 0.719 |
| 단승식 | gap_wide | 2168 | 0.616 | 0.426 | 0.723 |
| 단승식 | weak_or_open | 333 | 0.095 | 0.258 | 0.643 |
| 복승식 | all | 3519 | 1.000 | 0.180 | 0.392 |
| 복승식 | very_strong_pull | 1180 | 0.335 | 0.234 | 0.446 |
| 복승식 | strong_pull | 1826 | 0.519 | 0.214 | 0.418 |
| 복승식 | gap_wide | 1221 | 0.347 | 0.230 | 0.440 |
| 복승식 | weak_or_open | 590 | 0.168 | 0.134 | 0.359 |
| 삼복승식 | all | 3534 | 1.000 | 0.110 | 0.249 |
| 삼복승식 | very_strong_pull | 667 | 0.189 | 0.204 | 0.376 |
| 삼복승식 | strong_pull | 1684 | 0.477 | 0.140 | 0.275 |
| 삼복승식 | gap_wide | 1120 | 0.317 | 0.163 | 0.300 |
| 삼복승식 | weak_or_open | 725 | 0.205 | 0.079 | 0.225 |
| 삼쌍승식 | all | 3534 | 1.000 | 0.029 | 0.081 |
| 삼쌍승식 | very_strong_pull | 12 | 0.003 | 0.083 | 0.333 |
| 삼쌍승식 | strong_pull | 144 | 0.041 | 0.056 | 0.146 |
| 삼쌍승식 | gap_wide | 92 | 0.026 | 0.076 | 0.141 |
| 삼쌍승식 | weak_or_open | 1686 | 0.477 | 0.018 | 0.071 |
| 쌍승식 | all | 3519 | 1.000 | 0.107 | 0.263 |
| 쌍승식 | very_strong_pull | 406 | 0.115 | 0.212 | 0.374 |
| 쌍승식 | strong_pull | 1011 | 0.287 | 0.147 | 0.308 |
| 쌍승식 | gap_wide | 456 | 0.130 | 0.195 | 0.353 |
| 쌍승식 | weak_or_open | 998 | 0.284 | 0.083 | 0.232 |

## Model-market disagreement
| split | races | coverage | model_top1 | market_top1 |
|---|---:|---:|---:|---:|
| all | 2859 | 0.458 | 0.193 | 0.300 |
| 2026 | 562 | 0.425 | 0.221 | 0.319 |
| fresh_from_20260622 | 59 | 0.440 | 0.169 | 0.322 |

## Verdict
- Pure measurement only; no promotion claim.
- Exotic coverage starts at 2025-01-03 in the local official dividend table; 2024 exotic boards are absent from the local table.
