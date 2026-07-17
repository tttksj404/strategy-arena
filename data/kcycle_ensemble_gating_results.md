# KCYCLE ensemble + strong-pull gating

generated_at: 2026-07-12T02:12:28+00:00
records: 13900
available_deployable_candidates: 120
selection: train/val only; selected top 20 by val exact

## Rank-average ensemble
| top_k | val exact | val board lift | test exact | test board lift | test top1 |
|---:|---:|---:|---:|---:|---:|
| 10 | 0.1534 | +0.317pp | 0.1828 | +1.084pp | 0.6402 |
| 20 | 0.1546 | +0.436pp | 0.1821 | +1.012pp | 0.6387 |
| 40 | 0.1530 | +0.277pp | 0.1792 | +0.723pp | 0.6373 |

## Strong-pull precision-coverage tradeoff
This table is not a lift claim; it reports precision vs coverage for the board favorite under stronger market concentration.
| tier | n | coverage | exact | board_exact | top1 |
|---|---:|---:|---:|---:|---:|
| all | 13900 | 1.0000 | 0.1653 | 0.1653 | 0.6154 |
| strong_pull_all | 2275 | 0.1637 | 0.3516 | 0.3516 | 0.8132 |
| strong_pull_top50pct | 1138 | 0.0819 | 0.3822 | 0.3822 | 0.8102 |
| strong_pull_top16pct | 364 | 0.0262 | 0.4341 | 0.4341 | 0.8544 |
