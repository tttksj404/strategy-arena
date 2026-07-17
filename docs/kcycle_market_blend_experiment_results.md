# KCYCLE market blend experiment

elapsed_sec: 70.1

Validation chooses on first 60% of each year; reported test is last 40%.

| year | split | name | top1 | n | flip_rate | rule |
|---:|---|---|---:|---:|---:|---|
| 2025 | test | blend_w0.30 | 0.6217 | 978 | 0.050 | (1-w)*model+w*market, w=0.30 |
| 2025 | test | blend_w0.40 | 0.6217 | 978 | 0.067 | (1-w)*model+w*market, w=0.40 |
| 2025 | test | blend_norm_w0.50 | 0.6217 | 978 | 0.081 | (1-w)*race-normalized-model+w*market, w=0.50 |
| 2025 | test | blend_norm_w0.35 | 0.6207 | 978 | 0.057 | (1-w)*race-normalized-model+w*market, w=0.35 |
| 2025 | test | blend_norm_w0.40 | 0.6207 | 978 | 0.066 | (1-w)*race-normalized-model+w*market, w=0.40 |
| 2025 | test | blend_w0.50 | 0.6207 | 978 | 0.081 | (1-w)*model+w*market, w=0.50 |
| 2025 | test | blend_norm_w0.25 | 0.6196 | 978 | 0.041 | (1-w)*race-normalized-model+w*market, w=0.25 |
| 2025 | test | disagree_guard_g0.18_w0.30 | 0.6196 | 978 | 0.046 | normal w=0.30; if model-market disagree and model_gap>=0.18 use .05; if weak market gap use .10 |
| 2025 | test | disagree_guard_g0.24_w0.30 | 0.6196 | 978 | 0.046 | normal w=0.30; if model-market disagree and model_gap>=0.24 use .05; if weak market gap use .10 |
| 2025 | test | tiered_b0.05_m0.30 | 0.6186 | 978 | 0.019 | market weight 0.05; if p>=.55/gap>=.08 then 0.30; if p>=.75/gap>=.16 then .70 |
| 2025 | test | tiered_b0.10_m0.30 | 0.6186 | 978 | 0.028 | market weight 0.10; if p>=.55/gap>=.08 then 0.30; if p>=.75/gap>=.16 then .70 |
| 2025 | test | blend_norm_w0.30 | 0.6186 | 978 | 0.048 | (1-w)*race-normalized-model+w*market, w=0.30 |
| 2025 | test | blend_w0.35 | 0.6186 | 978 | 0.060 | (1-w)*model+w*market, w=0.35 |
| 2025 | test | tiered_b0.05_m0.35 | 0.6176 | 978 | 0.021 | market weight 0.05; if p>=.55/gap>=.08 then 0.35; if p>=.75/gap>=.16 then .70 |
| 2025 | test | tiered_b0.10_m0.35 | 0.6176 | 978 | 0.030 | market weight 0.10; if p>=.55/gap>=.08 then 0.35; if p>=.75/gap>=.16 then .70 |
| 2025 | test | guarded_w35_p0.55_gap0.08 | 0.6176 | 978 | 0.030 | base market weight 0.10; only p>=0.55, gap>=0.08 uses weight 0.35 |
| 2025 | test | guarded_w35_p0.55_gap0.12 | 0.6176 | 978 | 0.030 | base market weight 0.10; only p>=0.55, gap>=0.12 uses weight 0.35 |
| 2025 | test | guarded_w35_p0.55_gap0.16 | 0.6176 | 978 | 0.030 | base market weight 0.10; only p>=0.55, gap>=0.16 uses weight 0.35 |
| 2025 | test | blend_w0.20 | 0.6176 | 978 | 0.033 | (1-w)*model+w*market, w=0.20 |
| 2025 | test | blend_w0.25 | 0.6176 | 978 | 0.040 | (1-w)*model+w*market, w=0.25 |
| 2025 | test | disagree_guard_g0.18_w0.35 | 0.6176 | 978 | 0.055 | normal w=0.35; if model-market disagree and model_gap>=0.18 use .05; if weak market gap use .10 |
| 2025 | test | strong_p0.60_gap0.08 | 0.6176 | 978 | 0.514 | market only if p>=0.60 and gap>=0.08; else model |
| 2025 | test | strong_p0.60_gap0.12 | 0.6176 | 978 | 0.514 | market only if p>=0.60 and gap>=0.12; else model |
| 2025 | test | strong_p0.60_gap0.16 | 0.6176 | 978 | 0.514 | market only if p>=0.60 and gap>=0.16; else model |
| 2025 | test | tiered_b0.05_m0.25 | 0.6166 | 978 | 0.017 | market weight 0.05; if p>=.55/gap>=.08 then 0.25; if p>=.75/gap>=.16 then .70 |
| 2025 | test | tiered_b0.10_m0.25 | 0.6166 | 978 | 0.026 | market weight 0.10; if p>=.55/gap>=.08 then 0.25; if p>=.75/gap>=.16 then .70 |
| 2025 | test | guarded_w35_p0.60_gap0.08 | 0.6166 | 978 | 0.027 | base market weight 0.10; only p>=0.60, gap>=0.08 uses weight 0.35 |
| 2025 | test | guarded_w35_p0.60_gap0.12 | 0.6166 | 978 | 0.027 | base market weight 0.10; only p>=0.60, gap>=0.12 uses weight 0.35 |
| 2025 | test | guarded_w35_p0.60_gap0.16 | 0.6166 | 978 | 0.027 | base market weight 0.10; only p>=0.60, gap>=0.16 uses weight 0.35 |
| 2025 | test | disagree_guard_g0.12_w0.30 | 0.6166 | 978 | 0.042 | normal w=0.30; if model-market disagree and model_gap>=0.12 use .05; if weak market gap use .10 |
| 2025 | test | disagree_guard_g0.24_w0.35 | 0.6166 | 978 | 0.056 | normal w=0.35; if model-market disagree and model_gap>=0.24 use .05; if weak market gap use .10 |
| 2025 | test | strong_p0.65_gap0.08 | 0.6166 | 978 | 0.411 | market only if p>=0.65 and gap>=0.08; else model |
| 2025 | test | strong_p0.65_gap0.12 | 0.6166 | 978 | 0.411 | market only if p>=0.65 and gap>=0.12; else model |
| 2025 | test | strong_p0.65_gap0.16 | 0.6166 | 978 | 0.411 | market only if p>=0.65 and gap>=0.16; else model |
| 2025 | test | guarded_w35_p0.65_gap0.08 | 0.6155 | 978 | 0.020 | base market weight 0.10; only p>=0.65, gap>=0.08 uses weight 0.35 |
| 2025 | test | guarded_w35_p0.65_gap0.12 | 0.6155 | 978 | 0.020 | base market weight 0.10; only p>=0.65, gap>=0.12 uses weight 0.35 |
| 2025 | test | guarded_w35_p0.65_gap0.16 | 0.6155 | 978 | 0.020 | base market weight 0.10; only p>=0.65, gap>=0.16 uses weight 0.35 |
| 2025 | test | disagree_guard_g0.18_w0.25 | 0.6155 | 978 | 0.037 | normal w=0.25; if model-market disagree and model_gap>=0.18 use .05; if weak market gap use .10 |
| 2025 | test | disagree_guard_g0.24_w0.25 | 0.6155 | 978 | 0.037 | normal w=0.25; if model-market disagree and model_gap>=0.24 use .05; if weak market gap use .10 |
| 2025 | test | blend_norm_w0.70 | 0.6155 | 978 | 0.109 | (1-w)*race-normalized-model+w*market, w=0.70 |
| 2025 | test | strong_p0.55_gap0.08 | 0.6155 | 978 | 0.611 | market only if p>=0.55 and gap>=0.08; else model |
| 2025 | test | strong_p0.55_gap0.12 | 0.6155 | 978 | 0.611 | market only if p>=0.55 and gap>=0.12; else model |
| 2025 | test | strong_p0.55_gap0.16 | 0.6155 | 978 | 0.611 | market only if p>=0.55 and gap>=0.16; else model |
| 2025 | test | blend_norm_w0.20 | 0.6145 | 978 | 0.033 | (1-w)*race-normalized-model+w*market, w=0.20 |
| 2025 | test | disagree_guard_g0.12_w0.25 | 0.6145 | 978 | 0.036 | normal w=0.25; if model-market disagree and model_gap>=0.12 use .05; if weak market gap use .10 |
| 2025 | test | blend_w0.70 | 0.6145 | 978 | 0.112 | (1-w)*model+w*market, w=0.70 |
| 2025 | test | model | 0.6135 | 978 | 0.000 | model only |
| 2025 | test | blend_w0.05 | 0.6135 | 978 | 0.008 | (1-w)*model+w*market, w=0.05 |
| 2025 | test | blend_norm_w0.05 | 0.6135 | 978 | 0.008 | (1-w)*race-normalized-model+w*market, w=0.05 |
| 2025 | test | blend_norm_w0.10 | 0.6135 | 978 | 0.017 | (1-w)*race-normalized-model+w*market, w=0.10 |
| 2025 | test | blend_w0.10 | 0.6135 | 978 | 0.018 | (1-w)*model+w*market, w=0.10 |
| 2025 | test | guarded_w35_p0.70_gap0.08 | 0.6135 | 978 | 0.018 | base market weight 0.10; only p>=0.70, gap>=0.08 uses weight 0.35 |
| 2025 | test | guarded_w35_p0.70_gap0.12 | 0.6135 | 978 | 0.018 | base market weight 0.10; only p>=0.70, gap>=0.12 uses weight 0.35 |
| 2025 | test | guarded_w35_p0.70_gap0.16 | 0.6135 | 978 | 0.018 | base market weight 0.10; only p>=0.70, gap>=0.16 uses weight 0.35 |
| 2025 | test | blend_w0.15 | 0.6135 | 978 | 0.026 | (1-w)*model+w*market, w=0.15 |
| 2025 | test | blend_norm_w0.15 | 0.6135 | 978 | 0.028 | (1-w)*race-normalized-model+w*market, w=0.15 |
| 2025 | test | disagree_guard_g0.12_w0.35 | 0.6135 | 978 | 0.049 | normal w=0.35; if model-market disagree and model_gap>=0.12 use .05; if weak market gap use .10 |
| 2025 | test | strong_p0.70_gap0.08 | 0.6135 | 978 | 0.274 | market only if p>=0.70 and gap>=0.08; else model |
| 2025 | test | strong_p0.70_gap0.12 | 0.6135 | 978 | 0.274 | market only if p>=0.70 and gap>=0.12; else model |
| 2025 | test | strong_p0.70_gap0.16 | 0.6135 | 978 | 0.274 | market only if p>=0.70 and gap>=0.16; else model |
| 2025 | test | market | 0.6104 | 978 | 1.000 | market favorite only |

chosen_2025: blend_w0.70 test_top1=0.6145 n=978 flip=0.112
robust_2025: blend_w0.30 test_top1=0.6217 n=978 flip=0.050
