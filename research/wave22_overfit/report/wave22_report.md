# Wave-22 리포트 -- G1 과최적화 정밀판정

GA(wave-21)가 산출하고 gross 1x 제약 복원으로 확정된 G1 구성(research/STRATEGY_CARD.md "G1 확정")이 진짜 엣지인지 과최적화인지를 6종 독립 검증으로 정밀판정한다. G1은 wave21 H1/H2/H3/H5를 통과했지만 IS-OOS 격차 33.5%p(I5 대비 상대 25.3%p)라는 약한 과최적화 신호가 있었다 -- 이 wave의 존재 이유는 그 신호의 실체를 가리는 것이다.

**G1 유전자**: {'entry_threshold_apr': 0.11818955034178509, 'exit_threshold_ratio': 0.3873937165748336, 'window_days': 14, 'top_k_pairs': 1, 'leg_fraction': 0.5, 'universe_breadth': 100, 'idle_mode': 'usdt_lend'}
**I5 유전자(기준선)**: {'entry_threshold_apr': 0.15, 'exit_threshold_ratio': 0.5, 'window_days': 7, 'top_k_pairs': 1, 'leg_fraction': 0.5, 'universe_breadth': 200, 'idle_mode': 'tiered'}

이 리포트는 결과를 유리하게 쓰지 않는다. 과최적화 신호가 확인되면 FAIL로 명시하고 G1 승격 철회를 권고한다 (사용자 지시 원문).

## 0. 엔진 재현성 확인

wave22 자체 캐시/엔진(`fitness.build_market_cache`/`run_backtest`/`cagr`/`oos_slice`)으로 G1을 재평가해 STRATEGY_CARD.md의 실측값과 대조했다. 이 재현이 어긋나면 이후 6종 검증 전부가 무의미하므로 가장 먼저 확인한다.

| 지표 | wave22 재계산 | STRATEGY_CARD 실측 | 차이 |
|---|---|---|---|
| 전기간 CAGR | 12.35% | 12.35% | -0.00%p |
| OOS CAGR(자체구간) | 4.04% | 4.04% | -0.00%p |

차이가 소수점 이하 수준이면(반올림 오차) 재현 성공으로 간주하고 이후 절을 진행한다.

## 1. 파라미터 안정성 지형 (가장 중요)

G1의 7개 유전자를 각각 +-10%/+-20% (범위 내 값은 gene 종류별 정의, methodology 참조) 흔들어 재평가했다. '이웃 평균 성과 / G1 성과' 비율이 1에 가까울수록 완만한 고원(엣지), 낮을수록 뾰족한 봉우리(과최적화)에 가깝다.

| 축 | G1 값 | 평가가능/전체 | 이웃평균비율 | 이웃최소비율 | 안정(>=0.8) | 경계값 |
|---|---|---|---|---|---|---|
| 진입임계(APR) | 0.11818955034178509 | 4/4 | 0.976 | 0.881 | 예 |  |
| 청산비율 | 0.3873937165748336 | 4/4 | 1.092 | 0.986 | 예 |  |
| 윈도우(일) | 14 | 2/4 | 0.913 | 0.905 | 예 | 예 |
| top_k | 1 | 2/4 | 2.300 | 1.904 | 예 | 예 |
| leg 비중 | 0.5 | 2/4 | 0.852 | 0.802 | 예 | 예 |
| 유니버스 폭 | 100 | 3/4 | 0.892 | 0.780 | 예 | 예 |
| 유휴모드 | usdt_lend | 3/3 | 0.978 | 0.960 | 예 |  |

**종합 안정성 비율 (7축 중 최솟값, 판정에 사용) = 0.852** (최약축: leg 비중)
참고 -- 7축 평균 = 1.143, 0.8 미만 축 수 = 0/7, 0.6 미만 축 수 = 0/7

### 축별 상세

**진입임계(APR)** (entry_threshold_apr)

| tier | gene값 | 전기간CAGR | G1대비비율 | 비고 |
|---|---|---|---|---|
| -20% | 0.09455164027342808 | 10.88% | 0.881 |  |
| -10% | 0.10637059530760658 | 12.01% | 0.972 |  |
| +10% | 0.13000850537596362 | 12.54% | 1.016 |  |
| +20% | 0.1418274604101421 | 12.77% | 1.034 |  |

**청산비율** (exit_threshold_ratio)

| tier | gene값 | 전기간CAGR | G1대비비율 | 비고 |
|---|---|---|---|---|
| -20% | 0.30991497325986694 | 12.18% | 0.986 |  |
| -10% | 0.34865434491735026 | 12.26% | 0.993 |  |
| +10% | 0.42613308823231705 | 14.74% | 1.194 |  |
| +20% | 0.4648724598898003 | 14.78% | 1.197 |  |

**윈도우(일)** (window_days)

| tier | gene값 | 전기간CAGR | G1대비비율 | 비고 |
|---|---|---|---|---|
| -2step | 7 | 11.18% | 0.905 |  |
| -1step | 10 | 11.38% | 0.921 |  |
| +1step | - | - | - | index 4+1=5 outside choice range [0, 4] (3, 5, 7, 10, 14) -- G1 is at the range boundary on this axis |
| +2step | - | - | - | index 4+2=6 outside choice range [0, 4] (3, 5, 7, 10, 14) -- G1 is at the range boundary on this axis |

**top_k** (top_k_pairs)

| tier | gene값 | 전기간CAGR | G1대비비율 | 비고 |
|---|---|---|---|---|
| -2step | - | - | - | index 0-2=-2 outside choice range [0, 2] (1, 2, 3) -- G1 is at the range boundary on this axis |
| -1step | - | - | - | index 0-1=-1 outside choice range [0, 2] (1, 2, 3) -- G1 is at the range boundary on this axis |
| +1step | 2 | 23.51% | 1.904 |  |
| +2step | 3 | 33.29% | 2.696 |  |

**leg 비중** (leg_fraction)

| tier | gene값 | 전기간CAGR | G1대비비율 | 비고 |
|---|---|---|---|---|
| -20% | 0.4 | 9.91% | 0.802 |  |
| -10% | 0.45 | 11.12% | 0.901 |  |
| +10% | - | - | - | +10% of 0.5 clips to registry bound [0.3, 0.5] and collapses onto baseline -- G1 sits at/near this bound |
| +20% | - | - | - | +20% of 0.5 clips to registry bound [0.3, 0.5] and collapses onto baseline -- G1 sits at/near this bound |

**유니버스 폭** (universe_breadth)

| tier | gene값 | 전기간CAGR | G1대비비율 | 비고 |
|---|---|---|---|---|
| -2step | - | - | - | index 1-2=-1 outside choice range [0, 3] (30, 100, 200, 300) -- G1 is at the range boundary on this axis |
| -1step | 30 | 10.41% | 0.843 |  |
| +1step | 200 | 13.01% | 1.054 |  |
| +2step | 300 | 9.63% | 0.780 |  |

**유휴모드** (idle_mode)

| tier | gene값 | 전기간CAGR | G1대비비율 | 비고 |
|---|---|---|---|---|
| alt1 | none | 11.86% | 0.960 |  |
| alt2 | majors_low_thr | 12.00% | 0.972 |  |
| alt3 | tiered | 12.38% | 1.003 |  |

### 2축 동시변동 격자 (top_k_pairs x leg_fraction 격자는 gross 1x 제약을 넘는 셀도 포함해 원시 신호 품질을 확인한다)

**진입임계(APR) x 청산비율** -- 유효 25칸, 1x초과 0칸, 비율 최소 0.876 / 평균 0.991

| tier_entry_threshold_apr | tier_exit_threshold_ratio | 전기간CAGR | G1대비비율 |
|---|---|---|---|
| -2 | -2 | 10.83% | 0.877 |
| -2 | -1 | 10.81% | 0.876 |
| -2 | 0 | 10.88% | 0.881 |
| -2 | 1 | 10.99% | 0.890 |
| -2 | 2 | 10.98% | 0.890 |
| -1 | -2 | 11.75% | 0.952 |
| -1 | -1 | 11.80% | 0.955 |
| -1 | 0 | 12.01% | 0.972 |
| -1 | 1 | 12.00% | 0.972 |
| -1 | 2 | 12.20% | 0.988 |
| 0 | -2 | 12.18% | 0.986 |
| 0 | -1 | 12.26% | 0.993 |
| 0 | 0 | 12.35% | 1.000 |
| 0 | 1 | 14.74% | 1.194 |
| 0 | 2 | 14.78% | 1.197 |
| 1 | -2 | 12.17% | 0.986 |
| 1 | -1 | 12.39% | 1.003 |
| 1 | 0 | 12.54% | 1.016 |
| 1 | 1 | 12.60% | 1.020 |
| 1 | 2 | 12.50% | 1.012 |
| 2 | -2 | 12.51% | 1.013 |
| 2 | -1 | 12.76% | 1.034 |
| 2 | 0 | 12.77% | 1.034 |
| 2 | 1 | 12.54% | 1.016 |
| 2 | 2 | 12.52% | 1.014 |

**top_k x leg 비중** -- 유효 9칸, 1x초과 6칸, 비율 최소 0.802 / 평균 1.669

| tier_top_k_pairs | tier_leg_fraction | 전기간CAGR | G1대비비율 |
|---|---|---|---|
| -2 | -2 | - | 범위밖 |
| -2 | -1 | - | 범위밖 |
| -2 | 0 | - | 범위밖 |
| -2 | 1 | - | 범위밖 |
| -2 | 2 | - | 범위밖 |
| -1 | -2 | - | 범위밖 |
| -1 | -1 | - | 범위밖 |
| -1 | 0 | - | 범위밖 |
| -1 | 1 | - | 범위밖 |
| -1 | 2 | - | 범위밖 |
| 0 | -2 | 9.91% | 0.802 |
| 0 | -1 | 11.12% | 0.901 |
| 0 | 0 | 12.35% | 1.000 |
| 0 | 1 | - | 범위밖 |
| 0 | 2 | - | 범위밖 |
| 1 | -2 | 18.59% | 1.505(1x초과) |
| 1 | -1 | 21.03% | 1.703(1x초과) |
| 1 | 0 | 23.51% | 1.904(1x초과) |
| 1 | 1 | - | 범위밖 |
| 1 | 2 | - | 범위밖 |
| 2 | -2 | 26.05% | 2.110(1x초과) |
| 2 | -1 | 29.62% | 2.399(1x초과) |
| 2 | 0 | 33.29% | 2.696(1x초과) |
| 2 | 1 | - | 범위밖 |
| 2 | 2 | - | 범위밖 |

## 2. 시간 안정성 (롤링 워크포워드, 6개월 비중첩)

**G1 전체 승률 = 57.1%** (8/14구간)
- IS전용 구간 승률: 50.0% (12구간) / OOS포함 구간 승률: 100.0% (2구간)
- 전반부 승률: 57.1% / 후반부 승률: 57.1%
- 최장 G1 연승: 3구간 / 최장 I5 연승: 2구간

| 구간시작 | 구간끝 | G1 CAGR | I5 CAGR | 차이 | 승자 | 비고 |
|---|---|---|---|---|---|---|
| 2019-09-01 | 2020-03-01 | 9.52% | 10.74% | -1.22%p | I5승 |  |
| 2020-03-01 | 2020-09-01 | 14.98% | 20.28% | -5.31%p | I5승 |  |
| 2020-09-01 | 2021-03-01 | 38.63% | 36.69% | +1.94%p | G1승 |  |
| 2021-03-01 | 2021-09-01 | 27.05% | 27.87% | -0.83%p | I5승 |  |
| 2021-09-01 | 2022-03-01 | 7.26% | 5.88% | +1.37%p | G1승 |  |
| 2022-03-01 | 2022-09-01 | 5.16% | 2.20% | +2.96%p | G1승 |  |
| 2022-09-01 | 2023-03-01 | 2.48% | 2.23% | +0.24%p | G1승 |  |
| 2023-03-01 | 2023-09-01 | -0.08% | 0.65% | -0.73%p | I5승 |  |
| 2023-09-01 | 2024-03-01 | 43.43% | 7.83% | +35.60%p | G1승 |  |
| 2024-03-01 | 2024-09-01 | 32.71% | 30.91% | +1.80%p | G1승 |  |
| 2024-09-01 | 2025-03-01 | -2.02% | 0.24% | -2.27%p | I5승 |  |
| 2025-03-01 | 2025-09-01 | -3.18% | -0.93% | -2.25%p | I5승 |  |
| 2025-09-01 | 2026-03-01 | 3.04% | 2.78% | +0.25%p | G1승 | OOS포함 |
| 2026-03-01 | 2026-09-01 | 5.24% | 3.29% | +1.96%p | G1승 | OOS포함 |

**한계**:
- only 14 independent (non-overlapping) windows exist over the whole 2019-09~2026-07 history -- a win-rate statistic from this few samples has a wide confidence interval (e.g. a true 50% win rate could easily produce 9/14 or 5/14 by chance alone)
- 0 window(s) are flagged low_confidence (fewer than 60 obs, typically the final partial window)
- only 1-2 windows touch the OOS period (2025-10~) at all, since OOS itself is <1 year old -- the OOS-touching win rate is not independently informative beyond what validation #3's regime split already shows

## 3. 레짐 분해 (고펀딩 2020/2021/2024 vs 저펀딩 2022/2023/2025/2026)

| 연도 | 레짐 | G1 CAGR | I5 CAGR | 차이 | G1승 | 비고 |
|---|---|---|---|---|---|---|
| 2019 | 미분류 | 2.91% | 3.31% | -0.40%p | 아니오 | 부분연도 |
| 2020 | 고펀딩 | 16.60% | 19.20% | -2.59%p | 아니오 |  |
| 2021 | 고펀딩 | 30.93% | 30.45% | +0.48%p | 예 |  |
| 2022 | 저펀딩 | 3.37% | 1.75% | +1.61%p | 예 |  |
| 2023 | 저펀딩 | 18.86% | 3.00% | +15.86%p | 예 |  |
| 2024 | 고펀딩 | 17.18% | 17.28% | -0.11%p | 아니오 |  |
| 2025 | 저펀딩 | -2.03% | -0.03% | -2.00%p | 아니오 | OOS경계 |
| 2026 | 저펀딩 | 4.76% | 2.76% | +2.00%p | 예 | 부분연도 |

- 고펀딩기 평균 개선분: -0.74%p (중앙값 -0.11%p, 3개년, G1 1승)
- 저펀딩기 평균 개선분: +4.37%p (중앙값 +1.81%p, 4개년, G1 3승)
- **개선분 기여가 더 큰 레짐: 저펀딩기**
- 두 레짐 모두 개선(양수)인가: 아니오 / 한쪽에만 존재하는 개선: low_funding

**한계**:
- only 3 high-funding years and 4 low-funding years exist in the data -- per-bucket means rest on very few (3-4) independent annual observations each
- 2019 (partial, Sep-Dec only) is excluded from both buckets by the task's own year lists; 2026 is also partial (through the cache's own last date) and IS included in the low-funding bucket as specified, so its contribution is a part-year annualized figure, not a full year
- 2025 straddles OOS_SPLIT -- its bucket membership (low-funding) is a funding-regime statement, independent of validation #2's IS/OOS framing; do not conflate the two axes

## 4. DSR 재계산 (누적 시행 반영)

| 대상 | 시행수 | DSR score | probability |
|---|---|---|---|
| G1 (top_k=1, 이 wave 재계산) | 이 wave만 (7,500) | -0.75572 | 0.22491 |
| G1 (top_k=1, 이 wave 재계산) | 누적 (121+7,500=7,621) | -0.75970 | 0.22372 |
| GA_FINAL (top_k=3, wave21 원본, 참고) | 이 wave만 (7,500) | 0.23594 | - |
| GA_FINAL (top_k=3, wave21 원본, 참고) | 누적 (7,621) | 0.23196 | - |
| GA_FINAL (top_k=3, 이 wave 재계산 대조) | 이 wave만 (7,500) | 0.23594 | - |
| GA_FINAL (top_k=3, 이 wave 재계산 대조) | 누적 (7,621) | 0.23196 | - |

**G1의 누적시행 DSR = -0.75970 (양수 여부: 아니오)**

**한계**:
- DSR's own trial-count convention is inherently a judgment call (does a 'trial' count every GA individual, every generation's best, or every genome ever backtested); this module follows wave21_ga's own already-disclosed convention (gates21.GA_TRIALS/PRIOR_CUMULATIVE_TRIALS) rather than inventing a new one, for comparability
- the 121 prior-wave figure is a frozen, disclosed constant (gates21.PRIOR_CUMULATIVE_TRIALS), not independently re-derived by this wave -- see gates21.py's own comment for that precedent
- DSR assumes daily returns are the right unit and penalizes skew/kurtosis parametrically; it is a multiple-testing correction, not a guarantee -- a positive score is necessary, not sufficient, evidence of a real edge

## 5. 유전자 기여도 분해 (I5 -> G1, one-at-a-time)

I5->G1 전체 격차: 전기간 +2.08%p, OOS +0.98%p. 변경된 축은 5/7개(진입임계(APR), 청산비율, 윈도우(일), 유니버스 폭, 유휴모드), 동일값 축은 top_k, leg 비중.

| 축 | I5->G1 값변화 | 정방향 기여(I5+1축) | 역방향 기여(G1-1축) | 비고 |
|---|---|---|---|---|
| 진입임계(APR) | 0.15 -> 0.11818955034178509 | +1.76%p | -0.41%p |  |
| 청산비율 | 0.5 -> 0.3873937165748336 | +2.19%p | -0.08%p |  |
| 윈도우(일) | 7 -> 14 | +2.87%p | +1.17%p |  |
| top_k | 1 (동일) | - | - | - |
| leg 비중 | 0.5 (동일) | - | - | - |
| 유니버스 폭 | 200 -> 100 | -0.72%p | -0.67%p |  |
| 유휴모드 | tiered -> usdt_lend | -0.03%p | -0.03%p |  |

- 정방향 최대기여축: 윈도우(일) (양의 기여분 중 42.1%) -> 단일축 집중: 아니오 / 2축이상 분산: 예
- 역방향 최대기여축: 윈도우(일) (양의 기여분 중 100.0%) -> 단일축 집중: 예 / 2축이상 분산: 아니오
- 정방향/역방향 일치 여부: 아니오
- 정방향 기여 합 = +6.07%p (실제 격차 대비 상호작용 잔차 = -3.99%p)

**한계**:
- 2 of 7 genes (top_k_pairs, leg_fraction) are IDENTICAL between I5 and G1, so only 5 axes can possibly contribute to the gap -- 'spread across >=2 axes' is evaluated against this smaller active set, not all 7
- one-at-a-time attribution ignores higher-order interactions between genes; the interaction_residual_pp figure quantifies how much of the total gap the linear one-at-a-time sum does NOT explain -- a large residual means the genes interact and no single-axis story (concentrated or spread) fully describes the result
- forward and backward decompositions can disagree when interactions are strong; both are reported rather than only the more favorable one

## 6. 거짓 발견 대조 (무작위 유전자 30개)

G1 전기간 CAGR 12.35% / OOS CAGR 4.04%
- 전기간 CAGR 기준: G1은 무작위 30개 중 30개보다 우수 (상위 3.2%, 상위5%이내: 예)
- OOS CAGR 기준: G1은 무작위 30개 중 30개보다 우수 (상위 3.2%, 상위5%이내: 예)
- 시도 횟수(gross<=1x 조건 재추첨 포함): 96회 시도 -> 30개 채택

무작위 대조군 상위 5개 (전기간 CAGR 기준):

| idx | 전기간CAGR | OOS CAGR | gross |
|---|---|---|---|
| 0 | 12.06% | 1.91% | $87.97 |
| 19 | 9.95% | 2.71% | $87.35 |
| 29 | 9.26% | 1.62% | $87.39 |
| 15 | 9.13% | 1.13% | $88.14 |
| 20 | 8.98% | 0.53% | $81.86 |

무작위 대조군 하위 5개:

| idx | 전기간CAGR | OOS CAGR | gross |
|---|---|---|---|
| 10 | 5.98% | 0.00% | $89.28 |
| 13 | 5.87% | 1.65% | $59.50 |
| 18 | 5.19% | 2.05% | $60.78 |
| 1 | 4.83% | -3.94% | $60.95 |
| 28 | 4.36% | 0.60% | $59.51 |

**한계**:
- n=30 random draws is a small null sample -- percentile estimates have coarse resolution (1 draw = 3.3 percentage points); a 'top 5%' claim from n=30 rests on roughly the single best 1-2 draws' position
- the sizing constraint forces top_k_pairs=1 on every draw (see forced_axis_note) -- this control tests whether G1's OTHER 6 gene choices beat random chance, not whether its sizing choice does (sizing feasibility already settles that separately)
- an unconstrained variant (no gross filter, comparing on a leverage-normalized basis) was not run -- out of this validation's scope as specified

## 종합 판정

- 안정성 비율(최솟값): 0.852
- 롤링 승률: 57.1%
- 개선분 집중축(정방향): 윈도우(일) (단일축집중: 아니오, 2축이상분산: 예)
- DSR(누적시행): -0.75970 (양수: 아니오)
- 무작위대조 상위%(전기간CAGR): 3.2% (상위5%이내: 예)
- 개선분 기여 우세 레짐: low_funding

### PASS 기준 충족 현황

| 기준 | 충족여부 |
|---|---|
| 안정성비율 >=0.8 | 충족 |
| 롤링승률 >=55% | 충족 |
| 개선분 2축이상 분산 | 충족 |
| DSR>0 | 미충족 |
| 무작위대조 상위5% | 충족 |

### FAIL 사유 (하나라도 있으면 즉시 FAIL, 최우선 판정)

| 사유 |
|---|
| (없음) |

# 종합판정: CONDITIONAL

**권고**: G1 승격을 'paper 전진검증 필수' 조건부로 유지 -- 일부 기준만 충족(미충족: dsr_positive). 실거래 자본 투입 전 paper 전진검증(실시간, out-of-sample) 결과를 추가로 확인할 것.
