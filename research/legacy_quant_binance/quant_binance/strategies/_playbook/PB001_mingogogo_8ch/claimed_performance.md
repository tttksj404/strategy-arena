# PB001 — Mingogogo 주장 성과 (claimed performance)

> 출처: `coin_program_recommendation_with_rationale.csv` (41행) + `coin_program_dataset_summary_metrics_clean.csv` (17행 post-level 요약). 모두 블로거 본인이 발표한 사후 인증 포스트에서 추출.
> 마지막 갱신: 2026-04-28
> **이 통계는 블로거 자체 보고이며 cherry-pick 가능성이 매우 높음. §6 한계 섹션 필독.**

---

## 1. 요약 한 줄

**41건 (top winners only) 기준: 평균 +29.22%, 중앙값 +21.92%, win rate 100% (최소 +7.93%, 최대 +115.55%, 단 손실 0건). 그러나 17개 post 의 총 추천 수는 153 → 41/153 = 26.8% 만 공개. 나머지 73.2% (112건) 의 결과는 비공개 → 실제 win rate 는 미상.**

---

## 2. Post-level 요약 (n=17 post, summary_metrics.csv)

| 메트릭 | 값 |
|---|---|
| 인증 post 수 | 17 |
| 총 추천 코인 수 (분석에 포함) | **153** |
| post 별 평균 추천 코인 | 9.0 (range 3-12) |
| post 평균 수익률의 평균 | **+15.04%** |
| post 최고 수익률의 평균 | **+41.34%** |
| accuracy_pct (블로거 자체 보고) | **모든 post 100.0%** ← 시그널 |

→ "accuracy 100% × 17개 post 전부" 라는 점에서 통계적으로 매우 의심스러움 (실제 알고리즘이라면 분포 polarized).

---

## 3. Coin-level 분포 (n=41, recommendation_with_rationale.csv)

> 주의: 이 41행은 **각 post 가 자랑하는 top winner 만** 추출됨. 153건 전수가 아님.

### 기술 통계

| 메트릭 | 값 |
|---|---:|
| 표본수 (n) | **41** |
| 고유 post | 17 |
| 고유 코인 | 32 |
| **평균 수익률** | **+29.22%** |
| **중앙값** | **+21.92%** |
| 표준편차 | 24.59% |
| 최소 | +7.93% |
| 최대 | +115.55% (METEORA, post 218) |
| **win rate (>0%)** | **100.0% (41/41)** |

### 임계별 분포

| 임계 | 건수 | 비율 |
|---|---:|---:|
| > +10% | 36 / 41 | **87.8%** |
| > +30% | 13 / 41 | 31.7% |
| > +50% | 5 / 41 | 12.2% |
| > +100% | 2 / 41 | 4.9% (LISK +112.75, METEORA +115.55) |
| < 0% (손실) | 0 / 41 | **0%** |
| < -10% | 0 / 41 | 0% |

→ 단 1건의 손실도 없음. **공개된 41건이 사후 best-3-per-post 큐레이션** 임을 시사 (153 - 41 = 112 건의 패자 케이스 비공개).

### Top 10 winners

| post | 코인 | 티커 | 수익률 |
|---:|---|---|---:|
| 218 | 메테오라 | METEORA | +115.55% |
| 217 | 리스크 | LISK | +112.75% |
| 223 | 펏지펭귄 | PUDGY PENGUINS | +70.81% |
| 237 | 하이퍼레인 | HYPERLANE | +66.77% |
| 218 | 지토 | JITO | +51.76% |
| 222 | 히포프로토콜 | HIPPO PROTOCOL | +49.51% |
| 218 | 모나드 | MONAD | +41.74% |
| 217 | 커널다오 | KERNELDAO | +38.22% |
| 221 | 봉크 | BONK | +37.21% |
| 222 | 무브먼트 | MOVEMENT | +36.95% |

---

## 4. 시계열 분포 (cherry-pick 진단)

### Post 작성일 (블로그에 인증글이 올라온 날)

- 범위: **2025-11-27 ~ 2025-11-30** (4일간 17개 post 집중 발행)
- post 수: 11/27=4, 11/28=6, 11/29=3, 11/30=4
- → **4일에 17개 인증글 몰빵 발행** = 신규 가입자 유치용 마케팅 burst 의심

### 분석 기준 시점 (analysis_period — 추천 시점)

- 범위: **2025-07-10 ~ 2025-11-25** (4.5개월)
- 분포: 7월 13건 (76%), 11월 4건 (24%)
- → **블로그 발행일(11월 말)보다 4-5개월 전의 7월 추천을 11월 말에 사후 인증글로 발행** = 100% 사후선택 (look-ahead bias)
  - 즉 7월 14일 추천을 7월 17일 (3일 후) 결과로 평가했어야 하는데, 실제로는 4개월 후인 11월 28일에 "역사적으로 좋았던 케이스만" 골라서 발행
  - 전형적 **survivor bias 패턴**

### 5개월 라이브 운용 vs 17개 post

- 청구된 라이브 기간: **2025-07 ~ 2026-04** (9개월+)
- 그러나 인증글은 7월·11월의 일부 좋은 날만 17건
- **9개월 × 일평균 9건 추천 ≈ 2400+ 건 가능** vs **공개 41건 (top winners)** = 1.7%
- 진짜 win rate 산정 불가 (252:1 미공개 비율)

---

## 5. 자산 / 거래소 분포

### 거래소

- **모든 가격이 KRW (원화)** → **빗썸/업비트** (post 263 이 빗썸 명시, post 308 이 업비트 명시 → 두 거래소 혼합)
- 글로벌 USDT 페어 데이터 없음

### 자산 분류

- **알트코인 100%** — BTC/ETH 추천 0건
- **소형주 우세** (start_price KRW 0.032 ~ 713,600 — 가격 범위 7자릿수)
  - sub-$1 KRW: BONK (0.041원, 0.032원)
  - 1-100원: 다수 (PUDGY PENGUINS, MOVEMENT, ANKR 등)
  - 100-1000원: 다수
  - 1000원~ : 메이저 알트
  - BITCOIN CASH (713,600 KRW) — 유일한 메이저
- 심볼 매칭 품질: high 33건 (80%) / low 8건 (20%)

### 코인 카테고리 (32 unique)

- meme coin: BONK, PUDGY PENGUINS, MOO DENG, DOGECOIN, CAT IN A DOGS WORLD (5)
- L1/L2: MOVEMENT, MONAD, SEI, AELF, JITO, METEORA (Solana DeFi) (6)
- DeFi: UNISWAP, KERNELDAO, HYPERLANE, BOUNCE
- 기타: STORY, 0G, DOODLES, ANKR, LISK, ARDOR, HIPPO PROTOCOL, TETHER (?), STRATIS, CALDERA, NEXPACE, HUNT, SPACE ID, ONTOLOGY GAS, IMMUTABLE X, SAHARA AI, TEZOS, BITCOIN CASH, SIGN, PUNDI X

→ **알트코인 + 모멘텀 + meme/Solana DeFi 편중**. 단타 모멘텀 셋업에 최적화된 universe.

---

## 6. 신뢰도 평가 / 한계 (CRITICAL)

### Survivor / cherry-pick bias

1. **153건 추천 중 41건 (top 26.8%) 만 공개** → 나머지 112건의 결과는 미공개. 진짜 win rate 추정 불가.
2. **17개 post 모두 accuracy 100%** 보고 → 손실 케이스가 단 한 건도 없는 통계는 알고리즘 트레이딩에서 사실상 불가능. 손실 post 자체를 발행하지 않는 selection.
3. **11월 28-30일 4일간 17개 post burst** → 마케팅 캠페인성 사후 publish.
4. **수익률 측정 = peak_price_3d / start_price (3일 내 최고가)** → 실거래 exit 가 아님. 실제 청산은 peak 보다 훨씬 낮은 가격일 것 (TP 분할 청산 -3% buffer 가정 시 실현 수익률은 절반 이하 추정).

### 측정 방법론 결함

- entry/exit 모두 KRW spot 가격 (perp futures 환산 안 됨)
- 슬리피지·수수료·원화↔USDT 환손실 미반영
- 빗썸 illiquid 알트 (예: 가격 0.041 KRW = 0.00003 USD 수준) 의 실제 체결성 불명

### Forward-looking 검증 가능성

- **2025-12 ~ 2026-04 의 5개월간 라이브 추천 일자별 데이터는 posts.json (6MB) 에 누적되어 있음** → 일자별 추천 시점에서 **이후 3일 가격을 binance/upbit 공개 캔들로 재추적** 하면 진짜 win rate 산출 가능 (subagent 작업 권장)
- 단, 추천 코인 중 빗썸-only 종목은 binance 매핑 불가

### 결론

| 평가 항목 | 점수 (0-5) | 근거 |
|---|---:|---|
| 표본 충분성 | 1 | 41건 / 9개월, top-only |
| 셀렉션 bias 통제 | **0** | 100% accuracy 17개 post 전부 = 명백한 cherry-pick |
| 측정 방법 정직성 | 2 | peak_3d 사용 (실거래 exit 아님), 슬리피지 무시 |
| Forward-looking 재현성 | 3 | posts.json 으로 자체 재검증 가능 (별도 작업 필요) |
| **공개 성과의 신뢰도** | **1.5 / 5** | **참고 정도, 의사결정 근거 X** |

**권고**: 본 claimed_performance 는 "블로거 마케팅 자료" 로만 취급. 실제 PB001 게이트 통과 판단은 **자체 백테스트 (CH1+CH4+CH7+CH8 4채널 MVP) + posts.json 일자별 추천 forward-looking 재추적** 후 결정.

---

## 7. 다음 단계 (subagent 위임)

1. **posts.json 6MB 마이닝**: 230개 포스트에서 "추천", "리스트", "오전 N시", "코인" 키워드로 **일자별 raw 추천** 전수 추출 → CSV 화
2. **3일 후 가격 자동 추적**: upbit/bithumb API 로 추천 시점부터 +72h 가격 캔들 수집 → 진짜 max return + min drawdown + close-3d return 3종 계산
3. **진짜 win rate / Sharpe / MDD 산출**: 위 forward-tracking 결과로 `true_performance.md` 생성
4. **블로거 보고치 vs 실측치 gap 측정**: claimed +29.22% 평균 vs 실측 — 차이가 클수록 신뢰도 낮음
