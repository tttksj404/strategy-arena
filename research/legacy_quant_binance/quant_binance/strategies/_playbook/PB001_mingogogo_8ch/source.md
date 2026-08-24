# PB001 — Mingogogo "민지네 컨트롤타워 v3.0" 8채널 컨센서스

## 출처

- 블로그/오픈톡 운영자: Mingogogo (paming, 카카오 오픈톡)
- 크롤 데이터 위치: `quant_binance/data/external/mingogogo/raw/` (460MB, 214+ posts, 2025-11 ~ 2026-04 라이브 추천 누적)
- 핵심 산출물 파일:
  - `coin_program_inferred_algorithm_blueprint.json` — 8단계 파이프라인 역추적
  - `coin_program_target_posts_algorithm_clues.csv` (495 lines) — 알고리즘 단서 라벨링
  - `coin_program_recommendation_with_rationale.csv` — entry/exit/3일 수익률 (백테스트 +28~+115% 샘플)
  - `coin_program_reco_rationale_algorithm_report.md` (12.4KB) — 정제 요약
  - `posts.json` (6.0MB) — raw 포스트 전수
- handoff 2026-04-27 §2 의 "🥇 최고 가치 자산" 으로 식별

## 셋업 한 줄

8개 기술분석 채널 가중 합산 점수 (10~173) → 임계 통과 시 진입 → 멀티-TF (5m/15m/1h/4h) 동기화 + 하모닉 패턴 + Z-score reversal + 세력 매집 검증 → 알트코인 / 업비트 / 3-day horizon

## 인디케이터 가중치 (실측 추정)

| 인디케이터 | 가중치 |
|---|---:|
| RSI | 15% |
| MFI | 12% |
| Stochastic | 12% |
| BB%B | 10% |
| ATR | 7% |
| MACD | 추정 ~12% |
| OBV | 추정 ~10% |
| ADX | 추정 ~10% |
| 합계 | 100% (잔여 ~12% 미식별) |

> 정확한 가중치 + 임계값은 PB001 본격 마이닝 (subagent) 시 `coin_program_inferred_algorithm_blueprint.json` 에서 추출.

## 점수 분포 / 임계

- 점수 범위: 10 ~ 173
- 강력추천 임계: ~150+ (실제 적중 사례: 173점 = 단일 종목 +14% lottery)
- 일반 추천: 100~150 (분할 매수 안내 동반)

## 타깃 / 핏

- **거래소**: 업비트 (원본). Bitget/Binance USDT-perp 으로 포팅 시 페어 매핑 필요
- **자산**: 알트코인 only (BTC/ETH 비중 매우 낮음)
- **호라이즌**: 3-day (단기), 분할 매수 + 손절가 권장
- **사용자 자본 컨텍스트 핏**: $50 도박성·lottery 추구와 ◎ — 추천 종목당 +14% 단일 lottery 사례 다수

## 신뢰도 평가

| 항목 | 점수 (0~5) | 근거 |
|---|---:|---|
| 라이브 운용 기간 | 4 | 2025-11 ~ 2026-04 (5개월+) 일일 라이브 추천 누적, posts.json 6MB |
| 공개 손익 | 3 | 추천 + 3일 수익률 csv 공개. 단 cherry-pick 가능성 미검증 |
| 커뮤니티 검증 | 2 | 카카오 오픈톡 비공개 → 외부 검증 어려움 |
| 룰 명시도 | 4 | 8채널 + 가중치 + 점수 임계 모두 라벨링됨 |
| 백테스트 재현성 | 3 | 데이터셋 정제 완료 (`*_clean.csv`) → 재실행 가능 |
| **종합** | **3.2** | **G-전략 발급 게이트 (≥3.0) 통과** |

## 알려진 risk / 한계

- 원본 거래소가 업비트라 Bitget/Binance perp 로 옮길 시 유동성·페어 매핑 차이
- 점수 173 = 사후 적중 강조 가능성 (생존자 편향)
- 8채널 중 정확한 1~2개는 미식별 (마이닝으로 채워야)
- "세력 매집" 신호의 정량 정의 불명확 → OBV·매수세 강도로 근사 필요

## 다음 작업 (subagent 위임)

1. `posts.json` 6MB 마이닝 → 일자별 추천 + 실제 가격 추적 → 진짜 win rate 계산
2. `coin_program_inferred_algorithm_blueprint.json` → 정확한 8채널 식별 + 가중치 → `rules.md`
3. `coin_program_recommendation_with_rationale.csv` 의 entry/exit/3일 수익률 → 분포 통계 → `claimed_performance.md`
4. quant_binance pandas-ta 로 8채널 재구현 → `implementation_notes.md`
