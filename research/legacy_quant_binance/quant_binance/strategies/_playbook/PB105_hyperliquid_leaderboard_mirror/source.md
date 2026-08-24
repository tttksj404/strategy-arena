# PB105 — Hyperliquid Top Trader Leaderboard 미러링

## 출처

- Hyperliquid 공식 leaderboard: 모든 포지션 100% 온체인 verifiable
- Coinglass Whale Tracker: https://www.coinglass.com/hyperliquid
- API: Hyperliquid Info API (/info endpoint)

## 셋업 한 줄

Hyperliquid 상위 트레이더 (예: The White Whale +$50M/30일, Machi 92.11% WR / 76 trades) **온체인 포지션 변화 실시간 미러링** → Bitget perp 에 같은 페어로 비례 사이징 진입.

## 동작 원리

1. Hyperliquid Info API → top N traders user state polling (5초 간격)
2. 새 포지션 open / 기존 포지션 변동 감지
3. 동일 심볼 / 동일 방향 / 비례 사이징으로 Bitget 미러
4. Top trader 청산 시 우리도 청산
5. **Ensemble**: 3+ top trader 동일 페어 confirm 시만 진입 (alpha decay 방어)

## 사용자 6축 fit ⭐ 6/6

| 축 | fit |
|---|---|
| $55 | ✓ 비례 사이징 자유 |
| lottery | ✓ top trader 5-30x lev 양방향 |
| 5-10x | ✓ |
| 양방향 | ✓ long/short 모두 트리거 |
| 단기 | ✓ top trader 평균 단타 |
| ≥3건/일 | ✓ top trader 일평균 10+회 |

## 신뢰도 평가

| 항목 | 점수 | 근거 |
|---|---:|---|
| 라이브 운용 기간 | 5 | Top trader 6개월+ verified (온체인 raw) |
| 공개 손익 | 5 | 100% 온체인 공개, P&L verifiable |
| 커뮤니티 검증 | 4 | Coinglass / DeBank tracking |
| 룰 명시도 | 2 | top trader 의 룰은 black box (포지션만 공개) |
| 백테스트 재현성 | 2 | 미러링은 백테스트 어려움 (동일 latency 재현 X) |
| **종합** | **3.6** | **G-전략 발급 게이트 통과** |

## 사용자 G135 와의 관계

- G135 = ETH/SOL/DOGE/PEPE 자체 score gate
- PB105 = 외부 알파 (top trader 의 의사결정) 도입
- 결합 가능: G135 score 70+ AND top trader 동일 방향 진입 → 이중 confirm
- 또는 단독: ensemble 3+ 미러만으로

## 도입 비용 / 위험

**도입 cost**: 5-7일
- Hyperliquid Info API 통합 (rate-limited, 무료)
- Top trader 식별 + ensemble 로직
- Bitget 주문 미러러 + 사이징 비례화
- Latency 5초 이내 검증

**위험**:
- **Alpha decay**: 한 트레이더 wipeout 시 미러 loss 동시 발생
- **Latency arbitrage**: 우리 미러가 top trader 보다 5초 늦으면 entry edge 소실
- **Hyperliquid → Bitget price gap**: 같은 가격 진입 불가능, 슬리피지 발생
- **Top trader 익명성**: 내부자 정보 가능성 (불법 X 단 alpha 안정성 미보장)

## 다음 작업

1. Hyperliquid Info API integration (Python urllib)
2. Top 10 trader 6개월 leaderboard scrape + ensemble selection
3. Mirror engine PoC (paper mode)
4. 30일 paper-live 실험 (사용자 G135 와 별개 ID)
5. Latency / slippage 측정 보고
