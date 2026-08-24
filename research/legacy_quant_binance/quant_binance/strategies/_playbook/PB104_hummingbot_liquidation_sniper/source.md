# PB104 — Hummingbot V2 Liquidation Sniper Controller

## 출처

- repo: https://github.com/hummingbot/hummingbot (Hummingbot 본체, ⭐7k+, MIT)
- 공식 블로그: https://hummingbot.org/blog/coding-a-liquidation-sniper-v2-strategy-controller/
- 코드: `liquidations_sniper.py` (V2 strategy controller)
- 최신: v2.5.0 (2025), production-grade

## 셋업 한 줄

**Binance 청산 피드 실시간 모니터링** → 임계 청산량 발생 시 단방향 DCA 진입 (default 5x lev, TP 1% / SL 3% / 30분 timeout). "**liquidation cascade reversal**" 셋업.

## 동작 원리

1. Binance public liquidation WebSocket subscribe
2. 임계 청산량 (예: ETH $1M+ in 5min) 발생 감지
3. 청산 방향 inverse 진입 (LONG cascade → reversal SHORT, SHORT cascade → reversal LONG)
4. DCA: 가격 추가 -X% 이동 시 추가 entry
5. Exit: TP +1% / SL -3% / 30min timeout 중 first hit

## 사용자 6축 fit ⭐ 6/6

| 축 | fit |
|---|---|
| $55 자본 | ✓ DCA 사이즈 자유 조정 |
| lottery | ✓ cascade 후 강한 reversal 노림 |
| 5-10x lev | ✓ default 5x, 10x 까지 확장 |
| 양방향 | ✓ LONG/SHORT 양 트리거 |
| 단기 | ✓ 30분 timeout |
| ≥3건/일 | ✓ BTC/ETH 변동장 일평균 5-15회 트리거 |

## 신뢰도 평가

| 항목 | 점수 (0~5) | 근거 |
|---|---:|---|
| 라이브 운용 기간 | 4 | Hummingbot 다년 production. v2.5.0 안정 |
| 공개 손익 | 3 | 백테스트 수치 미공개. 단 production 사용자 多 |
| 커뮤니티 검증 | 5 | Hummingbot Foundation 공식 게재 |
| 룰 명시도 | 5 | 코드 100% 공개 (Python) |
| 백테스트 재현성 | 4 | 자체 청산 데이터 fetch 후 재현 가능 |
| **종합** | **4.2** | **G-전략 발급 게이트 통과 (상위)** |

## 사용자 G135 와의 관계 (직교 알파)

- G135 = score 76 + ETH/SOL/DOGE/PEPE 기반 진입 score gate (regime engine)
- PB104 = 외부 이벤트 (청산 cascade) 트리거 + 5x lev DCA reversal
- **두 신호 발생 조건 다름** → 동시 운용 시 알파 가산 가능
- 권장: G135 main + PB104 overlay (capital 70/30 분배)

## 도입 비용 / 위험

**도입 cost**: 2-3일
- Hummingbot 별도 환경 설치 (Docker 권장)
- Binance 청산 WS 통합 (이미 Hummingbot 내장)
- Bitget 거래 어댑터 brick 작성 (KuCoin 디폴트라 ⚠️)
- 자체 백테스트 (3-6개월 청산 데이터 재처리)

**위험**:
- 청산 피드 = **Binance 한정**, Bitget 자체 청산 데이터 별도 통합 필요
- KuCoin 디폴트 → Bitget 어댑터 신선도 검증 필수
- Cascade reversal 가설 → 약세 추세 지속 시 반복 손실 (DCA 위험)

## 다음 작업

1. Hummingbot V2 환경 설치 (별도 venv)
2. Binance liquidation WS subscribe 검증
3. Bitget 거래 어댑터 brick 작성 또는 ccxt bridge
4. **자체 60일 청산 데이터 백테스트** (자체 OOS 검증 — 9-point 일부 충족)
5. PB104-S001 strategy ID 발급 후 paper-live
