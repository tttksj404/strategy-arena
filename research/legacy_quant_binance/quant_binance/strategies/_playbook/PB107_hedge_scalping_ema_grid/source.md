# PB107 — EMA 5-Grid Hedge Scalping (nikita-doronin)

## 출처

- **Repo**: https://github.com/nikita-doronin/crypto_hedge_scalping_bot
- **별 / 포크**: 31 ⭐ / 6 fork (게이트 28+ 통과)
- **언어**: Python 60% + Jupyter 40%
- **라이선스**: CC BY-NC (비상업 OK)
- **마지막 업데이트**: 활성 (2024~2025 commits)
- **TradingView Pine Script port** 별도 제공 (룰 즉시 검증 가능)

## 한 줄 셋업

EMA(N) 기반 **5-grid hedge mode**: long limit 4단 + short market 1단을 동시 깔고 단기 풀백 수익 회수.
**Bitget hedge mode 양방향 native 지원**과 룰 구조가 일치 (PB101 Passivbot과 유사 계열, but **단기 scalping** 지향).

## 신뢰도 평가 (1~5)

| 항목 | 점수 | 근거 |
|---|---:|---|
| 다년 라이브 + 공개 손익 | 2 | 별도 라이브 PnL 공개 X |
| 1년+ 라이브 / 공개 백테스트 / 활발 유지보수 | 3 | 백테스트 인프라 + 활성 commit |
| 백테스트 공개 + OSS 검증 가능 | 4 | tick-level historical backtest + Pine Script 교차 검증 |
| 출처 신뢰성 | 3 | 31⭐, 코드 100% 공개, JSON config 룰 명시 |
| **종합** | **3.0** | **G-전략 발급 게이트 통과 (기준선)** |

## 6축 fit (사용자 컨텍스트)

| 축 | 기준 | PB107 적합 |
|---|---|---|
| $55 자본 | OK (config로 size 조정) | ✓ |
| Lottery / 거래당 큰 수익 | grid 4단 풀백 + short market = 구조적 lottery 아님, 회수형 | △ (lottery는 아니지만 빈도형) |
| 5-10x lev | config `leverage` 조정 가능 | ✓ |
| 양방향 (롱·숏) | **hedge mode 명시** (long limit 4 + short market 1) | ✓✓ |
| 단기 (intraday) | EMA + grid % 기반 풀백 = 분/시간 단위 | ✓ |
| ≥3건/일 | 5-grid 구조상 빈도 高 | ✓ |

**5/6 충족** (lottery 축만 부분 미충족 — 회수형이지만 단기·양방향·고빈도로 보완)

## 9-point 사전 가능성 (production candidate 9/9 게이트 사전 점검)

1. **룰 완전 공개**: ✓ (Python + Pine Script 동등 구현)
2. **재현 가능 backtest**: ✓ (Binance tick data + 코드)
3. **Live infra 존재**: ✓ (Telegram + SQL DB)
4. **JSON config**: ✓ (timeframe, EMA len, grid %, lev, TP)
5. **Bitget 호환**: △ (Binance Futures 기반, ccxt 또는 직접 어댑터 필요)
6. **Hedge mode (양방향)**: ✓ (Bitget hedge mode native 지원)
7. **scalping 시간 지평**: ✓
8. **Stop logic**: △ (코드 점검 필요 — TP 명시, SL 별도 검증)
9. **검증 자료**: △ (저자 제공 라이브 PnL 부재 — 자체 paper-live 6주 필수)

**6.0/9.0 사전 통과** — paper-live 검증 후 G-시리즈 발급 가능.

## 사용자 G135 와의 보완 관계

- G135: regime + score 76 + ETH/SOL/DOGE/PEPE — **방향성 진입 + score 게이팅** 패턴
- PB107: **양방향 grid hedge** — G135 의 단방향 한계 보완. 횡보 regime 에서 G135 disable 시 hedge grid 가 풀백 수익 회수
- **레이어드 운용 후보**: G135 = 추세 / PB107 = 횡보·풀백 (regime 분기 결합)

## 도입 cost / 위험

- Cost: ccxt 어댑터 1~2일 + JSON config 튜닝 + paper 6주 (총 ~7주)
- 위험: 별 31개로 작은 커뮤니티 — 버그·엣지케이스 자체 검증 필수
- 위험: hedge mode grid 는 강한 추세장에서 한쪽이 무한 손실 → S005·G135 처럼 regime 게이트 결합 필수

## 다음 단계

1. repo clone + 룰 정독 → `rules.md` 작성
2. Bitget USDT-perp ccxt 어댑터 작성 (hedge mode + grid limit)
3. 4주 paper-live 백테스트 (BTC + ETH + SOL)
4. paper PnL +5% 이상 + MDD <20% 달성 시 G201 발급 후보
