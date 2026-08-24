# PB102 — Jesse — 양방향·레버리지 first-class 지원 Python 백테스트·라이브 프레임워크

## 출처

- repo: https://github.com/jesse-ai/jesse
- 공식 사이트: https://jesse.trade/
- 라이선스: MIT
- ⭐ 7.8k / fork 1.1k / watch 204
- 언어: Python (전략 작성) + JavaScript (대시보드)
- 활발한 유지보수 (2026년 4월 현재 활동 중, 3,200+ commits)
- 커뮤니티: 공식 Discord (jesse.trade/discord) + jesse.trade 공식 사이트

## 셋업 한 줄

**"first-class support for leveraged trading and short-selling"** 을 명시하는 Python 백테스트·라이브 프레임워크. **단일 전략이 아니라 프레임워크** — 300+ indicator + multi-symbol/timeframe + zero look-ahead bias 백테스트 + 페이퍼·라이브 단일 코드. 우리 자체 short 전략 개발의 토대로 활용 가능.

## 주요 운용 파라미터 (Jesse 권장 → 우리 환경 매핑)

| Jesse 권장 | 값 | 우리 환경 (Bitget perp) 매핑 |
|---|---|---|
| 시장 타입 | spot + futures (양방향) | Bitget USDT-perp 호환 (CCXT 어댑터 또는 자체 어댑터) |
| 거래소 어댑터 | Binance / Binance Futures / Bybit / FTX 호환 등 다수 | Bitget 직접 어댑터 미확인 → CCXT bridge 필요 가능 |
| 백테스트 정확도 | 룩어헤드 bias 제거 (zero look-ahead) | 우리 vectorbt 벤치마크 대비 신뢰도 검증 가치 ↑ |
| 청산 옵션 | partial fills + 다중 TP/SL 단계 | $50 lottery 컨텍스트에 부분 청산 유리 |
| 알림 | Telegram / Slack / Discord 실시간 | 우리 paper-live-shell 알림 통합 가능 |
| 백테스트 + 라이브 | 동일 코드 양쪽 실행 (코드 분기 X) | quant_binance 핵심 원칙과 일치 |

## 신뢰도 평가

| 항목 | 점수 (0~5) | 근거 |
|---|---:|---|
| 라이브 운용 기간 | 4 | 2019+ 부터 활발한 프레임워크. 라이브 운용 사례 다수 (jesse.trade 사이트 testimonial) |
| 공개 손익 | 2 | 프레임워크 자체는 손익 미공개 (전략 비공개가 정상). 커뮤니티 사례만 산재 |
| 커뮤니티 검증 | 5 | ⭐7.8k / fork 1.1k / watch 204. Python crypto 백테스트 프레임워크 중 최상위 |
| 룰 명시도 | 4 | 프레임워크 코드 명시. **단, 전략은 사용자가 작성** — 우리가 short 룰 직접 발명 필요 |
| 백테스트 재현성 | 5 | zero look-ahead + multi-symbol 검증된 엔진. CCXT bridge 통해 Bitget 백필 가능 |
| **종합** | **4.0** | **G-전략 발급 게이트 통과 (프레임워크로서)**, **단일 전략 PB가 아님 — 자체 short 전략 검증 인프라로 채택** |

## 우리 시스템 포팅 옵션

**옵션 A — Jesse 별도 설치, 우리 short 가설 검증 인프라로 활용** (시간 ↓, 학습 효과 ↑↑)
- `pip install jesse` → Bitget USDT-perp 백필 (CCXT) → 우리 자체 short 가설 (예: liquidation reversal short, funding flip short) 을 Jesse 전략 파일로 작성 → 빠른 백테스트
- pro: vectorbt 와 다른 엔진으로 cross-check 가능 (zero look-ahead 신뢰도 검증)
- con: 전략 자체는 우리가 작성 — Jesse 는 도구 제공만 (PB100/PB101 처럼 "기성 전략" 아님)

**옵션 B — Jesse 의 "전략 community examples" 마이닝** (시간 중간, 학습 효과 중간)
- jesse.trade 커뮤니티에 공유된 short/futures 전략 검색 → 검증된 것만 우리 G-전략 후보로
- pro: 사용자 검증된 전략 발굴 가능
- con: 커뮤니티 전략은 신뢰도 낮음 (공개 ≠ 검증). PB 신뢰도 ≥3 게이트 통과 어려움

**옵션 C — Jesse 의 백테스트 엔진 코드 일부 채택 (zero look-ahead 검증 로직)** (시간 ↑↑)
- Jesse 의 candle-by-candle simulator 의 look-ahead 방지 로직 → quant_binance/runtime 의 replay mode 강화
- pro: 우리 백테스트 신뢰도 향상
- con: 시간 ↑↑. 우리 엔진 재작성 정도 부담

**권장: A 만 채택.** Jesse 는 **기성 전략 PB 가 아니라 우리 short 가설을 빠르게 검증하는 인프라**. PB100(NFI 기성 전략) + PB101(Passivbot 기성 봇) + PB102(Jesse 검증 프레임워크) 3종 보완 구조.

## 알려진 risk / 한계

- **단일 전략 아님**: PB100/PB101 처럼 "이 룰 그대로 쓰면 됨"이 아님. 우리가 short 전략 룰을 직접 발명·검증 필요
- **Bitget 직접 어댑터 미확인**: CCXT bridge 또는 자체 어댑터 작성 필요할 수 있음. 검증 필수
- **MIT 라이선스**: 우리 코드 통합·재배포 자유 ◎
- **JesseGPT / 클라우드 유료 옵션**: 무료 OSS 부분만 사용 가능. 유료 기능 없이도 라이브 가능
- **Python only**: Rust 백테스트 가속 없음 (passivbot 대비 백테스트 속도 ↓ 가능)

## 다음 작업

1. **옵션 A 부분 실행**: `pip install jesse` → Bitget USDT-perp 백필 가능성 확인 (CCXT bridge or 자체 어댑터) → 1심볼·30일 toy 백테스트로 vectorbt 와 결과 비교
2. 우리 자체 short 가설 (예: "8h 평균 funding > +0.05% 시 단기 short", "대규모 long liquidation 발생 후 30분 mean reversion short") → Jesse 에서 빠르게 prototype 백테스트
3. 양수 가설 발견 → `rules.md` 작성 후 G004+ 발급 후보. **PB102 자체는 G 전략 발급 X — 도구로만 활용**
