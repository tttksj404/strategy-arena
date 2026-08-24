# PB100 — NostalgiaForInfinity (NFI) — freqtrade 커뮤니티 표준 알트 단기

## 출처

- repo: https://github.com/iterativv/NostalgiaForInfinity
- 라이선스: GPL-3.0
- ⭐ 3.2k / fork 712 / watch 91
- 최신 버전: **v17.3.1079** (2026-04-09)
- 총 515 릴리스 (활발한 유지보수)
- freqtrade 호환 (별도 봇 프레임워크 없이 freqtrade strategy 파일 1개 형태로 배포)

## 셋업 한 줄

다중 진입 시그널 (수십 종) + ROI 계단형 청산 + Trailing stop + 페어별 동적 stoploss 를 가진 **알트코인 5분봉 long-only 봇**. 수년간 freqtrade 커뮤니티에서 가장 많이 운용된 전략 중 하나.

## 주요 운용 파라미터 (NFI 권장 → 우리 환경 매핑)

| NFI 권장 | 값 | 우리 환경 (Bitget perp) 매핑 |
|---|---|---|
| Timeframe | 5m (필수) | 그대로 사용 가능 |
| Pair quote | USDT, USDC | Bitget USDT-perp ◎ |
| Pair count | 40~80개 | 알트 USDT-perp top 50 |
| 거래소 권장 | Binance, Bitget, Bybit, Kucoin, Gate, OKX, MEXC, HTX, BitMart | **Bitget 명시 지원 ✓** (사용자 보유 API) |
| 레버리지 토큰 (BULL/BEAR) | 블랙리스트 | 적용 |
| use_exit_signal | true (필수) | execution.py 에 신호 hook 필요 |
| exit_profit_only | false (필수) | 동일 |
| ignore_roi_if_entry_signal | true (필수) | freqtrade-only 옵션 → 우리 ROI 룰 재해석 필요 |

## 신뢰도 평가

| 항목 | 점수 (0~5) | 근거 |
|---|---:|---|
| 라이브 운용 기간 | 5 | 2018+ 부터 활발한 운용. 515 릴리스 |
| 공개 손익 | 3 | 개별 commit 댓글에 백테스트 결과 산재. README 에는 미통합 |
| 커뮤니티 검증 | 5 | freqtrade discord 사실상 표준. fork 712, watch 91 |
| 룰 명시도 | 5 | 코드 자체가 룰 (Python 단일 파일) → 직접 읽고 포팅 가능 |
| 백테스트 재현성 | 5 | freqtrade 설치 + 백필 데이터로 누구나 재현 |
| **종합** | **4.6** | **G-전략 발급 게이트 통과 (최상위)** |

## 우리 시스템 포팅 옵션

**옵션 A — freqtrade 별도 설치, 그대로 운용** (시간 ↓, 학습 효과 ↑)
- `pipx install freqtrade` → NFI strategy 파일 download → `freqtrade backtesting --strategy NostalgiaForInfinityX --timerange 20260301-`
- pro: 즉시 실측 백테스트 결과 확인 가능 (Bitget 백필 freqtrade 가 알아서 처리)
- con: quant_binance 와 별도 런타임. Strategy Registry 와 결과 통합 별도 작업 필요

**옵션 B — quant_binance 로 룰 포팅** (시간 ↑↑, 통합 ↑↑)
- NFI strategy.py 의 `populate_entry_trend` / `populate_exit_trend` / `custom_stoploss` → quant_binance/strategy/regime.py override 로 변환
- pro: G-시리즈 정식 등록 가능. Strategy Registry 일관 관리
- con: 시간 (10+ 시간) + NFI 의 freqtrade-specific 기능 (커스텀 indicator, dataprovider) 호환 작업

**옵션 C — 룰 추출 후 vectorbt 로 grid search** (시간 중간, 학습 ↑↑)
- NFI 의 진입 시그널만 룰 단위로 추출 → vectorbt 로 100심볼×180일 일괄 평가
- pro: 어떤 시그널이 진짜 알파인지 ablation
- con: ROI/stoploss 동적 부분 손실

**권장: A → C → B 순.** A 로 NFI 가 우리 환경에서 정말 양수인지 1시간 내 확인 → 양수면 C 로 알파 분해 → 그 결과로 B 의 어떤 부분만 포팅할지 결정.

## 알려진 risk / 한계

- **Long-only**: 사용자 자본 컨텍스트 ($50, 도박성 OK, both 권장) 와 부분 mismatch. NFI 자체는 short 미지원
- **현물 지향**: NFI 는 원래 현물 봇. perp/leverage 로 옮기면 funding cost 추가
- **5m 만 권장**: 더 짧은 timeframe (1m) 비추천 → lottery 빈도 한계
- **GPL-3.0**: 우리 코드에 룰을 가져오면 derivative work 가능성. 사적 사용은 무관, 공개 재배포 시 GPL 준수 필요

## 다음 작업

1. **옵션 A 즉시 실행**: 별도 가상환경에 freqtrade 설치 → NFI strategy + 30일 Bitget USDT-perp top 50 페어 백테스트 → 결과 PnL/win rate/MDD/Sharpe 보고
2. 결과 양수면 → `claimed_performance.md` 작성 (NFI 우리 환경 실측치)
3. 양수면 → G002 발급 후보 (`/strategy-new G002 --base S001 --playbook PB100`)
