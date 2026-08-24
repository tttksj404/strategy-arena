# PB101 — Passivbot — 다중거래소 perp 양방향 grid + trailing 봇

## 출처

- repo: https://github.com/enarjord/passivbot
- 라이선스: Unlicense (퍼블릭 도메인, 사실상 제약 없음)
- ⭐ 2.0k / fork 646 / watch 약 90+
- 최신 버전: **v7.10.0** (2026-04-25)
- 언어: Python 85.5% + Rust 13.4% (백테스트·라이브 공유 Rust 오케스트레이터)
- 활발한 유지보수 (2026년 4월 현재 활동 중)
- 커뮤니티: Discord + Telegram 공식 채널

## 셋업 한 줄

**다중거래소 perp 시장에서 grid 진입 + trailing 청산 + Martingale 성격의 unstucking 메커니즘**으로 양방향 포지션을 자동 관리하는 contrarian market maker. Bitget·Binance·Bybit·OKX·Gate·Kucoin·Hyperliquid 네이티브 지원, **Bitget USDT-perp 우리 환경 그대로 운용 가능**.

## 주요 운용 파라미터 (Passivbot 권장 → 우리 환경 매핑)

| Passivbot 권장 | 값 | 우리 환경 (Bitget perp) 매핑 |
|---|---|---|
| 거래소 | Bybit, Bitget, OKX, GateIO, Binance, Kucoin, Hyperliquid | **Bitget 네이티브 ✓** (사용자 보유 API) |
| 시장 타입 | USDT/USDC perpetual futures | Bitget USDT-perp ◎ |
| 포지션 방향 | long + short 동시 운용 가능 | **양방향 ✓** (gap 충족) |
| 진입 방식 | Grid (다층 limit) + trailing entry | $50 자본에서는 grid 폭 좁게 (1~2단) 권장 |
| 청산 방식 | Trailing TP + unstucking (martingale doubling) | 5~10x 레버리지에 unstucking 위험 ↑↑ |
| Forager 모드 | 변동성 큰 시장 자동 선택 | 알트 단기 (3-day) 컨텍스트와 정합 |
| 백테스트 | 빌트인 + evolutionary optimizer | Bitget 백필 자동 처리 |

## 신뢰도 평가

| 항목 | 점수 (0~5) | 근거 |
|---|---:|---|
| 라이브 운용 기간 | 5 | 2018+ 부터 활발한 운용. Discord/Telegram 운용자 다수 |
| 공개 손익 | 3 | Discord·Telegram 채널에 운용자 손익 공유 산재. README 통합 미흡 |
| 커뮤니티 검증 | 4 | ⭐2k / fork 646 / 7개 거래소 네이티브. perp 봇 중 상위권 |
| 룰 명시도 | 5 | 코드 자체가 룰 (Python+Rust). config JSON 으로 파라미터 노출 |
| 백테스트 재현성 | 5 | 빌트인 백테스트 + evolutionary optimizer. Bitget 백필 자동 |
| **종합** | **4.4** | **G-전략 발급 게이트 통과 (양방향 gap 충족 최우선)** |

## 우리 시스템 포팅 옵션

**옵션 A — Passivbot 별도 설치, Bitget API 직결 운용** (시간 ↓, 학습 효과 중간)
- `git clone` → config JSON 작성 (Bitget API 키 + symbol 리스트) → `passivbot run config.json`
- pro: 즉시 실측 가능. 별도 코드 작성 X. unstucking 메커니즘 검증 가능
- con: $50 자본에 grid 다층 진입 시 마진콜 위험 ↑. 초기 grid_span 매우 좁게 필수

**옵션 B — passivbot grid+trailing 룰만 quant_binance 로 포팅** (시간 ↑↑, 통합 ↑↑)
- Rust 오케스트레이터의 entry/exit 로직 → `quant_binance/strategy/regime.py` 변형 override
- pro: G-시리즈 정식 등록. unstucking 룰 우리 컨텍스트에 맞게 단순화 가능 ($50 환경)
- con: Rust 코드 독해 + Python 포팅 시간 (15+ 시간)

**옵션 C — Forager 알고리즘만 추출 (변동성 큰 시장 자동 선택)** (시간 중간, 부분 채택)
- Forager 의 symbol selection logic → 우리 universe filter 로 채택
- pro: 진입 빈도 ≥3건/일 목표 달성에 직접 기여
- con: passivbot 의 핵심 알파(grid+unstucking) 미채택 → 부분적 활용

**권장: A → C → B 순.** A 로 small notional($5~10) 페이퍼/마이크로 라이브 1주 → unstucking 발동 빈도·MDD 측정 → 양수면 C 로 forager 채택, B 는 마지막에 핵심 알파만 선별.

## 알려진 risk / 한계

- **Martingale 성격 (unstucking)**: 손실 포지션에 doubling down. 5~10x 레버리지 + $50 자본에서는 1회 unstucking 발동으로 청산 위험. **반드시 max_n_unstuck=1 또는 0 으로 운용**
- **Grid 다층 진입**: 좁은 grid_span 필수 ($50 자본). 권장 기본값 그대로 쓰면 마진콜
- **Funding cost 무시**: passivbot 자체는 funding rate 회피 로직 없음. 우리 G-전략 포팅 시 funding 필터 추가 권장
- **Unlicense (퍼블릭 도메인)**: 라이선스 자유 ◎. 우리 코드 통합·재배포 무제한
- **3-day horizon vs grid 봇 본질**: passivbot 은 무기한 운용 가정. 3-day 단기 lottery 컨텍스트와 부분 mismatch → trailing TP threshold 짧게 조정 필수

## 다음 작업

1. **옵션 A 즉시 실행**: 별도 venv 에 passivbot v7.10.0 설치 → Bitget testnet/소액 → 알트 perp 5~10심볼 1주 paper run → unstucking 발동 횟수·실효 PnL 측정
2. 결과 양수 + unstucking 통제 가능하면 → `claimed_performance.md` (우리 환경 실측치) + `rules.md` (config 파라미터 + grid/trailing/unstucking 룰 추출)
3. → G003 발급 후보 (`/strategy-new G003 --base S001 --playbook PB101`). **양방향 gap 충족 최우선 후보**
