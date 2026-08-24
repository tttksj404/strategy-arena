# S001 — Baseline (현 기본 config)

- **Status**: bt-only
- **부모**: —
- **변경 변수 (vs 부모)**: (베이스라인 — 변경 변수 없음)
- **TF / Symbol**: 5m, 15m / 현 universe (CLI `--universe` 또는 `UNIVERSE_SYMBOLS` 기준)
- **레버리지**: 기본값 (settings.py 의 LeverageConfig)
- **방향**: 현 config 그대로 (대체로 롱 위주)

## 가설 (정량 예측 포함)

> 베이스라인이므로 가설 없음. **모든 후속 전략의 비교 기준선**.

## 진입 룰

> `quant_binance/strategy/regime.py` + `config.example.json` 기본값.
> 핵심:
> - `compute_predictability_score()` 가 임계값 이상
> - `passes_cost_gate()` 통과
> - 각 코인별 `coin_profiles.py` 게이트 통과
> - 매크로/펀딩 오버레이 게이트 통과 (활성 시)

## 청산 룰

> `policy/execution.py` + `risk/` 의 기본 룰.
> TP/SL/트레일/시간정지는 config 의 `execution.exit_*` 그룹 참조.

## 백테스트 결과

> `/strategy-eval S001` 실행 시 자동 갱신.

### 2차 평가 (2026-04-27, **실데이터 30일** batch_backtest)

**Fixture**: 30일 (2026-03-28 ~ 04-27) BTCUSDT/ETHUSDT/SOLUSDT, 1m+5m+1h+4h klines (Bitget API)
**모드**: batch_backtest (4h holding, 16 bps round-trip cost), 자본 $50 / capacity $125
**결과 파일**: [batch_2026-04-27.json](runs/batch_2026-04-27.json)

#### 전체 메트릭

| 항목 | 값 | 해석 |
|---|---:|---|
| 결정 수 | 1,788 | 1h step × 596 windows × 3 심볼 |
| 진입 수 (cash 제외) | 1,258 | **진입률 70.4%** |
| 승 / 패 | 452 / 806 | **승률 35.9%** |
| **Avg Gross bps/trade** | **−1.16** | **신호 자체 효과 ≈ 0 (random)** |
| Avg Net bps/trade | −17.16 | Cost 16 bps 차감 후 |
| 누적 Net PnL | −21,583 bps | (1,258 거래 × −17.16) |

#### 심볼별 (충격적 — 같은 베이스라인이어도 심볼별 격차 큼)

| 심볼 | 거래 | Gross bps | Net bps | 승률 | 진단 |
|---|---:|---:|---:|---:|---|
| **BTCUSDT** | 455 | **+3.17** | −12.83 | 36.3% | 신호 약간 살아있음. **Cost 줄이면 viable** |
| ETHUSDT | 407 | +0.00 | −16.00 | 35.4% | 순수 noise. 신호 무효 |
| **SOLUSDT** | 396 | **−7.31** | −23.31 | 36.1% | **역신호 (random 보다 나쁨)**. 제외 후보 |

### 1차 평가 (2026-04-27, sample fixture, 참고용)

[replay_2026-04-27.json](runs/replay_2026-04-27.json) — 2 snapshots, replay 모드. 통계 의미 없음. 게이트 동작만 확인 (BTC 진입 +23 bps, ETH 거절 — `FUTURES_OVERHEAT` + `SUPPORT_NOT_CONFIRMED`).

## 페이퍼·실거래 결과

| 기간 | 모드 | 거래 | 승률 | PnL | 평균 슬리피지 | 결과 파일 |
|---|---|---:|---:|---:|---:|---|
| (미실행) | — | — | — | — | — | — |

> 참고: `iCloudDrive/quant_archive/quant_runtime/output/paper-live-shell/` 에 2026-04-21 전후 페이퍼 데이터 있음. 6일 경과 stale 상태. S001 평가에 쓰려면 신규 페이퍼 사이클 필요 (사용자 결정).

## 결론 (다음 전략에 상속할 지식)

> **2차 평가 (실데이터 30일) 기반 정식 결론.**

- **가설 적중 여부**: (베이스라인이므로 가설 없음)

- **의외 발견 1 — 진입 빈도는 충분, 진짜 문제는 expectancy**:
  - 백테스트 30일 1,258 거래 = 일평균 **42거래/일** (3 심볼 합) — 사용자 호소 "진입 안 들어감" 과 모순
  - 따라서 라이브에서의 "진입 안 됨" 은 베이스라인 게이트 외 추가 필터 (소액 자본 사이즈 거절, 매크로/지지 라이브 게이트 등) 의심 — **별도 추적 과제**

- **의외 발견 2 — 베이스라인은 손실 전략**:
  - **승률 35.9%**, Gross 기댓값 거의 0 (−1.16 bps), Net −17 bps/거래
  - 30일 누적 −21,583 bps (≈ −215%) — 통계 평가지만 \$50 자본은 첫 수십 거래 안에 청산
  - **신호가 random 수준**: 진입 빈도 늘려도 손실만 커짐. **진입 게이트 완화 = 잘못된 방향**

- **의외 발견 3 — 심볼별 격차가 큼**:
  - BTC: gross **+3.17 bps** (살아있음, cost 줄이면 viable)
  - ETH: gross 0.00 bps (noise)
  - SOL: gross **−7.31 bps** (역신호)
  - **Universe 축소 (SOL 제외)** 가 단일 변수로 가장 깨끗한 효과 가능

- **다음 후보 (변수 1개 — 우선순위 재정렬, 데이터 기반)**:
  1. **🥇 Universe = BTC 단독** (SOL 제외, ETH 제외 또는 보조): gross 신호 살아있는 심볼만. 즉시 expectancy +
  2. **🥈 Cost 절감 — 시장가 → 지정가 진입**: 16 bps → 8~10 bps 가능. BTC gross +3 가 net +로 전환
  3. **🥉 Holding 4h → 12h 또는 1d**: cost 분모 키움. 다만 forward-return 통계 새로 빌드 필요
  4. ❌ 레버리지 2x → 5x: **expectancy 마이너스 상태에서 leverage 늘리면 손실만 5배**. 절대 금지
  5. ❌ 진입 게이트 완화: **gross 가 zero 인 상태에서 진입 늘리면 손실만 커짐**. 금지

- **\$50 자본·도박성 목표 재조명**:
  - 베이스라인은 \$50 자본 게임에 부적합 (signal 없음 + cost drag). 도박성 풀자고 leverage 올리면 단지 빨리 망함
  - 진짜 답: signal 자체를 다른 셋업(breakout / squeeze / liquidation reversal / funding flip)으로 갈아엎는 게 변수 1개 룰 로 안 풀림 → **별도 baseline (S100~ 시리즈?) 으로 분리** 검토

## 변경 이력

| 일자 | 변경 | 사유 |
|---|---|---|
| 2026-04-27 | 초기 작성 (Strategy Registry 도입) | 전략 카탈로그 시스템 시작점 |
| 2026-04-27 | 1차 평가 (sample fixture, 2 snapshots) | 베이스라인 첫 데이터 — replay 모드 한계 확인 |
| 2026-04-27 | **2차 평가 (실데이터 30일 batch_backtest)** | **베이스라인 expectancy 측정 — 신호 random 수준, cost 못 이김. 정식 결론 작성** |
