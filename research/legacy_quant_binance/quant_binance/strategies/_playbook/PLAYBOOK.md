# Playbook Master Index

> 새 G-시리즈 전략은 **반드시 PB(Playbook) 항목 인용** 필수.
> handoff 2026-04-27 §1 의 메타 깨달음: "이론·통설 기반 룰 발명 = 매번 random. 검증된 외부 셋업부터 가져와야 함".
> 각 PB 는 **출처·룰·주장 성과·구현 노트** 4개 파일로 구성.

## 스키마

```
_playbook/<PB_ID>_<slug>/
├── source.md              # 출처 (URL, 논문, 블로그, OSS repo) + 신뢰도 평가 (1~5)
├── rules.md               # 진입·청산·필터링 룰 (코드화 가능 형식)
├── claimed_performance.md # 원작자가 주장한 백테스트/라이브 성과 (수치 + 기간)
└── implementation_notes.md # 우리 시스템(quant_binance) 으로 포팅 시 주의/변환점
```

## ID 체계

- **PB001~PB099**: 사용자 자체 발견 자산 (Mingogogo, Naver gten 등)
- **PB100~PB199**: 검증된 OSS / 커뮤니티 (freqtrade-strategies, CryptoSignal 등)
- **PB200~PB299**: 학술 논문 / 백서 출처
- **PB900+**: 폐기/검증 실패 (참고용 보존)

## 신뢰도 평가 기준 (각 PB source.md 의 "신뢰도 평가" 섹션)

| 점수 | 기준 |
|---|---|
| 5 | 다년간 라이브 운용 + 공개 손익 + 커뮤니티 검증 |
| 4 | 1년 이상 라이브 + 공개 백테스트 + 활발한 유지보수 |
| 3 | 백테스트만 공개 + 활발한 OSS 또는 검증 가능 출처 |
| 2 | 출처 있으나 성과 검증 어려움 |
| 1 | 추측/통설 — G-시리즈 발급 불허 |

**G-전략 발급 게이트**: PB 신뢰도 종합 ≥ 3.0 필수.

## 등록된 PB 목록

| PB ID | 셋업 한 줄 | 신뢰도 | 카테고리 | 상태 |
|---|---|---:|---|---|
| [PB001](PB001_mingogogo_8ch/source.md) | 10-지표 가중 CH1 + 멀티TF/하모닉/Z-rev (Mingogogo v3.0) | 3.2 (claimed perf 1.5/5 cherry-picked) | 알트 단기, **bull-regime only** ⚠️ | **검증 결과: alpha 존재하나 regime 의존 (Q1+361/Q2-26/Q3+487/Q4-21). Portfolio 시뮬 -28.8%. paper-live 必필** |
| [PB100](PB100_freqtrade_nfi/source.md) | NostalgiaForInfinity (freqtrade 커뮤니티 표준) | 4.6 | 알트 5m long | source 작성 완료, rules 추출 대기 |
| [PB101](PB101_passivbot/source.md) | Passivbot — Bitget 네이티브 양방향 grid+trailing+unstucking | 4.4 | 알트 perp 양방향 | source 작성 완료, paper 검증 대기 |
| [PB102](PB102_jesse_framework/source.md) | Jesse — short-selling first-class 백테스트·라이브 프레임워크 | 4.0 | 인프라 (도구) | source 작성 완료, 자체 short 가설 검증 토대 |
| [PB103](PB103_funding_rate_arb/source.md) | aoki-h-jp funding-rate-arb — 펀딩 극단치 lottery 시그널 차용 | 3.0 | 알트 펀딩 양방향 | source 작성 완료, 펀딩 극단 시그널 자체 백테스트 필요 |
| [PB104](PB104_hummingbot_liquidation_sniper/source.md) | Hummingbot Liquidation Sniper | **dead (4/9)** | — | ⚠️ **PoC 결과: 91 trades WR 9.9% avg -35.6bps 모든 시기 음수. Binance forceOrders 엔드포인트 maintenance + 메이저 자산 30분 vol 부족 (TP +1% 도달 <5%). 다음 시도: 알트 universe + 짧은 hold 별도 작업** |
| [PB105](PB105_hyperliquid_leaderboard_mirror/source.md) | Hyperliquid Top trader 미러 | **dead (4/9, anti-alpha)** | — | ⚠️ **PoC 결과: real WR 21% < null WR 46%. n=142, alpha vs null +7bps noise. leaderboard endpoint 부재 + 데이터 부족. 어떤 형태로도 deploy X** |
| [PB106](PB106_cga_agent_meta/source.md) | CGA-Agent — GA + 멀티에이전트 메타 도구 | 2.2 | 메타 (운용 X) | 학술 단계. Strategy Registry 50+ 누적 시 재검토 |
| [PB107](PB107_hedge_scalping_ema_grid/source.md) | EMA 5-grid hedge scalping (long 4 limit + short 1 market) — nikita-doronin | 3.0 | 알트 perp 양방향 단기 scalping | source 작성 완료. 31⭐, Pine Script 교차 검증 가능. ccxt Bitget 어댑터 + 4주 paper 검증 후 G201 발급 후보 |

## 폐기/검증 실패 (PB900+)

| PB ID | repo | 사유 |
|---|---|---|
| (메모) CryptoGnome/LickHunterPRO | https://github.com/CryptoGnome/LickHunterPRO | 2021년 이후 dormant, ⭐144, Bitget 미지원 — PB 발급 불허 |
| (메모) alex-bormotov/breakout-trader | https://github.com/alex-bormotov/breakout-trader | ⭐35 (게이트 미달), Binance only — PB 발급 불허 |
| (메모, 2026-04-28) ailoglab.org BTC bot | https://ailoglab.org/코인-자동매매-프로그램-ai-바이브코딩 | RSI/CCI/Supertrend 6지표 vibe-coded. claimed BT +1283%/15mo/WR 61.5%/MDD 45%, **but live -50%**. 코드 비공개. 신뢰도 2.0 |
| (메모, 2026-04-28) HintBot / CoinDoriBot (Bitget) | https://www.ddengle.com/en/develop/16177514 | Bitget pullback 봇, free + referral 게이트, **closed-source SaaS** — 자동 reject |
| (메모, 2026-04-28) aitutor21 Bitget RSI&MACD | https://aitutor21.com/ailink/878 | RSI(30/70) + MACD cross 1m BTCUSDT — 튜토리얼 코드만, 백테스트·라이브 PnL 0건. 신뢰도 1.5 |
| (메모, 2026-04-28) liampgrichardson/Cryptocurrency_Trading_Bot | https://github.com/liampgrichardson/Cryptocurrency_Trading_Bot | LSTM-RNN+MA spot scalping, BT 6mo +54.49% Sharpe>2 MDD 33.78%, 1289 orders. **long-only spot Binance** — 양방향 미지원으로 6축 fit 부족. 신뢰도 2.5 |
| (메모, 2026-04-28) Hyperliquid 데이터 도구 (HyperStats / HyperTracker / Beacon / Dexly / ASXN) | n/a | leaderboard 도구지 전략 아님. PB105 dead 후 같은 미러링 시도는 동일 한계 (selection bias + variance). **PB 미발급** — 단, 90D ROI + grade≥A + 6mo+ 필터로 미래 PoC 시 데이터 소스 활용 가능 |
| (메모, 2026-04-28) 한국 retail (네이버/Velog/Tistory/디시) | n/a | 거의 모두 upbit-spot 변동성돌파 (Larry Williams) — 양방향·perp·5-10x lev 부적합. 검증된 코드+6mo 라이브 트랙 동시 충족 0건. **PB 발급 후보 없음** |

## 다음 작업

1. PB001: Mingogogo 460MB 마이닝 (subagent 위임, 1~2시간) → rules.md / claimed_performance.md 작성
2. PB100: freqtrade-strategies repo clone → NFI 룰 추출 → rules.md
3. 신뢰도 평가 ≥3 인 PB 만 G002+ 발급 게이트 통과
4. PB002 (하모닉 단독) / PB003 (Z-score reversal) / PB004~PB005 (Naver overlay) 후속

## 변경 이력

| 날짜 | 내용 |
|---|---|
| 2026-04-28 | PLAYBOOK 초기 + PB001 / PB100 스텁 |
| 2026-04-28 | PB101 (Passivbot 4.4) / PB102 (Jesse 4.0) / PB103 (funding-rate-arb 3.0) source 추가. SHORT/양방향/lottery gap 충족 후보 3종. LickHunterPRO·breakout-trader 는 게이트 미달로 메모만 |
| 2026-04-28 | **PB107** (EMA 5-grid hedge scalping, nikita-doronin) 신뢰도 3.0 등록. 한국 retail / Twitter PnL 추가 리서치 결과: **신뢰도 ≥3.0 통과 1건만** (PB107). 한국어 자료는 upbit-spot 변동성돌파 일색이라 양방향·perp·lottery 6축 부적합. ailoglab/HintBot/aitutor21/liampgrichardson 는 신뢰도 미달 메모 처리. Hyperliquid 신규 도구(HyperStats/HyperTracker)는 PB105 dead 메커니즘과 동일하므로 PB 미발급 |
