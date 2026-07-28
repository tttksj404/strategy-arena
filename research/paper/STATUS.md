# Paper forward-validation status

생성 시각: `2026-07-28T01:16:29.306728+00:00`

실주문: **금지**. 주문 엔드포인트·API 키·서명 기능을 사용하지 않는다.
메이커 체결가정: 최신 공개 1D 바의 종가를 mid 추정 체결가로 사용하고, 진입·청산 각 leg에 0.02% maker fee를 적용하며 슬리피지는 0으로 둔다.
펀딩: 보유 perp notional × 공개 funding rate × 경과시간/8h로 가상 적립한다. 양수 funding에서 perp short는 수취한다.

| 후보 | 가상 에쿼티(USDT) | 오픈 포지션 | 누적 펀딩(USDT) | 최근 실행일 |
|---|---:|---|---:|---|
| W2c | 299.9200 | 현금 | 0.000000 | 2026-07-28 |
| F1e | 300.4291 | BTCUSDT spot long 300.43 USDT @ 63634<br>BTCUSDT perp short 300.43 USDT @ 63632.9 | 0.518089 | 2026-07-28 |
| W3c | 297.5778 | BANKUSDT perp long 2.76 USDT @ 0.34143<br>KAITOUSDT perp long 9.54 USDT @ 1.2826<br>TLMUSDT perp long 6.38 USDT @ 0.001641<br>DEXEUSDT perp short 1.61 USDT @ 2.607<br>HOMEUSDT perp short 5.34 USDT @ 0.005597<br>SYNUSDT perp short 5.35 USDT @ 0.1427 | 0.072514 | 2026-07-28 |
| W3d | 300.0000 | 현금 | 0.000000 | 2026-07-28 |
| G1 **FIDELITY_FAIL** | 300.0000 | 현금 | 0.000000 | 2026-07-28 |

## 참고 — 표시된 FIDELITY_FAIL은 과거 기록

- `G1`: 위 표의 FIDELITY_FAIL은 `2026-07-28`에 이미 기록된, 유니버스 요건 미달 상태의 과거 판정이다. 방금 재확인한 실제 커버리지는 정상(PASS, 204/100) — 원장은 하루 1건 정책상 다음 UTC 날짜의 실행부터 새 판정을 반영한다.

## 후보별 신호

- `W2c`: W2c carry selected: cash; 최근 손익 0.000000 USDT; maker fee 0.000000 USDT
- `F1e`: F1e carry selected: BTCUSDT; 최근 손익 -0.001093 USDT; maker fee 0.000134 USDT
- `W3c`: W3c weekly momentum targets: BANKUSDT:long, KAITOUSDT:long, TLMUSDT:long, DEXEUSDT:short, HOMEUSDT:short, SYNUSDT:short; 최근 손익 1.212918 USDT; maker fee 0.000054 USDT
- `W3d`: W3d weekly momentum targets: cash; 최근 손익 0.000000 USDT; maker fee 0.000000 USDT
- `G1` [FIDELITY_FAIL]: G1 carry selected: cash; 최근 손익 0.000000 USDT; maker fee 0.000000 USDT

## 후보별 유니버스 커버리지

| 후보 | 요구 유니버스 | 실제 커버 | 판정 |
|---|---:|---:|---|
| W2c | 200 | 204 | PASS |
| F1e | 2 | 2 | PASS |
| W3c | 150 | 171 | PASS |
| W3d | 150 | 171 | PASS |
| G1 | 100 | 204 | PASS |

## 오염 기록 처리

- `G1` 유효 기록 시작일: **아직 없음 — 다음 실행일부터 (오늘 날짜는 이미 오염 기록으로 점유됨)** (오염 기록 2건 집계 제외: 2026-07-27, 2026-07-28)

## 수집 상태

수집 시간: 88.3s (목표 5분 이내).
funding_series 커버 심볼: 204개.

원장 경로: `research/paper/ledger/paper_ledger.jsonl`
