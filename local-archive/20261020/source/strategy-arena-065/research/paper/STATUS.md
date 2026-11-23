# Paper forward-validation status

생성 시각: `2026-07-30T00:12:12.358066+00:00`

실주문: **금지**. 주문 엔드포인트·API 키·서명 기능을 사용하지 않는다.
메이커 체결가정: 최신 공개 1D 바의 종가를 mid 추정 체결가로 사용하고, 진입·청산 각 leg에 0.02% maker fee를 적용하며 슬리피지는 0으로 둔다.
펀딩: 보유 perp notional × 공개 funding rate × 경과시간/8h로 가상 적립한다. 양수 funding에서 perp short는 수취한다.

| 후보 | 가상 에쿼티(USDT) | 오픈 포지션 | 누적 펀딩(USDT) | 최근 실행일 |
|---|---:|---|---:|---|
| W2c | 299.9200 | 현금 | 0.000000 | 2026-07-30 |
| F1e | 300.6895 | BTCUSDT spot long 300.48 USDT @ 64042<br>BTCUSDT perp short 300.48 USDT @ 64020.8 | 0.684485 | 2026-07-30 |
| W3c | 296.3505 | BANKUSDT perp long 2.47 USDT @ 0.16854<br>KAITOUSDT perp long 9.67 USDT @ 1.2689<br>TLMUSDT perp long 6.63 USDT @ 0.001606<br>DEXEUSDT perp short 1.61 USDT @ 2.732<br>HOMEUSDT perp short 5.36 USDT @ 0.005352<br>SYNUSDT perp short 5.61 USDT @ 0.1326 | 0.132752 | 2026-07-30 |
| W3d | 300.0000 | 현금 | 0.000000 | 2026-07-30 |
| G1 | 300.0000 | 현금 | 0.000000 | 2026-07-30 |

## 후보별 신호

- `W2c`: W2c carry selected: cash; 최근 손익 0.000000 USDT; maker fee 0.000000 USDT
- `F1e`: F1e carry selected: BTCUSDT; 최근 손익 0.213270 USDT; maker fee 0.000019 USDT
- `W3c`: W3c weekly momentum targets: BANKUSDT:long, KAITOUSDT:long, TLMUSDT:long, DEXEUSDT:short, HOMEUSDT:short, SYNUSDT:short; 최근 손익 0.199648 USDT; maker fee 0.000184 USDT
- `W3d`: W3d weekly momentum targets: cash; 최근 손익 0.000000 USDT; maker fee 0.000000 USDT
- `G1`: G1 carry selected: cash; 최근 손익 0.000000 USDT; maker fee 0.000000 USDT

## 후보별 유니버스 커버리지

| 후보 | 요구 유니버스 | 실제 커버 | 판정 |
|---|---:|---:|---|
| W2c | 200 | 203 | PASS |
| F1e | 2 | 2 | PASS |
| W3c | 150 | 170 | PASS |
| W3d | 150 | 170 | PASS |
| G1 | 100 | 203 | PASS |

## 오염 기록 처리

- `G1` 유효 기록 시작일: **2026-07-29** (오염 기록 2건 집계 제외: 2026-07-27, 2026-07-28)

## 수집 상태

수집 시간: 118.8s (목표 5분 이내).
funding_series 커버 심볼: 203개.

원장 경로: `research/paper/ledger/paper_ledger.jsonl`
