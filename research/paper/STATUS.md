# Paper forward-validation status

생성 시각: `2026-07-27T08:51:56.953074+00:00`

실주문: **금지**. 주문 엔드포인트·API 키·서명 기능을 사용하지 않는다.
메이커 체결가정: 최신 공개 1D 바의 종가를 mid 추정 체결가로 사용하고, 진입·청산 각 leg에 0.02% maker fee를 적용하며 슬리피지는 0으로 둔다.
펀딩: 보유 perp notional × 공개 funding rate × 경과시간/8h로 가상 적립한다. 양수 funding에서 perp short는 수취한다.

| 후보 | 가상 에쿼티(USDT) | 오픈 포지션 | 누적 펀딩(USDT) | 최근 실행일 |
|---|---:|---|---:|---|
| W2c | 299.9200 | 현금 | 0.000000 | 2026-07-27 |
| F1e | 300.4302 | BTCUSDT spot long 300.09 USDT @ 65192<br>BTCUSDT perp short 300.09 USDT @ 65182.2 | 0.480206 | 2026-07-27 |
| W3c | 296.3649 | BANKUSDT perp long 2.80 USDT @ 0.38234<br>KAITOUSDT perp long 9.63 USDT @ 1.1901<br>TLMUSDT perp long 6.39 USDT @ 0.00168<br>DEXEUSDT perp short 1.63 USDT @ 3.539<br>HOMEUSDT perp short 5.27 USDT @ 0.006081<br>SYNUSDT perp short 5.38 USDT @ 0.1436 | 0.041926 | 2026-07-27 |
| W3d | 300.0000 | 현금 | 0.000000 | 2026-07-27 |
| G1 | 300.0000 | 현금 | 0.000000 | 2026-07-27 |

## 후보별 신호

- `W2c`: W2c carry selected: cash; 최근 손익 0.000000 USDT; maker fee 0.000000 USDT
- `F1e`: F1e carry selected: BTCUSDT; 최근 손익 0.335592 USDT; maker fee 0.000107 USDT
- `W3c`: W3c weekly momentum targets: BANKUSDT:long, KAITOUSDT:long, TLMUSDT:long, DEXEUSDT:short, HOMEUSDT:short, SYNUSDT:short; 최근 손익 -1.488915 USDT; maker fee 0.000824 USDT
- `W3d`: W3d weekly momentum targets: cash; 최근 손익 0.000000 USDT; maker fee 0.000000 USDT
- `G1`: G1 carry selected: cash; 최근 손익 0.000000 USDT; maker fee 0.000000 USDT

원장 경로: `research/paper/ledger/paper_ledger.jsonl`
