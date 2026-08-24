# Trend Continuation Research

## 현재 결론

이 모듈은 Bitget 공개 시장 데이터로 추세 지속 후보를 연구하는 도구다. 현재 공개 API와 UI는 항상 `NO_TRADE`를 반환하고 `execution.enabled=false`를 유지한다. KORUUSDT 실제 과거 데이터 실행도 캔들/펀딩 수집은 완료됐지만, 2026-06-22 16:00 UTC부터 2026-08-18 08:00 UTC까지의 KORUUSDT 실효 수수료 정책 이력을 공식 공개 자료로 증명하지 못해 `NO_RELIABLE_WINNER`로 닫혔다.

따라서 이 기능은 다음 추세, 반등, 청산 회피, 손실 복구를 예측하는 프로그램이 아니다. 자동 주문, 반자동 주문, 레버리지 설정, 계정 조회, 포지션 조회도 하지 않는다.

## 하드 경계

허용된 외부 접점은 Bitget v2 공개 시장 `GET` 요청 6개뿐이다.

| 허용 경로 | 용도 |
| --- | --- |
| `/api/v2/mix/market/contracts` | 공개 RWA/USDT-FUTURES 계약 확인 |
| `/api/v2/mix/market/candles` | 최근 완료 캔들 |
| `/api/v2/mix/market/history-candles` | 범위 지정 과거 완료 캔들 |
| `/api/v2/mix/market/ticker` | forward 관찰 시점의 공개 호가 증거 |
| `/api/v2/mix/market/current-fund-rate` | forward 현재 펀딩 보조 증거 |
| `/api/v2/mix/market/history-fund-rate` | 과거/forward 라벨 구간 펀딩 증거 |

금지 경계:

- `POST`, 주문, 취소, 수정, 계정, 포지션, 레버리지, private/signed endpoint.
- API key, secret, passphrase, UID 기반 account endpoint.
- historical bid/ask/spread 복원 또는 임의 추정. historical fixture에 `bid`, `ask`, `spread`, `bid_ask_spread_pct`가 있으면 차단한다.
- 공개 API/UI에서 `CONTINUATION_OBSERVE`를 매매 추천으로 승격. 내부 진단에 존재해도 public status는 `NO_TRADE`다.

## 버전과 프로필

| 항목 | 값 |
| --- | --- |
| research payload schema | `trend-continuation.v1` |
| feature schema | `trend-continuation-features.v1` |
| strategy version | `trend-continuation-strategy.v1` |
| selection payload schema | `trend-continuation-selection.v1` |
| API cache schema | `trend-continuation-api-cache.v1` |
| rule versions | `baseline_mtf_v1`, `technical_score_v1` |

| profile | 의미 | trigger interval | 후보 horizon |
| --- | --- | --- | --- |
| `standard_15m_5m` | 15m context + 5m confirmation | 5m | 6, 12, 18 bars |
| `triple_leveraged_15m_5m_1m` | 15m context + 5m confirmation + 1m trigger | 1m | 5, 10, 15 bars |

`triple_leveraged_15m_5m_1m`이라는 이름은 연구 프로필 이름일 뿐 레버리지 사용을 허용하지 않는다. `candidate_horizon_bars`는 trigger interval 기준이다. 예를 들어 triple profile의 10 bars는 10분 라벨 구간이다.

## 완료 캔들 feature registry

모든 feature는 point-in-time 완료 캔들만 사용한다. 각 timeframe 입력은 최소 60개, timezone-aware, 중복 없음, 시간 오름차순, interval 연속, future/open candle 없음, OHLCV 양수 조건을 통과해야 한다.

| feature | 공식 |
| --- | --- |
| `SMA20` | `mean(close[-20:])` |
| `SMA50` | `mean(close[-50:])` |
| `SMA20_SLOPE` | `SMA20(t)-SMA20(t-1)` |
| `EMA12` | SMA12 seed 후 alpha `2/13` |
| `EMA26` | SMA26 seed 후 alpha `2/27` |
| `MACD_LINE` | `EMA12-EMA26` |
| `MACD_SIGNAL` | `EMA9(MACD_LINE)` |
| `MACD_HIST` | `MACD_LINE-MACD_SIGNAL` |
| `RSI14` | Wilder RSI 14 |
| `ROC1` | `(close_t/close_t-1-1)*100` |
| `ATR14` | Wilder ATR 14 |
| `BOLLINGER_WIDTH20` | `(upper-lower)/SMA20*100` |
| `RVOL20` | `volume_t/mean(volume_t-20_to_t-1)` |
| `ADX14` | Wilder ADX 14 |

방향 판정:

- UP: `close > SMA20 > SMA50`이고 `SMA20_SLOPE > 0`.
- DOWN: `close < SMA20 < SMA50`이고 `SMA20_SLOPE < 0`.
- 그 외는 `NEUTRAL`.

무효화 가격은 trigger timeframe에서 최근 directional pivot을 찾고 ATR14의 0.25배를 더해 만든다. UP은 pivot low - `0.25*ATR14`, DOWN은 pivot high + `0.25*ATR14`다.

`baseline_mtf_v1`은 15m/5m 방향 정렬, 5m RSI/ROC confirmation, triple profile의 1m RVOL trigger를 요구한다. `technical_score_v1`은 70점 이상이어야 하며 점수 항목은 방향 정렬 30, RSI 10, MACD 10, ROC 15, RVOL 15, ATR% 범위 10, ADX 10이다.

## 평가와 승격 계약

historical selection은 deterministic이다.

- 방향은 UP/DOWN을 모두 평가한다.
- profile별 horizon 전체와 rule version 2개를 평가한다.
- historical rows는 `ObservationSource.HISTORICAL_EXPLORATORY`만 받는다.
- 라벨 구간은 서로 겹치지 않는 rows만 사용한다.
- split은 50/60/70% boundary 기준 rolling development folds다.
- standard profile embargo는 14시간, triple profile embargo는 12시간 45분이다.
- validation rows는 boundary+embargo 이후부터 next boundary 이전만 쓴다.
- 선택 후보는 validation resolved count 80개 이상, Wilson lower 95% 50 이상, mean return lower 95% 양수여야 한다.
- historical selection은 항상 `exploratory_only=true`, `promotable=false`다.
- fee policy는 시작 시점, 종료 직전, freeze 시점 모두에서 유효해야 한다.
- selection manifest는 digest 기반으로 고정한다. 이후 forward validation은 정확히 80개, 그 뒤 holdout은 더 나중의 정확히 20개만 허용한다.
- forward batch는 phase별로 한 번만 소비되며, 중복 evidence time, interval overlap, 검증 전 holdout 접근은 rejection으로 닫힌다.
- validation/holdout 결과를 반복 접근하거나 순서를 어기면 terminal rejection으로 닫는다.

historical cost는 bid/ask/spread를 만들지 않는다. historical exploratory cost는 수수료 정책이 증명된 경우에만 `0.20%` floor와 방향성 funding cost를 더한다. forward cost는 관찰 시점의 실제 공개 ticker bid/ask spread와 fee policy, funding proof가 모두 있을 때만 계산한다.

## 수수료 이력 caveat

문서 작성 시 확인한 Bitget 공식 URL:

- https://www.bitget.com/api-doc/contract/market/Get-VIP-Fee-Rate
- https://www.bitget.com/api-doc/uta/account/Get-Account-Fee-Rate
- https://www.bitget.com/api-doc/uta/account/Get-All-Fee-Rate
- https://www.bitget.com/api-doc/uta/public/Get-Fee-Group
- https://www.bitget.com/api-doc/uta/public/Instruments
- https://www.bitget.com/api-doc/uta/changelog

이 URL들은 현재 수수료율, 계정별 현재 fee rate, group fee configuration, instrument 정보, changelog를 확인하는 데만 쓴다. 현재 확인된 공식 문서 기준으로는 KORUUSDT의 2026-06-22 16:00 UTC부터 2026-08-18 08:00 UTC까지 timestamped historical effective fee policy를 공개 `GET`으로 증명하는 endpoint가 없다.

중요한 제한:

- 현재 fee rate는 과거 기간에 backfill할 수 없다.
- account fee endpoints는 private/account read permission이 필요하므로 이 연구 경계 밖이다.
- changelog는 API 변경 이력이지, 특정 심볼의 과거 실효 수수료율 snapshot이 아니다.
- 수수료 이력을 증명하려면 해당 기간에 수집해 둔 user-owned 공식 fee-policy snapshot, 원문 hash, effective range가 필요하다.

이 증거가 없으면 evaluator는 `historical_fee_policy_coverage_unproven`으로 실패 종료한다.

## KORUUSDT 실제 실행 결과

재현 명령:

```bash
./.venv/bin/python manage.py evaluate_bitget_trend_continuation --symbol KORUUSDT --profile triple_leveraged_15m_5m_1m --start 2026-06-22T16:00:00+00:00 --end 2026-08-18T08:00:00+00:00 --max-pages 720
```

기대 결과는 성공 종료가 아니라 `exit 1`이다.

| 항목 | 관찰값 |
| --- | --- |
| mode | `research_only` |
| execution | `enabled=false` |
| exploratory | `true` |
| decision | `NO_RELIABLE_WINNER` |
| failure | `historical_fee_policy_coverage_unproven` |
| 15m candles | 5,440 rows, complete/continuous |
| 5m candles | 16,320 rows, complete/continuous |
| 1m candles | 81,600 rows, complete/continuous |
| funding | 2 pages, 170 filtered events, complete |
| research selection | `candidate_count=0`, `candidate_frozen=false` |

증거 파일:

- `.omo/evidence/trend-continuation/task5-koru-actual-manualqa.json`
- `.omo/evidence/trend-continuation/task5-koru-actual-cli-transcript.txt`
- `.omo/evidence/trend-continuation/real-data-koru-15m-coverage-20260818.json`
- `.omo/evidence/trend-continuation/real-data-koru-5m-coverage-20260818.json`
- `.omo/evidence/trend-continuation/real-data-koru-1m-coverage-20260818.json`
- `.omo/evidence/trend-continuation/real-data-koru-funding-coverage-20260818.json`

raw response hash는 Bitget 응답이 재수집될 때 달라질 수 있다. 문서에서 고정해 쓰는 것은 안전 판단, row count, complete/continuous 여부, failure code다.

## API/UI 동작

Endpoint:

```text
GET /api/bitget/trend-continuation/
```

동작:

- 인증된 요청만 연구 facade를 호출한다. 비인증 요청은 시장 데이터를 조회하지 않는다.
- cache key는 최신 완료 1m boundary를 사용한다.
- TTL은 60초다.
- cache wrapper schema, boundary, payload identity가 맞지 않으면 cache를 버린다.
- service payload에 내부 `CONTINUATION_OBSERVE` diagnostic이 있어도 public `decision_status`는 `NO_TRADE`다.
- public payload는 forbidden key/value 검사 후 반환된다.
- UI panel은 `NO_TRADE · 검증 미완료`, `실행 비활성`을 표시한다.

forward watcher:

- `watch_bitget_trend_continuation` command는 연구 전용 forward evidence 수집/정산 루프다.
- 생성된 관찰은 manifest lifecycle의 validation 80개, holdout 20개 계약을 통과해야만 평가된다.
- 이 watcher도 주문, 레버리지, 계정/포지션 endpoint를 호출하지 않는다.

## 검증 명령

API/UI smoke:

```bash
./.venv/bin/python manage.py test pulse.test_trend_continuation_api pulse.test_trend_continuation_panel --verbosity 1
```

검증 결과: 13 tests OK.

owned diff:

```bash
node scripts/verify_trend_continuation_owned_diff.mjs
```

검증 결과: `{"status":"PASS" ...}`.

historical evaluator boundary spot check:

```bash
rg -n "collect_forward_quote|/api/v2/mix/market/ticker|/api/v2/mix/order|/api/v2/mix/account|/api/v2/mix/position|set-leverage|leverage|POST|bid_price|ask_price|spread_pct" pulse/management/commands/evaluate_bitget_trend_continuation.py pulse/services/trend_continuation_historical_research.py pulse/services/trend_continuation_history.py pulse/services/trend_continuation_funding.py pulse/services/trend_continuation_historical_selection.py pulse/services/trend_continuation_evaluation.py pulse/services/trend_continuation_contract.py
```

검증 결과: historical evaluator chain에서는 order/account/position/leverage/private/quote-spread reconstruction 호출이 나오지 않는다. 허용 allowlist의 `/api/v2/mix/market/ticker` 문자열과 profile 이름만 정적 매치된다.

KORU actual command:

```bash
./.venv/bin/python manage.py evaluate_bitget_trend_continuation --symbol KORUUSDT --profile triple_leveraged_15m_5m_1m --start 2026-06-22T16:00:00+00:00 --end 2026-08-18T08:00:00+00:00 --max-pages 720
```

검증 결과: `exit 1`, `CommandError: historical_fee_policy_coverage_unproven`, `NO_RELIABLE_WINNER`.

## 한계와 운영 기준

- 이 연구는 현재 추세의 다음 방향을 보장하지 않는다.
- 어떤 레버리지에서도 청산 방지나 손실 복구를 주장하지 않는다.
- KORUUSDT historical run은 비용 이력 증명 부재 때문에 사용할 수 있는 승격 결과가 아니다.
- forward 관찰에서도 정확히 80개 validation과 그 이후 정확히 20개 holdout을 모두 통과하기 전까지 public 결과는 `NO_TRADE`다.
- 공개 market data availability는 exchange/vendor 상태에 의존한다.
- Bitget API 사용 약관, 데이터 라이선스, 관할권별 금융 규제 검토는 별도다.
