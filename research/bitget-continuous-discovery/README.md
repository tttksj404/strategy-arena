# Bitget 알트코인 연속 탐색과 X 게시물 실험

이 폴더는 2026-08-24 이후 대화에서 수행한 공개 시장 데이터 기반 연구를 모은다. 자동 주문·텔레그램 발송·계좌 변경은 포함하지 않는다. **수익 사례가 있는 것과 실전 승격 기준을 통과한 것은 다르며, 현재 실전 승격은 하지 않는다.**

## 먼저 읽을 파일

- `X_SOURCE_REVIEW_KO.md`: 사용자가 제공한 5개 게시물의 내용, 직접 재현 가능한 규칙, 대체 가설과 미검증 범위.
- `work/x_discovery_20260910_recheck/report.json`: 245개 후보의 최종 재실행. 최초 결과는 별도 폴더에 보존.
- `work/x_volume_control_20260910_recheck/report.json`: 기존 활성 종목 동일가중 Donchian과 5분 거래량 청산 규칙의 별도 대조.
- `work/X_RUN_ERRATA_20260910.md`: 사전등록 표현과 실제 분모·test 잠금 계약 차이, Windows 해시 교정 이력.
- `data-refresh/REFRESH_REPORT.json`: 최신 14코인·10개 시장가격 시간봉·4개 마크가격 시간봉, 645,134개 봉과 무인증 수집 검증.
- `output/pdf/strategy-arena-top-return-candidates-ko.pdf`: 일반적인 차트 설명이 아닌 기존 상위 전략의 규칙·실제 차트·배율 경로 설명.
- `CONTEXT_HANDOFF_KO.md`: 요구 변화, 이전 실험, 실패와 수정 이력, 현재 판정.
- `AUDIT_PACKET.md`: 검증 증거와 남은 금융 감사 조건.
- `BUNDLE_MANIFEST.json`: 복사 원본·저장본의 크기와 SHA-256. 과거 실패 보고서도 삭제하지 않는다.

## 기존 수익률과 새 결과를 혼동하지 않기

이전 Donchian 고정 2배 frontier의 역사 test 스트레스 수익 +161.32%는 기존 close-to-close 계산·편도 0.12% 스트레스 비용·펀딩 미포함 조건이다. 새 계산은 다음 시가 체결, 실제 수량의 재조정·청산 비용, 편도 0.18% 스트레스 비용, 별도 보유비용 가정 및 계좌 전체 시장봉 청산 프록시다. 둘은 같은 체결 계약이 아니다.

또한 245개 탐색의 Donchian은 전체 알트 13개를 분모로 한 비중 변형이다. 기존 전략은 신호가 활성화된 종목 수 N으로 나눈 뒤 개별 유동성 필터를 적용한다. 이 차이를 숨기지 않고 v2 결과를 보존했으며 원본 신호를 직접 호출한 별도 통제 실험을 추가했다. `fixed_5` 역시 원천 비중에 5를 곱한 계수이며, 계좌가 항상 5배 노출이라는 뜻이 아니다.

## 재현

저장소 루트가 아니라 **이 폴더를 작업 디렉터리**로 사용한다. Python 3.12 이상과 uv를 사용한다.

```powershell
uv run work/run_x_discovery.py --archive <intraday-v5> --daily-archive <daily-v2>
uv run work/run_x_volume_control.py --intraday-archive <intraday-v5> --daily-archive <daily-v2>
uv run work/refresh_bitget_mtf.py --output <new-output-directory> --start <UTC-ISO8601>
uv run --with httpx2 --with numpy --with scipy --with scikit-learn --with reportlab --with matplotlib --with pydantic --with pytest --with statsmodels --with typer --with rich python -m pytest work -q --confcutdir=.
uv run work/generate_top_strategy_pdf.py
uv run --with basedpyright --with numpy --with scipy --with statsmodels --with httpx2 --with pydantic --with typer --with rich basedpyright --project work/pyright_x_strict.json
```

신규 실행기의 `.py.lock` 파일을 함께 보관한다. PDF 생성기는 Windows의 Malgun Gothic 폰트에 의존한다. 다른 환경에서는 같은 한글 지원 폰트를 지정하고 다시 렌더 검증해야 한다.

공개 캔들 수집에는 API 키가 필요하지 않다. 사용자가 허용한 키 파일은 네트워크·저장소로 보내지 않았으며, 실제 키 사용·계좌 인증 성공으로 보고하지 않는다. `SETTINGS.json`이 이미 있으면 처음 고정한 종료 시점을 재사용하고, 신규·재개 모두 30일 이내의 양수 구간만 허용한다. 더 긴 과거 수집에는 별도 전역사 수집기를 사용한다.

기존 대용량 intraday 전역사 원시 아카이브는 이 Git 묶음에 포함하지 않는다. `data-manifests/`의 파일·해시 목록, 수집기 `work/fetch_bitget_history.py`, 재개·감사 도구를 보관한다. 예외적으로 PDF 전략 진단·실제 데이터 테스트에 쓰는 잠금 일봉 `work/bitget_history_full_daily_20260824_v2/`는 506개 파일·1,317,753바이트라 원시 파일까지 함께 보관했다. 원래 모든 시간봉 실험을 그대로 재현하려면 별도 보관한 intraday 해시의 원시 파일도 필요하다. 새로 API를 수집하면 관측 시점·상장 범위가 달라지므로 같은 데이터라고 간주할 수 없다. 현재 연구 공통 패널은 14개 생존 종목의 약 2021-12~2026-08이며 모든 상장·상폐 코인의 전 역사가 아니다.

별도 `data-refresh/`에는 2026-08-24 00:00 UTC부터 2026-09-10 00:29:55.707 UTC 스냅샷까지의 **완료봉 원시 갱신분도 포함**한다. 시장가격 1m/3m/5m/15m/30m/1H/4H/6H/12H/1D, 마크가격 5m/1H/4H/1D다. 시간봉별 마지막 완료 시각은 다르며 일봉 등은 거래소 기본 시간대 경계를 유지한다. 개인 API 키 없이 공개 GET으로 수집했고, 기존 245개 역사 평가에 합치지 않았다. 최신 마크가격을 확보한 것이 과거 전체 청산 입력을 모두 채운 것은 아니다.

10MB를 넘는 과거 JSON 3개는 내용 손실 없이 `.json.gz`로 저장한다. 해당 구형 실행기/분석기가 필요하면 같은 폴더에 원래 이름으로 압축 해제한다. `BUNDLE_MANIFEST.json`의 원본 SHA-256과 대조해야 한다. 외부 X 글의 전체 원문·영상은 저작권 때문에 재배포하지 않고 링크와 로컬 취득 해시만 남긴다.

## 한계

모든 실험은 역사 데이터 재사용과 생존편향의 영향을 받는다. 실제 mark price, 계좌별 유지증거금 구간, 격리/교차 모드, 주문장 충격, 체결 실패를 완전히 재현하지 않는다. 공개 주문 흐름 API의 짧은 응답을 수년짜리 기관 주문 데이터로 대체하지 않았다. 별도 Fable5 금융 승격 감사는 대기 상태다.
