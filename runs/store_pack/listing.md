# RaceLens Play Store Listing

## App title
RaceLens

## Short description
경륜·경마 출전 정보와 모델 신호를 한눈에 정리합니다.

## Full description
RaceLens는 경륜·경마 데이터를 정보 분석 관점에서 정리하는 모바일 앱입니다.

공식 출전 정보, 참가자 기록, 모델 신호, 배당 자료 상태, 검증 상태를 한 화면에서 확인할 수 있습니다. 화면은 홈, 분석, Pro 안내로 나뉘며 사용자는 종목, 개최장, 경기일, 경주 번호를 고른 뒤 데이터 상태를 확인합니다.

주요 기능
- 경륜·경마 개최일과 경주 번호 선택
- 공식 출전 정보 기반 참가자 카드
- 모델 신호와 근거 데이터 요약
- 배당 자료 상태와 갱신 시각 분리 표시
- 경주 종료 후 실제 착순이 확인된 경우 복기용 결과 표시
- 무료 이용 상태와 Pro 준비 상태 안내

RaceLens는 정보 분석 도구입니다. 앱 안에서 참여 연결, 금액 산정, 구매 대행, 외부 참여 사이트 이동 기능을 제공하지 않습니다. 만 19세 이상 사용자를 대상으로 하며, 사용자는 거주 지역의 법령과 스토어 정책을 준수해야 합니다.

## Reviewer notes
RaceLens is a non-betting data analysis app for Korean keirin and horse racing information. The app does not include betting links, stake sizing, purchase agency flows, or gambling-site ads. The Pro screen is configured as a preparation/entitlement surface unless store billing is enabled. Legal pages are served by the API server:
- Privacy: https://168-107-2-218.sslip.io/legal/privacy
- Terms: https://168-107-2-218.sslip.io/legal/terms
- Account deletion: https://168-107-2-218.sslip.io/legal/account-deletion
- Support: https://168-107-2-218.sslip.io/legal/support
- Third-party SDK disclosure: the Android release uses Google Firebase Analytics, Crashlytics, and Google Mobile Ads SDK. After 3 free analyses, a user may voluntarily view a rewarded ad for 1 additional analysis. Gambling-site ads and betting links are not part of the product flow.

## Data safety answers
Based on runs/audit_gpt55.md:
- Anonymous device ID: collected for free usage limits, session continuity, abuse prevention, and entitlement state.
- App instance ID / Firebase installation identifier: collected by Google Firebase Analytics and Crashlytics for analytics, app stability, and crash diagnostics.
- IP address and User-Agent derived data: collected server-side for rate limiting and security diagnostics.
- UX events: app_open, screen_view, tab_select, race context, analysis request/result/error; payload blocks participant names, selections, user IDs, and device IDs.
- App activity events: coarse Firebase Analytics events such as tab, sport, race number, latency, error kind, and rounded confidence percentages for analytics and app functionality/stability.
- Diagnostics: Crashlytics crash logs, device model, OS, app version, stack traces, and error context for crash analysis and app stability.
- Analysis context and result metadata: sport, date, venue, race number, model response, market snapshot status for service operation and diagnostics.
- Subscription verification result: product/status/expiry only when store verification is configured.
- Email: not collected in app; only received if the user contacts support or requests deletion.
- Third-party ad/analytics SDK identifiers: Google Firebase Analytics, Crashlytics, and Google Mobile Ads SDK are present. Google may process advertising identifiers, IP address, device information, and rewarded-ad interactions for ad delivery, measurement, and fraud prevention.
- Third-party sharing / processing: Firebase data can be transmitted to and processed by Google as the Firebase service provider; Google Mobile Ads data can be processed by Google as the advertising service provider.
- Transport encryption: yes. Production URLs use HTTPS, so data is encrypted in transit.
- Deletion path: https://168-107-2-218.sslip.io/legal/account-deletion and support email on legal pages.
