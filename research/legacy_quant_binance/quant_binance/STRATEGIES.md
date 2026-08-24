# quant_binance — Strategy Workflow

전략은 **코드가 아니라 파라미터 세트**다. 단일 결정 엔진(`strategy/regime.py`)을 `STRATEGY_OVERRIDE_PATH=<file.json>` 으로 갈아끼우는 구조이므로, **새 전략 = 새 JSON 한 파일**.

## 디렉터리

```
strategies/
├── REGISTRY.md                       # 마스터 인덱스 (한 화면 비교)
├── _template/                        # 신규 전략 스캐폴드 원본
│   ├── card.md
│   └── overrides.json
├── S001_baseline/                    # 베이스라인 (변경 금지)
│   ├── card.md                       # 가설·룰·결과 (사용자 작성)
│   ├── overrides.json                # 빈 = config.example.json 기본값
│   └── runs/                         # /strategy-eval 결과 (요약 JSON 누적)
└── Sxxx_*/...                        # 사용자가 추가
```

## 사이클 (반복할 것)

```
1. 부모 전략 card.md 의 "결론" 작성  ← 변수 1개 정함
2. /strategy-new <NEW_ID> --base <PARENT_ID>  ← 폴더·파일 스캐폴드
3. card.md 가설/변경 변수 작성, overrides.json 단일 키 변경
4. /strategy-eval <NEW_ID>  ← 백테스트 + 페이퍼 결과 자동 채움
5. card.md 결론 1줄 작성  ← 다음 사이클 진입점
```

**규칙 위반 시 학습 누적 안 됨** (CLAUDE.md "처방 B/C" 참조):
- 변수 ≥ 2개 동시 변경 → 원인 추적 불가. 별도 ID 분리.
- 결론 미작성 채로 다음 전략 → 같은 함정 반복.
- card.md 없이 코드만 수정 → 추후 같은 전략 재개발.

## 실행 명령 (수동 시)

```bash
# 백테스트 (전략 S007)
STRATEGY_OVERRIDE_PATH=quant_binance/strategies/S007_*/overrides.json \
python -m quant_binance.runtime --mode replay \
  --config quant_binance/config.example.json \
  --equity-usd 50 --capacity-usd 50 \
  --output-base quant_runtime

# 페이퍼라이브 (전략 S007)
STRATEGY_OVERRIDE_PATH=quant_binance/strategies/S007_*/overrides.json \
python -m quant_binance.runtime --mode paper-live-shell \
  --config quant_binance/config.example.json \
  --equity-usd 50 --capacity-usd 50 \
  --output-base quant_runtime
```

대부분의 경우 **`/strategy-eval S007`** 한 줄로 충분 (Claude 스킬이 위 명령을 알아서 구성).

## 결과 보관 정책

- **요약 메트릭**: `strategies/<ID>/runs/<mode>_<date>.json` (git 추적 — 작은 JSON)
- **원본 트레이스**: `quant_runtime/replay/<timestamp>/` (git 무시 — 큰 JSONL/로그)
- **장기 보관**: `iCloudDrive/quant_archive/quant_runtime/` (4.4GB 아카이브, iCloud 동기화)

## $50 자본 컨텍스트

CLAUDE.md "처방 A" 참조. 후속 전략 설계 시 다음 방향 우선:
- 진입 빈도 ≥ 3건/일 (현 베이스라인은 진입 거의 없음)
- 5~10x 레버리지
- 롱·숏 양방향
- breakout / squeeze release / liquidation reversal / funding flip 셋업
- TP +1.5R~+3R, SL −1R, 시간정지 N봉

## 관련 도구

- 슬래시 스킬: `/strategy-new`, `/strategy-eval` (iCloudDrive/.claude/skills/)
- 위임 에이전트: `quant-backtester`, `quant-monitor`, `quant-validator`, `quant-reporter`, `quant-supervisor` (iCloudDrive/.claude/agents/)
- 안전망: `risk/kill_switch.py` — 라이브 시 항상 활성
