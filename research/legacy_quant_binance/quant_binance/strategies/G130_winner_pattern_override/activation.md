# G130 활성화 가이드

## 즉시 paper-live 시작 (3 단계)

### Step 1: 환경 변수 설정

```bash
export STRATEGY_OVERRIDE_PATH="$HOME/Desktop/first_repo/quant_binance/strategies/G130_winner_pattern_override/overrides.json"
```

### Step 2: 파일 검증

```bash
# JSON 문법 OK?
python -c "import json; json.load(open('$STRATEGY_OVERRIDE_PATH')); print('OK')"

# universe / score 임계 확인
python -c "import json; d=json.load(open('$STRATEGY_OVERRIDE_PATH')); print('universe:', d['universe']); print('score_min:', d['mode_thresholds']['futures_score_min']); print('size:', d['risk']['per_trade_equity_risk']); print('lev:', d['risk']['target_futures_leverage'])"
```

### Step 3a: PowerShell (Windows) — paper-live daemon 시작

```powershell
$env:STRATEGY_OVERRIDE_PATH = "$HOME\Desktop\first_repo\quant_binance\strategies\G130_winner_pattern_override\overrides.json"
$env:PAPER_TRADING = "1"
cd $HOME\Desktop\first_repo
python -m quant_binance.daemon
```

### Step 3b: Bash (Mac/Linux/WSL)

```bash
cd ~/Desktop/first_repo
STRATEGY_OVERRIDE_PATH="$HOME/Desktop/first_repo/quant_binance/strategies/G130_winner_pattern_override/overrides.json" \
PAPER_TRADING=1 \
python -m quant_binance.daemon
```

## 모니터링

### 실시간 상태
```bash
tail -f quant_runtime/output/paper-live-shell/latest/overview.json
```

### 진입/청산 이벤트
```bash
tail -f quant_runtime/forensics/closed_trades.jsonl
tail -f quant_runtime/forensics/decisions.jsonl
```

### 일일 요약 (커스텀 스크립트)
```bash
cd ~/Desktop/first_repo && python -m quant_binance.report_performance
```

## 단계별 평가 (verify-first 대신 paper-live 검증)

| 일차 | 통과 기준 |
|---|---|
| Day 1-3 | daemon 안정 동작 / API 에러 0 / 진입 1건+ |
| Day 7 | 누적 PnL > 0 / 진입 ≥ 10건 / score 70+ 진입만 발화 (filter 검증) |
| Day 14 | 누적 PnL > +5% / max DD < 20% / Sharpe > 0.5 |
| Day 30 | 누적 PnL > +20% / WR ≥ 55% / annualized > +50% / Sharpe > 1.0 |

5/5 통과 시 → micro-live $5 실거래 검증 30일 → real $55 deploy.

## 비상 정지

```bash
# kill daemon
ps aux | grep quant_binance.daemon
kill <PID>

# 모든 포지션 시장가 청산 (사용자 본인 Bitget 앱 사용 권장)
```

## 변경 / 튜닝

7일/14일/30일 평가 결과 기반 파라미터 조정:
- 진입 빈도 너무 낮음 → `futures_score_min` 70 → 65
- 진입 너무 많음 → 70 → 75
- 빈번한 stop-loss → `stop_loss_roe_percent` -28 → -22
- TP 너무 빠름 → `proactive_take_profit_thresholds` [18, 35, 50] → [25, 45, 65]

각 변경은 별도 G131/G132/... ID 부여 (변수 1개 룰 + verify-first).
