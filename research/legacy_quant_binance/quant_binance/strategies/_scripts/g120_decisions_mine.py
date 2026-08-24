"""
G120 — decisions.jsonl + closed_trades 매칭으로 winner vs loser feature 분석.

decisions = 시스템이 발화한 모든 진입 결정 (208건)
closed_trades = 그 결과 종료된 거래 (78건)

매칭: decision_id 또는 symbol+timestamp 기준 → outcome 부여 → feature 차별화 식별.
"""
import json, sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

ROOT = Path.home() / "iCloudDrive" / "quant_archive"
DECISIONS = ROOT / "quant_runtime" / "forensics" / "decisions.jsonl"


def load_decisions():
    with open(DECISIONS, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def load_all_trades():
    trades = []
    for p in ROOT.rglob("closed_trades.jsonl"):
        try:
            with open(p, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try: trades.append(json.loads(line))
                        except: pass
        except: pass
    return trades


def main():
    decs = load_decisions()
    trades = load_all_trades()
    print(f"decisions: {len(decs)} / closed_trades: {len(trades)}")

    if not decs:
        print("NO DECISIONS"); return

    # 모든 decision feature 후보 식별
    sample = decs[0]
    print("\n=== decision keys ===")
    for k in sorted(sample.keys()):
        v = sample[k]
        t = type(v).__name__
        preview = str(v)[:50] if not isinstance(v, dict) else f"dict({len(v)})"
        print(f"  {k}: {t} = {preview}")

    df_dec = pd.DataFrame(decs)
    print(f"\ndecisions df shape: {df_dec.shape}")

    # 핵심 numeric features
    numeric_cols = []
    for c in df_dec.columns:
        if df_dec[c].dtype in [np.float64, np.int64, np.float32, np.int32]:
            numeric_cols.append(c)
    print(f"\nNumeric features: {len(numeric_cols)}")
    for c in numeric_cols[:30]:
        s = df_dec[c]
        print(f"  {c}: min={s.min():.3g} max={s.max():.3g} mean={s.mean():.3g}")

    # 필수 컬럼 확인
    print("\n=== 핵심 컬럼 sample ===")
    for k in ["decision_id", "symbol", "side", "decision", "decision_action", "predictability_score",
              "entry_predictability_score", "entry_signal", "should_enter"]:
        if k in df_dec.columns:
            uniq = df_dec[k].value_counts().head(3) if df_dec[k].dtype == object else df_dec[k].describe()
            print(f"\n  {k}:")
            print(uniq)

    # decision 결과 분류 (어떤 게 entry 인지)
    if "candidate_mode" in df_dec.columns:
        print(f"\ncandidate_mode: {df_dec['candidate_mode'].value_counts().to_dict()}")

    # symbol + open_time 매칭으로 trade 매핑
    if trades:
        df_tr = pd.DataFrame(trades)
        if "entry_time" in df_tr.columns:
            df_tr["entry_time_dt"] = pd.to_datetime(df_tr["entry_time"], utc=True, errors="coerce")
        # decisions 의 timestamp / symbol 으로 매칭
        if "timestamp" in df_dec.columns and "symbol" in df_dec.columns:
            df_dec["ts_dt"] = pd.to_datetime(df_dec["timestamp"], utc=True, errors="coerce")
            print(f"\ndecisions timestamps range: {df_dec['ts_dt'].min()} ~ {df_dec['ts_dt'].max()}")

        if "entry_time_dt" in df_tr.columns:
            print(f"\ntrades entry_time range: {df_tr['entry_time_dt'].min()} ~ {df_tr['entry_time_dt'].max()}")

    # predictability_score 관련 컬럼 검색
    pred_cols = [c for c in df_dec.columns if "predict" in c.lower()]
    print(f"\n\npredictability 관련 컬럼: {pred_cols}")
    for c in pred_cols:
        s = df_dec[c]
        if s.dtype in [np.float64, np.int64]:
            print(f"  {c}: min={s.min():.2f} max={s.max():.2f} mean={s.mean():.2f} median={s.median():.2f}")

    # 가장 중요: 각 decision 이 실제로 entry 됐는지 + winner/loser 인지 매칭
    if "decision" in df_dec.columns or "should_enter" in df_dec.columns:
        col = "decision" if "decision" in df_dec.columns else "should_enter"
        print(f"\n{col} distribution: {df_dec[col].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
