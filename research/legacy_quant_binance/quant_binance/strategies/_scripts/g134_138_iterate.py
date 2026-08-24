"""G134-G138 자동 변형 + 백테스트 + 비교."""
import json, os, sys, subprocess, re
from pathlib import Path

REPO = Path.home() / "Desktop" / "first_repo"
STRATS = REPO / "quant_binance" / "strategies"
G131 = STRATS / "G131_score75_locked" / "overrides.json"
PYTHON = sys.executable
ENV = os.environ.copy(); ENV["PYTHONIOENCODING"] = "utf-8"


def make(name, **changes):
    """G131 override 복사 + changes 적용."""
    src_d = json.load(open(G131, encoding="utf-8"))
    folder = STRATS / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "runs").mkdir(exist_ok=True)
    for path, val in changes.items():
        keys = path.split(".")
        cur = src_d
        for k in keys[:-1]:
            cur = cur[k]
        cur[keys[-1]] = val
    src_d["_strategy_id"] = name.split("_")[0]
    src_d["_parent_id"] = "G131"
    src_d["_changed_keys_vs_parent"] = list(changes.keys())
    out = folder / "overrides.json"
    json.dump(src_d, open(out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    return out


def backtest(override_path, days=60, hold="4h"):
    """analyze_backtest 실행 → score bucket 결과 파싱."""
    e = ENV.copy(); e["STRATEGY_OVERRIDE_PATH"] = str(override_path)
    cmd = [PYTHON, "-m", "quant_binance.backtest.analyze_backtest",
           "--symbols", "ETHUSDT,SOLUSDT,DOGEUSDT,PEPEUSDT",
           "--skip-download", "--equity-usd", "55",
           "--days", str(days), "--holding-period", hold]
    res = subprocess.run(cmd, cwd=REPO, env=e, capture_output=True, text=True, timeout=300)
    out = res.stdout
    # score buckets 파싱
    buckets = {}
    for line in out.splitlines():
        m = re.match(r"\s+(\d+)\s+(\d+)\s+([\d.]+)%\s+([+\-]?[\d.]+)\s+([+\-]?\d+)", line)
        if m:
            score, n, wr, avg, total = m.groups()
            buckets[int(score)] = {"n": int(n), "wr": float(wr), "avg": float(avg), "total": int(total)}
    # cost=16 result for score 75
    cost16 = {}
    for line in out.splitlines():
        m = re.match(r"\s+(\d{2})\s+16\s+(\d+)\s+([\d.]+)%\s+([+\-]?[\d.]+)\s+([+\-]?\d+)", line)
        if m:
            cost16[int(m.group(1))] = {"n": int(m.group(2)), "wr": float(m.group(3)), "avg": float(m.group(4)), "total": int(m.group(5))}
    return buckets, cost16, out


def main():
    # 변형 정의
    variants = [
        # name, changes
        ("G134_lev10", {"risk.target_futures_leverage": 10.0}),
        ("G135_score76", {"mode_thresholds.futures_score_min": 76}),
        ("G136_strict_size", {"risk.per_trade_equity_risk": 0.20}),
        ("G137_lev10_size20", {"risk.target_futures_leverage": 10.0, "risk.per_trade_equity_risk": 0.20}),
        ("G138_safe_combo", {"risk.target_futures_leverage": 10.0, "mode_thresholds.futures_score_min": 76}),
    ]

    # baseline G131
    print(f"{'strategy':<24} {'score75 n/total/avg':<28} {'best score':<20}")
    print("-" * 80)
    def fmt(s75, best):
        a = "n={} total={} avg={}".format(s75.get('n','-'), s75.get('total','-'), s75.get('avg','-'))
        b = "best score {}: total={}".format(best[0], best[1].get('total','-'))
        return a, b

    bk, cost16, _ = backtest(G131, days=60, hold="4h")
    s75 = cost16.get(75, {})
    best = max(cost16.items(), key=lambda x: x[1].get("total", -99999)) if cost16 else (0, {})
    a, b = fmt(s75, best)
    print(f"{'G131 baseline':<24} {a:<35} {b:<28}")

    for name, changes in variants:
        path = make(name, **changes)
        bk, cost16, _ = backtest(path, days=60, hold="4h")
        s75 = cost16.get(75, {})
        best = max(cost16.items(), key=lambda x: x[1].get("total", -99999)) if cost16 else (0, {})
        chg_str = ",".join("{}={}".format(k.split('.')[-1], v) for k, v in changes.items())
        a, b = fmt(s75, best)
        print(f"{name:<24} {a:<35} {b:<28}  ({chg_str})")


if __name__ == "__main__":
    main()
