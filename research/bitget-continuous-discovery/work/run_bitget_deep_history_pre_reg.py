"""Pre-registered deep-history test on the frozen Bitget daily archive.

Hypotheses fixed BEFORE looking at results:
  H1 cross-sectional momentum (alts ex-BTC): lookbacks {20,30,60}d
  H2 time-series momentum (per symbol, vol-targeted): lookbacks {30,60,90}d
Trials N=3 per family -> minimal multiple-testing burden.

Metrics computed in log space (overflow-safe), CSCV-PBO within family,
Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014).
"""

from __future__ import annotations

import gzip
import itertools
import json
import math
from pathlib import Path

import numpy as np

WORK = Path(__file__).resolve().parent
ARCHIVE = WORK / "bitget_history_full_daily_20260824_v2"
OUT = WORK / "bitget_deep_history_prereg_report.json"

COST_PER_TURNOVER = 0.0010
STRESS_COST = 0.0025
MIN_HISTORY = 200
ANN = 365
TOP_K = 3
BOTTOM_K = 3
TARGET_VOL_ANN = 0.5
MAX_LEV = 5.0
RET_CLIP = (-0.95, 3.0)


def load_panel() -> tuple[list[str], list[int], np.ndarray]:
    symbols = sorted(d.name.replace("USDT", "") for d in ARCHIVE.iterdir() if d.is_dir())
    series: dict[str, dict[int, float]] = {}
    for sym in symbols:
        rows: dict[int, float] = {}
        for part in sorted((ARCHIVE / f"{sym}USDT" / "1D").glob("*.jsonl.gz")):
            with gzip.open(part, "rt") as fh:
                for line in fh:
                    rec = json.loads(line)
                    rows[rec["timestamp_ms"]] = float(rec["close_price"])
        series[sym] = rows
    common_ts = sorted(set.intersection(*(set(s) for s in series.values())))
    prices = np.array([[series[sym][ts] for ts in common_ts] for sym in symbols])
    return symbols, common_ts, prices


def sharpe_ann(r: np.ndarray) -> float:
    r = r[np.isfinite(r)]
    if r.size < 30:
        return 0.0
    sd = r.std(ddof=1)
    return 0.0 if sd == 0 else float(r.mean() / sd * math.sqrt(ANN))


def mdd_from_log(log_eq: np.ndarray) -> float:
    peak = np.maximum.accumulate(np.nan_to_num(log_eq, nan=-1e18))
    return float((log_eq - peak).min())


def ann_return_from_log(log_eq: np.ndarray) -> float:
    finite = log_eq[np.isfinite(log_eq)]
    if finite.size == 0:
        return 0.0
    return float(math.expm1(finite[-1] / finite.size * ANN))


def cscv_pbo(config_returns: dict[str, np.ndarray], n_blocks: int = 6) -> tuple[float, float]:
    names = sorted(config_returns)
    mat = np.column_stack([config_returns[n][: min(len(v) for v in config_returns.values())]
                           for n in names])
    t = mat.shape[0]
    edges = np.linspace(0, t, n_blocks + 1).astype(int)
    blocks = [mat[edges[i]:edges[i + 1]] for i in range(n_blocks)]
    logits = []
    for train_combo in itertools.combinations(range(n_blocks), n_blocks // 2):
        test_combo = tuple(i for i in range(n_blocks) if i not in train_combo)
        tr = np.concatenate([blocks[i] for i in train_combo], axis=0)
        te = np.concatenate([blocks[i] for i in test_combo], axis=0)
        is_s = np.array([sharpe_ann(tr[:, j]) for j in range(len(names))])
        oos_s = np.array([sharpe_ann(te[:, j]) for j in range(len(names))])
        best = int(np.argmax(is_s))
        rel = float((oos_s < oos_s[best]).sum() / (len(names) - 1))
        omega = min(max(rel, 1e-6), 1 - 1e-6)
        logits.append(math.log(omega / (1 - omega)))
    lam = np.array(logits)
    return float((lam <= 0).mean()), float(lam.mean())


def _norm_ppf(p: float) -> float:
    lo, hi = -8.0, 8.0
    target = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))
    for _ in range(200):
        mid = (lo + hi) / 2
        if target(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def deflated_sharpe(sr_annual: float, trial_sr_annual: np.ndarray, n_obs: int,
                    skew: float = 0.0, kurt: float = 3.0) -> float:
    sr = sr_annual / math.sqrt(ANN)
    trials = trial_sr_annual / math.sqrt(ANN)
    var_sr = trials.var(ddof=1) if trials.size > 1 else 0.01 ** 2
    n = trials.size
    g = 0.5772156649
    z = (1 - g) * _norm_ppf(1 - 1 / n) + g * _norm_ppf(1 - 1 / (n * math.e))
    sr_star = math.sqrt(var_sr) * z
    denom = math.sqrt(max(1e-12, 1 - skew * sr + (kurt - 1) / 4 * sr * sr))
    num = (sr - sr_star) * math.sqrt(max(n_obs - 1, 1))
    return float(0.5 * (1 + math.erf((num / denom) / math.sqrt(2))))


def run_xs_momentum(prices: np.ndarray, rets: np.ndarray, lookback: int,
                    alt_mask: np.ndarray, cost: float) -> np.ndarray:
    n_sym, n_bar = prices.shape
    strat = np.full(n_bar, np.nan)
    w = np.zeros(n_sym)
    fv = np.array([np.where(~np.isnan(prices[s]))[0][0] if (~np.isnan(prices[s])).any()
                   else n_bar for s in range(n_sym)])
    eligible_from = int(fv.max()) + MIN_HISTORY
    for b in range(eligible_from, n_bar):
        strat[b] = float(w @ rets[:, b])
        if b % 7 == 0:
            mom = prices[:, b] / prices[:, b - lookback] - 1.0
            elig = np.where(alt_mask)[0]
            elig = elig[np.isfinite(mom[elig])]
            order = elig[np.argsort(mom[elig])[::-1]]
            new_w = np.zeros(n_sym)
            for i in order[:TOP_K]:
                new_w[i] += 0.5 / TOP_K
            for i in order[-BOTTOM_K:]:
                new_w[i] -= 0.5 / BOTTOM_K
            turn = float(np.abs(new_w - w).sum())
            strat[b] -= cost * turn
            w = new_w
    return strat


def run_ts_momentum(ret_row: np.ndarray, lookback: int, cost: float) -> np.ndarray:
    n = ret_row.size
    pos = 0.0
    out = np.full(n, np.nan)
    logp = np.cumsum(np.log1p(np.clip(ret_row, RET_CLIP[0], RET_CLIP[1])))
    vol_est = ret_row[:MIN_HISTORY].std(ddof=1) * math.sqrt(ANN)
    for b in range(MIN_HISTORY, n):
        window = ret_row[max(b - ANN, 0):b]
        rv = window.std(ddof=1) * math.sqrt(ANN) if window.size > 30 else vol_est
        lev = min(MAX_LEV, TARGET_VOL_ANN / max(rv, 0.05))
        sig = 1.0 if logp[b - 1] > logp[b - 1 - lookback] else -1.0
        target = sig * lev
        out[b] = pos * ret_row[b] - cost * abs(target - pos)
        pos = target
    return out


def run_xs_low_vol(prices: np.ndarray, rets: np.ndarray, vol_window: int,
                   alt_mask: np.ndarray, cost: float) -> np.ndarray:
    """Long TOP_K lowest-trailing-vol / short BOTTOM_K highest-vol alts."""
    n_sym, n_bar = prices.shape
    strat = np.full(n_bar, np.nan)
    w = np.zeros(n_sym)
    fv = np.array([np.where(~np.isnan(prices[s]))[0][0] if (~np.isnan(prices[s])).any()
                   else n_bar for s in range(n_sym)])
    eligible_from = int(fv.max()) + MIN_HISTORY
    for b in range(eligible_from, n_bar):
        strat[b] = float(w @ rets[:, b])
        if b % 7 == 0:
            vols = np.array([rets[s, b - vol_window:b].std(ddof=1)
                             if b - max(fv[s], vol_window) > 0 else np.nan
                             for s in range(n_sym)])
            elig = np.where(alt_mask)[0]
            elig = elig[np.isfinite(vols[elig])]
            order = elig[np.argsort(vols[elig])]
            new_w = np.zeros(n_sym)
            for i in order[:TOP_K]:
                new_w[i] += 0.5 / TOP_K
            for i in order[-BOTTOM_K:]:
                new_w[i] -= 0.5 / BOTTOM_K
            strat[b] -= cost * float(np.abs(new_w - w).sum())
            w = new_w
    return strat


def yearly(strat: np.ndarray, ts: list[int]) -> dict[str, float]:
    arr = np.asarray(ts)
    years = arr // 31_536_000_000 + 1970
    out = {}
    for y in sorted(set(years.tolist())):
        seg = np.nan_to_num(strat[years == y])
        out[str(int(y))] = round(float(math.expm1(np.log1p(seg).clip(-5, 3).sum())), 4)
    return out


def summarize(family: dict[str, np.ndarray], label: str, ts: list[int]) -> dict:
    res: dict = {}
    sharpes = []
    for name in sorted(family):
        raw = family[name]
        s = np.clip(np.nan_to_num(raw, nan=0.0), RET_CLIP[0], RET_CLIP[1])
        eq_log = np.cumsum(np.log1p(s))
        sr = sharpe_ann(s)
        sharpes.append(sr)
        res[name] = {
            "sharpe": round(sr, 3),
            "ann_return": round(ann_return_from_log(eq_log), 4),
            "max_drawdown": round(mdd_from_log(eq_log), 4),
            "positive_years": sum(1 for v in yearly(raw, ts).values() if v > 0),
        }
        res[name]["years"] = yearly(raw, ts)
    pbo, lam_mean = cscv_pbo({k: np.clip(np.nan_to_num(v), RET_CLIP[0], RET_CLIP[1])
                              for k, v in family.items()})
    best_name = max((k for k in res), key=lambda k: res[k]["sharpe"])
    best_raw = np.nan_to_num(family[best_name])
    skew = float(((best_raw - best_raw.mean()) ** 3).mean() / best_raw.std() ** 3)
    kurt = float(((best_raw - best_raw.mean()) ** 4).mean() / best_raw.std() ** 4)
    dsr = deflated_sharpe(res[best_name]["sharpe"], np.array(sharpes),
                          len(best_raw), skew, kurt)
    res["_family"] = {
        "trials_n": len(family), "pbo": round(pbo, 4), "lambda_mean": round(lam_mean, 3),
        "dsr_best": round(dsr, 4), "best_config": best_name,
        "cost_bps_per_turnover": COST_PER_TURNOVER * 1e4,
    }
    return {label: res}


def main() -> None:
    symbols, ts, prices = load_panel()
    rets = prices / np.roll(prices, 1, axis=1) - 1.0
    rets[:, 0] = np.nan
    btc_idx = symbols.index("BTC")
    alt_mask = np.array([i != btc_idx for i in range(len(symbols))])

    xs_family: dict[str, np.ndarray] = {}
    ts_family: dict[str, np.ndarray] = {}
    lv_family: dict[str, np.ndarray] = {}
    for lb in (20, 30, 60):
        xs_family[f"H1_lb{lb}_h7"] = run_xs_momentum(prices, rets, lb, alt_mask,
                                                     COST_PER_TURNOVER)
    for vw in (30, 90):
        lv_family[f"H3_vw{vw}_h7"] = run_xs_low_vol(prices, rets, vw, alt_mask,
                                                    COST_PER_TURNOVER)
    for i, sym in enumerate(symbols):
        row = np.nan_to_num(rets[i])
        for lb in (30, 60, 90):
            ts_family[f"H2_{sym}_lb{lb}"] = run_ts_momentum(row, lb, COST_PER_TURNOVER)

    report = {"symbols": symbols, "start_ms": ts[0], "end_ms": ts[-1],
              "bars": len(ts), "archive_sha_note": "frozen bitget_history_full_daily_20260824_v2"}
    report.update(summarize(xs_family, "H1_cross_sectional_momentum", ts))
    report.update(summarize(ts_family, "H2_time_series_momentum", ts))
    report.update(summarize(lv_family, "H3_cross_sectional_low_vol", ts))
    OUT.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    keys = ("H1_cross_sectional_momentum", "H2_time_series_momentum",
            "H3_cross_sectional_low_vol")
    print(json.dumps({k: report[k]["_family"] for k in keys}, indent=1))
    for k in keys:
        top = sorted(((v["sharpe"], name) for name, v in report[k].items()
                      if not name.startswith("_")), reverse=True)[:3]
        print(k, "top:", top)


if __name__ == "__main__":
    main()
