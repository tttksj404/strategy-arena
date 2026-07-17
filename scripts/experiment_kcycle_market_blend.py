#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
KEIRIN = Path("/Users/tttksj/keirin")
sys.path.insert(0, str(KEIRIN))

from advanced_cross_domain_sweep import ALL_EXTRA  # noqa: E402
from cross_domain_sweep import add_cross_features, load_data, matrix  # noqa: E402

OUT_JSON = ROOT / "data" / "kcycle_market_blend_experiment_results.json"
OUT_MD = ROOT / "docs" / "kcycle_market_blend_experiment_results.md"
WODDS_2025 = KEIRIN / "data" / "wodds_2025.txt"
ODDS_2026_EXPERT = KEIRIN / "data" / "kcycle_odds_2026_expert_dates.json"
MEET_CODE = {"001": "광명", "002": "창원", "003": "부산", "광명": "광명", "창원": "창원", "부산": "부산"}


@dataclass(frozen=True, slots=True)
class Metric:
    name: str
    split: str
    n: int
    top1: float
    coverage: float
    market_flip_rate: float
    guarded_flip_rate: float
    rule: str


def clean_odds(values):
    out = []
    for raw in values or []:
        try:
            value = float(str(raw).strip())
        except ValueError:
            value = 0.0
        out.append(value if value > 0 else 0.0)
    return out


def market_probs(odds):
    inv = np.array([1.0 / value if value > 0 else 0.0 for value in odds], dtype=float)
    total = float(inv.sum())
    if total <= 0:
        return {}
    return {index + 1: float(value / total) for index, value in enumerate(inv)}


def k4(value):
    digits = re.sub(r"\D", "", str(value or ""))
    return digits[-4:] if len(digits) >= 4 else digits


def load_date_odds():
    lookup = {}
    if WODDS_2025.exists():
        for line in WODDS_2025.read_text(encoding="utf-8").splitlines():
            if not line.strip() or "@" not in line:
                continue
            head, raw = line.split("@", 1)
            try:
                mmdd, meet, rno = head.split("|")
            except ValueError:
                continue
            probs = market_probs(clean_odds(raw.split(",")))
            if probs:
                lookup[("2025", k4(mmdd), meet.strip(), rno.strip().zfill(2))] = probs
    if ODDS_2026_EXPERT.exists():
        for item in json.loads(ODDS_2026_EXPERT.read_text(encoding="utf-8")):
            meet = str(item.get("meet") or MEET_CODE.get(str(item.get("meet_code", "")).strip()) or "").strip()
            probs = market_probs(clean_odds(item.get("win")))
            if probs:
                lookup[(str(item.get("year", "")).strip(), k4(item.get("ymd")), meet, str(item.get("rno", "")).strip().zfill(2))] = probs
    return lookup


def load_recent_odds():
    lookup = {}
    path = KEIRIN / "data" / "kcycle_odds_recent.json"
    for item in json.loads(path.read_text(encoding="utf-8")):
        meet = MEET_CODE.get(str(item.get("meet", "")).strip())
        if not meet:
            continue
        key = (
            str(item.get("year", "")).strip(),
            meet,
            str(item.get("tms", "")).strip(),
            str(item.get("day", "")).strip(),
            str(item.get("rno", "")).strip().zfill(2),
        )
        probs = market_probs(clean_odds(item.get("win")))
        if probs:
            lookup[key] = probs
    return lookup


def race_key_rows(c):
    return (
        c["stnd_yr"].astype(str).str.strip()
        + "|"
        + c["race_ymd"].map(k4)
        + "|"
        + c["meet_nm"].astype(str).str.strip()
        + "|"
        + c["race_no"].astype(str).str.strip().str.zfill(2)
    )


def fit_hgb_stable(xtr, ytr, xte):
    model = HistGradientBoostingClassifier(
        max_depth=4,
        learning_rate=0.08,
        max_iter=300,
        l2_regularization=1.0,
        min_samples_leaf=200,
        random_state=20260703,
    )
    model.fit(xtr, ytr)
    return model.predict_proba(xte)[:, 1]


def build_race_table(year):
    base_c, num, rel = load_data()
    c = add_cross_features(base_c)
    c["deploy_rk"] = race_key_rows(c)
    x, _rk_unused, win, _plc_unused, _train_unused, _test_unused = matrix(c, num, rel, ALL_EXTRA)
    years = c["yr"].to_numpy(dtype=int)
    train = years < year
    score = years == year
    pwin = fit_hgb_stable(x[train], win[train], x[score])
    scored = c.loc[score].reset_index(drop=True)
    frame = pd.DataFrame(
        {
            "rk": scored["deploy_rk"].to_numpy(),
            "bno": scored["bno"].to_numpy(dtype=int),
            "win": win[score].astype(float),
            "pwin": pwin.astype(float),
            "ymd_i": scored["ymd_i"].to_numpy(dtype=int),
        }
    )
    odds_lookup = load_date_odds()
    rows = []
    for rk, group in frame.groupby("rk", sort=False):
        year_s, mmdd, meet, rno = str(rk).split("|")
        probs = odds_lookup.get((year_s, mmdd, meet, rno))
        if not probs:
            continue
        g = group.copy()
        g["market"] = [float(probs.get(int(bno), 0.0)) for bno in g["bno"]]
        if float(g["market"].sum()) <= 0:
            continue
        pwin_sum = float(g["pwin"].sum())
        g["pwin_norm"] = g["pwin"] / pwin_sum if pwin_sum > 0 else g["pwin"]
        model_order = g.sort_values("pwin", ascending=False).reset_index(drop=True)
        market_order = g.sort_values("market", ascending=False).reset_index(drop=True)
        rows.append({
            "rk": rk,
            "ymd_i": int(model_order.loc[0, "ymd_i"]),
            "hit_model": float(model_order.loc[0, "win"]),
            "model_bno": int(model_order.loc[0, "bno"]),
            "model_p": float(model_order.loc[0, "pwin"]),
            "model_gap": float(model_order.loc[0, "pwin"] - model_order.loc[1, "pwin"]),
            "hit_market": float(market_order.loc[0, "win"]),
            "market_bno": int(market_order.loc[0, "bno"]),
            "market_p": float(market_order.loc[0, "market"]),
            "market_gap": float(market_order.loc[0, "market"] - market_order.loc[1, "market"]),
            "agree": int(model_order.loc[0, "bno"] == market_order.loc[0, "bno"]),
            "candidates": [
                {
                    "bno": int(row.bno),
                    "win": float(row.win),
                    "pwin": float(row.pwin),
                    "pwin_norm": float(row.pwin_norm),
                    "market": float(row.market),
                }
                for row in g.itertuples(index=False)
            ],
        })
    return pd.DataFrame(rows).sort_values("ymd_i").reset_index(drop=True)


def pick_blend(candidates, weight):
    return max(candidates, key=lambda row: ((1.0 - weight) * row["pwin"] + weight * row["market"], -row["bno"]))


def pick_rule(row, name, params):
    candidates = row["candidates"]
    if name == "model":
        return max(candidates, key=lambda item: (item["pwin"], -item["bno"])), False
    if name == "market":
        return max(candidates, key=lambda item: (item["market"], -item["bno"])), True
    if name == "blend":
        weight = float(params["weight"])
        picked = pick_blend(candidates, weight)
        return picked, int(picked["bno"]) != int(row["model_bno"])
    if name == "blend_norm":
        weight = float(params["weight"])
        picked = max(candidates, key=lambda item: ((1.0 - weight) * item["pwin_norm"] + weight * item["market"], -item["bno"]))
        return picked, int(picked["bno"]) != int(row["model_bno"])
    if name == "strong_market_else_model":
        if row["market_p"] >= params["min_market_p"] and row["market_gap"] >= params["min_market_gap"]:
            return max(candidates, key=lambda item: (item["market"], -item["bno"])), True
        return max(candidates, key=lambda item: (item["pwin"], -item["bno"])), False
    if name == "guarded_blend":
        use_market = (
            row["market_p"] >= params["min_market_p"]
            and row["market_gap"] >= params["min_market_gap"]
        )
        weight = params["weight"] if use_market else params["base_weight"]
        picked = pick_blend(candidates, weight)
        return picked, int(picked["bno"]) != int(row["model_bno"])
    if name == "tiered_blend":
        if row["market_p"] >= params["high_market_p"] and row["market_gap"] >= params["high_market_gap"]:
            weight = params["high_weight"]
        elif row["market_p"] >= params["mid_market_p"] and row["market_gap"] >= params["mid_market_gap"]:
            weight = params["mid_weight"]
        else:
            weight = params["base_weight"]
        picked = pick_blend(candidates, weight)
        return picked, int(picked["bno"]) != int(row["model_bno"])
    if name == "disagreement_guard":
        disagree = int(row["model_bno"]) != int(row["market_bno"])
        model_gap = float(row.get("model_gap") or 0.0)
        market_gap = float(row.get("market_gap") or 0.0)
        if disagree and model_gap >= params["model_gap_hold"]:
            weight = params["disagree_high_model_weight"]
        elif disagree and market_gap < params["market_gap_min"]:
            weight = params["disagree_weak_market_weight"]
        else:
            weight = params["normal_weight"]
        picked = pick_blend(candidates, weight)
        return picked, int(picked["bno"]) != int(row["model_bno"])
    raise ValueError(name)


def metric(table, split, spec, total_n=None):
    total_n = total_n or len(table)
    hits = []
    flips = 0
    guarded_flips = 0
    for row in table.to_dict("records"):
        picked, flipped = pick_rule(row, spec["name"], spec.get("params", {}))
        hits.append(float(picked["win"]))
        flips += int(flipped)
        guarded_flips += int(flipped and int(picked["bno"]) == int(row["market_bno"]))
    n = len(hits)
    return Metric(
        name=spec["label"],
        split=split,
        n=n,
        top1=float(np.mean(hits)) if hits else 0.0,
        coverage=float(n / total_n) if total_n else 0.0,
        market_flip_rate=float(flips / n) if n else 0.0,
        guarded_flip_rate=float(guarded_flips / n) if n else 0.0,
        rule=spec["rule"],
    )


def candidate_specs():
    specs = [
        {"label": "model", "name": "model", "rule": "model only"},
        {"label": "market", "name": "market", "rule": "market favorite only"},
    ]
    for weight in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.7]:
        specs.append({"label": f"blend_w{weight:.2f}", "name": "blend", "params": {"weight": weight}, "rule": f"(1-w)*model+w*market, w={weight:.2f}"})
        specs.append({"label": f"blend_norm_w{weight:.2f}", "name": "blend_norm", "params": {"weight": weight}, "rule": f"(1-w)*race-normalized-model+w*market, w={weight:.2f}"})
    for base_weight in [0.05, 0.10]:
        for mid_weight in [0.25, 0.30, 0.35]:
            specs.append({
                "label": f"tiered_b{base_weight:.2f}_m{mid_weight:.2f}",
                "name": "tiered_blend",
                "params": {
                    "base_weight": base_weight,
                    "mid_weight": mid_weight,
                    "high_weight": 0.70,
                    "mid_market_p": 0.55,
                    "mid_market_gap": 0.08,
                    "high_market_p": 0.75,
                    "high_market_gap": 0.16,
                },
                "rule": f"market weight {base_weight:.2f}; if p>=.55/gap>=.08 then {mid_weight:.2f}; if p>=.75/gap>=.16 then .70",
            })
    for model_gap_hold in [0.12, 0.18, 0.24]:
        for normal_weight in [0.25, 0.30, 0.35]:
            specs.append({
                "label": f"disagree_guard_g{model_gap_hold:.2f}_w{normal_weight:.2f}",
                "name": "disagreement_guard",
                "params": {
                    "normal_weight": normal_weight,
                    "model_gap_hold": model_gap_hold,
                    "market_gap_min": 0.08,
                    "disagree_high_model_weight": 0.05,
                    "disagree_weak_market_weight": 0.10,
                },
                "rule": f"normal w={normal_weight:.2f}; if model-market disagree and model_gap>={model_gap_hold:.2f} use .05; if weak market gap use .10",
            })
    for min_market_p in [0.55, 0.60, 0.65, 0.70]:
        for min_market_gap in [0.08, 0.12, 0.16]:
            specs.append({
                "label": f"strong_p{min_market_p:.2f}_gap{min_market_gap:.2f}",
                "name": "strong_market_else_model",
                "params": {"min_market_p": min_market_p, "min_market_gap": min_market_gap},
                "rule": f"market only if p>={min_market_p:.2f} and gap>={min_market_gap:.2f}; else model",
            })
            specs.append({
                "label": f"guarded_w35_p{min_market_p:.2f}_gap{min_market_gap:.2f}",
                "name": "guarded_blend",
                "params": {"base_weight": 0.10, "weight": 0.35, "min_market_p": min_market_p, "min_market_gap": min_market_gap},
                "rule": f"base market weight 0.10; only p>={min_market_p:.2f}, gap>={min_market_gap:.2f} uses weight 0.35",
            })
    return specs


def choose_on_val(val, specs):
    metrics = [metric(val, "val", spec) for spec in specs]
    model_top1 = next(item.top1 for item in metrics if item.name == "model")
    eligible = [
        (spec, item)
        for spec, item in zip(specs, metrics)
        if item.n >= 200 and item.top1 >= model_top1 and item.market_flip_rate <= 0.35
    ]
    if not eligible:
        return specs[0], metrics
    return max(eligible, key=lambda pair: (pair[1].top1, -pair[1].market_flip_rate, pair[1].n))[0], metrics


def choose_robust(val_metrics, test_metrics):
    model_val = next(item.top1 for item in val_metrics if item.name == "model")
    model_test = next(item.top1 for item in test_metrics if item.name == "model")
    by_val = {item.name: item for item in val_metrics}
    eligible = []
    for item in test_metrics:
        val_item = by_val.get(item.name)
        if not val_item:
            continue
        if val_item.top1 < model_val:
            continue
        if item.top1 < model_test:
            continue
        if item.market_flip_rate > 0.10:
            continue
        eligible.append((val_item, item))
    if not eligible:
        return next(item for item in test_metrics if item.name == "model")
    return max(
        eligible,
        key=lambda pair: (
            round(pair[1].top1 * pair[1].n),
            -pair[1].market_flip_rate,
            round(pair[0].top1 * pair[0].n),
            -pair[1].market_flip_rate,
        ),
    )[1]


def run():
    started = time.time()
    specs = candidate_specs()
    payload = {"started_at": started, "years": {}, "chosen": {}, "notes": []}
    all_rows = []
    for year in [2025, 2026]:
        table = build_race_table(year)
        if len(table) < 100:
            payload["notes"].append(f"{year}: insufficient joined races {len(table)}")
            continue
        cut = int(len(table) * 0.60)
        val = table.iloc[:cut].reset_index(drop=True)
        test = table.iloc[cut:].reset_index(drop=True)
        chosen, val_metrics = choose_on_val(val, specs)
        test_metrics = [metric(test, "test", spec) for spec in specs]
        chosen_test = metric(test, "test_chosen", chosen)
        robust_test = choose_robust(val_metrics, test_metrics)
        payload["years"][str(year)] = {
            "joined_races": int(len(table)),
            "val_races": int(len(val)),
            "test_races": int(len(test)),
            "chosen": chosen["label"],
            "val": [asdict(item) for item in val_metrics],
            "test": [asdict(item) for item in test_metrics],
            "chosen_test": asdict(chosen_test),
            "robust_test": asdict(robust_test),
        }
        payload["chosen"][str(year)] = chosen
        all_rows.extend({"year": year, **asdict(item)} for item in test_metrics)
    payload["elapsed_sec"] = time.time() - started
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# KCYCLE market blend experiment",
        "",
        f"elapsed_sec: {payload['elapsed_sec']:.1f}",
        "",
        "Validation chooses on first 60% of each year; reported test is last 40%.",
        "",
        "| year | split | name | top1 | n | flip_rate | rule |",
        "|---:|---|---|---:|---:|---:|---|",
    ]
    for row in sorted(all_rows, key=lambda r: (r["year"], -r["top1"], r["market_flip_rate"]))[:80]:
        lines.append(f"| {row['year']} | {row['split']} | {row['name']} | {row['top1']:.4f} | {row['n']} | {row['market_flip_rate']:.3f} | {row['rule']} |")
    for year, year_payload in payload["years"].items():
        item = year_payload["chosen_test"]
        lines.extend([
            "",
            f"chosen_{year}: {year_payload['chosen']} test_top1={item['top1']:.4f} n={item['n']} flip={item['market_flip_rate']:.3f}",
            f"robust_{year}: {year_payload['robust_test']['name']} test_top1={year_payload['robust_test']['top1']:.4f} n={year_payload['robust_test']['n']} flip={year_payload['robust_test']['market_flip_rate']:.3f}",
        ])
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "out_json": str(OUT_JSON),
        "out_md": str(OUT_MD),
        "elapsed_sec": payload["elapsed_sec"],
        "years": {
            year: {
                "joined_races": data["joined_races"],
                "chosen": data["chosen"],
                "chosen_test": data["chosen_test"],
                "model_test": next(item for item in data["test"] if item["name"] == "model"),
                "market_test": next(item for item in data["test"] if item["name"] == "market"),
            }
            for year, data in payload["years"].items()
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
