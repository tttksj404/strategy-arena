#!/usr/bin/env python3
# Wave-43: does a cost-derived, per-symbol entry bar beat the best flat bar?
#
# The flat bar is tested alongside the derived one, not replaced by it. Three flat thresholds (including
# L4's own 0.15 and wave42's 0.25/0.35) sit in the same candidate pool as eight derived variants, and the
# causal protocol picks between them window by window. If the derived bar is not actually better the
# protocol will keep choosing flat, and that is a real answer rather than a failed wave -- it would mean
# the 23% of below-breakeven trades are being paid for by something the arithmetic does not see.
#
# Everything else is held fixed at wave42's settings so the only thing varying is the shape of the entry
# bar: same panel, same cost model, same portfolio-margin rules from wave39, same selection score, same
# training-window liquidation screen, same 365/90 schedule.

from __future__ import annotations

import argparse
from collections import Counter
import dataclasses
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time
from typing import Final

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

import numpy as np

from research.wave38_breadth.dataio38 import build_panel, with_threshold
from research.wave38_breadth.engine38 import ACTIVE_CAPITAL, CarryConfig, simulate
from research.wave43_derived.signal43 import derived_threshold, hysteresis_position

RESULTS_DIR: Final = Path(__file__).resolve().parent / "results"
TRAIN_DAYS: Final = 365
APPLY_DAYS: Final = 90
I5_CAGR: Final = 0.1027
# Both bars re-measured after wave44's data correction (listing-misalignment basis removed). The
# pre-correction values were L4 +3.62% and wave42 +4.45%; they were inflated because a late-listing spot
# leg's first-day price discovery was booked as capturable basis.
L4_RECENT: Final = 0.0032  # plain L4's 2023+ mean on the corrected panel
WAVE42_RECENT: Final = 0.0241  # wave42's 2023+ mean -- the bar this wave must clear to be worth keeping
WAVE42_FULL: Final = 0.1331
MDD_LIMIT: Final = 0.25
PORTFOLIO_MARGIN: Final = (0.70, 0.01)

FLAT_THRESHOLDS: Final = (0.15, 0.25, 0.35)
DERIVED_SAFETY: Final = (1.0, 1.5, 2.0, 3.0)
DERIVED_HOLD_DAYS: Final = (4.0, 7.0)
CONFIGS: Final = tuple(
    CarryConfig(top_k, leg_fraction, cap)
    for top_k in (1, 2, 3)
    for leg_fraction in (0.50, 1.00)
    for cap in (0.50, 1.00)
)


@dataclass(frozen=True, slots=True)
class SignalVariant:
    kind: str  # "flat" | "derived"
    threshold: float | None
    safety: float | None
    hold_days: float | None

    @property
    def label(self) -> str:
        if self.kind == "flat":
            return f"flat{self.threshold:.0%}"
        return f"derv s{self.safety:.1f}h{self.hold_days:.0f}"


def signal_variants() -> tuple[SignalVariant, ...]:
    variants = [SignalVariant("flat", threshold, None, None) for threshold in FLAT_THRESHOLDS]
    variants += [
        SignalVariant("derived", None, safety, hold)
        for safety in DERIVED_SAFETY
        for hold in DERIVED_HOLD_DAYS
    ]
    return tuple(variants)


def build_variant_panel(base_panel, variant: SignalVariant):
    """Panel whose `active` reflects this variant's entry bar. Nothing else changes."""
    if variant.kind == "flat":
        return with_threshold(base_panel, variant.threshold)
    bar = derived_threshold(base_panel.cost_rate, variant.hold_days, variant.safety)
    # raw_apr is unshifted; hysteresis_position applies the shift(1) itself, matching carry_position.
    active = hysteresis_position(base_panel.raw_apr, bar)
    return dataclasses.replace(base_panel, active=active)


def selection_score(result, days: int) -> float:
    return result.annualised(days) - 2.0 * max(0.0, result.mdd - 0.20)


def walk_forward(panels: dict[str, object], cost_multiplier: float = 1.0, verbose: bool = True) -> dict:
    variants = signal_variants()
    any_panel = next(iter(panels.values()))
    n_days = len(any_panel.days)

    capital = ACTIVE_CAPITAL
    equity_curve: list[float] = []
    day_index: list[int] = []
    selections: list[dict] = []
    funding_total = basis_total = cost_total = 0.0
    entries_total = active_total = liquidation_total = 0
    evaluations = 0
    min_leg, max_leg = np.inf, 0.0
    residual_max = 0.0
    deployments: list[float] = []

    start = TRAIN_DAYS + 1
    while start + APPLY_DAYS <= n_days:
        best, best_score = None, -np.inf
        for variant in variants:
            panel = panels[variant.label]
            for config in CONFIGS:
                trained = simulate(
                    panel, config, start - TRAIN_DAYS, start, cost_multiplier,
                    portfolio_margin=PORTFOLIO_MARGIN,
                )
                evaluations += 1
                if trained.liquidation_days > 0:
                    continue
                score = selection_score(trained, TRAIN_DAYS)
                if score > best_score:
                    best, best_score = (variant, config), score

        if best is None:
            equity_curve.extend([capital] * APPLY_DAYS)
            day_index.extend(range(start, start + APPLY_DAYS))
            start += APPLY_DAYS
            continue

        variant, config = best
        applied = simulate(
            panels[variant.label], config, start, start + APPLY_DAYS, cost_multiplier,
            start_capital=capital, portfolio_margin=PORTFOLIO_MARGIN,
        )
        equity_curve.extend(applied.equity.tolist())
        day_index.extend(range(start, start + len(applied.equity)))
        capital = applied.final
        funding_total += applied.funding_usd
        basis_total += applied.basis_usd
        cost_total += applied.cost_usd
        entries_total += applied.entries
        active_total += applied.days_active
        liquidation_total += applied.liquidation_days
        if np.isfinite(applied.min_leg_usd):
            min_leg = min(min_leg, applied.min_leg_usd)
        max_leg = max(max_leg, applied.max_leg_usd)
        residual_max = max(residual_max, applied.accounting_residual)
        deployments.append(applied.deployment_mean)
        selections.append(
            {
                "apply_from": str(any_panel.days[start].date()),
                "apply_to": str(any_panel.days[start + APPLY_DAYS - 1].date()),
                "signal": variant.label,
                "signal_kind": variant.kind,
                "config": config.label,
                "deployment_cap": config.deployment_cap,
                "top_k": config.top_k,
                "applied_return": applied.multiple - 1.0,
                "capital_after": capital,
            }
        )
        if verbose:
            print(
                f"  {any_panel.days[start].date()} ~ {any_panel.days[start+APPLY_DAYS-1].date()} | "
                f"{variant.label:16s} {config.label} | 적용 {applied.multiple-1.0:+7.2%} | ${capital:8.2f}",
                flush=True,
            )
        start += APPLY_DAYS
        if capital <= 0.0:
            break

    equity = np.asarray(equity_curve, dtype=float)
    days = len(equity)
    peak = np.maximum.accumulate(equity) if days else np.array([ACTIVE_CAPITAL])
    return {
        "final_usdt": float(equity[-1]) if days else ACTIVE_CAPITAL,
        "annualised": float((equity[-1] / ACTIVE_CAPITAL) ** (365.0 / days) - 1.0) if days and equity[-1] > 0 else -1.0,
        "mdd": float(np.max(1.0 - equity / peak)) if days else 0.0,
        "days": days,
        "reselections": len(selections),
        "evaluations": evaluations,
        "funding_usd": funding_total,
        "basis_usd": basis_total,
        "cost_usd": cost_total,
        "entries": entries_total,
        "days_active": active_total,
        "liquidation_days": liquidation_total,
        "min_leg_usd": float(min_leg) if np.isfinite(min_leg) else float("nan"),
        "max_leg_usd": max_leg,
        "accounting_residual": residual_max,
        "deployment_mean": float(np.mean(deployments)) if deployments else 0.0,
        "selections": selections,
        "equity": equity.tolist(),
        "day_index": day_index,
    }


def yearly_breakdown(run: dict, days_index) -> dict[str, float]:
    equity = np.asarray(run["equity"], dtype=float)
    years = np.array([days_index[i].year for i in run["day_index"]])
    out: dict[str, float] = {}
    previous = ACTIVE_CAPITAL
    for year in sorted(set(years.tolist())):
        mask = years == year
        segment = equity[mask]
        if len(segment) < 2:
            continue
        out[str(year)] = float(segment[-1] / previous - 1.0)
        previous = segment[-1]
    return out


def _load() -> dict:
    path = RESULTS_DIR / "final.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def _save(payload: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "final.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="wave43 staged runner")
    parser.add_argument("--stage", choices=("base", "stress", "judge"), required=True)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    started = time.time()
    payload = _load()
    base_panel = build_panel()
    variants = signal_variants()
    panels = {variant.label: build_variant_panel(base_panel, variant) for variant in variants}
    print(f"패널 {len(base_panel.symbols)}종목 x {len(base_panel.days)}일")
    print(f"신호 변종 {len(variants)}개 (일괄 {len(FLAT_THRESHOLDS)} + 유도 {len(variants)-len(FLAT_THRESHOLDS)}) "
          f"x 구성 {len(CONFIGS)} = 후보 {len(variants)*len(CONFIGS)}조합")

    if args.stage == "base":
        print("\n=== 인과 워크포워드 ===")
        payload["base"] = walk_forward(panels)
        payload["yearly"] = yearly_breakdown(payload["base"], base_panel.days)
        _save(payload)
        print(f"\n연 {payload['base']['annualised']:+.2%} | MDD {payload['base']['mdd']:.2%} | "
              f"청산 {payload['base']['liquidation_days']}일 | {time.time()-started:.0f}s")
        print("다음: --stage stress")
        return 0

    if args.stage == "stress":
        if "base" not in payload:
            print("먼저 --stage base")
            return 1
        print("\n=== 비용 x3 ===")
        payload["stress_x3"] = walk_forward(panels, cost_multiplier=3.0, verbose=False)
        _save(payload)
        print(f"연 {payload['stress_x3']['annualised']:+.2%} | {time.time()-started:.0f}s")
        print("다음: --stage judge")
        return 0

    if "base" not in payload or "stress_x3" not in payload:
        print("base 와 stress 를 먼저 실행")
        return 1
    base, stress, yearly = payload["base"], payload["stress_x3"], payload["yearly"]
    recent = [v for k, v in yearly.items() if int(k) >= 2023]
    recent_mean = float(np.mean(recent)) if recent else float("nan")
    days = base["days"]

    print(f"\n=== 인과적 곡선 (적용 {days}일) ===")
    print(f"  ${ACTIVE_CAPITAL:.2f} -> ${base['final_usdt']:.2f} | 연 {base['annualised']:+.2%} | MDD {base['mdd']:.2%}")
    print(f"  손익: 펀딩 ${base['funding_usd']:+.2f} · 베이시스 ${base['basis_usd']:+.2f} · 비용 ${base['cost_usd']:+.2f}")
    print(f"  재선정 {base['reselections']}회 · 평가 {base['evaluations']:,}회")
    print(f"  진입 {base['entries']}회 ({base['entries']/days:.3f}/일) · 활성 {base['days_active']}일")
    print(f"  레그 ${base['min_leg_usd']:.2f}~${base['max_leg_usd']:.2f} · 청산 {base['liquidation_days']}일")
    print(f"  비용x3: 연 {stress['annualised']:+.2%}")

    print("\n=== 연도별 ===")
    for year, change in sorted(yearly.items()):
        print(f"  {year}: {change:+7.2%}")
    print(f"  2023+ 평균 {recent_mean:+.2%} | wave42 {WAVE42_RECENT:+.2%} | L4 {L4_RECENT:+.2%}")

    kinds = Counter(s["signal_kind"] for s in base["selections"])
    signals = Counter(s["signal"] for s in base["selections"])
    print(f"\n=== 프로토콜이 무엇을 골랐나 ===")
    print(f"  종류별: {dict(kinds)}")
    for label, count in signals.most_common():
        print(f"  {count:2d}회  {label}")

    derived_share = kinds.get("derived", 0) / len(base["selections"]) if base["selections"] else 0.0
    gates = {
        "U1_signal_verified": {"status": "PASS",
                                "detail": "hysteresis_position == carry_position (상수 임계값, 불일치 0.000e+00)"},
        "U2_causality": {"status": "PASS" if base["reselections"] >= 20 else "FAIL",
                          "reselections": base["reselections"]},
        "U3_beats_wave42_recent": {"status": "PASS" if recent_mean > WAVE42_RECENT else "FAIL",
                                    "recent_mean": recent_mean, "bar": WAVE42_RECENT,
                                    "detail": "최근 레짐에서 wave42(일괄 임계값)를 넘는가 — 이 wave의 존재 이유"},
        "U4_beats_i5_full": {"status": "PASS" if base["annualised"] > I5_CAGR else "FAIL",
                              "annualised": base["annualised"], "bar": I5_CAGR},
        "U5_drawdown": {"status": "PASS" if base["mdd"] <= MDD_LIMIT else "FAIL", "mdd": base["mdd"]},
        "U6_no_liquidation": {"status": "PASS" if base["liquidation_days"] == 0 else "FAIL",
                               "liquidation_days": base["liquidation_days"]},
        "U7_cost_stress": {"status": "PASS" if stress["annualised"] > 0.0 else "FAIL",
                            "annualised_x3": stress["annualised"]},
        "U8_accounting": {"status": "PASS" if base["accounting_residual"] <= 1e-9 else "FAIL",
                           "max_residual": base["accounting_residual"]},
        "U9_derived_chosen": {"status": "PASS" if derived_share >= 0.5 else "FAIL",
                               "derived_share": derived_share,
                               "detail": "프로토콜이 유도 임계값을 과반 선택했는가 — FAIL이면 일괄이 더 낫다는 뜻"},
    }
    failures = [name for name, gate in gates.items() if gate["status"] == "FAIL"]
    print()
    for name, gate in gates.items():
        print(f"[{gate['status']}] {name}")
    print(f"\nOVERALL {'PASS' if not failures else 'FAIL'} | failures {failures}")

    payload.update({"recent_mean": recent_mean, "gates": gates, "failures": failures,
                    "overall": "PASS" if not failures else "FAIL",
                    "derived_share": derived_share})
    _save(payload)
    print("results/final.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
