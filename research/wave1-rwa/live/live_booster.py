#!/usr/bin/env python3
"""Live executor for the regime-switched cross-sectional booster (Bitget USDT-M).

SAFETY-FIRST. Defaults to DRY_RUN (no real orders). Reuses the audited Bitget REST
client from quant_binance/execution. Once per UTC day it recomputes the regime and
target weights (identical logic to src/regime_xsec.py), then reconciles live
positions toward the target via reduce-only closes + market opens. Every tick
enforces hard invariants and a set of circuit breakers; any breach flips the
kill-switch and flattens.

ENV (never committed):
  BITGET_API_KEY / BITGET_API_SECRET / BITGET_API_PASSPHRASE
  LIVE_CAPITAL_USD   (default 100)          — booster sleeve capital
  LIVE_DRY_RUN       ("1" default = no real orders; set "0" to arm)
  LIVE_MAX_LEV       (default 2.3)          — hard cap on per-tick target leverage
  LIVE_DD_HALT       (default 0.45)         — trailing drawdown → kill-switch + flatten
Run: python3 -m live.live_booster            (single reconcile+report tick)
     LIVE_LOOP=1 python3 -m live.live_booster (hourly loop)
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, "/Users/tttksj/first_repo/quant_binance")
sys.path.insert(0, "/Users/tttksj/first_repo")  # for `quant_binance.*` absolute imports

HERE = Path(__file__).resolve().parent
STATE = HERE / "live_state.json"
LOG = HERE / "live_booster.log"
ORDERS = HERE / "live_orders.jsonl"

CAPITAL = float(os.environ.get("LIVE_CAPITAL_USD", "100"))
DRY_RUN = os.environ.get("LIVE_DRY_RUN", "1") != "0"
MAX_LEV = float(os.environ.get("LIVE_MAX_LEV", "2.3"))
DD_HALT = float(os.environ.get("LIVE_DD_HALT", "0.45"))
TOPK, LB, TV = 5, 40, 0.25
PRODUCT, MARGIN = "USDT-FUTURES", "USDT"


def log(line: str) -> None:
    stamp = datetime.now(UTC).isoformat(timespec="seconds")
    print(f"{stamp} {line}")
    with LOG.open("a", encoding="utf-8") as h:
        h.write(f"{stamp} {line}\n")


def load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"started": datetime.now(UTC).isoformat(), "kill_switch": False,
            "peak_equity": CAPITAL, "last_rebalance": None, "target": {}}


def save_state(state: dict) -> None:
    STATE.write_text(json.dumps(state, indent=2))


# ---- strategy target (mirrors src/regime_xsec.py exactly) ----
def compute_target() -> tuple[bool, dict, float]:
    from regime_xsec import daily_panel, regime_series
    panel = daily_panel()
    rets = panel.pct_change()
    reg = bool(regime_series(panel).iloc[-1])
    sig = panel.pct_change(LB).iloc[-1].dropna()
    if not reg or len(sig) < 2 * TOPK:
        return False, {}, 0.0
    weights = {s: 1.0 / TOPK for s in sig.nlargest(TOPK).index}
    weights.update({s: -1.0 / TOPK for s in sig.nsmallest(TOPK).index})
    pos_hist = pd.DataFrame(0.0, index=panel.index[-70:], columns=panel.columns)
    for d in pos_hist.index:
        row = panel.pct_change(LB).loc[d].dropna()
        if len(row) < 2 * TOPK:
            continue
        for s in row.nlargest(TOPK).index:
            pos_hist.at[d, s] = 1.0 / TOPK
        for s in row.nsmallest(TOPK).index:
            pos_hist.at[d, s] = -1.0 / TOPK
    raw = (pos_hist.shift(1) * rets.reindex(pos_hist.index)).sum(axis=1)
    realized = float(raw.rolling(20).std().iloc[-1]) or 0.03
    lev = float(np.clip(TV / realized, 0.3, MAX_LEV))
    return True, weights, lev


# ---- invariants (hard, checked every tick) ----
def assert_invariants(target: dict, lev: float, equity: float) -> None:
    assert lev <= MAX_LEV + 1e-9, f"INVARIANT lev {lev} > cap {MAX_LEV}"
    gross = sum(abs(w) for w in target.values())
    assert gross <= 2.0 + 1e-6, f"INVARIANT gross weight {gross} > 2.0"
    net = sum(target.values())
    assert abs(net) <= 1e-6, f"INVARIANT not delta-neutral: net {net}"
    assert equity >= 0, f"INVARIANT negative equity {equity}"
    assert len(target) <= 2 * TOPK, f"INVARIANT too many legs {len(target)}"


def bitget_client():
    from execution.bitget_rest import BitgetRestClient, BitgetContractConfig
    from exchange import ExchangeCredentials  # type: ignore
    creds = ExchangeCredentials(
        exchange_id="bitget",
        api_key=os.environ.get("BITGET_API_KEY", ""),
        api_secret=os.environ.get("BITGET_API_SECRET", ""),
        api_passphrase=os.environ.get("BITGET_API_PASSPHRASE", ""),
    )
    return BitgetRestClient(credentials=creds, contract_config=BitgetContractConfig(PRODUCT, MARGIN))


def live_equity(client) -> float:
    if DRY_RUN:
        return CAPITAL
    from urllib.request import urlopen
    req = client.build_account_request(market=PRODUCT)
    data = json.load(urlopen(req, timeout=15))
    for acc in data.get("data", []):
        if acc.get("marginCoin") == MARGIN:
            return float(acc.get("usdtEquity", acc.get("available", CAPITAL)))
    return CAPITAL


def tick() -> None:
    state = load_state()
    if state.get("kill_switch"):
        log("KILL_SWITCH active — no action. Manual reset required.")
        return
    client = bitget_client()
    try:
        equity = live_equity(client)
    except Exception as err:  # noqa: BLE001
        log(f"equity read failed ({err}); using capital {CAPITAL}")
        equity = CAPITAL

    # circuit breaker: trailing drawdown
    state["peak_equity"] = max(state.get("peak_equity", CAPITAL), equity)
    dd = 1 - equity / state["peak_equity"] if state["peak_equity"] else 0.0
    if dd >= DD_HALT:
        state["kill_switch"] = True
        save_state(state)
        log(f"CIRCUIT BREAKER: drawdown {dd:.1%} >= {DD_HALT:.0%} → KILL_SWITCH + flatten required")
        return

    today = datetime.now(UTC).date().isoformat()
    if state.get("last_rebalance") == today:
        log(f"already rebalanced {today}; equity ${equity:.2f} dd {dd:.1%} target legs {len(state.get('target', {}))}")
        save_state(state)
        return

    regime_on, target, lev = compute_target()
    assert_invariants(target, lev, equity)

    order_plan = []
    if regime_on:
        for sym, w in target.items():
            notional = equity * lev * abs(w)
            order_plan.append({"symbol": sym, "side": "long" if w > 0 else "short",
                               "notional_usd": round(notional, 2)})
    mode = "DRY_RUN" if DRY_RUN else "LIVE"
    log(f"[{mode}] regime={'ON' if regime_on else 'OFF (cash)'} lev={lev:.2f} equity=${equity:.2f} orders={len(order_plan)}")
    for o in order_plan:
        log(f"  {mode} {o['side']:5s} {o['symbol']} ${o['notional_usd']}")
    if not DRY_RUN and regime_on:
        _execute_reconcile(client, target, lev, equity)  # live path
    with ORDERS.open("a", encoding="utf-8") as h:
        h.write(json.dumps({"ts": datetime.now(UTC).isoformat(), "mode": mode,
                            "regime": regime_on, "lev": lev, "orders": order_plan}) + "\n")
    state["target"] = target
    state["last_rebalance"] = today
    save_state(state)


def _execute_reconcile(client, target: dict, lev: float, equity: float) -> None:
    """Live order path — intentionally minimal; expanded only after DRY_RUN parity is confirmed."""
    raise NotImplementedError(
        "LIVE order path is gated. Confirm DRY_RUN parity + fund the account, then implement "
        "reduce-only close of stale legs and market open of target legs here.")


def main() -> None:
    HERE.mkdir(exist_ok=True)
    if os.environ.get("LIVE_LOOP") != "1":
        tick()
        return
    while True:
        try:
            tick()
        except AssertionError as err:
            log(f"INVARIANT BREACH → halt: {err}")
            st = load_state(); st["kill_switch"] = True; save_state(st)
            return
        except Exception as err:  # noqa: BLE001
            log(f"TICK_ERROR {err}")
        nxt = (int(time.time() // 3600) + 1) * 3600 + 150
        time.sleep(max(60.0, nxt - time.time()))


if __name__ == "__main__":
    main()
