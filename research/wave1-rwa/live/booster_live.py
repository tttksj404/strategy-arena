#!/usr/bin/env python3
"""Booster live executor — regime-switched cross-sectional momentum on Bitget USDT-M.

Safety-first design (constitution HIGH-risk WRITE gates):
  * LIVE guard: orders are sent only when env LIVE=1 AND live/CONFIRM_LIVE exists.
    Otherwise every run is a dry-run that logs intended orders and touches nothing.
  * Kill switch: live/KILL file → flatten (live) and halt.
  * Circuit breakers: equity < 75% of high-water, or daily loss <= -20% → flatten+halt.
  * Runtime invariants (asserted every run, violations abort before any order):
      I1 gross planned notional <= equity * MAX_GROSS_LEV * 1.05
      I2 per-symbol notional <= equity * PER_SYMBOL_CAP
      I3 regime OFF → target book