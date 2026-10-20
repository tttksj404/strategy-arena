from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from research.wave3.engine import AssetMarket, CandidateConfig, run_candidate
from research.wave3.universe import AssetListing


dates = pd.date_range("2026-01-01", periods=70, freq="D", tz="UTC")
funding_dates = pd.date_range("2026-01-01", periods=210, freq="8h", tz="UTC")
markets: dict[str, AssetMarket] = {}
for offset, symbol in enumerate(("AAAUSDT", "BBBUSDT", "CCCUSDT")):
    close = pd.Series(
        100.0 + offset + dates.dayofyear.to_numpy() * (offset + 1) * 0.1,
        index=dates,
    )
    bars = pd.DataFrame({"close": close, "quote_volume": 100.0 + offset}, index=dates)
    listing = AssetListing(symbol, "crypto", pd.Timestamp("2025-01-01", tz="UTC"), True, True)
    markets[symbol] = AssetMarket(listing, bars, bars, pd.Series(0.0001, index=funding_dates))

result = run_candidate(markets, CandidateConfig("W3e", "max_z", 3, 1, hysteresis=True))
assert result.candidate_id == "W3e"
assert len(result.equity) == 70
assert result.equity.notna().all()
print("PUBLIC_RUN_CANDIDATE=PASS")
print(f"candidate_id={result.candidate_id}")
print(f"equity_len={len(result.equity)}")
print(f"equity_all_non_null={result.equity.notna().all()}")
