from pathlib import Path
import sys

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path.cwd()))
from research.wave8_alternative import run_wave8 as w8


N_DAYS = 70
INDEX = pd.date_range("2025-01-01", periods=N_DAYS, freq="D", tz="UTC")
SYMBOLS = list(w8.SYMBOLS)

close = pd.DataFrame(
    {
        symbol: 100.0 + np.arange(N_DAYS) * 0.2 + symbol_index * 0.1
        for symbol_index, symbol in enumerate(SYMBOLS)
    },
    index=INDEX,
)
volume = pd.DataFrame(
    {
        symbol: 100.0 + (np.arange(N_DAYS) % 10) * 2 + symbol_index
        for symbol_index, symbol in enumerate(SYMBOLS)
    },
    index=INDEX,
)

volume.iloc[-2, volume.columns.get_loc("BTC")] = 1000.0
base = w8._volume_signal(close, volume, 1, reversal=False)

mutated_volume = volume.copy()
mutated_volume.iloc[-1, mutated_volume.columns.get_loc("BTC")] = 1_000_000.0
mutated = w8._volume_signal(close, mutated_volume, 1, reversal=False)

probe = INDEX[-1]
base_row = base.loc[probe]
mutated_row = mutated.loc[probe]
row_l1_diff = float((base_row - mutated_row).abs().sum())
print(f"probe={probe.isoformat()}")
print(f"base_btc={base_row['BTC']:.12f} mutated_btc={mutated_row['BTC']:.12f}")
print(f"row_l1_diff={row_l1_diff:.12f}")
print(
    "future_mutation_invariant=",
    bool(np.allclose(base_row.to_numpy(), mutated_row.to_numpy())),
)
