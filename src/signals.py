from __future__ import annotations

import numpy as np
import pandas as pd


def positions_from_zscore(zscore: pd.Series, entry: float, exit: float) -> pd.Series:
    """Return spread position: +1 long spread, -1 short spread, 0 flat."""
    position = 0
    out = []
    for z in zscore:
        if np.isnan(z):
            out.append(0)
            continue
        if position == 0 and z <= -entry:
            position = 1
        elif position == 0 and z >= entry:
            position = -1
        elif position != 0 and abs(z) <= exit:
            position = 0
        out.append(position)
    return pd.Series(out, index=zscore.index, name="signal")
