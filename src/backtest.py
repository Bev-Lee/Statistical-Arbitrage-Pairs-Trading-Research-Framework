from __future__ import annotations

import numpy as np
import pandas as pd


def run_backtest(spread: pd.Series, signal: pd.Series, commission_bps: float, slippage_bps: float,
                 target_daily_vol: float = 0.01, max_gross_exposure: float = 1.0) -> pd.DataFrame:
    """Backtest a spread. Signals are lagged one bar before any return is earned."""
    frame = pd.DataFrame({"spread": spread, "signal": signal}).dropna()
    frame["spread_return"] = frame["spread"].diff().fillna(0.0)
    rolling_vol = frame["spread_return"].rolling(20, min_periods=10).std().replace(0, np.nan)
    scale = (target_daily_vol / rolling_vol).clip(upper=max_gross_exposure).fillna(0.0)
    frame["executed_position"] = frame["signal"].shift(1).fillna(0.0) * scale.shift(1).fillna(0.0)
    frame["turnover"] = frame["executed_position"].diff().abs().fillna(frame["executed_position"].abs())
    cost_rate = (commission_bps + slippage_bps) / 10_000
    frame["cost"] = frame["turnover"] * cost_rate
    frame["gross_return"] = frame["executed_position"] * frame["spread_return"]
    frame["net_return"] = frame["gross_return"] - frame["cost"]
    frame["equity"] = (1 + frame["net_return"]).cumprod()
    frame["drawdown"] = frame["equity"] / frame["equity"].cummax() - 1
    return frame
