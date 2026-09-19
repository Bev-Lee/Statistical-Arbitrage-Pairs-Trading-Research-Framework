from __future__ import annotations

import numpy as np
import pandas as pd


def performance_metrics(backtest: pd.DataFrame, annualization: int = 252) -> dict[str, float]:
    returns = backtest["net_return"]
    volatility = returns.std(ddof=0) * np.sqrt(annualization)
    annual_return = (1 + returns).prod() ** (annualization / max(len(returns), 1)) - 1
    sharpe = (returns.mean() / returns.std(ddof=0) * np.sqrt(annualization)) if returns.std(ddof=0) else np.nan
    active = backtest["executed_position"] != 0
    changes = backtest["executed_position"].diff().fillna(0).ne(0)
    return {"annual_return": annual_return, "annual_volatility": volatility, "sharpe": sharpe,
            "max_drawdown": backtest["drawdown"].min(), "turnover": backtest["turnover"].sum(),
            "trading_days": int(active.sum()), "position_changes": int(changes.sum())}
