from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint


def estimate_hedge_ratio(y: pd.Series, x: pd.Series) -> float:
    aligned = pd.concat([y, x], axis=1).dropna()
    model = sm.OLS(np.log(aligned.iloc[:, 0]), sm.add_constant(np.log(aligned.iloc[:, 1]))).fit()
    return float(model.params.iloc[1])


def spread(y: pd.Series, x: pd.Series, hedge_ratio: float) -> pd.Series:
    return np.log(y) - hedge_ratio * np.log(x)


def rolling_zscore(values: pd.Series, window: int) -> pd.Series:
    mean = values.rolling(window, min_periods=window).mean()
    std = values.rolling(window, min_periods=window).std(ddof=0).replace(0, np.nan)
    return (values - mean) / std


def screen_cointegrated_pairs(prices: pd.DataFrame, significance: float = 0.05) -> pd.DataFrame:
    rows = []
    for i, left in enumerate(prices.columns):
        for right in prices.columns[i + 1:]:
            score, pvalue, _ = coint(np.log(prices[left]), np.log(prices[right]))
            rows.append({"left": left, "right": right, "coint_stat": score, "pvalue": pvalue,
                         "cointegrated": pvalue < significance})
    return pd.DataFrame(rows).sort_values("pvalue").reset_index(drop=True)
