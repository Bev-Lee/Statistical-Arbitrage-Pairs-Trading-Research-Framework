from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


def make_synthetic_prices(periods: int = 750, seed: int = 7) -> pd.DataFrame:
    """Create an offline, deterministic universe with one cointegrated pair.

    This is exclusively a test/demo fixture. It avoids presenting downloaded or
    simulated results as real trading performance.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=periods)
    market = np.cumsum(rng.normal(0.0002, 0.012, periods))
    mean_reverting = np.zeros(periods)
    for i in range(1, periods):
        mean_reverting[i] = 0.88 * mean_reverting[i - 1] + rng.normal(0, 0.012)
    log_msft = np.log(100) + market + rng.normal(0, 0.004, periods)
    log_aapl = np.log(95) + 1.04 * (log_msft - np.log(100)) + mean_reverting
    prices = {"MSFT": np.exp(log_msft), "AAPL": np.exp(log_aapl)}
    for ticker in ("GOOGL", "META", "JPM", "BAC", "KO", "PEP"):
        prices[ticker] = np.exp(np.log(rng.uniform(40, 180)) + np.cumsum(rng.normal(0.0002, 0.018, periods)))
    return pd.DataFrame(prices, index=dates)


def download_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    import yfinance as yf

    frame = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    prices = frame["Close"] if isinstance(frame.columns, pd.MultiIndex) else frame[["Close"]]
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(tickers[0])
    return clean_prices(prices)


def clean_prices(prices: pd.DataFrame) -> pd.DataFrame:
    cleaned = prices.copy().sort_index().replace([np.inf, -np.inf], np.nan).ffill().dropna(axis=1)
    cleaned = cleaned.loc[:, (cleaned > 0).all()]
    if cleaned.shape[1] < 2:
        raise ValueError("At least two clean price series are required.")
    return cleaned


def all_pairs(columns: pd.Index) -> list[tuple[str, str]]:
    return list(combinations(columns, 2))


def save_prices(prices: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(path)
