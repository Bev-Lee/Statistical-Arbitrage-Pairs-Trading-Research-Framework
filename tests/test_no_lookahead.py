import pandas as pd

from src.backtest import run_backtest


def test_signal_does_not_earn_same_bar_return():
    index = pd.date_range("2024-01-01", periods=35, freq="D")
    increments = [0.0] + ([1.0, -0.5, 1.5, -1.0] * 9)[:34]
    spread = pd.Series(increments, index=index).cumsum()
    signal = pd.Series(0, index=index)
    signal.iloc[15] = 1
    result = run_backtest(spread, signal, 0, 0, target_daily_vol=1, max_gross_exposure=1)
    # The entry signal at t cannot earn the return observed at t.
    assert result.iloc[15]["gross_return"] == 0
    assert result.iloc[16]["executed_position"] > 0
