import pandas as pd

from src.signals import positions_from_zscore


def test_positions_enter_and_exit():
    z = pd.Series([0.0, -2.1, -1.0, -0.2, 2.2, 0.1])
    assert positions_from_zscore(z, 2.0, 0.5).tolist() == [0, 1, 1, 0, -1, 0]
