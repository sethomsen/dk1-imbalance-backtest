import numpy as np
import pandas as pd

from dk1.forecasts import correlation, signed_rule


def test_correlation_matches_numpy():
    rng = np.random.default_rng(4)
    index = pd.date_range("2025-09-01", periods=40 * 96, freq="15min", tz="UTC")
    x = pd.Series(rng.normal(size=len(index)), index=index)
    y = 0.5 * x + pd.Series(rng.normal(size=len(index)), index=index)
    r, low, high = correlation(x, y, pd.Series(index.date, index=index))
    assert np.isclose(r, np.corrcoef(x, y)[0, 1])
    assert low < r < high


def test_signed_rule_trades_against_a_large_signal_only():
    # Six months of training with signals between -1 and 1, then a test
    # month: a large positive signal is sold when the direction is -1.
    index = pd.date_range("2025-03-01", "2025-09-30 23:45", freq="15min", tz="Europe/Copenhagen").tz_convert("UTC")
    signal = pd.Series(np.tile([-1.0, -0.5, 0.5, 1.0], len(index) // 4), index=index)
    test = index >= pd.Timestamp("2025-09-01", tz="Europe/Copenhagen")
    signal[test] = np.tile([5.0, 0.1, -5.0, 0.0], test.sum() // 4)
    frame = pd.DataFrame({"signal": signal, "spread": 1.0, "local_date": index.date}, index=index)
    trades = signed_rule(frame, "signal", -1)
    assert list(trades["position"].iloc[:4]) == [-1.0, 0.0, 1.0, 0.0]
