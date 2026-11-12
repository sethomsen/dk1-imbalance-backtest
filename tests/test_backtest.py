import numpy as np
import pandas as pd
import pytest

from dk1.backtest import day_bootstrap, max_drawdown, reality_check, report
from dk1.holdout import against_zero, verdict


def periods(days: int) -> tuple[pd.DatetimeIndex, pd.Series]:
    index = pd.date_range("2025-09-01", periods=days * 96, freq="15min", tz="UTC")
    return index, pd.Series(index.date, index=index)


def test_per_mwh_figures_do_not_depend_on_position_size():
    index, days = periods(30)
    rng = np.random.default_rng(1)
    spread = pd.Series(rng.normal(0, 50, len(index)), index=index)
    position = pd.Series(rng.choice([-1.0, 0.0, 1.0], len(index)), index=index)
    full = report("full", spread, position, days)
    half = report("half", spread, position / 2, days)
    for key in ["eur_per_mwh", "ci95_low", "ci95_high", "median_per_mwh", "hit_rate", "without_top_0.1pct"]:
        assert np.isclose(full[key], half[key]), key
    assert np.isclose(full["pnl_eur"], 2 * half["pnl_eur"])


def test_blocks_keep_consecutive_days_together():
    index, days = periods(20)
    rng = np.random.default_rng(2)
    pnl = pd.Series(rng.normal(0, 10, len(index)), index=index)
    mwh = pd.Series(0.25, index=index)
    # A block as long as the sample is the whole sample, rotated, so every
    # resample gives the same figure.
    samples = day_bootstrap(pnl, mwh, days, block_days=20)
    assert np.allclose(samples, pnl.sum() / mwh.sum())


def test_reality_check_adjusts_for_the_number_of_strategies():
    rng = np.random.default_rng(3)
    daily = pd.DataFrame(rng.normal(0, 100, (300, 8)), columns=[f"noise {i}" for i in range(8)])
    daily["edge"] = rng.normal(40, 100, 300)
    result = reality_check(daily)
    assert (result["p_adjusted"] >= result["p"]).all()
    assert result.loc["edge", "p_adjusted"] < 0.01
    # The luckiest of eight strategies without an edge should not pass.
    assert result.drop("edge")["p_adjusted"].min() > 0.05


def test_a_strategy_that_never_trades_is_no_evidence():
    rng = np.random.default_rng(5)
    daily = pd.DataFrame({"noise": rng.normal(0, 100, 200), "edge": rng.normal(30, 100, 200)})
    with_idle = reality_check(daily.assign(idle=0.0))
    assert with_idle.loc["idle", "p"] == 1.0
    assert with_idle.loc["idle", "p_adjusted"] == 1.0
    # A strategy with a positive t-statistic is tested as if the idle one were not there.
    assert with_idle.loc["edge", "p_adjusted"] == reality_check(daily).loc["edge", "p_adjusted"]


@pytest.mark.filterwarnings("ignore::RuntimeWarning")  # PnL per MWh of no trades
def test_a_strategy_that_never_trades_does_not_pass_the_holdout_test():
    index, days = periods(30)
    trades = pd.DataFrame({"spread": 10.0, "position": 0.0, "local_date": days}, index=index)
    row = against_zero("idle", trades)
    assert row["p_one_sided"] == 1.0
    assert not verdict(row).startswith("passes")


def test_drawdown_counts_a_fall_from_the_start():
    assert max_drawdown(pd.Series([-10.0, -20.0, -15.0, -10.0])) == -20.0
    assert max_drawdown(pd.Series([10.0, 30.0, 5.0, 20.0])) == -25.0
