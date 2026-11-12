"""Newer forecasts and disagreement between weather models (hypothesis 4).

The look-ahead test in sensitivity.py showed where the profit is: in the
error of the wind forecast, that is, in how far the forecast the day-ahead
market traded on ends up from what happens. Two things known at gate closure
might say something about that error.

  - How the forecast has just changed. If forecasts are revised in steps, a
    run that lowered tomorrow's wind is followed by further cuts more often
    than by a rise. Less wind than the market planned for leaves the system
    short.
  - How much the models disagree. Where they disagree, one of them is wrong,
    and the error after gate closure should be larger. If the truth lies
    between the models more often than not, a model that stands apart from
    the others tends to be wrong in the direction in which it stands apart.

The strategies use the walk-forward setup and the 80% quantile of
backtest.py:

  - Wind rule, newest run: the rule of backtest.py on the 00 UTC run of the
    day of gate closure, in place of the forecast made one or two days
    ahead. This is the forecast a trader has, so it shows whether the rule
    survives realistic information.
  - Revision, 12 h and 24 h: buy when the newest run lowered the wind
    forecast for the delivery hour, compared with the run 12 or 24 hours
    before it, and sell when it raised it, in the periods where the absolute
    revision is above the quantile. The 12-hour revision is the main one.
  - Wind rule when models disagree: the rule of backtest.py, traded only
    when the spread between the four models is above its median in the
    training window.
  - ECMWF against the other models: buy when ECMWF forecasts more wind than
    the mean of the other three and sell when it forecasts less, in the
    periods where the absolute difference is above the quantile.
  - The two short-system classifiers of model.py with the five new features.

Before the strategies, the premises are checked on wind itself, which is
measured far more precisely than PnL. Energinet's revision of its wind
forecast after 18:00 (the signal of the look-ahead strategy in
sensitivity.py) stands in for the error after gate closure. That revision
is correlated from one day to the next, so the interval from resampling
single days is too narrow, and a second interval resamples blocks of 14
consecutive days.

Three further tables describe "ECMWF against the other models", since it
came out best: how the models' forecasts compare, the strategy without the
hours in which the archive mixes forecast runs (see features.py), and the
strategy without its largest gains. They were added after the first run
and change nothing in the strategies.

The last table tests every tradable strategy in the project against zero
and adjusts for having tried all of them on the same months.

The strategies and the direction of each trade were fixed before this
module was run for the first time, and nothing was tuned on the results.

Usage: uv run python -m dk1.forecasts
Takes about ten minutes. Writes tables to stdout and a figure to
results/figures/.
"""

import numpy as np
import pandas as pd

from dk1 import backtest, model, sensitivity, two_sided
from dk1.backtest import BOOTSTRAP_SAMPLES, FIRST_TEST_MONTH, MWH_PER_PERIOD, QUANTILE, TRAIN_MONTHS, reality_check, report
from dk1.features import TZ, build, cubed_wind, development, forecast_features, load_weather, model_features, select_lead
from dk1.fetch import OTHER_MODELS

NEW = ["run_wind_cubed", "run_revision_12h", "run_revision_24h", "model_spread", "ecmwf_vs_models"]


BLOCK_DAYS = 14


def correlation(x: pd.Series, y: pd.Series, days: pd.Series, seed: int = 0, block_days: int = 1) -> tuple[float, float, float]:
    """Correlation of two series, with a 95% interval from resampling whole
    delivery days, or blocks of consecutive days if block_days is above 1."""
    data = pd.DataFrame({"x": x, "y": y, "day": days}).dropna()
    data = data.assign(n=1.0, xx=data["x"] ** 2, yy=data["y"] ** 2, xy=data["x"] * data["y"])
    sums = data.groupby("day")[["n", "x", "y", "xx", "yy", "xy"]].sum().to_numpy()

    def from_sums(total: np.ndarray) -> np.ndarray:
        n, x, y, xx, yy, xy = total.T
        return (xy / n - x * y / n**2) / np.sqrt((xx / n - (x / n) ** 2) * (yy / n - (y / n) ** 2))

    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(sums), size=(BOOTSTRAP_SAMPLES, -(-len(sums) // block_days)))
    picks = ((starts[:, :, None] + np.arange(block_days)) % len(sums)).reshape(BOOTSTRAP_SAMPLES, -1)[:, : len(sums)]
    low, high = np.percentile(from_sums(sums[picks].sum(axis=1)), [2.5, 97.5])
    return float(from_sums(sums.sum(axis=0)[None, :])[0]), low, high


def signed_rule(frame: pd.DataFrame, signal: str, direction: int) -> pd.DataFrame:
    """Trade where the signal is large: with its sign if direction is 1,
    against it if direction is -1."""
    months = pd.date_range(FIRST_TEST_MONTH, frame.index.max().tz_convert(TZ), freq="MS")
    pieces = []
    for month in months:
        train = frame[(frame.index >= month - pd.DateOffset(months=TRAIN_MONTHS)) & (frame.index < month)]
        test = frame[(frame.index >= month) & (frame.index < month + pd.DateOffset(months=1))]
        threshold = train[signal].abs().quantile(QUANTILE)
        piece = test[["spread", "local_date"]].copy()
        piece["position"] = np.where(test[signal].abs() > threshold, direction * np.sign(test[signal]), 0.0)
        pieces.append(piece)
    return pd.concat(pieces)


def wind_rule_when_models_disagree(frame: pd.DataFrame) -> pd.DataFrame:
    months = pd.date_range(FIRST_TEST_MONTH, frame.index.max().tz_convert(TZ), freq="MS")
    pieces = []
    for month in months:
        train = frame[(frame.index >= month - pd.DateOffset(months=TRAIN_MONTHS)) & (frame.index < month)]
        test = frame[(frame.index >= month) & (frame.index < month + pd.DateOffset(months=1))]
        windy = test["wind_cubed"] > train["wind_cubed"].quantile(QUANTILE)
        disagree = test["model_spread"] > train["model_spread"].median()
        piece = test[["spread", "local_date"]].copy()
        piece["position"] = (windy & disagree).astype(float)
        pieces.append(piece)
    return pd.concat(pieces)


def model_levels(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """How the four models' wind forecasts compare, apart from any trading.

    "ECMWF against the other models" trades on a raw difference, so a
    constant gap between the models goes into its signal. Returns each
    model's mean forecast of cubed wind, and the share of periods in which
    ECMWF is below the mean of the others: in all periods, and in the fifth
    with the largest absolute difference.
    """
    others = select_lead(load_weather("weather_models"), frame.index)
    mean = {"ecmwf_ifs025": frame["wind_cubed"].mean()}
    mean.update({name: cubed_wind(others.filter(like=f"wind_speed_100m_{name}_")).mean() for name in OTHER_MODELS})
    difference = frame["ecmwf_vs_models"]
    largest = difference[difference.abs() > difference.abs().quantile(QUANTILE)]
    below = pd.Series({"all periods": (difference < 0).mean(), "largest fifth of differences": (largest < 0).mean()})
    return pd.Series(mean), below


def premises(frame: pd.DataFrame) -> pd.DataFrame:
    """Do the signals say anything about the wind forecast error after gate closure?"""
    later = frame["energinet_current"] - frame["energinet_day_ahead"]
    checks = [
        ("newest run vs latest Energinet forecast", frame["run_wind_cubed"], frame["energinet_current"], "close to 1"),
        ("forecast made 1-2 days ahead vs latest Energinet forecast", frame["wind_cubed"], frame["energinet_current"], "lower than the row above"),
        ("12 h revision before the gate vs revision after 18:00", frame["run_revision_12h"], later, "positive"),
        ("24 h revision before the gate vs revision after 18:00", frame["run_revision_24h"], later, "positive"),
        ("model spread vs size of the revision after 18:00", frame["model_spread"], later.abs(), "positive"),
        ("ECMWF minus other models vs revision after 18:00", frame["ecmwf_vs_models"], later, "negative"),
    ]
    rows = []
    for name, x, y, expected in checks:
        r, low, high = correlation(x, y, frame["local_date"])
        _, block_low, block_high = correlation(x, y, frame["local_date"], block_days=BLOCK_DAYS)
        rows.append(
            {
                "check": name, "correlation": r, "ci95_low": low, "ci95_high": high,
                "blocks_low": block_low, "blocks_high": block_high, "expected": expected,
            }
        )
    return pd.DataFrame(rows).set_index("check")


def without_mixed_runs(frame: pd.DataFrame) -> pd.DataFrame:
    """ECMWF against the other models without the hours in which the archive
    mixes forecast runs, and with 2-day forecasts for every hour.

    The hours are given in features.py: where the 1-day forecast is used, 01
    to 03 UTC carry real weight from a run that was not available at gate
    closure, and 22 and 23 UTC a few per cent; where the 2-day forecast is
    used, 19 to 22 UTC carry a few per cent.
    """
    trades = signed_rule(frame, "ecmwf_vs_models", 1)
    hour = trades.index.hour
    day1 = frame["forecast_lead_days"].reindex(trades.index) == 1
    most = day1 & np.isin(hour, [1, 2, 3])
    every = most | (day1 & np.isin(hour, [22, 23])) | (~day1 & np.isin(hour, [19, 20, 21, 22]))

    delivery = frame.index
    ecmwf = cubed_wind(select_lead(load_weather(), delivery, only_day2=True).filter(like="wind_speed_100m_"))
    others = select_lead(load_weather("weather_models"), delivery, only_day2=True)
    two_day = frame.assign(ecmwf_vs_models=model_features(others, ecmwf)["ecmwf_vs_models"]).dropna(subset=["ecmwf_vs_models"])

    two_day_trades = signed_rule(two_day, "ecmwf_vs_models", 1)

    cases = [
        ("as reported", trades, trades["position"]),
        ("flat in the hours most affected", trades, trades["position"].where(~most, 0.0)),
        ("flat in every affected hour", trades, trades["position"].where(~every, 0.0)),
        ("2-day forecasts for every hour", two_day_trades, two_day_trades["position"]),
    ]
    rows = []
    for name, case, position in cases:
        row = report(name, case["spread"], position, case["local_date"])
        row["eur_per_day"] = row["pnl_eur"] / case["local_date"].nunique()
        rows.append(row)
    columns = ["periods_traded", "pnl_eur", "eur_per_day", "eur_per_mwh", "ci95_low", "ci95_high"]
    return pd.DataFrame(rows).set_index("strategy")[columns]


def without_largest_gains(trades: pd.DataFrame, share: float = 0.001) -> float:
    """PnL per MWh without the periods with the largest gains.

    report() leaves out the periods with the largest absolute PnL. For a
    strategy that mostly sells those are losses, which flatters it. This is
    the comparable figure.
    """
    pnl = (trades["position"] * trades["spread"] * MWH_PER_PERIOD)[trades["position"] != 0]
    mwh = trades["position"].abs() * MWH_PER_PERIOD
    largest = pnl.sort_values(ascending=False).index[: max(1, int(share * len(pnl)))]
    return pnl.drop(largest).sum() / mwh[pnl.index].drop(largest).sum()


def main() -> None:
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda v: f"{v:,.3f}")

    built = development(build())
    frame = model.add_model_columns(built.join(forecast_features(built)).join(sensitivity.energinet_wind(built.index)))
    earlier_frame = frame.dropna(subset=two_sided.FEATURES + ["spread", "direction"])
    frame = frame.dropna(subset=model.FEATURES + NEW + ["spread", "direction"])

    print("Premises, checked on wind (correlation with a 95% interval)")
    print(premises(frame).to_string(), "\n")

    skill, classifiers = two_sided.feature_skill(frame, model.FEATURES + NEW, "forecast features")
    print("Skill at predicting a short system, with and without the forecast features")
    print("(AUC 0.5 = no skill; lower Brier is better; the interval is for the change in Brier score)")
    print(skill.T.to_string(), "\n")

    new = {
        "wind rule, newest run": backtest.walk_forward(frame, "run_wind_cubed"),
        "revision, 12 h": signed_rule(frame, "run_revision_12h", -1),
        "revision, 24 h": signed_rule(frame, "run_revision_24h", -1),
        "wind rule when models disagree": wind_rule_when_models_disagree(frame),
        "ECMWF against the other models": signed_rule(frame, "ecmwf_vs_models", 1),
        **{name: trades for name, trades in classifiers.items() if name.endswith("forecast features")},
    }
    local = frame.index.tz_convert(TZ)
    print(f"Development data with every feature: {local.min():%Y-%m-%d} to {local.max():%Y-%m-%d}, {len(frame):,} periods\n")

    pd.set_option("display.float_format", lambda v: f"{v:,.2f}")
    # The rule of backtest.py on the same periods, for comparison.
    same_rows = {"wind rule, forecast made 1-2 days ahead": backtest.walk_forward(frame), **new}
    rows = [report(name, trades["spread"], trades["position"], trades["local_date"]) for name, trades in same_rows.items()]
    print("Strategies on the forecast features")
    print(pd.DataFrame(rows).set_index("strategy").T.to_string(), "\n")

    rows = []
    for name in ["revision, 12 h", "revision, 24 h", "ECMWF against the other models"]:
        trades = new[name]
        for leg, position in [("buy", trades["position"].clip(lower=0)), ("sell", trades["position"].clip(upper=0))]:
            if (position != 0).any():
                rows.append(report(f"{name}: {leg}", trades["spread"], position, trades["local_date"]))
    columns = ["periods_traded", "pnl_eur", "eur_per_mwh", "ci95_low", "ci95_high", "median_per_mwh", "hit_rate", "without_top_0.1pct"]
    print("The buy and sell sides separately")
    print(pd.DataFrame(rows).set_index("strategy")[columns].to_string(), "\n")

    mean, below = model_levels(frame)
    print("Mean forecast of cubed wind per model")
    print(mean.round(0).to_string())
    print("Share of periods in which ECMWF is below the mean of the other models")
    print(below.to_string(), "\n")

    print("ECMWF against the other models, without the hours in which the archive mixes forecast runs")
    print(without_mixed_runs(frame).to_string(), "\n")
    print("PnL per MWh without the 0.1% of periods with the largest gains")
    for name in ["wind rule, newest run", "ECMWF against the other models"]:
        print(f"  {name}: {without_largest_gains(new[name]):.2f}")
    print()

    rows = [two_sided.risk(name, trades["spread"], trades["position"], trades["local_date"]) for name, trades in new.items()]
    print("Risk, 1 MW positions")
    print(pd.DataFrame(rows).set_index("strategy").to_string(), "\n")

    _, earlier = two_sided.build_strategies(earlier_frame)
    daily = two_sided.daily_pnl({**earlier, **new})
    print(f"Mean daily PnL against zero, adjusted for trying {daily.shape[1]} strategies on the same months")
    pd.set_option("display.float_format", lambda v: f"{v:,.3f}")
    print(reality_check(daily).to_string())
    print(f"\nTotal PnL over {len(daily)} days (EUR)")
    print(daily.sum().round(0).to_string())

    two_sided.plot(new, "forecasts.png", "Strategies on newer forecasts and on disagreement between models")


if __name__ == "__main__":
    main()
