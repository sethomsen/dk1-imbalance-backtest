"""The holdout run: delivery from 1 August to 31 October 2026.

Everything in this module down to "Notes added after the run" was fixed
before the holdout was looked at. It was run once.

Primary test. "ECMWF against the other models", exactly as in forecasts.py:
buy when ECMWF forecasts more wind than the mean of the other three models
and sell when it forecasts less, in the periods where the absolute
difference is above the 80% quantile of the previous six months. Each of
the three holdout months is traded with a threshold from the six months
before it. The strategy passes if its mean daily PnL is above zero at the
one-sided 5% level, by a bootstrap over delivery days. It is the only
strategy tested in this sense, so no correction for several strategies is
applied.

In development the strategy sold in more than nine trades out of ten,
because ECMWF forecasts less wind than ICON and GFS as a rule. What the
holdout can confirm is therefore that narrower thing: selling when ICON
and GFS forecast much more wind than ECMWF.

How to read the outcome. The test is weak. In development the strategy
made about 26 EUR a day on 1 MW with a standard deviation of about 250 EUR
between days. On 92 days that gives a standard error of about 26 EUR, so
if the effect is real and as large as in development, the chance of
passing is about 1 in 4, and about 1 in 8 if it is half as large. A pass
is strong evidence. A loss counts against the strategy. A profit that does
not pass is inconclusive. The report also gives the profit without the
largest 0.1% of periods, and a pass that rests on those is weaker
evidence: three months is short enough for a few spikes to decide it.
Delivery from 1 November 2026 is not used here and remains for a later
test.

Secondary tests, both fixed earlier. The wind rule of backtest.py, with
the same test. And the heavy-traffic test of ev_test.py with the travel
days of the holdout added, as its docstring planned: the difference in
differences of the evening spread over all heavy-traffic days.

For description only, with no test and no conclusion drawn: the other
strategies of the project on the holdout, the checks on wind of
forecasts.py, and how the models' forecasts compare.

Notes added after the run. They change nothing in the code of the test.

  - The pass rule is less strict than it reads. For a strategy that mostly
    sells, whose losses come as a few large days, a test on 92 days passes
    too often when it has no edge, because those days are often absent from
    the sample (power.py simulates this). A pass would have been weaker
    evidence than "strong". The strategy did not pass.
  - The secondary test of the wind rule runs on the periods that have every
    forecast feature, so its thresholds differ slightly from those of
    backtest.py. The wind rule is also listed among the other strategies,
    on the periods of two_sided.py.
  - report() in backtest.py printed "nan" for the interval of the buy side,
    because the buys fell on 11 days and one resample of days contained none
    of them. It now leaves such resamples out. No other figure is affected.
  - The AIFS forecasts in the signal of the primary strategy carry some
    weight from a run that was not available at gate closure (see
    features.py). forecasts.py shows that this did not raise the result on
    development data. It was not measured on the holdout.

With --rehearsal the same code runs on the last three months of the
development data (May to July 2026) and cannot read the holdout. It exists
to find errors in this module before the one real run, and it leaves out
the slow strategies that are only described.

Usage: uv run python -m dk1.holdout [--rehearsal]
Takes about fifteen minutes.
"""

import sys

import numpy as np
import pandas as pd

from dk1 import backtest, ev_test, forecasts, model, sensitivity, two_sided
from dk1.backtest import reality_check, report
from dk1.features import HOLDOUT_START, TZ, build, development, forecast_features

# The holdout run covers delivery before this date (Danish time).
END = pd.Timestamp("2026-11-01", tz=TZ)
REHEARSAL = (pd.Timestamp("2026-05-01", tz=TZ), HOLDOUT_START)
LEVEL = 0.05

PRIMARY = "ECMWF against the other models"
SECONDARY = "wind rule"


def window(trades: pd.DataFrame, start: pd.Timestamp) -> pd.DataFrame:
    """The trades whose delivery is in the window. The frame they come from
    already ends where the window ends."""
    return trades[trades.index >= start]


def against_zero(name: str, trades: pd.DataFrame) -> dict:
    """PnL statistics and the one-sided p-value of the mean daily PnL."""
    row = report(name, trades["spread"], trades["position"], trades["local_date"])
    row["days"] = trades["local_date"].nunique()
    row["p_one_sided"] = float(reality_check(two_sided.daily_pnl({name: trades}))["p"].iloc[0])
    return row


def verdict(row: dict) -> str:
    if row["p_one_sided"] < LEVEL:
        return "passes"
    if row["pnl_eur"] < 0:
        return "does not pass, and made a loss"
    return "does not pass; a profit that is not told apart from zero is inconclusive"


def heavy_traffic(frame: pd.DataFrame, start: pd.Timestamp) -> None:
    diffs = ev_test.matched_differences(ev_test.daily_table(frame.dropna(subset=["spread", "direction"])))
    in_window = diffs.index >= start.tz_localize(None)
    print(f"Heavy-traffic days: {len(diffs)} in all, of which {in_window.sum()} in the window")
    values = diffs["did_spread"].to_numpy()
    low, high = ev_test.bootstrap_ci(values)
    print("  evening minus rest of day, spread against control days (EUR/MWh), all days")
    print(f"    mean {values.mean():.2f}   median {np.median(values):.2f}   95% CI [{low:.2f}, {high:.2f}]   sign-flip p = {ev_test.sign_flip_p(values):.3f}")
    print("  the days in the window")
    print(diffs.loc[in_window, ["controls", "evening_spread", "did_spread"]].to_string(), "\n")


def main() -> None:
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda v: f"{v:,.2f}")

    rehearsal = "--rehearsal" in sys.argv
    if rehearsal:
        start, end = REHEARSAL
        built = development(build())
    else:
        start, end = HOLDOUT_START, END
        built = build()
    built = built[built.index < end]
    print(f"{'REHEARSAL on development data' if rehearsal else 'HOLDOUT'}: delivery from {start:%Y-%m-%d} to {end - pd.Timedelta(days=1):%Y-%m-%d}\n")

    frame = model.add_model_columns(built.join(forecast_features(built)).join(sensitivity.energinet_wind(built.index)))
    earlier_frame = frame.dropna(subset=two_sided.FEATURES + ["spread", "direction"])
    frame = frame.dropna(subset=model.FEATURES + forecasts.NEW + ["spread", "direction"])
    in_window = frame[frame.index >= start]
    print(f"Periods in the window with every feature: {len(in_window):,} on {in_window['local_date'].nunique()} days\n")

    primary = window(forecasts.signed_rule(frame, "ecmwf_vs_models", 1), start)
    row = against_zero(PRIMARY, primary)
    print(f"PRIMARY TEST: {PRIMARY}")
    print(pd.Series(row).drop("strategy").to_string())
    print(f"  -> {verdict(row)} (one-sided p = {row['p_one_sided']:.3f}, level {LEVEL})\n")
    for leg, position in [("buy", primary["position"].clip(lower=0)), ("sell", primary["position"].clip(upper=0))]:
        if (position != 0).any():
            side = report(leg, primary["spread"], position, primary["local_date"])
            print(f"  {leg:4s}: {side['periods_traded']:5,d} periods, {side['eur_per_mwh']:6.2f} EUR/MWh, 95% CI [{side['ci95_low']:.2f}, {side['ci95_high']:.2f}], total {side['pnl_eur']:8,.0f} EUR")
    always_sell = report("always sell", in_window["spread"], pd.Series(-1.0, index=in_window.index), in_window["local_date"])
    print(f"  for comparison, selling in every period: {always_sell['eur_per_mwh']:.2f} EUR/MWh\n")

    secondary = window(backtest.walk_forward(frame), start)
    row = against_zero(SECONDARY, secondary)
    print(f"SECONDARY TEST: {SECONDARY}")
    print(pd.Series(row).drop("strategy").to_string())
    print(f"  -> {verdict(row)} (one-sided p = {row['p_one_sided']:.3f})\n")

    print("SECONDARY TEST: heavy-traffic days")
    heavy_traffic(built, start)

    print("DESCRIPTION ONLY, no test and no conclusion drawn\n")
    print("Checks on wind in the window (correlation with a 95% interval)")
    print(forecasts.premises(in_window).to_string(), "\n")
    mean, below = forecasts.model_levels(in_window)
    print("Mean forecast of cubed wind per model")
    print(mean.round(0).to_string())
    print("Share of periods in which ECMWF is below the mean of the other models")
    print(below.to_string(), "\n")

    others = {
        "wind rule, newest run": backtest.walk_forward(frame, "run_wind_cubed"),
        "revision, 12 h": forecasts.signed_rule(frame, "run_revision_12h", -1),
        "revision, 24 h": forecasts.signed_rule(frame, "run_revision_24h", -1),
        "wind rule when models disagree": forecasts.wind_rule_when_models_disagree(frame),
    }
    if not rehearsal:
        _, earlier = two_sided.build_strategies(earlier_frame)
        _, classifiers = two_sided.feature_skill(frame, model.FEATURES + forecasts.NEW, "forecast features")
        others = {**earlier, **others, **{name: trades for name, trades in classifiers.items() if name.endswith("forecast features")}}
    rows = []
    for name, trades in others.items():
        trades = window(trades, start)
        rows.append(report(name, trades["spread"], trades["position"], trades["local_date"]))
    columns = ["periods_traded", "pnl_eur", "eur_per_mwh", "ci95_low", "ci95_high", "hit_rate"]
    print("The other strategies in the window")
    print(pd.DataFrame(rows).set_index("strategy")[columns].to_string())


if __name__ == "__main__":
    main()
