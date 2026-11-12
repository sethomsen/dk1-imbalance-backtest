"""How much look-ahead inflates the wind rule.

Runs the wind threshold rule from backtest.py with three wind forecasts:

  - ECMWF forecasts available at day-ahead gate closure (the main test).
  - Energinet's day-ahead wind forecast, published at 18:00 Danish time on
    the day before delivery, six hours after gate closure.
  - Energinet's current wind forecast, updated until shortly before
    delivery.

Only the first could have been traded. Swapping in a fresher forecast does
not make the rule better, because the rule profits from the error in an
early forecast, not from wind itself: when the early forecast says strong
wind, reality more often falls short of what the day-ahead market expected.

The look-ahead that does inflate a backtest is knowledge of that error. The
last strategy uses Energinet's revision of its wind forecast after the
18:00 forecast (current minus day-ahead, in MW): it sells when the forecast
was revised up and buys when it was revised down, in the periods where the
revision is in the largest 20% of the training window. It could never have
been traded, since the revision is only known close to delivery.

Usage: uv run python -m dk1.sensitivity
"""

import pandas as pd

import numpy as np

from dk1.backtest import FIRST_TEST_MONTH, QUANTILE, TRAIN_MONTHS, report, walk_forward
from dk1.features import TZ, build, development, load_energinet_wind

SIGNALS = {
    "ECMWF at gate closure (tradable)": "wind_cubed",
    "Energinet day-ahead, 18:00 (6 h late)": "energinet_day_ahead",
    "Energinet current (minutes before delivery)": "energinet_current",
}


def energinet_wind(delivery: pd.DatetimeIndex) -> pd.DataFrame:
    """Energinet's onshore plus offshore wind forecasts in MW, per delivery period."""
    hourly = load_energinet_wind().reindex(delivery.floor("h"))
    hourly.index = delivery
    return hourly.rename(columns={"ForecastDayAhead": "energinet_day_ahead", "ForecastCurrent": "energinet_current"})


def revision_walk_forward(frame: pd.DataFrame) -> pd.DataFrame:
    """Sell when wind was revised up after 18:00, buy when revised down."""
    revision = frame["energinet_current"] - frame["energinet_day_ahead"]
    months = pd.date_range(FIRST_TEST_MONTH, frame.index.max().tz_convert(TZ), freq="MS")
    pieces = []
    for month in months:
        in_train = (frame.index >= month - pd.DateOffset(months=TRAIN_MONTHS)) & (frame.index < month)
        in_test = (frame.index >= month) & (frame.index < month + pd.DateOffset(months=1))
        threshold = revision[in_train].abs().quantile(QUANTILE)
        piece = frame.loc[in_test, ["spread", "local_date"]].copy()
        big = revision[in_test].abs() > threshold
        piece["position"] = np.where(big, -np.sign(revision[in_test]), 0).astype(int)
        pieces.append(piece)
    return pd.concat(pieces)


def main() -> None:
    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", lambda v: f"{v:,.2f}")

    frame = development(build())
    frame = frame.join(energinet_wind(frame.index))
    frame = frame.dropna(subset=["spread", *SIGNALS.values()])

    rows = []
    for name, signal in SIGNALS.items():
        trades = walk_forward(frame, signal)
        rows.append(report(name, trades["spread"], trades["position"], trades["local_date"]))
    trades = revision_walk_forward(frame)
    rows.append(report("Energinet revision after 18:00 (look-ahead)", trades["spread"], trades["position"], trades["local_date"]))
    columns = ["periods_traded", "eur_per_mwh", "ci95_low", "ci95_high", "median_per_mwh", "hit_rate", "without_top_0.1pct", "max_drawdown_eur"]
    print(pd.DataFrame(rows).set_index("strategy")[columns].to_string())


if __name__ == "__main__":
    main()
