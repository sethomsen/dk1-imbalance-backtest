"""Household consumption on heavy-traffic days (hypothesis 2, the first step).

ev_test.py tests the end of the hypothesis: is the spread higher in the
evening of a heavy-traffic day? That is the noisiest place to look. This
module tests the step the rest depends on: do households use more power
that evening than on comparable days? Consumption is smooth, so the same
days say much more here than they do on prices.

The test was fixed before the consumption data was looked at:

  - Series: hourly consumption of private customers in the three regions
    that make up DK1, summed.
  - Days and controls: as in ev_test.py. The heavy-traffic days in
    traffic.py, each compared with the days on the same weekday within four
    weeks either side that are neither public holidays nor heavy-traffic
    days.
  - Unit: per cent of the control days' consumption in the same hour, so
    that winter and summer days count the same.
  - Main measure: difference in differences, as in ev_test.py. The evening
    (17:00 to 23:00 Danish time) minus the rest of the day.
  - Second measure: the night after (00:00 to 06:00 on the following day)
    minus the rest of the heavy-traffic day. Grid tariffs are lowest at
    night, so cars that come home in the evening may be set to charge
    after midnight.
  - Inference: bootstrap over heavy-traffic days, and a sign-flip
    permutation test of the mean.
  - The profile by hour is descriptive. It shows where in the day a
    difference falls, if there is one.

A holiday changes when people are at home, so more consumption in the
evening would not prove that cars are charging. No difference would be
evidence against the first step of the hypothesis. The series has no
forecast to compare with, so this says nothing about whether those who
forecast consumption are surprised.

Usage: uv run python -m dk1.ev_consumption
Writes tables to stdout and a figure to results/figures/.
"""

import holidays
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dk1.ev_test import CONTROL_WINDOW_DAYS, bootstrap_ci, detectable_effect, sign_flip_p
from dk1.features import HOLDOUT_START, RAW, TZ, read_parquets
from dk1.figures import BLUE, FIGURES, TEXT_SECONDARY, save, style
from dk1.traffic import high_traffic_days

EVENING = range(17, 23)
# Hours are counted from midnight of the heavy-traffic day, so 24 to 29 is
# 00:00 to 06:00 on the day after.
NIGHT_AFTER = range(24, 30)
REST = [hour for hour in range(24) if hour not in EVENING]


def load_private_consumption() -> pd.Series:
    """Hourly consumption of private customers in DK1, in MWh."""
    raw = read_parquets(RAW / "eds" / "ConsumptionConsumerCategoryHour")
    raw = raw[raw["ConsumerCategory3"] == "Privat"].drop_duplicates(["TimeUTC", "RegionName"])
    hour = pd.to_datetime(raw["TimeUTC"]).dt.tz_localize("UTC").rename("hour")
    regions = raw["RegionName"].nunique()
    return raw.groupby(hour)["ConsumptionkWh"].sum(min_count=regions) / 1000


def hourly_table(consumption: pd.Series) -> pd.DataFrame:
    """One row per Danish day and one column per hour, 0 to 29, where 24 to
    29 are the first six hours of the following day."""
    local = consumption.index.tz_convert(TZ)
    table = consumption.groupby([pd.to_datetime(local.date), local.hour]).mean().unstack()
    table = table[table.index < HOLDOUT_START.tz_localize(None)]
    after = table[list(range(6))].copy()
    after.index = after.index - pd.Timedelta(days=1)
    after.columns = list(NIGHT_AFTER)
    return table.join(after)


def matched_profiles(table: pd.DataFrame) -> pd.DataFrame:
    """For each heavy-traffic day, its consumption per hour in per cent above
    the mean of its control days."""
    traffic = pd.to_datetime(sorted(high_traffic_days(range(table.index.min().year, table.index.max().year + 1))))
    public = pd.to_datetime(sorted(holidays.Denmark(years=range(table.index.min().year, table.index.max().year + 1))))
    controls_pool = table[~table.index.isin(traffic) & ~table.index.isin(public)]
    rows = {}
    for day in table.index[table.index.isin(traffic)]:
        window = controls_pool[
            (controls_pool.index.day_of_week == day.day_of_week)
            & (np.abs((controls_pool.index - day).days) <= CONTROL_WINDOW_DAYS)
        ]
        if window.empty:
            continue
        rows[day] = 100 * (table.loc[day] / window.mean() - 1)
    return pd.DataFrame(rows).T


def plot(profiles: pd.DataFrame) -> None:
    style()
    FIGURES.mkdir(parents=True, exist_ok=True)
    mean = profiles.mean()
    bounds = np.array([bootstrap_ci(profiles[hour].dropna().to_numpy()) for hour in profiles.columns])
    fig, ax = plt.subplots(figsize=(9, 3.8))
    for hours, label in [(EVENING, "evening"), (NIGHT_AFTER, "night after")]:
        ax.axvspan(hours[0] - 0.5, hours[-1] + 0.5, color=TEXT_SECONDARY, alpha=0.08, linewidth=0)
        ax.annotate(label, (np.mean(hours), 1), xycoords=("data", "axes fraction"), xytext=(0, -12), textcoords="offset points", ha="center", color=TEXT_SECONDARY, fontsize=9)
    ax.fill_between(profiles.columns, bounds[:, 0], bounds[:, 1], color=BLUE, alpha=0.15, linewidth=0)
    ax.plot(profiles.columns, mean, color=BLUE, marker="o", markersize=4)
    ax.axhline(0, color=TEXT_SECONDARY, linewidth=0.8)
    ax.set_xticks(range(0, 30, 3), [f"{hour % 24:02d}:00" for hour in range(0, 30, 3)])
    ax.set_xlabel("hour of the heavy-traffic day, continuing into the next morning (Danish time)")
    ax.set_ylabel("% above comparison days")
    ax.set_title("Household consumption on heavy-traffic days")
    save(fig, "ev_consumption.png")


def main() -> None:
    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", lambda v: f"{v:,.2f}")

    profiles = matched_profiles(hourly_table(load_private_consumption()))
    print(f"Heavy-traffic days tested: {len(profiles)}\n")

    measures = pd.DataFrame(
        {
            "evening": profiles[list(EVENING)].mean(axis=1),
            "rest": profiles[REST].mean(axis=1),
            "night_after": profiles[list(NIGHT_AFTER)].mean(axis=1),
        }
    )
    measures["did_evening"] = measures["evening"] - measures["rest"]
    measures["did_night_after"] = measures["night_after"] - measures["rest"]

    print("Household consumption, heavy-traffic day against its controls (% of the controls), mean over days")
    for column, label in [
        ("rest", "rest of the day"),
        ("evening", "evening"),
        ("did_evening", "evening minus rest of the day  <- main measure"),
        ("night_after", "night after"),
        ("did_night_after", "night after minus rest of the day"),
    ]:
        values = measures[column].dropna().to_numpy()
        low, high = bootstrap_ci(values)
        print(f"  {label}")
        print(f"    mean {values.mean():6.2f}   median {np.median(values):6.2f}   95% CI [{low:.2f}, {high:.2f}]   sign-flip p = {sign_flip_p(values):.3f}")
    main_values = measures["did_evening"].dropna().to_numpy()
    print(f"\n  smallest effect the main measure could detect with 80% power: {detectable_effect(main_values):.2f} percentage points\n")

    profile = pd.DataFrame({"mean": profiles.mean(), "median": profiles.median()})
    bounds = np.array([bootstrap_ci(profiles[hour].dropna().to_numpy()) for hour in profiles.columns])
    profile["ci95_low"], profile["ci95_high"] = bounds[:, 0], bounds[:, 1]
    profile.index = [f"{hour % 24:02d}:00{' next day' if hour >= 24 else ''}" for hour in profile.index]
    print("By hour (% above the controls)")
    print(profile.to_string(), "\n")

    print("Per day")
    print(measures.assign(weekday=measures.index.strftime("%a")).to_string())
    plot(profiles)


if __name__ == "__main__":
    main()
