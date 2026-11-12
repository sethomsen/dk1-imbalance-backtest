"""Test of hypothesis 2: heavy-traffic days and EV charging.

Hypothesis: on days with heavy road traffic, more EVs arrive with low
batteries and charge in the evening. Consumption forecasts do not fully
capture this, so the system is short more often in the evening, and the
imbalance price is above day-ahead.

The test was fixed before it was run:

  - Days: the heavy-traffic days in traffic.py that fall in the development
    period. The ones in the holdout are added in the holdout run, by
    holdout.py.
  - Hours: evening, 17:00 to 23:00 Danish time.
  - Controls: for each heavy-traffic day, days on the same weekday within
    four weeks either side that are neither public holidays nor
    heavy-traffic days.
  - Main measure: difference in differences. For each day, the mean spread
    in the evening minus the mean spread in the rest of the day; then the
    heavy-traffic day's value minus the mean over its control days. This
    separates an evening EV effect from a general holiday effect and from
    the general evening effect.
  - Inference: bootstrap over heavy-traffic days, and a sign-flip
    permutation test of the mean difference.
  - Secondary measures: the same comparison for the share of evening
    periods with the system short, and the PnL of buying day-ahead in the
    evening on heavy-traffic days.

With about two dozen days, only a large effect can be detected. A null
result does not rule out a small one.

Usage: uv run python -m dk1.ev_test
"""

import holidays
import numpy as np
import pandas as pd

from dk1.features import TZ, build, development

MWH_PER_PERIOD = 0.25
CONTROL_WINDOW_DAYS = 28
SAMPLES = 10000


def daily_table(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per delivery day with evening and rest-of-day statistics."""
    frame = frame.assign(short=(frame["direction"] == 1).astype(float))
    grouped = frame.groupby(["local_date", "evening"])
    spread = grouped["spread"].mean().unstack()
    short = grouped["short"].mean().unstack()
    table = pd.DataFrame(
        {
            "evening_spread": spread[True],
            "rest_spread": spread[False],
            "evening_short": short[True],
            "rest_short": short[False],
        }
    )
    table["spread_diff"] = table["evening_spread"] - table["rest_spread"]
    table["short_diff"] = table["evening_short"] - table["rest_short"]
    flags = frame.groupby("local_date")["high_traffic"].first()
    table["high_traffic"] = flags
    years = range(table.index.min().year, table.index.max().year + 1)
    public = set(holidays.Denmark(years=years))
    table["public_holiday"] = [d.date() in public for d in table.index]
    return table.dropna()


def matched_differences(table: pd.DataFrame) -> pd.DataFrame:
    """For each heavy-traffic day, its values minus the mean of its controls."""
    controls_pool = table[~table["high_traffic"] & ~table["public_holiday"]]
    rows = []
    for day, values in table[table["high_traffic"]].iterrows():
        window = controls_pool[
            (controls_pool.index.day_of_week == day.day_of_week)
            & ((controls_pool.index - day).days.__abs__() <= CONTROL_WINDOW_DAYS)
        ]
        if window.empty:
            continue
        rows.append(
            {
                "day": day,
                "controls": len(window),
                "evening_spread": values["evening_spread"] - window["evening_spread"].mean(),
                "did_spread": values["spread_diff"] - window["spread_diff"].mean(),
                "evening_short": values["evening_short"] - window["evening_short"].mean(),
                "did_short": values["short_diff"] - window["short_diff"].mean(),
            }
        )
    return pd.DataFrame(rows).set_index("day")


def bootstrap_ci(values: np.ndarray, seed: int = 0, statistic=np.mean) -> tuple[float, float]:
    """95% interval for a statistic of the days, by default their mean."""
    rng = np.random.default_rng(seed)
    resampled = statistic(values[rng.integers(0, len(values), size=(SAMPLES, len(values)))], axis=1)
    return tuple(np.percentile(resampled, [2.5, 97.5]))


def trimmed_mean(values: np.ndarray, axis: int = -1) -> np.ndarray:
    """Mean without the lowest and the highest tenth of the days."""
    cut = round(0.1 * values.shape[axis])
    kept = np.sort(values, axis=axis).take(range(cut, values.shape[axis] - cut), axis=axis)
    return kept.mean(axis=axis)


def detectable_effect(values: np.ndarray) -> float:
    """The smallest true effect that a 5% two-sided test on these days would
    find four times out of five: 2.8 standard errors of the mean."""
    low, high = bootstrap_ci(values)
    return 2.8 * (high - low) / (2 * 1.96)


def sign_flip_p(values: np.ndarray, seed: int = 0) -> float:
    """Two-sided p-value for a mean of zero, flipping the sign of each day."""
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(SAMPLES, len(values)))
    null = (signs * values).mean(axis=1)
    return float((np.abs(null) >= abs(values.mean())).mean())


def main() -> None:
    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", lambda v: f"{v:,.2f}")

    frame = development(build()).dropna(subset=["spread", "direction"])
    table = daily_table(frame)
    diffs = matched_differences(table)
    print(f"Heavy-traffic days tested: {len(diffs)} (controls per day: {diffs['controls'].min()}-{diffs['controls'].max()})\n")

    print("Heavy-traffic day minus its controls, mean over days")
    for column, label in [
        ("evening_spread", "evening spread (EUR/MWh)"),
        ("did_spread", "evening minus rest of day, spread (EUR/MWh)  <- main measure"),
        ("evening_short", "evening share short"),
        ("did_short", "evening minus rest of day, share short"),
    ]:
        values = diffs[column].to_numpy()
        low, high = bootstrap_ci(values)
        print(f"  {label}")
        print(f"    mean {values.mean():7.2f}   median {np.median(values):7.2f}   95% CI [{low:.2f}, {high:.2f}]   sign-flip p = {sign_flip_p(values):.3f}")

    evening = frame[frame["high_traffic"] & frame["evening"]]
    daily_pnl = (evening["spread"] * MWH_PER_PERIOD).groupby(evening["local_date"]).sum()
    mwh_per_day = evening.groupby("local_date").size() * MWH_PER_PERIOD
    per_mwh = daily_pnl.sum() / mwh_per_day.sum()
    rng = np.random.default_rng(0)
    picks = rng.integers(0, len(daily_pnl), size=(SAMPLES, len(daily_pnl)))
    boot = daily_pnl.to_numpy()[picks].sum(axis=1) / mwh_per_day.to_numpy()[picks].sum(axis=1)
    low, high = np.percentile(boot, [2.5, 97.5])
    print("\nTrade: buy 1 MW day-ahead in the evening on heavy-traffic days")
    print(f"  {len(evening)} periods, PnL {daily_pnl.sum():,.0f} EUR, {per_mwh:.2f} EUR/MWh, 95% CI [{low:.2f}, {high:.2f}], days positive {int((daily_pnl > 0).sum())} of {len(daily_pnl)}\n")

    # None of this is a new test. It describes how much the main measure
    # could have shown, and how much it rests on single days.
    values = diffs["did_spread"].to_numpy()
    print("How far the main measure can be trusted")
    print(f"  smallest effect it could detect with 80% power: {detectable_effect(values):.0f} EUR/MWh")
    for label, statistic in [("median", np.median), ("mean without the lowest and highest tenth", trimmed_mean)]:
        low, high = bootstrap_ci(values, statistic=statistic)
        print(f"  {label}: {statistic(values):.2f}, 95% CI [{low:.2f}, {high:.2f}]")
    without_one = (values.sum() - values) / (len(values) - 1)
    largest = np.abs(values).argmax()
    print(f"  mean with one day left out: from {without_one.min():.2f} to {without_one.max():.2f}")
    print(f"  mean without {diffs.index[largest]:%Y-%m-%d} ({values[largest]:.2f}): {without_one[largest]:.2f}\n")

    print("Per day")
    print(diffs.assign(weekday=diffs.index.strftime("%a")).to_string())


if __name__ == "__main__":
    main()
