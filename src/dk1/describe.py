"""Descriptive look at the spread (imbalance minus day-ahead) on development data.

Uses only delivery periods before the holdout, and leaves out the
heavy-traffic days, which are kept for their own test.

Usage: uv run python -m dk1.describe
Writes tables to stdout and figures to results/figures/.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dk1.features import TZ, build, development

FIGURES = Path(__file__).resolve().parents[2] / "results" / "figures"

# A 1 MW position held for one 15-minute period is 0.25 MWh.
MWH_PER_PERIOD = 0.25


def summarise(spread: pd.Series) -> pd.Series:
    return pd.Series(
        {
            "periods": len(spread),
            "mean": spread.mean(),
            "median": spread.median(),
            "share_positive": (spread > 0).mean(),
            "share_zero": (spread == 0).mean(),
            "share_negative": (spread < 0).mean(),
        }
    )


def spike_dependence(spread: pd.Series) -> pd.DataFrame:
    """How the total spread changes when the largest periods are removed."""
    order = spread.abs().sort_values(ascending=False).index
    rows = []
    for share in [0, 0.001, 0.005, 0.01, 0.05]:
        dropped = order[: int(round(share * len(spread)))]
        rest = spread.drop(dropped)
        rows.append({"dropped_share": share, "periods_dropped": len(dropped), "mean": rest.mean()})
    return pd.DataFrame(rows)


def by_bucket(frame: pd.DataFrame, column: str, buckets: pd.Series) -> pd.DataFrame:
    grouped = frame.groupby(buckets, observed=True)
    return pd.DataFrame(
        {
            column: grouped[column].mean(),
            "periods": grouped.size(),
            "mean_spread": grouped["spread"].mean(),
            "median_spread": grouped["spread"].median(),
            "share_short": grouped["direction"].apply(lambda s: (s == 1).mean()),
            "share_long": grouped["direction"].apply(lambda s: (s == -1).mean()),
        }
    )


def plot_wind(table: pd.DataFrame) -> None:
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.8))
    x = np.arange(len(table))
    left.bar(x - 0.2, table["mean_spread"], 0.4, label="mean")
    left.bar(x + 0.2, table["median_spread"], 0.4, label="median")
    left.axhline(0, color="black", linewidth=0.8)
    left.set_xlabel("forecast wind decile (1 = calmest)")
    left.set_ylabel("imbalance minus day-ahead (EUR/MWh)")
    left.set_xticks(x, [str(i + 1) for i in x])
    left.legend(frameon=False)
    right.plot(x, table["share_short"], marker="o", label="system short")
    right.plot(x, table["share_long"], marker="o", label="system long")
    right.set_xlabel("forecast wind decile (1 = calmest)")
    right.set_ylabel("share of periods")
    right.set_xticks(x, [str(i + 1) for i in x])
    right.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "spread_by_wind.png", dpi=150)
    plt.close(fig)


def plot_hour(table: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(table.index, table["mean_spread"], marker="o", label="mean")
    ax.plot(table.index, table["median_spread"], marker="o", label="median")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("hour of delivery (Danish time)")
    ax.set_ylabel("imbalance minus day-ahead (EUR/MWh)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "spread_by_hour.png", dpi=150)
    plt.close(fig)


def main() -> None:
    pd.set_option("display.width", 140)
    pd.set_option("display.float_format", lambda v: f"{v:,.3f}")
    FIGURES.mkdir(parents=True, exist_ok=True)

    frame = development(build()).dropna(subset=["wind_cubed"])
    frame = frame[~frame["high_traffic"]]
    local = frame.index.tz_convert(TZ)
    print(f"Development data: {local.min():%Y-%m-%d} to {local.max():%Y-%m-%d}, heavy-traffic days excluded\n")

    print("Spread, EUR/MWh")
    print(summarise(frame["spread"]).to_string(), "\n")

    print("Mean spread after removing the largest periods (by absolute spread)")
    print(spike_dependence(frame["spread"]).to_string(index=False), "\n")

    print("Naive strategies, 1 MW in every period, EUR per MWh traded")
    print(f"  always buy day-ahead (profit when short):  {frame['spread'].mean():8.3f}")
    print(f"  always sell day-ahead (profit when long):  {-frame['spread'].mean():8.3f}")
    total = frame["spread"].sum() * MWH_PER_PERIOD
    print(f"  always buy, total over the period:         {total:12,.0f} EUR\n")

    deciles = pd.qcut(frame["wind_cubed"], 10, labels=False)
    wind = by_bucket(frame, "wind_mean", deciles)
    print("By forecast wind (deciles of mean cubed wind speed; wind_mean in m/s)")
    print(wind.to_string(), "\n")
    plot_wind(wind)

    hour = by_bucket(frame, "wind_mean", pd.Series(local.hour, index=frame.index))
    print("By hour of delivery (Danish time)")
    print(hour.drop(columns="wind_mean").to_string(), "\n")
    plot_hour(hour)

    month = by_bucket(frame, "wind_mean", pd.Series(local.strftime("%Y-%m"), index=frame.index))
    print("By month")
    print(month.to_string())


if __name__ == "__main__":
    main()
