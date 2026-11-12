"""Figures for the guide, from development data only.

Usage: uv run python -m dk1.figures
Writes PNG files to results/figures/.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dk1 import backtest, ev_test, model, sensitivity
from dk1.features import TZ, build, development

FIGURES = Path(__file__).resolve().parents[2] / "results" / "figures"

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
# Series colours in a fixed order, chosen to stay distinguishable for
# colour-blind readers. Aqua and yellow have low contrast against the
# surface, so lines are also labelled directly.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
MWH_PER_PERIOD = 0.25
# An ordinary day: the system is long in most quarter-hours and short in
# about a quarter of them, and its largest spread is a few hundred EUR/MWh.
EXAMPLE_DAY = "2026-01-03"


def style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT_SECONDARY,
            "axes.titlecolor": TEXT,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            "font.size": 10,
            "lines.linewidth": 2,
            "legend.frameon": False,
        }
    )


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=160)
    plt.close(fig)


def cumulative_pnl(frame: pd.DataFrame) -> None:
    colors = dict(zip(["Wind rule", "Gradient boosting", "Logistic regression", "Always buy"], [BLUE, ORANGE, AQUA, YELLOW]))
    strategies = {name: (trades, colors[name]) for name, trades in strategy_trades(frame).items()}

    fig, ax = plt.subplots(figsize=(9, 4.2))
    for name, (trades, color) in strategies.items():
        pnl = (trades["position"] * trades["spread"] * MWH_PER_PERIOD).groupby(trades["local_date"]).sum().cumsum()
        ax.plot(pnl.index, pnl.to_numpy(), color=color, label=name)
        ax.annotate(name, (pnl.index[-1], pnl.iloc[-1]), xytext=(6, 0), textcoords="offset points", va="center", color=TEXT_SECONDARY, fontsize=9)
    ax.axhline(0, color=TEXT_SECONDARY, linewidth=0.8)
    ax.set_title("Cumulative PnL, 1 MW positions, walk-forward test", pad=28)
    ax.set_ylabel("EUR")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=4, fontsize=9, handlelength=1.5)
    ax.margins(x=0.02)
    ax.set_xlim(right=ax.get_xlim()[1] + 45)
    save(fig, "cumulative_pnl.png")


def strategy_trades(frame: pd.DataFrame) -> dict:
    # Same rows as model.py: periods without a settled direction are left out.
    model_frame = model.add_model_columns(frame).dropna(subset=model.FEATURES + ["direction"])
    return {
        "Wind rule": backtest.walk_forward(frame),
        "Gradient boosting": model.walk_forward(model_frame, model.MODELS["gradient boosting"]),
        "Logistic regression": model.walk_forward(model_frame, model.MODELS["logistic regression"]),
        "Always buy": backtest.walk_forward(frame).assign(position=1),
    }


def daily_pnl(frame: pd.DataFrame) -> None:
    """Each strategy's net PnL per delivery day, one panel per strategy."""
    strategies = strategy_trades(frame)
    fig, axes = plt.subplots(len(strategies), 1, figsize=(9, 8), sharex=True, sharey=True)
    for ax, (name, trades) in zip(axes, strategies.items()):
        pnl = (trades["position"] * trades["spread"] * MWH_PER_PERIOD).groupby(trades["local_date"]).sum()
        colors = [BLUE if v >= 0 else ORANGE for v in pnl.to_numpy()]
        ax.bar(pnl.index, pnl.to_numpy(), width=1.0, color=colors, linewidth=0)
        ax.axhline(0, color=TEXT_SECONDARY, linewidth=0.8)
        ax.set_title(f"{name}: total {pnl.sum():+,.0f} EUR, {(pnl > 0).sum()} days up, {(pnl < 0).sum()} down", fontsize=10)
        ax.set_ylabel("EUR")
        ax.grid(axis="x", visible=False)
    fig.suptitle("Net PnL per day, 1 MW positions, before trading costs", x=0.01, ha="left", fontweight="bold", fontsize=12, color=TEXT)
    save(fig, "daily_pnl.png")


def spike_dependence(frame: pd.DataFrame) -> None:
    clean = frame[~frame["high_traffic"]]
    order = clean["spread"].abs().sort_values(ascending=False).index
    shares = [0, 0.001, 0.005, 0.01, 0.05]
    means = [clean["spread"].drop(order[: int(round(s * len(clean)))]).mean() for s in shares]
    labels = ["none", "largest 0.1%", "largest 0.5%", "largest 1%", "largest 5%"]

    fig, ax = plt.subplots(figsize=(7, 3.4))
    y = np.arange(len(shares))[::-1]
    ax.barh(y, means, color=BLUE, height=0.55)
    for yi, value in zip(y, means):
        ax.annotate(f"{value:+.1f}", (value, yi), xytext=(4 if value >= 0 else -4, 0), textcoords="offset points", va="center", ha="left" if value >= 0 else "right", color=TEXT, fontsize=9)
    ax.axvline(0, color=TEXT_SECONDARY, linewidth=0.8)
    ax.set_yticks(y, labels)
    ax.set_xlabel("mean spread, imbalance minus day-ahead (EUR/MWh)")
    ax.set_title("A few spikes decide the mean: periods removed by size")
    ax.set_xlim(left=min(means) - 3)
    ax.grid(axis="y", visible=False)
    save(fig, "spike_dependence.png")


def monthly_median(frame: pd.DataFrame) -> None:
    local = frame.index.tz_convert(TZ)
    median = frame["spread"].groupby(local.strftime("%Y-%m")).median()
    fig, ax = plt.subplots(figsize=(9, 3.6))
    x = np.arange(len(median))
    ax.bar(x, median.to_numpy(), color=BLUE, width=0.7)
    change = list(median.index).index("2025-10")
    ax.axvline(change - 0.5, color=TEXT_SECONDARY, linewidth=1, linestyle="--")
    ax.annotate("day-ahead moves to 15-minute resolution", (change - 0.4, median.min()), color=TEXT_SECONDARY, fontsize=9, va="bottom")
    for xi, value in zip(x, median.to_numpy()):
        if value == 0:
            ax.annotate("0", (xi, 0), xytext=(0, -12), textcoords="offset points", ha="center", color=TEXT_SECONDARY, fontsize=9)
    ax.axhline(0, color=TEXT_SECONDARY, linewidth=0.8)
    ax.set_xticks(x, [m[2:] for m in median.index], rotation=45, ha="right")
    ax.set_ylabel("EUR/MWh")
    ax.set_title("Median spread by month: the market shifted in October 2025")
    ax.grid(axis="x", visible=False)
    save(fig, "monthly_median.png")


def share_short_by_wind(frame: pd.DataFrame) -> None:
    clean = frame[~frame["high_traffic"]]
    deciles = pd.qcut(clean["wind_cubed"], 10, labels=False)
    share = (clean["direction"] == 1).groupby(deciles).mean()
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.plot(share.index + 1, share.to_numpy() * 100, color=BLUE, marker="o", markersize=7)
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("forecast wind at gate closure, decile (1 = calmest)")
    ax.set_ylabel("% of periods")
    ax.set_title("System short more often when strong wind is forecast")
    save(fig, "share_short_by_wind.png")


def strategy_intervals(frame: pd.DataFrame) -> None:
    with_energinet = frame.join(sensitivity.energinet_wind(frame.index)).dropna(subset=list(sensitivity.SIGNALS.values()))
    rows = []
    for name, signal in sensitivity.SIGNALS.items():
        t = backtest.walk_forward(with_energinet, signal)
        rows.append(backtest.report(name, t["spread"], t["position"], t["local_date"]))
    t = sensitivity.revision_walk_forward(with_energinet)
    rows.append(backtest.report("Forecast revision after 18:00 (look-ahead)", t["spread"], t["position"], t["local_date"]))
    table = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8, 3.4))
    y = np.arange(len(table))[::-1]
    for yi, (_, row) in zip(y, table.iterrows()):
        ax.plot([row["ci95_low"], row["ci95_high"]], [yi, yi], color=BLUE, linewidth=2, solid_capstyle="round")
        ax.plot(row["eur_per_mwh"], yi, "o", color=BLUE, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
        ax.annotate(f"{row['eur_per_mwh']:+.1f}", (row["ci95_high"], yi), xytext=(6, 0), textcoords="offset points", va="center", color=TEXT, fontsize=9)
    ax.axvline(0, color=TEXT_SECONDARY, linewidth=0.8)
    ax.set_yticks(y, table["strategy"])
    ax.set_xlabel("PnL per MWh with 95% bootstrap interval (EUR)")
    ax.set_title("Only the untradable signal looks good", loc="left")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(right=ax.get_xlim()[1] + 2)
    save(fig, "lookahead.png")


def ev_days(frame: pd.DataFrame) -> None:
    diffs = ev_test.matched_differences(ev_test.daily_table(frame.dropna(subset=["direction"])))
    values = diffs["did_spread"].to_numpy()
    low, high = ev_test.bootstrap_ci(values)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.axhspan(low, high, color=BLUE, alpha=0.12, linewidth=0)
    ax.axhline(values.mean(), color=BLUE, linewidth=2)
    ax.text(0.99, 0.04, f"band: mean {values.mean():+.1f} EUR/MWh, 95% CI [{low:.1f}, {high:.1f}]", transform=ax.transAxes, ha="right", color=TEXT_SECONDARY, fontsize=9)
    ax.plot(diffs.index, values, "o", color=BLUE, markersize=7, markeredgecolor=SURFACE, markeredgewidth=2)
    ax.axhline(0, color=TEXT_SECONDARY, linewidth=0.8)
    ax.set_ylabel("EUR/MWh")
    ax.set_title("Heavy-traffic days: evening effect vs matched control days")
    save(fig, "ev_test.png")


def example_day(frame: pd.DataFrame) -> None:
    """Both prices and the spread for each quarter-hour of one ordinary day."""
    local = frame.index.tz_convert(TZ)
    day = frame[local.strftime("%Y-%m-%d") == EXAMPLE_DAY]
    hours = (day.index - day.index[0]) / pd.Timedelta(hours=1)
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(9, 5.6), sharex=True, height_ratios=[3, 2])
    top.step(hours, day["day_ahead"], where="post", color=TEXT_SECONDARY, label="Day-ahead price")
    top.step(hours, day["imbalance"], where="post", color=BLUE, label="Imbalance price")
    top.set_ylabel("EUR/MWh")
    top.set_title(f"One day, {pd.Timestamp(EXAMPLE_DAY):%-d %B %Y}: prices per quarter-hour", pad=24)
    top.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=9, handlelength=1.5)
    bottom.bar(hours, day["spread"], width=0.2, align="edge", color=BLUE, linewidth=0)
    bottom.axhline(0, color=TEXT_SECONDARY, linewidth=0.8)
    bottom.set_ylabel("spread, EUR/MWh")
    bottom.set_xlabel("hour of delivery (Danish time)")
    bottom.set_xticks(range(0, 25, 3), [f"{h:02d}:00" for h in range(0, 25, 3)])
    bottom.grid(axis="x", visible=False)
    save(fig, "example_day.png")


def spread_histogram(frame: pd.DataFrame) -> None:
    """How often each spread occurs, with the periods at exactly zero apart."""
    spread = frame.loc[~frame["high_traffic"], "spread"]
    edges = np.arange(-200, 420, 20)
    nonzero = spread[spread != 0].clip(-199.9, 399.9)
    share = 100 * np.histogram(nonzero, bins=edges)[0] / len(spread)
    zero = 100 * (spread == 0).mean()
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.bar(edges[:-1], share, width=18, align="edge", color=BLUE, linewidth=0)
    ax.bar(0, zero, width=4, color=TEXT_SECONDARY, linewidth=0)
    ax.annotate(f"exactly 0: {zero:.0f}%", (0, zero), xytext=(8, -4), textcoords="offset points", va="top", color=TEXT, fontsize=9)
    for value, label, align in [(spread.median(), "median", "right"), (spread.mean(), "mean", "left")]:
        ax.annotate(
            f"{label} {value:+.1f}", (value, 0), xytext=(-10 if align == "right" else 10, -32), textcoords="offset points",
            ha=align, color=TEXT, fontsize=9, arrowprops={"arrowstyle": "-", "color": TEXT_SECONDARY, "shrinkB": 0},
        )
    ax.set_xlabel("imbalance minus day-ahead (EUR/MWh); the outermost bars hold everything beyond the axis", labelpad=26)
    ax.set_ylabel("% of quarter-hours")
    ax.set_title("Most quarter-hours lose a buyer a little; a thin tail pays")
    ax.grid(axis="x", visible=False)
    save(fig, "spread_histogram.png")


def comparison_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """The periods on which two_sided.py compares all strategies: those
    where every model has all its inputs."""
    from dk1 import two_sided  # imported here because two_sided imports this module

    return model.add_model_columns(frame).dropna(subset=two_sided.FEATURES + ["direction"])


def edge_and_luck(frame: pd.DataFrame) -> None:
    """Cumulative PnL of the wind rule beside a strategy that uses information from after gate closure."""
    wind = backtest.walk_forward(comparison_rows(frame))
    with_energinet = frame.join(sensitivity.energinet_wind(frame.index)).dropna(subset=list(sensitivity.SIGNALS.values()))
    strategies = {
        "Wind rule": (wind, BLUE),
        "Always buy": (wind.assign(position=1), ORANGE),
        "Forecast revision (not tradable)": (sensitivity.revision_walk_forward(with_energinet), AQUA),
    }
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for name, (trades, color) in strategies.items():
        pnl = (trades["position"] * trades["spread"] * MWH_PER_PERIOD).groupby(trades["local_date"]).sum().cumsum()
        ax.plot(pnl.index, pnl.to_numpy(), color=color, label=name)
        ax.annotate(f"{name.split(' (')[0]} {pnl.iloc[-1]:+,.0f}", (pnl.index[-1], pnl.iloc[-1]), xytext=(6, 0), textcoords="offset points", va="center", color=TEXT_SECONDARY, fontsize=9)
    ax.axhline(0, color=TEXT_SECONDARY, linewidth=0.8)
    ax.set_title("Cumulative PnL on 1 MW: a few jumps against a steady climb", pad=28)
    ax.set_ylabel("EUR")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, fontsize=9, handlelength=1.5)
    ax.margins(x=0.02)
    ax.set_xlim(right=ax.get_xlim()[1] + 70)
    save(fig, "edge_and_luck.png")


def main() -> None:
    style()
    FIGURES.mkdir(parents=True, exist_ok=True)
    frame = development(build()).dropna(subset=["wind_cubed", "spread"])
    example_day(frame)
    spread_histogram(frame)
    edge_and_luck(frame)
    cumulative_pnl(frame)
    daily_pnl(frame)
    spike_dependence(frame)
    monthly_median(frame)
    share_short_by_wind(frame)
    strategy_intervals(frame)
    ev_days(frame)
    print("\n".join(sorted(p.name for p in FIGURES.glob("*.png"))))


if __name__ == "__main__":
    main()
