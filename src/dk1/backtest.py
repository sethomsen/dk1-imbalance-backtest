"""Walk-forward backtest of the wind baseline rule (hypothesis 1).

Rule: buy 1 MW in day-ahead for a delivery period when the forecast wind
(mean cubed wind speed) is above a quantile of the training window, and
settle at the imbalance price. Otherwise stay flat.

Each test month gets a threshold computed only from the preceding training
window. The window rolls rather than expands, because the spread
distribution shifted around October 2025.

Usage: uv run python -m dk1.backtest
"""

import numpy as np
import pandas as pd

from dk1.features import TZ, build, development

MWH_PER_PERIOD = 0.25
TRAIN_MONTHS = 6
QUANTILE = 0.8
FIRST_TEST_MONTH = pd.Timestamp("2025-09-01", tz=TZ)
REGIME_CHANGE = pd.Timestamp("2025-10-01", tz=TZ)
BOOTSTRAP_SAMPLES = 5000
COSTS = [0.0, 1.0, 2.0, 5.0]


def walk_forward(frame: pd.DataFrame, signal: str = "wind_cubed", quantile: float = QUANTILE) -> pd.DataFrame:
    """Positions for every test period: 1 for buy, 0 for flat."""
    months = pd.date_range(FIRST_TEST_MONTH, frame.index.max().tz_convert(TZ), freq="MS")
    pieces = []
    for month in months:
        train = frame[(frame.index >= month - pd.DateOffset(months=TRAIN_MONTHS)) & (frame.index < month)]
        test = frame[(frame.index >= month) & (frame.index < month + pd.DateOffset(months=1))]
        threshold = train[signal].quantile(quantile)
        piece = test[["spread", signal, "local_date"]].copy()
        piece["threshold"] = threshold
        piece["position"] = (piece[signal] > threshold).astype(int)
        pieces.append(piece)
    return pd.concat(pieces)


def day_bootstrap(pnl: pd.Series, mwh: pd.Series, days: pd.Series, seed: int = 0, block_days: int = 1) -> np.ndarray:
    """Bootstrap the PnL per MWh by resampling whole delivery days.

    With block_days above 1, runs of that many consecutive days are
    resampled together (a circular block bootstrap), which keeps any
    dependence between neighbouring days.
    """
    daily = pd.DataFrame({"pnl": pnl, "mwh": mwh, "day": days}).groupby("day")[["pnl", "mwh"]].sum()
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(daily), size=(BOOTSTRAP_SAMPLES, -(-len(daily) // block_days)))
    picks = (starts[:, :, None] + np.arange(block_days)) % len(daily)
    picks = picks.reshape(BOOTSTRAP_SAMPLES, -1)[:, : len(daily)]
    pnl_sums = daily["pnl"].to_numpy()[picks].sum(axis=1)
    mwh_sums = daily["mwh"].to_numpy()[picks].sum(axis=1)
    return pnl_sums / mwh_sums


def reality_check(daily_pnl: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Test each strategy's mean daily PnL against zero, allowing for the search.

    `daily_pnl` has one row per delivery day and one column per strategy.
    Trying several strategies on the same months and reporting the best one
    overstates it. The adjusted p-value is the share of bootstrap samples in
    which the best of all the strategies, with their means set to zero,
    reaches a t-statistic at least as high as the strategy's own (a
    studentised version of White's reality check). The unadjusted p-value
    treats the strategy as the only one tried. Both are one-sided.
    """
    values = daily_pnl.to_numpy()
    days = len(values)
    mean = values.mean(axis=0)
    error = values.std(axis=0, ddof=1) / np.sqrt(days)
    # A strategy whose PnL never varies (one that never trades, say) has no
    # t-statistic. It counts as no evidence: t = 0.
    t = np.divide(mean, error, out=np.zeros_like(mean), where=error > 0)

    rng = np.random.default_rng(seed)
    resampled = values[rng.integers(0, days, size=(BOOTSTRAP_SAMPLES, days))]
    difference = resampled.mean(axis=1) - mean
    resampled_error = resampled.std(axis=1, ddof=1) / np.sqrt(days)
    # The same holds for a resample in which a strategy's PnL does not vary.
    null_t = np.divide(difference, resampled_error, out=np.zeros_like(difference), where=resampled_error > 0)
    best = null_t.max(axis=1)
    return pd.DataFrame(
        {
            "eur_per_day": mean,
            "t": t,
            "p": (null_t >= t).mean(axis=0),
            "p_adjusted": (best[:, None] >= t).mean(axis=0),
        },
        index=daily_pnl.columns,
    )


def max_drawdown(cumulative: pd.Series) -> float:
    """The largest fall from an earlier peak, where the start at zero counts as a peak."""
    return float((cumulative - cumulative.cummax().clip(lower=0)).min())


def report(name: str, spread: pd.Series, position: pd.Series, days: pd.Series) -> dict:
    """PnL statistics for positions in MW: positive buys, negative sells, 0 is flat."""
    traded = position != 0
    mwh = position.abs() * MWH_PER_PERIOD
    pnl = position * spread * MWH_PER_PERIOD
    per_mwh = pnl.sum() / mwh.sum()
    # A resample of days in which nothing was traded has no PnL per MWh. That
    # only happens to a strategy that trades on few days, and is left out.
    low, high = np.nanpercentile(day_bootstrap(pnl, mwh, days), [2.5, 97.5])

    # Spread from the position's point of view: positive means it made money.
    traded_spread = (np.sign(position) * spread)[traded]
    largest = pnl[traded].abs().sort_values(ascending=False).index[: max(1, int(0.001 * traded.sum()))]
    without_spikes = pnl.drop(largest).sum() / mwh.drop(largest).sum()

    return {
        "strategy": name,
        "periods_traded": int(traded.sum()),
        "mwh_traded": mwh.sum(),
        "pnl_eur": pnl.sum(),
        "eur_per_mwh": per_mwh,
        "ci95_low": low,
        "ci95_high": high,
        "median_per_mwh": traded_spread.median(),
        "hit_rate": (traded_spread > 0).mean(),
        "without_top_0.1pct": without_spikes,
        "max_drawdown_eur": max_drawdown(pnl.cumsum()),
        "break_even_cost": per_mwh,
    }


def main() -> None:
    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", lambda v: f"{v:,.2f}")

    frame = development(build()).dropna(subset=["wind_cubed", "spread"])
    trades = walk_forward(frame)
    local = trades.index.tz_convert(TZ)
    print(f"Test period: {local.min():%Y-%m-%d} to {local.max():%Y-%m-%d}")
    print(f"Rule: buy when forecast wind is above the {QUANTILE:.0%} quantile of the previous {TRAIN_MONTHS} months\n")

    always = pd.Series(1, index=trades.index)
    rows = [
        report("wind rule", trades["spread"], trades["position"], trades["local_date"]),
        report("always buy", trades["spread"], always, trades["local_date"]),
    ]
    for label, mask in [("before Oct 2025", trades.index < REGIME_CHANGE), ("from Oct 2025", trades.index >= REGIME_CHANGE)]:
        part = trades[mask]
        if len(part):
            rows.append(report(f"wind rule, {label}", part["spread"], part["position"], part["local_date"]))
            rows.append(report(f"always buy, {label}", part["spread"], pd.Series(1, index=part.index), part["local_date"]))
    print(pd.DataFrame(rows).set_index("strategy").T.to_string(), "\n")

    # Does the signal add anything over buying unconditionally? Compare the
    # traded periods with the periods the rule stayed out of.
    traded = trades[trades["position"] == 1]
    flat = trades[trades["position"] == 0]
    # Same seed, same set of days: both bootstraps draw the same days, so the
    # difference is paired.
    on = day_bootstrap(trades["spread"] * trades["position"], trades["position"].astype(float), trades["local_date"])
    off = day_bootstrap(trades["spread"] * (1 - trades["position"]), (1 - trades["position"]).astype(float), trades["local_date"])
    low, high = np.percentile(on - off, [2.5, 97.5])
    print("Spread when the rule trades vs when it stays out (EUR/MWh)")
    print(f"  traded: {traded['spread'].mean():.2f}   flat: {flat['spread'].mean():.2f}   difference 95% CI: [{low:.2f}, {high:.2f}]\n")

    # Weather systems last several days. If that made neighbouring days move
    # together, resampling single days would give too narrow an interval.
    pnl = trades["spread"] * trades["position"] * MWH_PER_PERIOD
    mwh = trades["position"] * MWH_PER_PERIOD
    print("Wind rule 95% CI when blocks of consecutive days are resampled together (EUR/MWh)")
    for block_days in [1, 7, 28]:
        low, high = np.percentile(day_bootstrap(pnl, mwh, trades["local_date"], block_days=block_days), [2.5, 97.5])
        print(f"  {block_days:2d}-day blocks: [{low:.2f}, {high:.2f}]")
    daily = pnl.groupby(trades["local_date"]).sum()
    print(f"  autocorrelation of daily PnL at lags 1 to 3: {', '.join(f'{daily.autocorr(lag):.2f}' for lag in [1, 2, 3])}\n")

    # The 80% quantile was chosen by hand, after the descriptive analysis.
    # The other quantiles show how much of the result depends on that choice.
    print("Wind rule at other quantiles")
    for quantile in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        other = walk_forward(frame, quantile=quantile)
        row = report("", other["spread"], other["position"], other["local_date"])
        print(
            f"  {quantile:4.0%}: {row['periods_traded']:6,d} periods, {row['eur_per_mwh']:5.2f} EUR/MWh, "
            f"95% CI [{row['ci95_low']:.2f}, {row['ci95_high']:.2f}], total {row['pnl_eur']:6,.0f} EUR"
        )
    print()

    rule = rows[0]
    print("Net PnL per MWh after trading costs (wind rule)")
    for cost in COSTS:
        print(f"  cost {cost:4.1f} EUR/MWh: {rule['eur_per_mwh'] - cost:7.2f} EUR/MWh, total {rule['pnl_eur'] - cost * rule['mwh_traded']:10,.0f} EUR")

    monthly = (trades["spread"] * trades["position"] * MWH_PER_PERIOD).groupby(local.strftime("%Y-%m")).sum()
    print("\nWind rule PnL by test month (EUR)")
    print(monthly.round(0).to_string())


if __name__ == "__main__":
    main()
