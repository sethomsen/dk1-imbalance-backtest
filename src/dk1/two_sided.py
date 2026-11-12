"""Two-sided strategies that forecast the spread (hypothesis 3).

The strategies in backtest.py and model.py only buy, and the models there
predict the direction of the imbalance, not its price. Here a position can
also be sold, and the forecast is of the spread itself:

  - Wind rule, two-sided: buy above the 80% quantile of forecast wind (the
    rule in backtest.py) and sell below the 20% quantile.
  - Direction model: a classifier for a short, balanced or long system. Its
    forecast is P(short) times the mean spread of the short periods in the
    training window, plus P(long) times the mean spread of the long ones.
    Balanced periods settle at the day-ahead price and add nothing.
  - Spread regression: gradient boosting on the spread with squared loss.

A model trades 1 MW in the direction of its forecast when the forecast is,
in absolute terms, above the 80% quantile of the out-of-fold forecasts on
the training window. That keeps the trading frequency comparable with the
buy-only strategies. The sized variant of the direction model trades every
period, with the position scaled down in proportion to the forecast below
that threshold.

The models use the features of model.py plus five that features.py derives
from the same data (EXTRA). Whether those add information is tested on the
short-system classifiers of model.py, with and without them, since skill
at predicting the direction is measured far more precisely than PnL.

The last table tests the ten strategies so far against zero and adjusts
for having tried all of them on the same months. forecasts.py repeats it
with its own strategies added.

The strategies, features, thresholds and model settings were fixed before
the first run, and nothing was tuned on the results.

Usage: uv run python -m dk1.two_sided
Takes about six minutes. Writes tables to stdout and a figure to
results/figures/.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict

from dk1 import backtest, model
from dk1.backtest import FIRST_TEST_MONTH, MWH_PER_PERIOD, QUANTILE, TRAIN_MONTHS, day_bootstrap, reality_check, report
from dk1.features import TZ, build, development
from dk1.figures import BLUE, FIGURES, SURFACE, TEXT, TEXT_SECONDARY, save, style

EXTRA = ["wind_ramp", "wind_sensitivity", "wind_revision_gate", "wind_error_d2", "day_ahead_d1"]
FEATURES = model.FEATURES + EXTRA

# Same settings as the gradient boosting classifier in model.py.
BOOSTING = {"max_iter": 200, "learning_rate": 0.05, "max_leaf_nodes": 15, "random_state": 0}


def direction_forecast(train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Expected spread from the probability and the mean spread of each direction."""
    mean_spread = train.groupby("direction")["spread"].mean()
    months_in_train = train.index.tz_convert(TZ).strftime("%Y-%m")
    out_of_fold = cross_val_predict(
        HistGradientBoostingClassifier(**BOOSTING), train[FEATURES], train["direction"],
        groups=months_in_train, cv=GroupKFold(n_splits=TRAIN_MONTHS), method="predict_proba",
    )
    fitted = HistGradientBoostingClassifier(**BOOSTING).fit(train[FEATURES], train["direction"])
    # Both probability matrices have one column per direction, in sorted order.
    weights = mean_spread.reindex(fitted.classes_).to_numpy()
    return out_of_fold @ weights, fitted.predict_proba(test[FEATURES]) @ weights


def regression_forecast(train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    months_in_train = train.index.tz_convert(TZ).strftime("%Y-%m")
    out_of_fold = cross_val_predict(
        HistGradientBoostingRegressor(**BOOSTING), train[FEATURES], train["spread"],
        groups=months_in_train, cv=GroupKFold(n_splits=TRAIN_MONTHS),
    )
    fitted = HistGradientBoostingRegressor(**BOOSTING).fit(train[FEATURES], train["spread"])
    return out_of_fold, fitted.predict(test[FEATURES])


def walk_forward(frame: pd.DataFrame, forecast) -> pd.DataFrame:
    """Spread forecasts for every test period, and the positions they give."""
    months = pd.date_range(FIRST_TEST_MONTH, frame.index.max().tz_convert(TZ), freq="MS")
    pieces = []
    for month in months:
        train = frame[(frame.index >= month - pd.DateOffset(months=TRAIN_MONTHS)) & (frame.index < month)]
        test = frame[(frame.index >= month) & (frame.index < month + pd.DateOffset(months=1))]
        out_of_fold, predicted = forecast(train, test)
        piece = test[["spread", "local_date"]].copy()
        piece["forecast"] = predicted
        piece["threshold"] = np.quantile(np.abs(out_of_fold), QUANTILE)
        pieces.append(piece)
    trades = pd.concat(pieces)
    strong = trades["forecast"].abs() > trades["threshold"]
    trades["position"] = np.where(strong, np.sign(trades["forecast"]), 0.0)
    trades["sized"] = (trades["forecast"] / trades["threshold"]).clip(-1, 1)
    return trades


def wind_two_sided(frame: pd.DataFrame) -> pd.DataFrame:
    """Buy on a strong wind forecast, sell on a calm one."""
    months = pd.date_range(FIRST_TEST_MONTH, frame.index.max().tz_convert(TZ), freq="MS")
    pieces = []
    for month in months:
        train = frame[(frame.index >= month - pd.DateOffset(months=TRAIN_MONTHS)) & (frame.index < month)]
        test = frame[(frame.index >= month) & (frame.index < month + pd.DateOffset(months=1))]
        calm, strong = train["wind_cubed"].quantile([1 - QUANTILE, QUANTILE])
        piece = test[["spread", "local_date"]].copy()
        piece["position"] = (test["wind_cubed"] > strong).astype(float) - (test["wind_cubed"] < calm).astype(float)
        pieces.append(piece)
    return pd.concat(pieces)


def risk(name: str, spread: pd.Series, position: pd.Series, days: pd.Series) -> dict:
    """What a strategy can lose. An imbalance position cannot be closed early."""
    pnl = position * spread * MWH_PER_PERIOD
    daily = pnl.groupby(days).sum()
    worst_days = daily.sort_values().iloc[: max(1, round(0.05 * len(daily)))]
    return {
        "strategy": name,
        "worst_period_eur": pnl.min(),
        "worst_day_eur": daily.min(),
        "mean_of_worst_5pct_days_eur": worst_days.mean(),
        "daily_std_eur": daily.std(),
        "share_of_days_down": (daily < 0).mean(),
        "sharpe_annualised": daily.mean() / daily.std() * np.sqrt(365),
    }


def feature_skill(frame: pd.DataFrame, features: list[str] = FEATURES, label: str = "extra features") -> tuple[pd.DataFrame, dict]:
    """Score the short-system classifiers with the features of model.py and
    with `features`.

    Returns the skill table and the trades of each classifier.
    """
    rows, trades = [], {}
    for name, make_model in model.MODELS.items():
        base = model.walk_forward(frame, make_model)
        extended = model.walk_forward(frame, make_model, features)
        trades[name] = base
        trades[f"{name}, {label}"] = extended

        # Paired: the change in squared error per period, resampled by day.
        change = (extended["probability"] - extended["short"]) ** 2 - (base["probability"] - base["short"]) ** 2
        low, high = np.percentile(day_bootstrap(change, pd.Series(1.0, index=change.index), base["local_date"]), [2.5, 97.5])
        rows.append(
            {
                "model": name,
                "auc": roc_auc_score(base["short"], base["probability"]),
                "auc_extra": roc_auc_score(extended["short"], extended["probability"]),
                "brier_base_rate": brier_score_loss(base["short"], base["base_rate"]),
                "brier": brier_score_loss(base["short"], base["probability"]),
                "brier_extra": brier_score_loss(extended["short"], extended["probability"]),
                "brier_change_ci95_low": low,
                "brier_change_ci95_high": high,
            }
        )
    return pd.DataFrame(rows).set_index("model"), trades


def build_strategies(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """The skill table of feature_skill, and the trades of every strategy
    so far, as frames with the spread, the position and the delivery day."""
    skill, classifiers = feature_skill(frame)
    direction = walk_forward(frame, direction_forecast)
    strategies = {
        "wind rule": backtest.walk_forward(frame),
        "always buy": backtest.walk_forward(frame).assign(position=1),
        **classifiers,
        "wind rule, two-sided": wind_two_sided(frame),
        "direction model": direction,
        "direction model, sized": direction.assign(position=direction["sized"]),
        "spread regression": walk_forward(frame, regression_forecast),
    }
    return skill, strategies


def daily_pnl(strategies: dict) -> pd.DataFrame:
    """PnL in EUR per delivery day, one column per strategy. A day on which
    a strategy has no periods counts as zero."""
    daily = {
        name: (trades["position"] * trades["spread"] * MWH_PER_PERIOD).groupby(trades["local_date"]).sum()
        for name, trades in strategies.items()
    }
    return pd.DataFrame(daily).fillna(0.0)


def plot(strategies: dict, filename: str, title: str) -> None:
    """PnL per MWh with its bootstrap interval, one row per strategy."""
    table = pd.DataFrame(
        [report(name, trades["spread"], trades["position"], trades["local_date"]) for name, trades in strategies.items()]
    )
    style()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 1 + 0.4 * len(table)))
    y = np.arange(len(table))[::-1]
    for yi, (_, row) in zip(y, table.iterrows()):
        ax.plot([row["ci95_low"], row["ci95_high"]], [yi, yi], color=BLUE, linewidth=2, solid_capstyle="round")
        ax.plot(row["eur_per_mwh"], yi, "o", color=BLUE, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
        ax.annotate(f"{row['eur_per_mwh']:+.1f}", (row["ci95_high"], yi), xytext=(6, 0), textcoords="offset points", va="center", color=TEXT, fontsize=9)
    ax.axvline(0, color=TEXT_SECONDARY, linewidth=0.8)
    ax.set_yticks(y, [name[0].upper() + name[1:] for name in table["strategy"]])
    ax.set_xlabel("PnL per MWh with 95% bootstrap interval (EUR)")
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold", fontsize=12, color=TEXT)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(right=ax.get_xlim()[1] + 1.5)
    save(fig, filename)


def main() -> None:
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda v: f"{v:,.3f}")

    frame = model.add_model_columns(development(build())).dropna(subset=FEATURES + ["spread", "direction"])

    skill, strategies = build_strategies(frame)
    print("Skill at predicting a short system, with and without the extra features")
    print("(AUC 0.5 = no skill; lower Brier is better; the interval is for the change in Brier score)")
    print(skill.T.to_string(), "\n")

    wind = strategies["wind rule, two-sided"]
    local = wind.index.tz_convert(TZ)
    print(f"Test period: {local.min():%Y-%m-%d} to {local.max():%Y-%m-%d}, {len(wind):,} periods\n")

    pd.set_option("display.float_format", lambda v: f"{v:,.2f}")
    new = ["wind rule, two-sided", "direction model", "direction model, sized", "spread regression"]
    rows = [report(name, strategies[name]["spread"], strategies[name]["position"], strategies[name]["local_date"]) for name in new]
    print("Two-sided strategies")
    print(pd.DataFrame(rows).set_index("strategy").T.to_string(), "\n")

    rows = []
    for name in ["wind rule, two-sided", "direction model", "spread regression"]:
        trades = strategies[name]
        for leg, position in [("buy", trades["position"].clip(lower=0)), ("sell", trades["position"].clip(upper=0))]:
            if (position != 0).any():
                rows.append(report(f"{name}: {leg}", trades["spread"], position, trades["local_date"]))
    columns = ["periods_traded", "pnl_eur", "eur_per_mwh", "ci95_low", "ci95_high", "median_per_mwh", "hit_rate", "without_top_0.1pct"]
    print("The buy and sell sides separately")
    print(pd.DataFrame(rows).set_index("strategy")[columns].to_string(), "\n")

    rows = [risk(name, trades["spread"], trades["position"], trades["local_date"]) for name, trades in strategies.items()]
    print("Risk, 1 MW positions")
    print(pd.DataFrame(rows).set_index("strategy").to_string(), "\n")

    daily = daily_pnl(strategies)
    print(f"Mean daily PnL against zero, adjusted for trying {daily.shape[1]} strategies on the same months")
    pd.set_option("display.float_format", lambda v: f"{v:,.3f}")
    print(reality_check(daily).to_string())
    print(f"\nTotal PnL over {len(daily)} days (EUR)")
    print(daily.sum().round(0).to_string())

    # The 80% quantile of the wind rule was chosen by hand. backtest.py shows
    # the other quantiles on its own periods; these are on the periods here.
    print("\nWind rule at other quantiles, on the periods of this comparison")
    for quantile in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        other = backtest.walk_forward(frame, quantile=quantile)
        row = report("", other["spread"], other["position"], other["local_date"])
        print(f"  {quantile:4.0%}: {row['periods_traded']:6,d} periods, {row['eur_per_mwh']:5.2f} EUR/MWh, 95% CI [{row['ci95_low']:.2f}, {row['ci95_high']:.2f}]")
    wind_daily = daily["wind rule"]
    print(f"Wind rule: {wind_daily.sum():,.0f} EUR on {len(wind_daily)} days, of which the five best days made {wind_daily.nlargest(5).sum():,.0f}")

    plot(strategies, "two_sided.png", "The first ten strategies: every interval includes zero")


if __name__ == "__main__":
    main()
