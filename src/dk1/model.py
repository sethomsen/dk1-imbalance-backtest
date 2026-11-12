"""Walk-forward models of whether the system will be short (hypothesis 1).

Two classifiers predict, at day-ahead gate closure, the probability that a
delivery period ends with up-regulation (the system short). A period is
bought when its predicted probability is in the top 20% of out-of-fold
predictions on the training window, which keeps the trading frequency
comparable with the wind threshold rule in backtest.py. Out-of-fold means
each training month is predicted by a model fitted on the other months, so
the threshold is not set on predictions the model has already seen.

The heavy-traffic flag is not a feature; it is kept for its own test.

Usage: uv run python -m dk1.model
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dk1.backtest import FIRST_TEST_MONTH, REGIME_CHANGE, TRAIN_MONTHS, report
from dk1.features import TZ, build, development

QUANTILE = 0.8

FEATURES = [
    "wind_mean",
    "wind_offshore",
    "wind_spread",
    "wind_cubed",
    "solar",
    "temperature",
    "hour_sin",
    "hour_cos",
    "weekend",
    "spread_mean_d2",
    "spread_abs_d2",
    "short_share_d2",
    "forecast_lead_days",
]

MODELS = {
    "logistic regression": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "gradient boosting": lambda: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, random_state=0),
}


def add_model_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    angle = 2 * np.pi * (frame["hour"] + frame.index.tz_convert(TZ).minute / 60) / 24
    frame["hour_sin"] = np.sin(angle)
    frame["hour_cos"] = np.cos(angle)
    frame["weekend"] = (frame["weekday"] >= 5).astype(int)
    frame["short"] = (frame["direction"] == 1).astype(int)
    return frame


def walk_forward(frame: pd.DataFrame, make_model, features: list[str] = FEATURES) -> pd.DataFrame:
    months = pd.date_range(FIRST_TEST_MONTH, frame.index.max().tz_convert(TZ), freq="MS")
    pieces = []
    for month in months:
        train = frame[(frame.index >= month - pd.DateOffset(months=TRAIN_MONTHS)) & (frame.index < month)]
        test = frame[(frame.index >= month) & (frame.index < month + pd.DateOffset(months=1))]
        months_in_train = train.index.tz_convert(TZ).strftime("%Y-%m")
        out_of_fold = cross_val_predict(
            make_model(), train[features], train["short"],
            groups=months_in_train, cv=GroupKFold(n_splits=TRAIN_MONTHS), method="predict_proba",
        )[:, 1]
        threshold = np.quantile(out_of_fold, QUANTILE)
        model = make_model().fit(train[features], train["short"])
        piece = test[["spread", "short", "local_date"]].copy()
        piece["probability"] = model.predict_proba(test[features])[:, 1]
        piece["position"] = (piece["probability"] > threshold).astype(int)
        # Base rate a forecaster would use with no model: the training share.
        piece["base_rate"] = train["short"].mean()
        pieces.append(piece)
    return pd.concat(pieces)


def main() -> None:
    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", lambda v: f"{v:,.3f}")

    frame = add_model_columns(development(build())).dropna(subset=FEATURES + ["spread", "direction"])

    rows, skill = [], []
    for name, make_model in MODELS.items():
        trades = walk_forward(frame, make_model)
        skill.append(
            {
                "model": name,
                "auc": roc_auc_score(trades["short"], trades["probability"]),
                "brier": brier_score_loss(trades["short"], trades["probability"]),
                "brier_base_rate": brier_score_loss(trades["short"], trades["base_rate"]),
            }
        )
        rows.append(report(name, trades["spread"], trades["position"], trades["local_date"]))
        for label, mask in [("before Oct 2025", trades.index < REGIME_CHANGE), ("from Oct 2025", trades.index >= REGIME_CHANGE)]:
            part = trades[mask]
            rows.append(report(f"{name}, {label}", part["spread"], part["position"], part["local_date"]))

        local = trades.index.tz_convert(TZ)
        monthly = (trades["spread"] * trades["position"] * 0.25).groupby(local.strftime("%Y-%m")).sum()
        print(f"{name}: PnL by test month (EUR)")
        print(monthly.round(0).to_string(), "\n")

    print("Out-of-sample skill at predicting a short system (AUC 0.5 = no skill; lower Brier is better)")
    print(pd.DataFrame(skill).set_index("model").to_string(), "\n")
    print(pd.DataFrame(rows).set_index("strategy").T.to_string())


if __name__ == "__main__":
    main()
