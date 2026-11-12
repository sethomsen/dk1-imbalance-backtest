"""How large an effect the tests could have found, and how often the
one-sided test passes when there is no effect.

For "ECMWF against the other models" and the wind rule, in the development
test and in the holdout window, this gives the mean and the standard
deviation of the daily PnL, the smallest effect the test would find four
times out of five (80% power at the one-sided 5% level), and how many days
a test would need to find the development effect, or half of it, with that
power.

A simulation then checks the level of the test. Daily PnL with a mean of
exactly zero is drawn in three shapes: normal, with a few large gains (a
buyer who is paid by spikes) and with a few large losses (a seller who pays
for them). The share of samples in which the test passes should be 5%.

It reads the holdout window that holdout.py has already used, and nothing
after it.

Usage: uv run python -m dk1.power
"""

import numpy as np
import pandas as pd

from dk1 import backtest, forecasts, model, sensitivity, two_sided
from dk1.backtest import reality_check
from dk1.features import HOLDOUT_START, build, forecast_features
from dk1.holdout import END, LEVEL

# Standard normal quantiles for a one-sided 5% test with 80% power.
REQUIRED_ERRORS = 1.645 + 0.842
REPETITIONS = 400


def passes_without_effect(draw, days: int, seed: int = 0) -> float:
    """Share of samples of `days` days, drawn with mean zero, in which the test passes."""
    rng = np.random.default_rng(seed)
    passed = 0
    for repetition in range(REPETITIONS):
        daily = pd.DataFrame({"pnl": draw(rng, days)})
        passed += reality_check(daily, seed=repetition)["p"].iloc[0] < LEVEL
    return passed / REPETITIONS


def spikes(sign: int):
    """Daily PnL with mean zero: small normal noise and a spike of 1,800 EUR
    on 1.5% of days, upwards for sign 1 and downwards for sign -1."""
    return lambda rng, days: rng.normal(0, 40, days) + sign * 1800 * ((rng.random(days) < 0.015) - 0.015)


def main() -> None:
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda v: f"{v:,.1f}")

    built = build()
    built = built[built.index < END]
    frame = model.add_model_columns(built.join(forecast_features(built)).join(sensitivity.energinet_wind(built.index)))
    frame = frame.dropna(subset=model.FEATURES + forecasts.NEW + ["spread", "direction"])
    strategies = {
        "ECMWF against the other models": forecasts.signed_rule(frame, "ecmwf_vs_models", 1),
        "wind rule": backtest.walk_forward(frame),
    }

    rows, needed = [], []
    for name, trades in strategies.items():
        parts = {"development": trades[trades.index < HOLDOUT_START], "holdout": trades[trades.index >= HOLDOUT_START]}
        daily = {label: two_sided.daily_pnl({name: part})[name] for label, part in parts.items()}
        for label, part in parts.items():
            mwh_per_day = (part["position"].abs() * backtest.MWH_PER_PERIOD).sum() / len(daily[label])
            error = daily[label].std() / np.sqrt(len(daily[label]))
            rows.append(
                {
                    "strategy": name, "window": label, "days": len(daily[label]),
                    "eur_per_day": daily[label].mean(), "sd_between_days": daily[label].std(),
                    "detectable_eur_per_day": REQUIRED_ERRORS * error,
                    "detectable_eur_per_mwh": REQUIRED_ERRORS * error / mwh_per_day,
                }
            )
        effect = daily["development"].mean()
        for label in parts:
            for share in [1.0, 0.5]:
                days = (REQUIRED_ERRORS * daily[label].std() / (share * effect)) ** 2
                needed.append({"strategy": name, "variation as in": label, "effect": f"{share:.0%} of development", "days": days, "years": days / 365})

    print("Smallest effect each test would find four times out of five")
    print(pd.DataFrame(rows).set_index(["strategy", "window"]).to_string(), "\n")
    print("Days of data needed to find the development effect four times out of five")
    print(pd.DataFrame(needed).set_index(["strategy", "variation as in", "effect"]).to_string(), "\n")

    shapes = {"normal": lambda rng, days: rng.normal(0, 100, days), "a few large gains": spikes(1), "a few large losses": spikes(-1)}
    print(f"Share of samples with no effect in which the test passes (it should be {LEVEL:.0%})")
    table = pd.DataFrame({f"{days} days": {name: passes_without_effect(draw, days) for name, draw in shapes.items()} for days in [92, 333]})
    pd.set_option("display.float_format", lambda v: f"{v:.3f}")
    print(table.to_string())


if __name__ == "__main__":
    main()
