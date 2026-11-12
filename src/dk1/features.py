"""Build one row per 15-minute delivery period with the target and the features.

Every feature must be known at day-ahead gate closure, 12:00 Danish time on
the day before delivery. The places where this needs care:

  - Weather forecasts. Open-Meteo's `previous_day1` value for hour h was
    predicted about 24 hours before h. ECMWF runs every 6 hours and its open
    data arrives a few hours after the run, so a 1-day forecast is treated as
    available 16 hours before h (24 hours, less an 8-hour margin). It is used
    only when that is no later than gate closure; otherwise the 2-day
    forecast is used, which is always available in time.

    This rule is not exact for the two ECMWF models. The archive stores them
    every 3 hours (IFS) and every 6 hours (AIFS) and fills the hours between
    with a spline over the stored values on both sides, which can come from
    later runs. For AIFS that puts real weight on the 06 UTC run of the day
    of gate closure in the hours 01 to 03 UTC, and a few per cent in some
    other hours; for IFS it is a few per cent at 01 and 02 UTC. GFS and ICON
    are stored hourly and are not affected. forecasts.py measures what this
    does to the one strategy that depends on AIFS.
  - Recent spreads. Imbalance prices are settled after the fact, so only days
    that ended before the day of gate closure are used (the day two days
    before delivery and earlier). The same holds for Energinet's forecast
    errors, which are only final once the day is over.
  - Day-ahead prices. The price for the delivery day is the outcome of the
    auction being bid into, so only the previous day's price is used. It
    was published about a day before gate closure.
  - Forecast runs. A run is treated as available 8 hours after it was
    started, the same margin as above. At gate closure, 10:00 or 11:00 UTC,
    that makes the 00 UTC run of the same day the newest one. The 06 UTC
    run is left out, since it may or may not have been published by then.
  - Settled values. The imbalance prices and directions are the final ones,
    downloaded long afterwards. At gate closure only preliminary values for
    the day two days before existed, and how far they are revised is not
    known here.
"""

import glob
from pathlib import Path

import numpy as np
import pandas as pd

from dk1.fetch import OTHER_MODELS, RAW, RUN_HOURS, RUN_MODEL, WEATHER_POINTS
from dk1.traffic import high_traffic_days

TZ = "Europe/Copenhagen"
FORECAST_MARGIN = pd.Timedelta(hours=8)
# Wind speed in m/s above which a turbine produces at rated power.
RATED_SPEED = 13

# Delivery periods from this date (Danish time) are the locked holdout. They
# are used once, after the method is fixed, and never during development.
HOLDOUT_START = pd.Timestamp("2026-08-01", tz=TZ)


def development(frame: pd.DataFrame) -> pd.DataFrame:
    """The part of the data that may be used while developing the method."""
    return frame[frame.index < HOLDOUT_START]


def read_parquets(directory: Path) -> pd.DataFrame:
    files = sorted(glob.glob(str(directory / "*.parquet")))
    return pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)


def load_market() -> pd.DataFrame:
    raw = read_parquets(RAW / "eds" / "ImbalancePrice")
    frame = pd.DataFrame(
        {
            "imbalance": raw["ImbalancePriceEUR"].to_numpy(),
            "day_ahead": raw["SpotPriceEUR"].to_numpy(),
            "direction": raw["DominatingDirection"].to_numpy(),
        },
        index=pd.to_datetime(raw["TimeUTC"]).dt.tz_localize("UTC"),
    )
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    frame["spread"] = frame["imbalance"] - frame["day_ahead"]
    return frame.dropna(subset=["imbalance", "day_ahead"])


def gate_closure(delivery: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """12:00 Danish time on the day before each delivery period, in UTC."""
    # Work in naive local wall time, so that the arithmetic is not shifted by
    # an hour when the delivery day and the day before have different offsets.
    local_day = delivery.tz_convert(TZ).tz_localize(None).normalize()
    gate_local = (local_day - pd.Timedelta(days=1) + pd.Timedelta(hours=12)).tz_localize(TZ)
    return gate_local.tz_convert("UTC")


def load_weather(source: str = "weather") -> pd.DataFrame:
    """Hourly weather forecasts, one column per variable, point and lead time.

    The source "weather_models" holds the wind forecasts of the other
    models, with the model in each column name.
    """
    frames = []
    for point in WEATHER_POINTS:
        raw = read_parquets(RAW / source / point)
        raw = raw.set_index(pd.to_datetime(raw.pop("time")).dt.tz_localize("UTC"))
        raw = raw.drop(columns="point")
        raw.columns = [f"{c}_{point}" for c in raw.columns]
        frames.append(raw)
    weather = pd.concat(frames, axis=1)
    return weather[~weather.index.duplicated(keep="last")].sort_index()


def select_lead(
    weather: pd.DataFrame, delivery: pd.DatetimeIndex, hours_back: int = 0, only_day2: bool = False
) -> pd.DataFrame:
    """For each delivery period, the newest forecast available at gate closure.

    With hours_back, the forecast for that many hours before the delivery
    hour, at the same lead time as the delivery hour's own forecast, so the
    two can be compared without mixing lead times. With only_day2, the 2-day
    forecast is used for every hour.
    """
    hour = delivery.floor("h")
    use_day1 = (hour - pd.Timedelta(hours=24) + FORECAST_MARGIN) <= gate_closure(delivery)
    if only_day2:
        use_day1 = np.zeros(len(delivery), dtype=bool)

    target = hour - pd.Timedelta(hours=hours_back)
    day1 = weather.filter(like="_previous_day1_").reindex(target)
    day2 = weather.filter(like="_previous_day2_").reindex(target)
    day1.columns = [c.replace("_previous_day1", "") for c in day1.columns]
    day2.columns = [c.replace("_previous_day2", "") for c in day2.columns]
    # The two are combined by position below.
    assert list(day1.columns) == list(day2.columns)

    chosen = day2.to_numpy().copy()
    chosen[use_day1] = day1.to_numpy()[use_day1]
    result = pd.DataFrame(chosen, index=delivery, columns=day2.columns)
    result["forecast_lead_days"] = np.where(use_day1, 1, 2)
    return result


def cubed_wind(wind: pd.DataFrame) -> pd.Series:
    """Mean cubed wind speed over the points, capped at rated speed.

    Turbines produce roughly with the cube of wind speed below rated power,
    so this tracks production better than the mean wind speed.
    """
    return (wind.clip(upper=RATED_SPEED) ** 3).mean(axis=1)


def weather_summary(chosen: pd.DataFrame, hour_before: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the point forecasts into a few interpretable features.

    `hour_before` holds the forecasts for the hour before each delivery
    hour, from select_lead with hours_back=1.
    """
    wind = chosen.filter(like="wind_speed_100m_")
    offshore = chosen[[f"wind_speed_100m_{p}" for p in ("horns_rev", "thor", "anholt")]]
    return pd.DataFrame(
        {
            "wind_mean": wind.mean(axis=1),
            "wind_offshore": offshore.mean(axis=1),
            "wind_spread": wind.max(axis=1) - wind.min(axis=1),
            "wind_cubed": cubed_wind(wind),
            # Change in forecast production from the hour before. A front
            # that arrives earlier or later than forecast matters most while
            # production is ramping.
            "wind_ramp": cubed_wind(wind) - cubed_wind(hour_before.filter(like="wind_speed_100m_")),
            # Slope of the cubed curve: how far a forecast error of 1 m/s
            # moves production. Zero above rated speed.
            "wind_sensitivity": (3 * wind**2 * (wind < RATED_SPEED)).mean(axis=1),
            "solar": chosen["shortwave_radiation_central_jutland"],
            "temperature": chosen["temperature_2m_central_jutland"],
            "forecast_lead_days": chosen["forecast_lead_days"],
        },
        index=chosen.index,
    )


def revision_at_gate(weather: pd.DataFrame, delivery: pd.DatetimeIndex) -> pd.Series:
    """How far the newest forecast moved the wind forecast, per delivery day.

    The mean of the 1-day minus the 2-day forecast of cubed wind speed over
    the hours whose 1-day forecast is available at gate closure: from gate
    closure until 16 hours after it. For most of the delivery day only the
    2-day forecast is in the features, so this tells them in which direction
    the newer forecast, which the market has seen, has moved.
    """
    revision = cubed_wind(weather.filter(like="wind_speed_100m_previous_day1_")) - cubed_wind(
        weather.filter(like="wind_speed_100m_previous_day2_")
    )
    revision = revision.asfreq("h")
    hours = int((pd.Timedelta(hours=24) - FORECAST_MARGIN) / pd.Timedelta(hours=1)) + 1
    # Mean over the hours from h to h + 16, labelled by h.
    ahead = revision[::-1].rolling(hours, min_periods=hours).mean()[::-1]
    return pd.Series(ahead.reindex(gate_closure(delivery)).to_numpy(), index=delivery, name="wind_revision_gate")


def load_runs() -> pd.Series:
    """Cubed wind speed in each forecast run, indexed by run and forecast hour."""
    raw = read_parquets(RAW / "runs" / RUN_MODEL)
    raw["cubed"] = raw["wind_speed_100m"].clip(upper=RATED_SPEED) ** 3
    keys = [pd.to_datetime(raw[column]).dt.tz_localize("UTC") for column in ("run", "time")]
    return raw.groupby(keys)["cubed"].mean()


def run_features(runs: pd.Series, delivery: pd.DatetimeIndex) -> pd.DataFrame:
    """The wind forecast of the newest run at gate closure, and how it changed.

    `runs` is indexed by run and forecast hour, as returned by load_runs. A
    revision is the newest run minus an earlier run's forecast for the same
    delivery hour: positive when the forecast for that hour was raised.
    """
    hour = delivery.floor("h")
    step = pd.Timedelta(hours=RUN_HOURS[1] - RUN_HOURS[0])
    newest_run = (gate_closure(delivery) - FORECAST_MARGIN).floor(step)

    def forecast(run: pd.DatetimeIndex) -> np.ndarray:
        return runs.reindex(pd.MultiIndex.from_arrays([run, hour])).to_numpy()

    newest = forecast(newest_run)
    return pd.DataFrame(
        {
            "run_wind_cubed": newest,
            "run_revision_12h": newest - forecast(newest_run - step),
            "run_revision_24h": newest - forecast(newest_run - 2 * step),
        },
        index=delivery,
    )


def model_features(chosen: pd.DataFrame, ecmwf: pd.Series) -> pd.DataFrame:
    """How much the weather models disagree about the wind.

    `chosen` holds the other models' forecasts from select_lead, and `ecmwf`
    the cubed wind speed of the ECMWF forecast at the same lead time.
    """
    others = pd.DataFrame({model: cubed_wind(chosen.filter(like=f"wind_speed_100m_{model}_")) for model in OTHER_MODELS})
    return pd.DataFrame(
        {
            "model_spread": others.assign(ecmwf=ecmwf).std(axis=1, skipna=False),
            # Positive when ECMWF forecasts more wind than the other models.
            "ecmwf_vs_models": ecmwf - others.mean(axis=1, skipna=False),
        },
        index=chosen.index,
    )


def forecast_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Features from the individual forecast runs and from the other models.

    Kept out of build(), so that the analyses that do not use them can run
    without the slow download of the runs. `frame` is the result of build().
    """
    delivery = frame.index
    models = select_lead(load_weather("weather_models"), delivery)
    return pd.concat([run_features(load_runs(), delivery), model_features(models, frame["wind_cubed"])], axis=1)


def load_energinet_wind() -> pd.DataFrame:
    """Energinet's onshore plus offshore wind forecasts in MW, per hour.

    ForecastDayAhead is published at 18:00 Danish time on the day before
    delivery, after gate closure. ForecastCurrent is updated until shortly
    before delivery.
    """
    raw = read_parquets(RAW / "eds" / "Forecasts_Hour")
    raw = raw[raw["ForecastType"].isin(["Onshore Wind", "Offshore Wind"])]
    hour = pd.to_datetime(raw["HourUTC"]).dt.tz_localize("UTC").rename("hour")
    return raw.groupby(hour)[["ForecastDayAhead", "ForecastCurrent"]].sum(min_count=2)


def recent_forecast_error(energinet: pd.DataFrame) -> pd.DataFrame:
    """Energinet's wind forecast error on the day two days before delivery.

    The mean of the current minus the day-ahead forecast, in MW. Indexed by
    the local delivery date, so it can be joined on it.
    """
    error = energinet["ForecastCurrent"] - energinet["ForecastDayAhead"]
    daily = error.groupby(error.index.tz_convert(TZ).date).mean().rename("wind_error_d2").to_frame()
    daily.index = pd.to_datetime(daily.index) + pd.Timedelta(days=2)
    return daily


def previous_day_price(market: pd.DataFrame) -> pd.Series:
    """Day-ahead price 24 hours before each delivery period.

    That is a price of the day before, published before gate closure, with
    one exception. On the 25-hour day in October, the last hour of the day
    gets the price of the first hour of the same day, which is published
    after gate closure: four periods a year.
    """
    earlier = market["day_ahead"].reindex(market.index - pd.Timedelta(hours=24))
    return pd.Series(earlier.to_numpy(), index=market.index, name="day_ahead_d1")


def recent_spreads(market: pd.DataFrame) -> pd.DataFrame:
    """Spread statistics from the day two days before delivery.

    Indexed by the local delivery date, so they can be joined on it.
    """
    local = market.index.tz_convert(TZ)
    by_day = market.groupby(local.date)["spread"]
    daily = pd.DataFrame(
        {
            "spread_mean_d2": by_day.mean(),
            "spread_abs_d2": by_day.apply(lambda s: s.abs().mean()),
            "short_share_d2": market.groupby(local.date)["direction"].apply(lambda s: (s == 1).mean()),
        }
    )
    daily.index = pd.to_datetime(daily.index) + pd.Timedelta(days=2)
    return daily


def calendar_features(delivery: pd.DatetimeIndex) -> pd.DataFrame:
    local = delivery.tz_convert(TZ)
    dates = local.date
    traffic = high_traffic_days(range(local.year.min(), local.year.max() + 1))
    return pd.DataFrame(
        {
            "local_date": pd.to_datetime(dates),
            "hour": local.hour,
            "weekday": local.weekday,
            "month": local.month,
            "high_traffic": np.isin(dates, list(traffic)),
            "evening": (local.hour >= 17) & (local.hour < 23),
        },
        index=delivery,
    )


def build() -> pd.DataFrame:
    market = load_market()
    delivery = market.index
    forecasts = load_weather()
    weather = weather_summary(select_lead(forecasts, delivery), select_lead(forecasts, delivery, hours_back=1))
    weather["wind_revision_gate"] = revision_at_gate(forecasts, delivery)
    calendar = calendar_features(delivery)
    frame = pd.concat([market, weather, calendar, previous_day_price(market)], axis=1)
    frame = frame.join(recent_spreads(market), on="local_date")
    frame = frame.join(recent_forecast_error(load_energinet_wind()), on="local_date")
    return frame
