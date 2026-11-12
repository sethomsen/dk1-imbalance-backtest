import numpy as np
import pandas as pd

from dk1.features import (
    FORECAST_MARGIN,
    gate_closure,
    model_features,
    previous_day_price,
    recent_forecast_error,
    recent_spreads,
    revision_at_gate,
    run_features,
    select_lead,
)


def utc(*stamps):
    return pd.DatetimeIndex(pd.to_datetime(list(stamps))).tz_localize("UTC")


def test_gate_closure_is_noon_danish_time_the_day_before():
    # Summer (UTC+2): delivery 2025-06-10 15:00 local -> gate 2025-06-09 10:00 UTC.
    # Winter (UTC+1): delivery 2026-01-10 15:00 local -> gate 2026-01-09 11:00 UTC.
    gates = gate_closure(utc("2025-06-10 13:00", "2026-01-10 14:00"))
    assert list(gates) == list(utc("2025-06-09 10:00", "2026-01-09 11:00"))


def test_gate_closure_across_daylight_saving_change():
    # Delivery on 2025-10-26 (clocks go back that night); the day before is
    # still summer time, so the gate is 10:00 UTC.
    gates = gate_closure(utc("2025-10-26 12:00"))
    assert list(gates) == list(utc("2025-10-25 10:00"))


def test_day1_forecast_only_used_when_available_before_gate():
    delivery = utc("2025-06-10 00:00", "2025-06-10 06:00")  # 02:00 and 08:00 local
    gate = gate_closure(delivery)
    available = delivery.floor("h") - pd.Timedelta(hours=24) + FORECAST_MARGIN
    assert list(available <= gate) == [True, False]


def hourly(start, end):
    return pd.date_range(start, end, freq="h", tz="UTC")


def test_hour_before_is_taken_at_the_delivery_hours_lead_time():
    index = hourly("2025-06-08", "2025-06-11")
    weather = pd.DataFrame(
        {
            "wind_speed_100m_previous_day1_a": 100.0 + np.arange(len(index)),
            "wind_speed_100m_previous_day2_a": 200.0 + np.arange(len(index)),
        },
        index=index,
    )
    # 02:00 local uses the 1-day forecast, 08:00 local the 2-day forecast.
    delivery = utc("2025-06-10 00:00", "2025-06-10 06:00")
    before = select_lead(weather, delivery, hours_back=1)
    expected = [
        weather.loc["2025-06-09 23:00", "wind_speed_100m_previous_day1_a"],
        weather.loc["2025-06-10 05:00", "wind_speed_100m_previous_day2_a"],
    ]
    assert list(before["wind_speed_100m_a"]) == expected
    assert list(before["forecast_lead_days"]) == [1, 2]


def test_revision_at_gate_only_uses_forecasts_available_at_gate():
    # Delivery 2025-06-10 15:00 local -> gate 2025-06-09 10:00 UTC. A 1-day
    # forecast is available 16 hours ahead, so until 2025-06-10 02:00 UTC.
    index = hourly("2025-06-08", "2025-06-11")
    available = (index >= "2025-06-09 10:00") & (index <= "2025-06-10 02:00")
    weather = pd.DataFrame(
        {
            # 2 m/s (8 when cubed) in the hours known at the gate, 10 m/s elsewhere.
            "wind_speed_100m_previous_day1_a": np.where(available, 2.0, 10.0),
            "wind_speed_100m_previous_day2_a": 0.0,
        },
        index=index,
    )
    revision = revision_at_gate(weather, utc("2025-06-10 13:00"))
    assert revision.iloc[0] == 8.0


def test_previous_day_price_is_24_hours_earlier():
    index = pd.date_range("2025-06-09", "2025-06-11", freq="15min", tz="UTC", inclusive="left")
    market = pd.DataFrame({"day_ahead": np.arange(len(index), dtype=float)}, index=index)
    earlier = previous_day_price(market)
    assert earlier.iloc[:96].isna().all()
    assert (earlier.iloc[96:] == market["day_ahead"].iloc[96:] - 96).all()


def test_recent_forecast_error_is_from_two_days_before_delivery():
    # Danish days 9, 10 and 11 June, with a forecast error equal to the day.
    index = hourly("2025-06-08 22:00", "2025-06-11 21:00")
    day = index.tz_convert("Europe/Copenhagen").day
    energinet = pd.DataFrame({"ForecastDayAhead": 1000.0, "ForecastCurrent": 1000.0 + day}, index=index)
    error = recent_forecast_error(energinet)["wind_error_d2"]
    assert error[pd.Timestamp("2025-06-11")] == 9
    assert error[pd.Timestamp("2025-06-13")] == 11


def test_newest_run_at_gate_is_the_midnight_run_of_the_day_before_delivery():
    # Summer delivery 2025-06-10 15:00 local (gate 10:00 UTC the day before)
    # and winter delivery 2026-01-10 15:00 local (gate 11:00 UTC).
    delivery = utc("2025-06-10 13:00", "2026-01-10 14:00")
    values = {
        ("2025-06-08 00:00", "2025-06-10 13:00"): 100.0,
        ("2025-06-08 12:00", "2025-06-10 13:00"): 110.0,
        ("2025-06-09 00:00", "2025-06-10 13:00"): 125.0,
        # Started after the newest run that had been published at the gate.
        ("2025-06-09 12:00", "2025-06-10 13:00"): 999.0,
        ("2026-01-08 00:00", "2026-01-10 14:00"): 40.0,
        ("2026-01-08 12:00", "2026-01-10 14:00"): 30.0,
        ("2026-01-09 00:00", "2026-01-10 14:00"): 20.0,
        ("2026-01-09 12:00", "2026-01-10 14:00"): 999.0,
    }
    index = pd.MultiIndex.from_tuples(
        [(pd.Timestamp(run, tz="UTC"), pd.Timestamp(hour, tz="UTC")) for run, hour in values]
    )
    features = run_features(pd.Series(list(values.values()), index=index), delivery)
    assert list(features["run_wind_cubed"]) == [125.0, 20.0]
    assert list(features["run_revision_12h"]) == [15.0, -10.0]
    assert list(features["run_revision_24h"]) == [25.0, -20.0]


def test_model_features_need_every_model():
    delivery = utc("2025-06-10 13:00", "2025-06-10 14:00")
    chosen = pd.DataFrame(
        {
            "wind_speed_100m_icon_seamless_a": [2.0, 2.0],  # 8 when cubed
            "wind_speed_100m_gfs_seamless_a": [3.0, np.nan],  # 27
            "wind_speed_100m_ecmwf_aifs025_single_a": [1.0, 1.0],  # 1
        },
        index=delivery,
    )
    features = model_features(chosen, pd.Series([12.0, 12.0], index=delivery))
    assert features["ecmwf_vs_models"].iloc[0] == 0.0
    assert np.isclose(features["model_spread"].iloc[0], np.std([8.0, 27.0, 1.0, 12.0], ddof=1))
    assert features.iloc[1].isna().all()


def test_gate_closure_across_the_spring_clock_change():
    # The clocks go forward on 2026-03-29. For delivery that day the gate is
    # still in winter time (11:00 UTC on the 28th); for the day after it is
    # in summer time (10:00 UTC on the 29th).
    gates = gate_closure(utc("2026-03-29 12:00", "2026-03-30 12:00"))
    assert list(gates) == list(utc("2026-03-28 11:00", "2026-03-29 10:00"))


def test_one_day_forecast_is_used_until_five_in_the_morning_in_both_seasons():
    index = hourly("2025-06-08", "2026-01-11")
    weather = pd.DataFrame(
        {"wind_speed_100m_previous_day1_a": 1.0, "wind_speed_100m_previous_day2_a": 2.0}, index=index
    )
    # 04:00 and 05:00 Danish time, in summer (UTC+2) and in winter (UTC+1).
    delivery = utc("2025-06-10 02:00", "2025-06-10 03:00", "2026-01-10 03:00", "2026-01-10 04:00")
    assert list(select_lead(weather, delivery)["forecast_lead_days"]) == [1, 2, 1, 2]
    assert list(select_lead(weather, delivery, only_day2=True)["wind_speed_100m_a"]) == [2.0] * 4


def test_recent_spreads_are_from_two_days_before_delivery():
    # Danish days 9 to 11 June, with a spread equal to the day of the month.
    index = pd.date_range("2025-06-08 22:00", "2025-06-11 21:45", freq="15min", tz="UTC")
    day = index.tz_convert("Europe/Copenhagen").day
    market = pd.DataFrame({"spread": day.astype(float), "direction": 1.0}, index=index)
    recent = recent_spreads(market)
    assert recent.loc[pd.Timestamp("2025-06-11"), "spread_mean_d2"] == 9.0
    assert recent.loc[pd.Timestamp("2025-06-13"), "spread_mean_d2"] == 11.0
    assert pd.Timestamp("2025-06-10") not in recent.index
