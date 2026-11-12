import numpy as np
import pandas as pd

from dk1.ev_consumption import hourly_table, matched_profiles
from dk1.ev_test import trimmed_mean


def test_trimmed_mean_drops_a_tenth_at_each_end():
    values = np.array([-100.0, 1, 2, 3, 4, 5, 6, 7, 8, 500])
    assert trimmed_mean(values) == 4.5
    assert list(trimmed_mean(np.stack([values, values]), axis=1)) == [4.5, 4.5]


def test_hours_after_midnight_come_from_the_following_day():
    index = pd.date_range("2025-10-10 00:00", "2025-10-11 23:00", freq="h", tz="Europe/Copenhagen").tz_convert("UTC")
    consumption = pd.Series(np.arange(len(index), dtype=float), index=index)
    table = hourly_table(consumption)
    # 00:00 on 11 October is the 25th hour of the series.
    assert table.loc["2025-10-10", 24] == 24.0
    assert table.loc["2025-10-10", 29] == 29.0
    assert np.isnan(table.loc["2025-10-11", 24])


def test_profile_is_per_cent_above_the_same_weekday_in_ordinary_weeks():
    # 10 October 2025 is the Friday before the autumn break.
    days = pd.date_range("2025-09-01", "2025-11-15", freq="D")
    table = pd.DataFrame(100.0, index=days, columns=range(30))
    table.loc["2025-10-10", list(range(17, 23))] = 110.0
    # A different weekday must not be used as a control.
    table.loc["2025-10-09"] = 1000.0
    profiles = matched_profiles(table)
    friday = profiles.loc["2025-10-10"]
    assert np.isclose(friday[17], 10.0)
    assert np.isclose(friday[3], 0.0)
