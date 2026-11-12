"""Days with heavy road traffic, known well before day-ahead gate closure.

Daily traffic counts are not openly available, and a trade could not use them
anyway, since they are only known after the fact. The days are therefore
defined from the calendar: departure and return days around Danish school
holidays and long public-holiday weekends.

Each function returns a set of dates. The rules are deliberately simple and
written down here so they can be checked and argued with.
"""

from datetime import date, timedelta

import holidays


def iso_week_day(year: int, week: int, weekday: int) -> date:
    """Date of the given ISO weekday (1 = Monday) in the given ISO week."""
    return date.fromisocalendar(year, week, weekday)


def school_holiday_travel_days(year: int) -> set[date]:
    """Friday before and Sunday at the end of the main school holidays.

    Winter break is week 7 in most of Jutland, autumn break is week 42, and
    the summer break covers roughly weeks 27 to 32. School holiday dates
    vary slightly between municipalities, so these are approximations.
    """
    days = set()
    for start_week, end_week in [(7, 7), (42, 42), (27, 32)]:
        days.add(iso_week_day(year, start_week, 1) - timedelta(days=3))  # Friday before
        days.add(iso_week_day(year, start_week, 1) - timedelta(days=2))  # Saturday it starts
        days.add(iso_week_day(year, end_week, 7))  # last Sunday
    return days


def long_weekend_travel_days(year: int) -> set[date]:
    """Departure and return days around Easter, Ascension and Whitsun, and the
    departure day before Christmas."""
    dk = holidays.Denmark(years=year, language="en_US")
    by_name = {name: day for day, name in dk.items()}

    maundy_thursday = by_name["Maundy Thursday"]
    easter_monday = by_name["Easter Monday"]
    ascension = by_name["Ascension Day"]
    whit_monday = by_name["Pentecost Monday"]

    return {
        maundy_thursday - timedelta(days=1),  # Wednesday before Easter
        easter_monday,
        ascension - timedelta(days=1),  # Wednesday before Ascension
        ascension + timedelta(days=3),  # Sunday after
        whit_monday - timedelta(days=3),  # Friday before Whitsun
        whit_monday,
        date(year, 12, 23),  # Christmas departure
    }


def high_traffic_days(years: range) -> set[date]:
    days = set()
    for year in years:
        days |= school_holiday_travel_days(year)
        days |= long_weekend_travel_days(year)
    return days
