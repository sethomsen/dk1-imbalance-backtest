"""Download the raw data and cache it under data/raw/.

Sources:
  - Energi Data Service (Energinet): imbalance prices (the dataset also
    holds the day-ahead price of each period), Energinet's own wind and
    solar forecasts for DK1, and hourly consumption by consumer category
    for the three regions that make up DK1.
  - Open-Meteo Previous Runs API: archived ECMWF IFS forecasts at fixed lead
    times, used as the signal because they existed before day-ahead gate
    closure (see features.py for how the lead time is chosen per hour). The
    same archive gives the wind forecasts of three other models, used to
    measure how much the models disagree.
  - Open-Meteo Single Runs API: every 00 and 12 UTC run of the ECMWF IFS
    9 km model as it was issued, used to give the features the forecast a
    trader had at gate closure and how it changed from the run before.

Each source is fetched in monthly chunks and written as one parquet file per
chunk, so a rerun only downloads what is missing. A month is fetched again
until its file was written well after the month ended, since its data keeps
arriving for a while. Forecast runs are written as one file per run and
never refetched.

The free Open-Meteo API allows 5,000 calls an hour and 10,000 a day, and a
request for seven points counts as seven. The runs are therefore fetched
slowly: a first full download takes about two hours, and can be interrupted
and resumed.

Usage: uv run python -m dk1.fetch
"""

import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"

# Energinet changed how the imbalance price is calculated on this date, and
# its ImbalancePrice dataset starts here. Earlier prices are not comparable.
START = date(2025, 3, 19)

EDS_URL = "https://api.energidataservice.dk/dataset/{name}"
OPEN_METEO_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
SINGLE_RUNS_URL = "https://single-runs-api.open-meteo.com/v1/forecast"

# Each dataset: (time column, first date, last date). None means up to today.
EDS_DATASETS = {
    "ImbalancePrice": ("TimeUTC", START, None),
    "Forecasts_Hour": ("HourUTC", START, None),
    "ConsumptionConsumerCategoryHour": ("TimeUTC", START, None),
}

# Most datasets are split by price area. Consumption by consumer category is
# split by region instead; DK1 is Jutland and Funen, which is these three.
DK1_FILTER = '{"PriceArea":["DK1"]}'
EDS_FILTERS = {
    "ConsumptionConsumerCategoryHour": '{"RegionName":["Region Midtjylland","Region Nordjylland","Region Syddanmark"]}',
}

# Points that cover the main DK1 wind areas, plus one inland point for solar
# and temperature. Coordinates are approximate.
WEATHER_POINTS = {
    "horns_rev": (55.53, 7.85),
    "thor": (56.35, 7.95),
    "anholt": (56.60, 11.20),
    "west_jutland": (56.10, 8.40),
    "north_jutland": (57.10, 9.20),
    "south_jutland": (55.10, 9.00),
    "central_jutland": (56.20, 9.40),
}
WEATHER_MODEL = "ecmwf_ifs025"
WEATHER_VARIABLES = ["wind_speed_100m", "shortwave_radiation", "temperature_2m"]
LEAD_DAYS = [1, 2]

# Other global models in the same archive with forecasts one and two days
# ahead for the whole period: DWD ICON, NCEP GFS and ECMWF's AI model.
OTHER_MODELS = ["icon_seamless", "gfs_seamless", "ecmwf_aifs025_single"]

# Individual forecast runs. Each run is fetched 72 hours ahead, which reaches
# the end of the delivery day from the 00 UTC run two days before it.
RUN_MODEL = "ecmwf_ifs"
RUN_HOURS = [0, 12]
RUN_FORECAST_DAYS = 3
# A run is requested once it is this old, well after it has been published.
RUN_PUBLISHED_AFTER = timedelta(hours=12)
RUN_PAUSE_SECONDS = 6


def month_starts(start: date, end: date) -> list[date]:
    months = []
    current = date(start.year, start.month, 1)
    while current <= end:
        months.append(current)
        current = date(current.year + current.month // 12, current.month % 12 + 1, 1)
    return months


def chunk_bounds(month: date, start: date, end: date) -> tuple[date, date]:
    """First and last day of `month` that fall within [start, end]."""
    next_month = date(month.year + month.month // 12, month.month % 12 + 1, 1)
    last = min(end, next_month - timedelta(days=1))
    return max(start, month), last


class NotAvailable(Exception):
    """The server kept failing on a request that was sent correctly."""


def get_json(url: str, params: dict, retries: int = 6) -> dict:
    """Fetch JSON, retrying what is likely to pass: a dropped connection, a
    rate limit, a server error, or a response that is not JSON. The servers
    now and then answer a valid request with a 500 or an empty body, and the
    same request succeeds later."""
    for attempt in range(retries):
        wait = 30 * (attempt + 1)
        try:
            response = requests.get(url, params=params, timeout=120)
        except (requests.ConnectionError, requests.Timeout) as error:
            print(f"  {type(error).__name__}, retrying in {wait} s", flush=True)
            time.sleep(wait)
            continue
        if response.status_code == 429 or response.status_code >= 500:
            reason = "rate limited" if response.status_code == 429 else f"HTTP {response.status_code}"
            print(f"  {reason}, waiting {wait} s", flush=True)
            time.sleep(wait)
            continue
        response.raise_for_status()
        try:
            return response.json()
        except requests.JSONDecodeError:
            print(f"  response is not JSON, retrying in {wait} s", flush=True)
            time.sleep(wait)
    raise NotAvailable(f"giving up on {url} after {retries} attempts")


def fetch_eds(name: str, time_column: str, first: date, last: date) -> pd.DataFrame:
    params = {
        "start": first.isoformat(),
        "end": (last + timedelta(days=1)).isoformat(),
        "filter": EDS_FILTERS.get(name, DK1_FILTER),
        "sort": f"{time_column} ASC",
        "limit": 0,
    }
    records = get_json(EDS_URL.format(name=name), params)["records"]
    return pd.DataFrame.from_records(records)


def fetch_weather(
    point: str, first: date, last: date, models: list[str] = [WEATHER_MODEL], variables: list[str] = WEATHER_VARIABLES
) -> pd.DataFrame:
    """Forecasts made one and two days ahead. With several models, each
    column name ends in the model it comes from."""
    lat, lon = WEATHER_POINTS[point]
    columns = [f"{var}_previous_day{d}" for var in variables for d in LEAD_DAYS]
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(columns),
        "models": ",".join(models),
        "start_date": first.isoformat(),
        "end_date": last.isoformat(),
        "wind_speed_unit": "ms",
        "timezone": "GMT",
    }
    hourly = get_json(OPEN_METEO_URL, params)["hourly"]
    frame = pd.DataFrame(hourly)
    frame["point"] = point
    return frame


def run_times(start: date, now: datetime) -> list[datetime]:
    """Initialisation times of the runs to fetch, all 00 UTC runs first.

    Starts two days before `start`, so that the first delivery day has the
    runs from the two days before it.
    """
    first = datetime(start.year, start.month, start.day, tzinfo=timezone.utc) - timedelta(days=2)
    days = (now - first).days + 1
    runs = [first + timedelta(days=d, hours=hour) for hour in RUN_HOURS for d in range(days)]
    return [run for run in runs if run + RUN_PUBLISHED_AFTER <= now]


def fetch_run(run: datetime) -> pd.DataFrame | None:
    """One forecast run for all weather points, or None if the archive cannot serve it.

    A run that is not archived gives a 400. A run on which the server keeps
    failing, after the retries in get_json, is skipped as well, so that one
    bad run does not stop the download. A skipped run has no cache file and
    is asked for again on the next download.
    """
    params = {
        "latitude": ",".join(str(lat) for lat, _ in WEATHER_POINTS.values()),
        "longitude": ",".join(str(lon) for _, lon in WEATHER_POINTS.values()),
        "hourly": ",".join(WEATHER_VARIABLES),
        "models": RUN_MODEL,
        "run": f"{run:%Y-%m-%dT%H:%M}",
        "forecast_days": RUN_FORECAST_DAYS,
        "wind_speed_unit": "ms",
        "timezone": "GMT",
    }
    try:
        locations = get_json(SINGLE_RUNS_URL, params)
    except requests.HTTPError as error:
        if error.response.status_code != 400:
            raise
        print(f"  not fetched: {error.response.text[:200]}", flush=True)
        return None
    except NotAvailable as error:
        print(f"  not fetched: {error}"[:200], flush=True)
        return None
    frames = []
    for point, location in zip(WEATHER_POINTS, locations):
        frame = pd.DataFrame(location["hourly"])
        frame["point"] = point
        frames.append(frame)
    frame = pd.concat(frames, ignore_index=True)
    frame["run"] = f"{run:%Y-%m-%dT%H:%M}"
    return frame


# Energinet's consumption data arrives up to 15 days late.
COMPLETE_AFTER = timedelta(days=16)


def cached(path: Path, month: date) -> bool:
    """Whether the file of a month holds the whole month.

    It does if it was written long enough after the month ended. A file
    written earlier is fetched again, so that a month first downloaded
    half-way through is completed later.
    """
    if not path.exists():
        return False
    next_month = date(month.year + month.month // 12, month.month % 12 + 1, 1)
    return date.fromtimestamp(path.stat().st_mtime) >= next_month + COMPLETE_AFTER


def fetch_eds_dataset(name: str, today: date) -> None:
    """Download the months of one Energi Data Service dataset that are not cached.

    The missing months are fetched in one request and written as one file
    per month. Some datasets allow only one request every few minutes, so
    a request per month would take hours.
    """
    time_column, start, end = EDS_DATASETS[name]
    folder = RAW / "eds" / name
    missing = [m for m in month_starts(start, end or today) if not cached(folder / f"{m:%Y-%m}.parquet", m)]
    if not missing:
        return
    first, _ = chunk_bounds(missing[0], start, end or today)
    _, last = chunk_bounds(missing[-1], start, end or today)
    print(f"{name} {first} to {last}", flush=True)
    frame = fetch_eds(name, time_column, first, last)
    if frame.empty:
        return
    # The request is in Danish dates, so the months are split on Danish time.
    month_of_row = frame[time_column.replace("UTC", "DK")].str.slice(0, 7)
    folder.mkdir(parents=True, exist_ok=True)
    for month, rows in frame.groupby(month_of_row):
        rows.reset_index(drop=True).to_parquet(folder / f"{month}.parquet")
    time.sleep(2)


def main() -> None:
    today = date.today()
    months = month_starts(START, today)

    for name in EDS_DATASETS:
        fetch_eds_dataset(name, today)

    for point in WEATHER_POINTS:
        for month in months:
            first, last = chunk_bounds(month, START, today)
            path = RAW / "weather" / point / f"{month:%Y-%m}.parquet"
            if cached(path, month):
                continue
            print(f"weather {point} {month:%Y-%m}")
            frame = fetch_weather(point, first, last)
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(path)
            time.sleep(1)

    for point in WEATHER_POINTS:
        for month in months:
            first, last = chunk_bounds(month, START, today)
            path = RAW / "weather_models" / point / f"{month:%Y-%m}.parquet"
            if cached(path, month):
                continue
            print(f"weather models {point} {month:%Y-%m}")
            frame = fetch_weather(point, first, last, models=OTHER_MODELS, variables=["wind_speed_100m"])
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(path)
            time.sleep(1)

    for run in run_times(START, datetime.now(timezone.utc)):
        path = RAW / "runs" / RUN_MODEL / f"{run:%Y-%m-%dT%H}.parquet"
        if path.exists():
            continue
        print(f"run {run:%Y-%m-%d %H} UTC", flush=True)
        frame = fetch_run(run)
        if frame is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(path)
        time.sleep(RUN_PAUSE_SECONDS)


if __name__ == "__main__":
    main()
