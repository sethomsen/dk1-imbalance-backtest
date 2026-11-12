# Day-ahead vs imbalance in DK1: a backtest

Every day at noon, tomorrow's electricity is sold in an auction, at the day-ahead price. Whatever the auction gets wrong is corrected in real time, at a second price, the imbalance price. This repository tests whether the gap between the two prices in western Denmark (DK1) can be predicted when the auction closes, from public data. It tests four hypotheses with 17 trading strategies, first on eleven months of data and then once on three months that were held back.

With what I tried, the answer is no. No strategy earns a profit that can be told apart from luck, and the tests were too small to prove a profit of the size seen. The findings that hold are about the weather: the profit is in the error of the wind forecast, and disagreement between weather models predicts that error.

The whole account is in [GUIDE.md](GUIDE.md): the market and the trade, the data and how look-ahead is kept out, each hypothesis with its strategies and results, the holdout, the limits and the conclusion. It assumes no background in power markets.

## Use of AI

This project was developed with heavy use of AI. I worked with Claude Code (Anthropic) throughout. It wrote most of the code and drafted much of the analysis and the text, and some of the ideas that were tested came out of that work. I set the direction, decided what to test and reviewed the results.

## Running it

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run python -m dk1.fetch           # download and cache the raw data under data/raw/
uv run python -m dk1.describe        # the spread, by wind, hour and month
uv run python -m dk1.backtest        # hypothesis 1: the wind rule
uv run python -m dk1.model           # hypothesis 1: the two models
uv run python -m dk1.sensitivity     # the wind rule with later forecasts
uv run python -m dk1.ev_test         # hypothesis 2 on prices
uv run python -m dk1.ev_consumption  # hypothesis 2 on household consumption
uv run python -m dk1.two_sided       # hypothesis 3 and the comparison, about six minutes
uv run python -m dk1.forecasts       # hypothesis 4, about ten minutes
uv run python -m dk1.holdout         # the test on the holdout, about fifteen minutes
uv run python -m dk1.power           # how large an effect the tests could find
uv run python -m dk1.figures         # the figures in results/figures/
uv run pytest
```

The first download takes about two hours. Most of that is the individual forecast runs, which are fetched slowly to stay within the limits of the free Open-Meteo API. It can be interrupted and resumed.

Every script except `holdout.py` and `power.py` reads only the development data, which is delivery before 1 August 2026. `holdout.py` is there to show how the one test was done. Running it again on the same months is not a new test.

## Layout

| File | What it does |
|---|---|
| `src/dk1/fetch.py` | Downloads and caches the raw data |
| `src/dk1/features.py` | Builds one row per quarter-hour with its features, each known at gate closure, and holds the holdout boundary |
| `src/dk1/traffic.py` | The heavy-traffic days, from the calendar |
| `src/dk1/describe.py` | Describes the spread |
| `src/dk1/backtest.py` | The wind rule, the bootstrap, and the correction for trying several strategies |
| `src/dk1/model.py` | Two models that predict a short system |
| `src/dk1/sensitivity.py` | The wind rule with later forecasts |
| `src/dk1/ev_test.py`, `src/dk1/ev_consumption.py` | The heavy-traffic hypothesis, on prices and on household consumption |
| `src/dk1/two_sided.py` | Strategies that can also sell, and the comparison of the first ten |
| `src/dk1/forecasts.py` | Strategies on newer forecasts and on disagreement between weather models |
| `src/dk1/holdout.py` | The one test on the holdout |
| `src/dk1/power.py` | How large an effect the tests could find, and how often the test passes with no effect |
| `src/dk1/figures.py` | The figures |
| `tests/` | Tests of when each feature is available, of the statistics and of the downloader |
| `results/` | The figures, more than the guide uses, and the printed output of the holdout run |

## Data

The raw data is not in the repository. `fetch.py` downloads it from [Energi Data Service](https://www.energidataservice.dk/) (Energinet) and from [Open-Meteo](https://open-meteo.com/), and the terms of those services apply to it. The free Open-Meteo API is for non-commercial use.

## Licence

The code is under the MIT licence. See [LICENSE](LICENSE).
