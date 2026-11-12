import os
from datetime import date, datetime

import requests

from dk1 import fetch


class Response:
    def __init__(self, status_code, payload=None):
        self.status_code = status_code
        self.payload = payload
        self.text = ""

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(response=self)

    def json(self):
        if self.payload is None:
            raise requests.JSONDecodeError("Expecting value", "", 0)
        return self.payload


def serve(monkeypatch, responses):
    """Answer successive requests with the given responses, without waiting."""
    queue = list(responses)
    monkeypatch.setattr(fetch.requests, "get", lambda *args, **kwargs: queue.pop(0))
    monkeypatch.setattr(fetch.time, "sleep", lambda seconds: None)


def test_server_errors_and_unreadable_responses_are_retried(monkeypatch):
    serve(monkeypatch, [Response(500), Response(200), Response(200, {"ok": True})])
    assert fetch.get_json("https://example.org", {}) == {"ok": True}


def test_a_run_the_server_keeps_failing_on_is_skipped(monkeypatch):
    serve(monkeypatch, [Response(500)] * 6)
    assert fetch.fetch_run(fetch.datetime(2025, 11, 26, 12, tzinfo=fetch.timezone.utc)) is None


def test_a_run_that_is_not_archived_is_skipped(monkeypatch):
    serve(monkeypatch, [Response(400)])
    assert fetch.fetch_run(fetch.datetime(2030, 1, 1, tzinfo=fetch.timezone.utc)) is None


def test_a_month_file_written_before_the_month_was_complete_is_fetched_again(tmp_path):
    path = tmp_path / "2026-10.parquet"
    assert not fetch.cached(path, date(2026, 10, 1))
    path.write_text("")

    def written(day):
        stamp = datetime(*day, 12).timestamp()
        os.utime(path, (stamp, stamp))

    written((2026, 10, 20))  # half-way through the month
    assert not fetch.cached(path, date(2026, 10, 1))
    written((2026, 11, 10))  # the month is over, but late data may still arrive
    assert not fetch.cached(path, date(2026, 10, 1))
    written((2026, 11, 20))
    assert fetch.cached(path, date(2026, 10, 1))
