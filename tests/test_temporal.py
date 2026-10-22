"""Calendar-gap and chronological-boundary regression tests."""

import pandas as pd
import pytest

from atelier_ai.temporal import (
    build_daily_activity,
    build_evaluation_windows,
    interval_mask,
    summarize_temporal_coverage,
)


def test_daily_activity_retains_unobserved_date_flag():
    chunks = [
        pd.DataFrame({"t_dat": ["2020-01-03", "2020-01-01"]}, dtype="string"),
        pd.DataFrame({"t_dat": ["2020-01-01"]}, dtype="string"),
    ]
    daily, rows = build_daily_activity(chunks)
    assert rows == 3
    assert daily["transaction_rows"].tolist() == [2, 0, 1]
    assert daily["observed_in_source"].tolist() == [True, False, True]
    assert summarize_temporal_coverage(daily).loc[0, "days_without_records"] == 1


def test_interval_includes_start_and_excludes_end():
    dates = pd.Series(pd.to_datetime(["2020-09-08", "2020-09-09", "2020-09-15", "2020-09-16"]))
    assert interval_mask(dates, "2020-09-09", "2020-09-16").tolist() == [False, True, True, False]


def test_evaluation_windows_partition_calendar_and_rows():
    source = pd.DataFrame({"t_dat": pd.date_range("2020-01-01", "2020-01-15").strftime("%Y-%m-%d")}, dtype="string")
    daily, _ = build_daily_activity([source])
    result = build_evaluation_windows(daily, validation_start="2020-01-02", test_start="2020-01-09", end_exclusive="2020-01-16")
    assert result["calendar_days"].tolist() == [1, 7, 7]
    assert result["transaction_rows"].tolist() == [1, 7, 7]
    with pytest.raises(ValueError, match="Validation target"):
        build_evaluation_windows(daily, validation_start="2020-01-02", test_start="2020-01-08", end_exclusive="2020-01-16")
    with pytest.raises(ValueError, match="Source coverage"):
        build_evaluation_windows(daily, validation_start="2020-01-02", test_start="2020-01-09", end_exclusive="2020-01-17")


@pytest.mark.parametrize("date", ["", "2020-1-01", "2020-02-30"])
def test_daily_activity_rejects_malformed_dates(date):
    with pytest.raises(ValueError):
        build_daily_activity([pd.DataFrame({"t_dat": [date]}, dtype="string")])
