"""Daily source coverage and chronological evaluation-window diagnostics."""

from collections import Counter
from collections.abc import Iterable

import pandas as pd


def interval_mask(dates: pd.Series, start, end) -> pd.Series:
    """Select an inclusive-start, exclusive-end interval."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if not start < end:
        raise ValueError("Interval start must precede its end.")
    return dates.ge(start) & dates.lt(end)


def build_daily_activity(
    chunks: Iterable[pd.DataFrame],
) -> tuple[pd.DataFrame, int]:
    """Aggregate exact source dates onto a complete observed calendar.

    Dates must be complete valid ISO calendar dates. Absent dates receive
    a zero source-row count and an explicit False observation flag.
    Input order does not affect daily aggregation.
    """
    counts = Counter()
    total_rows = 0
    for chunk in chunks:
        total_rows += len(chunk)
        counts.update(chunk["t_dat"].value_counts(dropna=False).to_dict())
    if total_rows == 0:
        raise ValueError("No transaction dates were inspected.")
    observed = pd.Series(counts, name="transaction_rows").rename_axis("date").reset_index()
    date_strings = observed["date"].astype("string")
    if not date_strings.str.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", na=False).all():
        raise ValueError("Dates must be complete ISO dates.")
    observed["date"] = pd.to_datetime(date_strings, format="%Y-%m-%d", errors="raise")
    calendar = pd.date_range(observed["date"].min(), observed["date"].max(), freq="D", name="date")
    daily = observed.set_index("date").reindex(calendar)
    daily["observed_in_source"] = daily["transaction_rows"].notna()
    daily["transaction_rows"] = daily["transaction_rows"].fillna(0).astype("int64")
    daily = daily.reset_index()
    if int(daily["transaction_rows"].sum()) != total_rows:
        raise ValueError("Daily transaction counts do not reconcile.")
    return daily, total_rows


def summarize_temporal_coverage(daily: pd.DataFrame) -> pd.DataFrame:
    """Summarize the complete-calendar output of build_daily_activity."""
    if daily.empty:
        raise ValueError("Daily activity is empty.")
    return pd.DataFrame([{
        "first_observed_date": daily["date"].min().date(),
        "last_observed_date": daily["date"].max().date(),
        "calendar_days": len(daily),
        "days_with_records": int(daily["observed_in_source"].sum()),
        "days_without_records": int((~daily["observed_in_source"]).sum()),
        "transaction_rows": int(daily["transaction_rows"].sum()),
    }])


def build_evaluation_windows(
    daily: pd.DataFrame,
    *,
    validation_start,
    test_start,
    end_exclusive,
    horizon_days: int = 7,
) -> pd.DataFrame:
    """Validate and summarize fixed local windows over a complete calendar.

    Dates are explicit parameters; model features and targets are not built.
    The supplied daily table must be produced by build_daily_activity.
    """
    if daily.empty:
        raise ValueError("Daily activity is empty.")
    if horizon_days <= 0:
        raise ValueError("horizon_days must be positive.")
    validation_start = pd.Timestamp(validation_start)
    test_start = pd.Timestamp(test_start)
    end_exclusive = pd.Timestamp(end_exclusive)
    boundaries = (validation_start, test_start, end_exclusive)
    if any(pd.isna(value) or value != value.normalize() for value in boundaries):
        raise ValueError("Window boundaries must be complete midnight dates.")
    horizon = pd.Timedelta(days=horizon_days)
    source_start, source_end = daily["date"].min(), daily["date"].max()
    if source_end + pd.Timedelta(days=1) != end_exclusive:
        raise ValueError("Source coverage differs from the documented evaluation protocol.")
    if not source_start < validation_start:
        raise ValueError("No pre-validation history is available.")
    if test_start - validation_start != horizon:
        raise ValueError("Validation target must match horizon_days.")
    if end_exclusive - test_start != horizon:
        raise ValueError("Test target must match horizon_days.")
    rows = []
    for name, start, end in (
        ("pre_validation_history", source_start, validation_start),
        ("validation_target", validation_start, test_start),
        ("test_target", test_start, end_exclusive),
    ):
        period = daily.loc[interval_mask(daily["date"], start, end)]
        rows.append({
            "period": name, "start_inclusive": start.date(),
            "end_exclusive": end.date(), "calendar_days": len(period),
            "days_with_records": int(period["observed_in_source"].sum()),
            "transaction_rows": int(period["transaction_rows"].sum()),
        })
    summary = pd.DataFrame(rows)
    if int(summary["transaction_rows"].sum()) != int(daily["transaction_rows"].sum()):
        raise ValueError("Evaluation-window counts do not reconcile.")
    return summary
