"""Exact repetition tests across chunk boundaries and ordering failures."""

from collections import Counter

import pandas as pd
import pytest

from atelier_ai.transactions import (
    TRANSACTION_COLUMNS,
    scan_transactions,
    summarize_repetitions,
)


def transaction_frame():
    a = ["2020-01-01", "c1", "a1", "0.10", "1"]
    b = ["2020-01-01", "c1", "a1", "0.1", "1"]
    c = ["2020-01-02", "c2", "a2", "0.20", "2"]
    return pd.DataFrame([a, a, a, b, c, c], columns=TRANSACTION_COLUMNS, dtype="string")


def test_patterns_cross_chunks_and_preserve_lexical_price_variants():
    frame = transaction_frame()
    original = frame.copy(deep=True)
    result = scan_transactions([frame.iloc[:2], frame.iloc[2:5], frame.iloc[5:]])
    oracle = Counter(frame.value_counts(dropna=False).tolist())
    assert result["multiplicity_counts"] == oracle == {1: 1, 2: 1, 3: 1}
    assert result["rows"] == 6
    assert result["channel_counts"] == {"1": 4, "2": 2}
    assert result["price_min"] == 0.1
    assert result["price_max"] == 0.2
    summary, rates = summarize_repetitions(result["multiplicity_counts"], 6)
    metrics = summary.set_index("metric")["value"].to_dict()
    assert metrics["rows_in_repeated_patterns"] == 5
    assert metrics["additional_repeated_rows"] == 3
    assert rates.loc[1, "pct_source_rows"] == 50
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize("across_chunks", [False, True])
def test_scan_rejects_unordered_dates(across_chunks):
    frame = transaction_frame().iloc[[4, 0]]
    chunks = [frame.iloc[:1], frame.iloc[1:]] if across_chunks else [frame]
    with pytest.raises(ValueError, match="not ordered"):
        scan_transactions(chunks)


@pytest.mark.parametrize("date", ["", "2020-1-01", "2020-02-30"])
def test_scan_rejects_invalid_dates(date):
    frame = transaction_frame().iloc[:1].copy()
    frame.loc[:, "t_dat"] = date
    with pytest.raises(ValueError, match="valid ISO dates"):
        scan_transactions([frame])


def test_repetition_summary_rejects_wrong_source_total():
    with pytest.raises(ValueError, match="do not reconcile"):
        summarize_repetitions({1: 1, 2: 1}, total_rows=4)
