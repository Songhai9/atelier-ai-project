"""Tests for numerical data-quality diagnostic contracts."""

import pandas as pd
import pytest

from atelier_ai.data_quality import (
    numeric_diagnostics,
    quality_table,
)


def test_numeric_diagnostics_separates_issues_and_preserves_input():
    raw = pd.Series(
        [
            "",
            "  ",
            pd.NA,
            "invalid",
            "NaN",
            "inf",
            "-inf",
            "-2",
            "0",
            "18",
            "18.5",
        ],
        index=[
            "blank",
            "spaces",
            "null",
            "invalid",
            "nan_text",
            "positive_infinity",
            "negative_infinity",
            "negative",
            "zero",
            "adult",
            "fractional",
        ],
        dtype="string",
    )

    original = raw.copy(deep=True)

    numeric, finite, counts = numeric_diagnostics(raw)

    assert counts == {
        "missing": 3,
        "unparseable": 2,
        "nonfinite": 2,
        "negative": 1,
        "zero": 1,
    }

    assert numeric.loc["adult"] == 18
    assert numeric.loc["fractional"] == 18.5
    assert numeric.dtype == pd.Float64Dtype()
    assert int(finite.sum()) == 4

    pd.testing.assert_index_equal(numeric.index, raw.index)
    pd.testing.assert_index_equal(finite.index, raw.index)
    pd.testing.assert_series_equal(raw, original)


def test_quality_table_uses_source_rows_for_overlapping_checks():
    result = quality_table(
        counts={
            "rule_a": 2,
            "rule_b": 2,
        },
        total_rows=3,
    ).set_index("check")

    assert result.loc["rule_a", "affected_rows"] == 2
    assert result.loc["rule_b", "affected_rows"] == 2

    assert result.loc["rule_a", "pct_source_rows"] == pytest.approx(
        200 / 3
    )
    assert result.loc["rule_b", "pct_source_rows"] == pytest.approx(
        200 / 3
    )


@pytest.mark.parametrize("total_rows", [0, -1])
def test_quality_table_rejects_nonpositive_denominator(total_rows):
    with pytest.raises(
        ValueError,
        match="positive source row count",
    ):
        quality_table(
            counts={"missing": 0},
            total_rows=total_rows,
        )