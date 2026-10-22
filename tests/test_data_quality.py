"""Tests for numerical data-quality diagnostic contracts."""

import pandas as pd
import pytest

from atelier_ai.data_quality import (
    identifier_format_profile,
    numeric_diagnostics,
    profile_categories,
    profile_code_labels,
    profile_completeness,
    profile_entity_keys,
    profile_product_grouping,
    profile_references,
    quality_table,
    summarize_categories,
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


def test_entity_duplicates_include_all_repeated_rows():
    frame = pd.DataFrame({"id": ["01", "02", "01", "", " "]}, dtype="string")
    original = frame.copy(deep=True)
    profile, checks = profile_entity_keys(frame, "id", "entity")
    counts = checks.set_index("check")["affected_rows"].to_dict()
    assert counts == {"missing_keys": 2, "duplicate_key_rows": 2, "surrounding_whitespace": 1}
    assert profile.loc[0, "distinct_nonmissing_keys"] == 2
    pd.testing.assert_frame_equal(frame, original)


def test_reference_missing_and_unknown_are_separate():
    chunks = [pd.DataFrame({"id": ["01", "absent", "", " "]}, dtype="string")]
    result = profile_references(chunks, {"id": pd.Index(["01"])})
    assert result.loc[0, "matched_rows"] == 1
    assert result.loc[0, "unknown_rows"] == 1
    assert result.loc[0, "missing_rows"] == 2
    assert result.loc[0, "matched_pct"] == 25
    assert not result.loc[0, "passed"]


def test_completeness_preserves_textual_placeholders_and_denominator():
    chunks = [
        pd.DataFrame({"value": ["", "NONE"]}, dtype="string"),
        pd.DataFrame({"value": [" ", "NaN"]}, dtype="string"),
    ]
    result = profile_completeness(chunks, "example")
    assert result.loc[0, "rows_examined"] == 4
    assert result.loc[0, "missing_rows"] == 2
    assert result.loc[0, "missing_pct"] == 50


def test_completeness_rejects_changed_chunk_columns():
    chunks = [pd.DataFrame({"a": ["x"]}), pd.DataFrame({"b": ["x"]})]
    with pytest.raises(ValueError, match="columns differ"):
        profile_completeness(chunks, "example")


def test_categories_keep_case_variants_and_blank_values_separate():
    chunks = [
        pd.DataFrame({"category": ["NONE", "None", ""]}, dtype="string"),
        pd.DataFrame({"category": [" ", "Unknown", "NONE"]}, dtype="string"),
    ]
    profile, rows = profile_categories(chunks, ["category"], {"none", "unknown"})
    assert rows == 6
    assert profile.set_index("raw_value")["rows"].to_dict() == {
        "NONE": 2, "None": 1, "": 1, " ": 1, "Unknown": 1,
    }
    summary = summarize_categories(profile, ["category"])
    assert summary.loc[0, "distinct_nonmissing_values"] == 3
    assert summary.loc[0, "missing_rows"] == 2
    assert summary.loc[0, "placeholder_rows"] == 4
    assert summary.loc[0, "whitespace_rows"] == 1
    assert summary.loc[0, "placeholder_pct"] == pytest.approx(200 / 3)


def test_identifier_format_uses_exact_ascii_source_values():
    values = pd.Series(["0123456789", "123456789", "１２３４５６７８９０", "0123456789 ", ""], dtype="string")
    result = identifier_format_profile(values, r"[0-9]{10}")
    assert result["nonmissing_rows"] == 4
    assert result["format_mismatch_rows"] == 3


def test_code_conflicts_are_distinct_from_shared_labels():
    frame = pd.DataFrame({"code": ["01", "02", "03", "03"], "label": ["same", "same", "x", "y"]}, dtype="string")
    summary, shared = profile_code_labels(frame, {"code": "label"})
    assert summary.loc[0, "codes_with_multiple_labels"] == 1
    assert summary.loc[0, "affected_article_rows"] == 2
    assert summary.loc[0, "labels_with_multiple_codes"] == 1
    assert set(shared["code"]) == {"01", "02"}


def test_product_grouping_preserves_article_granularity():
    frame = pd.DataFrame({"article_id": ["a1", "a2", "a3"], "product_code": ["p1", "p1", "p2"]}, dtype="string")
    result = profile_product_grouping(frame)
    assert result.loc[0, "distinct_articles"] == 3
    assert result.loc[0, "distinct_product_codes"] == 2
    assert result.loc[0, "product_codes_with_multiple_articles"] == 1
    assert result.loc[0, "maximum_articles_per_product_code"] == 2
