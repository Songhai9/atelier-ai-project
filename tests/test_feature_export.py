"""Feature exports preserve keys, types, missingness and complete-run semantics."""

import json

import duckdb
import pandas as pd
import pytest

from atelier_ai.feature_export import export_feature_batches, verify_feature_export


@pytest.fixture
def export_args(tmp_path):
    notebook = tmp_path / "pipeline.ipynb"
    notebook.write_text(json.dumps({"cells": []}))
    return {
        "customer_ids": pd.Index(["001", "002"], dtype="string"),
        "candidate_pools": {"001": ["01", "02"], "002": ["02", "03"]},
        "feature_tables": {"cutoff": pd.Timestamp("2020-09-09")},
        "output_base": tmp_path / "features",
        "feature_columns": ["x", "has_history"],
        "nullable_features": ["x"],
        "sql_schema": {
            "customer_id": "VARCHAR", "article_id": "VARCHAR", "cutoff": "TIMESTAMP",
            "product_group_name": "VARCHAR", "x": "DOUBLE", "has_history": "BOOLEAN",
        },
        "provenance": {"fixture_version": 1}, "notebook_path": notebook,
        "candidates_per_customer": 2, "batch_customers": 1,
    }


def build_fixture(pairs, tables):
    result = pairs.copy()
    result["product_group_name"] = result["customer_id"].map({"001": "A"}).astype("string")
    result["x"] = result["article_id"].map({"01": 1.5, "03": 3.5})
    result["has_history"] = result["customer_id"].eq("001").astype("boolean")
    return result


def test_roundtrip_preserves_nulls_strings_and_types_across_customer_batches(export_args):
    result = export_feature_batches(build_batch=build_fixture, **export_args)
    assert not result["reused"]
    assert result["checks"]["rows"] == 4
    assert result["checks"]["customers"] == 2
    assert result["checks"]["parts"] == 2
    assert result["checks"]["missing_counts"] == {"x": 2, "has_history": 0}
    files = [str(result["directory"] / part["file"]) for part in result["manifest"]["parts"]]
    with duckdb.connect() as connection:
        rows = connection.execute(
            "SELECT customer_id, article_id, product_group_name, x FROM read_parquet(?) "
            "ORDER BY customer_id, article_id", [files]
        ).fetchall()
    assert rows == [("001", "01", "A", 1.5), ("001", "02", "A", None),
                    ("002", "02", None, None), ("002", "03", None, 3.5)]
    assert all(part["customers"] == 1 for part in result["manifest"]["parts"])
    assert result["manifest"]["schema"][3] == {"name": "product_group_name", "type": "VARCHAR"}


def test_identical_export_reuses_verified_files_without_rebuilding(export_args):
    first = export_feature_batches(build_batch=build_fixture, **export_args)
    part = first["directory"] / first["manifest"]["parts"][0]["file"]
    before = part.stat().st_mtime_ns

    def must_not_rebuild(*args):
        raise AssertionError("An identical complete export must be reused.")

    second = export_feature_batches(build_batch=must_not_rebuild, **export_args)
    assert second["reused"] and second["directory"] == first["directory"]
    assert part.stat().st_mtime_ns == before


def test_corrupted_partition_fails_checksum_validation_and_reuse(export_args):
    result = export_feature_batches(build_batch=build_fixture, **export_args)
    path = result["directory"] / result["manifest"]["parts"][0]["file"]
    content = bytearray(path.read_bytes())
    content[-1] ^= 1
    path.write_bytes(content)
    with pytest.raises(ValueError, match="checksum"):
        verify_feature_export(result["directory"])
    with pytest.raises(ValueError, match="checksum"):
        export_feature_batches(build_batch=build_fixture, **export_args)


def test_failed_later_batch_publishes_no_partial_dataset(export_args):
    def failing_builder(pairs, tables):
        if pairs["customer_id"].iloc[0] == "002":
            raise RuntimeError("Synthetic second-batch failure")
        return build_fixture(pairs, tables)

    with pytest.raises(RuntimeError, match="second-batch"):
        export_feature_batches(build_batch=failing_builder, **export_args)
    assert list(export_args["output_base"].iterdir()) == []


def test_builder_cannot_silently_change_prediction_keys(export_args):
    def wrong_cutoff(pairs, tables):
        frame = build_fixture(pairs, tables)
        frame["cutoff"] = pd.Timestamp("2020-09-16")
        return frame

    with pytest.raises(ValueError, match="prediction keys"):
        export_feature_batches(build_batch=wrong_cutoff, **export_args)
    assert list(export_args["output_base"].iterdir()) == []
