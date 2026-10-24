"""Explicit split names keep training and validation exports distinct."""

import json
import pandas as pd
import pytest

from atelier_ai.feature_export import export_feature_batches


def test_training_and_validation_exports_have_distinct_split_identities(tmp_path):
    notebook = tmp_path / "pipeline.ipynb"
    notebook.write_text(json.dumps({"cells": []}))
    def builder(pairs, tables):
        return pairs.assign(x=1)
    args = dict(customer_ids=pd.Index(["001"], dtype="string"), candidate_pools={"001": ["01"]},
                feature_tables={"cutoff": pd.Timestamp("2020-08-12")}, build_batch=builder,
                output_base=tmp_path/"features", feature_columns=["x"], nullable_features=[],
                sql_schema={"customer_id":"VARCHAR", "article_id":"VARCHAR", "cutoff":"TIMESTAMP", "x":"BIGINT"},
                provenance={"fixture":1}, notebook_path=notebook, candidates_per_customer=1)
    training = export_feature_batches(**args, dataset_split="training")
    validation = export_feature_batches(**args)
    assert training["directory"].name.startswith("training_")
    assert validation["directory"].name.startswith("validation_")
    assert training["manifest"]["fingerprint"] != validation["manifest"]["fingerprint"]
    assert training["manifest"]["dataset_split"] == "training"
    with pytest.raises(ValueError, match="Unknown dataset split"):
        export_feature_batches(**args, dataset_split="invalid")
