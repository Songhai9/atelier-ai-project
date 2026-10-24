"""Training-group eligibility, complete groups and aligned ranking inputs."""

import duckdb
import pandas as pd
import pytest

from atelier_ai.ranking_data import (
    inspect_training_groups,
    load_eligible_training_rows,
    prepare_ranker_inputs,
)
from atelier_ai.recommendation_features import FEATURE_COLUMNS, feature_storage_schema


@pytest.fixture
def source(tmp_path):
    rows = []
    for customer, labels in [
        ("001", [1, 0, 0]),
        ("002", [0, 0, 0]),
        ("003", [1, 1, 1]),
        ("004", [0, 1, 0]),
    ]:
        for number, label in enumerate(labels):
            row = {
                "cutoff": pd.Timestamp("2020-08-12"),
                "customer_id": customer,
                "article_id": f"0{number}",
                "label": label,
            }
            for feature, kind in feature_storage_schema().items():
                if feature in FEATURE_COLUMNS:
                    row[feature] = (
                        True if kind == "BOOLEAN" else 1.0 if kind == "DOUBLE" else 1
                    )
            rows.append(row)
    frame = pd.DataFrame(rows)
    path = tmp_path / "candidates.parquet"
    with duckdb.connect() as connection:
        connection.register("_fixture", frame)
        connection.sql("SELECT * FROM _fixture").write_parquet(str(path))
    return [path], frame


def audit(paths):
    return inspect_training_groups(
        paths, validation_cutoff="2020-09-09", candidates_per_group=3
    )


def test_eligibility_selects_only_groups_with_both_label_values(source):
    paths, _ = source
    groups = audit(paths).set_index("customer_id")
    assert groups["eligible"].to_dict() == {
        "001": True,
        "002": False,
        "003": False,
        "004": True,
    }
    assert groups.loc["002", "reason"] == "all_zero_labels"
    assert groups.loc["003", "reason"] == "all_one_labels"


def test_filter_keeps_every_positive_and_negative_in_selected_groups(source):
    paths, _ = source
    selected = load_eligible_training_rows(paths, audit(paths))
    assert len(selected) == 6 and selected["label"].sum() == 2
    assert selected.groupby("customer_id").size().to_dict() == {"001": 3, "004": 3}
    assert set(selected["article_id"]) == {"00", "01", "02"}


def test_inputs_sort_and_align_X_y_keys_and_contiguous_groups(source):
    paths, _ = source
    selected = load_eligible_training_rows(paths, audit(paths))
    inputs = prepare_ranker_inputs(selected.iloc[::-1], validation_cutoff="2020-09-09")
    assert inputs["groups"].tolist() == [3, 3]
    assert inputs["groups"].sum() == len(inputs["X"]) == len(inputs["y"]) == 6
    assert inputs["X"].columns.tolist() == FEATURE_COLUMNS
    assert inputs["y"].tolist() == [1, 0, 0, 0, 1, 0]
    assert inputs["keys"]["customer_id"].tolist() == ["001"] * 3 + ["004"] * 3
    assert inputs["group_index"]["row_start"].tolist() == [0, 3]
    assert inputs["group_index"]["row_end_exclusive"].tolist() == [3, 6]


def test_validation_filtering_and_future_training_cutoffs_are_rejected(source):
    paths, frame = source
    with pytest.raises(ValueError, match="restricted to training"):
        inspect_training_groups(
            paths, validation_cutoff="2020-09-09", dataset_split="validation"
        )
    with pytest.raises(ValueError, match="precede validation"):
        prepare_ranker_inputs(
            frame.assign(cutoff=pd.Timestamp("2020-09-09")),
            validation_cutoff="2020-09-09",
        )


def test_target_cannot_enter_feature_contract_and_constant_groups_are_rejected(source):
    _, frame = source
    with pytest.raises(ValueError, match="ordered feature contract"):
        prepare_ranker_inputs(
            frame,
            validation_cutoff="2020-09-09",
            feature_columns=FEATURE_COLUMNS + ["label"],
        )
    with pytest.raises(ValueError, match="positive and negative"):
        prepare_ranker_inputs(frame, validation_cutoff="2020-09-09")


def test_expected_missing_values_preserved_but_invalid_features_rejected(source):
    paths, _ = source
    selected = load_eligible_training_rows(paths, audit(paths))
    selected["customer_article_recency_days"] = float("nan")
    assert (
        prepare_ranker_inputs(selected, validation_cutoff="2020-09-09")["X"][
            "customer_article_recency_days"
        ]
        .isna()
        .all()
    )
    selected["customer_history_rows"] = float("nan")
    with pytest.raises(ValueError, match="Unexpected missing"):
        prepare_ranker_inputs(selected, validation_cutoff="2020-09-09")


def test_duplicate_keys_and_infinite_features_are_rejected(source):
    paths, _ = source
    selected = load_eligible_training_rows(paths, audit(paths))
    with pytest.raises(ValueError, match="unique"):
        prepare_ranker_inputs(
            pd.concat([selected, selected.iloc[:1]]), validation_cutoff="2020-09-09"
        )
    selected["customer_group_share"] = float("inf")
    with pytest.raises(ValueError, match="Infinite"):
        prepare_ranker_inputs(selected, validation_cutoff="2020-09-09")


def test_fractional_labels_cannot_hide_between_binary_min_and_max(source, tmp_path):
    _, frame = source
    frame["label"] = frame["label"].astype("float64")
    frame.loc[1, "label"] = 0.5
    path = tmp_path / "invalid.parquet"
    with duckdb.connect() as connection:
        connection.register("_fixture", frame)
        connection.sql("SELECT * FROM _fixture").write_parquet(str(path))
    with pytest.raises(ValueError, match="binary"):
        audit([path])
