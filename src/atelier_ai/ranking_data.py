"""Training-only eligibility and aligned inputs for grouped recommendation ranking."""

import duckdb
import pandas as pd

from atelier_ai.recommendation_features import (
    FEATURE_COLUMNS,
    EXPECTED_MISSING_FEATURES,
)


GROUP_KEYS = ["cutoff", "customer_id"]
ROW_KEYS = GROUP_KEYS + ["article_id"]


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def inspect_training_groups(
    parquet_paths,
    *,
    validation_cutoff,
    candidates_per_group=300,
    dataset_split="training",
):
    """Audit all historical groups; select groups with both binary label values."""
    _require(
        dataset_split == "training", "Eligibility filtering is restricted to training."
    )
    _require(len(parquet_paths) > 0, "Historical training Parquet files are required.")
    with duckdb.connect(config={"threads": 2, "memory_limit": "512MB"}) as connection:
        groups = connection.execute(
            "SELECT cutoff, customer_id, count(*) AS candidate_rows, "
            "count(label) AS labelled_rows, sum(label) AS positive_rows, "
            "min(label) AS min_label, max(label) AS max_label "
            "FROM read_parquet(?) GROUP BY cutoff, customer_id ORDER BY cutoff, customer_id",
            [list(map(str, parquet_paths))],
        ).df()
    _require(not groups[GROUP_KEYS].isna().any().any(), "Group keys must be present.")
    _require(
        groups["cutoff"].lt(pd.Timestamp(validation_cutoff)).all(),
        "Ranker training cutoffs must precede validation.",
    )
    _require(
        groups["candidate_rows"].eq(candidates_per_group).all(),
        "Each group must preserve the declared candidate count.",
    )
    _require(
        groups["labelled_rows"].eq(groups["candidate_rows"]).all()
        and groups[["min_label", "max_label"]].isin([0, 1]).all().all(),
        "Every candidate must have a nonmissing binary label.",
    )
    # Binary min/max alone do not catch an intermediate fractional value.
    with duckdb.connect(config={"threads": 2, "memory_limit": "512MB"}) as connection:
        invalid = connection.execute(
            "SELECT count(*) FROM read_parquet(?) WHERE label NOT IN (0, 1) OR label IS NULL",
            [list(map(str, parquet_paths))],
        ).fetchall()[0][0]
    _require(invalid == 0, "Every candidate must have a nonmissing binary label.")
    groups["positive_rows"] = groups["positive_rows"].astype("int64")
    groups["negative_rows"] = groups["candidate_rows"] - groups["positive_rows"]
    groups["eligible"] = groups["positive_rows"].gt(0) & groups["negative_rows"].gt(0)
    groups["reason"] = "has_positive_and_negative"
    groups.loc[groups["positive_rows"].eq(0), "reason"] = "all_zero_labels"
    groups.loc[groups["negative_rows"].eq(0), "reason"] = "all_one_labels"
    return groups.drop(columns=["labelled_rows", "min_label", "max_label"])


def load_eligible_training_rows(
    parquet_paths, group_audit, *, dataset_split="training"
):
    """Fetch complete eligible groups, retaining all their candidate rows."""
    _require(
        dataset_split == "training", "Eligibility filtering is restricted to training."
    )
    _require(
        not group_audit.duplicated(GROUP_KEYS).any(), "Group audit keys must be unique."
    )
    selected = group_audit.loc[group_audit["eligible"], GROUP_KEYS].copy()
    _require(not selected.empty, "No informative training groups are available.")
    with duckdb.connect(config={"threads": 2, "memory_limit": "512MB"}) as connection:
        connection.register("_eligible_groups", selected)
        frame = connection.execute(
            "SELECT candidates.* FROM read_parquet(?) candidates "
            "INNER JOIN _eligible_groups USING(cutoff, customer_id) "
            "ORDER BY candidates.cutoff, candidates.customer_id, candidates.article_id",
            [list(map(str, parquet_paths))],
        ).df()
    expected_rows = int(
        group_audit.loc[group_audit["eligible"], "candidate_rows"].sum()
    )
    _require(
        len(frame) == expected_rows, "Eligible group rows were lost or multiplied."
    )
    actual_sizes = frame.groupby(GROUP_KEYS).size()
    expected_sizes = group_audit.loc[group_audit["eligible"]].set_index(GROUP_KEYS)[
        "candidate_rows"
    ]
    _require(
        actual_sizes.sort_index().equals(expected_sizes.sort_index().rename(None)),
        "Selected groups differ from the audited population.",
    )
    return frame


def prepare_ranker_inputs(frame, *, validation_cutoff, feature_columns=None):
    """Return X, y, contiguous group sizes and audit keys without fitted transforms."""
    columns = FEATURE_COLUMNS if feature_columns is None else list(feature_columns)
    _require(columns == FEATURE_COLUMNS, "Use the exact ordered feature contract.")
    _require(
        not frame.empty and set(ROW_KEYS + columns + ["label"]).issubset(frame.columns),
        "Candidate keys, features and labels are required.",
    )
    _require(
        not frame[ROW_KEYS].isna().any().any() and not frame.duplicated(ROW_KEYS).any(),
        "Candidate keys must be present and unique.",
    )
    _require(
        frame["cutoff"].lt(pd.Timestamp(validation_cutoff)).all(),
        "Ranker training cutoffs must precede validation.",
    )
    _require(
        frame["label"].isin([0, 1]).all(), "Training labels must be binary and present."
    )
    _require(
        all(pd.api.types.is_numeric_dtype(frame[column]) for column in columns),
        "Model features must remain numeric or boolean.",
    )
    required = [column for column in columns if column not in EXPECTED_MISSING_FEATURES]
    _require(
        not frame[required].isna().any().any(), "Unexpected missing model feature."
    )
    _require(
        not frame[columns].isin([float("inf"), -float("inf")]).any().any(),
        "Infinite model features are not supported.",
    )
    ordered = frame.sort_values(ROW_KEYS, kind="stable").reset_index(drop=True)
    group_index = (
        ordered.groupby(GROUP_KEYS, sort=False)
        .agg(
            group_size=("label", "size"),
            positive_rows=("label", "sum"),
        )
        .reset_index()
    )
    _require(
        (
            group_index["positive_rows"].gt(0)
            & group_index["positive_rows"].lt(group_index["group_size"])
        ).all(),
        "Training groups must contain positive and negative candidates.",
    )
    group_index["row_end_exclusive"] = group_index["group_size"].cumsum()
    group_index["row_start"] = (
        group_index["row_end_exclusive"] - group_index["group_size"]
    )
    group_sizes = group_index["group_size"].to_numpy(dtype="int32")
    _require(
        int(group_sizes.sum()) == len(ordered),
        "Group sizes must sum to candidate rows.",
    )
    return {
        "X": ordered[columns].copy(),
        "y": ordered["label"].astype("int8").copy(),
        "groups": group_sizes,
        "keys": ordered[ROW_KEYS].copy(),
        "group_index": group_index,
    }
