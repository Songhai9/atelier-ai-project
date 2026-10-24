"""Temporal isolation, complete populations and observed-purchase label semantics."""

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from atelier_ai.training_targets import (
    build_training_targets,
    collect_training_transactions,
    label_candidate_pairs,
    make_training_schedule,
    validate_training_schedule,
)


BOUNDARY = pd.Timestamp("2020-09-09")


@pytest.fixture
def schedule():
    return make_training_schedule(validation_cutoff=BOUNDARY, n_weeks=2)


@pytest.fixture
def chunks():
    columns = ["t_dat", "customer_id", "article_id"]
    return [pd.DataFrame(rows, columns=columns, dtype="string") for rows in [
        [["2020-08-25", "001", "past"], ["2020-08-26", "001", "01"],
         ["2020-09-01", "001", "01"], ["2020-09-02", "001", "02"],
         ["2020-09-09", "001", "validation"], ["2020-09-16", "001", "test"]],
        [["2020-08-27", "001", "01"], ["2020-09-08", "001", "01"],
         ["2020-08-26", "outside", "99"]],
    ]]


def test_schedule_uses_four_complete_weeks_before_validation():
    result = make_training_schedule(validation_cutoff=BOUNDARY)
    assert result["cutoff"].dt.strftime("%Y-%m-%d").tolist() == [
        "2020-08-12", "2020-08-19", "2020-08-26", "2020-09-02"]
    assert result["target_end"].iloc[-1] == BOUNDARY
    assert (result["cutoff"] - result["history_start"]).eq(pd.Timedelta(days=180)).all()
    with pytest.raises(ValueError, match="positive integer"):
        make_training_schedule(validation_cutoff=BOUNDARY, n_weeks=0)


def test_schedule_rejects_validation_labels_and_overlapping_windows(schedule):
    future = schedule.copy()
    future.iloc[-1] += pd.Timedelta(days=7)
    with pytest.raises(ValueError, match="validation starts"):
        validate_training_schedule(future, validation_cutoff=BOUNDARY)
    overlap = schedule.copy()
    overlap.iloc[-1] -= pd.Timedelta(days=1)
    with pytest.raises(ValueError, match="overlap"):
        validate_training_schedule(overlap, validation_cutoff=BOUNDARY)


def test_single_pass_boundaries_repetitions_and_input_immutability(schedule, chunks):
    originals = [frame.copy(deep=True) for frame in chunks]
    result = collect_training_transactions(iter(chunks), ["001", "002"], schedule,
                                           validation_cutoff=BOUNDARY)
    assert len(result) == 5
    assert set(result["article_id"]) == {"01", "02"}
    assert result["cutoff"].value_counts().sort_index().tolist() == [3, 2]
    assert result["t_dat"].min() == pd.Timestamp("2020-08-26")
    assert result["t_dat"].max() == pd.Timestamp("2020-09-08")
    for actual, original in zip(chunks, originals):
        assert_frame_equal(actual, original)


def test_targets_deduplicate_per_week_and_keep_nonbuyers(schedule, chunks):
    transactions = collect_training_transactions(chunks, ["001", "002"], schedule,
                                                 validation_cutoff=BOUNDARY)
    relevance, population = build_training_targets(transactions, ["001", "002"], schedule,
                                                   validation_cutoff=BOUNDARY)
    assert len(relevance) == 3 and relevance["label"].eq(1).all()
    # Article 01 remains relevant in both distinct weeks.
    assert relevance["article_id"].eq("01").sum() == 2
    assert len(population) == 4
    assert population["n_target_articles"].sum() == 3
    assert population.loc[population["customer_id"].eq("002"), "n_target_articles"].eq(0).all()
    assert not population.loc[population["customer_id"].eq("002"), "has_target_purchase"].any()


def test_empty_targets_preserve_all_customer_weeks(schedule):
    transactions = collect_training_transactions([], ["001", "002"], schedule,
                                                 validation_cutoff=BOUNDARY)
    relevance, population = build_training_targets(transactions, ["001", "002"], schedule,
                                                   validation_cutoff=BOUNDARY)
    assert relevance.empty and len(population) == 4
    assert population["n_target_articles"].eq(0).all()


def test_target_builder_rejects_wrong_week_assignment(schedule, chunks):
    transactions = collect_training_transactions(chunks, ["001", "002"], schedule,
                                                 validation_cutoff=BOUNDARY)
    transactions.loc[0, "t_dat"] = BOUNDARY
    with pytest.raises(ValueError, match="outside its assigned"):
        build_training_targets(transactions, ["001", "002"], schedule,
                               validation_cutoff=BOUNDARY)


def test_labels_join_on_week_without_injecting_missing_positives(schedule, chunks):
    transactions = collect_training_transactions(chunks, ["001", "002"], schedule,
                                                 validation_cutoff=BOUNDARY)
    relevance, _ = build_training_targets(transactions, ["001", "002"], schedule,
                                          validation_cutoff=BOUNDARY)
    candidates = pd.DataFrame({
        "cutoff": [schedule["cutoff"].iloc[0]] * 3 + [schedule["cutoff"].iloc[1]] * 2,
        "customer_id": pd.Series(["001", "001", "002", "001", "001"], dtype="string"),
        "article_id": pd.Series(["01", "02", "01", "01", "99"], dtype="string"),
    })
    labelled = label_candidate_pairs(candidates, relevance, schedule=schedule,
                                     customer_ids=["001", "002"], validation_cutoff=BOUNDARY)
    assert labelled["label"].tolist() == [1, 0, 0, 1, 0]
    assert len(labelled) == len(candidates)
    # The second week's purchased article 02 is absent from candidates and stays absent.
    assert not ((labelled["cutoff"] == schedule["cutoff"].iloc[1])
                & labelled["article_id"].eq("02")).any()


def test_labelling_rejects_duplicate_keys_and_unknown_cutoffs(schedule):
    candidates = pd.DataFrame({"cutoff": [schedule["cutoff"].iloc[0]] * 2,
                               "customer_id": ["001"] * 2, "article_id": ["01"] * 2})
    relevance = candidates.iloc[:1].copy().assign(label=1)
    with pytest.raises(ValueError, match="unique"):
        label_candidate_pairs(candidates, relevance, schedule=schedule,
                              customer_ids=["001"], validation_cutoff=BOUNDARY)
    candidates = candidates.iloc[:1].copy()
    candidates["cutoff"] = BOUNDARY
    with pytest.raises(ValueError, match="Unknown target cutoff"):
        label_candidate_pairs(candidates, relevance, schedule=schedule,
                              customer_ids=["001"], validation_cutoff=BOUNDARY)


def test_empty_relevance_labels_existing_candidates_zero(schedule):
    transactions = collect_training_transactions([], ["001"], schedule,
                                                 validation_cutoff=BOUNDARY)
    relevance, _ = build_training_targets(transactions, ["001"], schedule,
                                          validation_cutoff=BOUNDARY)
    candidates = pd.DataFrame({"cutoff": [schedule["cutoff"].iloc[0]],
                               "customer_id": pd.Series(["001"], dtype="string"),
                               "article_id": pd.Series(["01"], dtype="string")})
    labelled = label_candidate_pairs(candidates, relevance, schedule=schedule,
                                     customer_ids=["001"], validation_cutoff=BOUNDARY)
    assert labelled["label"].tolist() == [0]


def test_duplicate_cohort_ids_and_missing_article_ids_are_rejected(schedule, chunks):
    with pytest.raises(ValueError, match="unique"):
        collect_training_transactions(chunks, ["001", "001"], schedule,
                                      validation_cutoff=BOUNDARY)
    invalid = chunks[0].copy()
    invalid.loc[1, "article_id"] = " "
    with pytest.raises(ValueError, match="nonblank"):
        collect_training_transactions([invalid], ["001"], schedule,
                                      validation_cutoff=BOUNDARY)
