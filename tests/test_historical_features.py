"""Historical isolation, source semantics, feature values and split naming."""

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from atelier_ai.historical_contexts import collect_historical_contexts
from atelier_ai.recommendation_features import (
    FEATURE_COLUMNS, build_candidate_pools, build_candidate_feature_batch, build_checked_feature_batch,
    prepare_feature_tables, feature_storage_schema,
)
from atelier_ai.training_targets import label_candidate_pairs, make_training_schedule


@pytest.fixture
def fixture():
    ids = pd.Index(["001", "002"], dtype="string")
    cutoff = pd.Timestamp("2020-08-26")
    columns = ["t_dat", "customer_id", "article_id"]
    chunks = [pd.DataFrame(rows, columns=columns, dtype="string") for rows in [
        [["2020-02-28", "001", "01"], ["2020-08-25", "001", "02"],
         ["2020-08-25", "outside", "03"], ["2020-08-26", "001", "04"]],
        [["2020-08-25", "001", "02"], ["2020-08-27", "outside", "04"]],
    ]]
    metadata = pd.DataFrame({"article_id": ["01", "02", "03", "04"],
                             "product_group_name": ["A", "A", "B", "B"]}, dtype="string")
    return ids, cutoff, chunks, metadata


def test_contexts_exclude_cutoff_then_include_observed_rows_at_later_cutoff(fixture):
    ids, cutoff, chunks, _ = fixture
    contexts = collect_historical_contexts(iter(chunks), ids, [cutoff, cutoff+pd.Timedelta(days=7)])
    first, later = contexts[cutoff], contexts[cutoff+pd.Timedelta(days=7)]
    assert set(first["history"]["article_id"]) == {"01", "02"}
    assert set(first["popularity"]["article_id"]) == {"02", "03"}
    assert set(later["history"]["article_id"]) == {"02", "04"}  # oldest row expires
    assert first["history"].set_index("article_id").loc["01", "days_since_last_purchase"] == 180
    assert first["history"].set_index("article_id").loc["02", "history_transaction_rows"] == 2
    assert first["popularity"].set_index("article_id")["transaction_rows"].to_dict() == {"02": 2, "03": 1}


def test_future_rows_do_not_change_earlier_context_or_input_frames(fixture):
    ids, cutoff, chunks, _ = fixture
    original = [frame.copy(deep=True) for frame in chunks]
    without_future = [frame.loc[frame["t_dat"].lt("2020-08-26")].copy() for frame in chunks]
    first = collect_historical_contexts(chunks, ids, [cutoff])[cutoff]
    second = collect_historical_contexts(without_future, ids, [cutoff])[cutoff]
    for name in ["history", "popularity"]:
        assert_frame_equal(first[name], second[name])
    for frame, saved in zip(chunks, original):
        assert_frame_equal(frame, saved)


def test_features_preserve_cold_history_missingness_and_observed_counts(fixture):
    ids, cutoff, chunks, metadata = fixture
    context = collect_historical_contexts(chunks, ids, [cutoff])[cutoff]
    tables = prepare_feature_tables(ids, context["history"], context["popularity"], metadata,
                                    cutoff=cutoff, history_days=180, popularity_days=7,
                                    history_limit=1, category_limit=1, global_limit=2)
    pairs = pd.DataFrame({"customer_id": pd.Series(["001", "001", "002"], dtype="string"),
                          "article_id": pd.Series(["02", "01", "03"], dtype="string"),
                          "cutoff": [cutoff]*3})
    # Probe an observed old article outside all capped retrieval sources as well.
    result = build_candidate_feature_batch(pairs, tables)
    assert result.loc[0, "customer_article_rows"] == 2
    assert result.loc[0, "customer_article_recency_days"] == 1
    assert result.loc[1, "customer_article_recency_days"] == 180
    assert not result.loc[1, "from_history"]  # outside the capped top-one source
    assert result.loc[1, "has_purchased_article"]
    assert not result.loc[2, "customer_has_history"]
    assert pd.isna(result.loc[2, "customer_recency_days"])
    assert result.loc[2, "customer_history_rows"] == 0
    assert "label" not in result
    assert list(result.columns) == list(feature_storage_schema())
    assert len(FEATURE_COLUMNS) == 17 and "label" not in FEATURE_COLUMNS


def test_candidates_remain_historical_unique_and_cold_customer_uses_global(fixture):
    ids, cutoff, chunks, metadata = fixture
    context = collect_historical_contexts(chunks, ids, [cutoff])[cutoff]
    pools, audit = build_candidate_pools(ids, context["history"], context["popularity"], metadata,
                                         budget=2, history_limit=1, category_limit=1)
    assert pools["001"] == ["02", "03"]
    assert pools["002"] == ["02", "03"]
    assert audit[["history_added", "category_added", "global_added"]].sum(axis=1).eq(2).all()
    assert all(len(pool) == len(set(pool)) == 2 and "04" not in pool for pool in pools.values())
    with pytest.raises(ValueError, match="at least budget"):
        build_candidate_pools(ids, context["history"], context["popularity"], metadata,
                              budget=3, history_limit=1, category_limit=1)


def test_feature_builder_rejects_wrong_cutoff_and_label_inputs(fixture):
    ids, cutoff, chunks, metadata = fixture
    context = collect_historical_contexts(chunks, ids, [cutoff])[cutoff]
    tables = prepare_feature_tables(ids, context["history"], context["popularity"], metadata,
                                    cutoff=cutoff, history_days=180, popularity_days=7,
                                    history_limit=1, category_limit=1, global_limit=2)
    pairs = pd.DataFrame({"customer_id": ["001"], "article_id": ["02"], "cutoff": [cutoff]})
    with pytest.raises(ValueError, match="exactly"):
        build_checked_feature_batch(pairs.assign(label=1), tables)
    with pytest.raises(ValueError, match="cutoff differs"):
        build_checked_feature_batch(pairs.assign(cutoff=cutoff+pd.Timedelta(days=7)), tables)


def test_labelling_adds_only_target_column_after_feature_computation(fixture):
    ids, cutoff, chunks, metadata = fixture
    context = collect_historical_contexts(chunks, ids, [cutoff])[cutoff]
    tables = prepare_feature_tables(ids, context["history"], context["popularity"], metadata,
                                    cutoff=cutoff, history_days=180, popularity_days=7,
                                    history_limit=1, category_limit=1, global_limit=2)
    pairs = pd.DataFrame({"customer_id": pd.Series(["001", "001"], dtype="string"),
                          "article_id": pd.Series(["02", "03"], dtype="string"), "cutoff": [cutoff]*2})
    features = build_checked_feature_batch(pairs, tables)
    relevance = pd.DataFrame({"cutoff": [cutoff, cutoff], "customer_id": pd.Series(["001", "001"], dtype="string"),
                              "article_id": pd.Series(["02", "04"], dtype="string"), "label": [1, 1]})
    schedule = make_training_schedule(validation_cutoff="2020-09-09", n_weeks=2)
    labelled = label_candidate_pairs(features, relevance, schedule=schedule,
                                     customer_ids=ids, validation_cutoff="2020-09-09")
    assert_frame_equal(labelled.drop(columns="label"), features)
    assert labelled["label"].tolist() == [1, 0]
    assert "04" not in labelled["article_id"].tolist()


def test_empty_histories_still_have_typed_contexts(fixture):
    ids, cutoff, _, _ = fixture
    context = collect_historical_contexts([], ids, [cutoff])[cutoff]
    assert context["history"].empty and context["popularity"].empty
    with pytest.raises(ValueError, match="midnight"):
        collect_historical_contexts([], ids, [cutoff+pd.Timedelta(hours=1)])
