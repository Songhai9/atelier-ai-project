"""Historical isolation, source caps, affinities and complete-reference stopping."""

import duckdb
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from atelier_ai.historical_contexts import build_historical_context
from atelier_ai.recommendation_features import (
    FEATURE_COLUMNS,
    retrieve_candidates,
    prepare_feature_tables,
    build_checked_feature_batch,
    feature_storage_schema,
)
from atelier_ai.workflow import full_reference_ndcg


@pytest.fixture
def fixture():
    cutoff = pd.Timestamp("2020-09-02")
    ids = pd.Index(["u", "cold"], dtype="string")
    transactions = pd.DataFrame(
        [
            ("2020-03-06", "u", "01", 0.1, 1),  # exactly 180 days
            ("2020-09-01", "u", "02", 0.2, 2),
            ("2020-09-01", "u", "02", 0.2, 2),
            ("2020-08-20", "outside", "03", 0.3, 1),  # 28-day fill
            ("2020-09-01", "outside", "04", 0.4, 1),
            ("2020-09-02", "u", "05", 0.5, 2),  # target; unavailable at cutoff
            ("2020-09-03", "outside", "05", 0.5, 2),
        ],
        columns=["t_dat", "customer_id", "article_id", "price", "sales_channel_id"],
    )
    transactions["t_dat"] = pd.to_datetime(transactions["t_dat"])
    metadata = pd.DataFrame(
        {
            "article_id": ["01", "02", "03", "04", "05"],
            "product_group_name": ["A", "A", "B", "B", "C"],
            "product_type_name": ["T1", "T2", "T3", "T2", "T4"],
            "colour_group_name": ["Blue", "Red", "Unknown", "Blue", "Red"],
        }
    )
    return cutoff, ids, transactions, metadata


def context(fixture, *, remove_future=False):
    cutoff, ids, transactions, metadata = fixture
    if remove_future:
        transactions = transactions.loc[transactions["t_dat"].lt(cutoff)]
    with duckdb.connect() as con:
        con.register("transactions", transactions)
        return build_historical_context(con, ids, metadata, cutoff=cutoff)


def test_future_transactions_cannot_change_history_or_global_popularity(fixture):
    first, second = context(fixture), context(fixture, remove_future=True)
    for name in ["history", "popularity"]:
        assert_frame_equal(
            first[name].sort_values(list(first[name])).reset_index(drop=True),
            second[name].sort_values(list(second[name])).reset_index(drop=True),
        )
    assert "05" not in set(first["popularity"]["article_id"])
    assert set(first["history"]["article_id"]) == {"01", "02"}


def test_sources_preserve_caps_and_use_28_day_popularity_for_fill(fixture):
    _, ids, _, _ = fixture
    pools, sources, _, audit = retrieve_candidates(
        ids,
        context(fixture),
        budget=3,
        history_limit=1,
        global_limit=1,
        preferred_types=2,
        per_type=1,
    )
    assert sources["u"]["history"] == ["02"]
    assert sources["u"]["global"] == ["02"]
    assert "03" in pools["cold"]
    assert "03" in sources["cold"]["fallback"]
    assert all(len(values) == len(set(values)) == 3 for values in pools.values())
    assert not any("05" in values for values in pools.values())
    assert audit["history_proposed"].le(1).all()
    assert audit["category_proposed"].le(2).all()


def test_features_preserve_missing_recency_and_exact_affinity_denominators(fixture):
    cutoff, ids, _, _ = fixture
    ctx = context(fixture)
    pools, sources, _, _ = retrieve_candidates(
        ids,
        ctx,
        budget=3,
        history_limit=1,
        global_limit=1,
        preferred_types=2,
        per_type=1,
    )
    tables = prepare_feature_tables(ids, ctx, sources)
    pairs = pd.DataFrame(
        [
            (customer, article, cutoff)
            for customer in ids
            for article in pools[customer]
        ],
        columns=["customer_id", "article_id", "cutoff"],
    )
    result = build_checked_feature_batch(pairs, tables)
    row = result.loc[
        result["customer_id"].eq("u") & result["article_id"].eq("02")
    ].iloc[0]
    assert row["customer_history_rows"] == 3
    assert row["customer_rows_7d"] == 2
    assert row["customer_rows_28d"] == 2
    assert row["customer_type_share"] == pytest.approx(2 / 3)
    assert row["customer_colour_share"] == pytest.approx(2 / 3)
    assert row["customer_article_rows"] == 2
    assert row["customer_article_recency_days"] == 1
    cold = result.loc[result["customer_id"].eq("cold")]
    assert cold["customer_recency_days"].isna().all()
    assert cold["customer_history_rows"].eq(0).all()
    assert list(result.columns) == list(feature_storage_schema())
    assert len(FEATURE_COLUMNS) == 31 and "label" not in FEATURE_COLUMNS
    with pytest.raises(ValueError, match="exactly"):
        build_checked_feature_batch(pairs.assign(label=1), tables)
    with pytest.raises(ValueError, match="cutoff differs"):
        build_checked_feature_batch(
            pairs.assign(cutoff=cutoff + pd.Timedelta(days=1)), tables
        )


def test_stopping_metric_keeps_retrieval_misses_in_ideal_ranking():
    metric = full_reference_ndcg([2, 1, 0], budget=3, k=2)
    name, score, higher = metric(
        np.array([1, 0, 0, 0, 0, 0, 0, 0, 0]), np.array([3, 2, 1, 3, 2, 1, 3, 2, 1])
    )
    assert higher and name == "ndcg_full_reference_at_12"
    assert score == pytest.approx((1 / (1 + 1 / np.log2(3))) / 2)
