"""Historical candidate retrieval and feature computation shared across cutoffs.

Core functions are extracted unchanged from notebook 03's validated prototype.
No target labels enter these functions.
"""

from collections import Counter
from collections.abc import Iterable
import pandas as pd
from atelier_ai.temporal import interval_mask

feature_contract = pd.DataFrame(
    [
        ('customer_history_rows', 'customer', '0', 'Observed source purchase rows in customer history'),
        ('customer_distinct_articles', 'customer', '0', 'Distinct purchased article IDs in customer history'),
        ('customer_recency_days', 'customer', 'NaN without observed history', 'Days since last observed customer purchase'),
        ('customer_has_history', 'customer', 'False', 'Any observed customer purchase in the history window'),
        ('article_recent_rows', 'article', '0 if absent from recent purchases', 'Global source purchase rows for this article in the recent window'),
        ('article_recent_rank', 'article', 'NaN if absent from recent purchases', 'Global recent popularity rank; count descending, ID ascending'),
        ('customer_group_rows', 'customer-group', '0', 'Customer historical purchase rows in the candidate product group'),
        ('customer_group_share', 'customer-group', '0', 'Group rows divided by all customer historical rows'),
        ('customer_article_rows', 'customer-article', '0', 'Customer historical purchase rows for this article'),
        ('customer_article_recency_days', 'customer-article', 'NaN without observed purchase', 'Days since this customer last bought this article'),
        ('has_purchased_article', 'customer-article', 'False', 'Any observed purchase of this article by this customer'),
        ('from_history', 'retrieval', 'False', 'Article belongs to the bounded history retrieval proposal'),
        ('from_category', 'retrieval', 'False', 'Article belongs to the bounded dominant-group retrieval proposal'),
        ('from_global', 'retrieval', 'False', 'Article belongs to global top candidate-budget retrieval proposal'),
        ('history_source_rank', 'retrieval', 'NaN outside bounded source', 'Position within the bounded history retrieval proposal'),
        ('category_source_rank', 'retrieval', 'NaN outside bounded source', 'Position within the bounded dominant-group proposal'),
        ('n_retrieval_sources', 'retrieval', '0 for articles outside every source', 'Number of retrieval proposals containing the article'),
    ],
    columns=["feature", "family", "missing_policy", "meaning"],
)
FEATURE_COLUMNS = feature_contract["feature"].tolist()

EXPECTED_MISSING_FEATURES = [
    "customer_recency_days", "article_recent_rank", "customer_article_recency_days",
    "history_source_rank", "category_source_rank",
]

def aggregate_recent_popularity(
    chunks: Iterable[pd.DataFrame],
    *,
    cutoff,
    window_days: int,
) -> pd.DataFrame:
    """Rank articles by global source-row count in [cutoff - window, cutoff)."""
    if window_days <= 0:
        raise ValueError("Popularity window must be positive.")
    cutoff = pd.Timestamp(cutoff)
    start = cutoff - pd.Timedelta(days=window_days)
    if not start < cutoff:
        raise ValueError("Popularity dates must define a valid interval.")

    article_counts = Counter()
    for chunk in chunks:
        dates = pd.to_datetime(chunk["t_dat"], format="%Y-%m-%d", errors="raise")
        recent_articles = chunk.loc[interval_mask(dates, start, cutoff), "article_id"]
        article_counts.update(recent_articles.value_counts().to_dict())

    popularity = (
        pd.Series(article_counts, dtype="int64", name="transaction_rows")
        .rename_axis("article_id")
        .reset_index()
    )
    popularity["article_id"] = popularity["article_id"].astype("string")
    return popularity.sort_values(
        ["transaction_rows", "article_id"], ascending=[False, True]
    ).reset_index(drop=True)


def build_customer_article_history(
    chunks: Iterable[pd.DataFrame],
    customer_ids: pd.Index,
    *,
    cutoff,
    window_days: int,
) -> pd.DataFrame:
    """Aggregate per-customer article recency and row counts strictly before cutoff."""
    if window_days <= 0:
        raise ValueError("History window must be positive.")
    cutoff = pd.Timestamp(cutoff)
    start = cutoff - pd.Timedelta(days=window_days)
    if not start < cutoff:
        raise ValueError("History dates must define a valid interval.")

    parts = []
    for chunk in chunks:
        selected = chunk.loc[
            chunk["customer_id"].isin(customer_ids),
            ["t_dat", "customer_id", "article_id"],
        ].copy()
        if selected.empty:
            continue
        selected["t_dat"] = pd.to_datetime(
            selected["t_dat"], format="%Y-%m-%d", errors="raise"
        )
        selected = selected.loc[interval_mask(selected["t_dat"], start, cutoff)]
        if not selected.empty:
            parts.append(selected.groupby(["customer_id", "article_id"], as_index=False).agg(
                last_purchase_date=("t_dat", "max"),
                history_transaction_rows=("t_dat", "size"),
            ))

    if not parts:
        return pd.DataFrame({
            "customer_id": pd.Series(dtype="string"),
            "article_id": pd.Series(dtype="string"),
            "last_purchase_date": pd.Series(dtype="datetime64[ns]"),
            "history_transaction_rows": pd.Series(dtype="int64"),
            "days_since_last_purchase": pd.Series(dtype="int64"),
        })

    history = pd.concat(parts, ignore_index=True).groupby(
        ["customer_id", "article_id"], as_index=False
    ).agg(
        last_purchase_date=("last_purchase_date", "max"),
        history_transaction_rows=("history_transaction_rows", "sum"),
    )
    history["days_since_last_purchase"] = (cutoff - history["last_purchase_date"]).dt.days
    return history.sort_values(
        ["customer_id", "last_purchase_date", "article_id"],
        ascending=[True, False, True],
    ).reset_index(drop=True)


def attach_product_groups(frame: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """Attach exactly one group per observed article, rejecting unmatched references."""
    result = frame.merge(
        metadata[["article_id", "product_group_name"]],
        on="article_id", how="left", validate="many_to_one", indicator=True,
    )
    if len(result) != len(frame) or result["_merge"].ne("both").any():
        raise ValueError("Product-group join must match every input row without multiplication.")
    if result["product_group_name"].isna().any() or result["product_group_name"].str.strip().eq("").any():
        raise ValueError("Matched product groups must not be missing or blank.")
    return result.drop(columns="_merge")


def build_category_tables(history, popularity, metadata):
    """Return all customer-group affinities, dominant groups and article popularity by group."""
    grouped_history = attach_product_groups(history, metadata)
    category_affinities = grouped_history.groupby(
        ["customer_id", "product_group_name"], as_index=False
    ).agg(
        group_transaction_rows=("history_transaction_rows", "sum"),
        n_history_articles=("article_id", "size"),
    )
    category_affinities["group_share"] = (
        category_affinities["group_transaction_rows"]
        / category_affinities.groupby("customer_id")["group_transaction_rows"].transform("sum")
    )
    preferred_groups = (
        category_affinities.sort_values(
            ["customer_id", "group_transaction_rows", "product_group_name"],
            ascending=[True, False, True],
        )
        .drop_duplicates("customer_id")
        .reset_index(drop=True)
    )
    category_popularity = attach_product_groups(popularity, metadata).sort_values(
        ["product_group_name", "transaction_rows", "article_id"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    return category_affinities, preferred_groups, category_popularity


def combine_candidate_sources(
    history_articles,
    category_articles,
    global_articles,
    *,
    budget: int,
    history_limit: int,
    category_limit: int,
) -> tuple[list, dict]:
    """Union bounded historical sources, then fill globally without target inputs."""
    if budget <= 0 or min(history_limit, category_limit) < 0:
        raise ValueError("Budget must be positive and source limits nonnegative.")
    if history_limit + category_limit > budget:
        raise ValueError("Source limits exceed the candidate budget.")
    global_order = list(dict.fromkeys(global_articles))[:budget]
    if len(global_order) < budget:
        raise ValueError("Global fallback must supply at least budget distinct articles.")
    source_orders = [
        ("history_added", list(dict.fromkeys(history_articles))[:history_limit]),
        ("category_added", list(dict.fromkeys(category_articles))[:category_limit]),
        ("global_added", global_order),
    ]
    chosen, seen, audit = [], set(), {}
    for source_name, articles in source_orders:
        before = len(chosen)
        for article_id in articles:
            if len(chosen) == budget:
                break
            if article_id not in seen:
                chosen.append(article_id)
                seen.add(article_id)
        audit[source_name] = len(chosen) - before
    return chosen, audit


def prepare_feature_tables(
    customer_ids, history, popularity, metadata, *, cutoff, history_days,
    popularity_days, history_limit, category_limit, global_limit,
) -> dict:
    """Prepare historical lookups for one declared prediction cutoff."""
    cutoff = pd.Timestamp(cutoff)
    if history_days <= 0 or popularity_days <= 0 or min(history_limit, category_limit, global_limit) <= 0:
        raise ValueError("Feature windows and retrieval limits must be positive.")
    if not customer_ids.is_unique or history.duplicated(["customer_id", "article_id"]).any():
        raise ValueError("Customer IDs and historical customer–article keys must be unique.")
    if not history["customer_id"].isin(customer_ids).all() or not history["history_transaction_rows"].gt(0).all():
        raise ValueError("History must contain known customers and positive observed counts.")
    if not interval_mask(history["last_purchase_date"], cutoff - pd.Timedelta(days=history_days), cutoff).all():
        raise ValueError("History contains dates outside the declared pre-cutoff interval.")

    affinities, dominant, grouped_popularity = build_category_tables(history, popularity, metadata)
    customer_observed = history.groupby("customer_id", as_index=False).agg(
        customer_history_rows=("history_transaction_rows", "sum"),
        customer_distinct_articles=("article_id", "size"),
        last_purchase_date=("last_purchase_date", "max"),
    )
    customers = pd.DataFrame({"customer_id": customer_ids}).merge(
        customer_observed, on="customer_id", how="left", validate="one_to_one",
    ).merge(
        dominant[["customer_id", "product_group_name"]].rename(columns={"product_group_name": "dominant_product_group"}),
        on="customer_id", how="left", validate="one_to_one",
    )
    customers["customer_recency_days"] = (cutoff - customers["last_purchase_date"]).dt.days
    customers = customers.drop(columns="last_purchase_date")

    recent = popularity.sort_values(["transaction_rows", "article_id"], ascending=[False, True]).reset_index(drop=True)
    recent = recent.rename(columns={"transaction_rows": "article_recent_rows"})
    recent["article_recent_rank"] = range(1, len(recent) + 1)
    grouped_popularity = grouped_popularity.copy()
    grouped_popularity["_group_popularity_rank"] = grouped_popularity.groupby("product_group_name").cumcount() + 1
    observed_ids = pd.concat([history["article_id"], popularity["article_id"]], ignore_index=True).drop_duplicates()
    articles = attach_product_groups(observed_ids.to_frame(name="article_id"), metadata).merge(
        recent, on="article_id", how="left", validate="one_to_one",
    ).merge(
        grouped_popularity[["article_id", "_group_popularity_rank"]],
        on="article_id", how="left", validate="one_to_one",
    )
    groups = affinities[["customer_id", "product_group_name", "group_transaction_rows", "group_share"]].rename(
        columns={"group_transaction_rows": "customer_group_rows", "group_share": "customer_group_share"}
    )
    interactions = history.sort_values(
        ["customer_id", "last_purchase_date", "article_id"], ascending=[True, False, True],
    ).copy()
    interactions["_history_rank"] = interactions.groupby("customer_id").cumcount() + 1
    interactions["history_source_rank"] = interactions["_history_rank"].where(interactions["_history_rank"].le(history_limit))
    interactions["customer_article_recency_days"] = (cutoff - interactions["last_purchase_date"]).dt.days
    interactions = interactions.rename(columns={"history_transaction_rows": "customer_article_rows"})
    interactions = interactions[["customer_id", "article_id", "customer_article_rows", "customer_article_recency_days", "history_source_rank"]]
    return {
        "cutoff": cutoff, "history_days": history_days, "popularity_days": popularity_days,
        "category_limit": category_limit, "global_limit": global_limit,
        "customers": customers, "articles": articles, "groups": groups, "interactions": interactions,
    }


def build_candidate_feature_batch(candidate_pairs: pd.DataFrame, tables: dict) -> pd.DataFrame:
    """Construct historical features without labels, fitted transforms or target inputs."""
    keys = ["customer_id", "article_id", "cutoff"]
    if set(candidate_pairs.columns) != set(keys) or candidate_pairs.empty:
        raise ValueError("A nonempty batch must contain exactly customer_id, article_id and cutoff.")
    if candidate_pairs[keys].isna().any().any() or candidate_pairs.duplicated(keys).any():
        raise ValueError("Prediction keys must be present and unique.")
    if not candidate_pairs["cutoff"].eq(tables["cutoff"]).all():
        raise ValueError("Candidate cutoff differs from historical feature-table context.")
    if not candidate_pairs["customer_id"].isin(tables["customers"]["customer_id"]).all():
        raise ValueError("Candidate batch contains a customer outside the source context.")
    if not candidate_pairs["article_id"].isin(tables["articles"]["article_id"]).all():
        raise ValueError("Candidate article lacks a historical source reference.")

    result = candidate_pairs[keys].reset_index(drop=True).merge(
        tables["customers"], on="customer_id", how="left", validate="many_to_one",
    ).merge(
        tables["articles"], on="article_id", how="left", validate="many_to_one",
    ).merge(
        tables["groups"], on=["customer_id", "product_group_name"], how="left", validate="many_to_one",
    ).merge(
        tables["interactions"], on=["customer_id", "article_id"], how="left", validate="one_to_one",
    )
    for column in [
        "customer_history_rows", "customer_distinct_articles", "article_recent_rows",
        "customer_group_rows", "customer_article_rows",
    ]:
        result[column] = result[column].fillna(0).astype("int64")
    result["customer_group_share"] = result["customer_group_share"].fillna(0).astype("float64")
    result["customer_has_history"] = result["customer_distinct_articles"].gt(0)
    result["has_purchased_article"] = result["customer_article_rows"].gt(0)
    result["from_history"] = result["history_source_rank"].notna()
    result["from_category"] = (
        result["product_group_name"].eq(result["dominant_product_group"]).fillna(False)
        & result["_group_popularity_rank"].le(tables["category_limit"]).fillna(False)
    )
    result["from_global"] = result["article_recent_rank"].le(tables["global_limit"]).fillna(False)
    result["category_source_rank"] = result["_group_popularity_rank"].where(result["from_category"])
    result["n_retrieval_sources"] = result[["from_history", "from_category", "from_global"]].sum(axis=1).astype("int64")
    for column in [
        "customer_recency_days", "article_recent_rank", "customer_article_recency_days",
        "history_source_rank", "category_source_rank",
    ]:
        result[column] = result[column].astype("float64")
    audit_columns = keys + ["product_group_name", "dominant_product_group"]
    if len(result) != len(candidate_pairs):
        raise ValueError("Feature joins changed the candidate-row grain.")
    return result[audit_columns + FEATURE_COLUMNS]


def build_checked_feature_batch(pairs, tables):
    batch = build_candidate_feature_batch(pairs, tables)
    count_columns = [
        "customer_history_rows", "customer_distinct_articles", "article_recent_rows",
        "customer_group_rows", "customer_article_rows",
    ]
    if not batch[count_columns].ge(0).all().all():
        raise ValueError("Historical counts must be nonnegative.")
    if not batch["customer_group_share"].between(0, 1).all():
        raise ValueError("Group shares must be between zero and one.")
    if not batch["n_retrieval_sources"].between(1, 3).all():
        raise ValueError("Every generated candidate must belong to a retrieval source.")
    for column, observed in [
        ("customer_recency_days", batch["customer_has_history"]),
        ("customer_article_recency_days", batch["has_purchased_article"]),
    ]:
        if not batch[column].notna().eq(observed).all():
            raise ValueError(f"Observed-history flag differs from recency availability: {column}")
        if not batch[column].dropna().between(1, tables["history_days"]).all():
            raise ValueError(f"Recency lies outside historical bounds: {column}")
    if not batch["has_purchased_article"].eq(batch["customer_article_rows"].gt(0)).all():
        raise ValueError("Interaction flag differs from historical purchase count.")
    if not batch["article_recent_rank"].notna().eq(batch["article_recent_rows"].gt(0)).all():
        raise ValueError("Recent article rank differs from observed recent purchases.")
    if not batch["customer_group_rows"].le(batch["customer_history_rows"]).all():
        raise ValueError("Group frequency exceeds customer historical frequency.")
    if not batch["customer_article_rows"].le(batch["customer_history_rows"]).all():
        raise ValueError("Interaction frequency exceeds customer historical frequency.")
    return batch


def build_candidate_pools(customer_ids, history, popularity, metadata, *,
                          budget=300, history_limit=100, category_limit=100):
    """Apply notebook 03's historical retrieval policy without target inputs."""
    _, preferred, grouped_popularity = build_category_tables(history, popularity, metadata)
    global_order = popularity.sort_values(
        ["transaction_rows", "article_id"], ascending=[False, True]
    ).head(budget)["article_id"].tolist()
    history_orders = (history.sort_values(
        ["customer_id", "last_purchase_date", "article_id"], ascending=[True, False, True]
    ).groupby("customer_id", sort=False).head(history_limit)
        .groupby("customer_id")["article_id"].agg(list).to_dict())
    category_orders = (grouped_popularity.groupby("product_group_name", sort=False)
                       .head(category_limit).groupby("product_group_name")["article_id"]
                       .agg(list).to_dict())
    dominant = preferred.set_index("customer_id")["product_group_name"].to_dict()
    pools, audit = {}, []
    if not customer_ids.is_unique:
        raise ValueError("Candidate customers must be unique.")
    for customer_id in customer_ids:
        pool, additions = combine_candidate_sources(
            history_orders.get(customer_id, []), category_orders.get(dominant.get(customer_id), []),
            global_order, budget=budget, history_limit=history_limit, category_limit=category_limit,
        )
        if len(pool) != budget or len(set(pool)) != budget:
            raise ValueError("Candidate pool must meet the distinct candidate budget.")
        pools[customer_id] = pool
        audit.append({"customer_id": customer_id, **additions})
    return pools, pd.DataFrame(audit)


def feature_storage_schema(*, include_label=False):
    """Declare the same stable storage types as the validation feature export."""
    result = {"customer_id": "VARCHAR", "article_id": "VARCHAR", "cutoff": "TIMESTAMP",
              "product_group_name": "VARCHAR", "dominant_product_group": "VARCHAR"}
    boolean_features = {"customer_has_history", "has_purchased_article", "from_history",
                        "from_category", "from_global"}
    floating_features = set(EXPECTED_MISSING_FEATURES) | {"customer_group_share"}
    for column in FEATURE_COLUMNS:
        result[column] = ("BOOLEAN" if column in boolean_features else
                          "DOUBLE" if column in floating_features else "BIGINT")
    if include_label:
        result["label"] = "BIGINT"
    return result
