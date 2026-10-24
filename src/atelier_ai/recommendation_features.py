"""Historical retrieval and numeric feature definitions for fashion ranking."""

import numpy as np
import pandas as pd

FEATURE_DEFINITIONS = [
    (
        "customer_history_rows",
        "customer",
        "0",
        "Purchase rows in the preceding 180 days",
    ),
    (
        "customer_distinct_articles",
        "customer",
        "0",
        "Distinct articles in the preceding 180 days",
    ),
    (
        "customer_recency_days",
        "customer",
        "NaN",
        "Days since the last observed purchase",
    ),
    (
        "customer_has_history",
        "customer",
        "False",
        "Any purchase in the preceding 180 days",
    ),
    ("customer_rows_7d", "customer", "0", "Purchase rows in the preceding 7 days"),
    ("customer_rows_28d", "customer", "0", "Purchase rows in the preceding 28 days"),
    ("customer_rows_90d", "customer", "0", "Purchase rows in the preceding 90 days"),
    (
        "customer_distinct_product_types",
        "customer",
        "0",
        "Distinct product types in 180-day history",
    ),
    (
        "customer_distinct_colours",
        "customer",
        "0",
        "Distinct colour groups in 180-day history",
    ),
    (
        "article_recent_rows",
        "article",
        "0",
        "Global purchase rows in the preceding 7 days",
    ),
    ("article_recent_rank", "article", "NaN", "Global 7-day rank; ties use article ID"),
    (
        "article_rows_28d",
        "article",
        "0",
        "Global purchase rows in the preceding 28 days",
    ),
    (
        "article_popularity_trend",
        "article",
        "0",
        "7-day rows divided by one quarter of 28-day rows",
    ),
    (
        "article_recency_days",
        "article",
        "NaN",
        "Days since last global purchase within 180 days",
    ),
    (
        "customer_group_rows",
        "affinity",
        "0",
        "Customer rows for the candidate broad product group",
    ),
    (
        "customer_group_share",
        "affinity",
        "0",
        "Broad-group rows divided by customer history rows",
    ),
    (
        "customer_type_share",
        "affinity",
        "0",
        "Product-type rows divided by customer history rows",
    ),
    (
        "customer_colour_share",
        "affinity",
        "0",
        "Colour-group rows divided by customer history rows",
    ),
    (
        "customer_article_rows",
        "interaction",
        "0",
        "Customer purchase rows for the candidate article",
    ),
    (
        "customer_article_recency_days",
        "interaction",
        "NaN",
        "Days since the customer last bought this article",
    ),
    (
        "has_purchased_article",
        "interaction",
        "False",
        "Candidate article observed in customer history",
    ),
    (
        "prediction_weekday",
        "context",
        "defined",
        "Weekday of prediction cutoff, Monday=0",
    ),
    ("prediction_iso_week", "context", "defined", "ISO week of prediction cutoff"),
    (
        "customer_channel_2_share",
        "context",
        "0",
        "History-row share with source channel code 2",
    ),
    (
        "from_history",
        "retrieval",
        "False",
        "In the latest 30 distinct personal articles",
    ),
    (
        "from_category",
        "retrieval",
        "False",
        "In a top-50 proposal from a preferred product type",
    ),
    ("from_global", "retrieval", "False", "In the global 7-day top-100 proposal"),
    ("from_fallback_28d", "retrieval", "False", "Added by the 28-day popularity fill"),
    (
        "history_source_rank",
        "retrieval",
        "NaN",
        "Rank in the bounded personal-history proposal",
    ),
    (
        "category_source_rank",
        "retrieval",
        "NaN",
        "Rank in the combined preferred-type proposal",
    ),
    ("n_retrieval_sources", "retrieval", "0", "Number of proposal or fill memberships"),
]
feature_contract = pd.DataFrame(
    FEATURE_DEFINITIONS, columns=["feature", "family", "missing_policy", "meaning"]
)
FEATURE_COLUMNS = feature_contract["feature"].tolist()
EXPECTED_MISSING_FEATURES = feature_contract.loc[
    feature_contract["missing_policy"].eq("NaN"), "feature"
].tolist()
BOOLEAN_FEATURES = {
    "customer_has_history",
    "has_purchased_article",
    "from_history",
    "from_category",
    "from_global",
    "from_fallback_28d",
}
FLOAT_FEATURES = set(EXPECTED_MISSING_FEATURES) | {
    "article_popularity_trend",
    "customer_group_share",
    "customer_type_share",
    "customer_colour_share",
    "customer_channel_2_share",
}


def feature_storage_schema(*, include_label=False):
    result = {
        "customer_id": "VARCHAR",
        "article_id": "VARCHAR",
        "cutoff": "TIMESTAMP",
        "product_group_name": "VARCHAR",
        "dominant_product_group": "VARCHAR",
    }
    for column in FEATURE_COLUMNS:
        result[column] = (
            "BOOLEAN"
            if column in BOOLEAN_FEATURES
            else "DOUBLE"
            if column in FLOAT_FEATURES
            else "BIGINT"
        )
    if include_label:
        result["label"] = "BIGINT"
    return result


def ordered_union(*sources, limit):
    """Keep the first occurrence of each article, respecting source priority."""
    result, seen = [], set()
    for source in sources:
        for article in source:
            if article not in seen:
                result.append(article)
                seen.add(article)
                if len(result) == limit:
                    return result
    return result


def retrieve_candidates(
    customer_ids,
    context,
    *,
    budget=300,
    history_limit=30,
    global_limit=100,
    preferred_types=3,
    per_type=50,
):
    """Propose history, global 7-day, preferred types, then fill on 28 days."""
    ids = pd.Index(customer_ids)
    if ids.empty or ids.has_duplicates or ids.isna().any():
        raise ValueError("Candidate customers must be nonempty, unique and present.")
    if min(budget, history_limit, global_limit, preferred_types, per_type) <= 0:
        raise ValueError("Candidate limits must be positive.")
    history, popularity = context["history"], context["popularity"]
    if not history.empty and (
        history["t_dat"].ge(context["cutoff"]).any()
        or history["t_dat"].lt(context["cutoff"] - pd.Timedelta(days=180)).any()
    ):
        raise ValueError("History must lie strictly before its cutoff within 180 days.")
    last = history.groupby(["customer_id", "article_id"], as_index=False)["t_dat"].max()
    last = last.sort_values(
        ["customer_id", "t_dat", "article_id"], ascending=[True, False, True]
    )
    personal = last.groupby("customer_id", sort=False)["article_id"].agg(list).to_dict()
    preference = (
        history.groupby(["customer_id", "product_type_name"])
        .size()
        .rename("rows")
        .reset_index()
        .sort_values(
            ["customer_id", "rows", "product_type_name"], ascending=[True, False, True]
        )
    )
    preferred = (
        preference.groupby("customer_id", sort=False)
        .head(preferred_types)
        .groupby("customer_id", sort=False)["product_type_name"]
        .agg(list)
        .to_dict()
    )
    recent = popularity.loc[popularity["article_recent_rows"].gt(0)].sort_values(
        ["article_recent_rows", "article_id"], ascending=[False, True]
    )
    by_type = (
        recent.groupby("product_type_name", sort=False)
        .head(per_type)
        .groupby("product_type_name", sort=False)["article_id"]
        .agg(list)
        .to_dict()
    )
    global_order = recent["article_id"].head(global_limit).tolist()
    fallback = (
        popularity.loc[popularity["article_rows_28d"].gt(0)]
        .sort_values(["article_rows_28d", "article_id"], ascending=[False, True])[
            "article_id"
        ]
        .tolist()
    )
    if len(fallback) < budget:
        raise ValueError(
            "28-day fallback must contain at least budget distinct articles."
        )
    pools, memberships, baselines, audits = {}, {}, {}, []
    for customer in ids:
        h = personal.get(customer, [])[:history_limit]
        c = ordered_union(
            *(by_type.get(t, []) for t in preferred.get(customer, [])),
            limit=preferred_types * per_type,
        )
        before_fill = ordered_union(h, global_order, c, limit=budget)
        pool = ordered_union(before_fill, fallback, limit=budget)
        fill = [article for article in pool if article not in set(before_fill)]
        pools[customer] = pool
        memberships[customer] = {
            "history": h,
            "category": c,
            "global": global_order,
            "fallback": fill,
        }
        baselines[customer] = {
            "recent_popularity_7d": ordered_union(global_order, fallback, limit=12),
            "preferred_product_types": ordered_union(
                c, global_order, fallback, limit=12
            ),
            "recent_purchases_180d": ordered_union(
                personal.get(customer, []), global_order, fallback, limit=12
            ),
        }
        audits.append(
            {
                "customer_id": customer,
                "history_proposed": len(h),
                "global_proposed": len(global_order),
                "category_proposed": len(c),
                "fallback_added": len(fill),
                "candidate_rows": len(pool),
            }
        )
    return pools, memberships, baselines, pd.DataFrame(audits)


def prepare_feature_tables(customer_ids, context, memberships):
    """Aggregate historical customer, article and affinity lookups without labels."""
    history, popularity, cutoff = (
        context["history"],
        context["popularity"],
        context["cutoff"],
    )
    customers = pd.DataFrame({"customer_id": pd.Index(customer_ids)})
    observed = history.groupby("customer_id").agg(
        customer_history_rows=("article_id", "size"),
        customer_distinct_articles=("article_id", "nunique"),
        last_purchase=("t_dat", "max"),
        customer_distinct_product_types=("product_type_name", "nunique"),
        customer_distinct_colours=("colour_group_name", "nunique"),
    )
    observed["customer_recency_days"] = (cutoff - observed.pop("last_purchase")).dt.days
    for days in (7, 28, 90):
        observed[f"customer_rows_{days}d"] = (
            history.loc[history["t_dat"].ge(cutoff - pd.Timedelta(days=days))]
            .groupby("customer_id")
            .size()
        )
    observed["customer_channel_2_share"] = (
        history["sales_channel_id"].eq(2).groupby(history["customer_id"]).mean()
    )
    customers = customers.merge(
        observed.reset_index(), how="left", on="customer_id", validate="one_to_one"
    )
    customers["customer_has_history"] = (
        customers["customer_history_rows"].fillna(0).gt(0)
    )
    affinities = {}
    for column, prefix in [
        ("product_group_name", "group"),
        ("product_type_name", "type"),
        ("colour_group_name", "colour"),
    ]:
        affinity = (
            history.groupby(["customer_id", column]).size().rename("rows").reset_index()
        )
        affinity[f"customer_{prefix}_share"] = affinity["rows"] / affinity.groupby(
            "customer_id"
        )["rows"].transform("sum")
        if prefix == "group":
            dominant = affinity.sort_values(
                ["customer_id", "rows", column], ascending=[True, False, True]
            ).drop_duplicates("customer_id")
            customers = customers.merge(
                dominant[["customer_id", column]].rename(
                    columns={column: "dominant_product_group"}
                ),
                how="left",
                on="customer_id",
                validate="one_to_one",
            )
            affinity = affinity.rename(columns={"rows": "customer_group_rows"})
        else:
            affinity = affinity.drop(columns="rows")
        affinities[prefix] = affinity
    interactions = history.groupby(["customer_id", "article_id"], as_index=False).agg(
        customer_article_rows=("t_dat", "size"), last_purchase=("t_dat", "max")
    )
    interactions["customer_article_recency_days"] = (
        cutoff - interactions.pop("last_purchase")
    ).dt.days
    articles = popularity.copy()
    articles["article_popularity_trend"] = np.where(
        articles["article_rows_28d"].gt(0),
        articles["article_recent_rows"]
        * 4
        / articles["article_rows_28d"].clip(lower=1),
        0,
    )
    recent = articles.loc[articles["article_recent_rows"].gt(0)].sort_values(
        ["article_recent_rows", "article_id"], ascending=[False, True]
    )
    ranks = dict(zip(recent["article_id"], range(1, len(recent) + 1)))
    articles["article_recent_rank"] = articles["article_id"].map(ranks)
    return {
        "cutoff": cutoff,
        "history_days": 180,
        "customers": customers,
        "articles": articles,
        "affinities": affinities,
        "interactions": interactions,
        "memberships": memberships,
    }


def build_candidate_feature_batch(candidate_pairs, tables):
    keys = ["customer_id", "article_id", "cutoff"]
    if set(candidate_pairs.columns) != set(keys) or candidate_pairs.empty:
        raise ValueError(
            "A nonempty batch must contain exactly the three prediction keys."
        )
    if (
        candidate_pairs[keys].isna().any().any()
        or candidate_pairs.duplicated(keys).any()
    ):
        raise ValueError("Candidate keys must be present and unique.")
    if not candidate_pairs["cutoff"].eq(tables["cutoff"]).all():
        raise ValueError("Candidate cutoff differs from historical context.")
    if (
        not candidate_pairs["customer_id"]
        .isin(tables["customers"]["customer_id"])
        .all()
    ):
        raise ValueError("Unknown customer in candidate batch.")
    if not candidate_pairs["article_id"].isin(tables["articles"]["article_id"]).all():
        raise ValueError(
            "Candidate article has not been observed in historical context."
        )
    result = (
        candidate_pairs[keys]
        .reset_index(drop=True)
        .merge(
            tables["customers"], on="customer_id", how="left", validate="many_to_one"
        )
    )
    result = result.merge(
        tables["articles"], on="article_id", how="left", validate="many_to_one"
    )
    for prefix, column in [
        ("group", "product_group_name"),
        ("type", "product_type_name"),
        ("colour", "colour_group_name"),
    ]:
        result = result.merge(
            tables["affinities"][prefix],
            on=["customer_id", column],
            how="left",
            validate="many_to_one",
        )
    result = result.merge(
        tables["interactions"],
        on=["customer_id", "article_id"],
        how="left",
        validate="one_to_one",
    )
    membership_rows = []
    for customer, pairs in candidate_pairs.groupby("customer_id", sort=False):
        sources = tables["memberships"][customer]
        hr = {a: i + 1 for i, a in enumerate(sources["history"])}
        cr = {a: i + 1 for i, a in enumerate(sources["category"])}
        source_sets = {k: set(v) for k, v in sources.items()}
        for article in pairs["article_id"]:
            membership_rows.append(
                {
                    "customer_id": customer,
                    "article_id": article,
                    "from_history": article in source_sets["history"],
                    "from_category": article in source_sets["category"],
                    "from_global": article in source_sets["global"],
                    "from_fallback_28d": article in source_sets["fallback"],
                    "history_source_rank": hr.get(article, np.nan),
                    "category_source_rank": cr.get(article, np.nan),
                }
            )
    result = result.merge(
        pd.DataFrame(membership_rows),
        on=["customer_id", "article_id"],
        how="left",
        validate="one_to_one",
    )
    result["has_purchased_article"] = result["customer_article_rows"].fillna(0).gt(0)
    result["prediction_weekday"] = tables["cutoff"].dayofweek
    result["prediction_iso_week"] = tables["cutoff"].isocalendar().week
    result["n_retrieval_sources"] = (
        result[["from_history", "from_category", "from_global", "from_fallback_28d"]]
        .fillna(False)
        .sum(axis=1)
    )
    for column in FEATURE_COLUMNS:
        if column in BOOLEAN_FEATURES:
            result[column] = result[column].fillna(False).astype(bool)
        elif column in FLOAT_FEATURES:
            result[column] = (
                result[column].astype(float)
                if column in EXPECTED_MISSING_FEATURES
                else result[column].fillna(0).astype(float)
            )
        else:
            result[column] = result[column].fillna(0).astype("int64")
    if len(result) != len(candidate_pairs):
        raise ValueError("Feature joins changed candidate grain.")
    return result[list(feature_storage_schema())]


def build_checked_feature_batch(pairs, tables):
    result = build_candidate_feature_batch(pairs, tables)
    required = [c for c in FEATURE_COLUMNS if c not in EXPECTED_MISSING_FEATURES]
    if (
        result[required].isna().any().any()
        or result[FEATURE_COLUMNS].isin([np.inf, -np.inf]).any().any()
    ):
        raise ValueError("Features contain invalid missing or infinite values.")
    if not result["n_retrieval_sources"].between(1, 4).all():
        raise ValueError(
            "Every candidate needs at least one historical retrieval source."
        )
    for column in [
        "customer_group_share",
        "customer_type_share",
        "customer_colour_share",
        "customer_channel_2_share",
    ]:
        if not result[column].between(0, 1).all():
            raise ValueError("Historical shares must lie in [0, 1].")
    for column in [
        "customer_recency_days",
        "customer_article_recency_days",
        "article_recency_days",
    ]:
        if not result[column].dropna().between(1, 180).all():
            raise ValueError("Recency exceeds the historical window.")
    return result
