"""Cutoff-specific contexts from typed transactions and a fixed metadata snapshot."""

import pandas as pd


def build_historical_context(
    connection, customer_ids, metadata, *, cutoff, history_days=180
):
    """Load bounded cohort history and global 7/28-day article statistics."""
    cutoff = pd.Timestamp(cutoff)
    if pd.isna(cutoff) or cutoff.tzinfo is not None or cutoff != cutoff.normalize():
        raise ValueError("Cutoff must be a timezone-naive midnight date.")
    if history_days != 180:
        raise ValueError("This feature contract requires 180-day history.")
    ids = pd.Index(customer_ids)
    if ids.empty or ids.has_duplicates or ids.isna().any():
        raise ValueError("Unique customers are required.")
    connection.register("_context_customers", pd.DataFrame({"customer_id": ids}))
    start = cutoff - pd.Timedelta(days=history_days)
    history = connection.execute(
        "SELECT t.* FROM transactions t INNER JOIN _context_customers c USING(customer_id) "
        "WHERE t_dat>=? AND t_dat<? ORDER BY customer_id,t_dat,article_id",
        [start, cutoff],
    ).df()
    connection.unregister("_context_customers")
    popularity = connection.execute(
        "SELECT article_id, count(*) FILTER (WHERE t_dat>=?) AS article_recent_rows, "
        "count(*) FILTER (WHERE t_dat>=?) AS article_rows_28d, max(t_dat) AS last_seen "
        "FROM transactions WHERE t_dat>=? AND t_dat<? GROUP BY article_id",
        [cutoff - pd.Timedelta(days=7), cutoff - pd.Timedelta(days=28), start, cutoff],
    ).df()
    popularity["article_recency_days"] = (
        cutoff - popularity.pop("last_seen")
    ).dt.days.astype(float)
    columns = [
        "article_id",
        "product_group_name",
        "product_type_name",
        "colour_group_name",
    ]
    if (
        metadata["article_id"].duplicated().any()
        or metadata[columns].isna().any().any()
    ):
        raise ValueError(
            "Article metadata must have unique complete classification keys."
        )
    for frame in [history, popularity]:
        if not frame["article_id"].isin(metadata["article_id"]).all():
            raise ValueError("Unmatched article metadata.")
    history = history.merge(
        metadata[columns], on="article_id", how="left", validate="many_to_one"
    )
    popularity = popularity.merge(
        metadata[columns], on="article_id", how="left", validate="many_to_one"
    )
    return {"cutoff": cutoff, "history": history, "popularity": popularity}
