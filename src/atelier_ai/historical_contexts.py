"""One source pass for historical customer histories and global popularity."""

from collections import Counter

import pandas as pd

from atelier_ai.recommendation_features import build_customer_article_history
from atelier_ai.temporal import interval_mask


def collect_historical_contexts(chunks, customer_ids, cutoffs, *, history_days=180,
                                popularity_days=7):
    """Aggregate cutoff-specific histories and popularity without target labels.

    Keep only cohort history rows over the union of historical windows, plus
    small article-count dictionaries per cutoff. Global popularity includes
    customers outside the cohort. Repetitions contribute to observed counts.
    """
    customer_ids = pd.Index(customer_ids)
    cutoffs = pd.DatetimeIndex(cutoffs)
    if (len(customer_ids) == 0 or not customer_ids.is_unique or customer_ids.isna().any()
            or len(cutoffs) == 0 or not cutoffs.is_unique or cutoffs.isna().any()
            or cutoffs.tz is not None or not cutoffs.equals(cutoffs.normalize())):
        raise ValueError("Unique customers and timezone-naive midnight cutoffs are required.")
    if not 0 < popularity_days <= history_days:
        raise ValueError("Historical windows must be positive and nested.")
    history_parts = []
    popularity_counts = {cutoff: Counter() for cutoff in cutoffs}
    history_start = cutoffs.min() - pd.Timedelta(days=history_days)
    for chunk in chunks:
        dates = pd.to_datetime(chunk["t_dat"], format="%Y-%m-%d", errors="raise")
        cohort_history = chunk.loc[
            chunk["customer_id"].isin(customer_ids)
            & interval_mask(dates, history_start, cutoffs.max()),
            ["t_dat", "customer_id", "article_id"],
        ].copy()
        if not cohort_history.empty:
            history_parts.append(cohort_history)
        for cutoff in cutoffs:
            recent = chunk.loc[interval_mask(
                dates, cutoff-pd.Timedelta(days=popularity_days), cutoff
            ), "article_id"]
            popularity_counts[cutoff].update(recent.value_counts().to_dict())
    if history_parts:
        retained_history = pd.concat(history_parts, ignore_index=True)
    else:
        retained_history = pd.DataFrame({column: pd.Series(dtype="string")
                                        for column in ["t_dat", "customer_id", "article_id"]})
    contexts = {}
    for cutoff in cutoffs:
        history = build_customer_article_history(
            [retained_history], customer_ids, cutoff=cutoff, window_days=history_days,
        )
        popularity = (pd.Series(popularity_counts[cutoff], dtype="int64", name="transaction_rows")
                      .rename_axis("article_id").reset_index())
        popularity["article_id"] = popularity["article_id"].astype("string")
        popularity = popularity.sort_values(
            ["transaction_rows", "article_id"], ascending=[False, True]
        ).reset_index(drop=True)
        contexts[cutoff] = {"history": history, "popularity": popularity}
    return contexts
