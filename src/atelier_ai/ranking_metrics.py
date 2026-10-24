"""Ranking metrics evaluated against complete observed target sets."""

import pandas as pd

def score_recommendation_list(
    relevant_articles,
    recommended_articles,
    *,
    k: int,
) -> dict:
    """Score a distinct top-k ranking against a nonempty, unfiltered target set."""
    if k <= 0:
        raise ValueError("k must be positive.")

    relevant = set(relevant_articles)
    if not relevant:
        raise ValueError("Target-based metrics require at least one relevant article.")

    predictions = list(recommended_articles)[:k]
    if len(predictions) != k or len(set(predictions)) != k:
        raise ValueError("Predictions must contain at least k distinct top-k articles.")

    n_hits = 0
    precision_sum = 0.0
    for rank, article_id in enumerate(predictions, start=1):
        if article_id in relevant:
            n_hits += 1
            precision_sum += n_hits / rank

    return {
        "n_target_articles": len(relevant),
        "n_hits": n_hits,
        "ap_at_k": precision_sum / min(k, len(relevant)),
        "recall_at_k": n_hits / len(relevant),
    }

def evaluate_recommendations(
    recommendation_lists,
    relevant_articles,
    customer_ids,
    *,
    k=12,
):
    """Evaluate one cutoff using all recipients and complete buyer targets."""
    ids = pd.Index(customer_ids)
    if ids.empty or ids.has_duplicates or ids.isna().any():
        raise ValueError("Recipient IDs must be nonempty, unique and present.")
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError("k must be a positive integer.")
    if set(recommendation_lists) != set(ids):
        raise ValueError("Recommendations must cover exactly all recipients.")

    reference = {
        customer: set(articles)
        for customer, articles in relevant_articles.items()
    }
    if (
        not reference
        or not set(reference).issubset(set(ids))
        or any(not articles for articles in reference.values())
    ):
        raise ValueError("Reference must contain nonempty target sets for cohort buyers.")

    top_lists = {}
    for customer in ids:
        predictions = list(recommendation_lists[customer])[:k]
        if len(predictions) != k or len(set(predictions)) != k:
            raise ValueError("Every recipient needs k distinct recommendations.")
        top_lists[customer] = predictions

    customer_scores = pd.DataFrame([
        {
            "customer_id": customer,
            **score_recommendation_list(
                reference[customer], top_lists[customer], k=k
            ),
        }
        for customer in ids
        if customer in reference
    ])

    excluded_customers = len(ids) - len(customer_scores)
    target_pairs = int(customer_scores["n_target_articles"].sum())
    retrieved_pairs = int(customer_scores["n_hits"].sum())

    return {
        "customer_scores": customer_scores,
        "summary": {
            "recommendation_customers": len(ids),
            "evaluated_customers": len(customer_scores),
            "excluded_customers": excluded_customers,
            "excluded_share_pct": 100 * excluded_customers / len(ids),
            "target_pairs": target_pairs,
            "retrieved_target_pairs": retrieved_pairs,
            "map_at_k": float(customer_scores["ap_at_k"].mean()),
            "recall_at_k": float(customer_scores["recall_at_k"].mean()),
            "micro_recall_at_k": retrieved_pairs / target_pairs,
        },
    }