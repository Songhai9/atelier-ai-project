"""Historical training windows and observed seven-day relevance targets."""

from collections.abc import Iterable

import pandas as pd

from atelier_ai.temporal import interval_mask


TARGET_KEYS = ["cutoff", "customer_id", "article_id"]
SCHEDULE_COLUMNS = ["cutoff", "history_start", "popularity_start", "target_end"]


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _customer_index(customer_ids):
    result = pd.Index(customer_ids, dtype="string", name="customer_id")
    _require(len(result) > 0 and result.is_unique and not result.isna().any(),
             "Customer IDs must be nonempty, unique and present.")
    _require((result.str.strip() != "").all(), "Customer IDs must not be blank.")
    return result


def validate_training_schedule(schedule, *, validation_cutoff, horizon_days=7,
                               history_days=180, popularity_days=7):
    """Require sorted disjoint daily target windows ending before validation."""
    _require(all(isinstance(value, int) and not isinstance(value, bool) and value > 0
                 for value in [horizon_days, history_days, popularity_days]),
             "Window lengths must be positive integers.")
    _require(popularity_days <= history_days, "Popularity window exceeds history.")
    _require(list(schedule.columns) == SCHEDULE_COLUMNS and not schedule.empty,
             "A nonempty schedule with the declared columns is required.")
    boundary = pd.Timestamp(validation_cutoff)
    _require(not pd.isna(boundary) and boundary.tzinfo is None
             and boundary == boundary.normalize(), "Validation must be a midnight date.")
    for column in SCHEDULE_COLUMNS:
        values = schedule[column]
        _require(pd.api.types.is_datetime64_any_dtype(values) and not values.isna().any(),
                 "Schedule columns must contain valid datetime values.")
        _require(values.dt.tz is None and values.eq(values.dt.normalize()).all(),
                 "Schedule must contain timezone-naive midnight dates.")
    _require(schedule["cutoff"].is_unique and schedule["cutoff"].is_monotonic_increasing,
             "Training cutoffs must be unique and chronological.")
    for left, right, days in [
        ("history_start", "cutoff", history_days),
        ("popularity_start", "cutoff", popularity_days),
        ("cutoff", "target_end", horizon_days),
    ]:
        _require((schedule[right] - schedule[left]).eq(pd.Timedelta(days=days)).all(),
                 "A schedule window differs from its declared length.")
    _require(schedule["target_end"].le(boundary).all(),
             "Training targets must end no later than validation starts.")
    ends = schedule["target_end"].iloc[:-1].reset_index(drop=True)
    starts = schedule["cutoff"].iloc[1:].reset_index(drop=True)
    _require(ends.le(starts).all(), "Training target windows must not overlap.")


def make_training_schedule(*, validation_cutoff, n_weeks=4, horizon_days=7,
                           history_days=180, popularity_days=7):
    """Choose the last n_weeks complete target windows before validation."""
    _require(isinstance(n_weeks, int) and not isinstance(n_weeks, bool) and n_weeks > 0,
             "n_weeks must be a positive integer.")
    for value in [horizon_days, history_days, popularity_days]:
        _require(isinstance(value, int) and not isinstance(value, bool) and value > 0,
                 "Window lengths must be positive integers.")
    boundary = pd.Timestamp(validation_cutoff)
    cutoffs = [boundary - pd.Timedelta(days=horizon_days * week)
               for week in range(n_weeks, 0, -1)]
    schedule = pd.DataFrame({
        "cutoff": cutoffs,
        "history_start": [cutoff - pd.Timedelta(days=history_days) for cutoff in cutoffs],
        "popularity_start": [cutoff - pd.Timedelta(days=popularity_days) for cutoff in cutoffs],
        "target_end": [cutoff + pd.Timedelta(days=horizon_days) for cutoff in cutoffs],
    })
    validate_training_schedule(schedule, validation_cutoff=boundary, horizon_days=horizon_days,
                               history_days=history_days, popularity_days=popularity_days)
    return schedule


def collect_training_transactions(chunks: Iterable[pd.DataFrame], customer_ids, schedule, *,
                                  validation_cutoff, horizon_days=7, history_days=180,
                                  popularity_days=7):
    """Collect only cohort target rows in one source pass, retaining repetitions."""
    ids = _customer_index(customer_ids)
    validate_training_schedule(schedule, validation_cutoff=validation_cutoff,
                               horizon_days=horizon_days, history_days=history_days,
                               popularity_days=popularity_days)
    selected_parts = []
    columns = ["t_dat", "customer_id", "article_id"]
    for chunk in chunks:
        selected = chunk.loc[chunk["customer_id"].isin(ids), columns].copy()
        if selected.empty:
            continue
        selected["t_dat"] = pd.to_datetime(selected["t_dat"], format="%Y-%m-%d", errors="raise")
        for window in schedule.itertuples(index=False):
            part = selected.loc[interval_mask(selected["t_dat"], window.cutoff, window.target_end)].copy()
            if part.empty:
                continue
            _require(part["article_id"].notna().all()
                     and part["article_id"].astype("string").str.strip().ne("").all(),
                     "Target article IDs must be present and nonblank.")
            part["cutoff"] = window.cutoff
            selected_parts.append(part[["cutoff", *columns]])
    if not selected_parts:
        return pd.DataFrame({
            "cutoff": pd.Series(dtype="datetime64[ns]"),
            "t_dat": pd.Series(dtype="datetime64[ns]"),
            "customer_id": pd.Series(dtype="string"),
            "article_id": pd.Series(dtype="string"),
        })
    return (pd.concat(selected_parts, ignore_index=True)
            .sort_values(["cutoff", "t_dat", "customer_id", "article_id"])
            .reset_index(drop=True))


def build_training_targets(transactions, customer_ids, schedule, *, validation_cutoff,
                           horizon_days=7, history_days=180, popularity_days=7):
    """Return distinct positive relevance and the full customer-week population."""
    ids = _customer_index(customer_ids)
    validate_training_schedule(schedule, validation_cutoff=validation_cutoff,
                               horizon_days=horizon_days, history_days=history_days,
                               popularity_days=popularity_days)
    _require(transactions["customer_id"].isin(ids).all(), "Target customer outside cohort.")
    _require(transactions["cutoff"].isin(schedule["cutoff"]).all(), "Unknown target cutoff.")
    _require(transactions["article_id"].notna().all()
             and transactions["article_id"].astype("string").str.strip().ne("").all(),
             "Target article IDs must be present and nonblank.")
    for window in schedule.itertuples(index=False):
        dates = transactions.loc[transactions["cutoff"].eq(window.cutoff), "t_dat"]
        _require(interval_mask(dates, window.cutoff, window.target_end).all(),
                 "A transaction lies outside its assigned target week.")
    relevance = (transactions[TARGET_KEYS].drop_duplicates()
                 .sort_values(TARGET_KEYS).reset_index(drop=True))
    relevance["label"] = pd.Series(1, index=relevance.index, dtype="int8")
    counts = relevance.groupby(["cutoff", "customer_id"]).size().rename("n_target_articles")
    population = pd.MultiIndex.from_product(
        [schedule["cutoff"], ids], names=["cutoff", "customer_id"]
    ).to_frame(index=False)
    population = population.merge(counts.reset_index(), on=["cutoff", "customer_id"],
                                  how="left", validate="one_to_one", sort=False)
    population["n_target_articles"] = population["n_target_articles"].fillna(0).astype("int64")
    population["has_target_purchase"] = population["n_target_articles"].gt(0)
    return relevance, population


def label_candidate_pairs(candidate_pairs, relevance, *, schedule, customer_ids,
                          validation_cutoff, horizon_days=7, history_days=180,
                          popularity_days=7):
    """Attach observed-purchase labels without inserting missing positive targets."""
    ids = _customer_index(customer_ids)
    validate_training_schedule(schedule, validation_cutoff=validation_cutoff,
                               horizon_days=horizon_days, history_days=history_days,
                               popularity_days=popularity_days)
    _require("label" not in candidate_pairs, "Candidate rows are already labelled.")
    _require(not candidate_pairs[TARGET_KEYS].isna().any().any()
             and not candidate_pairs.duplicated(TARGET_KEYS).any(),
             "Candidate keys must be present and unique.")
    for frame in [candidate_pairs, relevance]:
        _require(frame["cutoff"].isin(schedule["cutoff"]).all(), "Unknown target cutoff.")
        _require(frame["customer_id"].isin(ids).all(), "Customer outside cohort.")
    _require(not relevance[TARGET_KEYS].isna().any().any()
             and not relevance.duplicated(TARGET_KEYS).any() and relevance["label"].eq(1).all(),
             "Relevance must contain distinct positive keys.")
    labelled = candidate_pairs.merge(relevance[TARGET_KEYS + ["label"]], on=TARGET_KEYS,
                                     how="left", validate="one_to_one", sort=False)
    labelled["label"] = labelled["label"].fillna(0).astype("int8")
    _require(len(labelled) == len(candidate_pairs)
             and labelled[TARGET_KEYS].reset_index(drop=True).equals(
                 candidate_pairs[TARGET_KEYS].reset_index(drop=True)),
             "Labelling changed candidate keys or their order.")
    return labelled
