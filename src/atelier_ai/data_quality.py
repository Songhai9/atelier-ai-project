"""Source-quality diagnostics without source mutation or automatic cleaning."""

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence

import pandas as pd


def missing_mask(values: pd.Series) -> pd.Series:
    """Identify null, empty and whitespace-only values, preserving the index."""
    return values.fillna("").str.strip().eq("")


def numeric_diagnostics(
    raw_values: pd.Series,
) -> tuple[pd.Series, pd.Series, dict[str, int]]:
    """Return numeric values, a finite-value mask and diagnostic counts.

    Nonempty parsing failures are distinct from source missingness.
    Negative and zero counts include finite values only.
    The source Series and its index are preserved.
    """
    missing = missing_mask(raw_values)
    numeric = pd.to_numeric(
        raw_values.mask(missing), errors="coerce"
    ).astype("Float64")
    finite = numeric.notna() & ~numeric.isin([float("inf"), float("-inf")])
    counts = {
        "missing": int(missing.sum()),
        "unparseable": int((~missing & numeric.isna()).sum()),
        "nonfinite": int((numeric.notna() & ~finite).sum()),
        "negative": int((finite & numeric.lt(0)).sum()),
        "zero": int((finite & numeric.eq(0)).sum()),
    }
    return numeric, finite, counts


def quality_table(counts: Mapping[str, int], total_rows: int) -> pd.DataFrame:
    """Build counts and source-row percentages; checks may overlap."""
    if total_rows <= 0:
        raise ValueError("Quality rates require a positive source row count.")
    return pd.DataFrame([
        {"check": name, "affected_rows": count,
         "pct_source_rows": 100 * count / total_rows}
        for name, count in counts.items()
    ])


def profile_entity_keys(
    frame: pd.DataFrame, key_column: str, dataset_name: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Profile a complete entity-key column, not independent chunks."""
    values = frame[key_column]
    total_rows = len(values)
    if total_rows == 0:
        raise ValueError(f"Entity table is empty: {dataset_name}")
    missing = missing_mask(values)
    present = values.loc[~missing]
    counts = {
        "missing_keys": int(missing.sum()),
        "duplicate_key_rows": int(present.duplicated(keep=False).sum()),
        "surrounding_whitespace": int(
            values.ne(values.str.strip()).fillna(False).sum()
        ),
    }
    profile = pd.DataFrame([{
        "dataset": dataset_name, "key_column": key_column,
        "rows": total_rows, "distinct_nonmissing_keys": int(present.nunique()),
        "memory_mib": frame.memory_usage(index=True, deep=True).sum() / 1024**2,
    }])
    checks = pd.DataFrame([
        {"dataset": dataset_name, "check": name, "affected_rows": count,
         "affected_pct": 100 * count / total_rows, "passed": count == 0}
        for name, count in counts.items()
    ])
    return profile, checks


def identifier_format_profile(values: pd.Series, pattern: str) -> dict:
    """Report exact pattern mismatches among nonmissing identifiers."""
    present = values.loc[~missing_mask(values)]
    lengths = present.str.len()
    matches = present.str.fullmatch(pattern, na=False)
    return {
        "nonmissing_rows": len(present),
        "min_length": lengths.min(), "max_length": lengths.max(),
        "format_mismatch_rows": int((~matches).sum()),
    }


def profile_references(
    chunks: Iterable[pd.DataFrame], reference_indexes: Mapping[str, pd.Index]
) -> pd.DataFrame:
    """Classify exact references as missing, unknown or matched."""
    if not reference_indexes:
        raise ValueError("At least one reference index is required.")
    for name, index in reference_indexes.items():
        if not index.is_unique:
            raise ValueError(f"Reference index must be unique: {name}")
    counts = {
        name: Counter({"missing_rows": 0, "unknown_rows": 0, "matched_rows": 0})
        for name in reference_indexes
    }
    total_rows = 0
    for chunk in chunks:
        total_rows += len(chunk)
        for name, index in reference_indexes.items():
            values = chunk[name]
            missing = missing_mask(values)
            positions = index.get_indexer(values)
            counts[name].update({
                "missing_rows": int(missing.sum()),
                "unknown_rows": int((~missing & (positions < 0)).sum()),
                "matched_rows": int((~missing & (positions >= 0)).sum()),
            })
    if total_rows == 0:
        raise ValueError("No transaction records were read.")
    return pd.DataFrame([
        {"key_column": name, "rows_examined": total_rows, **dict(values),
         "matched_pct": 100 * values["matched_rows"] / total_rows,
         "passed": values["missing_rows"] == 0 and values["unknown_rows"] == 0}
        for name, values in counts.items()
    ])


def profile_completeness(
    chunks: Iterable[pd.DataFrame], dataset_name: str
) -> pd.DataFrame:
    """Count missing values across chunks from a single source table."""
    total_rows = 0
    counts = Counter()
    source_columns = None
    for chunk in chunks:
        if source_columns is None:
            source_columns = tuple(chunk.columns)
        elif tuple(chunk.columns) != source_columns:
            raise ValueError("Source columns differ between chunks.")
        total_rows += len(chunk)
        for column in chunk.columns:
            counts[column] += int(missing_mask(chunk[column]).sum())
    if total_rows == 0:
        raise ValueError(f"Source table is empty: {dataset_name}")
    return pd.DataFrame([
        {"dataset": dataset_name, "column": column, "rows_examined": total_rows,
         "missing_rows": count, "missing_pct": 100 * count / total_rows}
        for column, count in counts.items()
    ])


def profile_categories(
    chunks: Iterable[pd.DataFrame],
    columns: Sequence[str],
    placeholder_candidates: Iterable[str],
) -> tuple[pd.DataFrame, int]:
    """Count exact categories across chunks; normalization flags only.

    Return a detailed profile and the source row count. Every selected
    column uses the same source-row denominator. No raw values are merged.
    """
    if not columns:
        raise ValueError("At least one category column is required.")
    candidates = frozenset(value.strip().casefold() for value in placeholder_candidates)
    counts = {column: Counter() for column in columns}
    total_rows = 0
    for chunk in chunks:
        total_rows += len(chunk)
        for column in columns:
            counts[column].update(chunk[column].value_counts(dropna=False).to_dict())
    if total_rows == 0:
        raise ValueError("No category records were inspected.")
    rows = []
    for column, values in counts.items():
        if sum(values.values()) != total_rows:
            raise ValueError(f"Category counts do not reconcile for {column}.")
        for raw_value, count in values.items():
            missing = pd.isna(raw_value) or raw_value.strip() == ""
            rows.append({
                "column": column, "raw_value": raw_value,
                "value_display": repr(raw_value), "rows": count,
                "pct_rows": 100 * count / total_rows,
                "is_missing": missing,
                "placeholder_candidate": (
                    not missing and raw_value.strip().casefold() in candidates
                ),
                "surrounding_whitespace": (
                    False if pd.isna(raw_value) else raw_value != raw_value.strip()
                ),
            })
    profile = pd.DataFrame(rows).sort_values(
        ["column", "rows", "value_display"], ascending=[True, False, True]
    ).reset_index(drop=True)
    return profile, total_rows


def summarize_categories(profile: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Summarize a canonical category profile in the requested field order."""
    rows = []
    for column in columns:
        values = profile.loc[profile["column"].eq(column)]
        total_rows = int(values["rows"].sum())
        if total_rows <= 0:
            raise ValueError(f"No category counts for {column}.")
        placeholders = int(values.loc[values["placeholder_candidate"], "rows"].sum())
        rows.append({
            "column": column,
            "distinct_nonmissing_values": int((~values["is_missing"]).sum()),
            "missing_rows": int(values.loc[values["is_missing"], "rows"].sum()),
            "placeholder_rows": placeholders,
            "placeholder_pct": 100 * placeholders / total_rows,
            "whitespace_rows": int(
                values.loc[values["surrounding_whitespace"], "rows"].sum()
            ),
        })
    return pd.DataFrame(rows)


def profile_code_labels(
    articles: pd.DataFrame, pairs: Mapping[str, str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separate code-to-label conflicts from nonunique labels."""
    summary = []
    shared = []
    for code, label in pairs.items():
        missing = missing_mask(articles[code]) | missing_mask(articles[label])
        links = articles.loc[~missing, [code, label]].drop_duplicates()
        labels_per_code = links.groupby(code)[label].nunique()
        codes_per_label = links.groupby(label)[code].nunique()
        conflicts = labels_per_code.index[labels_per_code.gt(1)]
        shared_labels = codes_per_label.index[codes_per_label.gt(1)]
        summary.append({
            "code_column": code, "label_column": label,
            "missing_pair_rows": int(missing.sum()),
            "codes_with_multiple_labels": len(conflicts),
            "affected_article_rows": int(articles[code].isin(conflicts).sum()),
            "labels_with_multiple_codes": len(shared_labels),
        })
        for _, link in links.loc[links[label].isin(shared_labels)].iterrows():
            shared.append({"code_column": code, "label": link[label], "code": link[code]})
    return pd.DataFrame(summary), pd.DataFrame(
        shared, columns=["code_column", "label", "code"]
    )


def profile_product_grouping(articles: pd.DataFrame) -> pd.DataFrame:
    """Measure article/product granularity without inferring variant semantics."""
    if articles.empty:
        raise ValueError("Article table is empty.")
    per_article = articles.groupby("article_id")["product_code"].nunique()
    per_product = articles.groupby("product_code")["article_id"].nunique()
    return pd.DataFrame([{
        "article_rows": len(articles),
        "distinct_articles": articles["article_id"].nunique(),
        "distinct_product_codes": articles["product_code"].nunique(),
        "articles_with_multiple_product_codes": int(per_article.gt(1).sum()),
        "product_codes_with_multiple_articles": int(per_product.gt(1).sum()),
        "maximum_articles_per_product_code": int(per_product.max()),
    }])
