"""Numerical diagnostics and aggregate data-quality tables."""

from collections.abc import Mapping

import pandas as pd


def numeric_diagnostics(
    raw_values: pd.Series,
) -> tuple[pd.Series, pd.Series, dict[str, int]]:
    """Return numeric values, a finite-value mask and diagnostic counts.

    Missing values include null, empty and whitespace-only fields.
    Nonempty parsing failures are counted separately from source
    missingness. Negative and zero counts include finite values only.

    The source Series and its index are preserved.
    """
    missing = raw_values.fillna("").str.strip().eq("")

    numeric = pd.to_numeric(
        raw_values.mask(missing),
        errors="coerce",
    ).astype("Float64")

    finite = (
        numeric.notna()
        & ~numeric.isin([float("inf"), float("-inf")])
    )

    counts = {
        "missing": int(missing.sum()),
        "unparseable": int(
            (~missing & numeric.isna()).sum()
        ),
        "nonfinite": int(
            (numeric.notna() & ~finite).sum()
        ),
        "negative": int(
            (finite & numeric.lt(0)).sum()
        ),
        "zero": int(
            (finite & numeric.eq(0)).sum()
        ),
    }

    return numeric, finite, counts


def quality_table(
    counts: Mapping[str, int],
    total_rows: int,
) -> pd.DataFrame:
    """Build diagnostic counts and percentages of source rows.

    Checks may overlap. Each percentage uses the source row count,
    rather than the sum of diagnostic counts.

    Raises:
        ValueError: If total_rows is not positive.
    """
    if total_rows <= 0:
        raise ValueError(
            "Quality rates require a positive source row count."
        )

    return pd.DataFrame([
        {
            "check": name,
            "affected_rows": count,
            "pct_source_rows": 100 * count / total_rows,
        }
        for name, count in counts.items()
    ])