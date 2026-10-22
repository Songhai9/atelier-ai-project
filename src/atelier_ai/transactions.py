"""Chunked transaction validity and exact source-pattern repetitions."""

from collections import Counter
from collections.abc import Iterable, Mapping

import pandas as pd

from .data_quality import missing_mask, numeric_diagnostics, quality_table


TRANSACTION_COLUMNS = (
    "t_dat", "customer_id", "article_id", "price", "sales_channel_id",
)


def scan_transactions(chunks: Iterable[pd.DataFrame]) -> dict:
    """Scan all five source fields, preserving exact lexical comparisons.

    Require valid complete ISO dates in nondecreasing order, checked
    within and across chunks. A persistent day counter catches patterns
    split across chunk boundaries. Source frames are not modified.

    Return checks, source rows, finite price range/count, channel counts
    and the exact-pattern multiplicity distribution. No records are dropped.
    """
    checks = Counter()
    channel_counts = Counter()
    multiplicities = Counter()
    current_day_counts = Counter()
    current_date = None
    rows = 0
    finite_price_rows = 0
    price_min = price_max = None
    for chunk in chunks:
        if chunk.empty:
            continue
        rows += len(chunk)
        dates = chunk["t_dat"]
        date_missing = missing_mask(dates)
        date_format = dates.str.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", na=False)
        parsed_dates = pd.to_datetime(dates.mask(date_missing), format="%Y-%m-%d", errors="coerce")
        checks.update({
            "date_missing": int(date_missing.sum()),
            "date_format_mismatch": int((~date_missing & ~date_format).sum()),
            "date_unparseable": int((~date_missing & parsed_dates.isna()).sum()),
        })
        if date_missing.any() or (~date_format).any() or parsed_dates.isna().any():
            raise ValueError("Repetition scanning requires complete valid ISO dates.")
        if not dates.is_monotonic_increasing or (
            current_date is not None and dates.iloc[0] < current_date
        ):
            raise ValueError("Transactions are not ordered by date.")
        prices, finite, price_checks = numeric_diagnostics(chunk["price"])
        checks.update({f"price_{name}": count for name, count in price_checks.items()})
        finite_price_rows += int(finite.sum())
        if finite.any():
            lower, upper = float(prices.loc[finite].min()), float(prices.loc[finite].max())
            price_min = lower if price_min is None else min(price_min, lower)
            price_max = upper if price_max is None else max(price_max, upper)
        channels = chunk["sales_channel_id"]
        channel_missing = missing_mask(channels)
        checks.update({
            "channel_missing": int(channel_missing.sum()),
            "channel_format_mismatch": int((
                ~channel_missing & ~channels.str.fullmatch(r"[0-9]+", na=False)
            ).sum()),
        })
        channel_counts.update(channels.value_counts(dropna=False).to_dict())
        for day, day_rows in chunk.groupby("t_dat", sort=False):
            if day != current_date:
                multiplicities.update(current_day_counts.values())
                current_day_counts.clear()
                current_date = day
            current_day_counts.update(
                day_rows.loc[:, list(TRANSACTION_COLUMNS)]
                .value_counts(sort=False, dropna=False).to_dict()
            )
    if rows == 0:
        raise ValueError("No transaction records were inspected.")
    multiplicities.update(current_day_counts.values())
    if sum(size * count for size, count in multiplicities.items()) != rows:
        raise ValueError("Repetition counts do not reconcile.")
    if sum(channel_counts.values()) != rows:
        raise ValueError("Channel counts do not reconcile.")
    return {
        "checks": dict(checks), "rows": rows,
        "finite_price_rows": finite_price_rows,
        "price_min": price_min, "price_max": price_max,
        "channel_counts": dict(channel_counts),
        "multiplicity_counts": dict(multiplicities),
    }


def summarize_repetitions(
    multiplicities: Mapping[int, int], total_rows: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separate repeated patterns, affected rows and additional occurrences."""
    if total_rows <= 0 or not multiplicities:
        raise ValueError("Repetition summary requires nonempty counts.")
    if any(size <= 0 or count <= 0 for size, count in multiplicities.items()):
        raise ValueError("Multiplicities and pattern counts must be positive.")
    if sum(size * count for size, count in multiplicities.items()) != total_rows:
        raise ValueError("Repetition counts do not reconcile.")
    repeated_patterns = sum(count for size, count in multiplicities.items() if size > 1)
    repeated_rows = sum(size * count for size, count in multiplicities.items() if size > 1)
    additional = sum((size - 1) * count for size, count in multiplicities.items())
    summary = pd.DataFrame([
        {"metric": name, "value": value}
        for name, value in (
            ("source_transaction_rows", total_rows),
            ("distinct_exact_patterns", sum(multiplicities.values())),
            ("repeated_patterns", repeated_patterns),
            ("rows_in_repeated_patterns", repeated_rows),
            ("additional_repeated_rows", additional),
            ("maximum_pattern_multiplicity", max(multiplicities)),
        )
    ])
    rates = quality_table({
        "rows_in_repeated_patterns": repeated_rows,
        "additional_repeated_rows": additional,
    }, total_rows)
    return summary, rates
