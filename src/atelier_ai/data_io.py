"""Source CSV readers preserving lexical values and missing-value policy."""

from collections.abc import Sequence
from pathlib import Path

import pandas as pd


def read_source_csv(
    path: str | Path,
    *,
    columns: Sequence[str] | None = None,
    nrows: int | None = None,
) -> pd.DataFrame:
    """Read a source table or preview as strings, preserving textual markers."""
    return pd.read_csv(
        path,
        usecols=None if columns is None else list(columns),
        nrows=nrows,
        dtype="string",
        keep_default_na=False,
        encoding="utf-8",
    )


def read_source_chunks(
    path: str | Path,
    *,
    columns: Sequence[str] | None = None,
    chunk_rows: int = 100_000,
):
    """Return a chunk reader to consume inside a ``with`` statement.

    Chunks use the same string and missing-value policy as read_source_csv.
    The caller owns the context manager and closes the source file.
    """
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive.")
    return pd.read_csv(
        path,
        usecols=None if columns is None else list(columns),
        chunksize=chunk_rows,
        dtype="string",
        keep_default_na=False,
        encoding="utf-8",
    )
