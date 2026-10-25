"""Inspect and verify prepared Parquet artifacts."""

from hashlib import file_digest
import json
from pathlib import Path

import duckdb


TABLE_NAMES = ("articles", "customers", "transactions")

TRANSACTION_TYPES = {
    "t_dat": "DATE",
    "customer_id": "VARCHAR",
    "article_id": "VARCHAR",
    "price": "DOUBLE",
    "sales_channel_id": "SMALLINT",
}


def file_identity(path):
    """Return the size and SHA-256 of a required file."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing prepared file: {path}")

    with path.open("rb") as stream:
        digest = file_digest(stream, "sha256").hexdigest()

    return {"size_bytes": path.stat().st_size, "sha256": digest}


def inspect_prepared_files(
    directory,
    expected_rows,
    *,
    expected_artifacts=None,
):
    """Check file identities, row counts and storage types."""
    directory = Path(directory)
    records = {}

    with duckdb.connect(
        config={"threads": 2, "memory_limit": "512MB"}
    ) as connection:
        for name in TABLE_NAMES:
            path = directory / f"{name}.parquet"
            identity = file_identity(path)

            if expected_artifacts is not None:
                expected = expected_artifacts[name]
                if identity != {
                    key: expected[key]
                    for key in ("size_bytes", "sha256")
                }:
                    raise ValueError(f"Prepared checksum differs: {name}")

            description = connection.execute(
                "DESCRIBE SELECT * FROM read_parquet(?)",
                [str(path)],
            ).fetchall()
            types = {row[0]: row[1] for row in description}

            if name == "transactions":
                valid_types = types == TRANSACTION_TYPES
            else:
                identifier = (
                    "article_id" if name == "articles" else "customer_id"
                )
                valid_types = (
                    types.get(identifier) == "VARCHAR"
                    and all(value == "VARCHAR" for value in types.values())
                )

            if not valid_types:
                raise ValueError(f"Prepared types differ: {name}")

            rows = connection.execute(
                "SELECT count(*) FROM read_parquet(?)",
                [str(path)],
            ).fetchone()[0]

            if rows != expected_rows[name]:
                raise ValueError(f"Prepared row count differs: {name}")

            records[name] = {
                "file": path.name,
                "rows": rows,
                "schema": [
                    {"name": column, "type": dtype}
                    for column, dtype in types.items()
                ],
                **identity,
            }

    return records


def verify_prepared_sources(directory, *, expected_recipe=None):
    """Verify an existing prepared dataset without modifying it."""
    directory = Path(directory)
    manifest = json.loads(
        (directory / "manifest.json").read_text(encoding="utf-8")
    )

    if (
        manifest.get("schema_version") != 2
        or manifest.get("status") != "complete"
        or manifest.get("stage") != "prepare_sources"
    ):
        raise ValueError("A complete version-2 prepared manifest is required.")

    if expected_recipe is not None:
        if any(
            manifest.get(key) != value
            for key, value in expected_recipe.items()
        ):
            raise ValueError("Prepared recipe differs; use a new campaign.")

    if (
        set(manifest["rows"]) != set(TABLE_NAMES)
        or set(manifest["artifacts"]) != set(TABLE_NAMES)
    ):
        raise ValueError("The prepared manifest must cover all three tables.")

    observed = inspect_prepared_files(
        directory,
        manifest["rows"],
        expected_artifacts=manifest["artifacts"],
    )
    if observed != manifest["artifacts"]:
        raise ValueError("Prepared artifact metadata differs.")

    return manifest