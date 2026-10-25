"""Batch feature export with typed Parquet round trips and provenance checks."""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import duckdb
import pandas as pd


def file_sha256(path: Path) -> str:
    """Hash a file without loading it all into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _quoted(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _connection(scratch: str):
    return duckdb.connect(config={
        "threads": 2, "memory_limit": "512MB", "temp_directory": scratch,
    })


def _stored_schema(connection, path: Path) -> list[dict]:
    rows = connection.execute(
        "DESCRIBE SELECT * FROM read_parquet(?)", [str(path)]
    ).fetchall()
    return [{"name": row[0], "type": row[1]} for row in rows]


def verify_feature_export(dataset_directory, *, expected_fingerprint=None) -> dict:
    """Verify checksums, stored types, row counts, customers and missingness."""
    directory = Path(dataset_directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    _check(manifest["status"] == "complete", "Feature dataset is not complete.")
    if expected_fingerprint is not None:
        _check(manifest["fingerprint"] == expected_fingerprint, "Dataset identity differs.")
    paths = []
    artifacts = manifest["parts"] + manifest["support_files"]
    for artifact in artifacts:
        name = artifact["file"]
        _check(Path(name).name == name, "Artifact names must be plain filenames.")
        path = directory / name
        _check(path.is_file(), f"Missing exported artifact: {name}")
        _check(path.stat().st_size == artifact["size_bytes"], f"Artifact size differs: {name}")
        _check(file_sha256(path) == artifact["sha256"], f"Artifact checksum differs: {name}")
    expected_files = {part["file"] for part in manifest["parts"]}
    _check({path.name for path in directory.glob("part-*.parquet")} == expected_files,
           "Parquet partition list differs from the manifest.")
    cohort = pd.read_csv(directory / "cohort.csv", dtype="string", keep_default_na=False)
    _check(cohort["customer_id"].is_unique, "Exported cohort IDs must be unique.")
    with TemporaryDirectory(prefix="atelier-feature-verify-") as scratch:
        with _connection(scratch) as connection:
            for part in manifest["parts"]:
                path = directory / part["file"]
                _check(_stored_schema(connection, path) == manifest["schema"],
                       f"Stored schema differs: {path.name}")
                count = connection.execute(
                    "SELECT count(*) FROM read_parquet(?)", [str(path)]
                ).fetchall()[0][0]
                _check(count == part["rows"], f"Partition row count differs: {path.name}")
                paths.append(str(path))
            customers = connection.execute(
                "SELECT customer_id, count(*) AS row_count FROM read_parquet(?) "
                "GROUP BY customer_id", [paths],
            ).df()
            _check(set(customers["customer_id"]) == set(cohort["customer_id"]),
                   "Stored customer population differs from the exported cohort.")
            _check(customers["row_count"].eq(manifest["candidates_per_customer"]).all(),
                   "Stored candidate counts differ between customers.")
            rows = int(customers["row_count"].sum())
            _check(rows == manifest["rows"], "Dataset row total differs.")
            expressions = ", ".join(
                f"sum(CASE WHEN {_quoted(name)} IS NULL THEN 1 ELSE 0 END)"
                for name in manifest["feature_columns"]
            )
            missing = connection.execute(
                f"SELECT {expressions} FROM read_parquet(?)", [paths]
            ).fetchall()[0]
    missing_counts = dict(zip(manifest["feature_columns"], map(int, missing)))
    _check(missing_counts == manifest["missing_counts"], "Stored missing-value counts differ.")
    return {
        "rows": rows, "customers": len(customers), "parts": len(paths),
        "parquet_bytes": sum(part["size_bytes"] for part in manifest["parts"]),
        "missing_counts": missing_counts,
    }


def export_feature_batches(
    customer_ids, candidate_pools, feature_tables, build_batch, *, output_base,
    feature_columns, nullable_features, sql_schema, provenance, notebook_path=None,
    source_path=None,
    candidates_per_customer, batch_customers=500, dataset_split="validation",
) -> dict:
    """Export disjoint customer batches, verify contents, then publish the directory.

    A matching complete export is verified and reused without invoking build_batch.
    Generated staging files are removed on failure; existing datasets are never replaced.
    """
    _check(dataset_split in {"training", "validation", "test"}, "Unknown dataset split.")
    customer_ids = pd.Index(customer_ids)
    _check(len(customer_ids) > 0 and customer_ids.is_unique and not customer_ids.isna().any(),
           "A nonempty unique customer population is required.")
    _check(batch_customers > 0 and candidates_per_customer > 0,
           "Batch and candidate sizes must be positive.")
    _check(set(customer_ids) == set(candidate_pools), "Candidate population differs.")
    _check(set(feature_columns).issubset(sql_schema), "Feature schema is incomplete.")
    _check(set(nullable_features).issubset(feature_columns), "Unknown nullable feature.")
    _check((notebook_path is None) != (source_path is None),
           "Provide exactly one of notebook_path or source_path.")
    source = Path(notebook_path if notebook_path is not None else source_path)
    _check(source.is_file(), "Save the source file before exporting features.")
    snapshot_name = (
        "pipeline_notebook.ipynb" if notebook_path is not None
        else "pipeline_source" + source.suffix
    )
    allowed_types = {"VARCHAR", "TIMESTAMP", "BIGINT", "DOUBLE", "BOOLEAN"}
    _check(set(sql_schema.values()).issubset(allowed_types), "Unsupported SQL storage type.")
    candidate_digest = hashlib.sha256()
    for customer_id in customer_ids:
        pool = candidate_pools[customer_id]
        _check(len(pool) == candidates_per_customer and len(set(pool)) == len(pool),
               "Each customer must have the declared number of distinct candidates.")
        candidate_digest.update((_canonical_json([str(customer_id), pool]) + "\n").encode())
    schema = [{"name": name, "type": kind} for name, kind in sql_schema.items()]
    identity = {
        "schema_version": 1, "dataset_split": dataset_split,
        "cutoff": pd.Timestamp(feature_tables["cutoff"]).isoformat(),
        "customer_order": list(map(str, customer_ids)),
        "candidate_sha256": candidate_digest.hexdigest(),
        "candidates_per_customer": candidates_per_customer, "batch_customers": batch_customers,
        "feature_columns": list(feature_columns), "nullable_features": list(nullable_features),
        "schema": schema, "provenance": provenance,
    }
    if source_path is not None:
        identity["source_snapshot_sha256"] = file_sha256(source)
    fingerprint = hashlib.sha256(_canonical_json(identity).encode()).hexdigest()
    base = Path(output_base)
    base.mkdir(parents=True, exist_ok=True)
    name = f"{dataset_split}_{pd.Timestamp(feature_tables['cutoff']):%Y-%m-%d}_{fingerprint[:12]}"
    destination = base / name
    if destination.exists():
        checks = verify_feature_export(destination, expected_fingerprint=fingerprint)
        return {"directory": destination, "manifest": json.loads((destination / "manifest.json").read_text()),
                "checks": checks, "reused": True}

    staging = base / f".{name}.partial-{uuid4().hex}"
    staging.mkdir()
    parts, support_files = [], []
    missing_counts = dict.fromkeys(feature_columns, 0)
    projection = ", ".join(
        f"CAST({_quoted(column)} AS {kind}) AS {_quoted(column)}"
        for column, kind in sql_schema.items()
    )
    keys = ["customer_id", "article_id", "cutoff"]
    try:
        with TemporaryDirectory(prefix="atelier-feature-write-") as scratch:
            with _connection(scratch) as connection:
                for part_number, offset in enumerate(range(0, len(customer_ids), batch_customers)):
                    batch_ids = customer_ids[offset:offset + batch_customers]
                    start = time.perf_counter()
                    pairs = pd.DataFrame(
                        [(customer_id, article_id, feature_tables["cutoff"])
                         for customer_id in batch_ids for article_id in candidate_pools[customer_id]],
                        columns=keys,
                    )
                    batch = build_batch(pairs, feature_tables)
                    _check(list(batch.columns) == list(sql_schema), "Batch columns differ from storage schema.")
                    _check(len(batch) == len(pairs) and batch[keys].eq(pairs).all().all(),
                           "Feature builder changed prediction keys or row order.")
                    _check(all(pd.api.types.is_numeric_dtype(batch[column]) for column in feature_columns),
                           "Model features must be numeric or boolean.")
                    required = [column for column in feature_columns if column not in nullable_features]
                    _check(not batch[required].isna().any().any(), "Unexpected missing feature value.")
                    _check(not batch[feature_columns].isin([float("inf"), -float("inf")]).any().any(),
                           "Infinite feature values cannot be exported.")
                    build_seconds = time.perf_counter() - start
                    start = time.perf_counter()
                    path = staging / f"part-{part_number:05d}.parquet"
                    connection.register("_expected_batch", batch)
                    expected = f"SELECT {projection} FROM _expected_batch"
                    connection.sql(expected).write_parquet(str(path), compression="zstd")
                    _check(_stored_schema(connection, path) == schema, "Parquet types differ.")
                    differences = connection.execute(
                        f"SELECT count(*) FROM (({expected} EXCEPT ALL SELECT * FROM read_parquet(?)) "
                        f"UNION ALL (SELECT * FROM read_parquet(?) EXCEPT ALL {expected}))",
                        [str(path), str(path)],
                    ).fetchall()[0][0]
                    _check(differences == 0, "Parquet round trip changed feature rows or nulls.")
                    connection.unregister("_expected_batch")
                    for column in feature_columns:
                        missing_counts[column] += int(batch[column].isna().sum())
                    parts.append({
                        "file": path.name, "rows": len(batch), "customers": len(batch_ids),
                        "cohort_offset": offset, "size_bytes": path.stat().st_size,
                        "sha256": file_sha256(path), "build_seconds": build_seconds,
                        "write_roundtrip_seconds": time.perf_counter() - start,
                        "dataframe_deep_mb": float(batch.memory_usage(deep=True).sum() / 1024 ** 2),
                    })
                    if (part_number + 1) % 5 == 0 or offset + len(batch_ids) == len(customer_ids):
                        print(f"Verified feature partitions: {part_number + 1}; customers: {offset + len(batch_ids):,}")
                    del batch, pairs
        pd.DataFrame({"customer_id": customer_ids}).to_csv(staging / "cohort.csv", index=False)
        shutil.copyfile(source, staging / snapshot_name)
        for filename in ["cohort.csv", snapshot_name]:
            path = staging / filename
            support_files.append({"file": filename, "size_bytes": path.stat().st_size,
                                  "sha256": file_sha256(path)})
        manifest = {
            "schema_version": 1, "status": "complete", "fingerprint": fingerprint,
            "dataset_split": dataset_split,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "cutoff": identity["cutoff"], "rows": sum(part["rows"] for part in parts),
            "customers": len(customer_ids), "candidates_per_customer": candidates_per_customer,
            "batch_customers": batch_customers, "schema": schema,
            "feature_columns": list(feature_columns), "nullable_features": list(nullable_features),
            "candidate_sha256": identity["candidate_sha256"], "provenance": provenance,
            "parts": parts, "support_files": support_files, "missing_counts": missing_counts,
            "key_uniqueness": "Distinct article lists per unique customer; disjoint customer partitions; exact row round trips.",
        }
        if source_path is not None:
            manifest["source_snapshot_sha256"] = identity["source_snapshot_sha256"]
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        checks = verify_feature_export(staging, expected_fingerprint=fingerprint)
        staging.rename(destination)
    except BaseException:
        shutil.rmtree(staging)
        raise
    return {"directory": destination, "manifest": manifest, "checks": checks, "reused": False}
