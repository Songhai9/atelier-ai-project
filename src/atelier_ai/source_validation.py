"""Validate raw file identities against the acquisition manifest."""

from hashlib import file_digest, sha256
import json
from pathlib import Path
import sys


SOURCE_NAMES = (
    "articles.csv",
    "customers.csv",
    "transactions_train.csv",
)


def validate_sources(project_root: Path, manifest_path: Path) -> dict:
    """Check all expected raw files without modifying them.

    Returns:
        A deterministic report after every file passes validation.

    Raises:
        ValueError: If the manifest or a source identity differs.
        FileNotFoundError: If a required source is absent.
    """
    root = project_root.resolve()
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)

    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported source manifest schema.")

    records = manifest["files"]
    expected_paths = {
        f"data/raw/hm/{name}" for name in SOURCE_NAMES
    }

    if (
        len(records) != len(expected_paths)
        or {record["path"] for record in records} != expected_paths
    ):
        raise ValueError(
            "The manifest must list each expected source exactly once."
        )

    checked = []

    for record in sorted(records, key=lambda item: item["path"]):
        path = root / record["path"]

        if not path.is_file():
            raise FileNotFoundError(f"Missing source: {record['path']}")

        size_bytes = path.stat().st_size

        if size_bytes <= 0 or size_bytes != record["size_bytes"]:
            raise ValueError(f"Source size differs: {record['path']}")

        print(
            f"Checking SHA-256: {record['path']}",
            file=sys.stderr,
            flush=True,
        )

        with path.open("rb") as source:
            digest = file_digest(source, "sha256").hexdigest()

        if digest != record["sha256"]:
            raise ValueError(f"Source SHA-256 differs: {record['path']}")

        checked.append(
            {
                "path": record["path"],
                "size_bytes": size_bytes,
                "sha256": digest,
            }
        )

    return {
        "schema_version": 1,
        "stage": "validate_sources",
        "status": "complete",
        "scope": "raw_file_identity",
        "source_manifest_sha256": sha256(manifest_bytes).hexdigest(),
        "files": checked,
    }