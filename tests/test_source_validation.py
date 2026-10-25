"""Check raw-source integrity with tiny synthetic files."""

from hashlib import sha256
import json

import pytest

from atelier_ai.source_validation import SOURCE_NAMES, validate_sources


def make_sources(root):
    records = []

    for name in SOURCE_NAMES:
        path = root / "data/raw/hm" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        contents = b"id\n0001\n"
        path.write_bytes(contents)
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": len(contents),
                "sha256": sha256(contents).hexdigest(),
            }
        )

    manifest_path = root / "docs/data_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps({"schema_version": 1, "files": records}),
        encoding="utf-8",
    )
    return manifest_path


def test_validation_preserves_inputs(tmp_path):
    manifest_path = make_sources(tmp_path)
    paths = [manifest_path, *sorted((tmp_path / "data/raw/hm").iterdir())]
    before = {path: path.read_bytes() for path in paths}

    report = validate_sources(tmp_path, manifest_path)

    assert report["status"] == "complete"
    assert len(report["files"]) == 3
    assert {path: path.read_bytes() for path in paths} == before


def test_same_size_change_is_rejected(tmp_path):
    manifest_path = make_sources(tmp_path)
    path = tmp_path / "data/raw/hm/articles.csv"
    path.write_bytes(b"id\n0002\n")

    with pytest.raises(ValueError, match="SHA-256 differs"):
        validate_sources(tmp_path, manifest_path)