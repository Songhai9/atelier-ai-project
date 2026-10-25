"""Check isolated preparation with tiny synthetic source tables."""

from datetime import date
from hashlib import sha256
import json
import pytest

from atelier_ai.prepared_validation import verify_prepared_sources
from atelier_ai.preparation import open_prepared, prepare_sources


def test_preparation_uses_supplied_directory_and_preserves_observations(
    tmp_path,
):
    sources = {
        "articles.csv": (
            "article_id,product_group_name\n"
            "0000000001,Unknown\n"
        ),
        "customers.csv": "customer_id\n0001\n",
        "transactions_train.csv": (
            "t_dat,customer_id,article_id,price,sales_channel_id\n"
            "2020-09-01,0001,0000000001,0.25,2\n"
            "2020-09-01,0001,0000000001,0.25,2\n"
        ),
    }

    raw = tmp_path / "data/raw/hm"
    raw.mkdir(parents=True)
    records = []

    for name, text in sources.items():
        contents = text.encode("utf-8")
        path = raw / name
        path.write_bytes(contents)
        records.append(
            {
                "path": path.relative_to(tmp_path).as_posix(),
                "size_bytes": len(contents),
                "sha256": sha256(contents).hexdigest(),
            }
        )

    manifest_path = tmp_path / "docs/data_manifest.json"
    manifest_path.parent.mkdir()
    manifest_path.write_text(
        json.dumps({"schema_version": 1, "files": records}),
        encoding="utf-8",
    )

    destination = (
        tmp_path / "data/processed/pipeline_runs/test_run/prepared"
    )
    result = prepare_sources(
        tmp_path,
        output_directory=destination,
    )

    assert result == destination
    assert (destination / "manifest.json").is_file()
    assert not (tmp_path / "data/processed/prepared").exists()

    with open_prepared(
        tmp_path,
        prepared_directory=destination,
    ) as connection:
        rows = connection.execute(
            "SELECT customer_id, article_id, t_dat, price, "
            "sales_channel_id FROM transactions"
        ).fetchall()

        assert rows == [
            ("0001", "0000000001", date(2020, 9, 1), 0.25, 2),
        ] * 2

        assert connection.execute(
            "SELECT product_group_name FROM articles"
        ).fetchall() == [("Unknown",)]

    for name, text in sources.items():
        assert (raw / name).read_bytes() == text.encode("utf-8")
    manifest_before = (destination / "manifest.json").read_bytes()
    report = verify_prepared_sources(destination)
    assert report["rows"] == {
        "articles": 1,
        "customers": 1,
        "transactions": 2,
    }

    assert prepare_sources(
        tmp_path,
        output_directory=destination,
    ) == destination
    assert (destination / "manifest.json").read_bytes() == manifest_before

    article_path = destination / "articles.parquet"
    contents = article_path.read_bytes()
    corrupted = bytes([contents[0] ^ 1]) + contents[1:]
    article_path.write_bytes(corrupted)

    with pytest.raises(ValueError, match="checksum"):
        prepare_sources(tmp_path, output_directory=destination)

    assert article_path.read_bytes() == corrupted