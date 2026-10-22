"""Reader-policy regression tests using tiny local CSV fixtures."""

import pandas as pd
import pytest

from atelier_ai.data_io import read_source_chunks, read_source_csv


def test_full_and_chunked_readers_preserve_source_strings(tmp_path):
    path = tmp_path / "source.csv"
    path.write_text("id,category\n0001,NONE\n0002,\n0003,NaN\n", encoding="utf-8")
    expected = read_source_csv(path)
    with read_source_chunks(path, chunk_rows=2) as reader:
        chunks = list(reader)
    actual = pd.concat(chunks, ignore_index=True)
    pd.testing.assert_frame_equal(actual, expected)
    assert actual["id"].tolist() == ["0001", "0002", "0003"]
    assert actual["category"].tolist() == ["NONE", "", "NaN"]
    assert read_source_csv(path, columns=["id"], nrows=1).shape == (1, 1)


def test_chunk_reader_rejects_nonpositive_size(tmp_path):
    with pytest.raises(ValueError, match="positive"):
        read_source_chunks(tmp_path / "unused.csv", chunk_rows=0)
