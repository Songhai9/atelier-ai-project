"""Record acquisition metadata and SHA-256 hashes for the raw H&M files."""

import argparse
import hashlib
import json
from datetime import date, datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIRECTORY = PROJECT_ROOT / "data" / "raw" / "hm"
MANIFEST_PATH = PROJECT_ROOT / "docs" / "data_manifest.json"

EXPECTED_FILES = (
    "articles.csv",
    "customers.csv",
    "transactions_train.csv",
)

SOURCE_URL = (
    "https://www.kaggle.com/competitions/"
    "h-and-m-personalized-fashion-recommendations/data"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an acquisition manifest for the raw H&M files."
    )
    parser.add_argument(
        "--acquired-on",
        required=True,
        type=date.fromisoformat,
        help="Actual acquisition date, in YYYY-MM-DD format.",
    )
    args = parser.parse_args()

    records = []

    for filename in EXPECTED_FILES:
        path = RAW_DIRECTORY / filename

        if not path.is_file():
            raise FileNotFoundError(f"Required source file not found: {path}")

        size_bytes = path.stat().st_size
        if size_bytes == 0:
            raise ValueError(f"Source file is empty: {path}")

        print(f"Computing SHA-256: {filename}", flush=True)

        with path.open("rb") as source_file:
            sha256 = hashlib.file_digest(source_file, "sha256").hexdigest()

        records.append(
            {
                "path": path.relative_to(PROJECT_ROOT).as_posix(),
                "size_bytes": size_bytes,
                "sha256": sha256,
            }
        )

    manifest = {
        "schema_version": 1,
        "dataset": "H&M Personalized Fashion Recommendations",
        "source_url": SOURCE_URL,
        "acquisition_method": "manual_download_from_kaggle",
        "acquired_on": args.acquired_on.isoformat(),
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ),
        "files": records,
    }

    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Manifest written: {MANIFEST_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()