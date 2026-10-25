"""Typed local Parquet sources and deterministic customer selection."""

from hashlib import sha256
from pathlib import Path
import json
import duckdb
import pandas as pd

from tempfile import TemporaryDirectory

from atelier_ai.prepared_validation import (
    inspect_prepared_files,
    verify_prepared_sources,
)


def select_customer_cohort(customer_ids, *, size=20_000, seed=42):
    ids = pd.Series(customer_ids, dtype="string", name="customer_id")
    if ids.isna().any() or not ids.is_unique or ids.str.strip().eq("").any():
        raise ValueError("Customer identifiers must be present, nonblank and unique.")
    if not 0 < size <= len(ids):
        raise ValueError("Cohort size must fit the source population.")
    frame = ids.to_frame()
    frame["selection_key"] = frame["customer_id"].map(
        lambda value: sha256(f"{seed}:{value}".encode()).hexdigest()
    )
    return (
        frame.sort_values(["selection_key", "customer_id"])
        .head(size)
        .reset_index(drop=True)
    )


def prepare_sources(project_root, *, output_directory=None):
    """Create or verify typed Parquet sources in the requested directory."""
    root = Path(project_root)
    raw = root / "data/raw/hm"
    output = (
        root / "data/processed/prepared"
        if output_directory is None
        else Path(output_directory)
    )
    manifest_path = output / "manifest.json"

    source_bytes = (root / "docs/data_manifest.json").read_bytes()
    source_manifest = json.loads(source_bytes)

    recipe = {
        "source_manifest": source_manifest,
        "source_manifest_sha256": sha256(source_bytes).hexdigest(),
        "duckdb_version": duckdb.__version__,
        "source_code_sha256": {
            name: sha256(
                Path(__file__).with_name(name).read_bytes()
            ).hexdigest()
            for name in ("preparation.py", "prepared_validation.py")
        },
    }

    if output.exists():
        if not manifest_path.is_file():
            raise FileExistsError(
                "Prepared directory has no manifest; preserve it "
                "and use a different output directory."
            )

        previous = json.loads(manifest_path.read_text(encoding="utf-8"))

        # Preserve compatibility with the historical notebook campaign.
        if (
            output_directory is None
            and previous.get("schema_version") != 2
        ):
            if previous.get("source_manifest") == source_manifest and all(
                (output / f"{name}.parquet").is_file()
                for name in ("articles", "customers", "transactions")
            ):
                return output
            raise ValueError("Historical prepared-source identity differs.")

        verify_prepared_sources(output, expected_recipe=recipe)
        return output

    output.parent.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(
        prefix="atelier-prepared-",
        dir=output.parent,
    ) as temporary:
        staging = Path(temporary)
        counts = {}

        with duckdb.connect(
            config={"threads": 4, "memory_limit": "2GB"}
        ) as connection:
            for source, name in (
                ("articles", "articles"),
                ("customers", "customers"),
                ("transactions_train", "transactions"),
            ):
                relation = connection.read_csv(
                    str(raw / f"{source}.csv"),
                    all_varchar=True,
                )
                if name == "transactions":
                    relation = relation.project(
                        "CAST(t_dat AS DATE) t_dat, customer_id, article_id, "
                        "CAST(price AS DOUBLE) price, "
                        "CAST(sales_channel_id AS SMALLINT) sales_channel_id"
                    )

                path = staging / f"{name}.parquet"
                relation.write_parquet(str(path), compression="zstd")
                counts[name] = connection.execute(
                    "SELECT count(*) FROM read_parquet(?)",
                    [str(path)],
                ).fetchall()[0][0]
                print(f"Prepared {name}: {counts[name]:,} rows", flush=True)

        artifacts = inspect_prepared_files(staging, counts)
        manifest = {
            "schema_version": 2,
            "stage": "prepare_sources",
            "status": "complete",
            **recipe,
            "rows": counts,
            "artifacts": artifacts,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )
        staging.rename(output)

    return output


def open_prepared(project_root, *, prepared_directory=None):
    root = Path(project_root)
    output = (
        root / "data/processed/prepared"
        if prepared_directory is None
        else Path(prepared_directory)
    )
    if not (output / "manifest.json").exists():
        raise FileNotFoundError("Run source preparation first.")
    con = duckdb.connect(config={"threads": 4, "memory_limit": "2GB"})
    for name in ["transactions", "articles", "customers"]:
        con.read_parquet(str(output / f"{name}.parquet")).create_view(name)
    return con
