"""Typed local Parquet sources and deterministic customer selection."""

from hashlib import sha256
from pathlib import Path
import json
import duckdb
import pandas as pd


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


def prepare_sources(project_root):
    """Preserve raw files; create typed, compressed, reusable Parquet tables."""
    root = Path(project_root)
    raw = root / "data/raw/hm"
    output = root / "data/processed/prepared"
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    source_manifest = json.loads((root / "docs/data_manifest.json").read_text())
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("source_manifest") == source_manifest and all(
            (output / f"{name}.parquet").exists()
            for name in ["transactions", "articles", "customers"]
        ):
            return output
        raise ValueError(
            "Prepared-source identity differs; preserve it and create a new version explicitly."
        )
    counts = {}
    with duckdb.connect(config={"threads": 4, "memory_limit": "2GB"}) as con:
        for source, name in [
            ("articles", "articles"),
            ("customers", "customers"),
            ("transactions_train", "transactions"),
        ]:
            relation = con.read_csv(str(raw / f"{source}.csv"), all_varchar=True)
            if name == "transactions":
                relation = relation.project(
                    "CAST(t_dat AS DATE) t_dat, customer_id, article_id, CAST(price AS DOUBLE) price, CAST(sales_channel_id AS SMALLINT) sales_channel_id"
                )
            relation.write_parquet(str(output / f"{name}.parquet"), compression="zstd")
            counts[name] = con.execute(
                "SELECT count(*) FROM read_parquet(?)",
                [str(output / f"{name}.parquet")],
            ).fetchall()[0][0]
            print(f"Prepared {name}: {counts[name]:,} rows", flush=True)
    manifest_path.write_text(
        json.dumps({"source_manifest": source_manifest, "rows": counts}, indent=2)
        + "\n"
    )
    return output


def open_prepared(project_root):
    root = Path(project_root)
    output = root / "data/processed/prepared"
    if not (output / "manifest.json").exists():
        raise FileNotFoundError("Run source preparation first.")
    con = duckdb.connect(config={"threads": 4, "memory_limit": "2GB"})
    for name in ["transactions", "articles", "customers"]:
        con.read_parquet(str(output / f"{name}.parquet")).create_view(name)
    return con
