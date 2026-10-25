"""Assemble the frozen seven-week fitting dataset from verified cutoff exports."""

import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import duckdb
import numpy as np
import pandas as pd

from atelier_ai.feature_export import file_sha256, verify_feature_export
from atelier_ai.pipeline_features import SOURCE_NAMES, export_training_cutoff
from atelier_ai.pipeline_training import load_training_context
from atelier_ai.ranking_data import (
    GROUP_KEYS,
    ROW_KEYS,
    inspect_training_groups,
    load_eligible_training_rows,
    prepare_ranker_inputs,
)
from atelier_ai.workflow import provenance


ARTIFACT_NAMES = (
    "rows.parquet", "group_audit.parquet", "groups.parquet", "targets.parquet",
    "cohort.csv", "schedule.csv", "summary.csv",
)
SUMMARY_COLUMNS = (
    "cutoff", "target_end_exclusive", "candidate_rows", "audited_groups",
    "eligible_groups", "positive_candidate_pairs", "source_purchase_rows",
    "target_pairs", "target_buyers",
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _paths(plan):
    stage = next(s for s in plan["stages"] if s["name"] == "build_training")
    marker = Path(stage["planned_manifest"])
    return Path(plan["output_root"]), marker, marker.parent / "dataset"


def _identity(path):
    return {"size_bytes": path.stat().st_size, "sha256": file_sha256(path)}


def _recipe(root, plan, context):
    run_root, _, _ = _paths(plan)
    evidence = provenance(
        root, context["customer_ids"],
        snapshots_directory=run_root / "source_snapshots",
        source_names=(*SOURCE_NAMES, "ranking_data.py", "pipeline_dataset.py"),
    )
    evidence["versions"] = {
        "duckdb": duckdb.__version__, "pandas": pd.__version__,
        "numpy": np.__version__,
    }
    return {
        "dataset_role": "final_model_fitting",
        "cohort_sha256": context["cohort_sha256"],
        "selection_sha256": file_sha256(Path(plan["inputs"]["frozen_selection"])),
        "prepared_manifest_sha256": context["prepared_manifest_sha256"],
        "training_cutoffs": context["schedule"]["cutoff"].dt.strftime("%Y-%m-%d").tolist(),
        "fitting_boundary_exclusive": context["schedule"]["target_end"].max().date().isoformat(),
        "feature_columns": context["feature_columns"],
        "candidates_per_group": context["candidate_budget"],
        "uses_v1_targets": False,
        "fitting_order": ROW_KEYS,
        "provenance": evidence,
    }


def _write_frames(directory, frames):
    """Check exact multiset round trips, including duplicates and null values."""
    with TemporaryDirectory(prefix="atelier-dataset-duckdb-") as scratch:
        with duckdb.connect(config={
            "threads": 2, "memory_limit": "1GB", "temp_directory": scratch,
        }) as connection:
            for name, frame in frames.items():
                path = directory / f"{name}.parquet"
                connection.register("_payload", frame)
                connection.sql("SELECT * FROM _payload").write_parquet(
                    str(path), compression="zstd",
                )
                differences = connection.execute(
                    "SELECT count(*) FROM ((SELECT * FROM _payload EXCEPT ALL "
                    "SELECT * FROM read_parquet(?)) UNION ALL (SELECT * FROM "
                    "read_parquet(?) EXCEPT ALL SELECT * FROM _payload))",
                    [str(path), str(path)],
                ).fetchall()[0][0]
                _require(differences == 0, f"Dataset round trip differs: {name}")
                connection.unregister("_payload")


def _verify_payload(plan, directory, manifest, expected_recipe=None):
    run_root, marker, _ = _paths(plan)
    _require(
        manifest.get("schema_version") == 1
        and manifest.get("stage") == "build_training"
        and manifest.get("status") == "complete"
        and manifest.get("dataset_role") == "final_model_fitting"
        and manifest.get("uses_v1_targets") is False,
        "A complete final-training manifest is required.",
    )
    if expected_recipe is not None:
        _require(
            all(manifest.get(k) == v for k, v in expected_recipe.items()),
            "Training recipe differs; preserve this dataset and use a new campaign.",
        )
    _require(
        manifest["selection_sha256"] == file_sha256(Path(plan["inputs"]["frozen_selection"])),
        "The frozen selection differs from the training dataset.",
    )
    prepared = next(s for s in plan["stages"] if s["name"] == "prepare_sources")
    _require(
        manifest["prepared_manifest_sha256"] == file_sha256(Path(prepared["planned_manifest"])),
        "The prepared input manifest differs from the training dataset.",
    )
    _require(set(manifest["artifacts"]) == set(ARTIFACT_NAMES), "Dataset artifact list differs.")
    for name, record in manifest["artifacts"].items():
        path = directory / name
        _require(path.is_file(), f"Missing training artifact: {name}")
        _require(_identity(path) == record, f"Training checksum differs: {name}")

    exports = manifest["feature_exports"]
    _require(
        [entry["cutoff"] for entry in exports] == manifest["training_cutoffs"],
        "Feature exports do not match the declared calendar.",
    )
    total_rows = 0
    expected_sources = {
        name: manifest["provenance"]["source_sha256"][name]
        for name in SOURCE_NAMES
    }
    for entry in exports:
        feature_directory = (run_root / entry["directory"]).resolve()
        _require(
            feature_directory.is_relative_to((marker.parent / "features").resolve()),
            "A feature export lies outside this training campaign.",
        )
        _require(
            file_sha256(feature_directory / "manifest.json") == entry["manifest_sha256"],
            "A feature manifest differs from the assembled dataset.",
        )
        verify_feature_export(feature_directory, expected_fingerprint=entry["fingerprint"])
        feature = json.loads((feature_directory / "manifest.json").read_text())
        valid = (
            feature["dataset_split"] == "training"
            and pd.Timestamp(feature["cutoff"]).date().isoformat() == entry["cutoff"]
            and feature["feature_columns"] == manifest["feature_columns"]
            and feature["candidates_per_customer"] == manifest["candidates_per_group"]
            and feature["provenance"]["cohort_sha256"] == manifest["cohort_sha256"]
            and feature["provenance"]["source_sha256"] == expected_sources
            and feature["provenance"]["versions"] == manifest["provenance"]["versions"]
            and feature["provenance"]["prepared_manifest_sha256"] == manifest["prepared_manifest_sha256"]
            and feature["provenance"]["frozen_selection_sha256"] == manifest["selection_sha256"]
        )
        _require(valid, "Feature exports mix incompatible inputs or code.")
        total_rows += feature["rows"]
    _require(total_rows == manifest["raw_candidate_rows"], "Raw candidate total differs.")

    with duckdb.connect(config={"threads": 2, "memory_limit": "512MB"}) as connection:
        rows, positives = connection.execute(
            "SELECT count(*), sum(label) FROM read_parquet(?)",
            [str(directory / "rows.parquet")],
        ).fetchall()[0]
        groups, grouped_rows = connection.execute(
            "SELECT count(*), sum(group_size) FROM read_parquet(?)",
            [str(directory / "groups.parquet")],
        ).fetchall()[0]
        audited, eligible, candidate_rows = connection.execute(
            "SELECT count(*), sum(CASE WHEN eligible THEN 1 ELSE 0 END), "
            "sum(candidate_rows) FROM read_parquet(?)",
            [str(directory / "group_audit.parquet")],
        ).fetchall()[0]
        targets = connection.execute(
            "SELECT count(*) FROM read_parquet(?)",
            [str(directory / "targets.parquet")],
        ).fetchall()[0][0]
    _require(rows == grouped_rows == manifest["training_rows"], "Fitting row total differs.")
    _require(positives == manifest["positive_training_rows"], "Fitting positive total differs.")
    _require(groups == eligible == manifest["eligible_groups"], "Fitting group total differs.")
    _require(audited == manifest["audited_groups"], "Audit group total differs.")
    _require(candidate_rows == manifest["raw_candidate_rows"], "Audit candidate total differs.")
    _require(targets == manifest["target_pairs"], "Full target total differs.")


def verify_training_dataset(project_root, plan, *, expected_recipe=None):
    """Verify the published dataset and all of its referenced feature exports."""
    _, marker, directory = _paths(plan)
    manifest = json.loads(marker.read_text(encoding="utf-8"))
    stored = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    _require(manifest == stored, "Published training manifests differ.")
    _verify_payload(plan, directory, manifest, expected_recipe)
    return manifest


def _publish_marker(marker, manifest):
    if marker.exists():
        _require(
            json.loads(marker.read_text(encoding="utf-8")) == manifest,
            "An incompatible training manifest is already published.",
        )
    else:
        with marker.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(manifest, indent=2) + "\n")


def build_training_dataset(project_root, plan):
    """Export each frozen week and retain complete informative groups for fitting."""
    root = Path(project_root).resolve()
    context = load_training_context(root, plan)
    run_root, marker, directory = _paths(plan)
    recipe = _recipe(root, plan, context)

    if directory.exists():
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        _verify_payload(plan, directory, manifest, recipe)
        _publish_marker(marker, manifest)
        return {"manifest": manifest, "manifest_path": marker, "reused": True}
    _require(not marker.exists(), "The published dataset directory is missing.")

    paths, exports, target_parts, summaries = [], [], [], []
    for cutoff in context["schedule"]["cutoff"]:
        result = export_training_cutoff(root, plan, cutoff)
        summary = result["summary"]
        print(f"Training cutoff {cutoff.date()}: reused={summary['reused']}", flush=True)
        exported = result["feature_export"]
        paths.extend(exported["directory"] / p["file"] for p in exported["manifest"]["parts"])
        exports.append({
            "cutoff": cutoff.date().isoformat(),
            "directory": exported["directory"].relative_to(run_root).as_posix(),
            "fingerprint": exported["manifest"]["fingerprint"],
            "manifest_sha256": file_sha256(exported["directory"] / "manifest.json"),
        })
        target_parts.append(result["targets"].assign(cutoff=cutoff)[ROW_KEYS])
        summaries.append({key: summary[key] for key in SUMMARY_COLUMNS})

    ids = context["customer_ids"]
    boundary = context["schedule"]["target_end"].max()
    audit = inspect_training_groups(
        paths, validation_cutoff=boundary,
        candidates_per_group=context["candidate_budget"],
    )
    counts = audit.groupby("cutoff").size()
    _require(
        counts.index.strftime("%Y-%m-%d").tolist() == recipe["training_cutoffs"]
        and counts.eq(len(ids)).all()
        and audit["customer_id"].isin(ids).all(),
        "The audit does not preserve the complete customer-cutoff population.",
    )
    rows = load_eligible_training_rows(paths, audit)
    inputs = prepare_ranker_inputs(
        rows, validation_cutoff=boundary,
        feature_columns=context["feature_columns"],
    )
    _require(rows[ROW_KEYS].equals(inputs["keys"]), "Fitting key alignment differs.")
    groups = inputs["group_index"]
    del inputs
    targets = pd.concat(target_parts, ignore_index=True).sort_values(ROW_KEYS).reset_index(drop=True)
    _require(not targets.duplicated(ROW_KEYS).any(), "Full target keys are duplicated.")
    matched = rows.loc[rows["label"].eq(1), ROW_KEYS].merge(
        targets, on=ROW_KEYS, how="inner", validate="one_to_one",
    )
    _require(len(matched) == int(rows["label"].sum()), "A fitting positive has no reference target.")
    summary_table = pd.DataFrame(summaries)
    _require(
        int(summary_table["positive_candidate_pairs"].sum()) == int(audit["positive_rows"].sum())
        and int(summary_table["eligible_groups"].sum()) == int(audit["eligible"].sum()),
        "Per-cutoff summaries disagree with the complete audit.",
    )

    marker.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="atelier-fitting-", dir=marker.parent) as temporary:
        staging = Path(temporary)
        _write_frames(staging, {
            "rows": rows, "group_audit": audit, "groups": groups, "targets": targets,
        })
        pd.DataFrame({"customer_id": ids}).to_csv(staging / "cohort.csv", index=False)
        context["schedule"].to_csv(staging / "schedule.csv", index=False, date_format="%Y-%m-%d")
        summary_table.to_csv(staging / "summary.csv", index=False)
        manifest = {
            "schema_version": 1, "stage": "build_training", "status": "complete",
            **recipe,
            "feature_exports": exports,
            "raw_candidate_rows": int(audit["candidate_rows"].sum()),
            "audited_groups": len(audit),
            "eligible_groups": len(groups),
            "training_rows": len(rows),
            "positive_training_rows": int(rows["label"].sum()),
            "source_purchase_rows": int(summary_table["source_purchase_rows"].sum()),
            "target_pairs": len(targets),
            "artifacts": {name: _identity(staging / name) for name in ARTIFACT_NAMES},
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        _verify_payload(plan, staging, manifest, recipe)
        staging.rename(directory)
    _publish_marker(marker, manifest)
    return {"manifest": manifest, "manifest_path": marker, "reused": False}


def main():
    from atelier_ai.pipeline import build_plan

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    plan = build_plan(args.project_root, args.run_id)
    with redirect_stdout(sys.stderr):
        result = build_training_dataset(args.project_root, plan)
    report = result["manifest"]
    print(json.dumps({
        "stage": report["stage"], "status": report["status"],
        "manifest": str(result["manifest_path"]), "reused": result["reused"],
        **{key: report[key] for key in (
            "raw_candidate_rows", "audited_groups", "eligible_groups",
            "training_rows", "positive_training_rows", "target_pairs", "uses_v1_targets",
        )},
    }, indent=2))


if __name__ == "__main__":
    main()
