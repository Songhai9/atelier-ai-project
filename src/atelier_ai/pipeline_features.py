"""Export labelled candidates for one frozen final-training cutoff."""

import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys

import duckdb
import numpy as np
import pandas as pd

from atelier_ai.feature_export import export_feature_batches, file_sha256
from atelier_ai.pipeline_training import load_training_context
from atelier_ai.ranking_data import inspect_training_groups
from atelier_ai.recommendation_features import (
    EXPECTED_MISSING_FEATURES,
    build_checked_feature_batch,
    feature_storage_schema,
    prepare_feature_tables,
    retrieve_candidates,
)
from atelier_ai.training_targets import label_candidate_pairs
from atelier_ai.workflow import build_context, provenance, target_reference


# Only dependencies of this export belong to its computation identity.
SOURCE_NAMES = (
    "feature_export.py",
    "historical_contexts.py",
    "pipeline_features.py",
    "pipeline_training.py",
    "preparation.py",
    "prepared_validation.py",
    "recommendation_features.py",
    "temporal.py",
    "temporal_protocol.py",
    "training_targets.py",
    "workflow.py",
)


def export_training_cutoff(project_root, plan, cutoff):
    """Export complete groups while keeping full future targets separately."""
    root = Path(project_root).resolve()
    training = load_training_context(root, plan)
    ids = training["customer_ids"]
    schedule = training["schedule"]
    cutoff = pd.Timestamp(cutoff)
    window = schedule.loc[schedule["cutoff"].eq(cutoff)]
    if len(window) != 1:
        raise ValueError("Choose a cutoff from the frozen final-training calendar.")
    boundary = schedule["target_end"].max()

    selection_path = Path(plan["inputs"]["frozen_selection"])
    policy = json.loads(selection_path.read_text(encoding="utf-8"))[
        "decision"
    ]["retrieval"]
    if policy["fallback_days"] != 28:
        raise ValueError("The shared retrieval implementation requires 28-day fill.")

    historical = build_context(
        root, ids, cutoff, prepared_directory=training["prepared_directory"]
    )
    pools, memberships, _, _ = retrieve_candidates(
        ids,
        historical,
        budget=policy["candidate_budget"],
        history_limit=policy["history_limit"],
        global_limit=policy["global_limit_7d"],
        preferred_types=policy["preferred_product_types"],
        per_type=policy["per_preferred_type"],
    )
    tables = prepare_feature_tables(ids, historical, memberships)

    source_rows, targets = target_reference(
        root, ids, cutoff, prepared_directory=training["prepared_directory"]
    )
    relevance = targets.assign(cutoff=cutoff, label=1)

    def builder(pairs, feature_tables):
        features = build_checked_feature_batch(pairs, feature_tables)
        return label_candidate_pairs(
            features,
            relevance,
            schedule=schedule,
            customer_ids=ids,
            validation_cutoff=boundary,
        )

    run_root = Path(plan["output_root"])
    identity = provenance(
        root,
        ids,
        snapshots_directory=run_root / "source_snapshots",
        source_names=SOURCE_NAMES,
    )
    identity.update(policy)
    identity.update(
        {
            "prepared_manifest_sha256": training["prepared_manifest_sha256"],
            "frozen_selection_sha256": file_sha256(selection_path),
            "versions": {
                "duckdb": duckdb.__version__,
                "pandas": pd.__version__,
                "numpy": np.__version__,
            },
        }
    )

    exported = export_feature_batches(
        ids,
        pools,
        tables,
        builder,
        output_base=run_root / "training/features",
        feature_columns=training["feature_columns"],
        nullable_features=EXPECTED_MISSING_FEATURES,
        sql_schema=feature_storage_schema(include_label=True),
        provenance=identity,
        source_path=Path(__file__),
        candidates_per_customer=policy["candidate_budget"],
        batch_customers=500,
        dataset_split="training",
    )
    paths = [
        exported["directory"] / part["file"]
        for part in exported["manifest"]["parts"]
    ]
    audit = inspect_training_groups(
        paths,
        validation_cutoff=boundary,
        candidates_per_group=policy["candidate_budget"],
    )
    summary = {
        "stage": "export_training_cutoff",
        "status": "complete",
        "cutoff": cutoff.date().isoformat(),
        "target_end_exclusive": window["target_end"].iloc[0].date().isoformat(),
        "directory": str(exported["directory"]),
        "fingerprint": exported["manifest"]["fingerprint"],
        "candidate_rows": exported["checks"]["rows"],
        "audited_groups": len(audit),
        "eligible_groups": int(audit["eligible"].sum()),
        "positive_candidate_pairs": int(audit["positive_rows"].sum()),
        "source_purchase_rows": len(source_rows),
        "target_pairs": len(targets),
        "target_buyers": int(targets["customer_id"].nunique()),
        "reused": exported["reused"],
    }
    return {
        "summary": summary,
        "targets": targets,
        "group_audit": audit,
        "feature_export": exported,
    }


def main():
    """Run one declared cutoff and print its actual export summary."""
    from atelier_ai.pipeline import build_plan

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--cutoff", required=True)
    args = parser.parse_args()
    plan = build_plan(args.project_root, args.run_id)
    with redirect_stdout(sys.stderr):
        result = export_training_cutoff(args.project_root, plan, args.cutoff)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
