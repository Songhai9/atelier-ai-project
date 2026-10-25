"""Load and check the inputs required by the frozen training pipeline."""

from hashlib import sha256
import json
from pathlib import Path

import pandas as pd

from atelier_ai.preparation import (
    open_prepared,
    prepare_sources,
    select_customer_cohort,
)
from atelier_ai.recommendation_features import FEATURE_COLUMNS
from atelier_ai.temporal_protocol import load_temporal_protocol
from atelier_ai.training_targets import make_training_schedule


def load_training_context(project_root, plan):
    """Verify prepared inputs and reconstruct the frozen cohort and calendar.

    No candidate generation, target extraction or model fitting is performed.
    """
    root = Path(project_root).resolve()

    prepared_stage = next(
        stage
        for stage in plan["stages"]
        if stage["name"] == "prepare_sources"
    )
    prepared_manifest = Path(prepared_stage["planned_manifest"])
    prepared_directory = prepared_manifest.parent

    if not prepared_manifest.is_file():
        raise FileNotFoundError(
            "Complete prepare-sources for this run ID first."
        )

    # The existing dataset is verified through the reuse branch.
    prepare_sources(root, output_directory=prepared_directory)

    decision = json.loads(
        Path(plan["inputs"]["frozen_selection"]).read_text(
            encoding="utf-8"
        )
    )["decision"]

    protocol = load_temporal_protocol(root / "configs/temporal.yaml")

    with open_prepared(
        root,
        prepared_directory=prepared_directory,
    ) as connection:
        source_ids = connection.execute(
            "SELECT customer_id FROM customers"
        ).df()["customer_id"]

    selected = select_customer_cohort(
        source_ids,
        size=20_000,
        seed=42,
    )
    customer_ids = pd.Index(selected["customer_id"], dtype="string")
    cohort_hash = sha256(
        ("\n".join(customer_ids) + "\n").encode("utf-8")
    ).hexdigest()

    if cohort_hash != decision["training"]["cohort_sha256"]:
        raise ValueError("The reconstructed cohort differs from the frozen cohort.")

    schedule = make_training_schedule(
        validation_cutoff=protocol["v1_evaluation_start"],
        n_weeks=protocol["training_weeks"] + 1,
        horizon_days=protocol["horizon_days"],
        history_days=protocol["history_days"],
    )

    observed_cutoffs = (
        schedule["cutoff"].dt.strftime("%Y-%m-%d").tolist()
    )
    if observed_cutoffs != decision["refit_protocol"]["training_cutoffs"]:
        raise ValueError("The training calendar differs from the frozen decision.")

    if list(FEATURE_COLUMNS) != decision["feature_columns"]:
        raise ValueError("The feature names or their order have changed.")

    if (
        decision["refit_protocol"]["early_stopping"] is not False
        or decision["refit_protocol"]["uses_v1_targets"] is not False
    ):
        raise ValueError("The frozen final-refit protocol is incompatible.")

    return {
        "customer_ids": customer_ids,
        "cohort_sha256": cohort_hash,
        "schedule": schedule,
        "prepared_directory": prepared_directory,
        "prepared_manifest_sha256": sha256(
            prepared_manifest.read_bytes()
        ).hexdigest(),
        "feature_columns": list(FEATURE_COLUMNS),
        "candidate_budget": decision["retrieval"]["candidate_budget"],
    }