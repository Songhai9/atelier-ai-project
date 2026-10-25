"""Plan, execute and verify the frozen local recommendation pipeline."""

import argparse
from contextlib import redirect_stdout
from hashlib import sha256
import json
from pathlib import Path
import re
import sys


# Each specification contains: name, upstream stages, output directory.
STAGE_SPECS = (
    ("validate_sources", (), "source_validation"),
    ("prepare_sources", ("validate_sources",), "prepared"),
    ("build_training", ("prepare_sources",), "training"),
    ("fit_model", ("build_training",), "model"),
    ("predict_v1", ("prepare_sources", "fit_model"), "predictions"),
    ("evaluate_v1", ("predict_v1",), "evaluation"),
    (
        "package_service",
        ("fit_model", "predict_v1", "evaluate_v1"),
        "serving",
    ),
)


def build_plan(project_root: Path, run_id: str) -> dict:
    """Read the frozen decision and return a plan without writing files.

    Args:
        project_root: Root directory of the Atelier AI repository.
        run_id: Single directory name identifying the future campaign.

    Returns:
        Pipeline inputs, frozen settings and planned stage manifests.

    Raises:
        ValueError: If the run ID or frozen temporal contract is invalid.
    """
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", run_id) is None:
        raise ValueError(
            "run_id must contain letters, digits, underscores or hyphens "
            "and start with a letter or digit."
        )

    root = project_root.resolve()
    selection_path = (
        root / "data/processed/weekly_v1/final_selection.json"
    )
    temporal_path = root / "configs/temporal.yaml"

    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    decision = selection["decision"]

    temporal_hash = sha256(temporal_path.read_bytes()).hexdigest()
    if temporal_hash != decision["temporal_config_sha256"]:
        raise ValueError(
            "The temporal configuration differs from the frozen decision."
        )

    if decision["confirmation_reserved"] is not True:
        raise ValueError("Final confirmation must remain reserved.")

    output_root = root / "data/processed/pipeline_runs" / run_id
    v1 = decision["v1_evaluation"]

    return {
        "mode": "plan_only",
        "run_id": run_id,
        "output_root": str(output_root),
        "inputs": {
            "raw_sources": str(root / "data/raw/hm"),
            "source_manifest": str(root / "docs/data_manifest.json"),
            "temporal_config": str(temporal_path),
            "frozen_selection": str(selection_path),
        },
        "method": decision["method"],
        "feature_count": len(decision["feature_columns"]),
        "refit_parameters": decision["refit_parameters"],
        "training_cutoffs": decision["refit_protocol"]["training_cutoffs"],
        "v1_window": {
            "start_inclusive": v1["start_inclusive"],
            "end_exclusive": v1["end_exclusive"],
        },
        "confirmation_included": False,
        "stages": [
            {
                "name": name,
                "depends_on": list(dependencies),
                "planned_manifest": str(
                    output_root / directory / "manifest.json"
                ),
            }
            for name, dependencies, directory in STAGE_SPECS
        ],
    }


def main() -> None:
    """Expose all pipeline stages, resumable execution and read-only verification."""
    from atelier_ai.pipeline_control import STAGE_COMMANDS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=["plan", "training-plan", "status", "verify", "run", *STAGE_COMMANDS],
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--until", choices=list(STAGE_COMMANDS),
                        help="Last stage to execute; valid only with run.")
    args = parser.parse_args()
    if args.until is not None and args.command != "run":
        parser.error("--until is supported only by run")

    plan = build_plan(args.project_root, args.run_id)

    if args.command == "plan":
        print(json.dumps(plan, indent=2))
        return

    if args.command == "training-plan":
        from atelier_ai.pipeline_training import load_training_context

        context = load_training_context(args.project_root, plan)
        schedule = context["schedule"]
        customers = len(context["customer_ids"])
        groups = customers * len(schedule)

        print(
            json.dumps(
                {
                    "mode": "training_plan_only",
                    "cohort_size": customers,
                    "cohort_sha256": context["cohort_sha256"],
                    "feature_count": len(context["feature_columns"]),
                    "planned_customer_cutoff_groups": groups,
                    "planned_candidate_rows": (
                        groups * context["candidate_budget"]
                    ),
                    "schedule": schedule[
                        ["cutoff", "history_start", "target_end"]
                    ].astype(str).to_dict(orient="records"),
                },
                indent=2,
            )
        )
        return

    from atelier_ai.pipeline_control import (
        execute_stage, pipeline_status, run_pipeline, verify_pipeline,
    )

    with redirect_stdout(sys.stderr):
        if args.command == "status":
            result = pipeline_status(plan)
        elif args.command == "verify":
            result = verify_pipeline(args.project_root, plan)
        elif args.command == "run":
            result = run_pipeline(args.project_root, plan, until=args.until or "package-service")
        else:
            result = execute_stage(args.project_root, plan, args.command)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
