"""Unified stage execution and read-only status/integrity checks."""

from importlib import import_module
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile


STAGE_COMMANDS = {
    "validate-sources": "validate_sources", "prepare-sources": "prepare_sources",
    "build-training": "build_training", "fit-model": "fit_model", "predict-v1": "predict_v1",
    "evaluate-v1": "evaluate_v1", "package-service": "package_service",
}
BUILDERS = {
    "build_training": ("pipeline_dataset", "build_training_dataset"),
    "fit_model": ("pipeline_model", "fit_frozen_model"),
    "predict_v1": ("pipeline_predict", "predict_v1"),
    "evaluate_v1": ("pipeline_evaluate", "evaluate_v1"),
    "package_service": ("pipeline_package", "package_service"),
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _stage(plan, name):
    return next(stage for stage in plan["stages"] if stage["name"] == name)


def _record(stage):
    path = Path(stage["planned_manifest"])
    report = _read(path)
    _require(report.get("stage") == stage["name"] and report.get("status") == "complete",
             f"A complete {stage['name']} manifest is required.")
    return report


def _publish(path, report):
    if path.exists():
        _require(_read(path) == report, "An incompatible validation is already published; use a new campaign.")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                            prefix=".publish-", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(json.dumps(report, indent=2) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            os.link(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def pipeline_status(plan):
    """Inspect completion records only, without reading data or contacting MLflow."""
    stages = []
    for stage in plan["stages"]:
        path = Path(stage["planned_manifest"])
        state, error = "missing", None
        if path.exists():
            try:
                _record(stage)
                state = "recorded_complete"
            except (OSError, ValueError, TypeError, AttributeError) as exc:
                state, error = "invalid", str(exc)
        entry = {"name": stage["name"], "state": state, "manifest": str(path)}
        if error is not None:
            entry["error"] = error
        stages.append(entry)
    return {"mode": "manifest_status_only", "run_id": plan["run_id"],
            "artifacts_verified": False, "recorded_complete": sum(s["state"] == "recorded_complete" for s in stages),
            "stage_count": len(stages), "stages": stages}


def verify_pipeline(project_root, plan):
    """Verify raw sources and the recursive scientific/MLflow/serving lineage."""
    from atelier_ai.pipeline_package import verify_service_package
    from atelier_ai.source_validation import validate_sources

    for stage in plan["stages"]:
        _record(stage)
    expected = _record(_stage(plan, "validate_sources"))
    current = validate_sources(Path(project_root), Path(plan["inputs"]["source_manifest"]))
    _require(expected == current, "Current raw sources differ from the saved validation.")
    # The package verifier traverses evaluation, prediction, model, training and prepared inputs.
    serving = verify_service_package(project_root, plan)
    return {"mode": "full_verification", "run_id": plan["run_id"], "artifacts_verified": True,
            "verified_stages": len(plan["stages"]), "status": "complete",
            "bundle": str(Path(_stage(plan, "package_service")["planned_manifest"]).parent / "bundle"),
            "model_sha256": serving["model_sha256"], "confirmation_evaluated": serving["confirmation_evaluated"],
            "stages": [{"name": stage["name"], "state": "verified"} for stage in plan["stages"]]}


def _summary(result):
    report = result["manifest"]
    keys = (
        "rows", "raw_candidate_rows", "audited_groups", "eligible_groups", "training_rows",
        "training_groups", "positive_training_rows", "num_trees", "reload_max_abs_error",
        "recipients", "candidate_rows", "recommendation_rows", "target_pairs", "target_buyers",
        "customers", "catalogue_articles", "serving_mode", "historical_date", "history_rows",
    )
    return {"stage": report["stage"], "status": report["status"],
            "manifest": str(result["manifest_path"]), "reused": result["reused"],
            **{key: report[key] for key in keys if key in report}}


def execute_stage(project_root, plan, command):
    """Run one existing stage implementation; preserve its cache and manifest format."""
    _require(command in STAGE_COMMANDS, f"Unknown stage command: {command}")
    root = Path(project_root).resolve()
    name = STAGE_COMMANDS[command]
    stage = _stage(plan, name)
    for dependency in stage["depends_on"]:
        _record(_stage(plan, dependency))
    marker = Path(stage["planned_manifest"])
    if name in ("validate_sources", "prepare_sources"):
        from atelier_ai.source_validation import validate_sources

        validation = validate_sources(root, Path(plan["inputs"]["source_manifest"]))
        if name == "validate_sources":
            reused = marker.exists()
            _publish(marker, validation)
            return _summary({"manifest": validation, "manifest_path": marker, "reused": reused})
        _require(_record(_stage(plan, "validate_sources")) == validation,
                 "Current raw sources differ from the saved validation.")
        from atelier_ai.preparation import prepare_sources

        reused = marker.parent.exists()
        directory = prepare_sources(root, output_directory=marker.parent)
        return _summary({"manifest": _read(directory / "manifest.json"), "manifest_path": marker, "reused": reused})
    module_name, function_name = BUILDERS[name]
    builder = getattr(import_module(f"atelier_ai.{module_name}"), function_name)
    return _summary(builder(root, plan))


def run_pipeline(project_root, plan, *, until="package-service"):
    """Execute the ordered prefix; already completed stages are checked and reused."""
    _require(until in STAGE_COMMANDS, f"Unknown final stage command: {until}")
    commands = {name: command for command, name in STAGE_COMMANDS.items()}
    results = []
    for stage in plan["stages"]:
        results.append(execute_stage(project_root, plan, commands[stage["name"]]))
        if stage["name"] == STAGE_COMMANDS[until]:
            break
    return {"mode": "pipeline_run", "run_id": plan["run_id"], "status": "complete",
            "completed_through": STAGE_COMMANDS[until], "stages": results}
