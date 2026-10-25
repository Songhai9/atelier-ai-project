"""Fit and verify the frozen final ranker from the local training dataset."""

import argparse
from contextlib import redirect_stdout
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import shutil
import sys
from tempfile import NamedTemporaryFile, TemporaryDirectory
from time import perf_counter, time

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from atelier_ai.feature_export import file_sha256
from atelier_ai.pipeline_dataset import verify_training_dataset
from atelier_ai.ranking_data import ROW_KEYS, prepare_ranker_inputs
from atelier_ai.recommendation_features import FEATURE_COLUMNS


ARTIFACT_NAMES = ("model.txt", "reload_check.parquet", "feature_importance.csv")
EXPERIMENT_NAME = "atelier-ai-local-pipeline"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _identity(path):
    return {"size_bytes": path.stat().st_size, "sha256": file_sha256(path)}


def _paths(plan):
    stage = next(s for s in plan["stages"] if s["name"] == "fit_model")
    marker = Path(stage["planned_manifest"])
    training = next(s for s in plan["stages"] if s["name"] == "build_training")
    return marker, marker.parent / "bundle", Path(training["planned_manifest"])


def _publish_json(path, report):
    """Publish a complete JSON file exclusively; never replace existing evidence."""
    if path.exists():
        _require(_read(path) == report, f"Published evidence differs: {path.name}")
        return
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


def _contract(root, plan):
    """Check the saved input lineage and frozen temporal/model contract."""
    training = verify_training_dataset(root, plan)
    _, _, training_path = _paths(plan)
    selection_path = Path(plan["inputs"]["frozen_selection"])
    decision = _read(selection_path)["decision"]
    protocol = decision["refit_protocol"]
    columns = decision["feature_columns"]
    params = decision["refit_parameters"]
    boundary = decision["v1_evaluation"]["start_inclusive"]
    _require(
        protocol["early_stopping"] is False
        and protocol["uses_v1_targets"] is False
        and decision["confirmation_reserved"] is True,
        "The frozen refit must exclude V1 targets, confirmation and early stopping.",
    )
    _require(
        training["training_cutoffs"] == protocol["training_cutoffs"] == plan["training_cutoffs"]
        and training["fitting_boundary_exclusive"] == boundary == plan["v1_window"]["start_inclusive"]
        and training["cohort_sha256"] == decision["training"]["cohort_sha256"]
        and training["candidates_per_group"] == decision["retrieval"]["candidate_budget"]
        and training["feature_columns"] == columns == FEATURE_COLUMNS
        and plan["method"] == decision["method"]
        and plan["refit_parameters"] == params,
        "Training data or plan differs from the frozen refit contract.",
    )
    _require(
        params.get("objective") == "lambdarank"
        and type(params.get("n_estimators")) is int
        and params["n_estimators"] > 0
        and params.get("deterministic") is True
        and params.get("force_col_wise") is True
        and not any(k.startswith("early_stopping") for k in params),
        "Use the frozen deterministic ranker and its fixed iteration budget.",
    )
    runtime = {
        "python": platform.python_version(), "duckdb": duckdb.__version__,
        "numpy": np.__version__, "pandas": pd.__version__,
        "lightgbm": lgb.__version__, "scikit-learn": version("scikit-learn"),
        "mlflow": version("mlflow-skinny"),
    }
    provenance = training["provenance"]
    _require(
        all(runtime[k] == v for k, v in provenance["versions"].items())
        and runtime["python"] == provenance["python"],
        "The training runtime differs; preserve this campaign and use a new one.",
    )
    sources = dict(provenance["source_sha256"])
    for name, expected in sources.items():
        _require(
            file_sha256(root / "src/atelier_ai" / name) == expected,
            f"Training source differs: {name}; preserve this campaign.",
        )
    sources["pipeline_model.py"] = file_sha256(root / "src/atelier_ai/pipeline_model.py")
    recipe = {
        "method": decision["method"], "parameters": params,
        "feature_columns": columns, "input_dtype": "float32",
        "selection_sha256": file_sha256(selection_path),
        "training_manifest_sha256": file_sha256(training_path),
        "training_cutoffs": training["training_cutoffs"],
        "fitting_boundary_exclusive": boundary,
        "training_rows": training["training_rows"],
        "training_groups": training["eligible_groups"],
        "positive_training_rows": training["positive_training_rows"],
        "cohort_sha256": training["cohort_sha256"],
        "early_stopping": False, "uses_v1_targets": False,
        "confirmation_evaluated": False,
        "runtime": runtime, "source_sha256": sources,
    }
    return recipe, training_path.parent / "dataset"


def _read_parquet(path):
    with duckdb.connect(config={"threads": 2, "memory_limit": "1GB"}) as connection:
        return connection.execute("SELECT * FROM read_parquet(?)", [str(path)]).df()


def _verify_bundle(directory, recipe):
    report = _read(directory / "manifest.json")
    _require(
        report.get("schema_version") == 1
        and report.get("stage") == "fit_model"
        and report.get("status") == "complete"
        and report.get("role") == "final_train_plus_validation_refit"
        and all(report.get(k) == v for k, v in recipe.items()),
        "Model recipe differs; preserve this bundle and use a new campaign.",
    )
    _require(set(report["artifacts"]) == set(ARTIFACT_NAMES), "Model artifact list differs.")
    for name, identity in report["artifacts"].items():
        path = directory / name
        _require(path.is_file() and _identity(path) == identity, f"Model checksum differs: {name}")
    for name, digest in recipe["source_sha256"].items():
        _require(file_sha256(directory / "source" / name) == digest, f"Model source snapshot differs: {name}")
    booster = lgb.Booster(model_file=str(directory / "model.txt"))
    _require(
        booster.feature_name() == recipe["feature_columns"]
        and booster.num_trees() == report["num_trees"] == recipe["parameters"]["n_estimators"]
        and booster.params["objective"] == "lambdarank",
        "The saved native model differs from the frozen feature/tree contract.",
    )
    probes = _read_parquet(directory / "reload_check.parquet")
    positions = np.linspace(0, recipe["training_rows"] - 1,
                            num=min(1000, recipe["training_rows"]), dtype=int)
    _require(
        len(probes) == report["reload_probe_rows"] == len(positions)
        and np.array_equal(probes["row_position"].to_numpy(), positions),
        "Reload probe positions differ from the deterministic sampling rule.",
    )
    scores = booster.predict(probes[recipe["feature_columns"]].astype("float32"),
                             num_threads=recipe["parameters"]["n_jobs"], validate_features=True)
    before, after = probes["score_before"].to_numpy(), probes["score_after"].to_numpy()
    error = np.abs(before - after)
    _require(
        np.isfinite(before).all() and np.isfinite(after).all() and np.isfinite(scores).all()
        and np.allclose(scores, after, rtol=0, atol=1e-12)
        and np.allclose(before, after, rtol=0, atol=1e-12)
        and np.array_equal(probes["absolute_error"].to_numpy(), error)
        and float(error.max()) == report["reload_max_abs_error"],
        "Native model reload predictions differ.",
    )
    return report


def _fit_bundle(root, directory, recipe, dataset):
    rows = _read_parquet(dataset / "rows.parquet")
    inputs = prepare_ranker_inputs(rows, validation_cutoff=recipe["fitting_boundary_exclusive"],
                                   feature_columns=recipe["feature_columns"])
    _require(rows[ROW_KEYS].equals(inputs["keys"]), "Saved fitting rows are not in canonical order.")
    assert_frame_equal(inputs["group_index"], _read_parquet(dataset / "groups.parquet"))
    _require(
        len(rows) == recipe["training_rows"]
        and len(inputs["groups"]) == recipe["training_groups"]
        and int(inputs["y"].sum()) == recipe["positive_training_rows"],
        "Loaded fitting counts differ from the verified dataset.",
    )
    del rows
    X = inputs["X"].astype("float32")
    del inputs["X"]
    _require(not X.isin([np.inf, -np.inf]).any().any(), "Feature casting introduced infinity.")
    estimator = lgb.LGBMRanker(**recipe["parameters"])
    started = perf_counter()
    estimator.fit(X, inputs["y"], group=inputs["groups"])
    fit_seconds = perf_counter() - started
    estimator.booster_.save_model(str(directory / "model.txt"))
    booster = lgb.Booster(model_file=str(directory / "model.txt"))
    positions = np.linspace(0, len(X) - 1, num=min(1000, len(X)), dtype=int)
    probe = X.iloc[positions].reset_index(drop=True)
    before = estimator.predict(probe, num_threads=recipe["parameters"]["n_jobs"])
    after = booster.predict(probe, num_threads=recipe["parameters"]["n_jobs"], validate_features=True)
    probes = pd.concat([inputs["keys"].iloc[positions].reset_index(drop=True), probe], axis=1)
    probes = probes.assign(row_position=positions, score_before=before, score_after=after,
                           absolute_error=np.abs(before - after))
    with duckdb.connect() as connection:
        connection.register("_probes", probes)
        connection.sql("SELECT * FROM _probes").write_parquet(
            str(directory / "reload_check.parquet"), compression="zstd")
    pd.DataFrame({
        "feature": booster.feature_name(),
        "gain": booster.feature_importance(importance_type="gain"),
        "split_count": booster.feature_importance(importance_type="split"),
    }).to_csv(directory / "feature_importance.csv", index=False)
    source = directory / "source"
    source.mkdir()
    for name in recipe["source_sha256"]:
        shutil.copyfile(root / "src/atelier_ai" / name, source / name)
    report = {
        "schema_version": 1, "stage": "fit_model", "status": "complete",
        "role": "final_train_plus_validation_refit", **recipe,
        "num_trees": booster.num_trees(), "fit_seconds": fit_seconds,
        "reload_probe_rows": len(probes),
        "reload_max_abs_error": float(probes["absolute_error"].max()),
        "artifacts": {name: _identity(directory / name) for name in ARTIFACT_NAMES},
    }
    _publish_json(directory / "manifest.json", report)
    _verify_bundle(directory, recipe)
    return report


def _tracking_files(plan, directory, report):
    _, _, training_path = _paths(plan)
    files = {f"model/{name}": directory / name for name in (*ARTIFACT_NAMES, "manifest.json")}
    files["inputs/training/manifest.json"] = training_path
    selection = Path(plan["inputs"]["frozen_selection"])
    files[f"inputs/selection/{selection.name}"] = selection
    files.update({f"source/{name}": directory / "source" / name for name in report["source_sha256"]})
    return files


def _tracking_client(root):
    from mlflow.tracking import MlflowClient

    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    uri = (root / "mlruns").as_uri()
    return MlflowClient(tracking_uri=uri), uri


def _tracking_tags(plan, report):
    return {
        "pipeline.run_id": plan["run_id"], "pipeline.stage": "fit_model",
        "pipeline.training_sha256": report["training_manifest_sha256"],
        "pipeline.model_sha256": report["artifacts"]["model.txt"]["sha256"],
    }


def _verify_tracking(client, receipt, plan, directory, report):
    run = client.get_run(receipt["run_id"])
    _require(run.info.experiment_id == receipt["experiment_id"], "MLflow experiment differs.")
    tags = _tracking_tags(plan, report)
    _require(all(run.data.tags.get(k) == v for k, v in tags.items()), "MLflow input/model identity differs.")
    params = {k: str(v) for k, v in report["parameters"].items()}
    _require(all(run.data.params.get(k) == v for k, v in params.items()), "MLflow parameters differ.")
    for name in ("fit_seconds", "reload_max_abs_error"):
        _require(run.data.metrics.get(name) == report[name], f"MLflow metric differs: {name}")
    with TemporaryDirectory(prefix="atelier-mlflow-check-") as scratch:
        for name, path in _tracking_files(plan, directory, report).items():
            downloaded = Path(client.download_artifacts(receipt["run_id"], name, scratch))
            _require(_identity(downloaded) == _identity(path), f"MLflow artifact differs: {name}")
    return run


def _track_bundle(root, plan, directory, report):
    """Resume logging from an immutable fit bundle without refitting the model."""
    from mlflow.entities import Metric, Param

    client, uri = _tracking_client(root)
    receipt_path = directory.parent / "tracking.json"
    if receipt_path.exists():
        receipt = _read(receipt_path)
        _require(receipt["tracking_uri"] == uri, "MLflow tracking location differs.")
    else:
        experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
        experiment_id = experiment.experiment_id if experiment else client.create_experiment(EXPERIMENT_NAME)
        run = client.create_run(experiment_id, run_name=f"{plan['run_id']}-frozen-refit",
                                tags=_tracking_tags(plan, report))
        receipt = {"tracking_uri": uri, "experiment_id": experiment_id, "run_id": run.info.run_id}
        _publish_json(receipt_path, receipt)
    run = client.get_run(receipt["run_id"])
    _require(
        run.info.experiment_id == receipt["experiment_id"]
        and all(run.data.tags.get(k) == v for k, v in _tracking_tags(plan, report).items()),
        "MLflow input/model identity differs; no tracking artifacts were changed.",
    )
    if run.info.status != "FINISHED":
        _require(run.info.status == "RUNNING", "The saved MLflow run is not resumable.")
        client.log_batch(receipt["run_id"], params=[
            Param(k, str(v)) for k, v in report["parameters"].items()
        ], metrics=[Metric(k, report[k], int(time() * 1000), 0)
                    for k in ("fit_seconds", "reload_max_abs_error")])
        for name, path in _tracking_files(plan, directory, report).items():
            client.log_artifact(receipt["run_id"], str(path), artifact_path=str(Path(name).parent))
        _verify_tracking(client, receipt, plan, directory, report)
        client.set_terminated(receipt["run_id"], status="FINISHED")
    else:
        _verify_tracking(client, receipt, plan, directory, report)
    return receipt


def verify_fitted_model(project_root, plan):
    """Verify inputs, native reload scores and the completed local MLflow run."""
    root = Path(project_root).resolve()
    recipe, _ = _contract(root, plan)
    marker, directory, _ = _paths(plan)
    report = _verify_bundle(directory, recipe)
    published = _read(marker)
    receipt = _read(directory.parent / "tracking.json")
    _require(published == {**report, "mlflow": receipt}, "Published model manifest differs.")
    client, uri = _tracking_client(root)
    _require(receipt["tracking_uri"] == uri, "MLflow tracking location differs.")
    run = _verify_tracking(client, receipt, plan, directory, report)
    _require(run.info.status == "FINISHED", "MLflow logging is not complete.")
    return published


def fit_frozen_model(project_root, plan):
    """Fit once, verify reloads, then publish the stage after MLflow logging."""
    root = Path(project_root).resolve()
    marker, directory, _ = _paths(plan)
    if marker.exists():
        report = verify_fitted_model(root, plan)
        return {"manifest": report, "manifest_path": marker, "reused": True}
    recipe, dataset = _contract(root, plan)
    reused = directory.exists()
    if reused:
        report = _verify_bundle(directory, recipe)
    else:
        _require(not (marker.parent / "tracking.json").exists(), "Tracking evidence exists without a fit bundle.")
        marker.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix="atelier-model-", dir=marker.parent) as temporary:
            staging = Path(temporary)
            report = _fit_bundle(root, staging, recipe, dataset)
            staging.rename(directory)
    receipt = _track_bundle(root, plan, directory, report)
    published = {**report, "mlflow": receipt}
    _publish_json(marker, published)
    return {"manifest": published, "manifest_path": marker, "reused": reused}


def main():
    from atelier_ai.pipeline import build_plan

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    plan = build_plan(args.project_root, args.run_id)
    with redirect_stdout(sys.stderr):
        result = fit_frozen_model(args.project_root, plan)
    report = result["manifest"]
    print(json.dumps({
        "stage": report["stage"], "status": report["status"],
        "manifest": str(result["manifest_path"]), "reused": result["reused"],
        "feature_count": len(report["feature_columns"]),
        **{key: report[key] for key in (
            "num_trees", "training_rows", "training_groups", "positive_training_rows",
            "fit_seconds", "reload_probe_rows", "reload_max_abs_error", "uses_v1_targets",
        )},
        "model_sha256": report["artifacts"]["model.txt"]["sha256"],
        "mlflow_run_id": report["mlflow"]["run_id"],
    }, indent=2))


if __name__ == "__main__":
    main()
