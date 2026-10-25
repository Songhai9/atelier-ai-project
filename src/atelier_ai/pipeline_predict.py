"""Generate frozen V1 recommendations from unlabelled historical features."""

import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import platform
import shutil
import sys
from tempfile import TemporaryDirectory
from time import perf_counter

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd

from atelier_ai.feature_export import export_feature_batches, file_sha256, verify_feature_export
from atelier_ai.pipeline_model import (
    _identity, _publish_json, _read, _read_parquet, _tracking_client,
    verify_fitted_model,
)
from atelier_ai.pipeline_training import load_training_context
from atelier_ai.recommendation_features import (
    EXPECTED_MISSING_FEATURES, build_checked_feature_batch, feature_storage_schema,
    prepare_feature_tables, retrieve_candidates,
)
from atelier_ai.workflow import build_context, load_protocol


SOURCE_NAMES = (
    "feature_export.py", "historical_contexts.py", "pipeline_predict.py",
    "pipeline_model.py", "pipeline_training.py", "preparation.py",
    "prepared_validation.py", "recommendation_features.py", "temporal.py",
    "temporal_protocol.py", "training_targets.py", "workflow.py",
)
BASELINE_NAMES = (
    "recent_popularity_7d", "preferred_product_types", "recent_purchases_180d",
)
ARTIFACT_NAMES = (
    "recommendations.parquet", "baseline_recommendations.json",
    "candidate_audit.parquet", "historical_catalogue.parquet",
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _paths(plan):
    def marker(stage):
        return Path(next(s["planned_manifest"] for s in plan["stages"] if s["name"] == stage))
    prediction, model = marker("predict_v1"), marker("fit_model")
    return prediction, prediction.parent / "bundle", model


def _contract(root, plan):
    fitted = verify_fitted_model(root, plan)
    context = load_training_context(root, plan)
    decision = _read(Path(plan["inputs"]["frozen_selection"]))["decision"]
    protocol = load_protocol(root)
    v1, policy = decision["v1_evaluation"], decision["retrieval"]
    start, end = pd.Timestamp(v1["start_inclusive"]), pd.Timestamp(v1["end_exclusive"])
    _require(
        start == protocol["v1_evaluation_start"]
        and end == protocol["confirmation_start"]
        and end - start == pd.Timedelta(days=7)
        and plan["v1_window"] == {"start_inclusive": start.date().isoformat(),
                                    "end_exclusive": end.date().isoformat()}
        and fitted["fitting_boundary_exclusive"] == start.date().isoformat()
        and decision["confirmation_reserved"] is True,
        "Prediction dates must be the frozen V1 window before confirmation.",
    )
    _require(v1["k"] == 12 and policy["fallback_days"] == 28,
             "The shared retrieval contract requires top-12 lists and 28-day fill.")
    _require(
        fitted["cohort_sha256"] == context["cohort_sha256"]
        and fitted["feature_columns"] == context["feature_columns"]
        and fitted["method"] not in BASELINE_NAMES,
        "Prediction cohort, features or method differ from the fitted contract.",
    )
    _, _, model_manifest = _paths(plan)
    identity = {
        "protocol": protocol["protocol_id"],
        "temporal_config_sha256": file_sha256(root / "configs/temporal.yaml"),
        "source_manifest_sha256": file_sha256(root / "docs/data_manifest.json"),
        "cohort_sha256": context["cohort_sha256"],
        "source_sha256": {name: file_sha256(root / "src/atelier_ai" / name) for name in SOURCE_NAMES},
        **policy, "python": platform.python_version(),
        "versions": {"duckdb": duckdb.__version__, "pandas": pd.__version__, "numpy": np.__version__},
        "prepared_manifest_sha256": context["prepared_manifest_sha256"],
        "frozen_selection_sha256": fitted["selection_sha256"],
    }
    recipe = {
        "start_inclusive": start.date().isoformat(), "end_exclusive": end.date().isoformat(),
        "method": fitted["method"], "methods": [*BASELINE_NAMES, fitted["method"]],
        "feature_columns": context["feature_columns"], "input_dtype": "float32",
        "model_manifest_sha256": file_sha256(model_manifest),
        "model_sha256": fitted["artifacts"]["model.txt"]["sha256"],
        "model_mlflow_run_id": fitted["mlflow"]["run_id"],
        "recipients": len(context["customer_ids"]),
        "candidates_per_customer": policy["candidate_budget"],
        "recommendations_per_recipient": v1["k"],
        "target_labels_used_for_prediction": False, "confirmation_evaluated": False,
        "prior_development_exposure": v1["prior_development_exposure"],
        "provenance": identity,
    }
    return recipe, context, fitted


def _export_features(root, plan, recipe, context):
    policy = recipe["provenance"]
    ids = context["customer_ids"]
    historical = build_context(root, ids, recipe["start_inclusive"],
                               prepared_directory=context["prepared_directory"])
    pools, memberships, baseline_rows, audit = retrieve_candidates(
        ids, historical, budget=policy["candidate_budget"],
        history_limit=policy["history_limit"], global_limit=policy["global_limit_7d"],
        preferred_types=policy["preferred_product_types"], per_type=policy["per_preferred_type"],
    )
    tables = prepare_feature_tables(ids, historical, memberships)
    marker, _, _ = _paths(plan)
    exported = export_feature_batches(
        ids, pools, tables, build_checked_feature_batch,
        output_base=marker.parent / "features", feature_columns=recipe["feature_columns"],
        nullable_features=EXPECTED_MISSING_FEATURES, sql_schema=feature_storage_schema(),
        provenance=recipe["provenance"], source_path=Path(__file__),
        candidates_per_customer=recipe["candidates_per_customer"],
        batch_customers=500, dataset_split="test",
    )
    baselines = {name: {customer: values[name] for customer, values in baseline_rows.items()}
                 for name in BASELINE_NAMES}
    catalogue = historical["popularity"][["article_id", "product_group_name"]].copy()
    return exported, baselines, audit, catalogue


def _score_features(directory, manifest, booster, *, columns, k, threads):
    """Score disjoint complete customer batches; break score ties by article ID."""
    parts, scored, score_seconds = [], 0, 0.0
    with duckdb.connect(config={"threads": 2, "memory_limit": "512MB"}) as connection:
        for number, part in enumerate(manifest["parts"], start=1):
            batch = connection.execute("SELECT * FROM read_parquet(?)",
                                       [str(directory / part["file"])]).df()
            _require("label" not in batch.columns, "Prediction features must be unlabelled.")
            X = batch[columns].astype("float32")
            _require(not X.isin([np.inf, -np.inf]).any().any(), "Prediction casting introduced infinity.")
            started = perf_counter()
            scores = booster.predict(X, num_threads=threads, validate_features=True)
            score_seconds += perf_counter() - started
            _require(np.isfinite(scores).all(), "All ranking scores must be finite.")
            top = (batch[["customer_id", "article_id"]].assign(score=scores)
                   .sort_values(["customer_id", "score", "article_id"], ascending=[True, False, True])
                   .groupby("customer_id", sort=False).head(k))
            parts.append(top)
            scored += len(batch)
            if number % 10 == 0 or number == len(manifest["parts"]):
                print(f"Scored partitions: {number}; candidate rows: {scored:,}", flush=True)
    top = (pd.concat(parts, ignore_index=True)
           .sort_values(["customer_id", "score", "article_id"], ascending=[True, False, True])
           .reset_index(drop=True))
    top["rank"] = top.groupby("customer_id", sort=False).cumcount() + 1
    _require(scored == manifest["rows"], "Scored candidate total differs from the export.")
    return top, scored, score_seconds


def _write_frames(directory, frames):
    with duckdb.connect(config={"threads": 2, "memory_limit": "1GB"}) as connection:
        for name, frame in frames.items():
            path = directory / f"{name}.parquet"
            connection.register("_payload", frame)
            connection.sql("SELECT * FROM _payload").write_parquet(str(path), compression="zstd")
            differences = connection.execute(
                "SELECT count(*) FROM ((SELECT * FROM _payload EXCEPT ALL SELECT * FROM read_parquet(?)) "
                "UNION ALL (SELECT * FROM read_parquet(?) EXCEPT ALL SELECT * FROM _payload))",
                [str(path), str(path)],
            ).fetchall()[0][0]
            _require(differences == 0, f"Prediction round trip differs: {name}")
            connection.unregister("_payload")


def _verify_bundle(plan, directory, recipe, context):
    report = _read(directory / "manifest.json")
    _require(
        report.get("schema_version") == 1 and report.get("stage") == "predict_v1"
        and report.get("status") == "complete"
        and all(report.get(k) == v for k, v in recipe.items()),
        "Prediction recipe differs; preserve this bundle and use a new campaign.",
    )
    _require(set(report["artifacts"]) == set(ARTIFACT_NAMES), "Prediction artifact list differs.")
    for name, identity in report["artifacts"].items():
        path = directory / name
        _require(path.is_file() and _identity(path) == identity, f"Prediction checksum differs: {name}")
    for name, digest in recipe["provenance"]["source_sha256"].items():
        _require(file_sha256(directory / "source" / name) == digest, f"Prediction source snapshot differs: {name}")
    marker, _, _ = _paths(plan)
    entry = report["feature_export"]
    feature_directory = (Path(plan["output_root"]) / entry["directory"]).resolve()
    _require(feature_directory.is_relative_to((marker.parent / "features").resolve()),
             "V1 features lie outside this prediction campaign.")
    _require(file_sha256(feature_directory / "manifest.json") == entry["manifest_sha256"],
             "The prediction feature manifest differs.")
    checks = verify_feature_export(feature_directory, expected_fingerprint=entry["fingerprint"])
    feature = _read(feature_directory / "manifest.json")
    _require(
        feature["dataset_split"] == "test"
        and pd.Timestamp(feature["cutoff"]) == pd.Timestamp(recipe["start_inclusive"])
        and feature["feature_columns"] == recipe["feature_columns"]
        and feature["provenance"] == recipe["provenance"]
        and feature["schema"] == [{"name": k, "type": v} for k, v in feature_storage_schema().items()]
        and feature["source_snapshot_sha256"] == recipe["provenance"]["source_sha256"]["pipeline_predict.py"]
        and checks["rows"] == report["candidate_rows"] == recipe["recipients"] * recipe["candidates_per_customer"]
        and checks["customers"] == recipe["recipients"],
        "Prediction features differ from the complete unlabelled V1 contract.",
    )
    ids = set(context["customer_ids"])
    top = _read_parquet(directory / "recommendations.parquet")
    k = recipe["recommendations_per_recipient"]
    _require(
        top.columns.tolist() == ["customer_id", "article_id", "score", "rank"]
        and not top.isna().any().any()
        and set(top["customer_id"]) == ids
        and not top.duplicated(["customer_id", "article_id"]).any()
        and top.groupby("customer_id").size().eq(k).all()
        and len(top) == report["recommendation_rows"] == len(ids) * k
        and np.isfinite(top["score"]).all()
        and top.equals(top.sort_values(["customer_id", "score", "article_id"],
                                      ascending=[True, False, True]).reset_index(drop=True))
        and top["rank"].eq(top.groupby("customer_id", sort=False).cumcount() + 1).all(),
        "Each recipient must retain a distinct, deterministically ranked top-k list.",
    )
    audit = _read_parquet(directory / "candidate_audit.parquet")
    _require(set(audit["customer_id"]) == ids and audit["customer_id"].is_unique
             and audit["candidate_rows"].eq(recipe["candidates_per_customer"]).all(),
             "The prediction audit must preserve the complete cohort.")
    catalogue = _read_parquet(directory / "historical_catalogue.parquet")
    _require(catalogue.columns.tolist() == ["article_id", "product_group_name"]
             and catalogue["article_id"].is_unique and not catalogue.isna().any().any(),
             "The historical catalogue must have unique complete metadata.")
    catalogue_ids = set(catalogue["article_id"])
    baselines = _read(directory / "baseline_recommendations.json")
    _require(set(baselines) == set(BASELINE_NAMES), "Baseline method list differs.")
    for name, lists in baselines.items():
        _require(set(lists) == ids and all(
            len(articles) == len(set(articles)) == k and set(articles) <= catalogue_ids
            for articles in lists.values()), f"Incomplete historical baseline: {name}")
    _require(set(top["article_id"]) <= catalogue_ids, "A recommendation has no historical catalogue entry.")
    paths = [str(feature_directory / part["file"]) for part in feature["parts"]]
    with duckdb.connect(config={"threads": 2, "memory_limit": "1GB"}) as connection:
        connection.register("_recommendations", top)
        absent = connection.execute(
            "SELECT count(*) FROM _recommendations ANTI JOIN read_parquet(?) "
            "USING(customer_id, article_id)", [paths],
        ).fetchall()[0][0]
    _require(absent == 0, "A learned recommendation lies outside its candidate pool.")
    return report


def _tracking_files(directory, report):
    files = {f"v1/predictions/{name}": directory / name for name in (*ARTIFACT_NAMES, "manifest.json")}
    files.update({f"v1/prediction_source/{name}": directory / "source" / name
                  for name in report["provenance"]["source_sha256"]})
    return files


def _track_predictions(root, directory, report, *, write):
    client, _ = _tracking_client(root)
    run_id = report["model_mlflow_run_id"]
    for name, path in _tracking_files(directory, report).items():
        if write:
            client.log_artifact(run_id, str(path), artifact_path=str(Path(name).parent))
    with TemporaryDirectory(prefix="atelier-v1-tracking-") as scratch:
        for name, path in _tracking_files(directory, report).items():
            downloaded = Path(client.download_artifacts(run_id, name, scratch))
            _require(_identity(downloaded) == _identity(path), f"MLflow prediction artifact differs: {name}")


def verify_v1_predictions(project_root, plan):
    """Verify published predictions, their unlabelled export and MLflow artifacts."""
    root = Path(project_root).resolve()
    recipe, context, _ = _contract(root, plan)
    marker, directory, _ = _paths(plan)
    report = _verify_bundle(plan, directory, recipe, context)
    _require(_read(marker) == report, "Published prediction manifests differ.")
    _track_predictions(root, directory, report, write=False)
    return report


def predict_v1(project_root, plan):
    """Generate all recipients' V1 lists without extracting any future targets."""
    root = Path(project_root).resolve()
    marker, directory, model_manifest = _paths(plan)
    if marker.exists():
        report = verify_v1_predictions(root, plan)
        return {"manifest": report, "manifest_path": marker, "reused": True}
    recipe, context, fitted = _contract(root, plan)
    reused = directory.exists()
    if reused:
        report = _verify_bundle(plan, directory, recipe, context)
    else:
        exported, baselines, audit, catalogue = _export_features(root, plan, recipe, context)
        booster = lgb.Booster(model_file=str(model_manifest.parent / "bundle/model.txt"))
        top, scored, seconds = _score_features(
            exported["directory"], exported["manifest"], booster, columns=recipe["feature_columns"],
            k=recipe["recommendations_per_recipient"], threads=fitted["parameters"]["n_jobs"],
        )
        marker.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix="atelier-v1-", dir=marker.parent) as temporary:
            staging = Path(temporary)
            _write_frames(staging, {"recommendations": top, "candidate_audit": audit,
                                    "historical_catalogue": catalogue})
            _publish_json(staging / "baseline_recommendations.json", baselines)
            (staging / "source").mkdir()
            for name in recipe["provenance"]["source_sha256"]:
                shutil.copyfile(root / "src/atelier_ai" / name, staging / "source" / name)
            report = {
                "schema_version": 1, "stage": "predict_v1", "status": "complete", **recipe,
                "candidate_rows": scored, "recommendation_rows": len(top),
                "score_seconds": seconds,
                "feature_export": {
                    "directory": exported["directory"].relative_to(Path(plan["output_root"])).as_posix(),
                    "fingerprint": exported["manifest"]["fingerprint"],
                    "manifest_sha256": file_sha256(exported["directory"] / "manifest.json"),
                },
                "artifacts": {name: _identity(staging / name) for name in ARTIFACT_NAMES},
            }
            _publish_json(staging / "manifest.json", report)
            _verify_bundle(plan, staging, recipe, context)
            staging.rename(directory)
    _track_predictions(root, directory, report, write=True)
    _publish_json(marker, report)
    return {"manifest": report, "manifest_path": marker, "reused": reused}


def main():
    from atelier_ai.pipeline import build_plan

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    plan = build_plan(args.project_root, args.run_id)
    with redirect_stdout(sys.stderr):
        result = predict_v1(args.project_root, plan)
    report = result["manifest"]
    print(json.dumps({
        "stage": report["stage"], "status": report["status"],
        "manifest": str(result["manifest_path"]), "reused": result["reused"],
        **{key: report[key] for key in (
            "start_inclusive", "end_exclusive", "recipients", "candidate_rows", "recommendation_rows",
            "recommendations_per_recipient", "score_seconds", "model_sha256", "model_mlflow_run_id",
            "target_labels_used_for_prediction",
        )},
        "method_count": len(report["methods"]),
    }, indent=2))


if __name__ == "__main__":
    main()
