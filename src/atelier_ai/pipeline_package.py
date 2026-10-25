"""Package verified local V1 artifacts into an independent serving directory."""

import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory

import duckdb
import pandas as pd

from atelier_ai.feature_export import file_sha256
from atelier_ai.pipeline_evaluate import verify_v1_evaluation
from atelier_ai.pipeline_model import _publish_json, _read, _read_parquet, _tracking_client
from atelier_ai.pipeline_predict import _write_frames
from atelier_ai.recommendation_features import FEATURE_COLUMNS, feature_contract, prepare_feature_tables
from atelier_ai.serving_bundle import PAYLOAD_FILES, file_identity, verify_serving_bundle
from atelier_ai.workflow import build_context


PRESENTATION_FIELDS = (
    "article_id", "prod_name", "product_type_name", "product_group_name", "colour_group_name",
    "index_name", "section_name", "garment_group_name", "detail_desc",
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _stage(plan, name):
    return Path(next(s["planned_manifest"] for s in plan["stages"] if s["name"] == name))


def _contract(root, plan):
    evaluation = verify_v1_evaluation(root, plan)
    paths = {"selection": Path(plan["inputs"]["frozen_selection"]),
             "model": _stage(plan, "fit_model"), "predictions": _stage(plan, "predict_v1"),
             "evaluation": _stage(plan, "evaluate_v1")}
    decision = _read(paths["selection"])["decision"]
    model, prediction = _read(paths["model"]), _read(paths["predictions"])
    sources = dict(evaluation["source_sha256"])
    for name in ("pipeline_package.py", "serving_bundle.py"):
        sources[name] = file_sha256(root / "src/atelier_ai" / name)
    recipe = {
        "serving_mode": "precomputed_recommendations", "method": decision["method"],
        "historical_date": prediction["start_inclusive"], "history_days": 180,
        "profile_scope": "frozen_customer_cohort", "model_sha256": model["artifacts"]["model.txt"]["sha256"],
        "model_mlflow_run_id": prediction["model_mlflow_run_id"],
        "num_trees": model["num_trees"], "feature_columns": decision["feature_columns"],
        "cohort_sha256": prediction["provenance"]["cohort_sha256"],
        "recommendations_per_customer": prediction["recommendations_per_recipient"],
        "confirmation_evaluated": False, "input_sha256": {name: file_sha256(path) for name, path in paths.items()},
        "source_sha256": sources, "runtime": model["runtime"],
    }
    return recipe, paths, decision, prediction


def _historical_frames(root, plan, recipe, prediction):
    feature_directory = Path(plan["output_root"]) / prediction["feature_export"]["directory"]
    ids = pd.Index(pd.read_csv(feature_directory / "cohort.csv", dtype="string", keep_default_na=False)["customer_id"])
    prepared = _stage(plan, "prepare_sources").parent
    historical = build_context(root, ids, recipe["historical_date"], prepared_directory=prepared)
    # Serving uses saved rankings; these aggregates describe history and preferences.
    tables = prepare_feature_tables(ids, historical, memberships={})
    profiles = tables["customers"].copy()
    for name in profiles.columns:
        if name.startswith("customer_") and name not in ("customer_id", "customer_recency_days", "customer_has_history"):
            profiles[name] = profiles[name].fillna(0)
    for name in ("customer_history_rows", "customer_distinct_articles", "customer_distinct_product_types",
                 "customer_distinct_colours", "customer_rows_7d", "customer_rows_28d", "customer_rows_90d"):
        profiles[name] = profiles[name].astype("int64")
    profiles["customer_recency_days"] = profiles["customer_recency_days"].astype(float)
    profiles["customer_channel_2_share"] = profiles["customer_channel_2_share"].astype(float)
    history = historical["history"][["t_dat", "customer_id", "article_id", "price", "sales_channel_id"]].copy()
    history = history.rename(columns={"price": "price_source_units"})
    with duckdb.connect(config={"threads": 2, "memory_limit": "1GB"}) as connection:
        metadata = connection.execute("SELECT * FROM read_parquet(?)", [str(prepared / "articles.parquet")]).df()
    for name in PRESENTATION_FIELDS:
        if name not in metadata:
            metadata[name] = pd.Series(None, index=metadata.index, dtype="string")
    catalogue = tables["articles"][["article_id"]].merge(
        metadata[list(PRESENTATION_FIELDS)], on="article_id", how="left", validate="one_to_one")
    prediction_directory = _stage(plan, "predict_v1").parent / "bundle"
    baselines = _read(prediction_directory / "baseline_recommendations.json")
    fallback_ids = next(iter(baselines["recent_popularity_7d"].values()))
    fallback = pd.DataFrame({"article_id": fallback_ids, "rank": range(1, len(fallback_ids) + 1)})
    fallback = fallback.merge(tables["articles"][["article_id", "article_recent_rows", "article_rows_28d"]],
                              on="article_id", how="left", validate="one_to_one")
    fallback["score"] = fallback["article_recent_rows"].astype(float)
    frames = {"customer_profiles": profiles, "customer_history": history,
              "article_aggregates": tables["articles"], "customer_interactions": tables["interactions"],
              "catalogue": catalogue, "fallback": fallback}
    frames.update({f"affinity_{name}": table for name, table in tables["affinities"].items()})
    return frames, feature_directory


def _build_bundle(root, plan, directory, recipe, paths, decision, prediction):
    frames, feature_directory = _historical_frames(root, plan, recipe, prediction)
    _write_frames(directory, frames)
    model_directory, prediction_directory = paths["model"].parent / "bundle", paths["predictions"].parent / "bundle"
    evaluation_directory = paths["evaluation"].parent / "bundle"
    for name in ("model.txt", "reload_check.parquet"):
        shutil.copyfile(model_directory / name, directory / name)
    for name in ("recommendations.parquet", "baseline_recommendations.json", "candidate_audit.parquet"):
        shutil.copyfile(prediction_directory / name, directory / name)
    for name in ("comparison.parquet", "bootstrap_summary.parquet"):
        shutil.copyfile(evaluation_directory / name, directory / name)
    shutil.copyfile(feature_directory / "cohort.csv", directory / "cohort.csv")
    _publish_json(directory / "feature_contract.json", {
        "schema_version": 1, "feature_columns": FEATURE_COLUMNS, "input_dtype": "float32",
        "features": feature_contract.to_dict(orient="records"), "history_days": 180,
        "retrieval": decision["retrieval"], "customer_snapshot_attributes_used": False,
        "stock_availability_known": False, "metadata_timing": "fixed source snapshot; observation date unknown",
        "learned_score_kind": "uncalibrated_lightgbm_ranking_score",
        "fallback_score_kind": "historical_purchase_rows_7d",
        "fallback_order": "7-day popularity, then 28-day popularity fill; article-ID tie breaks",
        "price_units": "unresolved source units", "history_grain": "source purchase row; repeated rows retained",
        "serving_mode": "precomputed_recommendations",
    })
    (directory / "provenance").mkdir()
    for name, path in paths.items():
        shutil.copyfile(path, directory / "provenance" / f"{name}.json")
    (directory / "source").mkdir()
    for name in recipe["source_sha256"]:
        shutil.copyfile(root / "src/atelier_ai" / name, directory / "source" / name)
    filenames = [*PAYLOAD_FILES, *(f"source/{name}" for name in recipe["source_sha256"])]
    report = {
        "schema_version": 1, "stage": "package_service", "status": "complete", **recipe,
        "customers": len(frames["customer_profiles"]), "catalogue_articles": len(frames["catalogue"]),
        "history_rows": len(frames["customer_history"]), "recommendation_rows": prediction["recommendation_rows"],
        "artifacts": {name: file_identity(directory / name) for name in filenames},
    }
    _publish_json(directory / "manifest.json", report)
    verify_serving_bundle(directory)
    return report


def _track_package(root, directory, report, *, write):
    client, _ = _tracking_client(root)
    run_id = report["model_mlflow_run_id"]
    files = {f"serving/{name}": directory / name for name in (*report["artifacts"], "manifest.json")}
    if write:
        for name, path in files.items():
            client.log_artifact(run_id, str(path), artifact_path=str(Path(name).parent))
    with TemporaryDirectory(prefix="atelier-serving-tracking-") as scratch:
        for name, path in files.items():
            downloaded = Path(client.download_artifacts(run_id, name, scratch))
            _require(file_identity(downloaded) == file_identity(path), f"MLflow serving artifact differs: {name}")


def verify_service_package(project_root, plan):
    """Check campaign lineage and tracking in addition to independent bundle checks."""
    root = Path(project_root).resolve()
    recipe, _, _, _ = _contract(root, plan)
    marker = _stage(plan, "package_service")
    directory = marker.parent / "bundle"
    report = verify_serving_bundle(directory)
    _require(all(report.get(k) == v for k, v in recipe.items()), "Serving recipe differs; use a new campaign.")
    _require(_read(marker) == report, "Published serving manifests differ.")
    _track_package(root, directory, report, write=False)
    return report


def package_service(project_root, plan):
    """Build a self-contained historical package without changing scored systems."""
    root = Path(project_root).resolve()
    marker = _stage(plan, "package_service")
    directory = marker.parent / "bundle"
    if marker.exists():
        return {"manifest": verify_service_package(root, plan), "manifest_path": marker, "reused": True}
    recipe, paths, decision, prediction = _contract(root, plan)
    reused = directory.exists()
    if reused:
        report = verify_serving_bundle(directory)
        _require(all(report.get(k) == v for k, v in recipe.items()), "Serving recipe differs; use a new campaign.")
    else:
        marker.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix="atelier-serving-", dir=marker.parent) as temporary:
            staging = Path(temporary)
            report = _build_bundle(root, plan, staging, recipe, paths, decision, prediction)
            staging.rename(directory)
    _track_package(root, directory, report, write=True)
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
        result = package_service(args.project_root, plan)
    report = result["manifest"]
    print(json.dumps({
        "stage": report["stage"], "status": report["status"], "reused": result["reused"],
        "manifest": str(result["manifest_path"]), "bundle": str(result["manifest_path"].parent / "bundle"),
        **{key: report[key] for key in ("serving_mode", "historical_date", "customers", "catalogue_articles",
                                       "history_rows", "recommendation_rows", "model_sha256", "confirmation_evaluated")},
    }, indent=2))


if __name__ == "__main__":
    main()
