"""Evaluate immutable V1 predictions against full targets with paired uncertainty."""

import argparse
from contextlib import redirect_stdout
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
from time import time

import duckdb
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from atelier_ai.feature_export import file_sha256
from atelier_ai.pipeline_model import _identity, _publish_json, _read, _read_parquet, _tracking_client
from atelier_ai.pipeline_predict import _write_frames, verify_v1_predictions
from atelier_ai.ranking_metrics import evaluate_recommendations
from atelier_ai.workflow import target_reference


PAIR_KEYS = ["customer_id", "article_id"]
ARTIFACT_NAMES = (
    "source_purchases.parquet", "targets.parquet", "population.parquet",
    "comparison.parquet", "customer_scores.parquet", "bootstrap_samples.parquet",
    "bootstrap_summary.parquet",
)
BOOTSTRAP = {
    "unit": "target buyer", "paired": True, "resamples": 5000, "seed": 42,
    "confidence_level": 0.95, "batch_size": 250, "interval_method": "percentile",
    "scope": "Fixed fitted systems and observed V1 week",
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _paths(plan):
    def marker(name):
        return Path(next(s["planned_manifest"] for s in plan["stages"] if s["name"] == name))
    evaluation, prediction = marker("evaluate_v1"), marker("predict_v1")
    return evaluation, evaluation.parent / "bundle", prediction


def _contract(root, plan):
    prediction = verify_v1_predictions(root, plan)
    selection = Path(plan["inputs"]["frozen_selection"])
    decision = _read(selection)["decision"]
    v1 = decision["v1_evaluation"]
    _require(v1["full_reference_targets"] is True
             and v1["target_metric_population"] == "all target buyers"
             and decision["confirmation_reserved"] is True,
             "V1 must use complete buyer targets and preserve final confirmation.")
    evidence = root / "data/processed/weekly_v1/diagnostics" / decision["method"] / "segmentation_manifest.json"
    digest = file_sha256(evidence)
    _require(digest == decision["evidence_sha256"].get(evidence.relative_to(root).as_posix()),
             "The validation-selected comparator evidence differs from the frozen decision.")
    baseline = _read(evidence)["baseline"]
    _require(baseline in prediction["methods"] and baseline != prediction["method"],
             "The frozen comparator must be one of the predicted reference methods.")
    _, _, prediction_path = _paths(plan)
    feature_directory = Path(plan["output_root"]) / prediction["feature_export"]["directory"]
    ids = pd.Index(pd.read_csv(feature_directory / "cohort.csv", dtype="string",
                              keep_default_na=False)["customer_id"], dtype="string")
    _require(sha256(("\n".join(ids) + "\n").encode()).hexdigest()
             == prediction["provenance"]["cohort_sha256"], "Evaluation cohort identity differs.")
    sources = dict(prediction["provenance"]["source_sha256"])
    for name in ("pipeline_evaluate.py", "ranking_metrics.py"):
        sources[name] = file_sha256(root / "src/atelier_ai" / name)
    recipe = {
        "start_inclusive": prediction["start_inclusive"], "end_exclusive": prediction["end_exclusive"],
        "prediction_manifest_sha256": file_sha256(prediction_path),
        "selection_sha256": file_sha256(selection), "model_sha256": prediction["model_sha256"],
        "model_mlflow_run_id": prediction["model_mlflow_run_id"],
        "method": prediction["method"], "methods": prediction["methods"],
        "baseline_comparator": baseline, "baseline_evidence_sha256": digest,
        "recipients": len(ids), "cohort_sha256": prediction["provenance"]["cohort_sha256"],
        "recommendations_per_recipient": prediction["recommendations_per_recipient"],
        "target_metric_population": "all target buyers", "full_reference_targets": True,
        "prior_development_exposure": prediction["prior_development_exposure"],
        "confirmation_evaluated": False, "bootstrap": dict(BOOTSTRAP),
        "source_sha256": sources, "versions": dict(prediction["provenance"]["versions"]),
    }
    return recipe, prediction, ids


def paired_map_bootstrap(customer_scores, methods, model_method, baseline_method, *,
                         resamples=5000, seed=42, batch_size=250, confidence_level=0.95):
    """Resample the same sorted target buyers for all methods, including zero hits."""
    _require(len(set(methods)) == len(methods) and model_method in methods
             and baseline_method in methods and model_method != baseline_method,
             "Bootstrap method identities must be distinct and complete.")
    _require(type(resamples) is int and resamples > 0 and type(batch_size) is int
             and batch_size > 0 and 0 < confidence_level < 1, "Invalid bootstrap settings.")
    _require(set(customer_scores["method"]) == set(methods), "Bootstrap methods differ from predictions.")
    wide = (customer_scores.pivot(index="customer_id", columns="method", values="ap_at_k")
            .reindex(columns=methods).sort_index())
    _require(not wide.empty and not wide.isna().any().any()
             and wide.ge(0).all().all() and wide.le(1).all().all(),
             "Bootstrap requires the same complete buyer population for every method.")
    values = wide.to_numpy(dtype=float)
    observed = values.mean(axis=0)
    rng = np.random.default_rng(seed)
    means = np.empty((resamples, len(methods)))
    for offset in range(0, resamples, batch_size):
        stop = min(offset + batch_size, resamples)
        indices = rng.integers(0, len(wide), size=(stop - offset, len(wide)))
        means[offset:stop] = values[indices].mean(axis=1)
    samples = pd.DataFrame(means, columns=methods)
    samples["model_minus_baseline"] = samples[model_method] - samples[baseline_method]
    alpha = 1 - confidence_level
    summary = pd.DataFrame({
        "method": [*methods, "model_minus_baseline"],
        "metric": ["map_at_12"] * len(methods) + ["delta_map_at_12"],
        "observed": [*observed, wide[model_method].mean() - wide[baseline_method].mean()],
        "ci_lower": samples.quantile(alpha / 2).to_numpy(),
        "ci_upper": samples.quantile(1 - alpha / 2).to_numpy(),
        "evaluated_buyers": len(wide), "bootstrap_resamples": resamples,
    })
    return samples, summary


def _methods(prediction_directory, recipe):
    methods = _read(prediction_directory / "baseline_recommendations.json")
    top = _read_parquet(prediction_directory / "recommendations.parquet")
    methods[recipe["method"]] = (top.sort_values(["customer_id", "rank"])
                                  .groupby("customer_id", sort=False)["article_id"].agg(list).to_dict())
    return {name: methods[name] for name in recipe["methods"]}


def _population(ids, targets):
    population = pd.DataFrame({"customer_id": ids})
    counts = (targets.groupby("customer_id", as_index=False)
              .agg(target_articles=("article_id", "size"), candidate_target_articles=("in_candidates", "sum")))
    population = population.merge(counts, how="left", on="customer_id", validate="one_to_one", sort=False)
    population[["target_articles", "candidate_target_articles"]] = (
        population[["target_articles", "candidate_target_articles"]].fillna(0).astype("int64"))
    population["evaluated"] = population["target_articles"].gt(0)
    return population


def _metric_tables(prediction_directory, recipe, ids, targets):
    reference = targets.groupby("customer_id")["article_id"].agg(set).to_dict()
    methods = _methods(prediction_directory, recipe)
    catalogue = _read_parquet(prediction_directory / "historical_catalogue.parquet")
    group_map = catalogue.set_index("article_id")["product_group_name"].to_dict()
    coverage = int(targets["in_candidates"].sum()) / len(targets)
    comparisons, scores = [], []
    for name, lists in methods.items():
        result = evaluate_recommendations(lists, reference, ids, k=recipe["recommendations_per_recipient"])
        scores.append(result["customer_scores"].assign(method=name))
        comparisons.append({
            "method": name, **result["summary"], "candidate_coverage": coverage,
            "catalogue_coverage": len({article for articles in lists.values() for article in articles}) / len(catalogue),
            "mean_recommended_product_groups": float(np.mean([
                len({group_map[article] for article in articles}) for articles in lists.values()])),
        })
    comparison, scores = pd.DataFrame(comparisons), pd.concat(scores, ignore_index=True)
    config = recipe["bootstrap"]
    samples, summary = paired_map_bootstrap(
        scores, recipe["methods"], recipe["method"], recipe["baseline_comparator"],
        resamples=config["resamples"], seed=config["seed"], batch_size=config["batch_size"],
        confidence_level=config["confidence_level"],
    )
    return {"population": _population(ids, targets), "comparison": comparison,
            "customer_scores": scores, "bootstrap_samples": samples, "bootstrap_summary": summary}


def _calculate(root, plan, recipe, prediction, ids):
    prepared = Path(next(s["planned_manifest"] for s in plan["stages"]
                         if s["name"] == "prepare_sources")).parent
    purchases, full_targets = target_reference(root, ids, recipe["start_inclusive"], prepared_directory=prepared)
    _require(not full_targets.empty, "V1 evaluation requires at least one observed target buyer.")
    feature_directory = Path(plan["output_root"]) / prediction["feature_export"]["directory"]
    feature = _read(feature_directory / "manifest.json")
    paths = [str(feature_directory / part["file"]) for part in feature["parts"]]
    with duckdb.connect(config={"threads": 2, "memory_limit": "1GB"}) as connection:
        connection.register("_full_targets", full_targets)
        targets = connection.execute(
            "SELECT t.*, EXISTS (SELECT 1 FROM read_parquet(?) c "
            "WHERE c.customer_id=t.customer_id AND c.article_id=t.article_id) AS in_candidates "
            "FROM _full_targets t ORDER BY t.customer_id,t.article_id", [paths],
        ).df()
    _require(len(targets) == len(full_targets), "Candidate annotation lost full-reference targets.")
    _, _, prediction_path = _paths(plan)
    return {"source_purchases": purchases, "targets": targets,
            **_metric_tables(prediction_path.parent / "bundle", recipe, ids, targets)}


def _verify_bundle(plan, directory, recipe, ids):
    report = _read(directory / "manifest.json")
    _require(report.get("schema_version") == 1 and report.get("stage") == "evaluate_v1"
             and report.get("status") == "complete"
             and all(report.get(k) == v for k, v in recipe.items()),
             "Evaluation recipe differs; preserve this bundle and use a new campaign.")
    _require(set(report["artifacts"]) == set(ARTIFACT_NAMES), "Evaluation artifact list differs.")
    for name, identity in report["artifacts"].items():
        path = directory / name
        _require(path.is_file() and _identity(path) == identity, f"Evaluation checksum differs: {name}")
    for name, digest in recipe["source_sha256"].items():
        _require(file_sha256(directory / "source" / name) == digest, f"Evaluation source snapshot differs: {name}")
    purchases = _read_parquet(directory / "source_purchases.parquet")
    targets = _read_parquet(directory / "targets.parquet")
    _require(len(purchases) == report["source_purchase_rows"] and not purchases.empty
             and purchases["customer_id"].isin(ids).all()
             and purchases["t_dat"].ge(pd.Timestamp(recipe["start_inclusive"])).all()
             and purchases["t_dat"].lt(pd.Timestamp(recipe["end_exclusive"])).all(),
             "Source purchases must remain strictly within the declared V1 window.")
    reference = purchases[PAIR_KEYS].drop_duplicates().sort_values(PAIR_KEYS).reset_index(drop=True)
    assert_frame_equal(reference, targets[PAIR_KEYS], check_dtype=False, check_exact=True)
    _require(targets["in_candidates"].isin([True, False]).all()
             and len(targets) == report["target_pairs"]
             and int(targets["in_candidates"].sum()) == report["candidate_target_pairs"]
             and targets["customer_id"].nunique() == report["target_buyers"]
             and len(ids) - report["target_buyers"] == report["excluded_nonbuyers"],
             "Full target or buyer denominators differ from the evaluation manifest.")
    _, _, prediction_path = _paths(plan)
    expected = _metric_tables(prediction_path.parent / "bundle", recipe, ids, targets)
    for name, frame in expected.items():
        assert_frame_equal(frame, _read_parquet(directory / f"{name}.parquet"),
                           check_dtype=False, check_exact=True)
    return report


def _tracking_metrics(directory, report):
    selected = _read_parquet(directory / "comparison.parquet").set_index("method").loc[report["method"]]
    delta = (_read_parquet(directory / "bootstrap_summary.parquet")
             .set_index("method").loc["model_minus_baseline"])
    return {
        "v1_map_at_12": float(selected["map_at_k"]),
        "v1_recall_at_12": float(selected["recall_at_k"]),
        "v1_ndcg_at_12": float(selected["ndcg_at_k"]),
        "v1_delta_map": float(delta["observed"]),
        "v1_delta_map_ci_lower": float(delta["ci_lower"]),
        "v1_delta_map_ci_upper": float(delta["ci_upper"]),
    }


def _track_evaluation(root, directory, report, *, write):
    from mlflow.entities import Metric

    client, _ = _tracking_client(root)
    run_id = report["model_mlflow_run_id"]
    files = {f"v1/evaluation/{name}": directory / name for name in (*ARTIFACT_NAMES, "manifest.json")}
    files.update({f"v1/evaluation_source/{name}": directory / "source" / name for name in report["source_sha256"]})
    metrics = _tracking_metrics(directory, report)
    if write:
        for name, path in files.items():
            client.log_artifact(run_id, str(path), artifact_path=str(Path(name).parent))
        client.log_batch(run_id, metrics=[Metric(name, value, int(time() * 1000), 0)
                                           for name, value in metrics.items()])
    tracked = client.get_run(run_id)
    _require(all(tracked.data.metrics.get(k) == v for k, v in metrics.items()), "MLflow V1 metrics differ.")
    with TemporaryDirectory(prefix="atelier-evaluation-tracking-") as scratch:
        for name, path in files.items():
            downloaded = Path(client.download_artifacts(run_id, name, scratch))
            _require(_identity(downloaded) == _identity(path), f"MLflow evaluation artifact differs: {name}")


def verify_v1_evaluation(project_root, plan):
    """Verify all inputs, full target denominators, metric replay and tracking."""
    root = Path(project_root).resolve()
    recipe, _, ids = _contract(root, plan)
    marker, directory, _ = _paths(plan)
    report = _verify_bundle(plan, directory, recipe, ids)
    _require(_read(marker) == report, "Published evaluation manifests differ.")
    _track_evaluation(root, directory, report, write=False)
    return report


def evaluate_v1(project_root, plan):
    """Evaluate saved lists; publish only after the complete evaluation is tracked."""
    root = Path(project_root).resolve()
    marker, directory, _ = _paths(plan)
    if marker.exists():
        return {"manifest": verify_v1_evaluation(root, plan), "manifest_path": marker, "reused": True}
    recipe, prediction, ids = _contract(root, plan)
    reused = directory.exists()
    if reused:
        report = _verify_bundle(plan, directory, recipe, ids)
    else:
        frames = _calculate(root, plan, recipe, prediction, ids)
        marker.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix="atelier-evaluation-", dir=marker.parent) as temporary:
            staging = Path(temporary)
            _write_frames(staging, frames)
            (staging / "source").mkdir()
            for name in recipe["source_sha256"]:
                shutil.copyfile(root / "src/atelier_ai" / name, staging / "source" / name)
            targets = frames["targets"]
            report = {
                "schema_version": 1, "stage": "evaluate_v1", "status": "complete", **recipe,
                "source_purchase_rows": len(frames["source_purchases"]), "target_pairs": len(targets),
                "candidate_target_pairs": int(targets["in_candidates"].sum()),
                "target_buyers": int(targets["customer_id"].nunique()),
                "excluded_nonbuyers": len(ids) - int(targets["customer_id"].nunique()),
                "artifacts": {name: _identity(staging / name) for name in ARTIFACT_NAMES},
            }
            _publish_json(staging / "manifest.json", report)
            _verify_bundle(plan, staging, recipe, ids)
            staging.rename(directory)
    _track_evaluation(root, directory, report, write=True)
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
        result = evaluate_v1(args.project_root, plan)
    report = result["manifest"]
    print(json.dumps({
        "stage": report["stage"], "status": report["status"],
        "manifest": str(result["manifest_path"]), "reused": result["reused"],
        **{key: report[key] for key in (
            "recipients", "source_purchase_rows", "target_pairs", "target_buyers",
            "excluded_nonbuyers", "candidate_target_pairs", "baseline_comparator", "confirmation_evaluated",
        )},
        **_tracking_metrics(result["manifest_path"].parent / "bundle", report),
    }, indent=2))


if __name__ == "__main__":
    main()
