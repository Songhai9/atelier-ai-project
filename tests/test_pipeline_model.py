"""Real LightGBM/MLflow checks on a small synthetic training campaign."""

import json
from pathlib import Path

import duckdb
import numpy as np
import pytest

import atelier_ai.pipeline_dataset as dataset
import atelier_ai.pipeline_features as features
import atelier_ai.pipeline_model as model
from atelier_ai.recommendation_features import FEATURE_COLUMNS
from test_pipeline_dataset import campaign  # Reuse the synthetic source fixture.


@pytest.fixture
def model_campaign(campaign, monkeypatch):
    root, plan, run = campaign
    monkeypatch.setenv("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")
    frozen_path = Path(plan["inputs"]["frozen_selection"])
    frozen = json.loads(frozen_path.read_text())
    context = features.load_training_context(root, plan)
    parameters = {
        "objective": "lambdarank", "learning_rate": 0.05, "n_estimators": 8,
        "num_leaves": 63, "min_child_samples": 20, "max_depth": 8,
        "reg_lambda": 1.0, "random_state": 42, "n_jobs": 4,
        "deterministic": True, "force_col_wise": True, "verbosity": -1,
        "metric": "None",
    }
    cutoffs = ["2020-07-22", "2020-07-29"]
    frozen["decision"].update({
        "method": "lightgbm_leaves_63_min_20", "refit_parameters": parameters,
        "feature_columns": FEATURE_COLUMNS,
        "refit_protocol": {"early_stopping": False, "uses_v1_targets": False,
                           "training_cutoffs": cutoffs},
        "v1_evaluation": {"start_inclusive": "2020-08-05", "end_exclusive": "2020-08-12",
                          "k": 12, "prior_development_exposure": True},
        "confirmation_reserved": True,
        "training": {"cohort_sha256": context["cohort_sha256"]},
    })
    frozen_path.write_text(json.dumps(frozen))
    plan.update({
        "run_id": "unit", "method": frozen["decision"]["method"],
        "refit_parameters": parameters, "training_cutoffs": cutoffs,
        "v1_window": {"start_inclusive": "2020-08-05", "end_exclusive": "2020-08-12"},
    })
    plan["stages"].append({"name": "fit_model", "planned_manifest": str(run / "model/manifest.json")})
    dataset.build_training_dataset(root, plan)
    return root, plan, run


def test_frozen_fit_native_reload_tracking_and_verified_reuse(model_campaign, monkeypatch):
    root, plan, run = model_campaign
    before = (run / "training/manifest.json").read_bytes()
    calls = []
    original = model.lgb.LGBMRanker.fit

    def observed_fit(self, X, y, **kwargs):
        calls.append(kwargs)
        assert all(str(dtype) == "float32" for dtype in X.dtypes)
        assert X.columns.tolist() == FEATURE_COLUMNS
        return original(self, X, y, **kwargs)

    monkeypatch.setattr(model.lgb.LGBMRanker, "fit", observed_fit)
    result = model.fit_frozen_model(root, plan)
    report = result["manifest"]
    assert result["reused"] is False
    assert report["training_rows"] == 600 and report["training_groups"] == 2
    assert report["positive_training_rows"] == 2
    assert report["num_trees"] == 8
    assert report["reload_probe_rows"] == 600
    assert report["reload_max_abs_error"] == 0
    assert report["uses_v1_targets"] is False
    assert len(calls) == 1 and set(calls[0]) == {"group"}
    assert np.array_equal(calls[0]["group"], [300, 300])
    assert model.verify_fitted_model(root, plan) == report
    assert (run / "training/manifest.json").read_bytes() == before
    assert not (root / "data/processed/weekly_v1").exists()
    bundle = run / "model/bundle"
    for name in report["source_sha256"]:
        assert (bundle / "source" / name).read_bytes() == (root / "src/atelier_ai" / name).read_bytes()
    client, _ = model._tracking_client(root)
    tracked = client.get_run(report["mlflow"]["run_id"])
    assert tracked.info.status == "FINISHED"
    assert tracked.data.params["n_estimators"] == "8"
    assert tracked.data.tags["pipeline.model_sha256"] == report["artifacts"]["model.txt"]["sha256"]
    marker_before = result["manifest_path"].read_bytes()
    second = model.fit_frozen_model(root, plan)
    assert second["reused"] is True and len(calls) == 1
    assert second["manifest"] == report
    assert result["manifest_path"].read_bytes() == marker_before


@pytest.mark.parametrize("name", ["model.txt", "reload_check.parquet", "source/pipeline_model.py"])
def test_corrupted_model_evidence_is_rejected_and_preserved(model_campaign, name):
    root, plan, run = model_campaign
    result = model.fit_frozen_model(root, plan)
    path = run / "model/bundle" / name
    corrupted = path.read_bytes() + b"\ncorrupted"
    path.write_bytes(corrupted)
    with pytest.raises(ValueError, match="checksum|snapshot differs"):
        model.fit_frozen_model(root, plan)
    assert path.read_bytes() == corrupted
    assert result["manifest_path"].is_file()


def test_input_corruption_is_rejected_before_fitting(model_campaign):
    root, plan, run = model_campaign
    rows = run / "training/dataset/rows.parquet"
    rows.write_bytes(rows.read_bytes() + b"corrupted")
    with pytest.raises(ValueError, match="checksum"):
        model.fit_frozen_model(root, plan)
    assert not (run / "model").exists()


def test_saved_group_order_is_checked_before_fitting(model_campaign):
    root, plan, run = model_campaign
    group_path = run / "training/dataset/groups.parquet"
    reordered = group_path.with_name("reordered.parquet")
    with duckdb.connect() as connection:
        connection.sql("SELECT * FROM read_parquet(?) ORDER BY cutoff DESC",
                       params=[str(group_path)]).write_parquet(str(reordered))
    reordered.replace(group_path)
    manifest = model._read(run / "training/manifest.json")
    manifest["artifacts"]["groups.parquet"] = model._identity(group_path)
    for path in (run / "training/manifest.json", run / "training/dataset/manifest.json"):
        path.write_text(json.dumps(manifest))
    with pytest.raises(AssertionError):
        model.fit_frozen_model(root, plan)
    assert not (run / "model/manifest.json").exists()
    assert not (run / "model/bundle").exists()


def test_interrupted_mlflow_logging_resumes_same_fit_and_run(model_campaign, monkeypatch):
    from mlflow.tracking import MlflowClient

    root, plan, run = model_campaign
    original = MlflowClient.log_artifact

    def interrupt(*args, **kwargs):
        raise RuntimeError("Synthetic tracking interruption")

    monkeypatch.setattr(MlflowClient, "log_artifact", interrupt)
    with pytest.raises(RuntimeError, match="tracking interruption"):
        model.fit_frozen_model(root, plan)
    assert not (run / "model/manifest.json").exists()
    assert (run / "model/bundle/manifest.json").is_file()
    before = (run / "model/bundle/model.txt").read_bytes()
    receipt = model._read(run / "model/tracking.json")
    monkeypatch.setattr(MlflowClient, "log_artifact", original)

    def no_refit(*args):
        raise AssertionError("A verified native model must be reused.")

    monkeypatch.setattr(model, "_fit_bundle", no_refit)
    resumed = model.fit_frozen_model(root, plan)
    assert resumed["reused"] is True
    assert resumed["manifest"]["mlflow"] == receipt
    assert (run / "model/bundle/model.txt").read_bytes() == before
    assert model.verify_fitted_model(root, plan) == resumed["manifest"]


def test_publication_after_finished_tracking_resumes_without_refit(model_campaign, monkeypatch):
    root, plan, run = model_campaign
    publish = model._publish_json

    def interrupt(path, report):
        if path == run / "model/manifest.json":
            raise RuntimeError("Synthetic publication interruption")
        return publish(path, report)

    monkeypatch.setattr(model, "_publish_json", interrupt)
    with pytest.raises(RuntimeError, match="publication interruption"):
        model.fit_frozen_model(root, plan)
    receipt = model._read(run / "model/tracking.json")
    client, _ = model._tracking_client(root)
    assert client.get_run(receipt["run_id"]).info.status == "FINISHED"
    monkeypatch.setattr(model, "_publish_json", publish)
    resumed = model.fit_frozen_model(root, plan)
    assert resumed["reused"] is True
    assert resumed["manifest"]["mlflow"] == receipt


def test_changed_fit_code_preserves_existing_model(model_campaign):
    root, plan, run = model_campaign
    model.fit_frozen_model(root, plan)
    before = (run / "model/manifest.json").read_bytes()
    source = root / "src/atelier_ai/pipeline_model.py"
    source.write_text(source.read_text() + "\n# changed fitting dependency\n")
    with pytest.raises(ValueError, match="Model recipe differs"):
        model.fit_frozen_model(root, plan)
    assert (run / "model/manifest.json").read_bytes() == before


def test_plan_mismatch_cannot_publish_a_model(model_campaign):
    root, plan, run = model_campaign
    plan["v1_window"]["start_inclusive"] = "2020-08-12"
    with pytest.raises(ValueError, match="frozen refit contract"):
        model.fit_frozen_model(root, plan)
    assert not (run / "model").exists()


def test_orchestration_changes_keep_an_identical_frozen_fit_reusable(model_campaign):
    root, plan, run = model_campaign
    first = model.fit_frozen_model(root, plan)
    source = root / "src/atelier_ai/pipeline.py"
    source.write_text(source.read_text() + "\n# another CLI stage\n")
    second = model.fit_frozen_model(root, plan)
    assert second["reused"] is True
    assert second["manifest"] == first["manifest"]
