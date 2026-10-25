"""Synthetic V1 inference, temporal isolation, complete populations and reuse."""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

import atelier_ai.pipeline_features as features
import atelier_ai.pipeline_model as model
import atelier_ai.pipeline_predict as predict
import atelier_ai.workflow as workflow
from atelier_ai.recommendation_features import FEATURE_COLUMNS
from test_pipeline_model import campaign, model_campaign


@pytest.fixture
def prediction_campaign(model_campaign, monkeypatch):
    root, plan, run = model_campaign
    plan["stages"].append({"name": "predict_v1", "planned_manifest": str(run / "predictions/manifest.json")})
    monkeypatch.setattr(predict, "load_training_context", features.load_training_context)
    model.fit_frozen_model(root, plan)
    return root, plan, run


def _unreachable(*args, **kwargs):
    raise AssertionError("This operation must not run in the prediction stage.")


def test_prediction_keeps_cold_customers_and_matches_direct_scores_without_targets(prediction_campaign, monkeypatch):
    root, plan, run = prediction_campaign
    before = {name: (run / name).read_bytes() for name in ("model/manifest.json", "training/manifest.json")}
    monkeypatch.setattr(workflow, "target_reference", _unreachable)
    monkeypatch.setattr(features, "target_reference", _unreachable)
    result = predict.predict_v1(root, plan)
    report = result["manifest"]
    assert result["reused"] is False
    assert report["recipients"] == 2 and report["candidate_rows"] == 600
    assert report["recommendation_rows"] == 24 and report["recommendations_per_recipient"] == 12
    assert report["target_labels_used_for_prediction"] is False
    assert report["confirmation_evaluated"] is False
    assert report["methods"] == [*predict.BASELINE_NAMES, "lightgbm_leaves_63_min_20"]
    directory = Path(plan["output_root"]) / report["feature_export"]["directory"]
    manifest = model._read(directory / "manifest.json")
    assert manifest["dataset_split"] == "test"
    assert "label" not in [column["name"] for column in manifest["schema"]]
    with duckdb.connect() as connection:
        batch = connection.execute("SELECT * FROM read_parquet(?)", [
            [str(directory / part["file"]) for part in manifest["parts"]],
        ]).df()
    assert batch.loc[batch["customer_id"].eq("cold"), "customer_has_history"].eq(False).all()
    booster = predict.lgb.Booster(model_file=str(run / "model/bundle/model.txt"))
    expected = (batch[["customer_id", "article_id"]]
                .assign(score=booster.predict(batch[FEATURE_COLUMNS].astype("float32"),
                                             num_threads=4, validate_features=True))
                .sort_values(["customer_id", "score", "article_id"], ascending=[True, False, True])
                .groupby("customer_id", sort=False).head(12).reset_index(drop=True))
    expected["rank"] = expected.groupby("customer_id", sort=False).cumcount() + 1
    actual = model._read_parquet(run / "predictions/bundle/recommendations.parquet")
    assert_frame_equal(actual, expected, check_dtype=False)
    assert set(actual["customer_id"]) == {"u", "cold"}
    for name, content in before.items():
        assert (run / name).read_bytes() == content
    assert not list((run / "predictions").rglob("targets.*"))
    assert not (root / "data/processed/weekly_v1").exists()
    assert predict.verify_v1_predictions(root, plan) == report
    monkeypatch.setattr(predict, "_export_features", _unreachable)
    monkeypatch.setattr(predict, "_score_features", _unreachable)
    second = predict.predict_v1(root, plan)
    assert second["reused"] is True and second["manifest"] == report


@pytest.mark.parametrize("name", ["recommendations.parquet", "baseline_recommendations.json", "source/pipeline_predict.py"])
def test_corrupted_prediction_evidence_is_rejected_and_preserved(prediction_campaign, name):
    root, plan, run = prediction_campaign
    predict.predict_v1(root, plan)
    path = run / "predictions/bundle" / name
    damaged = path.read_bytes() + b"corrupted"
    path.write_bytes(damaged)
    with pytest.raises(ValueError, match="checksum|snapshot differs"):
        predict.predict_v1(root, plan)
    assert path.read_bytes() == damaged


def test_interrupted_scoring_reuses_complete_feature_export(prediction_campaign, monkeypatch):
    root, plan, run = prediction_campaign
    score = predict._score_features

    def interrupt(*args, **kwargs):
        raise RuntimeError("Synthetic scoring interruption")

    monkeypatch.setattr(predict, "_score_features", interrupt)
    with pytest.raises(RuntimeError, match="scoring interruption"):
        predict.predict_v1(root, plan)
    part = next((run / "predictions/features").rglob("part-*.parquet"))
    before = part.stat().st_mtime_ns
    assert not (run / "predictions/bundle").exists()
    assert not (run / "predictions/manifest.json").exists()
    monkeypatch.setattr(predict, "_score_features", score)
    monkeypatch.setattr(predict, "build_checked_feature_batch", _unreachable)
    result = predict.predict_v1(root, plan)
    assert result["manifest"]["recommendation_rows"] == 24
    assert part.stat().st_mtime_ns == before


def test_interrupted_prediction_tracking_resumes_without_scoring(prediction_campaign, monkeypatch):
    from mlflow.tracking import MlflowClient

    root, plan, run = prediction_campaign
    log = MlflowClient.log_artifact

    def interrupt(*args, **kwargs):
        raise RuntimeError("Synthetic prediction tracking interruption")

    monkeypatch.setattr(MlflowClient, "log_artifact", interrupt)
    with pytest.raises(RuntimeError, match="prediction tracking interruption"):
        predict.predict_v1(root, plan)
    assert not (run / "predictions/manifest.json").exists()
    before = (run / "predictions/bundle/recommendations.parquet").read_bytes()
    monkeypatch.setattr(MlflowClient, "log_artifact", log)
    monkeypatch.setattr(predict, "_export_features", _unreachable)
    monkeypatch.setattr(predict, "_score_features", _unreachable)
    result = predict.predict_v1(root, plan)
    assert result["reused"] is True
    assert (run / "predictions/bundle/recommendations.parquet").read_bytes() == before
    assert predict.verify_v1_predictions(root, plan) == result["manifest"]


def test_changed_prediction_source_requires_another_campaign(prediction_campaign):
    root, plan, run = prediction_campaign
    predict.predict_v1(root, plan)
    source = root / "src/atelier_ai/pipeline_predict.py"
    source.write_text(source.read_text() + "\n# changed prediction dependency\n")
    with pytest.raises(ValueError, match="Prediction recipe differs"):
        predict.predict_v1(root, plan)


def test_confirmation_window_is_rejected_before_prediction(prediction_campaign):
    root, plan, run = prediction_campaign
    plan["v1_window"]["end_exclusive"] = "2020-08-19"
    with pytest.raises(ValueError, match="frozen V1 window"):
        predict.predict_v1(root, plan)
    assert not (run / "predictions").exists()


@pytest.mark.parametrize("nonfinite", [False, True])
def test_score_ties_use_article_ids_and_nonfinite_scores_fail(tmp_path, nonfinite):
    # Reverse candidate order deliberately; every tied score must use article-ID order.
    batch = pd.DataFrame([(customer, f"{i:010d}", 0.0)
                          for customer in ("u", "cold") for i in range(300, 0, -1)],
                         columns=["customer_id", "article_id", "x"])
    with duckdb.connect() as connection:
        connection.register("_batch", batch)
        connection.sql("SELECT * FROM _batch").write_parquet(str(tmp_path / "part.parquet"))

    class TiedBooster:
        def predict(self, X, **kwargs):
            assert kwargs["validate_features"] is True
            return np.full(len(X), np.nan if nonfinite else 0.0)

    manifest = {"parts": [{"file": "part.parquet"}], "rows": len(batch)}
    if nonfinite:
        with pytest.raises(ValueError, match="finite"):
            predict._score_features(tmp_path, manifest, TiedBooster(), columns=["x"], k=12, threads=1)
    else:
        result, scored, _ = predict._score_features(tmp_path, manifest, TiedBooster(),
                                                   columns=["x"], k=12, threads=1)
        assert scored == 600
        for _, group in result.groupby("customer_id"):
            assert group["article_id"].tolist() == [f"{i:010d}" for i in range(1, 13)]
            assert group["rank"].tolist() == list(range(1, 13))
