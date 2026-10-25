"""Portable serving, historical profiles, fallback and interruption checks."""

import json
from pathlib import Path
import shutil

import duckdb
import pytest

import atelier_ai.pipeline_evaluate as evaluate
import atelier_ai.pipeline_package as package
from atelier_ai.serving_bundle import ServingBundle, file_identity, read_table, verify_serving_bundle
from test_pipeline_evaluate import evaluation_campaign


@pytest.fixture
def serving_campaign(evaluation_campaign):
    root, plan, run = evaluation_campaign
    plan["stages"].append({"name": "package_service", "planned_manifest": str(run / "serving/manifest.json")})
    evaluate.evaluate_v1(root, plan)
    return root, plan, run


def _unreachable(*args, **kwargs):
    raise AssertionError("An existing serving package must not rebuild historical data.")


def test_package_preserves_frozen_system_and_excludes_future_profile_purchases(serving_campaign, monkeypatch):
    root, plan, run = serving_campaign
    upstream = {name: (run / name / "manifest.json").read_bytes()
                for name in ("training", "model", "predictions", "evaluation")}
    result = package.package_service(root, plan)
    report = result["manifest"]
    assert result["reused"] is False
    assert report["serving_mode"] == "precomputed_recommendations"
    assert report["customers"] == 3 and report["recommendation_rows"] == 36
    assert report["catalogue_articles"] == 302 and report["history_rows"] == 6
    assert report["confirmation_evaluated"] is False
    bundle = run / "serving/bundle"
    assert "0000000303" not in set(read_table(bundle, "catalogue")["article_id"])
    service = ServingBundle(bundle)
    profile = service.profile("u")
    assert profile["historical_date"] == "2020-08-05"
    assert profile["summary"]["customer_history_rows"] == 6
    assert len(profile["recent_purchase_rows"]) == 6
    assert all(row["t_dat"] < "2020-08-05" for row in profile["recent_purchase_rows"])
    assert sum(row["article_id"] == "0000000001" for row in profile["recent_purchase_rows"]) == 3
    cold = service.profile("cold")
    assert cold["summary"]["customer_has_history"] is False
    assert cold["summary"]["customer_history_rows"] == 0
    assert cold["summary"]["customer_recency_days"] is None
    assert cold["recent_purchase_rows"] == []
    assert [row["customer_id"] for row in service.customers(limit=2)] == ["cold", "quiet"]
    assert [row["customer_id"] for row in service.customers(offset=2, limit=2)] == ["u"]
    for customer in ("u", "cold", "quiet"):
        response = service.recommendations(customer)
        assert response["strategy"] == report["method"]
        assert response["score_kind"] == "uncalibrated_lightgbm_ranking_score"
        assert len(response["items"]) == 12
        assert [item["rank"] for item in response["items"]] == list(range(1, 13))
        assert len({item["article_id"] for item in response["items"]}) == 12
        assert all(item["prod_name"] is None for item in response["items"])
        json.dumps(response, allow_nan=False)
    for method, lists in service.baselines.items():
        response = service.recommendations("u", method=method)
        assert response["strategy"] == method and response["score_kind"] == "baseline_rank_only"
        assert [item["article_id"] for item in response["items"]] == lists["u"]
        assert all(item["score"] is None for item in response["items"])
        json.dumps(response, allow_nan=False)
    with pytest.raises(ValueError, match="Unknown recommendation method"):
        service.recommendations("u", method="undefined")
    assert package.verify_service_package(root, plan) == report
    for name, content in upstream.items():
        assert (run / name / "manifest.json").read_bytes() == content
    monkeypatch.setattr(package, "_historical_frames", _unreachable)
    again = package.package_service(root, plan)
    assert again["reused"] is True and again["manifest"] == report


def test_detached_bundle_works_without_original_project_or_mlflow(serving_campaign, monkeypatch):
    root, plan, run = serving_campaign
    result = package.package_service(root, plan)
    detached = root.parent / f"{root.name}-detached"
    offline = root.parent / f"{root.name}-offline"
    shutil.copytree(run / "serving/bundle", detached)
    root.rename(offline)
    monkeypatch.setattr(package, "_contract", _unreachable)
    monkeypatch.setattr(package, "_tracking_client", _unreachable)
    try:
        service = ServingBundle(detached)
        assert service.manifest == result["manifest"]
        assert service.profile("u")["summary"]["customer_history_rows"] == 6
        assert service.profile("unknown") is None
        response = service.recommendations("unknown")
        assert response["strategy"] == "popularity_fallback_unknown_customer"
        assert response["score_kind"] == "historical_purchase_rows_7d"
        assert response["model_sha256"] is None
        assert response["historical_date"] == "2020-08-05"
        assert len(response["items"]) == 12
        assert [item["article_id"] for item in response["items"]] == read_table(detached, "fallback")["article_id"].tolist()
        json.dumps(response, allow_nan=False)
    finally:
        offline.rename(root)


@pytest.mark.parametrize("name", ["model.txt", "catalogue.parquet", "source/serving_bundle.py"])
def test_corrupted_bundle_is_rejected_and_preserved(serving_campaign, name):
    root, plan, run = serving_campaign
    package.package_service(root, plan)
    bundle = run / "serving/bundle"
    path = bundle / name
    damaged = path.read_bytes() + b"damaged"
    path.write_bytes(damaged)
    with pytest.raises(ValueError, match="checksum"):
        ServingBundle(bundle)
    assert path.read_bytes() == damaged


def test_future_history_rows_are_rejected_even_with_updated_checksums(serving_campaign):
    root, plan, run = serving_campaign
    package.package_service(root, plan)
    bundle = run / "serving/bundle"
    history = bundle / "customer_history.parquet"
    wrong = bundle / "rewritten.parquet"
    with duckdb.connect() as connection:
        connection.sql("SELECT CAST(? AS DATE) AS t_dat, * EXCLUDE(t_dat) FROM read_parquet(?)",
                       params=["2020-08-05", str(history)]).write_parquet(str(wrong))
    wrong.replace(history)
    manifest = json.loads((bundle / "manifest.json").read_text())
    manifest["artifacts"]["customer_history.parquet"] = file_identity(history)
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="outside the historical serving scope"):
        verify_serving_bundle(bundle)


def test_unlisted_data_file_is_rejected(serving_campaign):
    root, plan, run = serving_campaign
    package.package_service(root, plan)
    bundle = run / "serving/bundle"
    (bundle / "targets.parquet").write_bytes(b"not part of serving")
    with pytest.raises(ValueError, match="unexpected or missing"):
        verify_serving_bundle(bundle)


def test_tracking_interruption_reuses_complete_serving_bundle(serving_campaign, monkeypatch):
    from mlflow.tracking import MlflowClient

    root, plan, run = serving_campaign
    log = MlflowClient.log_artifact

    def interrupt(*args, **kwargs):
        raise RuntimeError("Synthetic serving tracking interruption")

    monkeypatch.setattr(MlflowClient, "log_artifact", interrupt)
    with pytest.raises(RuntimeError, match="serving tracking interruption"):
        package.package_service(root, plan)
    assert not (run / "serving/manifest.json").exists()
    before = (run / "serving/bundle/manifest.json").read_bytes()
    monkeypatch.setattr(MlflowClient, "log_artifact", log)
    monkeypatch.setattr(package, "_historical_frames", _unreachable)
    result = package.package_service(root, plan)
    assert result["reused"] is True
    assert (run / "serving/bundle/manifest.json").read_bytes() == before
    assert package.verify_service_package(root, plan) == result["manifest"]


def test_changed_packaging_source_requires_a_new_campaign(serving_campaign):
    root, plan, run = serving_campaign
    package.package_service(root, plan)
    before = (run / "serving/manifest.json").read_bytes()
    source = root / "src/atelier_ai/serving_bundle.py"
    source.write_text(source.read_text() + "\n# changed serving dependency\n")
    with pytest.raises(ValueError, match="Serving recipe differs"):
        package.package_service(root, plan)
    assert (run / "serving/manifest.json").read_bytes() == before
