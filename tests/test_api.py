"""HTTP contracts over a real synthetic serving bundle, without raw-data serving."""

from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import shutil

from fastapi.testclient import TestClient
import pytest

import atelier_ai.api as api
import atelier_ai.pipeline_package as package
from atelier_ai.api_models import ErrorResponse, RecommendationsResponse
from test_pipeline_package import evaluation_campaign, serving_campaign


@pytest.fixture
def api_bundle(serving_campaign):
    root, plan, run = serving_campaign
    package.package_service(root, plan)
    return root, run / "serving/bundle"


def identities(directory):
    return {path.relative_to(directory).as_posix(): sha256(path.read_bytes()).hexdigest()
            for path in directory.rglob("*") if path.is_file()}


def test_all_five_routes_pagination_methods_and_read_only_loading(api_bundle, monkeypatch):
    root, directory = api_bundle
    before = identities(directory)
    loader = api.ServingBundle
    loads = []

    def load(path):
        loads.append(path)
        return loader(path)

    monkeypatch.setattr(api, "ServingBundle", load)
    application = api.create_app(directory)
    assert loads == []
    with TestClient(application) as client:
        assert len(loads) == 1
        assert client.get("/health").json() == {"api_version": "0.1.0", "status": "ok"}
        ready = client.get("/ready")
        assert ready.status_code == 200
        state = ready.json()
        assert state["status"] == "ready" and state["customers"] == 3
        assert state["historical_date"] == "2020-08-05"
        assert state["recommendations_per_customer"] == 12 and len(state["methods"]) == 4
        assert str(root) not in ready.text and "tracking_uri" not in ready.text
        page = client.get("/customers", params={"limit": 2}).json()
        assert page["total"] == 3 and page["next_offset"] == 2
        assert [row["customer_id"] for row in page["items"]] == ["cold", "quiet"]
        last = client.get("/customers", params={"offset": page["next_offset"], "limit": 2}).json()
        assert [row["customer_id"] for row in last["items"]] == ["u"]
        assert last["next_offset"] is None
        empty = client.get("/customers", params={"offset": 100}).json()
        assert empty["items"] == [] and empty["next_offset"] is None
        profile = client.get("/customers/u/profile")
        assert profile.status_code == 200
        profile = profile.json()
        assert profile["summary"]["customer_history_rows"] == 6
        assert len(profile["recent_purchase_rows"]) == 6
        assert all(row["t_dat"] < "2020-08-05" for row in profile["recent_purchase_rows"])
        assert profile["bundle_version"] == state["bundle_version"]
        cold = client.get("/customers/cold/profile").json()
        assert cold["summary"]["customer_has_history"] is False
        assert cold["summary"]["customer_recency_days"] is None
        assert cold["recent_purchase_rows"] == []
        response = client.get("/recommendations", params={"customer_id": "u"})
        assert response.status_code == 200
        learned = RecommendationsResponse.model_validate(response.json())
        assert learned.strategy == state["default_method"]
        assert learned.model_sha256 == state["model_sha256"]
        assert learned.score_kind == "uncalibrated_lightgbm_ranking_score"
        assert [item.rank for item in learned.items] == list(range(1, 13))
        assert len({item.article_id for item in learned.items}) == 12
        assert all(item.article_id.startswith("0") and item.prod_name is None for item in learned.items)
        for method in state["methods"][1:]:
            baseline = client.get("/recommendations", params={"customer_id": "u", "method": method})
            assert baseline.status_code == 200
            result = baseline.json()
            assert result["strategy"] == method and result["score_kind"] == "baseline_rank_only"
            assert result["model_sha256"] is None and all(row["score"] is None for row in result["items"])
        assert client.get("/recommendations", params={"customer_id": "cold"}).json()["strategy"] == state["default_method"]
        requests = ["/recommendations?customer_id=u", "/customers/u/profile"] * 4
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(client.get, requests))
        assert all(result.status_code == 200 for result in results)
        assert all(result.json() == response.json() for result in results[::2])
        assert len(loads) == 1
    assert application.state.bundle is None
    assert identities(directory) == before


def test_unknown_ids_and_invalid_requests_have_explicit_http_semantics(api_bundle):
    _, directory = api_bundle
    with TestClient(api.create_app(directory)) as client:
        profile = client.get("/customers/unknown/profile")
        assert profile.status_code == 404
        assert ErrorResponse.model_validate(profile.json()).detail.code == "customer_not_found"
        fallback = client.get("/recommendations", params={"customer_id": "unknown"})
        assert fallback.status_code == 200
        result = RecommendationsResponse.model_validate(fallback.json())
        assert result.strategy == "popularity_fallback_unknown_customer"
        assert result.score_kind == "historical_purchase_rows_7d"
        assert result.model_sha256 is None and len(result.items) == 12
        invalid = [
            ("/customers", {"limit": 0}), ("/customers", {"limit": 101}),
            ("/customers", {"offset": -1}), ("/customers", {"offset": "wrong"}),
            ("/recommendations", {}), ("/recommendations", {"customer_id": ""}),
            ("/recommendations", {"customer_id": "u' OR 1=1"}),
            ("/recommendations", {"customer_id": "u" * 129}),
            ("/recommendations", {"customer_id": "u", "method": "undefined"}),
            ("/recommendations", {"customer_id": "u", "method": ""}),
            ("/recommendations", {"customer_id": "u", "method": "x" * 81}),
            ("/customers/has space/profile", {}),
        ]
        for path, params in invalid:
            response = client.get(path, params=params)
            assert response.status_code == 422, (path, params, response.text)
            ErrorResponse.model_validate(response.json())
        assert client.get("/recommendations", params={"customer_id": "u"}).status_code == 200


@pytest.mark.parametrize("failure", ["missing", "malformed", "empty_configuration"])
def test_unavailable_startup_keeps_health_live_but_blocks_business_routes(tmp_path, monkeypatch, failure):
    path = tmp_path / "missing"
    if failure == "malformed":
        path.mkdir()
        (path / "manifest.json").write_text("[]")
    if failure == "empty_configuration":
        monkeypatch.setenv("ATELIER_SERVING_BUNDLE", " ")
        application = api.create_app()
    else:
        application = api.create_app(path)
    with TestClient(application) as client:
        assert client.get("/health").status_code == 200
        for url in ["/ready", "/customers", "/customers/u/profile", "/recommendations?customer_id=u"]:
            response = client.get(url)
            assert response.status_code == 503
            assert ErrorResponse.model_validate(response.json()).detail.code == "bundle_unavailable"
            assert str(tmp_path) not in response.text
        assert client.get("/docs").status_code == 200
        assert client.get("/openapi.json").status_code == 200


def test_checksum_failure_prevents_readiness(api_bundle):
    _, directory = api_bundle
    path = directory / "model.txt"
    content = path.read_bytes()
    path.write_bytes(content + b"damaged")
    with TestClient(api.create_app(directory)) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 503
        assert client.get("/recommendations?customer_id=u").status_code == 503
    assert path.read_bytes() == content + b"damaged"


def test_environment_configuration_and_detached_bundle_need_no_pipeline_services(api_bundle, monkeypatch):
    root, directory = api_bundle
    detached = root.parent / f"{root.name}-api-detached"
    offline = root.parent / f"{root.name}-api-offline"
    shutil.copytree(directory, detached)
    application = api.create_app()  # Environment is intentionally set after factory creation.
    monkeypatch.setenv("ATELIER_SERVING_BUNDLE", str(detached))

    def no_tracking(*args, **kwargs):
        raise AssertionError("The HTTP application must not contact MLflow.")

    monkeypatch.setattr(package, "_tracking_client", no_tracking)
    root.rename(offline)
    try:
        with TestClient(application) as client:
            assert client.get("/ready").status_code == 200
            assert client.get("/customers/u/profile").status_code == 200
            assert client.get("/recommendations?customer_id=unknown").status_code == 200
        monkeypatch.setenv("ATELIER_SERVING_BUNDLE", str(detached / "invalid"))
        with TestClient(api.create_app(detached)) as client:
            assert client.get("/ready").status_code == 200  # Explicit factory path wins.
    finally:
        offline.rename(root)


def test_history_read_failure_switches_readiness_to_unavailable(api_bundle):
    _, directory = api_bundle
    path = directory / "customer_history.parquet"
    with TestClient(api.create_app(directory)) as client:
        path.unlink()
        assert client.get("/customers/u/profile").status_code == 503
        assert client.get("/ready").status_code == 503
        assert client.get("/recommendations?customer_id=u").status_code == 503
        assert client.get("/health").status_code == 200


def test_openapi_is_available_without_loading_data_and_covers_contracts(monkeypatch):
    def do_not_load(*args, **kwargs):
        raise AssertionError("OpenAPI generation must not access a bundle.")

    monkeypatch.setattr(api, "ServingBundle", do_not_load)
    application = api.create_app()
    schema = application.openapi()
    assert set(schema["paths"]) == {"/health", "/ready", "/customers", "/customers/{customer_id}/profile", "/recommendations"}
    assert schema["paths"]["/recommendations"]["get"]["responses"]["200"]["content"]["application/json"]["schema"]["$ref"].endswith("RecommendationsResponse")
    assert "503" in schema["paths"]["/ready"]["get"]["responses"]
    assert "404" in schema["paths"]["/customers/{customer_id}/profile"]["get"]["responses"]
    assert "422" in schema["paths"]["/recommendations"]["get"]["responses"]
    json.dumps(schema, allow_nan=False)


def test_unexpected_programming_error_is_not_silently_treated_as_bad_data(monkeypatch):
    def bug(*args, **kwargs):
        raise RuntimeError("unexpected application bug")

    monkeypatch.setattr(api, "ServingBundle", bug)
    with pytest.raises(RuntimeError, match="application bug"):
        with TestClient(api.create_app("unused")):
            pass
