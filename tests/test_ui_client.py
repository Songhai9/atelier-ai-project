"""HTTP consumer snapshot, contract and failure checks."""

from copy import deepcopy

from fastapi.testclient import TestClient
import httpx
import pytest

from atelier_ai.api import create_app
from atelier_ai.ui_client import AdvisorApi, UiApiError, checked_customer_id
from test_api import api_bundle, evaluation_campaign, serving_campaign


def test_consumer_accepts_real_api_contracts_and_unknown_fallback(api_bundle):
    _, directory = api_bundle
    with TestClient(create_app(directory)) as server:
        def respond(request):
            return server.request(request.method, str(request.url.copy_with(scheme="http", host="testserver")))
        with AdvisorApi("http://testserver", transport=httpx.MockTransport(respond)) as client:
            ready = client.ready()
            assert client.customers(ready, offset=0, limit=100).total == 3
            assert client.profile(ready, "cold").summary.customer_has_history is False
            assert client.profile(ready, "outside-cohort") is None
            assert len(client.recommendations(ready, "u").items) == 12
            assert client.recommendations(ready, "outside-cohort").strategy == "popularity_fallback_unknown_customer"
            assert client.analysis(ready).target_pairs == 3


def test_consumer_rejects_snapshot_customer_and_ranking_mismatches(api_bundle):
    _, directory = api_bundle
    with TestClient(create_app(directory)) as server:
        ready_body = server.get("/ready").json()
        original = server.get("/recommendations?customer_id=u").json()
        malformed = []
        wrong_snapshot = deepcopy(original)
        wrong_snapshot["bundle_version"] = "f" * 64
        malformed.append((wrong_snapshot, "snapshot_changed"))
        wrong_customer = deepcopy(original)
        wrong_customer["customer_id"] = "cold"
        malformed.append((wrong_customer, "invalid_response"))
        duplicate = deepcopy(original)
        duplicate["items"][1]["article_id"] = duplicate["items"][0]["article_id"]
        malformed.append((duplicate, "invalid_response"))
        for body, code in malformed:
            def respond(request):
                return httpx.Response(200, json=ready_body if request.url.path == "/ready" else body)
            with AdvisorApi("http://testserver", transport=httpx.MockTransport(respond)) as client:
                ready = client.ready()
                with pytest.raises(UiApiError) as error:
                    client.recommendations(ready, "u")
                assert error.value.code == code


@pytest.mark.parametrize("kind", ["timeout", "unreachable", "not_ready", "invalid_response", "unsupported_endpoint"])
def test_consumer_reports_safe_actionable_failures(kind):
    def respond(request):
        if kind == "timeout":
            raise httpx.ReadTimeout("internal path must remain private", request=request)
        if kind == "unreachable":
            raise httpx.ConnectError("private", request=request)
        if kind == "not_ready":
            return httpx.Response(503, text="private backend traceback")
        if kind == "unsupported_endpoint":
            return httpx.Response(404)
        return httpx.Response(200, text="not JSON")
    with AdvisorApi("http://testserver", transport=httpx.MockTransport(respond)) as client:
        with pytest.raises(UiApiError) as error:
            client.ready()
        assert error.value.code == kind
        assert "private" not in str(error.value) and "traceback" not in str(error.value)


def test_configuration_and_identifiers_are_validated_before_http():
    for value in ("", "file:///etc/passwd", "http://name:secret@host", "http://host/path", "http://host?query=1"):
        with pytest.raises(UiApiError, match="adresse"):
            AdvisorApi(value)
    assert checked_customer_id(" 00001 ") == "00001"
    for value in ("", "a/b", "bad id", "x" * 129):
        with pytest.raises(UiApiError, match="identifiant"):
            checked_customer_id(value)
