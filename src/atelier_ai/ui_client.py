"""Typed HTTP boundary for the advisor UI; no model or local-data access."""

from datetime import timedelta
from math import isclose
from typing import TypeVar

import httpx
from pydantic import BaseModel, TypeAdapter, ValidationError

from atelier_ai.api_models import (
    AnalysisResponse, CustomerId, CustomerPage, CustomerProfile, ReadyResponse,
    RecommendationsResponse, SnapshotResponse,
)


Response = TypeVar("Response", bound=BaseModel)
ID_ADAPTER = TypeAdapter(CustomerId)


class UiApiError(Exception):
    def __init__(self, message, *, code="api_error"):
        super().__init__(message)
        self.code = code


def checked_customer_id(value):
    try:
        return ID_ADAPTER.validate_python(value.strip())
    except (ValidationError, AttributeError):
        raise UiApiError("Saisissez un identifiant de 1 à 128 lettres, chiffres, tirets ou underscores.",
                         code="invalid_customer") from None


class AdvisorApi:
    def __init__(self, base_url, *, transport=None):
        try:
            url = httpx.URL(base_url)
        except (httpx.InvalidURL, TypeError):
            raise UiApiError("L’adresse du service est invalide.", code="configuration") from None
        if (url.scheme not in ("http", "https") or not url.host or url.path not in ("", "/")
                or url.query or url.fragment or url.username or url.password):
            raise UiApiError("L’adresse du service doit être une origine HTTP ou HTTPS sans identifiants.",
                             code="configuration")
        self.client = httpx.Client(base_url=str(url).rstrip("/") + "/", transport=transport,
                                   timeout=httpx.Timeout(8, connect=2), trust_env=False, follow_redirects=False)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.client.close()

    def _get(self, path, model: type[Response], *, params=None, missing_profile=False) -> Response | None:
        try:
            response = self.client.get(path, params=params)
        except httpx.TimeoutException:
            raise UiApiError("Le service met trop de temps à répondre. Réessayez.", code="timeout") from None
        except httpx.HTTPError:
            raise UiApiError("Le service est injoignable. Vérifiez qu’il est démarré.", code="unreachable") from None
        if response.status_code == 404 and missing_profile:
            try:
                if response.json().get("detail", {}).get("code") == "customer_not_found":
                    return None
            except (ValueError, AttributeError):
                pass
        if response.status_code == 503:
            raise UiApiError("Le service répond, mais son catalogue n’est pas prêt.", code="not_ready")
        if response.status_code == 404:
            raise UiApiError("Cette vue nécessite la version à jour de l’API.", code="unsupported_endpoint")
        if response.status_code == 422:
            raise UiApiError("Le service a refusé les paramètres demandés.", code="invalid_request")
        if response.status_code != 200:
            raise UiApiError("Le service a rencontré une erreur. Réessayez.", code="http_error")
        try:
            return model.model_validate(response.json())
        except (ValidationError, ValueError):
            raise UiApiError("La réponse du service est incompatible avec cette interface.", code="invalid_response") from None

    @staticmethod
    def _snapshot(value: SnapshotResponse, ready: ReadyResponse):
        if (value.bundle_version != ready.bundle_version or value.historical_date != ready.historical_date
                or value.api_version != ready.api_version):
            raise UiApiError("Le catalogue a changé pendant la consultation. Actualisez la page.", code="snapshot_changed")
        return value

    def ready(self):
        value = self._get("/ready", ReadyResponse)
        if not value.methods or len(set(value.methods)) != len(value.methods) or value.default_method not in value.methods:
            raise UiApiError("La liste des méthodes du service est incohérente.", code="invalid_response")
        return value

    def customers(self, ready, *, offset=0, limit=100):
        page = self._snapshot(self._get("/customers", CustomerPage, params={"offset": offset, "limit": limit}), ready)
        ids = [item.customer_id for item in page.items]
        if (page.offset != offset or page.limit != limit or page.total != ready.customers
                or len(ids) != len(set(ids)) or len(ids) > limit):
            raise UiApiError("La page de clients est incohérente.", code="invalid_response")
        return page

    def profile(self, ready, customer_id):
        customer_id = checked_customer_id(customer_id)
        profile = self._get(f"/customers/{customer_id}/profile", CustomerProfile, missing_profile=True)
        if profile is None:
            return None
        self._snapshot(profile, ready)
        if profile.customer_id != customer_id:
            raise UiApiError("Le profil reçu ne correspond pas au client demandé.", code="invalid_response")
        return profile

    def recommendations(self, ready, customer_id, *, method=None):
        customer_id = checked_customer_id(customer_id)
        if method is not None and method not in ready.methods:
            raise UiApiError("Choisissez une méthode disponible.", code="invalid_method")
        params = {"customer_id": customer_id}
        if method is not None:
            params["method"] = method
        result = self._snapshot(self._get("/recommendations", RecommendationsResponse, params=params), ready)
        if (result.customer_id != customer_id or len({item.article_id for item in result.items}) != 12
                or [item.rank for item in result.items] != list(range(1, 13))
                or result.strategy not in {method or ready.default_method, "popularity_fallback_unknown_customer"}):
            raise UiApiError("La sélection reçue ne correspond pas à la demande.", code="invalid_response")
        return result

    def analysis(self, ready):
        result = self._snapshot(self._get("/analysis", AnalysisResponse), ready)
        names = [item.method for item in result.comparison]
        intervals = [item for item in result.intervals if item.metric == "map_at_12"]
        deltas = [item for item in result.intervals if item.metric == "delta_map_at_12"]
        if (set(names) != set(ready.methods) or len(names) != len(set(names))
                or result.baseline_comparator not in ready.methods
                or result.baseline_comparator == ready.default_method
                or result.start_inclusive != ready.historical_date
                or result.end_exclusive - result.start_inclusive != timedelta(days=7)
                or result.recipients != ready.customers
                or result.target_buyers + result.excluded_nonbuyers != result.recipients
                or result.candidate_target_pairs > result.target_pairs
                or len(intervals) != len(names) or {item.method for item in intervals} != set(names)
                or len(deltas) != 1 or deltas[0].method != "model_minus_baseline"
                or any(item.ci_lower > item.ci_upper or item.evaluated_buyers != result.target_buyers
                       or item.bootstrap_resamples != result.bootstrap.resamples for item in result.intervals)
                or any(item.recommendation_customers != result.recipients or item.evaluated_customers != result.target_buyers
                       or item.target_pairs != result.target_pairs for item in result.comparison)):
            raise UiApiError("Le bilan reçu est incohérent.", code="invalid_response")
        metrics = {item.method: item.map_at_k for item in result.comparison}
        if (any(not isclose(item.observed, metrics[item.method], rel_tol=0, abs_tol=1e-12) for item in intervals)
                or not isclose(deltas[0].observed, metrics[ready.default_method] - metrics[result.baseline_comparator],
                               rel_tol=0, abs_tol=1e-12)):
            raise UiApiError("Les valeurs du bilan sont incohérentes.", code="invalid_response")
        return result
