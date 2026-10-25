"""Read-only FastAPI adapter for one independently verified serving bundle."""

from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path
from typing import Annotated

import duckdb
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from lightgbm.basic import LightGBMError
from starlette.concurrency import run_in_threadpool

from atelier_ai.api_models import (
    API_VERSION, CustomerId, CustomerPage, CustomerProfile, ErrorResponse,
    HealthResponse, ReadyResponse, RecommendationsResponse,
)
from atelier_ai.serving_bundle import ServingBundle


LOGGER = logging.getLogger(__name__)
DEFAULT_BUNDLE = Path("data/processed/pipeline_runs/local_v1/serving/bundle")
STARTUP_DATA_ERRORS = (OSError, ValueError, TypeError, KeyError, AttributeError,
                       IndexError, duckdb.Error, LightGBMError)
UNAVAILABLE = {"code": "bundle_unavailable", "message": "The serving bundle is unavailable."}
NOT_READY = {503: {"model": ErrorResponse, "description": "No valid serving bundle is loaded."}}
INVALID_REQUEST = {422: {"model": ErrorResponse, "description": "Invalid request parameter or method."}}


def get_bundle(request: Request) -> ServingBundle:
    service = getattr(request.app.state, "bundle", None)
    if service is None:
        raise HTTPException(status_code=503, detail=UNAVAILABLE)
    return service


BundleDependency = Annotated[ServingBundle, Depends(get_bundle)]
MethodQuery = Annotated[str | None, Query(min_length=1, max_length=80, pattern=r"^[A-Za-z0-9_-]+$")]


def create_app(bundle_directory: str | Path | None = None) -> FastAPI:
    """Create an app without accessing data; load the configured bundle at startup."""

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.bundle = None
        configured = bundle_directory if bundle_directory is not None else os.environ.get(
            "ATELIER_SERVING_BUNDLE", str(DEFAULT_BUNDLE))
        try:
            if not str(configured).strip():
                raise ValueError("ATELIER_SERVING_BUNDLE must be nonempty.")
            application.state.bundle = await run_in_threadpool(ServingBundle, Path(configured).expanduser().resolve())
        except STARTUP_DATA_ERRORS:
            LOGGER.exception("Serving bundle failed validation; fix the bundle and restart the API.")
        try:
            yield
        finally:
            application.state.bundle = None

    application = FastAPI(
        title="Atelier AI", version=API_VERSION, lifespan=lifespan,
        description="Historical client-advisor recommendations at the bundle's reference date. "
                    "Ranking scores are not calibrated purchase probabilities.",
    )
    application.state.bundle = None

    async def data_unavailable(request: Request, exc: Exception):
        request.app.state.bundle = None
        LOGGER.error("Serving data became unavailable; restore the bundle and restart the API.",
                     exc_info=(type(exc), exc, exc.__traceback__))
        return JSONResponse(status_code=503, content={"detail": UNAVAILABLE})

    application.add_exception_handler(OSError, data_unavailable)
    application.add_exception_handler(duckdb.Error, data_unavailable)

    @application.get("/health", response_model=HealthResponse, tags=["status"])
    def health():
        """Liveness: the process responds independently of bundle availability."""
        return HealthResponse()

    @application.get("/ready", response_model=ReadyResponse, responses=NOT_READY, tags=["status"])
    def ready(service: BundleDependency):
        """Readiness and public metadata for the bundle verified at startup."""
        report = service.manifest
        return {
            "historical_date": report["historical_date"], "bundle_version": service.version,
            "serving_mode": report["serving_mode"], "model_sha256": report["model_sha256"],
            "default_method": report["method"], "methods": [report["method"], *service.baselines],
            "customers": report["customers"], "catalogue_articles": report["catalogue_articles"],
            "recommendations_per_customer": report["recommendations_per_customer"],
        }

    @application.get("/customers", response_model=CustomerPage,
                     responses={**NOT_READY, **INVALID_REQUEST}, tags=["customers"])
    def customers(service: BundleDependency, offset: Annotated[int, Query(ge=0)] = 0,
                  limit: Annotated[int, Query(ge=1, le=100)] = 50):
        """List the frozen cohort in customer-ID order, including no-history profiles."""
        items = service.customers(offset=offset, limit=limit)
        total = service.manifest["customers"]
        next_offset = offset + len(items)
        return {
            "historical_date": service.manifest["historical_date"], "bundle_version": service.version,
            "total": total, "offset": offset, "limit": limit,
            "next_offset": next_offset if next_offset < total else None, "items": items,
        }

    @application.get(
        "/customers/{customer_id}/profile", response_model=CustomerProfile, tags=["customers"],
        responses={**NOT_READY, **INVALID_REQUEST,
                   404: {"model": ErrorResponse, "description": "No profile in the serving cohort."}},
    )
    def profile(customer_id: CustomerId, service: BundleDependency):
        """Return historical aggregates and at most 20 preceding purchase rows."""
        result = service.profile(customer_id)
        if result is None:
            raise HTTPException(status_code=404, detail={
                "code": "customer_not_found", "message": "No profile for this customer in the serving cohort.",
            })
        return {**result, "bundle_version": service.version}

    @application.get("/recommendations", response_model=RecommendationsResponse,
                     responses={**NOT_READY, **INVALID_REQUEST}, tags=["recommendations"])
    def recommendations(customer_id: CustomerId, service: BundleDependency, method: MethodQuery = None):
        """Return 12 distinct articles; unknown IDs receive explicit popularity fallback."""
        if method is not None and method not in {service.manifest["method"], *service.baselines}:
            raise HTTPException(status_code=422, detail={
                "code": "unknown_method", "message": "Choose a recommendation method listed by /ready.",
            })
        return service.recommendations(customer_id, method=method)

    return application


app = create_app()
