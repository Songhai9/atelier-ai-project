# Historical recommendation API

FastAPI serves one verified, immutable serving bundle. Loading happens once per process during application startup. Requests use the packaged reader and do not run training, generate candidates, access raw CSVs or contact MLflow.

## Install and launch

From the repository root, use the locked environment and the API extra:

```bash
uv sync --locked --extra api
.venv/bin/python -m pytest -q tests/test_api.py
```

The API dependencies are explicit pins of versions already present in the previous lock: FastAPI 0.162.0, Pydantic 2.14.1 and Uvicorn 0.54.0. HTTPX 0.28.1 is declared for development tests. All 148 existing locked package versions are preserved.

Start the local server with the completed bundle:

```bash
ATELIER_SERVING_BUNDLE=data/processed/pipeline_runs/local_v1/serving/bundle \
  .venv/bin/python -m uvicorn atelier_ai.api:app --host 127.0.0.1 --port 8000
```

Open [interactive API documentation](http://127.0.0.1:8000/docs) or the [OpenAPI schema](http://127.0.0.1:8000/openapi.json). The default bundle path is the same `local_v1` directory relative to the working directory; setting `ATELIER_SERVING_BUNDLE` explicitly also supports a detached absolute bundle path. No repository or MLflow directory is required at runtime when that path is set.

`create_app(bundle_directory)` provides an explicit factory for tests and embedding. Its argument takes precedence over the environment; otherwise configuration is read at startup. Importing the app or generating its OpenAPI schema does not load data.

## HTTP contract

| Route | Successful response | Other expected responses |
|---|---|---|
| `GET /health` | 200, process liveness | Independent of bundle readiness |
| `GET /ready` | 200, verified bundle metadata and available methods | 503 when no valid bundle is loaded |
| `GET /customers?offset=0&limit=50` | 200, customer-ID-sorted page, total and next offset | 422 for invalid pagination; 503 when unavailable |
| `GET /customers/{customer_id}/profile` | 200, historical aggregates and at most 20 preceding purchase rows | 404 for IDs outside the serving cohort; 422 for malformed IDs; 503 when unavailable |
| `GET /recommendations?customer_id=...&method=...` | 200, exactly 12 distinct ranked articles | 422 for malformed parameters or unavailable methods; 503 when unavailable |

Customer IDs remain strings, preserving leading zeros. They accept 1–128 ASCII letters, digits, underscores or hyphens. Pagination uses a nonnegative offset and a page size from 1 to 100. An offset beyond the final page returns an empty list with no next offset.

The default recommendation method is the frozen learned model. `/ready` lists it and the three packaged references. An unknown customer has no profile, but receives the explicit `popularity_fallback_unknown_customer` strategy from `/recommendations`. A known customer with no observed history retains the frozen learned recommendations. A valid requested reference method never creates a profile for an unknown ID.

Successful responses include an API version. Business responses also include the historical reference date and bundle version. Recommendations carry the strategy, score kind, model hash where applicable, article IDs, ranks and available factual catalogue fields. Missing optional metadata remains null.

| Score kind | Interpretation |
|---|---|
| `uncalibrated_lightgbm_ranking_score` | Native model ranking score; not a purchase probability |
| `baseline_rank_only` | Reference-method order; `score` is null |
| `historical_purchase_rows_7d` | Seven-day global purchase-row count for unknown-ID popularity fallback; 28-day fill resolves the remaining selection |

Profiles use only the 180 days strictly before the bundle's historical date. Repeated source purchase rows remain repeated; rows are not orders. Price values retain unresolved source units, and channel numbers retain source codes. Customer snapshot attributes and V1 target outcomes are not profile inputs. Catalogue metadata does not establish stock availability.

## Quick requests

```bash
curl -s http://127.0.0.1:8000/ready
curl -s 'http://127.0.0.1:8000/customers?limit=1'
curl -s 'http://127.0.0.1:8000/recommendations?customer_id=unknown-demo'
```

Copy a real ID from `/customers` into the profile path or recommendation query in `/docs` to inspect a known customer. Select a method listed by `/ready` to compare the learned system with a reference.

## Availability and verification

A missing, malformed or corrupted bundle fails startup data validation. The HTTP process remains live: `/health` returns 200 while `/ready` and business routes return 503 with `bundle_unavailable`. Detailed validation errors stay in server logs; response bodies do not expose local filesystem paths. Restore the bundle and restart the process to load it again. Unexpected application programming errors still fail startup.

The bundle must remain immutable while the server runs. If a profile-history file becomes unreadable, the request returns 503 and readiness becomes unavailable until restart. Startup validation is not a continuous filesystem integrity scan.

Pydantic models describe and validate successful responses in OpenAPI, including finite scores and exactly twelve recommendation items. The HTTP tests cover pagination, all methods, zero-history profiles, unknown IDs, invalid requests, readiness failures, checksum corruption, concurrent reads and serving after the original project/MLflow directory is moved away.

The full Python suite passed 144 tests. A live Uvicorn HTTP smoke check against the recorded `local_v1` bundle passed all five routes, a known customer's twelve recommendations, unknown-profile 404, unknown-ID fallback, Swagger and OpenAPI. The check used 20,000 packaged customers at historical date 2020-09-09, retained model SHA-256 `d6ca939c488a707c9a8c5be7480f738df640d2bc0a3496fdcb20b573f8353bf3`, and left the serving manifest unchanged. The temporary test server was stopped afterward.

The pinned Starlette test client currently reports a deprecation warning for its HTTPX compatibility layer. The HTTP tests pass with the locked versions; this warning is confined to the test client.

Source files and dependency declarations are DVC dependencies. After applying the API change, refresh their local lock records before committing:

```bash
uvx --from dvc==3.67.1 dvc repro
```

The existing scientific stage implementations and package reader are unchanged, so valid `local_v1` outputs remain reusable. DVC retains the existing output directories and finishes with full integrity verification.

This increment adds the API only. Streamlit, containers and automated CI follow as separate increments. Batch scoring duration is not an API latency measurement.

Framework references: [FastAPI lifespan](https://fastapi.tiangolo.com/advanced/events/), [response models](https://fastapi.tiangolo.com/tutorial/response-model/) and [HTTP testing](https://fastapi.tiangolo.com/tutorial/testing/).
