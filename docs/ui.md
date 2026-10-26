# Local advisor interface

Streamlit provides four advisor views over the historical recommendation API. It reads profiles, recommendations and aggregate evaluation exclusively through HTTP. It never imports the bundle reader, loads models, reads competition files or contacts MLflow.

## Install and launch

From the repository root:

```bash
uv sync --locked --extra api --extra ui
.venv/bin/python -m pytest tests -q
```

Streamlit 1.66.1 is explicitly pinned. The UI also declares HTTPX 0.28.1 and Pydantic 2.14.1. The updated lock adds eight packages and preserves every version among the previous 148 packages. Both extras are included for full application testing; UI interaction tests skip when Streamlit is absent.

In terminal 1, start or restart the API. Restarting matters because this increment adds the aggregate analysis route:

```bash
ATELIER_SERVING_BUNDLE=data/processed/pipeline_runs/local_v1/serving/bundle \
  .venv/bin/python -m uvicorn atelier_ai.api:app --host 127.0.0.1 --port 8000
```

In terminal 2, from the same repository root:

```bash
ATELIER_API_URL=http://127.0.0.1:8000 \
  .venv/bin/python -m streamlit run streamlit_app.py
```

Open [Atelier AI](http://127.0.0.1:8501). The default API origin is `http://127.0.0.1:8000`. `ATELIER_API_URL` accepts an HTTP or HTTPS origin, without a path, query, fragment or embedded credentials. The two processes run independently; the UI calls the API from its Python process. Browser CORS configuration is unnecessary for this local architecture.

`.streamlit/config.toml` supplies the light ivory/teal theme, serif headings, loopback address and port 8501. File watching uses polling to support environments where macOS filesystem event streams cannot start, and excludes bulk data and local environment directories. Usage statistics are disabled.

## Advisor workflow

| View | Behaviour |
|---|---|
| Profile | Historical 180-day activity, distinct articles, recency, factual preferences and up to 20 preceding source purchase rows |
| Selection | Twelve ranked article cards, method choice, retained-article checkboxes, reset and JSON download |
| Comparison | Two methods for the same customer, ranked factual tables and the number of shared articles |
| Analysis | Packaged V1 metrics for four methods, paired bootstrap MAP intervals, complete-reference counts and interpretation limits |

Browse the frozen cohort in pages of 100 or submit a customer identifier. IDs remain strings and retain leading zeros. An unknown ID has no profile and receives an explicitly labelled global popularity selection. A known customer with no preceding purchases keeps the packaged learned recommendations. Missing optional catalogue metadata is not replaced with invented content or images.

The selection initially retains all twelve proposals. Uncheck articles, use **Tout retenir** to restore all twelve, or download the retained subset. Navigation between views preserves the current draft, including the chosen method. Changing the customer or method starts a new draft; drafts are also cleared when the API origin or serving bundle version changes. A zero-item selection disables download. Drafts live in the browser session and are not saved to the backend.

The JSON export records the customer ID, historical date, bundle version, requested method, effective strategy, score kind, original list size, retained count and retained factual article records. Original recommendation ranks remain unchanged when articles are removed. Downloading does not create an order or contact a customer.

## Availability and consistency

Readiness is checked over HTTP on every Streamlit rerun. Profile, list, recommendation and analysis responses are cached for at most 30 seconds, keyed by API origin, full readiness metadata and request parameters. **Actualiser** clears this cache. All successful responses are checked against the active bundle version, date and API version; recommendations also verify the customer, strategy, twelve unique article IDs and contiguous ranks.

The analysis consumer validates method membership, evaluation denominators, paired interval metadata and the agreement between reported MAP values and their intervals. It reports a failed contract or a bundle change instead of mixing snapshots.

A missing or unready API, timeout, incompatible response or unavailable route produces an actionable message and **Réessayer**. Cached recommendations are not rendered when readiness fails on a rerun. An unavailable `/analysis` route indicates that the API process needs restarting with the updated code. Readiness is request-based; an idle browser is not continuously polled. Unexpected programming errors remain visible for diagnosis.

## Historical interpretation

The recorded bundle represents **2020-09-09**, with histories strictly before that date. Catalogue metadata does not establish current stock availability. Model scores are uncalibrated ranking scores, reference methods expose ranks without numeric scores, and unknown-ID fallback scores count historical purchase rows. Their scales must not be compared as purchase probabilities.

V1 covers `[2020-09-09, 2020-09-16)`. All 20,000 recipients receive recommendations. Relevance averages cover all 1,015 buyers and all 3,073 distinct customer/article targets, including purchases absent from candidates; 18,985 nonbuyers are excluded from those averages. Candidate retrieval covers 618 targets, approximately 20.11%.

The learned MAP@12 is 0.022927 against 0.022168 for the validation-selected recent-purchases reference. The paired 95% interval of the difference is `[-0.002626, 0.004346]`: a clear improvement is not established. The displayed intervals use 5,000 paired buyer resamples with seed 42. V1 has prior development exposure; final confirmation remains reserved and commercial effects are unmeasured.

## Verification scope

The complete synthetic suite passes **158 tests**, including eleven API tests, eight HTTP-consumer checks and five Streamlit AppTest workflows. The UI checks exercise draft retention through navigation, method persistence, customer changes, unknown and malformed IDs, no-history profiles, the aggregate analysis view and loss of API availability after a successful selection. Client tests also reject mismatched snapshots, customers, duplicate article IDs and invalid HTTP responses. The existing Starlette HTTPX test-client deprecation warning remains; the checks pass.

The complete real `local_v1` pipeline passes artifact verification with the added UI dependencies and retains native model SHA-256 `d6ca939c488a707c9a8c5be7480f738df640d2bc0a3496fdcb20b573f8353bf3`. A live temporary Uvicorn/Streamlit stack verifies the six API routes, actual V1 aggregates, browser rendering and JSON export. The downloaded eleven-item selection contains eleven distinct string article IDs with preserved leading zeros. This verifies the recorded campaign and serving workflow; it is not a fresh full-source training replay or a latency benchmark.

After applying these source/dependency changes, refresh local DVC records before committing:

```bash
uvx --from dvc==3.67.1 dvc repro
```

The scientific implementations and bundle reader remain unchanged. Valid local outputs remain reusable. Docker Compose and synthetic CI are the next increments.

Framework references: [Streamlit configuration](https://docs.streamlit.io/develop/api-reference/configuration/config.toml), [application testing](https://docs.streamlit.io/develop/concepts/app-testing) and [AppTest](https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest).
