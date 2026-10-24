# Atelier AI

Personalised fashion recommendations to help a client advisor prepare a ranked selection of twelve articles from observed customer purchase history.

## Task and approach

At each historical cutoff, retrieve 300 distinct candidate articles using earlier purchases, compute historical features, then rank a top-12 list. Evaluate against distinct purchases in the next seven days. Purchases are implicit relevance signals; scores are not calibrated purchase probabilities.

The project follows **data foundation → temporal protocol → behavioral EDA → candidates/features → reference methods → learned ranker → selection → local application**. The [implementation plan](docs/implementation_plan.md) records completed deliverables and the next increments.

## Current milestone

Plan stage 7 is complete: the initial ranker and six declared hyperparameter comparisons are recorded. The selected validation candidate now proceeds to stage 8.

- Source diagnostics cover 105,542 articles, 1,371,980 customer records and 31,788,324 purchase rows. Raw files remain unchanged; typed Parquet supports repeatable analysis.
- The shared calendar defines six historical training weeks, validation, intermediate V1 evaluation and final model confirmation.
- Behavioral EDA covers temporal/category activity, customer histories, repeat delays, popularity and top-seller renewal, category/colour breadth and affinities, prices and channels.
- A fixed future-independent cohort contains 20,000 customers. Candidates combine 30 personal articles, global top 100, three preferred types × 50, and 28-day popularity fill to 300.
- Historical preparation stores 36,000,000 labelled candidate rows across 120,000 groups. Informative fitting uses 2,533 complete groups / 759,900 rows and 31 features.
- All six LambdaRank configurations use at most 1,000 iterations and full-reference validation NDCG@12 stopping with patience 50. The selected validation candidate uses 63 leaves, minimum 20 examples per leaf and iteration 8; the initial 31/100 experiment selected iteration 16.
- Native model reload checks, verified Parquet exports, MLflow recording, complete-target comparisons and paired/history diagnostics are implemented.

## Validation results

| Method | MAP@12 | Recall@12 | NDCG@12 |
|---|---|---|---|
| recent_popularity_7d | 0.006091 | 0.022700 | 0.012546 |
| preferred_product_types | 0.005346 | 0.014352 | 0.009923 |
| recent_purchases_180d | 0.025445 | 0.055198 | 0.036811 |
| lightgbm_initial | 0.027023 | 0.056514 | 0.038934 |
| lightgbm_leaves_63_min_20 | **0.032710** | 0.059368 | 0.044496 |

All methods recommend twelve distinct articles to the same 20,000 recipients. Target-based means include all 1,065 validation buyers and all 3,428 distinct targets, including retrieval misses. These are retrospective validation results, not an independent-test or business-impact claim. The candidate is selected by MAP@12, while the 63/100 configuration leads NDCG and Recall. See [tuning results](docs/lightgbm_tuning_results.md) and [initial results](docs/first_ranker_results.md).

## Temporal contract

| Role | Start inclusive | End exclusive | Source transaction rows |
|---|---|---|---:|
| EDA source-history pool | 2018-09-20 | 2020-09-02 | 31,028,115 |
| Validation | 2020-09-02 | 2020-09-09 | 264,657 |
| Intermediate V1 evaluation | 2020-09-09 | 2020-09-16 | 255,241 |
| Final LightGBM/PyTorch confirmation | 2020-09-16 | 2020-09-23 | 240,311 |

Intervals include their start and exclude their end. Table counts cover all source customers and retain repeated rows. Six historical cutoffs run weekly from 22 July through 26 August 2020. Candidates/features use only observations strictly before their own cutoff, with a 180-day personal lookback. The [evaluation strategy](docs/evaluation.md) defines fitting, selection, final refit and development-exposure limits. V1 and final-confirmation ranking performance are not computed by the current notebooks.

## Repository guide

| Notebook | Purpose |
|---|---|
| [01 Data foundation](notebooks/01_data_foundation.ipynb) | Framing, source diagnostics and configured calendar |
| [02 Behavioral EDA](notebooks/02_behavioral_eda.ipynb) | Questions, complete-scope aggregates, figures and modelling hypotheses |
| [03 Candidates and features](notebooks/03_candidates_and_features.ipynb) | Fixed cohort, retrieval, three baselines, feature contract and validation export |
| [04 Training dataset](notebooks/04_training_dataset.ipynb) | Six historical target weeks, labels, group audit and aligned fitting inputs |
| [05 Initial LightGBM](notebooks/05_lightgbm_ranker.ipynb) | Planned initial fit, learning curve, full-reference comparison and diagnostics |
| [06 LightGBM tuning](notebooks/06_lightgbm_tuning.ipynb) | Six declared configurations, common-input checks, MAP heatmap and candidate selection |

Reusable computations live under `src/atelier_ai/`; synthetic tests under `tests/`; configuration under `configs/`. Raw data, prepared tables, generated figures, models and MLflow storage stay outside Git. Executed notebooks contain portable static figures alongside interactive Plotly outputs.

## Local reproduction

Use Python 3.12, Git and uv. Clone this repository and run `uv sync --locked`. On macOS, LightGBM needs the OpenMP runtime (`brew install libomp`). Plotly PNG export requires Chrome/Chromium supported by Kaleido.

Obtain the competition files under their access conditions and place `articles.csv`, `customers.csv` and `transactions_train.csv` in `data/raw/hm/`. The [acquisition manifest](docs/data_manifest.json) identifies the documented source version.

Run notebooks in numerical order with the project environment. The first notebook's complete-file diagnostics scan large sources; the following notebooks prepare and reuse typed Parquet. Feature exports are batched and verified. Run the whole learning sequence explicitly with:

```bash
uv run python scripts/run_notebooks.py
```

To rerun a selected notebook, supply its filename, for example:

```bash
uv run python scripts/run_notebooks.py 05_lightgbm_ranker.ipynb
```

Source and configuration hashes identify derived exports; modified computation creates a distinct dataset identity. Active run indices live under `data/processed/weekly_v1/`. They remove any need to copy an old machine-specific export directory into a notebook.

```bash
uv run pytest -q
uv run python scripts/check_temporal_protocol.py
```

Tests use synthetic data and do not need competition files. For the local MLflow file store, run `MLFLOW_ALLOW_FILE_STORE=true uv run mlflow ui --backend-store-uri ./mlruns` and open the printed local address. The tracking store is local, explicitly opted into file storage, and excluded from Git.

## Next work

1. Complete stage-8 diagnostics for the selected candidate: native SHAP examples, declared feature ablations, successes/failures, uncertainty and timings.
2. Freeze the selected method and iteration budget; refit a selected learned configuration on training plus validation, then report intermediate V1 performance with its exposure limitations.
3. Package the selected system with DVC, FastAPI, Streamlit, Docker Compose and synthetic CI.
4. Compare PyTorch under common candidates/periods/metrics before opening final confirmation; later add GCP and the advisor copilot.

## Documentation and limits

- [Project brief](docs/project_brief.md)
- [Implementation plan](docs/implementation_plan.md)
- [Evaluation strategy](docs/evaluation.md)
- [Training protocol](docs/training_protocol.md)
- [Feature dictionary](docs/feature_dictionary.md)
- [Initial ranking results](docs/first_ranker_results.md)
- [Controlled LightGBM tuning](docs/lightgbm_tuning_results.md)
- [Methodological decisions](docs/methodological_decisions.md)
- [Behavioral EDA summary](docs/eda_summary.md)
- [Data card](docs/data_card.md)
- [Data dictionary](docs/data_dictionary.md)

The project uses the [H&M recommendation benchmark](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/). Metadata observation dates, stock, price units, `FN`/`Active` semantics and channel business labels remain unresolved. Customer snapshot attributes are excluded from the initial feature matrix. Historical benchmark results do not establish performance in luxury retail or causal business benefits.
