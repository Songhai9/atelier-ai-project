# Atelier AI

Personalised fashion recommendations to help a client advisor prepare a ranked selection of twelve articles from observed customer purchase history.

## Task and approach

At each historical cutoff, retrieve 300 distinct candidate articles using earlier purchases, compute historical features, then rank a top-12 list. Evaluate against distinct purchases in the next seven days. Purchases are implicit relevance signals; scores are not calibrated purchase probabilities.

The project follows **data foundation → temporal protocol → behavioral EDA → candidates/features → reference methods → learned ranker → selection → local application**. The [implementation plan](docs/implementation_plan.md) records completed deliverables and the next increments.

## Current milestone

Scientific plan stages 1-8 are complete in notebooks 01-07. Diagnostics, frozen selection, the final train-plus-validation refit and intermediate V1 evaluation are recorded. LightGBM leads validation MAP; its small positive V1 MAP difference has an interval containing zero.

- Source diagnostics cover 105,542 articles, 1,371,980 customer records and 31,788,324 purchase rows. Raw files remain unchanged; typed Parquet supports repeatable analysis.
- The shared calendar defines six historical training weeks, validation, intermediate V1 evaluation and final model confirmation.
- Behavioral EDA covers temporal/category activity, customer histories, repeat delays, popularity and top-seller renewal, category/colour breadth and affinities, prices and channels.
- A fixed future-independent cohort contains 20,000 customers. Candidates combine 30 personal articles, global top 100, three preferred types × 50, and 28-day popularity fill to 300.
- Historical preparation stores 36,000,000 labelled candidate rows across 120,000 groups. Initial model-selection fitting uses 2,533 complete groups / 759,900 rows and 31 features.
- All six LambdaRank configurations use at most 1,000 iterations and full-reference validation NDCG@12 stopping with patience 50. The selected validation candidate uses 63 leaves, minimum 20 examples per leaf and iteration 8; the initial 31/100 experiment selected iteration 16.
- Notebook 07 records gain importance, native SHAP examples, three feature ablations, error/history diagnostics, customer bootstrap uncertainty and local batch timings.
- Final selection is frozen before the seven-week refit. The final model has eight trees and 31 features; save/reload checks pass on 1,000 probe rows. V1 predictions and complete-reference metrics are stored with MLflow provenance.

## Validation results

| Method | MAP@12 | Recall@12 | NDCG@12 |
|---|---|---|---|
| recent_popularity_7d | 0.006091 | 0.022700 | 0.012546 |
| preferred_product_types | 0.005346 | 0.014352 | 0.009923 |
| recent_purchases_180d | 0.025445 | 0.055198 | 0.036811 |
| lightgbm_initial | 0.027023 | 0.056514 | 0.038934 |
| lightgbm_leaves_63_min_20 | **0.032710** | 0.059368 | 0.044496 |

All methods recommend twelve distinct articles to the same 20,000 recipients. Target-based means include all 1,065 validation buyers and all 3,428 distinct targets, including retrieval misses. These are retrospective validation results, not an independent-test or business-impact claim. The candidate is selected by MAP@12, while the 63/100 configuration leads NDCG and Recall. See [tuning results](docs/lightgbm_tuning_results.md) and [initial results](docs/first_ranker_results.md).


## Intermediate V1 results

The frozen 63/20 configuration was refitted on seven training cutoffs using eight trees, 2,974 complete informative groups, 892,200 rows and 4,461 positive candidate pairs. V1 covers [2020-09-09, 2020-09-16): all 20,000 recipients receive lists; target-based means cover 1,015 buyers and all 3,073 distinct target pairs from 3,454 cohort source purchase rows. The other 18,985 recipients (94.925%) remain outside target-based means.

| Method | MAP@12 | Macro Recall@12 | NDCG@12 |
|---|---:|---:|---:|
| recent_popularity_7d | 0.005443 | 0.019836 | 0.011334 |
| preferred_product_types | 0.003475 | 0.012097 | 0.007408 |
| recent_purchases_180d | 0.022168 | 0.044635 | 0.032089 |
| lightgbm_leaves_63_min_20 | 0.022927 | 0.041459 | 0.031957 |

The MAP difference against the validation-selected recent-purchases baseline is **0.000759**, with a paired percentile 95% interval **[-0.002626, 0.004346]** from 5,000 buyer resamples. This does not establish a clear V1 advantage. The baseline has slightly higher Recall and NDCG. Preserve the frozen configuration and report the result without test-driven retuning. V1 is retrospective; final confirmation remains reserved. See the [stage-8 evaluation report](docs/stage8_evaluation_report.md) and [model card](docs/model_card.md).

## Temporal contract

| Role | Start inclusive | End exclusive | Source transaction rows |
|---|---|---|---:|
| EDA source-history pool | 2018-09-20 | 2020-09-02 | 31,028,115 |
| Validation | 2020-09-02 | 2020-09-09 | 264,657 |
| Intermediate V1 evaluation | 2020-09-09 | 2020-09-16 | 255,241 |
| Final LightGBM/PyTorch confirmation | 2020-09-16 | 2020-09-23 | 240,311 |

Intervals include their start and exclude their end. Table counts cover all source customers and retain repeated rows. Six historical cutoffs run weekly from 22 July through 26 August 2020. Candidates/features use only observations strictly before their own cutoff, with a 180-day personal lookback. The [evaluation strategy](docs/evaluation.md) defines fitting, selection, final refit and development-exposure limits. Intermediate V1 has been evaluated; final-confirmation ranking performance remains uncomputed.

## Repository guide

| Notebook | Purpose |
|---|---|
| [01 Data foundation](notebooks/01_data_foundation.ipynb) | Framing, source diagnostics and configured calendar |
| [02 Behavioral EDA](notebooks/02_behavioral_eda.ipynb) | Questions, complete-scope aggregates, figures and modelling hypotheses |
| [03 Candidates and features](notebooks/03_candidates_and_features.ipynb) | Fixed cohort, retrieval, three baselines, feature contract and validation export |
| [04 Training dataset](notebooks/04_training_dataset.ipynb) | Six historical target weeks, labels, group audit and aligned fitting inputs |
| [05 Initial LightGBM](notebooks/05_lightgbm_ranker.ipynb) | Planned initial fit, learning curve, full-reference comparison and diagnostics |
| [06 LightGBM tuning](notebooks/06_lightgbm_tuning.ipynb) | Six declared configurations, common-input checks, MAP heatmap and candidate selection |
| [07 Model diagnostics and V1](notebooks/07_model_diagnostics.ipynb) | Selected-model diagnostics, frozen decision, final refit, V1 recommendations and uncertainty |

Reusable computations live under `src/atelier_ai/`; synthetic tests under `tests/`; configuration under `configs/`. Raw data, prepared tables, generated figures, models and MLflow storage stay outside Git. Executed notebooks contain portable static figures alongside interactive Plotly outputs.

## Local reproduction

Use Python 3.12, Git and uv. Clone this repository and run `uv sync --locked`. On macOS, LightGBM needs the OpenMP runtime (`brew install libomp`). Plotly PNG export requires Chrome/Chromium supported by Kaleido.

Obtain the competition files under their access conditions and place `articles.csv`, `customers.csv` and `transactions_train.csv` in `data/raw/hm/`. The [acquisition manifest](docs/data_manifest.json) identifies the documented source version.

Run notebooks in numerical order with the project environment. The first notebook's complete-file diagnostics scan large sources; the following notebooks prepare and reuse typed Parquet. Feature exports are batched and verified. For a fresh derived-data campaign, run the whole learning sequence explicitly with:

```bash
uv run python scripts/run_notebooks.py
```

To rerun a selected notebook, supply its filename, for example:

```bash
uv run python scripts/run_notebooks.py 05_lightgbm_ranker.ipynb
```

Notebook 07 is a recorded selection/evaluation campaign. Preserve its completed frozen artifacts when reviewing saved outputs. Re-executing ablation or timing cells creates different evidence hashes; the freeze guard deliberately rejects a changed campaign. Full replay should use fresh derived-data storage while preserving the completed campaign. This closure reviewed saved outputs and did not execute a fresh end-to-end replay.

Source and configuration hashes identify derived exports; modified computation creates a distinct dataset identity. Active run indices live under `data/processed/weekly_v1/`. They remove any need to copy an old machine-specific export directory into a notebook.

```bash
uv run pytest -q
uv run python scripts/check_temporal_protocol.py
```

Tests use synthetic data and do not need competition files. For the local MLflow file store, run `MLFLOW_ALLOW_FILE_STORE=true uv run mlflow ui --backend-store-uri ./mlruns` and open the printed local address. The tracking store is local, explicitly opted into file storage, and excluded from Git.

## Next work

1. Extract the validated workflow into a reproducible local pipeline and version its dependencies with DVC.
2. Package the frozen system for FastAPI, Streamlit, Docker Compose and synthetic CI.
3. Compare PyTorch under common candidates/periods/metrics and freeze both fitting procedures before opening final confirmation.
4. Later add GCP and the advisor copilot according to the reference plan.

## Documentation and limits

- [Project brief](docs/project_brief.md)
- [Implementation plan](docs/implementation_plan.md)
- [Evaluation strategy](docs/evaluation.md)
- [Training protocol](docs/training_protocol.md)
- [Feature dictionary](docs/feature_dictionary.md)
- [Initial ranking results](docs/first_ranker_results.md)
- [Controlled LightGBM tuning](docs/lightgbm_tuning_results.md)
- [Stage-8 evaluation report](docs/stage8_evaluation_report.md)
- [Model card](docs/model_card.md)
- [Methodological decisions](docs/methodological_decisions.md)
- [Behavioral EDA summary](docs/eda_summary.md)
- [Data card](docs/data_card.md)
- [Data dictionary](docs/data_dictionary.md)

The project uses the [H&M recommendation benchmark](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/). Metadata observation dates, stock, price units, `FN`/`Active` semantics and channel business labels remain unresolved. Customer snapshot attributes are excluded from the initial feature matrix. Historical benchmark results do not establish performance in luxury retail or causal business benefits.
