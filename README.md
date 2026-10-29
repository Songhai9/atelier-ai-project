# Atelier AI

Personalised fashion recommendations to help a client advisor prepare a ranked selection of twelve articles from observed customer purchase history.

## Task and approach

At each historical cutoff, retrieve 300 distinct candidate articles using earlier purchases, compute historical features, then rank a top-12 list. Evaluate against distinct purchases in the next seven days. Purchases are implicit relevance signals; scores are not calibrated purchase probabilities.

The project follows **data foundation → temporal protocol → behavioral EDA → candidates/features → reference methods → learned ranker → selection → local application**. The [implementation plan](docs/implementation_plan.md) records completed deliverables and the next increments.

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

The MAP difference against the validation-selected recent-purchases baseline is **0.000759**, with a paired percentile 95% interval **[-0.002626, 0.004346]** from 5,000 buyer resamples. This does not establish a clear V1 advantage. The baseline has slightly higher Recall and NDCG. Preserve the frozen configuration and report the result without test-driven retuning. V1 is retrospective; the later frozen final confirmation is reported below. See the [stage-8 evaluation report](docs/stage8_evaluation_report.md) and [model card](docs/model_card.md).

## Frozen final confirmation

The local PyTorch extension and the frozen five-system confirmation are complete. The window is **[2020-09-16, 2020-09-23)**. All 20,000 recipients receive twelve distinct articles from the same 300-candidate policy. Target-based means cover all 1,027 buyers and 3,267 distinct customer–article targets from 3,736 cohort purchase rows; 18,973 nonbuyers (94.865%) are excluded from these means, but receive lists. Retrieval misses remain in the reference sets.

| Method | MAP@12 | Macro Recall@12 | NDCG@12 | Retrieved targets / 3,267 |
|---|---:|---:|---:|---:|
| recent_popularity_7d | 0.006555 | 0.019679 | 0.012617 | 64 |
| preferred_product_types | 0.005354 | 0.015364 | 0.010033 | 45 |
| recent_purchases_180d | **0.020584** | 0.044484 | 0.030262 | 101 |
| lightgbm_leaves_63_min_20 | 0.017709 | 0.037696 | 0.026555 | 93 |
| pytorch_hybrid | 0.020575 | **0.046191** | **0.031193** | 108 |

The declared paired contrast, PyTorch hybrid minus LightGBM MAP@12, is **0.002866**, with percentile 95% interval **[-0.002320, 0.008160]** from 5,000 buyer resamples (seed 42). It does not establish superiority or equivalence. The recent-purchases baseline has the highest MAP point estimate; no baseline-versus-PyTorch paired contrast is reported. Final outcomes do not trigger retuning or change the recorded validation-selected nomination.

PyTorch scores the 2,406 recipients present in its final training vocabulary; the other 17,594 use the fixed recent-purchases fallback. The table therefore measures the complete hybrid system. The API/UI still serve the packaged **2020-09-09 LightGBM V1 snapshot**. See [PyTorch methods, artifacts and confirmation report](docs/pytorch_results.md).

## Temporal contract

| Role | Start inclusive | End exclusive | Source transaction rows |
|---|---|---|---:|
| EDA source-history pool | 2018-09-20 | 2020-09-02 | 31,028,115 |
| Validation | 2020-09-02 | 2020-09-09 | 264,657 |
| Intermediate V1 evaluation | 2020-09-09 | 2020-09-16 | 255,241 |
| Final LightGBM/PyTorch confirmation | 2020-09-16 | 2020-09-23 | 240,311 |

Intervals include their start and exclude their end. Table counts cover all source customers and retain repeated rows. Six historical cutoffs run weekly from 22 July through 26 August 2020. Candidates/features use only observations strictly before their own cutoff, with a 180-day personal lookback. The [evaluation strategy](docs/evaluation.md) defines fitting, selection, final refit and development-exposure limits. Intermediate V1 and frozen final confirmation have been evaluated; final outcomes are now exposed and must not be reused for tuning.

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
| [08 PyTorch ranker](notebooks/08_pytorch_ranker.ipynb) | Groupwise pairwise learning, validation selection, fixed-budget final refit and freeze |
| [09 Frozen confirmation](notebooks/09_final_confirmation.ipynb) | Independent artifact loading, complete-reference confirmation, paired uncertainty and MLflow |

Reusable computations live under `src/atelier_ai/`; synthetic tests under `tests/`; configuration under `configs/`. Raw data, prepared tables, generated figures, models and MLflow storage stay outside Git. Executed notebooks contain portable static figures alongside interactive Plotly outputs.

## Local reproduction

Use Python 3.12, Git and uv. Clone this repository and run `uv sync --locked`. On macOS, LightGBM needs the OpenMP runtime (`brew install libomp`). Plotly PNG export requires Chrome/Chromium supported by Kaleido.

Obtain the competition files under their access conditions and place `articles.csv`, `customers.csv` and `transactions_train.csv` in `data/raw/hm/`. The [acquisition manifest](docs/data_manifest.json) identifies the documented source version.

Notebooks 01-08 record the learning/training sequence. Notebook 09 starts in a fresh kernel without running cells from notebook 08; its default verifies and loads saved predictions and completed evaluation artifacts. Generated data and model artifacts are required and are intentionally not committed. Do not use Run All in notebook 08 merely to review final results. Run training notebooks deliberately with the project environment. The first notebook's complete-file diagnostics scan large sources; the following notebooks prepare and reuse typed Parquet. Feature exports are batched and verified. For a fresh derived-data campaign, run the whole learning sequence explicitly with:

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

## Frozen local pipeline

The seven local stages have been executed for `local_v1`: source validation, preparation, training-data assembly, frozen refit, V1 prediction, evaluation and a portable serving bundle. The native model and recommendation lists match the historical V1 campaign. The later PyTorch comparison is a separate offline campaign, outside this DVC graph and serving bundle.

The unified CLI provides `plan`, `status`, `verify`, individual stage commands and resumable `run`. `status` reads completion records; `verify` checks artifact identities and lineage. DVC declarations record dependencies with persistent local outputs. DVC 3.67.1 has executed the graph against the completed `local_v1` campaign: all seven stage outputs were verified and reused, the final integrity check passed, and `dvc.lock` records all eight stages.

```bash
.venv/bin/python -m atelier_ai.pipeline status --run-id local_v1
.venv/bin/python -m atelier_ai.pipeline verify --run-id local_v1
```

See [local pipeline commands, prerequisites and DVC setup](docs/local_pipeline.md).

## Local advisor interface

Streamlit calls the HTTP API for customer profiles, a selectable top-12 list with JSON export, side-by-side method comparison and aggregate V1 analysis. The interface displays the historical date and evaluation limits. It does not load local data or models.

Install both extras, start or restart the API using the command above, then launch the UI in another terminal:

```bash
uv sync --locked --extra api --extra ui
ATELIER_API_URL=http://127.0.0.1:8000 \
  .venv/bin/python -m streamlit run streamlit_app.py
```

Open [the advisor interface](http://127.0.0.1:8501). See [UI setup, workflow and verification](docs/ui.md). The complete synthetic suite passes 158 tests with both extras installed; browser checks against `local_v1` also verify the live UI and JSON export.

The project uses the [H&M recommendation benchmark](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/). Metadata observation dates, stock, price units, `FN`/`Active` semantics and channel business labels remain unresolved. Customer snapshot attributes are excluded from the initial feature matrix. Historical benchmark results do not establish performance in luxury retail or causal business benefits.
