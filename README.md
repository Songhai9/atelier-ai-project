# Atelier AI

Personalised fashion recommendations to support client advisors in preparing product selections.

## Business Objective

Atelier AI explores how purchase histories and product attributes can help a client advisor identify relevant articles for an individual customer. The intended application provides a ranked selection that the advisor can review and adapt before a customer interaction.

## Recommendation Task

At a historical prediction cutoff, recommend twelve distinct `article_id` values using information available before that cutoff. Evaluate recommendations against distinct articles purchased during the following seven days.

Purchases are treated as implicit relevance signals. An unpurchased article is not automatically considered an article the customer dislikes. Ranking scores are not interpreted as calibrated purchase probabilities.

**MAP@12** is the primary model-selection metric. Supporting diagnostics include Recall@12, candidate coverage and catalogue coverage.

## Current Status

The first fixed LightGBM LambdaRank experiment is trained, evaluated and exported alongside the three reference baselines:

- Data foundation, behavioral EDA, historical retrieval and seventeen cutoff-specific features are implemented.
- Notebook 04 retains four historical target weeks, 24 million raw labelled candidate rows and the complete reference populations.
- Training-only eligibility retains 1,775 complete groups / 532,500 rows, including 2,753 positives; no within-group sampling or additional sample weights are used.
- Notebook 05 trains 100 boosting iterations with a fixed initial configuration, reloads the native model and scores all 6 million validation candidate rows in batches.
- All four methods recommend twelve distinct articles to each of 20,000 recipients. Full-reference metrics include all 1,015 validation buyers and all 3,073 distinct customer-article targets, including retrieval misses.
- The recent-purchase baseline leads validation MAP@12 (0.022168), ahead of the first LightGBM ranker (0.020495). It remains the current leading selection under the documented rule.
- Paired customer and observed-history diagnostics, recommendations, full references, model/configuration and source snapshots are exported with checksums.
- Reusable functions and synthetic tests cover temporal boundaries, retrieval/features, labels, grouped inputs, exports and complete-reference ranking metrics.

See the [first-ranker results](docs/first_ranker_results.md) for exact scope, results and artifact locations. This is one fixed learned configuration on one validation week, without hyperparameter search. The local test remains unevaluated; no business impact or broader temporal robustness is established.

## Validated Data Scope

| Source file | Rows | Columns |
|---|---:|---:|
| `articles.csv` | 105,542 | 25 |
| `customers.csv` | 1,371,980 | 7 |
| `transactions_train.csv` | 31,788,324 | 5 |

Article and customer keys are complete and unique. All transaction references match their corresponding entity tables.

Transactions span **2018-09-20 through 2020-09-22**, with records on all 734 calendar dates. Repeated transaction rows are retained; distinct article relevance is defined separately for ranking evaluation.

These checks establish properties of the acquired files. They do not establish complete real-world transaction capture or historical availability of every metadata field.

## Temporal Evaluation

All intervals have an inclusive start and an exclusive end.

| Period | Start inclusive | End exclusive | Source transaction rows |
|---|---|---|---:|
| Pre-validation history | 2018-09-20 | 2020-09-09 | 31,292,772 |
| Validation target | 2020-09-09 | 2020-09-16 | 255,241 |
| Local test target | 2020-09-16 | 2020-09-23 | 240,311 |

The history period is a source pool. Supervised candidates are constructed at four historical cutoffs: 2020-08-12, 2020-08-19, 2020-08-26 and 2020-09-02. Their target windows end no later than validation starts. Features and candidates use transactions strictly before their own cutoff. Training-group selection is not applied to validation/test recipients or their full reference relevance.

For V1, selected learned parameters remain fixed between validation and test. Historical features and candidates are recomputed at each cutoff. Test outcomes are reserved for the final assessment after model-selection choices are frozen.

The complete protocol and conventions are recorded in the [evaluation strategy](docs/evaluation.md).

## Repository Structure

```text
atelier-ai-project/
├── docs/                    # scope, EDA, evaluation, training protocol, source manifest
├── notebooks/
│   ├── 01_data_foundation.ipynb
│   ├── 02_behavioral_eda.ipynb
│   ├── 03_candidates_and_features.ipynb
│   ├── 04_training_dataset.ipynb
│   └── 05_lightgbm_ranker.ipynb
├── src/atelier_ai/          # readers, retrieval/features, exports, ranking inputs and metrics
├── tests/                  # synthetic contract and boundary tests
├── pyproject.toml
└── uv.lock
```

The notebooks present analysis, interpretation and the preparation workflow. The Python package holds reusable computations; tests cover their contracts and boundary cases. Raw source files are stored locally under `data/raw/hm/` and excluded from version control.

## Local Setup

### Requirements

- Git, uv and Python 3.12
- VS Code with the Microsoft Python and Jupyter extensions
- Local access to the required competition CSV files

### Installation

```bash
git clone https://github.com/Songhai9/atelier-ai-project.git
cd atelier-ai-project
uv sync --locked
```

Dependencies are declared in `pyproject.toml` and resolved in `uv.lock`, including `lightgbm[scikit-learn]==4.7.0`. On macOS, the LightGBM wheel also requires the OpenMP runtime: install `libomp` with Homebrew (`brew install libomp`) before importing LightGBM. This native runtime is separate from Python package resolution.

### Data Preparation

Obtain the three required CSV files from the source competition under its access conditions, then place them at:

```text
data/raw/hm/articles.csv
data/raw/hm/customers.csv
data/raw/hm/transactions_train.csv
```

The [data card](docs/data_card.md) describes scope and handling. The [manifest](docs/data_manifest.json) identifies the acquired version used for the documented diagnostics.

### Notebook Execution

Run the notebooks in numerical order using the project environment. Notebook 03 creates the fixed cohort export used by notebook 04. In notebooks 04 and 05, set `VALIDATION_FEATURE_DATASET` to the complete validation export directory printed by notebook 03 if its fingerprint differs from the documented local default.

Select `.venv/bin/python` on macOS/Linux or `.venv\Scripts\python.exe` on Windows as the kernel. Restart the kernel, run all cells top-to-bottom and save the outputs. Outputs not yet saved should be regenerated before sharing a notebook as executed evidence.

Notebook 05 also loads the selected `RANKER_DATASET` export from notebook 04. Save its source before artifact exports so notebook snapshots capture the implemented workflow. Match the fingerprints recorded in each manifest rather than silently substituting a different dataset.

Complete-file diagnostics scan the transaction CSV several times. Notebook 04 materialises historical candidates in customer batches and loads only eligible groups for ranker inputs. Runtime and memory depend on local hardware; per-DataFrame estimates are not process peak-RAM measurements.

Derived datasets are written under `data/processed/` and remain Git-ignored. Typed Parquet files and manifests support verification/reuse of matching exports. Notebook code fingerprints change when source changes; new versions can coexist without overwriting older exports. Static Plotly rendering in the exploratory notebooks requires Chrome/Chromium supported by Kaleido.

### Checks

```bash
uv pip check
uv run pytest -q
```

The unit tests use small synthetic fixtures and do not require competition data. They complement the complete-file diagnostics recorded in the notebook.

## Roadmap

1. **Completed:** data acquisition, full-file diagnostics and chronological evaluation protocol.
2. **Completed:** behavioral EDA and modelling hypotheses.
3. **Completed:** recommendation baselines, historical candidate features/labels and an explicit grouped-ranking training contract.
4. **First reference experiment completed:** fixed LightGBM fitting, full-reference validation comparison, paired/history diagnostics and immutable evidence snapshots.
5. **Remaining modelling work:** controlled validation comparisons from the decision register, freeze the selected system, then assess the reserved local test. Candidate-budget sensitivity remains a plan, not a completed experiment.
6. **Later:** reproducible local application, PyTorch comparison, GCP deployment and a GenAI assistant for the advisor.

## Documentation

- [Project brief](docs/project_brief.md)
- [Evaluation strategy](docs/evaluation.md)
- [Training protocol](docs/training_protocol.md)
- [First-ranker results](docs/first_ranker_results.md)
- [Methodological decision register](docs/methodological_decisions.md)
- [Behavioral EDA summary](docs/eda_summary.md)
- [Data card](docs/data_card.md)
- [Data dictionary](docs/data_dictionary.md)
- [Acquisition manifest](docs/data_manifest.json)

## Data Source and Limits

The project uses the [H&M Personalized Fashion Recommendations dataset](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/). Raw competition data is excluded from the Git repository.

Unresolved limitations include metadata observation dates, the business definitions of `FN` and `Active`, price units and scaling, and sales-channel labels. The initial protocol uses one validation week and one local test week. Results on this source will not establish performance in luxury retail or a measurable commercial benefit.
