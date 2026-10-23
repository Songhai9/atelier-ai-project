# Atelier AI

Personalised fashion recommendations to support client advisors in preparing product selections.

## Business Objective

Atelier AI explores how purchase histories and product attributes can help a client advisor identify relevant articles for an individual customer. The intended application provides a ranked selection that the advisor can review and adapt before a customer interaction.

## Recommendation Task

At a historical prediction cutoff, recommend twelve distinct `article_id` values using information available before that cutoff. Evaluate recommendations against distinct articles purchased during the following seven days.

Purchases are treated as implicit relevance signals. An unpurchased article is not automatically considered an article the customer dislikes. Ranking scores are not interpreted as calibrated purchase probabilities.

**MAP@12** is the primary model-selection metric. Supporting diagnostics include Recall@12, candidate coverage and catalogue coverage.

## Current Status

The data foundation and initial temporal evaluation protocol are implemented:

- Source acquisition is documented with file sizes and SHA-256 hashes.
- Complete-file diagnostics cover schemas, entity keys, references, missingness, categories, numeric values, classification mappings and repeated transaction patterns.
- Temporal coverage and local validation/test boundaries are established.
- Reusable reading and diagnostic functions are extracted into `src/atelier_ai` and covered by automated tests.
- The main notebook contains executed diagnostics, aggregate results and Plotly figures with static PNG outputs.
- Behavioral EDA has started with an explicit pre-validation scope, measurement definitions and a reconciled daily transaction-row aggregate.

Behavioral EDA covers customer activity, article popularity and transaction-row concentration within the declared pre-validation history. Validated article-metadata joins support descriptive product-group and product-type summaries, with reconciled tables and Plotly figures. Repeat-purchase patterns and temporal variation are the next analyses. Feature engineering, recommendation baselines, model training and application development remain planned. No recommendation performance or business impact is reported yet.

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

The history period is a source pool; supervised examples have not yet been constructed. Features and candidates at a prediction cutoff use earlier transactions only. Historical training target windows must end no later than validation starts.

For V1, selected learned parameters remain fixed between validation and test. Historical features and candidates are recomputed at each cutoff. Test outcomes are reserved for the final assessment after model-selection choices are frozen.

The complete protocol and conventions are recorded in the [evaluation strategy](docs/evaluation.md).

## Repository Structure

```text
atelier-ai-project/
├── docs/
│   ├── project_brief.md
│   ├── evaluation.md
│   ├── data_card.md
│   ├── data_dictionary.md
│   └── data_manifest.json
├── notebooks/
│   ├── data_foundation.ipynb
│   └── behavioral_eda.ipynb
├── src/atelier_ai/
│   ├── __init__.py
│   ├── data_io.py
│   ├── data_quality.py
│   ├── temporal.py
│   └── transactions.py
├── tests/
│   ├── test_data_io.py
│   ├── test_data_quality.py
│   ├── test_temporal.py
│   └── test_transactions.py
├── pyproject.toml
└── uv.lock
```

The notebook presents the analysis and its interpretation. The Python package holds reusable computations; tests cover their contracts and boundary cases. Raw source files are stored locally under `data/raw/hm/` and excluded from version control.

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

Dependencies are declared in `pyproject.toml` and resolved in `uv.lock`.

### Data Preparation

Obtain the three required CSV files from the source competition under its access conditions, then place them at:

```text
data/raw/hm/articles.csv
data/raw/hm/customers.csv
data/raw/hm/transactions_train.csv
```

The [data card](docs/data_card.md) describes scope and handling. The [manifest](docs/data_manifest.json) identifies the acquired version used for the documented diagnostics.

### Notebook Execution

Open [the main notebook](notebooks/data_foundation.ipynb) in VS Code. Select `.venv/bin/python` on macOS/Linux or `.venv\Scripts\python.exe` on Windows as the kernel.

Continue with [the behavioral EDA notebook](notebooks/behavioral_eda.ipynb). Each notebook runs independently from a fresh kernel using the project environment.

Restart the kernel and run all cells from top to bottom, then save the outputs. The environment cell should report the project virtual environment and an `atelier_ai` package location inside `src/atelier_ai`.

Complete-file diagnostics scan the transaction CSV several times. Their runtime depends on local storage and hardware. Plotly figures include PNG outputs; static rendering requires a Chrome/Chromium installation supported by Kaleido.

### Checks

```bash
uv pip check
uv run pytest -q
```

The unit tests use small synthetic fixtures and do not require competition data. They complement the complete-file diagnostics recorded in the notebook.

## Roadmap

1. **Completed:** data acquisition, structural diagnostics, data-quality checks, initial temporal protocol and extraction of reusable functions.
2. **In progress:** behavioural EDA on pre-validation history; customer activity, article popularity and category summaries are implemented, with repeat-purchase patterns and temporal variation next.
3. **Planned:** candidate generation, features, baselines and a LightGBM ranking model.
4. **Planned:** experiment tracking, evaluation, interpretation and a reproducible local demonstration.
5. **Subsequent milestones:** PyTorch comparison, GCP deployment and a GenAI assistant for the advisor.

## Documentation

- [Project brief](docs/project_brief.md)
- [Evaluation strategy](docs/evaluation.md)
- [Data card](docs/data_card.md)
- [Data dictionary](docs/data_dictionary.md)
- [Acquisition manifest](docs/data_manifest.json)

## Data Source and Limits

The project uses the [H&M Personalized Fashion Recommendations dataset](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/). Raw competition data is excluded from the Git repository.

Unresolved limitations include metadata observation dates, the business definitions of `FN` and `Active`, price units and scaling, and sales-channel labels. The initial protocol uses one validation week and one local test week. Results on this source will not establish performance in luxury retail or a measurable commercial benefit.
