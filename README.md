# Atelier AI

Personalised fashion recommendations to support client advisors in preparing product selections.

## Business Objective

Atelier AI explores how purchase histories and product attributes can help a client advisor identify relevant products for an individual customer.

The intended application provides a ranked selection that the advisor can review and adapt before a customer interaction.

## Recommendation Task

At a historical reference date, recommend twelve distinct products using only information available before that date.

Recommendation quality will be evaluated against purchases observed during the following seven days.

Purchases are treated as implicit relevance signals. An unpurchased product is not automatically considered a product the customer dislikes.

## Planned Approach

1. Inspect data quality and establish temporal evaluation periods.
2. Explore purchasing patterns and formulate modelling hypotheses.
3. Generate candidate products and construct customer–product features.
4. Compare simple recommendation baselines with a LightGBM ranking model.
5. Evaluate ranking quality, interpret results and analyse limitations.
6. Package the selected approach into a reproducible local application.

MAP@12 is the primary model-selection metric. Supporting diagnostics include recall, candidate coverage and catalogue coverage.

## Current Status

Business framing and the evaluation strategy are documented.

The Python environment is configured with uv. Package imports and the
notebook's execution environment have been verified locally in VS Code.

Data analysis, model training and application development are planned.
No predictive performance or business impact is reported at this stage.

## Local Setup

### Requirements

- Git
- uv
- Python 3.12
- VS Code with the Microsoft Python and Jupyter extensions

### Installation

```bash
git clone https://github.com/Songhai9/atelier-ai-project.git
cd atelier-ai-project
uv sync --locked
```

Dependencies are declared in `pyproject.toml` and resolved in `uv.lock`.
The local `.venv` directory is excluded from version control.

### Notebook

Open the repository folder in VS Code, then open
[the main notebook](notebooks/atelier_ai.ipynb).

Use the notebook kernel picker to select the Python interpreter located
at `.venv/bin/python` on macOS/Linux or `.venv\Scripts\python.exe`
on Windows.

Restart the kernel and run all cells from top to bottom. The environment
cell should report the project's virtual environment and an
`atelier_ai` package location inside `src/atelier_ai`.

### Dependency Check

```bash
uv pip check
```

## Roadmap

- **Local V1:** exploratory analysis, feature engineering, baselines, LightGBM, experiment tracking, tests and a local demonstration.
- **Subsequent milestones:** PyTorch comparison, GCP deployment and a GenAI assistant for the advisor.

## Documentation

- [Project brief](docs/project_brief.md)
- [Evaluation strategy](docs/evaluation.md)

## Data Source

The project will use the [H&M Personalized Fashion Recommendations dataset](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/).

Raw competition data will not be included in the Git repository.