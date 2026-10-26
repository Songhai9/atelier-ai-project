# Frozen local pipeline

The seven local stages replay the frozen scientific decision into an isolated campaign under `data/processed/pipeline_runs/<run_id>/`. The historical notebook campaign remains input evidence. No configuration selection or final-confirmation evaluation occurs here.

## Prerequisites

Use the existing Python 3.12 environment from `uv sync --locked`, the three documented CSV files in `data/raw/hm/`, and the recorded `docs/data_manifest.json` and `configs/temporal.yaml`.

Preserve these two exact local scientific inputs from the completed notebook campaign:

- `data/processed/weekly_v1/final_selection.json`;
- `data/processed/weekly_v1/diagnostics/lightgbm_leaves_63_min_20/segmentation_manifest.json`, whose checksum fixes the validation-selected comparator.

They are required inputs, not regenerated decisions. A fresh clone needs these recorded inputs as well as the source data. Large data, generated artifacts and MLflow storage remain excluded from Git.

## Unified commands

Run commands from the repository root. The original per-module commands remain available.

```bash
.venv/bin/python -m atelier_ai.pipeline plan --run-id local_v1
.venv/bin/python -m atelier_ai.pipeline status --run-id local_v1
.venv/bin/python -m atelier_ai.pipeline verify --run-id local_v1
```

`status` reads only stage completion records. It explicitly reports that artifact verification was not performed. `verify` checks all raw-source identities, complete stage records, prepared data, training exports, model reloads, predictions, evaluation replay, serving contents and local MLflow artifacts. It writes no campaign artifacts.

Execute all stages, or stop after an explicit stage:

```bash
.venv/bin/python -m atelier_ai.pipeline run --run-id local_v1
.venv/bin/python -m atelier_ai.pipeline run --run-id local_v1 --until prepare-sources
```

The available stage commands are `validate-sources`, `prepare-sources`, `build-training`, `fit-model`, `predict-v1`, `evaluate-v1` and `package-service`. Each command uses the same implementation and manifests as the corresponding original module. Completed valid artifacts are verified and reused. Interruptions resume through the existing stage logic; incompatible artifacts are preserved and rejected. There is no force-overwrite option.

For changed computation code, inputs or library versions, choose a fresh run ID. A new campaign regenerates its own prepared data and exports. Updating only this orchestration layer does not invalidate unchanged scientific artifacts.

Progress goes to stderr; the final JSON report goes to stdout. A failure stops the remaining stages and exits nonzero.

## DVC dependency graph

`dvc.yaml` declares the seven stages and a final integrity check. `configs/pipeline.yaml` selects the campaign ID for DVC; the direct Python CLI always uses its explicit `--run-id` argument.

DVC is an isolated tool, separate from the frozen model environment. The pinned version below is available in the [official releases](https://github.com/treeverse/dvc/releases/tag/3.67.1):

```bash
uvx --from dvc==3.67.1 dvc init
uvx --from dvc==3.67.1 dvc dag
uvx --from dvc==3.67.1 dvc repro --dry
uvx --from dvc==3.67.1 dvc repro
```

Initialize once; omit `init` when `.dvc/` already exists. These commands require network access the first time the isolated DVC tool is obtained. They do not install DVC into `.venv` or change `uv.lock`.

Every output uses `persist: true`, so DVC does not remove existing campaign artifacts before a stage command. `cache: false` keeps large local outputs out of the DVC object cache; this setup records the graph and local checksums, and does not provide artifact download or remote backup. See the [official output-field documentation](https://doc.dvc.org/user-guide/project-structure/dvcyaml-files).

The final `verify_pipeline` stage is always changed intentionally: every `dvc repro` finishes with full integrity checks, including dependencies in local MLflow that are not independent DVC outputs. `dvc status` may therefore list this verification stage as always changed even when scientific outputs are current.

The first `dvc repro` registers existing valid outputs through the same reusable stage commands. Keep `dvc.yaml`, `dvc.lock`, `configs/pipeline.yaml`, `.dvcignore` and the configuration files created by `dvc init` in Git; do not hand-write `dvc.lock`. Generated data and MLflow artifacts remain local.

## Verified local campaign

DVC 3.67.1 was initialized and executed for `local_v1`. The real `dvc repro` reused each of the seven complete stage outputs, then passed `verify_pipeline` with `artifacts_verified: true` and seven verified stages. The generated lock uses schema 2.0 and records all eight stage commands for `local_v1`. The model identity and reserved-confirmation state are unchanged.

This DVC run validates reuse and integrity of the existing campaign. It is not a new cold rebuild through DVC: the real artifacts were generated earlier through the individual pipeline stages. Synthetic tests also cover a fresh unified run, partial resume, failure handling and complete reuse; the full Python suite passed 134 tests at pipeline closure, 144 after the API increment and 158 with the API and UI extras installed.

The recorded `local_v1` campaign produced 42 million candidate rows, 2,974 informative groups and 892,200 fitting rows. The saved eight-tree model has SHA-256 `d6ca939c488a707c9a8c5be7480f738df640d2bc0a3496fdcb20b573f8353bf3`, identical to the historical refit.

V1 evaluates 1,015 buyers against 3,073 complete target pairs. MAP@12 is 0.02292699; the paired difference against the validation-selected recent-purchases reference is 0.00075868, with 95% interval [-0.00262641, 0.00434631]. This does not establish clear superiority. V1 has prior development exposure; final confirmation remains reserved.

The portable serving bundle contains 20,000 historical customer profiles, 50,383 observed catalogue articles, 113,678 preceding purchase rows and 240,000 learned recommendations at reference date 2020-09-09. Its reader works without raw data or MLflow. Unknown IDs have no profile and receive an explicit popularity fallback. Customer snapshot attributes are excluded; stock availability, price units and calibrated purchase probabilities are not established.

The application milestone now includes [FastAPI](api.md) and the [Streamlit advisor interface](ui.md). Containers and synthetic CI remain to implement. Full-source training stays an explicit local action.

## Later offline confirmation scope

This page documents the packaged 2020-09-09 LightGBM V1 snapshot and its completed local integration. Its stored captions/protocol fields predate the later frozen comparison. Final confirmation [2020-09-16, 2020-09-23) is now evaluated separately in notebook 09; neither PyTorch nor its final-week predictions are integrated into this serving bundle or DVC graph. See [the current neural/confirmation report](pytorch_results.md).
