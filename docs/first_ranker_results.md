# Atelier AI - First Ranker Results

## Scope and conclusion

The first fixed LightGBM LambdaRank configuration is trained, reloaded and compared with the three existing baselines in [notebook 05](../notebooks/05_lightgbm_ranker.ipynb). The recent-purchase baseline leads this validation comparison and remains eligible for the final system under the existing selection rule. The local test has not been evaluated.

This milestone establishes a runnable training/evaluation workflow and preserved evidence. It does not establish an optimised model, general temporal robustness, calibrated purchase probabilities or business impact.

## Population, time and denominators

| Quantity | Definition | Count |
|---|---|---:|
| Validation cutoff | Features/candidates use transactions strictly before 2020-09-09 | One cutoff |
| Target interval | 2020-09-09 inclusive to 2020-09-16 exclusive | Seven days |
| Recipients | Fixed cohort, every method | 20,000 |
| Candidate rows | 300 distinct articles per recipient | 6,000,000 |
| Final recommendation rows | Twelve articles per recipient, per method | 240,000 |
| Evaluated buyers | Recipients with at least one observed target article | 1,015 |
| Excluded from target metrics | No observed purchase in the target week; still recommended to | 18,985 (94.925%) |
| Source target rows | Retained cohort purchase rows, including repetitions | 3,454 |
| Distinct relevant pairs | Complete customer-article target reference | 3,073 |

MAP and macro Recall have 1,015 equally weighted buyer scores as their denominator. Micro Recall divides pooled hits by 3,073 target pairs. All missing-candidate targets remain in the reference; buyers with no retrieved positives are not removed.

## Training configuration

Four historical cutoffs (2020-08-12, 2020-08-19, 2020-08-26, 2020-09-02) precede validation. The training-only policy retains 1,775 complete groups / 532,500 rows, with 2,753 positives and 529,747 zeros. X has seventeen ordered historical features. These selected groups are not representative of all recipient weeks.

Experiment `lightgbm_lambdarank_v1_fixed100` uses LightGBM 4.7.0, binary LambdaRank, 100 iterations, learning rate 0.05 and up to 15 leaves per tree. It uses query normalisation, `label_gain=[0, 1]` and truncation level 15. Missingness is preserved. No extra sample weights, within-group sampling, learned preprocessing, validation early stopping, hyperparameter search or validation refit is introduced. Full parameters and versions are in model metadata and the experiment manifest. See the [training protocol](training_protocol.md).

The native saved model has 100 iterations and seventeen features. Scoring reloads that file; ties use score descending, then lexical article ID ascending.

## Full-reference validation results

All scores below are fractions, not percentages. Printed precision is nine decimal places; exported tables retain full floating-point precision.

| Method | MAP@12 | Macro Recall@12 | Micro Recall@12 | Retrieved targets |
|---|---:|---:|---:|---:|
| Recent purchases + popularity | 0.022168303 | 0.044635073 | 0.036121054 | 111 |
| LightGBM LambdaRank fixed100 | 0.020495038 | 0.037645798 | 0.029287341 | 90 |
| Recent popularity | 0.005442630 | 0.019836223 | 0.018548650 | 57 |
| Dominant group + popularity | 0.004458753 | 0.016324611 | 0.015294500 | 47 |

Model minus the leading baseline MAP@12 is **-0.0016732647833025346**. LightGBM retrieves 90 / 3,073 target pairs versus 111 / 3,073 for the baseline. The baseline remains the current leader; no test outcome or alternate metric overrides this primary-metric comparison.

## Paired and observed-history diagnostics

At an AP-difference tolerance of 1e-12, 37 of 1,015 buyers improve, 926 are unchanged and 52 worsen. Unchanged AP does not imply identical recommendations, and this count is not a significance test.

History membership uses the same 180-day pre-cutoff window: `[2020-03-13, 2020-09-09)`. No observed history does not mean a lifetime-new customer.

| Observed history | Buyers | Full target pairs | Baseline MAP@12 | Model MAP@12 | Baseline hits | Model hits |
|---|---:|---:|---:|---:|---:|---:|
| None in 180 days | 187 | 594 | 0.002338 | 0.003681 | 7 | 7 |
| Present in 180 days | 828 | 2,479 | 0.026647 | 0.024292 | 104 | 83 |

The first segment's positive MAP difference contributes approximately +0.000247 to the overall difference; the second contributes approximately -0.001921. Contributions use the segment AP-difference sum divided by all 1,015 buyers. Rounded displayed contributions may not sum exactly; the saved unrounded values reconcile.

The smaller no-history segment has identical hit totals but different ranking scores: AP rewards the positions of relevant articles as well as their retrieval. These segment summaries describe where performance differs; they do not identify a cause.

## Evidence locations and identity

All paths below are repository-relative and Git-ignored.

| Artifact | Location |
|---|---|
| Selected training export | `data/processed/ranker_training/ranker_v1_5e8154a40689/` |
| Complete validation features | `data/processed/features/validation_2020-09-09_1e4ed94c9d17/` |
| Native model and metadata | `data/processed/models/lightgbm_lambdarank_v1_fixed100_86c845e1cc70/` |
| Verified evidence snapshot | `data/processed/experiments/lightgbm_lambdarank_v1_fixed100/20261024T191410Z_740259cd/` |

Training dataset fingerprint: `5e8154a40689a6049fd808fa0b02b2a4e61c422391b3ed42f12750cb4d0387e9`.

Validation dataset fingerprint: `1e4ed94c9d1716f024c3105f0a095fe93952568a098f51645357d6efcf476d5a`.

Model SHA-256: `199f2b110fe74c96703435b1b899f2bc6d6d87ed11bb37a553fbc42c6409306c`.

The snapshot contains twelve Parquet tables: four full recommendation tables, pooled per-buyer scores, method comparison, source target transactions, complete relevance, recipient population, paired history scores, paired outcome aggregates and history-segment aggregates. It also copies model/configuration, the saved notebook, dependency files, source modules and upstream manifests. File size and SHA-256 checks cover all 31 listed files.

Each Parquet export is checked against its in-memory input using bidirectional `EXCEPT ALL`. Full-precision tables are authoritative for numerical reconciliation. A staged temporary directory is renamed to its final location after successful export, and each export has a new snapshot ID. A new export is not a new model-training experiment.

The recorded Git HEAD is `ab19dc575e82b1d19b087cd55f873dcc7737c002`, with `working_tree_clean=false`. The copied sources capture the local uncommitted implementation; the Git commit alone does not contain the first-ranker code. Data/model artifacts remain excluded from Git.

The captured fit duration is approximately 1.225 seconds; validation reading, checksum checks, scoring and ranking together take approximately 5.873 seconds in this local run. These are observed timings, not isolated prediction latency, peak-memory measurements or production guarantees.

## Reproduction and remaining work

Use the locked project environment and the matched upstream exports. Run notebooks in numerical order, preserve source before export and save executed outputs. Notebook 05 reconstructs inputs, fits a fresh fixed estimator, saves/reloads it, generates full-cohort recommendations, evaluates complete references, compares the three baselines and exports the evidence. For DuckDB 2.0, consume aggregate-check results with `fetchall()` before further operations on the connection.

The completed stage is the first fixed reference experiment. Controlled validation comparisons from the [decision register](methodological_decisions.md), catalogue-coverage reporting and broader temporal assessment remain outstanding. Do not report candidate-budget sensitivity as completed. Freeze the selected system before opening the local test `[2020-09-16, 2020-09-23)`; recompute histories/features/candidates at its cutoff, without fitting learned parameters on validation labels.
