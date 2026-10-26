# Atelier AI - Model Card

## Intended use and current evidence

A retrospective fashion recommendation prototype for a client advisor: rank twelve distinct articles per anonymised customer from 300 historical candidates. The frozen LightGBM configuration was selected on validation MAP@12. Its final V1 MAP point estimate is slightly higher than the selected baseline, but the paired interval includes zero; clear superiority is not established.

This artifact is not a calibrated purchase-probability model, current-stock system or measured business-impact intervention. The local FastAPI and Streamlit application serve the packaged historical LightGBM V1 bundle. A separate completed PyTorch/LightGBM confirmation is described below.

## Model and inputs

- LambdaRank gradient-boosted trees; 63 maximum leaves, maximum depth 8, minimum 20 examples per leaf, learning rate 0.05, L2=1, seed 42, four worker threads, deterministic column-wise fitting.
- Eight boosting iterations frozen before final refit; no V1 early stopping.
- Exactly 31 ordered numeric/boolean features, defined in the [feature dictionary](feature_dictionary.md).
- 180-day personal history and global historical aggregates; retrieval uses 30 personal articles, global seven-day top 100, 50 articles per three preferred types, then 28-day fill to 300.
- String IDs remain keys; snapshot customer attributes are excluded. Missing recency/rank values preserve unobserved-history semantics.

## Training and temporal scope

Final fitting cutoffs: 2020-07-22, 07-29, 08-05, 08-12, 08-19, 08-26 and 09-02. All histories precede their own cutoff and all target windows end by 2020-09-09. Of 140,000 audited customer-cutoff groups, 2,974 complete informative groups retain 892,200 candidate rows, including 4,461 observed positives. Eligibility is fitting-only; no within-group negative sampling is used.

The model is newly fitted on training plus former validation, rather than continued from the six-week selection model. Native reload checks on 1,000 deterministic training probes have maximum absolute error 0.0; this is a serialization check.

## Evaluation

Intermediate V1 [2020-09-09, 2020-09-16): 20,000 recipients, 1,015 target buyers, 3,073 full distinct targets and 3,454 source purchase rows. Nonbuyers: 18,985 (94.925%), excluded from target-based means but still given lists.

| Metric | Frozen LightGBM | Recent-purchases baseline |
|---|---:|---:|
| MAP@12 | 0.022927 | 0.022168 |
| Macro Recall@12 | 0.041459 | 0.044635 |
| NDCG@12 | 0.031957 | 0.032089 |
| Retrieved target pairs / 3,073 | 105 | 111 |
| Catalogue coverage / 50,383 articles | 5,944 (11.7976%) | 16,857 (33.4577%) |
| Mean broad product groups per list / 20,000 recipients | 3.33075 | 2.70180 |

Shared candidate coverage is 618/3,073 (20.1106%). Full-reference evaluation includes retrieval misses. Paired buyer bootstrap (5,000 resamples, seed 42) gives difference 0.000759, percentile 95% interval [-0.002626, 0.004346]. This is an inconclusive superiority comparison, not an equivalence result.

## Diagnostics and performance scope

Gain importance, native SHAP examples, feature ablations, errors and observed-history segments in notebook 07 refer to the pre-refit selected model, not the final fitted weights. The full feature variant has the highest validation MAP among declared ablations. Segments are descriptive, with small evaluated groups.

Recorded local validation inference stages sum to 59.07 s on 20,000 recipients / six million pairs (features 39.53 s, retrieval 15.57 s, scoring 0.90 s). This is one pre-refit validation pass, excluding several stages and with uncontrolled caches. Final fit-only time is 0.952 s. No API latency or production capacity claim follows.

## Limitations

- Validation and intermediate V1 were exposed during development; results are retrospective.
- Customer intervals condition on fixed models and one week, without training/selection/temporal uncertainty.
- Snapshot category/colour metadata lacks historical observation dates; no stock-availability claim is supported.
- The fixed cohort comes from customer snapshot IDs; historical account existence at every cutoff is not established.
- Purchase absence is not disinterest; no observed 180-day history is not lifetime newness.
- Historical benchmark data does not establish generalization to luxury retail, fairness across unmeasured groups, or causal commercial benefit.
- Use the frozen configuration for the recorded V1; do not select a replacement based on V1 outcomes.

## Identity and lifecycle

- Final model SHA-256: `d6ca939c488a707c9a8c5be7480f738df640d2bc0a3496fdcb20b573f8353bf3`.
- Final MLflow run: `8f90b653b86543f98358903363295f0e`.
- Frozen decision: `data/processed/weekly_v1/final_selection.json`.
- Training manifest: `data/processed/weekly_v1/final_training/manifest.json`.
- Model and evaluation manifests: `final_ranker/manifest.json` and `v1_evaluation/evaluation_manifest.json` under the same data root.
- Source hashes and MLflow notebook snapshots identify executed code, including saved working-tree changes; a Git base alone is not proof of a clean training tree.
- V1 and the frozen final confirmation [2020-09-16, 2020-09-23) are evaluated. Earlier protocol files retain their freeze-time status; do not rewrite them.

See the [stage-8 report](stage8_evaluation_report.md) and [evaluation strategy](evaluation.md).

## Later offline neural comparison

The final PyTorch hybrid uses customer/article embeddings (32 each), 37 numeric inputs, 128/64 ReLU layers and dropout 0.2. It was newly refitted for the validation-selected three epochs on the same seven fitting weeks, ending before 2020-09-09. The complete hybrid includes recent-purchases fallback for unknown customer IDs; unknown articles use the reserved zero embedding and historical numeric features.

On final confirmation, LightGBM MAP@12 is 0.017709, PyTorch hybrid 0.020575 and recent purchases 0.020584 over the same 1,027 buyers / 3,267 full targets. The paired PyTorch-minus-LightGBM interval [-0.002320, 0.008160] contains zero. Neither learned model leads the baseline on MAP point estimate; no equivalence claim follows. No final-week retuning or serving replacement was performed. See the [full neural/confirmation report](pytorch_results.md) for scope, identities, fallback proportions and reproducibility limits.
