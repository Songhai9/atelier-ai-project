# Atelier AI - Training Protocol

## Temporal inputs

Read dates from `configs/temporal.yaml`; use six historical cutoffs from 2020-07-22 through 2020-08-26. All targets are seven days; all personal feature histories are 180 days strictly before their own cutoff. Training targets end no later than 2020-09-02.

The fixed cohort SHA-256 is `5dcab30d63b42ad6f677fcec8cefade3964300890bba58cac12351060ed02b41`. Cohort selection uses seed 42 and source IDs, never future purchases. Global historical aggregates include source customers outside the cohort.

## Retrieval before labels

For each cutoff, retrieve 30 recent distinct personal articles, global 7-day top 100, 50 articles from each of three preferred product types, then fill to 300 distinct articles using 28-day popularity. Tie ordering and feature definitions are specified in [feature dictionary](feature_dictionary.md). Only articles observed before the cutoff are eligible.

Label 1 when a generated article occurs in that customer's next-week relevance set, otherwise 0. Do not insert missed future positives. Keep full future reference targets separately. Repeated source purchases are retained in transaction counts but deduplicated for article relevance.

## Complete groups and fitting eligibility

| Quantity | Executed count |
|---|---:|
| Historical target weeks | 6 |
| Complete customer/week groups | 120,000 |
| Raw labelled candidate rows | 36,000,000 |
| Informative fitting groups | 2,533 |
| Fitting rows | 759,900 |
| Positive fitting rows | 3,787 |
| Numeric/boolean features | 31 |

Retain complete training groups only when they contain both positive and negative labels. Constant-label groups remain in the audit; no within-group negative sampling or extra weights are used. This target-dependent eligibility is restricted to training. It must never alter validation recipients or full-reference denominators.

Sort rows by cutoff, customer ID and article ID. The ordered X feature columns exclude keys and label. y is binary. Contiguous group sizes sum exactly to the fitting row count. Counts/shares/flags use the declared zero policy; unobserved recencies/ranks remain missing. No fitted categorical encoder or scaler is needed by this numeric tree model.

## Initial learned configuration

LambdaRank, learning rate 0.05, at most 1,000 iterations, 31 leaves, maximum depth 8, minimum 100 examples per leaf, L2=1, seed 42, four worker threads and deterministic column-wise fitting. Stop after 50 iterations without improved full-reference validation NDCG@12.

The initial run selected iteration 16. Training uses the six historical weeks only. Save a native model and verify reloaded predictions numerically. This run is not the final train-plus-validation refit.

The planned search is six combinations of leaves {15,31,63} × minimum examples {20,100}, with common candidates, features, metric conventions and stopping. Final choice uses complete-reference MAP@12; model capacity is not chosen from test outcomes.

## Artifacts and reproducibility

`preparation.py` creates typed local Parquet sources. Feature exports use customer-complete batches of 500 with explicit SQL types, exact round-trip checks, per-file checksums, source/cohort/configuration/code identities and distinct output fingerprints. Matching complete exports are verified before reuse.

The active run indices live under `data/processed/weekly_v1/`: `validation.json`, `training.json` and `first_ranker/manifest.json`. Each selected dataset is identified explicitly rather than by an old hardcoded export directory. Raw and derived data and models remain Git-ignored.

MLflow records the initial run locally, including configuration, identities, metrics, duration, model and learning data. The run ID is `ad9a6d63d33245a68346b43d986d80a6`. The manifest records the Git base and hashes of the working-tree source files actually used; a Git base alone is not presented as proof of a clean training tree.

## Next model work

Complete the six-configuration comparison, SHAP examples and declared ablations; assess paired/history diagnostics, timings and customer-bootstrap uncertainty. Freeze selection and iteration budgets before a final train-plus-validation refit and V1 evaluation. Final confirmation remains reserved for a frozen LightGBM/PyTorch comparison. See [evaluation strategy](evaluation.md) for prior development exposure and interpretation limits.
