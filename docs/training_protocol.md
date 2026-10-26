# Atelier AI - Training Protocol

## Temporal inputs

Read dates from `configs/temporal.yaml`. Configuration selection uses six historical cutoffs from 2020-07-22 through 2020-08-26, whose targets end no later than 2020-09-02. All targets are seven days; all personal feature histories are 180 days strictly before their own cutoff. The final refit adds cutoff 2020-09-02, whose target window ends at 2020-09-09.

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

The six combinations of leaves {15,31,63} × minimum examples {20,100} have been executed with common candidates, features, metric conventions and stopping. Complete-reference validation MAP@12 selects 63/20 with best iteration 8 as the reference candidate for stage 8. The 63/100 configuration leads validation NDCG and Recall, but not the declared selection metric. See [tuning results](lightgbm_tuning_results.md).

## Artifacts and reproducibility

`preparation.py` creates typed local Parquet sources. Feature exports use customer-complete batches of 500 with explicit SQL types, exact round-trip checks, per-file checksums, source/cohort/configuration/code identities and distinct output fingerprints. Matching complete exports are verified before reuse.

The active run indices live under `data/processed/weekly_v1/`: historical `validation.json`, `training.json`, initial/search manifests, and the stage-8 `final_selection.json`, `final_training/manifest.json`, `final_ranker/manifest.json` and `v1_evaluation/evaluation_manifest.json`. Each selected dataset is identified explicitly rather than by an old hardcoded export directory. Raw and derived data and models remain Git-ignored.

MLflow records the initial experiment and all six search trials locally, including configuration, identities, metrics, duration, models, learning data and the comparison figure. Run IDs are read from the corresponding manifests. The selected search run is `b76ec3a4783c4b7296a07321d032df31`. The manifest records the Git base and hashes of the working-tree source files actually used; a Git base alone is not presented as proof of a clean training tree.

## Frozen final refit and V1

Stage 8 is complete. `final_selection.json` fixes the 63-leaf / minimum-20 configuration, all 31 ordered features and eight boosting iterations before refitting. The final fit uses the six historical cutoffs plus 2020-09-02. Every group's features retain their own cutoff; the last target window ends at 2020-09-09.

All 140,000 groups remain in the audit. The same training-only eligibility rule retains 2,974 complete groups / 892,200 rows / 4,461 positives. The added week contributes 441 groups / 132,300 rows / 674 positives. No V1 or confirmation target enters this fit.

Fit a new estimator for eight iterations without an evaluation set or early stopping. The saved final model has eight trees and 31 features. Scores before/after native reload agree on 1,000 deterministic probe rows, with maximum absolute error 0.0. Local fit-only time is 0.952 s, excluding data preparation and artifact checks.

Final MLflow run: `8f90b653b86543f98358903363295f0e`; final model SHA-256: `d6ca939c488a707c9a8c5be7480f738df640d2bc0a3496fdcb20b573f8353bf3`. The pre-refit selection model and its diagnostics retain their own identity. V1 and the later frozen final confirmation have been evaluated without changing the final LightGBM weights.

Historical candidate manifests remain records of earlier milestones. The frozen decision and final-fit/evaluation manifests describe the later lifecycle; do not rewrite the historical candidate file, whose hash is part of the frozen evidence.

Preserve completed frozen campaigns. Timing and ablation re-execution produces new evidence even when configurations are identical, so full reproduction uses fresh derived-data storage. A fresh top-to-bottom replay was not performed during this closure.

The local DVC pipeline, FastAPI and Streamlit integration are complete for the historical LightGBM V1 snapshot. The later neural experiment remains offline. See the [evaluation report](stage8_evaluation_report.md) and [evaluation strategy](evaluation.md).

## Local neural ranker procedure

Notebook 08 uses the same initial six-week fitting groups (2,533 groups / 759,900 rows / 3,787 positives), the 31 ordered numeric features and full-reference validation. Numeric preprocessing is fitted only on these fitting rows: median imputation, standard scaling and six fixed missingness flags give 37 numeric inputs. Customer/article vocabularies are fitted on training candidate rows, with index zero reserved for unseen IDs.

Two 32-dimensional embeddings concatenate with the 37 numeric inputs, giving 101 inputs to Linear(101,128), ReLU, dropout 0.2, Linear(128,64), ReLU, dropout 0.2 and Linear(64,1). The output is an unrestricted ranking score. `src/atelier_ai/neural_ranker.py` owns the shared architecture and identifier encoding.

For each informative group, compute `softplus(score_negative - score_positive)` over all positive–negative pairs. Average within the group, then across the batch's groups. There is no negative sampling; a group contains all 300 candidates. AdamW uses learning rate 0.001, weight decay 0.01, seed 42 and batches of 16 groups. Maximum budget: 20 epochs; stop after three epochs without strict improvement in full-reference validation MAP@12. Equal MAP retains the earlier epoch. Epoch 3 is selected; six epochs were executed. Validation MAP is 0.026914980, below LightGBM's 0.032710133.

The final neural refit starts from new weights, refits preprocessing and vocabularies on the same seven-week dataset as final LightGBM (2,974 groups / 892,200 rows / 4,461 positives), and runs exactly three epochs without early stopping or final-week labels. Its vocabulary has 2,406 known customers and 14,454 known articles plus reserved indices. No V1 or confirmation target enters fitting. The final state has 560,961 parameters; reloaded scores agree exactly on 1,000 probe rows.

Unknown customers use the frozen recent-purchases fallback; unknown articles retain their zero embedding while using historical numeric features. This hybrid policy is part of the evaluated system, not a new policy selected after confirmation. See [neural/confirmation artifacts](pytorch_results.md).
