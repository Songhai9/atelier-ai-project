# Atelier AI — Historical Training Targets V1

This document fixes the initial historical supervision windows. It supplements [the evaluation strategy](evaluation.md); it does not change the validation/test protocol.

## Fixed cohort and temporal scope

Use the same 20,000-customer cohort exported by notebook 03. Its sorted-ID SHA-256 is `f83b0240971fbc17545f900c497a904cb4e4c51b5bcaba64f0df6ceb4dc6b96f`.

Choose the last four complete seven-day target windows before validation, before inspecting their outcomes. Four weeks are an initial computational scope, not a demonstrated optimal duration. This is one historical training set followed by validation and test, not four cross-validation folds.

Every interval is inclusive at its start and exclusive at its end.

| Prediction cutoff | 180-day history starts | 7-day popularity starts | Target ends |
|---|---|---|---|
| 2020-08-12 | 2020-02-14 | 2020-08-05 | 2020-08-19 |
| 2020-08-19 | 2020-02-21 | 2020-08-12 | 2020-08-26 |
| 2020-08-26 | 2020-02-28 | 2020-08-19 | 2020-09-02 |
| 2020-09-02 | 2020-03-06 | 2020-08-26 | 2020-09-09 |

At cutoff `t`, candidates and features use records strictly before `t`. Labels use observed purchases in `[t, t + 7 days)`. Training targets end no later than the start of validation on 2020-09-09.

The four target windows do not overlap. Historical feature windows overlap intentionally. An earlier target purchase can be used as later history after that purchase has occurred. No features may use purchases from their own target week or subsequent weeks.

The same customer may appear in several weeks. The split measures future-period recommendations for the fixed population. It does not measure generalisation to entirely unseen customer IDs. Repeated customer–week observations are not assumed to be statistically independent.

## Three reference tables

- `target_transactions.parquet`: `(cutoff, t_dat, customer_id, article_id)`, retaining every selected source transaction row, including repetitions.
- `relevance.parquet`: unique `(cutoff, customer_id, article_id)` with observed-purchase `label = 1`. The same customer–article pair may be relevant in several distinct weeks.
- `population.parquet`: every `(cutoff, customer_id)`, with the distinct target-item count and an observed-purchase flag. It contains 80,000 rows, including customer–weeks without purchases.

All target purchases remain in reference relevance, including articles that will be absent from historical candidates. Reference relevance must not be restricted to retrieved candidates.

The candidate-feature training table is not yet built in milestone 4.4c. Relevance-row counts are not candidate-positive counts, candidate coverage, class balance in the eventual training matrix or model performance.

## Attaching labels to future candidate rows

Generate candidates from historical inputs first. Then left-join reference relevance on all three keys: `cutoff`, `customer_id`, `article_id`. Preserve the existing candidate rows and order. Fill an unmatched label with zero.

A zero means no matching observed purchase in the target week. It does not establish dislike, product exposure or availability. Do not inject missed relevant articles into candidates after reading labels. Keep missed positives in the reference table for coverage and evaluation denominators.

Customer–weeks without purchases remain represented. Final ranker training-group eligibility, treatment of groups without any retrieved positive, weighting and possible sampling are later explicit modelling decisions. Future MAP/Recall eligibility remains defined by the evaluation document.

## Recompute historical features

Notebook 03's 2020-09-09 feature export must not be relabelled as historical training data: it uses transactions that were future information at earlier cutoffs.

At each historical cutoff, recompute 180-day history, 7-day popularity, dominant group, candidate retrieval and the seventeen features. Preserve the existing candidate-policy limits initially; any subsequent changes must be recorded and compared under the same evaluation protocol.

No learned preprocessing is fitted in this target-construction milestone. Fit learned transformations on eligible training examples only. Validation selects model choices. For V1, selected learned parameters are not refitted on validation labels before local test.

## Export and provenance

Notebook `04_training_dataset.ipynb` exports the three reference tables, `schedule.csv`, a source notebook snapshot and `manifest.json` under Git-ignored `data/processed/training_targets/`.

The export identity records schedule, cohort, source manifest, code hashes and Python/pandas/DuckDB versions. Generated files receive fresh checksums; typed Parquet is compared against its in-memory input before publication. Identical complete exports are verified and reused.

Raw source hashes are inherited from the existing source manifest; current raw file sizes are checked. This stage does not rehash all raw CSV bytes. It scans the source CSV but retains no validation/test target rows or performance results.


## Materialised historical candidate features (milestone 4.4d)

Notebook 04 sections 8–13 materialise the unchanged retrieval and seventeen-feature definitions at each of the four historical cutoffs. Eight core functions are extracted unchanged from notebook 03 into `recommendation_features.py`; `historical_contexts.py` prepares the four historical contexts in one additional source scan. Global popularity includes all customers, while personal histories use the established cohort.

For each cutoff, retrieve up to 100 recent personal articles, up to 100 articles from the dominant historical product group, then fill globally to 300 distinct candidates per customer. Compute historical features before joining observed-purchase labels. No missed positives are inserted. The broad product-group historical-stability assumption remains explicit and unverified.

The complete exports contain 24,000,000 rows over 80,000 customer–weeks, in 160 partitions of 500 customers / 150,000 rows. Every row has 23 columns: three keys, two audit descriptors, seventeen model features and one binary label. The target and audit columns are excluded from `FEATURE_COLUMNS`.

Each stored customer's positive count is reconciled against the intersection of generated candidates and unchanged reference relevance. Micro/macro candidate coverage retains all reference purchases in its denominators. Candidate-positive rates are measured over all generated rows and are distinct from reference-positive counts.

No-purchase customer–weeks and buyer weeks without any retrieved positive are retained. Final ranker-training eligibility, negative sampling and weighting remain explicit later choices. No model or learned preprocessing is fitted here.

The exporter accepts an explicit `dataset_split="training"`, writes complete per-cutoff datasets and reuses matching verified exports. A collection manifest is written only after all four cutoff datasets and their label counts pass verification. Raw-file checks remain inherited source hashes plus current size checks; generated artifact checksums and round-trip comparisons are fresh.


## Initial ranker training policy (milestone 4.4e)

One training group is `(cutoff, customer_id)`. Keep a complete group only when its generated candidates contain both observed-purchase label values 0 and 1. Constant-label groups provide no positive-versus-negative preference within that group for the chosen binary grouped-ranking setup. This is a training objective choice, not a rule for all classification models.

For the current historical collection, 1,775 of 80,000 groups are eligible. Keeping all 300 candidates per eligible group gives 532,500 rows: 2,753 positive labels and 529,747 zero labels. No within-group negative sampling, positive duplication, synthetic oversampling or extra sample weights are introduced. The future loss configuration controls any model-specific query normalisation/weighting.

Selection depends on observed training labels and changes the training distribution. It does not yield a calibrated population purchase probability or demonstrate performance across unselected groups. The raw candidate exports, all 80,000-group audit and full reference relevance remain retained.

This filter is restricted to historical training. Validation/test recommendation recipients and reference targets remain unchanged, including buyers whose purchases are absent from candidates. Metric eligibility still follows the existing evaluation protocol; do not filter validation to only groups with retrieved positives.

`ranking_data.py` sorts selected rows by cutoff, customer ID, then article ID. X contains exactly the seventeen ordered features; y contains aligned binary relevance. Identifiers, group descriptors, dates and target labels are excluded from X. Group sizes refer to consecutive row blocks and sum to the number of X/y rows, following the [LGBMRanker grouped-input convention](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMRanker.html).

Expected missing features remain missing. No scaler, encoder, imputer or model is fitted during preparation. The selected Parquet, group index, complete audit, source snapshot and provenance manifest are exported under Git-ignored `data/processed/ranker_training/`. A future model can reconstruct aligned X, y and group sizes from these artifacts.
