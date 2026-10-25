# Atelier AI - Controlled LightGBM Tuning

## Question and shared protocol

Compare the six declared combinations of leaves {15,31,63} and minimum leaf examples {20,100}. Train on six historical cutoffs from 2020-07-22 through 2020-08-26, using 2,533 complete informative customer-week groups, 759,900 rows and the same 31 features.

Validation is [2020-09-02, 2020-09-09). Every method recommends twelve distinct article IDs to the same 20,000 recipients. Ranking means include all 1,065 target buyers and all 3,428 distinct customer-article targets, including retrieval misses. The other 18,935 recipients receive lists but are excluded from target-based means.

Only num_leaves and min_child_samples vary. Learning rate 0.05, maximum 1,000 iterations, depth 8, L2=1, seed 42 and four worker threads remain fixed. Each trial uses full-reference validation NDCG@12 stopping with patience 50. Compare stopped models by full-reference MAP@12.

## Results

| Maximum leaves | Minimum leaf examples | Best iteration | MAP@12 | Recall@12 | NDCG@12 |
|---|---|---|---|---|---|
| 15 | 20 | 51 | 0.027000 | 0.054792 | 0.038684 |
| 15 | 100 | 73 | 0.026248 | 0.055503 | 0.038284 |
| 31 | 20 | 85 | 0.028404 | 0.057459 | 0.040445 |
| 31 | 100 | 16 | 0.027023 | 0.056514 | 0.038934 |
| 63 | 20 | 8 | 0.032710 | 0.059368 | 0.044496 |
| 63 | 100 | 75 | 0.031051 | 0.063059 | 0.044591 |

The best baseline, recent personal purchases, reaches MAP@12=0.025445. The 31/100 search trial reproduces the initial experiment's ranking metrics.

## Candidate selection and interpretation

Select **63 leaves / minimum 20 / best iteration 8** as the reference candidate for stage 8, with MAP@12=0.032710. The MAP increase over the leading baseline is 0.007265 in absolute score units, or 28.55% relative. This is a validation-score comparison, not a measured increase in purchases or business performance.

The 63/100 model has slightly higher NDCG and Recall but lower MAP. The previously declared primary metric determines this candidate selection. More leaves and the lower minimum leaf-size setting improve MAP within this small grid; this does not establish a general monotonic rule or an optimum outside the grid.

The selected trial records 58 iterations: the best NDCG occurs at iteration 8, followed by 50 rounds without improvement. Its saved model contains eight trees. Measured fit time is 15.62 s and scoring the six million validation candidates takes 0.55 s. These measurements do not include the full candidate/feature pipeline and are not API-latency estimates.

For an exact MAP tie among learned configurations, sorting prefers fewer maximum leaves and then a higher minimum leaf-size constraint. This is a deterministic capacity preference, not a measurement of actual model size. A leading or exactly tied baseline requires explicit review and remains eligible. The current maximum MAP is unique.

## Inspectable evidence

Notebook 06 records the grid, checksum/input checks, all six trials, results, the MAP heatmap and the candidate decision. Each trial stores a native model, model checksum, dataset/code provenance, learning curve, per-customer scores and complete recommendation lists. MLflow records all six trials, models, metrics and the comparison figure.

Local artifacts under `data/processed/weekly_v1/ranker_search/` include `results.csv`, `comparison_with_baselines.csv`, `manifest.json` and `selected_candidate.json`, plus one directory per trial. The selected MLflow run is `b76ec3a4783c4b7296a07321d032df31`. The selected model SHA-256 is `796d32628387bea0a3d7bea09313ad51f57c786de309d0f2afcb053d4576ef11`.

## Status and limits

Plan stage 7 is complete, and notebook 07 subsequently completed stage 8. This page remains the historical six-configuration selection record. Stage 8 retained the 63/20 configuration and all 31 features, froze an eight-iteration budget, refitted on training plus validation and evaluated intermediate V1. Its small V1 MAP difference has an interval containing zero; see the [stage-8 report](stage8_evaluation_report.md) and [model card](model_card.md).

No V1 or final-confirmation ranking outcomes enter this search. The benchmark remains retrospective because validation and intermediate V1 weeks were accessed during development. Final confirmation remains reserved until the LightGBM/PyTorch comparison is frozen. See [evaluation strategy](evaluation.md).
