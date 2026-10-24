# Atelier AI - Initial Ranking Results

## Question and scope

Does the planned initial LambdaRank model improve complete-reference validation MAP@12 over three simple references? Train on six historical weeks, evaluate [2020-09-02, 2020-09-09), and recommend twelve distinct articles to each of 20,000 fixed recipients.

## Results

| Method | MAP@12 | Recall@12 | NDCG@12 |
|---|---|---|---|
| recent_popularity_7d | 0.006091 | 0.022700 | 0.012546 |
| preferred_product_types | 0.005346 | 0.014352 | 0.009923 |
| recent_purchases_180d | 0.025445 | 0.055198 | 0.036811 |
| lightgbm_initial | 0.027023 | 0.056514 | 0.038934 |

The leading method in this initial comparison is **lightgbm_initial**. This is a validation observation, not a final production decision or a claim of commercial benefit. The declared six-configuration search remains next.

The reference contains 1,065 buyers and 3,428 distinct target pairs from 3,803 cohort source rows. The other 18,935 recipients still receive recommendations but are excluded from target-based metric means. All target pairs remain in denominators, including those absent from the 300 candidates.

## Configuration and execution

- Six training cutoffs: 2020-07-22 through 2020-08-26.
- 2,533 informative fitting groups; 759,900 candidate rows; 31 features.
- LambdaRank; learning rate 0.05; leaves 31; depth 8; minimum leaf examples 100; L2=1; seed 42.
- Maximum 1,000 iterations; full-reference validation NDCG@12 stopping with patience 50; selected iteration 16.
- Measured fitting time: 15.88 s. Measured full validation scoring time: 0.65 s. These are local measurements, not serving-latency benchmarks or process peak-memory measurements.
- Native save/reload predictions checked numerically; MLflow run `ad9a6d63d33245a68346b43d986d80a6`.

## Diagnostic evidence

Notebook 05 contains the learning curve, common-reference comparison, gain importance, paired customer AP differences against the leading baseline, and activity segments based on observed 180-day history. Candidate/catalogue coverage and mean recommended product-group breadth accompany ranking metrics.

Full local evidence is stored under `data/processed/weekly_v1/first_ranker/`: model, manifest, comparison, learning curve, recommendations, per-customer scores, paired scores and history segments. Generated figures live under `artifacts/figures/`. These outputs are local; portable figures and aggregates are saved inside the versioned notebook.

## Remaining evidence and limitations

Hyperparameter search, SHAP examples, controlled ablations, customer-bootstrap uncertainty, final selection/refit and intermediate V1 performance are not completed. No final-confirmation ranking outcomes have been evaluated. Development exposure makes the V1 period retrospective; see [evaluation strategy](evaluation.md).

The initial ranker has the highest point estimate in this comparison. Customer-bootstrap uncertainty and the declared controlled comparisons remain necessary before a final selection claim. Do not attribute a causal business mechanism to feature importance or infer performance on unseen time periods from this single validation week.
