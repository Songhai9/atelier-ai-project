# Atelier AI - Validation selection and inconclusive intermediate V1 advantage

## Result and decision

LightGBM leads the declared validation MAP comparison. After freezing the 63/20 configuration, all 31 features and an eight-tree refit budget, the final model scores MAP@12=0.022927 on intermediate V1 versus 0.022168 for the validation-selected recent-purchases baseline. The difference is 0.000759, with paired percentile 95% interval [-0.002626, 0.004346].

The interval includes zero. V1 does not establish clear superiority or equivalence; baseline Recall and NDCG are slightly higher. Preserve the frozen configuration for the reported V1, without test-driven retuning. These are retrospective offline results, not evidence of commercial impact.

## Populations, periods and identities

Every method recommends twelve distinct article IDs to a fixed, future-independent cohort of 20,000 recipients. Relevance is distinct customer/article purchases in the next seven days. Purchase rows are not orders. All reference purchases remain in denominators, including retrieval misses.

| Evaluation | Interval, end exclusive | Source rows in cohort | Target pairs | Target buyers | Excluded nonbuyers |
|---|---|---:|---:|---:|---:|
| Validation | 2020-09-02 to 2020-09-09 | 3,803 | 3,428 | 1,065 | 18,935 (94.675%) |
| Intermediate V1 | 2020-09-09 to 2020-09-16 | 3,454 | 3,073 | 1,015 | 18,985 (94.925%) |

Ranking means cover target buyers only, including zero scores. Recommendation coverage, catalogue coverage and diversity cover all recipients. Personal histories are 180 days strictly before each cutoff; global aggregates include eligible source customers beyond the cohort.

Diagnostics below use the selected six-week model, SHA-256 `796d32628387bea0a3d7bea09313ad51f57c786de309d0f2afcb053d4576ef11`, MLflow run `b76ec3a4783c4b7296a07321d032df31`. V1 uses the independently refitted seven-week model, SHA-256 `d6ca939c488a707c9a8c5be7480f738df640d2bc0a3496fdcb20b573f8353bf3`, run `8f90b653b86543f98358903363295f0e`. Both have eight trees, but their learned weights differ.

## Selected-model explanations

Gain importance uses all 31 features as its denominator. The leading shares are article_popularity_trend 15.296848%, customer_article_recency_days 10.786110%, customer_recency_days 10.557425% and history_source_rank 9.919405%. This is model split-gain allocation, not a causal or sales attribution.

Native SHAP examples use the same illustrative customer with zero observed purchase rows in the 180-day history. Article 0918292001 is an observed target at rank 9 with score 0.199027; 0751471043 is not an observed target and appears at rank 1 with score 0.226804. Both share a base value of about -0.026657. All feature contributions reconstruct the raw scores. Missing-history contributions depend on the full feature context. No observed purchase is not evidence of dislike; no observed history is not proof of lifetime newness.

## Fixed-candidate ablations

| Variant | Features | Retained iteration | MAP@12 | Macro Recall@12 | NDCG@12 |
|---|---:|---:|---:|---:|---:|
| full_control | 31 | 8 | 0.032710 | 0.059368 | 0.044496 |
| without_recency_days | 28 | 6 | 0.027859 | 0.055650 | 0.039695 |
| without_affinities | 27 | 69 | 0.028637 | 0.061341 | 0.041939 |
| without_popularity | 27 | 65 | 0.025210 | 0.054116 | 0.037233 |

Recency removes three recency-days columns. Affinity removes four declared group/type/colour rows or shares. Popularity removes four declared article-count/rank/trend columns. Retrieval and related proxies remain fixed. Each variant uses the same validation stopping rule and reselects its iteration. These comparisons measure incremental utility of those columns under this pipeline; they are not causal effects of abstract concepts.

## Errors and observed-history segments

Of 3,428 validation target pairs, 2,754 are outside candidates, 535 are candidates outside the top twelve and 139 appear in the top twelve. Among 1,065 buyers, 624 have no target in candidates, 327 have candidate targets but no hit, and 114 have at least one hit. These are two different grains and denominators.

Illustrative buyer cases cover four targets / zero candidate targets / zero hits; four / two / zero; and six / two / one. They are examples, not a representative sample.

| Observed purchase rows over 180 days | Recipients | Buyers | Model MAP | Baseline MAP |
|---|---:|---:|---:|---:|
| No observed history | 9350 | 185 | 0.013674 | 0.004979 |
| 1-10 purchase rows | 7265 | 374 | 0.030460 | 0.024315 |
| 11-50 purchase rows | 3180 | 435 | 0.038960 | 0.033296 |
| 51+ purchase rows | 205 | 71 | 0.055869 | 0.036620 |

Distinct-article richness segments contain 185, 221, 431 and 228 buyers for no observed history, 1-5, 6-20 and 21+ distinct articles, respectively. Model MAP exceeds the baseline point estimate in each activity and richness segment. These descriptive cuts have no segment-specific intervals; the highest-activity group has only 71 buyers.

## Uncertainty and local timings

Validation paired buyer bootstrap: difference 0.007265, percentile 95% interval [0.002316, 0.012509], 1,065 buyers / 5,000 resamples / seed 42. This exploratory validation interval does not account for model selection.

One local pass on all 20,000 validation recipients and six million candidate pairs records 0.58 s for historical context, 15.57 s retrieval, 0.37 s lookups, 39.53 s batch features, 0.90 s model scoring and 2.13 s top-12 selection. The timed sum is 59.07 s. Retrieval includes source memberships, baseline lists and audit construction returned by the current helper. Batch features include pair creation, construction checks and model-matrix conversion. Feature/scoring/selection use 40 batches of 500 customers, with four prediction threads. These stage sums exclude model loading, training, persistence, plotting, evaluation and inter-stage checks; cache state is uncontrolled. They are not API latency.

## Frozen refit

The immutable decision precedes the final fit. Six historical training cutoffs plus 2020-09-02 retain each group's own earlier history. Targets for the last fitting group end at 2020-09-09. The audit contains 140,000 groups; fitting keeps 2,974 complete informative groups / 892,200 rows / 4,461 positive candidate pairs. Fit eight iterations without evaluation data or early stopping. Reloaded scores agree on 1,000 deterministic probes (maximum absolute error 0.0). Fit-only time is 0.952 s, measured separately from preparation.

## Intermediate V1 outcomes

| Method | MAP@12 | Macro Recall@12 | NDCG@12 |
|---|---:|---:|---:|
| recent_popularity_7d | 0.005443 | 0.019836 | 0.011334 |
| preferred_product_types | 0.003475 | 0.012097 | 0.007408 |
| recent_purchases_180d | 0.022168 | 0.044635 | 0.032089 |
| lightgbm_leaves_63_min_20 | 0.022927 | 0.041459 | 0.031957 |

Candidate coverage is 618/3,073=20.1106%. LightGBM finds 105 target pairs versus 111 for the baseline. The slightly higher MAP can coexist with lower Recall because AP weights relevant ranks and MAP averages buyers; the metrics are not interchangeable.

The historical catalogue contains 50,383 observed articles. LightGBM recommends 5,944 distinct articles (11.7976%), versus 16,857 (33.4577%) for the comparator. Its mean broad-group diversity is 3.33075 versus 2.70180 across all 20,000 recipients. This V1 diversity relationship differs from validation and must not be described as a universal tradeoff.

Bootstrap resamples all 1,015 eligible buyers with replacement, including zero APs, jointly across the four methods. Each sample has 1,015 rows; 5,000 samples use seed 42. Individual percentile 95% MAP intervals are [0.015908, 0.030432] for LightGBM and [0.015640, 0.029461] for the comparator. The preselected paired difference, not overlap of separate intervals, addresses the main comparison.

## Limits and next milestone

Validation and V1 were exposed during development. V1 is not an untouched independent test. One week's buyer bootstrap conditions on fitted systems and cannot quantify training, selection or temporal variation. Validation and V1 also differ in fitted model and period, so their score difference does not identify overfitting, drift or any single cause.

Snapshot classifications do not establish historical stock or present availability. Raw scores are uncalibrated; missing purchase signals do not prove negative preference. Training-only group eligibility must not restrict evaluation. No sales uplift, conversion gain, luxury-domain generalization or production-serving latency has been established.

All scientific stage-8 outputs are recorded. The next plan section is the local reproducible pipeline and application. Preserve final confirmation [2020-09-16, 2020-09-23) until both LightGBM/PyTorch fitting/comparison procedures are frozen.

## Evidence and review scope

Notebook [07](../notebooks/07_model_diagnostics.ipynb) retains portable figures and saved outputs. Local artifacts under `data/processed/weekly_v1/` include `diagnostics/lightgbm_leaves_63_min_20/`, `ablations/`, `final_selection.json`, `final_training/`, `final_ranker/` and `v1_evaluation/`. MLflow stores models, evidence, figures, source snapshots and checksums.

Closure review checked saved outputs and artifact/MLflow identities; independently recomputed V1 AP, Recall, NDCG, micro Recall, catalogue coverage and diversity from saved lists/targets; and checked candidate coverage against the saved feature export. It did not rerun training, bootstrap draws, a fresh full notebook replay or the test suite.
