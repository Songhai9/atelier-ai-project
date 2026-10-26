# Atelier AI - Evaluation Strategy

## Task and evidence scope

Recommend twelve distinct article IDs per customer at a historical prediction cutoff. Relevance is the set of distinct articles purchased in the next seven days. Evaluate predictive performance relative to simple methods; offline metrics do not establish sales, conversion or advisor time savings.

## Calendar

The shared configuration is `configs/temporal.yaml`. All intervals include their start and exclude their end.

| Role | Start inclusive | End exclusive | Source transaction rows |
|---|---|---|---:|
| EDA source-history pool | 2018-09-20 | 2020-09-02 | 31,028,115 |
| Validation | 2020-09-02 | 2020-09-09 | 264,657 |
| Intermediate V1 evaluation | 2020-09-09 | 2020-09-16 | 255,241 |
| Final LightGBM/PyTorch confirmation | 2020-09-16 | 2020-09-23 | 240,311 |

The row counts describe the complete source file and retain repetitions. They are not customer counts, orders or supervised examples. Six initial model-selection training cutoffs run weekly from 2020-07-22 to 2020-08-26. Their target windows end no later than validation starts. At each cutoff, recompute history, candidates and features using strictly earlier observations and a 180-day personal lookback.

## Populations and metrics

The fixed cohort contains 20,000 source customer IDs selected using seeded SHA-256 ordering independently of future purchases. All receive recommendations. Customer/week pairs without target purchases are excluded from MAP, macro Recall and NDCG averages; report their count and share.

For validation [2020-09-02, 2020-09-09), the cohort contains 3,803 source purchase rows, 3,428 distinct customer/article targets and 1,065 target buyers. These counts differ from complete-file weekly counts because they describe the fixed cohort.

- **AP@12:** accumulate precision at relevant ranks, divided by min(12, the full number of distinct targets).
- **MAP@12:** equal-weight mean AP over all target buyers; primary selection metric.
- **Macro Recall@12:** equal-weight mean of hits / all distinct targets per buyer.
- **Micro Recall@12:** all retrieved customer/article targets / all distinct target pairs.
- **NDCG@12:** binary discounted gain divided by ideal gain for min(12, the full target count).
- **Candidate coverage:** retrieved distinct target pairs / all distinct target pairs, including misses.
- **Catalogue coverage:** distinct recommended articles / articles observed in the declared 180-day global history.
- **Category diversity:** mean number of distinct broad product groups in each recipient's twelve recommendations.

Repeated source purchases do not create duplicate relevant articles. Never restrict references to retrieved candidates or add missed targets to a candidate pool. An evaluated buyer with all purchases missed receives zero ranking hits. Validate complete recipient coverage and twelve distinct recommendations before computing target-based metrics.

## Baselines and learned experiment

Compare recent global popularity, popularity in three preferred product types, recent personal purchases with popularity fill, and the initial LambdaRank model. Use identical cohort, target period and metric definitions. A baseline remains eligible for eventual serving.

The initial learned model uses the planned starting configuration and validation NDCG stopping with patience 50. Its stopping metric uses full target counts for ideal gain and excludes nonbuyers; it does not treat buyers with completely missed retrieval as perfect empty groups. The six-configuration comparison is complete: validation MAP@12 selects 63 leaves / minimum 20 with iteration 8 as the stage-8 candidate. These selected-model diagnostics and final method selection are completed in notebook 07; their scope is documented in the stage-8 report. See [tuning results](lightgbm_tuning_results.md).

## Selection, fitting and frozen outcomes

Select by validation MAP@12. Retain the simpler method on an exact tie. Freeze model choices and iteration budget, then refit a selected learned configuration on historical training plus validation. Neither the V1 nor confirmation targets enter this fitting or early stopping.

After freezing V1 choices, report intermediate V1 outcomes without revising that reported configuration. Freeze the LightGBM/PyTorch comparison and both fitting procedures before evaluating final confirmation. Notebook 07 now records intermediate V1 performance for the frozen final refit. Notebook 09 now records the frozen final confirmation; its results and uncertainty are documented below.

Predictions issued at a week's start do not incorporate purchases from inside that week. Earlier validation purchases may become historical inputs at later cutoffs without being future inputs.

## Interpretation limits

This is a retrospective local benchmark. The weeks beginning 2020-09-02 and 2020-09-09 were accessed during development. The intermediate V1 period is therefore not an untouched independent test. Recomputing under fixed chronological boundaries does not erase that exposure. Final-week ranking performance has now been exposed by the frozen LightGBM/PyTorch comparison; this week is no longer available as untouched data for further tuning.

Snapshot article classifications are used under an explicit stability assumption; their observation dates and stock availability are not established. Customer snapshot attributes with unresolved historical availability are excluded from model inputs. Price units and channel business labels remain unresolved. Report cold-history cases rather than silently deleting them.

The source is separate from the competition's hidden test data. One local week provides limited temporal evidence. SHAP is explanatory, not causal. Stage 8 records selected-model feature ablations, descriptive activity/history segments, timings and customer-bootstrap uncertainty. Validation diagnostics describe the pre-refit model; V1 metrics describe the separately fitted final model.

## Evidence

- [Initial results](first_ranker_results.md)
- [Training contract](training_protocol.md)
- [Feature definitions](feature_dictionary.md)
- [Reference plan](implementation_plan.md)


## Executed intermediate V1 evaluation

Period [2020-09-09, 2020-09-16), fixed cohort 20,000: 3,454 source purchase rows, 3,073 distinct target pairs and 1,015 target buyers. The other 18,985 recipients (94.925%) receive lists but are excluded from MAP, macro Recall and NDCG means.

The frozen final model gives MAP@12=0.022927, macro Recall@12=0.041459 and NDCG@12=0.031957. The validation-selected recent-purchases baseline gives 0.022168, 0.044635 and 0.032089, respectively.

Candidate coverage is 618/3,073=20.1106%. LightGBM retrieves 105 target pairs (micro Recall 3.4169%); the baseline retrieves 111 (3.6121%). The historical catalogue denominator is 50,383 observed articles. LightGBM recommends 5,944 distinct articles (11.7976%) versus 16,857 (33.4577%) for the baseline. Mean broad-group diversity across all recipients is 3.33075 versus 2.70180.

Bootstrap 5,000 target buyers samples with replacement, seed 42, keeping all methods' AP values paired. Each resample contains 1,015 buyers, including zero APs. The percentile 95% interval of LightGBM minus the fixed comparator is [-0.002626, 0.004346], around 0.000759. It includes zero: this is neither a demonstrated advantage nor an equivalence result.

Intervals condition on fitted systems and the observed week; they do not capture training, selection or between-week variation. The comparator was chosen on validation, not on V1. No configuration was changed after V1 outcomes. See [full report](stage8_evaluation_report.md).

## Executed final confirmation

Notebook 09 evaluates five frozen systems on [2020-09-16, 2020-09-23), after model weights, preprocessing, vocabularies, fallback policy and prediction lists were fixed. All target windows used for fitting end by 2020-09-09. Purchases between 2020-09-09 and 2020-09-16 may enter historical prediction inputs; they do not enter this refit.

For 20,000 recipients, 3,736 cohort source purchase rows become 3,267 distinct customer–article targets / 1,027 target buyers. Excluded nonbuyers: 18,973 (94.865%). All buyers, including complete retrieval misses and zero-AP cases, remain in MAP/Recall/NDCG means. Candidate coverage is 22.5895% of all 3,267 targets.

The primary contrast is **PyTorch hybrid minus LightGBM**, not PyTorch minus the best method observed on confirmation. Resampling uses the same 1,027 buyers for every method, 5,000 times with replacement, seed 42. The stored generic `model_minus_baseline` bootstrap row denotes this declared PyTorch/LightGBM contrast. The delta is 0.002865909; percentile 95% interval [-0.002320312, 0.008159653]. It is inconclusive for superiority and does not establish equivalence. Uncertainty conditions on fixed systems and this one week, excluding training, selection and temporal variation.

Recent purchases has the highest MAP point estimate (0.020584); PyTorch hybrid follows (0.020575), then LightGBM (0.017709). PyTorch has the highest descriptive macro Recall/NDCG; these secondary differences have no paired inference in this report. Preserve the validation nomination and frozen fitting procedures. Do not retune, rename a test-selected winner or claim business benefit from these outcomes. See [complete results](pytorch_results.md).
