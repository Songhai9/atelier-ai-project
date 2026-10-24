# Atelier AI — Evaluation Strategy

## Objective

Evaluate whether personalised recommendations retrieve and rank future observed purchases more effectively than simple reference methods.

The evaluation focuses on historical predictive performance. Business impact requires a separate assessment under actual usage conditions.

## Primary Metric

**MAP@12 — Mean Average Precision at 12** is the primary model-selection criterion.

Average Precision is computed for each customer–target-week pair, considering the first twelve recommendations. Precision is accumulated at positions containing relevant products and normalised by the smaller of twelve and the number of distinct purchased products.

MAP@12 averages these scores with equal weight across eligible customer–week pairs.

This choice follows the evaluation metric used in the [H&M Personalized Fashion Recommendations benchmark](https://www.kaggle.com/c/h-and-m-personalized-fashion-recommendations/).

## Evaluation Conventions

- Relevance is defined by a purchase observed during the seven-day target period.
- Repeated purchases of the same product count as one relevant product for ranking evaluation.
- Recommendation lists contain distinct product identifiers.
- All target purchases remain in the evaluation, including products absent from the candidate set.
- Customer–week pairs without target purchases are excluded from MAP@12 and Recall@12.
- The proportion of excluded pairs is reported.
- Recommendations remain available for customers without observed target purchases.
- Ranking scores are not interpreted as calibrated purchase probabilities.

## Baseline Methods

Notebook 05 compares the first learned model with:

1. **Recent popularity:** products ranked by recent global purchase activity.
2. **Category preference:** popular products within the customer’s preferred categories.
3. **Recent purchases:** previously purchased products, completed with recent popular products.

All four methods are evaluated using the same customer cohort, target period and metric definitions.

## Supporting Diagnostics

| Diagnostic | Purpose |
|---|---|
| Recall@12 | Measure the proportion of distinct target purchases retrieved in the recommendation list |
| Candidate coverage | Measure the proportion of target purchases present in the generated candidate set |
| Catalogue coverage | Measure the proportion of eligible catalogue products recommended across the evaluated cohort |

Ranking quality will also be examined across customer-history segments to identify where personalisation helps or fails.

## Temporal Protocol

The acquired transaction file covers 2018-09-20 through 2020-09-22.
Transactions occur on all 734 calendar dates in that observed range.

The initial local evaluation uses seven-day prediction targets.

All intervals have an inclusive start and an exclusive end.

| Period | Start inclusive | End exclusive | Source transaction rows |
|---|---|---|---:|
| Pre-validation history | 2018-09-20 | 2020-09-09 | 31,292,772 |
| Validation target | 2020-09-09 | 2020-09-16 | 255,241 |
| Local test target | 2020-09-16 | 2020-09-23 | 240,311 |

Volumes count source transaction rows, including retained repetitions.
They are not counts of customers, distinct relevance labels or
supervised training examples.

### Historical Training Examples

The pre-validation period is a source-history pool.

Supervised examples will be constructed at historical prediction
cutoffs. Each example uses features from records strictly before its
cutoff and relevance from the following seven days.

Training target windows must end no later than 2020-09-09.
Learned preprocessing is fitted using eligible training examples only.

The initial historical cutoffs are 2020-08-12, 2020-08-19, 2020-08-26 and 2020-09-02. Each has a seven-day target and cutoff-specific historical candidates/features. For initial grouped ranker fitting, retain complete training groups only when candidate labels contain both 0 and 1. This target-dependent selection is restricted to historical training; it does not change validation/test recipients, reference targets or metric denominators. See the [training protocol](training_protocol.md) for the group audit and input contract.


### Validation and Test Execution

Methods and hyperparameters are selected using the validation target.

For V1, selected learned parameters remain fixed between validation
and test. They are not refitted on validation targets.

Historical features and candidates are recomputed at each prediction
cutoff using only earlier observations. Validation-week transactions
are therefore available as history at the 2020-09-16 test cutoff.

Predictions issued at the beginning of a target week are not updated
using purchases inside that week.

Test results are inspected after model-selection choices are frozen
and are not used to revise those choices.

### Availability and Evaluation Scope

Metadata fields require evidence of historical availability or an
explicit, justified stability assumption before feature use.

Target purchases absent from the candidate set remain in evaluation.
Cold-start cases are reported rather than silently removed.

This test is a local historical holdout, separate from the competition's
hidden test data.

One validation week and one test week provide an initial assessment.
Broader temporal robustness requires additional historical cutoffs.

Coverage calculations and boundary definitions are documented in the
[analysis notebook](../notebooks/01_data_foundation.ipynb).

## Model Selection

The selected system is the method with the highest validation MAP@12. In the event of an exact tie, the simpler method is retained.

A baseline remains eligible for serving if the learned model does not improve the primary validation metric.

Test results are reported without using them to revise the selected configuration.

## Interpretation of Success

Predictive success is assessed relative to the baseline methods, with supporting diagnostics and uncertainty reported alongside the primary metric.

Functional evaluation will separately assess recommendation validity, response latency and the advisor workflow.

Improved offline ranking metrics do not establish increased sales, conversion or reduced preparation time. Those outcomes require dedicated usage studies or experiments.

## Implemented full-reference evaluation (stage 5)

`ranking_metrics.py` validates complete recipient coverage and distinct top-k lists, including recipients without target purchases. It computes per-buyer AP@12 and Recall@12 against the full distinct target set. Buyers whose purchases are absent from all candidates remain eligible and receive zero hits. No target reference is restricted to retrieved articles.

- **MAP@12:** equal-weight mean AP across the 1,015 validation buyers; AP's denominator is `min(12, full distinct target count)` for that buyer.
- **Macro Recall@12:** equal-weight mean of each buyer's hits divided by their full distinct target count.
- **Micro Recall@12:** total retrieved customer-article targets divided by all 3,073 distinct customer-article targets. This is a diagnostic separate from the macro mean.
- **Excluded share:** 18,985 / 20,000 recipients = 94.925% have no observed target purchase and are excluded from target-based averages. All 20,000 still receive recommendations.

The validation target is `[2020-09-09, 2020-09-16)`. Its cohort reference retains 3,454 source transaction rows and separately deduplicates them to 3,073 relevant customer-article pairs. Repeated source transactions are not interpreted as separate relevant articles or as orders.

Paired diagnostics align model and baseline scores by the same 1,015 buyers. History segments use observed purchases in `[2020-03-13, 2020-09-09)`, not lifetime customer tenure. Segment MAP differences are weighted by segment buyer count when reconciling the overall difference.

The first comparison and export identities are recorded in [first-ranker results](first_ranker_results.md). Recent purchases leads this comparison; this does not establish that it will remain best after further declared validation experiments. Catalogue coverage and broader temporal robustness are not established by this first learned experiment. Local-test outcomes remain unevaluated.
