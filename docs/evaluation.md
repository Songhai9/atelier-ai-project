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

The learned model will be compared with:

1. **Recent popularity:** products ranked by recent global purchase activity.
2. **Category preference:** popular products within the customer’s preferred categories.
3. **Recent purchases:** previously purchased products, completed with recent popular products.

The methods will be evaluated using the same customer cohort, target periods and metric definitions.

## Supporting Diagnostics

| Diagnostic | Purpose |
|---|---|
| Recall@12 | Measure the proportion of distinct target purchases retrieved in the recommendation list |
| Candidate coverage | Measure the proportion of target purchases present in the generated candidate set |
| Catalogue coverage | Measure the proportion of eligible catalogue products recommended across the evaluated cohort |

Ranking quality will also be examined across customer-history segments to identify where personalisation helps or fails.

## Temporal Protocol

Training data is used to fit models. Validation data is used to select methods and hyperparameters. Test data is evaluated after these choices are frozen.

Features and candidates must use only information available before each prediction date.

Actual date boundaries will be established after inspecting the dataset’s temporal coverage.

## Model Selection

The selected system is the method with the highest validation MAP@12. In the event of an exact tie, the simpler method is retained.

A baseline remains eligible for serving if the learned model does not improve the primary validation metric.

Test results are reported without using them to revise the selected configuration.

## Interpretation of Success

Predictive success is assessed relative to the baseline methods, with supporting diagnostics and uncertainty reported alongside the primary metric.

Functional evaluation will separately assess recommendation validity, response latency and the advisor workflow.

Improved offline ranking metrics do not establish increased sales, conversion or reduced preparation time. Those outcomes require dedicated usage studies or experiments.