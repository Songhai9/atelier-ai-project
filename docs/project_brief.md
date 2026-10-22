# Atelier AI — Project Brief

## Overview

Atelier AI is a fashion recommendation project designed to help client advisors prepare personalised product selections.

The proposed system combines purchase histories and product attributes to generate a ranked list of recommendations.

## Business Problem

A broad product catalogue makes exhaustive manual selection difficult. Purchase histories provide signals that could help advisors identify relevant products, but systematically using these signals requires a structured approach.

The project addresses the following question:

> How can a customer’s previous purchases help a client advisor identify which products to recommend next?

## Target User and Workflow

The primary user is a client advisor.

The intended workflow is to:

1. Select an anonymised customer.
2. Review the purchase history available at a reference date.
3. Retrieve twelve distinct products, ranked by estimated relevance.
4. Review the information associated with the selection.
5. Adapt the recommendations before the customer interaction.

The recommendations support preparation. The advisor retains control over the final selection.

## Machine Learning Formulation

At a given reference date, the system ranks products using only information available before that date.

Ranking quality will be evaluated against the products purchased by the customer during the following seven days.

| Dimension | Definition |
|---|---|
| Recommendation unit | A customer at a reference date |
| Available inputs | Previous purchases and available product attributes |
| Expected output | A ranked list of twelve distinct products |
| Evaluation horizon | Seven days |
| Relevance signal | A purchase observed during the target period |

Purchases provide an implicit relevance signal. An unpurchased product should not automatically be interpreted as a product the customer dislikes.

## Main Hypothesis

Combining observed customer preferences with recent product popularity could produce more relevant selections than recommending recent bestsellers alone.

This hypothesis will be assessed by comparing simple baseline methods with a learned ranking model.

## Success Criteria

The project will be assessed along two dimensions:

- **Predictive quality:** the ability to retrieve and rank future observed purchases, compared with baseline methods.
- **Functional usefulness:** the ability to provide a coherent, accessible selection supported by information useful to an advisor.

Improvements in historical evaluation metrics would demonstrate modelling performance. Measuring an effect on sales or preparation time would require an experiment under actual usage conditions.

## Assumptions and Limitations

- Information recorded after the reference date is excluded from model inputs.
- Observed purchases provide an incomplete view of customer preferences.
- Predictive relevance does not establish current product availability.
- A ranking score is not a calibrated purchase probability.
- Business benefits remain to be measured.