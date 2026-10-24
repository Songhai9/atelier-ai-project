# Atelier AI - Historical Feature Dictionary

One row represents `(customer_id, article_id, cutoff)`. All personal history uses the preceding 180 days; shorter windows are nested. Global article measurements include every source customer. Source purchase rows retain repetitions; category and colour shares use personal historical row counts as denominator. Classification labels come from a snapshot used under an explicit stability assumption.

| Feature | Family | Missing policy | Definition |
|---|---|---|---|
| customer_history_rows | customer | 0 | Purchase rows in the preceding 180 days |
| customer_distinct_articles | customer | 0 | Distinct articles in the preceding 180 days |
| customer_recency_days | customer | NaN | Days since the last observed purchase |
| customer_has_history | customer | False | Any purchase in the preceding 180 days |
| customer_rows_7d | customer | 0 | Purchase rows in the preceding 7 days |
| customer_rows_28d | customer | 0 | Purchase rows in the preceding 28 days |
| customer_rows_90d | customer | 0 | Purchase rows in the preceding 90 days |
| customer_distinct_product_types | customer | 0 | Distinct product types in 180-day history |
| customer_distinct_colours | customer | 0 | Distinct colour groups in 180-day history |
| article_recent_rows | article | 0 | Global purchase rows in the preceding 7 days |
| article_recent_rank | article | NaN | Global 7-day rank; ties use article ID |
| article_rows_28d | article | 0 | Global purchase rows in the preceding 28 days |
| article_popularity_trend | article | 0 | 7-day rows divided by one quarter of 28-day rows |
| article_recency_days | article | NaN | Days since last global purchase within 180 days |
| customer_group_rows | affinity | 0 | Customer rows for the candidate broad product group |
| customer_group_share | affinity | 0 | Broad-group rows divided by customer history rows |
| customer_type_share | affinity | 0 | Product-type rows divided by customer history rows |
| customer_colour_share | affinity | 0 | Colour-group rows divided by customer history rows |
| customer_article_rows | interaction | 0 | Customer purchase rows for the candidate article |
| customer_article_recency_days | interaction | NaN | Days since the customer last bought this article |
| has_purchased_article | interaction | False | Candidate article observed in customer history |
| prediction_weekday | context | defined | Weekday of prediction cutoff, Monday=0 |
| prediction_iso_week | context | defined | ISO week of prediction cutoff |
| customer_channel_2_share | context | 0 | History-row share with source channel code 2 |
| from_history | retrieval | False | In the latest 30 distinct personal articles |
| from_category | retrieval | False | In a top-50 proposal from a preferred product type |
| from_global | retrieval | False | In the global 7-day top-100 proposal |
| from_fallback_28d | retrieval | False | Added by the 28-day popularity fill |
| history_source_rank | retrieval | NaN | Rank in the bounded personal-history proposal |
| category_source_rank | retrieval | NaN | Rank in the combined preferred-type proposal |
| n_retrieval_sources | retrieval | 0 | Number of proposal or fill memberships |

Preferred retrieval types are selected by descending personal 180-day row count, then type name. Their article proposals use descending global 7-day row counts, then article ID. Personal candidates use descending last purchase date, then article ID. Candidate source priority is personal history, global 7-day top 100, preferred-type proposals, then 28-day fill. Source memberships may overlap; the fallback flag identifies actual fill additions. Counts and shares without observations use zero; an unobserved recency or source rank remains missing.

Audit fields `customer_id`, `article_id`, `cutoff`, `product_group_name` and `dominant_product_group` are excluded from X. Target labels are attached after retrieval and feature calculation. No raw identifier is interpreted as an ordered number. `prediction_iso_week` is an explicit calendar context feature, not a learned product/customer ID encoding.
