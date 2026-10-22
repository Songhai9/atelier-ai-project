# Atelier AI — Data Dictionary

## Scope and Evidence

This dictionary covers the 25 article columns, 7 customer columns and
5 transaction columns present in the acquired CSV headers.

Descriptions are working interpretations based on column names,
inspected values and source documentation. They do not constitute a
complete official business glossary.

Entity-key uniqueness, code-to-label consistency and reference coverage
remain subject to full-table validation.

Documenting a field does not establish its suitability as a model input
or its availability at historical prediction dates.

## Logical Roles and Current Reading Types

Logical roles describe how values should be interpreted. Reading types
describe their current representation in pandas.

| Fields | Current reading type |
|---|---|
| Article and customer metadata | string |
| Transaction customer, article and channel identifiers | string |
| Transaction date | pandas datetime |
| Transaction price | nullable Float64 |

Numeric-looking identifiers and category codes are not treated as
continuous measurements.

## Articles

Expected grain: one record per article.

`article_id` is the candidate entity key and recommendation identifier.
Its full-table uniqueness remains to be checked.

| Field(s) | Working meaning | Logical role and handling |
|---|---|---|
| `article_id` | Identifier of the article referenced by transactions. | Entity identifier; preserve its source representation. |
| `product_code` | Product code shared by multiple article identifiers in the inspected preview. | Grouping identifier; not interchangeable with article_id. |
| `prod_name` | Source product name. | Descriptive text; not assumed to uniquely identify a product. |
| `product_type_no`, `product_type_name` | Code and label for the source product-type classification. | Categorical pair; inspect code-to-label consistency. |
| `product_group_name` | Source product-group label. | Categorical descriptor. |
| `graphical_appearance_no`, `graphical_appearance_name` | Code and label describing graphical appearance. | Categorical pair; retain source terminology. |
| `colour_group_code`, `colour_group_name` | Code and label for a source colour group. | Categorical pair; preserve code formatting. |
| `perceived_colour_value_id`, `perceived_colour_value_name` | Code and label for the source perceived-colour-value classification. | Categorical pair; exact taxonomy semantics are not independently established. |
| `perceived_colour_master_id`, `perceived_colour_master_name` | Code and label for the source perceived-colour-master classification. | Categorical pair; retain the source classification. |
| `department_no`, `department_name` | Source department code and label. | Categorical pair; department numbers are not continuous quantities. |
| `index_code`, `index_name` | Source index code and label. | Categorical pair; business meaning follows the source labels. |
| `index_group_no`, `index_group_name` | Source index-group code and label. | Categorical pair; hierarchy relationships require inspection. |
| `section_no`, `section_name` | Source section code and label. | Categorical pair; inspect relationships with other classifications. |
| `garment_group_no`, `garment_group_name` | Source garment-group code and label. | Categorical pair. |
| `detail_desc` | Source article description. | Free text; possible future text-feature input, subject to evaluation. |

Classification labels are not assumed to be unique. Relationships
between product type, department, index, section and garment group are
not assumed to form a strict hierarchy.

## Customers

Expected grain: one record per customer.

`customer_id` is the candidate entity key. Its full-table uniqueness
remains to be checked.

| Field | Working meaning | Logical role and handling |
|---|---|---|
| `customer_id` | Customer identifier referenced by transactions. | Entity identifier; preserve as an opaque string. |
| `FN` | Source indicator with unresolved business semantics. The inspected preview contains blank fields and the value 1.0. | Preserve as supplied; do not interpret blanks as zero or include in initial model inputs without clarification. |
| `Active` | Source indicator with unresolved business semantics. The inspected preview contains blank fields and the value 1.0. | Preserve as supplied; do not equate it with recent purchasing activity. |
| `club_member_status` | Source membership-status label. | Categorical attribute; status definitions and historical availability require clarification. |
| `fashion_news_frequency` | Source label apparently describing fashion-news frequency. | Working categorical interpretation; definitions, missingness and timing require clarification. |
| `age` | Source customer-age attribute. | Numerical interpretation; reference date and validity require review before historical use. |
| `postal_code` | Postal-code-related source token. | Opaque categorical identifier; geographic decoding and granularity are not established. |

Customer metadata is supplied without a per-record observation date in
the inspected schema. Its availability at a historical prediction
cutoff must not be inferred from its presence in the file.

## Transactions

Expected grain: one recorded article purchase by a customer on a date.

The schema does not provide an event, order or basket identifier.
Repeated records are retained pending interpretation.

| Field | Working meaning | Logical role and handling |
|---|---|---|
| `t_dat` | Recorded purchase date. | Temporal field; parsed using YYYY-MM-DD. |
| `customer_id` | Reference to the customer associated with the purchase. | Entity reference; expected to match customers.customer_id. |
| `article_id` | Reference to the purchased article. | Entity reference and relevance identifier; expected to match articles.article_id. |
| `price` | Recorded numeric price field. | Numerical measure; currency and scaling are not established in this project. No rescaling is applied. |
| `sales_channel_id` | Source sales-channel code. | Categorical code; preserve raw values without assigning an unverified channel-label mapping. |

Blank transaction dates and prices are converted to missing values
during the typed scan. Invalid non-blank values raise conversion errors.

Successful parsing does not establish completeness, plausible ranges
or correct relationships between tables.

## Feature Eligibility

Potential uses include product-category preferences, historical
purchase recency and frequency, and historical price summaries.

These are hypotheses for later evaluation, not implemented features.

Transaction-derived features must use records preceding the prediction
cutoff. Metadata features require supporting temporal evidence or an
explicit, justified stability assumption.

Code and label columns may carry overlapping information. Their
representation will be selected after consistency checks and modelling
experiments.

## Open Questions

- Are candidate entity keys complete and unique?
- Do transaction references match the entity tables?
- Is each classification code consistently associated with its label?
- What are the precise definitions and missing-value semantics of
  FN and Active?
- What observation dates apply to customer metadata?
- What currency or transformation applies to price?
- What authoritative mapping defines sales-channel labels?

Unresolved questions remain documented rather than being replaced
with unsupported assumptions.

## References

- [Official dataset documentation](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data)
- [Data card](data_card.md)
- [Acquisition manifest](data_manifest.json)
- [Notebook: inspected structure and diagnostics](../notebooks/atelier_ai.ipynb)