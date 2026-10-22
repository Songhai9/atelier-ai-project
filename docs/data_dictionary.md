# Atelier AI — Data Dictionary

## Scope and Evidence

This dictionary covers the 25 article columns, 7 customer columns and
5 transaction columns present in the acquired CSV headers.

Descriptions are working interpretations based on column names,
inspected values and source documentation. They do not constitute a
complete official business glossary.

Complete-table diagnostics have established entity-key completeness and
uniqueness, transaction reference coverage and classification code-to-label
consistency for the acquired files. Results and limits are summarised below
and recorded in the analysis notebook.

Documenting a field does not establish its suitability as a model input
or its availability at historical prediction dates.

## Logical Roles and Current Reading Types

Logical roles describe how values should be interpreted. Reading types
describe their representation at a particular computation stage.

The reusable CSV readers initially preserve all fields as strings, with
automatic textual missing-value recognition disabled. Diagnostics use
explicit missingness rules and numerical or temporal copies as needed.
The initial typed loading scan additionally demonstrates these conversions:

| Fields | Raw reader type | Typed loading-scan type |
|---|---|---|
| Article and customer metadata | string | string |
| Transaction customer, article and channel identifiers | string | string |
| Transaction date | string | pandas datetime |
| Transaction price | string | nullable Float64 |

Source files and source lexical values remain unchanged. A converted
diagnostic series is not a cleaned or persisted training dataset.

Numeric-looking identifiers and category codes are not treated as
continuous measurements.

## Articles

Expected grain: one record per article.

`article_id` is the validated entity key and recommendation identifier.
All 105,542 keys are complete, unique, free of surrounding whitespace
and represented by exactly ten ASCII digits.

| Field(s) | Working meaning | Logical role and handling |
|---|---|---|
| `article_id` | Identifier of the article referenced by transactions. | Entity identifier; preserve its source representation. |
| `product_code` | Product grouping code; the complete catalogue has 47,224 codes for 105,542 articles. | Grouping identifier; not interchangeable with article_id. |
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

`customer_id` is the validated entity key. All 1,371,980 keys are
complete, unique, free of surrounding whitespace and represented by
exactly 64 lowercase hexadecimal characters.

| Field | Working meaning | Logical role and handling |
|---|---|---|
| `customer_id` | Customer identifier referenced by transactions. | Entity identifier; preserve as an opaque string. |
| `FN` | Source indicator with unresolved business semantics. The complete file contains blanks and the value '1.0'. | Preserve as supplied; do not interpret blanks as zero or include in initial model inputs without clarification. |
| `Active` | Source indicator with unresolved business semantics. The complete file contains blanks and the value '1.0'. | Preserve as supplied; do not equate it with recent purchasing activity. |
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
Repeated rows are retained: the source documentation describes them as
multiple purchases of the same item. Exact-pattern diagnostics use all
five source columns; no deduplication is applied.

| Field | Working meaning | Logical role and handling |
|---|---|---|
| `t_dat` | Recorded purchase date. | Temporal field; parsed using YYYY-MM-DD. |
| `customer_id` | Reference to the customer associated with the purchase. | Entity reference; all source rows match customers.customer_id. |
| `article_id` | Reference to the purchased article. | Entity reference and relevance identifier; all source rows match articles.article_id. |
| `price` | Recorded numeric price field. | Numerical measure; currency and scaling are not established in this project. No rescaling is applied. |
| `sales_channel_id` | Source sales-channel code. | Categorical code; preserve raw values without assigning an unverified channel-label mapping. |

During the initial typed scan, blank transaction dates and prices would
be converted to missing values and invalid non-blank values would raise
conversion errors. Subsequent complete-file diagnostics found neither
missing nor invalid dates or prices in this acquired version.

All dates are valid YYYY-MM-DD values in nondecreasing source order.
All prices are finite and strictly positive under the implemented
diagnostics. Sales-channel values are the source strings 1 and 2.
Successful checks do not establish price units or channel business labels.

## Complete-File Diagnostic Summary

| Area | Observed result | Handling consequence |
|---|---|---|
| Entity references | All 31,788,324 transactions match both entity tables; no missing or unknown references | Use the validated many-to-one join design and check later transformations |
| Missingness | Six fields contain null or blank values under the explicit policy | Preserve missingness; defer model-specific imputation |
| Customer age | 15,861 missing values; finite ages range from 16 to 99 | Keep historical availability separate from numerical validity |
| FN and Active | Only blanks and '1.0'; business semantics unresolved | Exclude from initial model inputs; do not infer zero from blanks |
| Textual markers | Nonempty Unknown, NONE and None values are retained as exact source categories | Review semantics before any replacement or merging |
| Classification pairs | Each code has one label across ten inspected pairs; some labels have multiple codes | Do not use labels as unique classification identifiers |
| Product grouping | 105,542 articles share 47,224 product codes; no article maps to multiple product codes | Recommend article_id and retain product_code for grouping |
| Exact transaction patterns | 2,974,905 additional occurrences beyond each pattern's first occurrence | Retain source rows; deduplicate relevance only per customer and target window |
| Temporal coverage | 2018-09-20 through 2020-09-22, with records on all 734 dates | Use the explicitly defined temporal protocol |

Missingness means null, empty or whitespace-only values. Nonempty textual
markers are inspected separately; a marker flag is not an invalidity rule.

In the classification mappings, `product_type_name`, `department_name`
and `section_name` each include labels associated with multiple codes.
This reverse nonuniqueness does not contradict code-to-label consistency.

The [data card](data_card.md) records detailed counts and handling limits.
The [evaluation strategy](evaluation.md) defines historical cutoffs and
local target windows. Passing structural checks does not establish
stock availability, metadata timing or predictive performance.

## Feature Eligibility

Potential uses include product-category preferences, historical
purchase recency and frequency, and historical price summaries.

These are hypotheses for later evaluation, not implemented features.

Transaction-derived features must use records preceding the prediction
cutoff. Metadata features require supporting temporal evidence or an
explicit, justified stability assumption.

Code and label columns may carry overlapping information. Their
representation will be selected during feature development and modelling
experiments, informed by the completed consistency checks.

## Open Questions

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
- [Notebook: inspected structure and diagnostics](../notebooks/data_foundation.ipynb)
