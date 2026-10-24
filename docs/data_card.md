# Atelier AI — Data Card

## Status

The three required CSV files have been acquired locally. File sizes and SHA-256 hashes are recorded in [data_manifest.json](data_manifest.json) and have been checked against the local files.

Complete-file structural and data-quality diagnostics have been executed. Temporal coverage and the initial local evaluation windows are established. Calculations, aggregate outputs and figures are available in the [analysis notebook](../notebooks/01_data_foundation.ipynb); reusable functions and automated tests are included in the repository.

Behavioural EDA, feature engineering and model evaluation remain subsequent work.

## Source

- Dataset: H&M Personalized Fashion Recommendations
- Publisher: H&M Group
- Platform: Kaggle
- Documentation review date: 2026-10-22

The source benchmark predicts purchases during the seven days following the supplied purchase history. Atelier AI defines historical validation and local test periods within that history.

## File Scope and Observed Dimensions

| File | V1 decision | Rows | Columns | Project use |
|---|---|---:|---:|---|
| `transactions_train.csv` | Required | 31,788,324 | 5 | Purchase histories and evaluation targets |
| `articles.csv` | Required | 105,542 | 25 | Article attributes and candidate feature exploration |
| `customers.csv` | Required | 1,371,980 | 7 | Customer population and metadata inspection |
| `sample_submission.csv` | Optional; outside the current diagnostic scope | — | — | Submission-format reference |
| `images/` | Deferred | — | — | Potential future visual or multimodal experiments |

Including a file in the inspection scope does not imply that every column is eligible for modelling.

## Grain and Relationships

- `articles.csv`: one record per `article_id`; all 105,542 keys are complete and unique.
- `customers.csv`: one record per `customer_id`; all 1,371,980 keys are complete and unique.
- `transactions_train.csv`: recorded article purchases, with no order, basket or event identifier supplied.

All 31,788,324 transaction rows reference existing customer and article keys. No missing or unknown references were detected. These results support many-to-one metadata joins on the validated entity keys; later transformations must preserve the intended grain.

Article keys contain exactly ten ASCII digits; customer keys contain exactly 64 lowercase hexadecimal characters in the acquired entity tables. Identifiers remain exact strings.

The 105,542 articles share 47,224 `product_code` values. Recommendations use `article_id`; `product_code` is a grouping identifier.

## Missingness and Categories

Missingness is defined as null, empty or whitespace-only values. Nonempty textual values such as `Unknown`, `NONE` and `None` remain separate source categories.

| Field | Missing rows | Percentage of source rows |
|---|---:|---:|
| `customers.Active` | 907,576 | 66.15% |
| `customers.FN` | 895,050 | 65.24% |
| `customers.fashion_news_frequency` | 16,009 | 1.17% |
| `customers.age` | 15,861 | 1.16% |
| `customers.club_member_status` | 6,062 | 0.44% |
| `articles.detail_desc` | 416 | 0.39% |

No missing values under this policy were detected in the other source columns. Percentages are rounded and use the row count of the corresponding file.

`FN` and `Active` contain blanks and `'1.0'`; their business and missing-value semantics remain unresolved. They are excluded from initial model inputs. Blank values are not recoded as zero.

`fashion_news_frequency` contains `'NONE'` in 877,711 rows and `'None'` in two rows. These categories remain separate. Article classification labels include nonempty `'Unknown'` categories, which are documented without automatic replacement.

## Numerical and Classification Checks

Finite recorded ages range from 16 to 99, with 15,861 missing ages. No nonmissing parsing failures, nonfinite values, nonpositive values, fractional ages or ages above the diagnostic review threshold of 100 were detected.

Transaction dates are complete, valid ISO dates in nondecreasing source order. Prices are complete, parseable, finite and strictly positive under the implemented diagnostics; units and scaling remain unresolved. Channel codes are `'1'` and `'2'`, without an assigned business-label mapping.

For all ten inspected article classification code/label pairs, each code maps to one label. Some labels map to multiple codes. Labels are therefore not assumed to be unique identifiers.

Passing these checks does not establish the business meaning or historical availability of a field.

## Repeated Transactions and Handling

Exact repetition is defined by equality of source values across all five transaction columns.

- Distinct exact patterns: 28,813,419.
- Patterns occurring more than once: 2,543,908.
- Rows belonging to repeated patterns: 5,518,813.
- Additional occurrences beyond each pattern's first occurrence: 2,974,905.

The source documentation describes repeated rows as multiple purchases of the same item. All transaction rows are retained. Repetition does not identify an order, basket or ingestion error.

Ranking relevance is defined separately: one distinct relevant `article_id` per customer and seven-day target window, regardless of repeated purchases.

Source files remain unchanged. No automatic filtering, imputation, deduplication or price rescaling has been applied.

## Temporal Coverage and Evaluation

Transactions cover 2018-09-20 through 2020-09-22, inclusive, with records on all 734 calendar dates. Calendar continuity does not establish that the source contains every real-world transaction.

All intervals below have an inclusive start and an exclusive end.

| Period | Start inclusive | End exclusive | Source transaction rows |
|---|---|---|---:|
| Pre-validation history | 2018-09-20 | 2020-09-09 | 31,292,772 |
| Validation target | 2020-09-09 | 2020-09-16 | 255,241 |
| Local test target | 2020-09-16 | 2020-09-23 | 240,311 |

Counts retain repeated source rows. The history pool is not an already constructed supervised training table.

Transaction-derived features and candidates must precede the prediction cutoff. Historical training target windows must end no later than validation starts. Completed behavioral EDA used for model-design decisions covers pre-validation history only. Its reconciled findings and testable modeling hypotheses are recorded in the [behavioral EDA summary](eda_summary.md) and [behavioral analysis notebook](../notebooks/02_behavioral_eda.ipynb).

For V1, selected learned parameters remain fixed between validation and test, while historical features and candidates are refreshed at each cutoff. See the [evaluation strategy](evaluation.md) for the complete protocol.

## Metadata Availability

Customer and article metadata lack sufficient per-record temporal provenance to establish availability at every historical cutoff. Metadata features require supporting evidence or an explicit, justified stability assumption before use.

Time-varying attributes are not assumed safe for historical modelling merely because they appear in the supplied files.

## Access and Publication

Access is subject to the competition rules. The rules describe non-commercial, academic and educational uses and restrict redistribution of competition data.

Source files are excluded from Git. Published notebook outputs contain structural diagnostics and aggregate summaries rather than source transaction or customer records. Future outputs and images require the same publication review.

## Interpretation Limits and Remaining Decisions

- Observed purchases are incomplete signals of customer preferences.
- Stock availability is not established by historical purchase records.
- `FN`, `Active`, price units, sales-channel labels and metadata observation dates remain unresolved.
- Imputation, transformations and model-input eligibility require decisions during feature development.
- One validation week and one local test week provide an initial assessment; wider temporal robustness remains to be evaluated.
- Offline results will not establish performance in luxury retail or measurable business impact.

## References

- [Dataset documentation](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data)
- [Competition rules](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/rules)
- [Project brief](project_brief.md)
- [Evaluation strategy](evaluation.md)
- [Data dictionary](data_dictionary.md)
- [Acquisition manifest](data_manifest.json)
- [Analysis notebook](../notebooks/01_data_foundation.ipynb)

- [Behavioral EDA summary](eda_summary.md)
- [Behavioral analysis notebook](../notebooks/02_behavioral_eda.ipynb)
