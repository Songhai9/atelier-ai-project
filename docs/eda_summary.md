# Atelier AI - Behavioral EDA Summary

## Status and Evidence

Behavioral EDA and its modeling-hypothesis register are complete. Results come from saved, executed outputs in the [behavioral EDA notebook](../notebooks/02_behavioral_eda.ipynb), including its reconciliation checks and Plotly figures with PNG outputs.

No supervised training examples, recommendation baselines or learned rankers have been implemented at this milestone. Predictive performance and business impact remain unmeasured.

## Scope and Measurement Contract

All results below use [2018-09-20, 2020-09-09), excluding validation and local-test targets.

| Population or measure | Observed count |
|---|---:|
| Retained transaction rows | 31,292,772 |
| Calendar dates with source records | 720 |
| Observed customers | 1,351,314 |
| Observed article IDs | 102,967 |
| Customer-article pairs | 26,882,193 |
| Observed product groups | 19 |

Counts retain source transaction repetitions. Transaction rows do not identify orders or baskets. Recommendation identity remains `article_id`; different IDs sharing a `product_code` are not merged. The source category `Unknown` remains retained.

## Main Findings

### Customer Activity - Notebook Section 4

- Exactly one transaction row is observed for 130,217 customers, approximately 9.636% of EDA customers.
- Exactly one purchase date is observed for 444,424 customers, approximately 32.89%.
- The median number of distinct purchased articles is eight.
- Median recency at the EDA cutoff is 149 days. Approximately 17.39% of observed customers have a purchase within the final 28 days.
- Activity, recency and history length describe different aspects of observed histories; they do not establish churn, loyalty or complete lifetime behavior.

### Article Popularity and Classification - Notebook Section 5

- The leading 1,030 articles, approximately 1% of observed articles, account for 18.554% of retained transaction rows.
- Reaching at least 80% of rows requires 21,338 articles, or 20.723% of observed articles.
- The top twenty articles by row count and distinct observed buyers share eighteen articles, while individual ranks can differ.
- Product-group and product-type joins preserve all 102,967 observed articles and 31,292,772 rows.
- Garment Upper body accounts for 39.354% of rows; Garment Lower body accounts for 22.156%.

### Repeat Purchases and Observed Concentration - Notebook Section 6

| Customer-article pattern | Share of observed pairs | Share of retained rows |
|---|---:|---:|
| One transaction row | 87.130% | 74.849% |
| Multiple rows on one date | 8.702% | 16.189% |
| Purchases on multiple dates | 4.168% | 8.962% |

Pair shares are not customer shares. Rows associated with multiple-date pairs include their initial observations and are not a count of repeat-purchase events.

Between one and three product groups are observed for 752,427 customers, or 55.681% of EDA customers. Among customers observed in exactly three groups, the median maximum within-customer group share is 50% under both row-based and distinct-article measures. The maximizing group can differ between measures; these summaries discard its identity.

### Temporal Variation - Notebook Section 7

- Daily row counts range from 12,760 on 2020-01-01 to 198,622 on 2019-09-28.
- The largest trailing seven-day mean is approximately 90,841.6 rows per day, dated 2019-06-25.
- Saturday has the highest mean daily volume; Thursday has the highest median. Weekday rankings depend on the statistic.
- June 2019 has the highest mean daily volume among complete months, approximately 63,540.1 rows. June 2020 records approximately 58,816.9 rows per day.
- The 25 retained calendar months include two partial months: September 2018 has eleven dates and September 2020 has eight. Their results describe those dates only.
- Monthly product-group counts form a complete 25-by-19 grid. Counts reconcile with monthly totals and shares sum to 100% within each month.
- Product-group composition changes over time. For example, Swimwear accounts for 15.37% of rows in June 2019 and 14.25% in June 2020, compared with 1.30% in October 2019.
- Relative shares and absolute counts can move in opposite directions: Swimwear rows rise from 212,776 in May 2020 to 251,523 in June, while its monthly share falls from 15.62% to 14.25%.

## Modeling Hypotheses - Notebook Section 8

All hypotheses remain untested.

| ID | Hypothesis | Planned evaluation |
|---|---|---|
| H1 | Recent popularity may outperform longer-window popularity. | Compare popularity baselines on validation MAP@12. |
| H2 | Historical category affinity may improve over global popularity. | Compare the category-preference baseline with recent global popularity, with fallback coverage. |
| H3 | Previously purchased articles may contribute useful recommendations. | Compare recent purchases completed with popular articles against the popularity baseline. |
| H4 | Personalization gains may vary with history amount and recency. | Report history-segment results and compare customer-activity feature groups. |
| H5 | Recent product-group activity may add ranking information. | Compare rankers with and without this feature group on fixed candidates. |

Follow the [evaluation strategy](evaluation.md): compute histories, features and candidates strictly before each cutoff; recompute aggregates for historical examples; fit learned preprocessing on training examples only. Compare feature groups on fixed candidates and candidate sources under a controlled budget.

Use validation MAP@12 for selection, with Recall@12, candidate coverage and catalogue coverage as supporting diagnostics. Keep all target articles in evaluation, including those absent from candidates. Report the exclusion of customer-week pairs without target purchases. Freeze model-selection choices before inspecting the local test.

## Interpretation Limits

- Observed histories do not establish complete customer lifetimes, intrinsic preferences or dislikes.
- Source rows and same-date repetitions do not identify orders, baskets or purchase sessions.
- Purchase counts do not establish product exposure or historical stock availability; zero month-group counts are not evidence of stockouts.
- Metadata snapshots do not establish availability at historical cutoffs. Feature use needs temporal evidence or an explicit, justified stability assumption.
- Descriptive temporal patterns do not establish causes, stable seasonality or future predictive value.
- One validation week and one test week provide an initial assessment; additional historical cutoffs are needed for broader temporal robustness.

## Next Milestone

Implement candidate generation, cutoff-safe features and the declared reference baselines before assessing a learned ranker. The EDA does not predetermine which method will perform best.

## References

- [Behavioral EDA notebook](../notebooks/02_behavioral_eda.ipynb)
- [Data foundation notebook](../notebooks/01_data_foundation.ipynb)
- [Evaluation strategy](evaluation.md)
- [Data card](data_card.md)
- [Project brief](project_brief.md)
