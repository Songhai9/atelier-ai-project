# Atelier AI - Behavioral EDA Summary

## Scope and measurement

Notebook 02 uses [2018-09-20, 2020-09-02), excluding all subsequent target weeks. Complete-scope aggregates cover 31,028,115 retained source purchase rows, 713 dates, 1,345,600 observed customers, 101,945 articles and 26,653,314 distinct customer/article pairs.

Source rows are not orders; a purchase date is not a basket. Preserve transaction repetitions, `Unknown` categories and exact article IDs. Snapshot classifications are descriptive, not evidence of historical stock or complete lifetime preference.

## Executed analyses and hypotheses

| Analysis | Evidence and implication |
|---|---|
| Daily and weekday activity | Daily rows, trailing seven-day mean, weekday means/medians and date counts; motivate bounded recent signals |
| Weekly categories | Category-share trajectories and complete category/week heatmap; motivate cutoff-specific affinity and popularity |
| Customer activity and recency | Median distinct articles 8; median cutoff recency 146 days; motivate history segments and missing-history fallback |
| Repeat purchases | Distinguish one row, multiple rows on one date and multiple purchase dates; keep previous articles eligible |
| Repeat delays | 1,233,464 transitions between distinct customer/article purchase dates, plus activity-band summaries; motivate interaction recency |
| Popularity concentration | Top approximately 1% of observed articles accounts for 18.491% of source rows; motivate popularity references and catalogue diagnostics |
| Top-seller renewal | Median 11 new articles in successive weekly top-20 lists; motivate 7/28-day article features |
| Breadth and colour affinity | Customer category/colour counts and row-normalised category/colour matrix; motivate diversity and affinity features |
| Price | Category medians and 10th-90th percentiles, plus weekly mean/median histories; keep source units and distinguish composition from price changes |
| Channel | Within-category source-channel proportions; retain codes without an unverified business mapping |

Figures retain titles, units, population and periods. The first/last calendar weeks are partial. Percentile bars show observation spread, not confidence intervals. Repeated-date delays exclude same-day row repetitions. Customer/article IDs and future targets do not determine descriptive cohort membership.

## Modelling hypotheses

1. Recent popularity may provide a useful reference; compare its validation ranking quality.
2. Historical product-type affinity may improve retrieval or ranking; compare preferred-type and global methods.
3. Previously purchased articles may be relevant again; compare recent-purchase recommendations and interaction features.
4. Personalisation quality may depend on activity and recency; use paired history segments.
5. Colour affinity, article trends and historical context may add information; test declared feature ablations with fixed candidates.

These observations do not establish causal effects, future predictive gains, calibrated purchase probabilities, churn or stock availability. Subsequent experiments preserve complete target references and the [temporal evaluation contract](evaluation.md).
