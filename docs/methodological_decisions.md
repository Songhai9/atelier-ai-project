# Atelier AI — Methodological Decision Register

This register distinguishes task definitions, practical starting choices and measured evidence. It is intended to support an honest technical discussion. It does not retroactively turn initial choices into experimentally established optima.

## Current preparation status

The four notebooks are executed and saved. The selected historical training export reconstructs X with 532,500 rows and 17 features, aligned y with 2,753 positive labels, and 1,775 groups of 300 consecutive rows. All 58 current tests pass. No learned ranker has been fitted yet.

## Decision register

| Decision | Current rationale | Evidence and limit | Comparison still needed |
|---|---|---|---|
| Recommend 12 distinct articles | Follow the project's MAP@12 task and evaluation convention | Fixed output-list length; not an optimised advisor-interface choice | A business study would be needed to justify 12 for real advisor usage |
| Retrieve 300 candidates per customer–week | A bounded initial scope: 6 million rows per 20,000-customer cutoff; compatible with the current bounded history/category proposals and global fill | A practical starting choice, not a measured optimal candidate count. Alternative retrieval policies were compared at 300; different total budgets were not compared | Candidate-budget sensitivity at 200, 300 and 500, with full-reference coverage, ranking quality and measured resource use |
| History proposal cap 100; category proposal cap 100 | Bound two personalised retrieval sources before global fill | These are maximum source proposals, not an equal three-way split. Overlap and short history often leave many more global-fill slots. With total budget 300, at least 100 first-addition slots remain for global fill | Source-cap ablations; do not confuse a plausible configuration with a proven optimum |
| 180-day personal history | A concrete retrospective window for recency, frequency and group-affinity features | Data coverage is available and extraction boundaries are tested. Predictive superiority over shorter/longer windows is not established | Compare alternative windows while controlling candidate policy and evaluation population |
| 7-day global popularity | A recent-activity reference for a weekly prediction task | The matching cadence is intuitive, not proof that this window is best | Compare recent-popularity windows after the first reference model |
| 7-day targets | Fixed local task definition, applied consistently to historical learning, validation and test | All intervals use inclusive start/exclusive end; no target-week activity enters its own features | Different horizons would define another task and require a separate evaluation |
| Four historical training cutoffs | Initial recent historical supervision scope before validation | The four target weeks are complete and disjoint. This is not four-fold cross-validation or proof of temporal robustness | More historical training weeks and additional chronological validation origins |
| Fixed 20,000-customer cohort | Keep the first local workflow tractable and comparisons aligned | Cohort selection uses seeded SHA-256 ordering independently of target purchase activity; sparse buyers are retained in reference population | Broader cohorts and sensitivity to cohort choice before generalisation claims |
| Seed 42 | Reproduce a declared initial run | The particular number has no performance justification or special statistical status | Replicated cohorts/seeds where uncertainty and robustness require it |
| Source purchase repetitions retained | Preserve acquired observations for frequency calculations | Relevance is separately deduplicated per customer–article–week | Do not silently treat every repeated source row as a data error |
| Seventeen historical features | Operationalise EDA motivations using customer, article, interaction and retrieval information | Historical boundaries, joins and storage are validated. Individual predictive benefit has not yet been demonstrated by a learned model | Feature-family ablations and validation error analysis |
| Product-group metadata | Use the broad classification with an explicit stability assumption | Historical observation dates are unavailable; stability is not verified | Reassess if timestamped catalogue data becomes available |
| Keep training groups with both 0 and 1 labels | Provide observed within-group preference contrasts for initial binary grouped ranking | Selection is restricted to historical training and is conditional on labels; it changes the training distribution | Evaluate the learned model on the complete validation population, not only on retrieved-positive groups |
| Keep all 300 candidates of selected groups | Avoid introducing another sampling policy in the first local experiment | 532,500 rows are manageable; class imbalance remains. No synthetic positives or extra sample weights were introduced | Negative-sampling/weighting experiments only if there is a justified need |
| Batch 500 customers | Bound the current feature DataFrame while keeping each group intact | This is an execution parameter, not a learned-model hyperparameter or a measured optimum. Per-frame memory is not peak process RAM | Measure actual runtime and process memory before tuning batch size |
| Learned-model configuration | Not yet selected or fitted | No claim of an optimised learning rate, tree count or architecture is currently supported | Record fixed initial parameters before the first run, then distinguish subsequent validation tuning |

## What the current evidence establishes

The full four-cutoff candidate collection has 24 million rows and 80,000 customer–week groups. It retrieves 2,753 of 13,732 reference positive pairs. Weekly micro coverage is 17.68%, 18.68%, 21.08% and 22.61%; macro coverage is 17.98%, 19.21%, 22.62% and 25.71%. These are retrieval diagnostics, not learned-model performance or business impact.

The selected training subset retains 1,775 complete groups and all 2,753 currently retrieved positives. It excludes constant-label training groups from ranker fitting while preserving raw exports and the audit. Validation/test recipient populations and full reference targets are unaffected.

Notebook 03's validation baselines and same-budget candidate-policy comparisons establish reference behaviour. They do not establish that 300 candidates, the source caps or historical windows are optimal.

## Candidate-budget sensitivity plan

### Purpose

Measure the trade-off between missed relevant articles, downstream ranking quality and execution cost. Treat the complete retrieval-plus-ranking pipeline as the system being compared.

### Controlled starting comparison

Compare total budgets **200, 300 and 500**, keeping the history/category proposal caps at 100 each, the cohort, cutoffs, target definitions, feature definitions and the initial ranker configuration fixed.

A total budget of 100 is incompatible with the current requirement that both source caps sum to at most the total budget. Testing 100 would require an explicitly different cap configuration. Even with fixed caps, changing the total budget changes the amount of global fill; report source additions and memberships rather than implying constant source proportions.

### Rebuild affected inputs

Regenerate candidates and cutoff-specific feature tables for each total budget. In particular, the `from_global` flag uses the top-budget global proposal, and `n_retrieval_sources` depends on these memberships. Do not silently reuse 300-budget feature values as if all retrieval-derived features were invariant.

Recompute historical labels and training eligibility after retrieval. Report how many groups and retrieved positives become available at each budget. These population changes are part of the end-to-end system effect; they prevent interpreting the comparison as only a ranking-cost intervention.

### Measurements

For every budget, report:

- Micro and macro candidate coverage using all reference purchases.
- Candidate/source counts, retrieved positive counts and eligible training-group counts.
- Full-reference validation MAP@12 and macro Recall@12 after retraining the ranker.
- Candidate construction, feature construction, training and prediction durations, plus file sizes and a clearly identified process-memory measurement where available.

Keep the final list length at 12. Rank recommendations for all 20,000 validation recipients. Exclude no buyer from target metrics merely because retrieval missed all their purchases. Keep the existing convention for weeks without observed target purchases and report their proportion.

### Selection and reporting

Choose using validation MAP@12 under the existing protocol. Retain the simpler method on an exact tie. Interpret changes alongside coverage and resource use; do not invent an unmeasured latency target. Reserve local-test outcomes until the configuration is frozen.

Report this study as a finite sensitivity comparison, not proof of a globally optimal candidate count or universal generalisation. Record unsuccessful comparisons too.

## Interview wording supported today

“I started with a bounded retrieval configuration so that I could validate the full historical pipeline locally. The current system retrieves 300 candidates from personal history, a dominant category and recent global popularity. I have compared source policies at that fixed size, but I have not yet compared total candidate budgets. The exact count remains an initial engineering choice. I plan to compare coverage, full-reference validation ranking quality and execution cost before defending it as the selected configuration.”

Replace this wording with actual experimental results once the budget comparison has been run. Do not present the planned experiment as completed work.
