# Atelier AI - Methodological Decisions

| Decision | Rationale | Verification or limitation |
|---|---|---|
| Seven-day article relevance and twelve distinct outputs | Matches the stated next-purchase ranking task | All targets remain in ranking denominators |
| Fixed 20,000-ID cohort, SHA-256 seed 42 | Bounds computation independently of future activity and source row order | Snapshot does not prove historical account existence |
| Six training weeks and three subsequent evaluation roles | Separates fitting, selection, V1 assessment and later model confirmation | Development exposure is disclosed in `evaluation.md` |
| 180-day personal history and global 7/28-day popularity | Explicit bounded historical signals | Initial choices, not measured optimal windows |
| Retain exact source repetitions | Rows may represent repeated purchases; no order key exists | Relevance separately deduplicates customer/article/week |
| 30 personal + 100 global + up to 150 preferred-type proposals, 28-day fill to 300 | Covers personal history, current popularity and three observed type affinities under a fixed budget | Proposal memberships overlap; do not sum them as distinct articles |
| Product types for retrieval; broad groups for additional affinity/diversity | Distinguishes classification granularity | Classifications use a documented snapshot stability assumption |
| Historical colour affinities and channel-code share | Adds conditional preference/context measurements without next-week fields | No causal preference or verified channel-name mapping |
| 31 numeric/boolean features; missing unobserved recencies | Preserves the difference between absence and measured zero | Raw IDs and customer snapshot attributes are not model inputs |
| Complete informative groups for fitting | Pairwise ordering needs positive and negative candidates | Training-only filter; keep complete evaluation recipients and references |
| Complete-reference MAP selection and NDCG stopping | Separates final metric choice from the smooth rank-sensitive stopping signal | Missing-retrieval buyers score zero; nonbuyers excluded with share reported |
| Initial configuration, then six declared comparisons | Bounds capacity search and makes choices inspectable | All six trials recorded; MAP selects 63/20 at iteration 8 for stage 8 |
| Native model reload check and MLflow provenance | Connects the scored model to the recorded data/configuration/code | Local fit/scoring durations are not serving latency |
| Final refit only after selection freeze | Uses training plus validation without choosing iterations from test outcomes | Intermediate V1 is retrospective; final model confirmation remains reserved |

## Completed controlled comparisons and final decision

The declared search of leaves {15,31,63} × minimum examples {20,100} is complete. Complete-reference validation MAP@12 selects 63/20 at iteration 8 as the reference candidate. Among exactly tied learned configurations, the declared sorting prefers fewer maximum leaves and then a higher minimum leaf size; a baseline remains preferred on an exact tie with the leading ranker. Notebook 07 completes ablations without the declared recency, affinity and popularity columns, keeping candidates and evaluation conventions fixed. Each variant reselects its stopping iteration under the common validation rule. The full feature set has the highest MAP of these variants. Native SHAP, errors, descriptive history segments, timings and customer bootstrap are recorded before final selection/refit.

No experiment presupposes LightGBM superiority. A baseline remains eligible for serving. Extra candidate-budget or window sensitivity experiments require a declared scope and do not replace the reference plan's current next increment.


The final decision freezes all 31 features and eight refit iterations. The seven-week refit uses no V1 targets or early stopping. V1 has a small positive MAP difference with a paired interval containing zero, while baseline Recall/NDCG remain slightly higher. This result is reported without retuning or replacing the frozen method using V1 outcomes.

Candidate coverage limits end-to-end micro Recall; its numerical rate is not a direct MAP ceiling because MAP uses per-buyer normalization and weighting. Catalogue coverage and category diversity remain separate measures: on V1, LightGBM has lower catalogue coverage but higher mean broad-group diversity than the comparator.

Selected-model validation SHAP/ablations/segments are not explanations or estimates for the final refitted model. Validation and V1 comparisons differ in both week and fitted model; a score change alone does not identify its cause. Final confirmation remains reserved. See [stage-8 report](stage8_evaluation_report.md).
