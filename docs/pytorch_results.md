# Atelier AI - Local PyTorch and Frozen Final Confirmation

## Result and scope

A pairwise neural ranker and its final refit are complete locally. The five frozen systems were compared on [2020-09-16, 2020-09-23). Recent purchases has the highest MAP@12 point estimate; PyTorch hybrid does not establish a clear MAP advantage over LightGBM. No model is retuned or replaced using final outcomes. FastAPI/Streamlit continue serving the 2020-09-09 LightGBM V1 snapshot; neural serving and DVC integration are separate work.

## Task, populations and evaluation

At a cutoff, generate 300 distinct historically observed candidate article IDs per customer, then recommend twelve. Use the fixed 20,000-customer cohort and 180-day personal lookback; global history can include customers outside the cohort. All candidates/features precede their own prediction cutoff. Customer and article IDs retain their string identities; article_id is not product_code. Purchase absence is an unobserved signal, not evidence of disinterest.

Confirmation includes 3,736 cohort source purchase rows, preserved with repetitions, and 3,267 distinct customer–article relevance pairs. All 1,027 target buyers enter equal-weight MAP/macro Recall/NDCG means, including buyers with no retrieved target. The 18,973 nonbuyers (94.865%) are excluded from these means, but all recipients receive lists. Candidate coverage is 22.5895% of all 3,267 targets. Metrics follow [the evaluation contract](evaluation.md); list coverage and diversity use all 20,000 recipients.

## Architecture and learning

Two 32-dimensional trainable embeddings represent customer and article IDs. A training-fitted median imputer/scaler transforms the 31 ordered numeric features; six fixed missingness flags produce 37 numeric inputs. The concatenated dimension is 101. The head is 128 ReLU/dropout(0.2), 64 ReLU/dropout(0.2), then one unrestricted score. Zero is reserved in both vocabularies and embeddings. The architecture and `encode_ids` live in `src/atelier_ai/neural_ranker.py`.

Within each complete informative 300-candidate group, loss is `mean(softplus(s_negative - s_positive))` over every positive–negative pair. Group means are averaged across the batch. No negative sampling or target-dependent evaluation filtering is used. AdamW: learning rate 0.001, weight decay 0.01, seed 42; batch size 16 groups. Maximum 20 epochs, patience three, strict full-reference validation MAP improvement; a tie keeps the earlier epoch.

Initial fitting uses six cutoffs, 2,533 groups / 759,900 rows / 3,787 positives / 1,128,529 positive–negative pairs. Its vocabularies contain 2,120 known customers and 13,528 known articles; 522,177 parameters. Validation selects epoch 3 (MAP 0.026914980) and stops after epoch 6. LightGBM leads validation MAP at 0.032710133; recent purchases gives 0.025444640. Decreasing training loss alone does not justify later epochs.

The final refit newly initializes weights and refits preprocessing/vocabularies on seven cutoffs, ending with 2020-09-02. All fitting targets end by 2020-09-09. Exactly three epochs use 2,974 groups / 892,200 rows / 4,461 positives, without early stopping or V1/confirmation targets. The final network has 560,961 parameters, 2,406 known customers and 14,454 known articles, plus reserved indices. The saved model.pt is 2,247,281 bytes; reload maximum absolute score error is 0.0 on 1,000 probes. Native LightGBM model.txt is 60,306 bytes; different serialization formats and runtime requirements limit this size comparison.

## Hybrid inference policy

Known customer IDs use neural scores. Unknown customer IDs use `recent_purchases_180d`, including that baseline's own historical/popularity fill. Unknown article IDs use a fixed zero embedding alongside historical numeric features. "Unknown to the fitting vocabulary" is not "new customer/article in lifetime history".

At confirmation, 2,406 of 20,000 recipients use the neural model; 17,594 use the fixed fallback. Among 1,027 evaluated buyers, 328 use neural scores and 699 fallback. Of 721,800 neural candidate rows (2,406 × 300), 168,380 have an unknown article ID. Reported results concern this full hybrid, not an embedding model applied to every recipient. All systems share candidates/targets and deterministic score/article-ID tie ordering.

## Confirmation results

| System | MAP@12 | Macro Recall@12 | NDCG@12 | Hits / 3,267 targets |
|---|---:|---:|---:|---:|
| recent_popularity_7d | 0.006555 | 0.019679 | 0.012617 | 64 |
| preferred_product_types | 0.005354 | 0.015364 | 0.010033 | 45 |
| recent_purchases_180d | **0.020584** | 0.044484 | 0.030262 | 101 |
| lightgbm_leaves_63_min_20 | 0.017709 | 0.037696 | 0.026555 | 93 |
| pytorch_hybrid | 0.020575 | **0.046191** | **0.031193** | 108 |

The declared contrast is **PyTorch hybrid minus LightGBM MAP@12**: 0.002865909, percentile 95% interval [-0.002320312, 0.008159653]. Resample the same 1,027 target buyers with replacement for every method, 5,000 times, seed 42, including zero APs. The generic stored bootstrap label `model_minus_baseline` refers to this explicit contrast. The interval crosses zero: neither superiority nor equivalence is demonstrated. No paired contrast against recent purchases was evaluated. Secondary Recall/NDCG rankings are descriptive.

These intervals condition on fixed systems and one observed week; they exclude model-selection, fitting and between-week variation. Earlier validation and V1 were exposed during development. Final confirmation is now also exposed and must not become another tuning set. Results do not establish causal business gains, calibrated probabilities, current stock availability, luxury-retail transfer or production performance.

## Recorded campaign and MLflow identities

Canonical comparison root: `data/processed/pytorch_v1/559f9ddac46f/`. Preserve it explicitly rather than selecting the latest directory. Notebook 08 also contains a later training campaign (`5699605587da`); notebook 09 intentionally pins `559f9ddac46f`. Do not merge/delete campaigns or infer identity from matching rounded metrics.

| Evidence | Local relative path / run ID |
|---|---|
| Initial selection | `selection.json`, `verification.json`; MLflow `09b11a08cea74116801184eef9e16977` |
| Fixed neural refit | `final_refit/{model.pt,numeric_preprocessor.joblib,vocabularies.json,protocol.json,manifest.json}` |
| Refit MLflow | `327c3e6b5622411ca4538a07ffb58346` |
| Frozen comparison | `confirmation/protocol.json`, `recommendations.json`, `source_purchases.csv`, `targets.csv` |
| Evaluation evidence | `confirmation/{comparison.csv,customer_scores.csv,bootstrap_samples.csv,bootstrap_summary.csv,candidate_audit.csv,manifest.json,comparison.html,comparison.png}` |
| Confirmation MLflow | `88fc1a0580a944acb65aec96a6da8e1f`, FINISHED; twelve logged artifacts match local bytes |

Neural final model SHA-256: `dd364f447c38138e192e6e3bf0451eefdf1db8c348485937d789960390ac9abe`.

Frozen LightGBM native model: `data/processed/pipeline_runs/local_v1/model/bundle/model.txt`, SHA-256 `d6ca939c488a707c9a8c5be7480f738df640d2bc0a3496fdcb20b573f8353bf3`.

Final comparison manifest SHA-256: `b6f96c92d5e4289923aa9edd6d6c53098ef969bbf735a2b2dce714297383f784`.

Freeze-time fields such as `confirmation_evaluated: false` remain unchanged in immutable selection/refit/protocol records; the later confirmation manifest records completion. Notebook/source snapshots and checksums, not a Git base alone, identify executed working-tree code.

## Reproduction and verification limits

- Notebook 08 contains training/selection/refit. Its organizational refactor retains historical outputs and was not followed by a fresh complete training replay. Do not present its saved counters as evidence of a new end-to-end run.
- Notebook 09 starts independently in a fresh kernel, imports the shared class, verifies the explicitly selected campaign and loads saved predictions by default. Its seven substantive code cells were executed and saved successfully. Completed tables are verified/reloaded, not re-estimated for each review.
- Required generated data are outside Git: prepared H&M files, cohort, frozen artifacts and the checksummed feature export `data/processed/weekly_v1/features/test_2020-09-16_fae8368dc524`. A public checkout alone does not include these licensed/derived bytes. Follow source access conditions and preparation steps in the README.
- `REBUILD_PREDICTIONS=True` is an explicit reproduction check against immutable lists; it never trains or silently accepts changed predictions. That optional branch in the reorganized notebook 09 was not newly replayed during closure.
- macOS native crashes were consistent with OpenMP runtime collisions, distinct from a post-restart `NameError` caused by lost imports. Notebook 09 sets `OMP_NUM_THREADS=1` and `OMP_THREAD_LIMIT=1` before numerical imports, imports PyTorch first, limits PyTorch threads and predicts LightGBM with one thread. This is a verified local workaround, not a general portability guarantee. Do not suppress duplicate-runtime errors with unsafe runtime overrides.
- Checks covered saved cell outputs/errors, immutable hashes, model reload probes, prediction/target metrics, stored bootstrap summaries and matching MLflow artifacts. No new training, resampling or full notebook replay was run by the reviewer. The previous 158-test application suite is historical and was not rerun for this neural milestone.
- Fit durations are recorded per epoch; they omit preparation and other stages. No controlled common inference-latency benchmark or production capacity claim was completed. Containers, CI, cloud and neural serving integration remain deferred.

## Primary method references

- [PyTorch embeddings](https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html): index lookup and reserved padding representation.
- [PyTorch softplus](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.softplus.html): smooth pairwise score-difference loss used here.
- [PyTorch autograd](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html): differentiation and parameter updates.
- [scikit-learn leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html): learn preprocessing on fitting rows only, reuse transformations on later data.
- [LightGBM OpenMP issue 6595](https://github.com/lightgbm-org/LightGBM/issues/6595): upstream runtime-collision reports; the local crash evidence and chosen workaround are documented separately above.
