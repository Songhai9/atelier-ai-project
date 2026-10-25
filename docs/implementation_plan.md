# Atelier AI - Implementation Plan

## Objective and working sequence

Support a client advisor with a ranked selection of twelve distinct fashion articles per anonymised customer. Build the scientific workflow first, then package it as a usable local service. Purchases are implicit relevance signals; ranking scores are not calibrated purchase probabilities.

The sequence is explanatory notebooks → reusable functions → reproducible pipeline → application. Completed deliverables are recorded from executed evidence.

## Scientific stages

| Plan stage | Deliverable | Current evidence |
|---|---|---|
| 1. Problem framing | Brief, dictionary, advisor workflow | `docs/project_brief.md`, `docs/data_dictionary.md` |
| 2. Data foundation | Immutable raw sources, checksums, diagnostics, typed Parquet and fixed cohort | Notebook 01; `preparation.py`; acquisition manifest |
| 3. Temporal protocol | Six training targets, validation, V1 evaluation, reserved final confirmation | `configs/temporal.yaml`; notebook 01; boundary tests |
| 4. Behavioral EDA | Questions, full-scope aggregates, figures, hypotheses and limits | Notebook 02; `docs/eda_summary.md` |
| 5. Retrieval and features | Historical candidates and feature dictionary | Notebook 03; `recommendation_features.py` |
| 6. Simple references | Popularity, preferred types, recent purchases; candidate coverage | Notebook 03; complete-reference validation |
| 7. LightGBM | Initial ranker, then controlled six-configuration search and MLflow | Completed: notebook 04 prepares groups, notebook 05 records the initial ranker, notebook 06 records all six configurations and the stage-8 candidate |
| 8. Explain and select | Importance, SHAP, ablations, robustness, final refit and V1 assessment | Completed in notebook 07: selected-model diagnostics, frozen eight-tree refit and intermediate V1; see evaluation report and model card |

Notebook numbers identify the learning sequence and are not a second definition of the original plan stages. The five French revision sheets follow this notebook sequence.

## Data and temporal contract

Use H&M transactions, customer and article tables. Preserve raw bytes, string IDs, valid source categories and repeated transaction rows. Distinct relevance is defined separately by customer, article and target week. Never automatically remove identical purchase rows.

Select 20,000 customers by stable seeded hashing of source customer IDs, independently of future activity. Calculate global popularity using all eligible source customers. Typed Parquet supports repeated bounded scans.

At each historical cutoff use the preceding 180 days and predict the next seven days. Training cutoffs are 22 July, 29 July, 5 August, 12 August, 19 August and 26 August 2020. Validation starts 2 September; intermediate V1 evaluation starts 9 September; final confirmation starts 16 September and ends 23 September, exclusively. Behavioral EDA ends before validation. See `evaluation.md` for development-exposure limits.

## EDA scope

Use question → calculation → chart → observation → hypothesis → modelling consequence. Cover source quality, daily activity and rolling means, weekday activity, weekly category composition/heatmaps, customer activity and recency, repeated purchases and repeat delays by activity, popularity concentration and top-seller renewal, category/colour breadth and affinities, prices by category and over time, and channel/category composition.

Use complete announced data for summary statistics. Declare any point sampling and its seed. Keep prices in source units; first observed purchase is not a product launch. Snapshot metadata does not establish historical stock.

## Candidate policy and feature families

At each cutoff concatenate the latest 30 distinct personal articles, global 7-day top 100, and 50 recent articles from each of three preferred product types. Deduplicate in that priority order; complete to 300 distinct articles using 28-day global popularity. Candidate articles must have been observed before prediction. Keep repurchases and define a popularity fallback for customers without observed history.

Feature families cover customer activity/recency and category diversity; article 7/28-day popularity, trend and last observation; previous customer/article purchases and recency; category/colour affinities; prediction date and historical channel share; retrieval memberships. Raw IDs do not become ordered numeric features. Fit any learned transformations on training only. Customer metadata without historical provenance remains exploratory.

## LightGBM and model selection

Use LambdaRank with customer/week groups and binary observed-purchase labels. Keep target purchases absent from candidates in the evaluation reference. Initial configuration: learning rate 0.05, maximum 1,000 iterations, 31 leaves, depth 8, minimum 100 examples per leaf, L2=1 and seed 42.

The six declared combinations of leaves {15,31,63} × minimum examples {20,100} are complete. All stop on full-reference validation NDCG@12 with patience 50. Selection by complete-reference MAP@12 retains 63 leaves / minimum 20 and iteration 8 as the stage-8 candidate. MLflow records configuration, identities, metrics, duration, model and figures. See [tuning results](lightgbm_tuning_results.md).

Stage 8 covers native SHAP examples, gain importance, ablations without recency/affinities/popularity, successes and failures, MAP/Recall/NDCG@12, candidate/catalogue coverage, recommended category diversity, activity/history segments and generation/scoring durations. Bootstrap customers for MAP uncertainty. Freeze choices and iteration budget, refit a selected learned model on training plus validation, then assess V1 without tuning on its outcomes. Retain the simpler method on an exact tie; a baseline can serve if it leads.

## Local product and reproducibility - subsequent milestone

Extract the validated workflow into preparation, targets, references, training, evaluation and serving stages. DVC records the stage dependencies. Large data, MLflow storage and generated models remain outside Git.

The serving artifact contains the selected method/model, transformations, required aggregates, catalogue, demonstration profiles and version manifest. FastAPI exposes `/health`, `/ready`, `/customers`, `/customers/{id}/profile` and `/recommendations`. An unknown customer receives an explicit popularity fallback; their profile remains absent. Responses contain article IDs, ranks, ranking scores, factual presentation fields, strategy, historical date and versions.

Streamlit provides customer, selection, comparison and analysis views. Display the historical reference date rather than implying current stock. Docker Compose runs API and UI. GitHub Actions tests a small synthetic pipeline and builds images; full training remains explicit. Validate data contracts, temporal safety, deterministic distinct recommendations, save/reload behavior, API fallbacks and the advisor workflow. Report measured latency and artifacts, not assumed business benefits.

## Subsequent enhancements

**PyTorch:** client/article embeddings of dimension 32, standardised numerical variables, dense layers 128/64, ReLU, dropout 0.2, within-group purchased/unobserved pairs, AdamW at 0.001, maximum 20 epochs with validation stopping. Compare common candidates/periods/metrics, training duration, latency and size; unknown IDs use fallback. Freeze both fitting procedures before final confirmation.

**GCP:** reuse containers with Cloud Storage artifacts, Artifact Registry images, Cloud Run API/UI, explicit Cloud Run batch Jobs, Terraform and federated GitHub Actions authentication. Target `europe-west1` after checking availability. MLflow remains local with exported results. Verify project and billing before provisioning; no cloud resource is required for local V1.

**GenAI:** a Gemini Flash advisor assistant calls tools for profiles, recommendations and article metadata. Verify the exact deployment model at that milestone. Prepare 30 cases covering usual requests, unknown clients, missing history/information and tool failures; measure grounded article references and tool success.


## Verified stage-8 completion

Notebook 07 covers every scientific stage-8 item: gain importance, native SHAP examples, three declared ablations, successes/failures, complete-reference ranking/coverage/diversity metrics, activity/history segments, local generation/scoring timings and paired customer bootstrap.

The 63/20 configuration with all 31 features was frozen with an eight-iteration refit budget. Seven cutoffs produce 2,974 informative groups / 892,200 fitting rows. Intermediate V1 gives MAP@12=0.022927 versus 0.022168 for the validation-selected recent-purchases baseline; the paired 95% interval of the difference is [-0.002626, 0.004346]. This does not establish clear superiority. V1 does not change the frozen decision.

The next reference-plan section is the reproducible local pipeline and application. Final confirmation stays reserved for a later frozen LightGBM/PyTorch comparison. See [evaluation report](stage8_evaluation_report.md) and [model card](model_card.md).


## Verified local pipeline completion

The seven local pipeline stages have been executed for `local_v1`, from source-file validation through the portable serving bundle. The unified CLI supports status, full verification, individual stages and resumable execution. DVC 3.67.1 executed its graph using all seven verified stage outputs, wrote the eight-stage `dvc.lock`, and completed the final integrity check. The Python suite passed 134 synthetic tests.

This replay preserves the historical native model identity and V1 results; final confirmation remains reserved. The bundle contains 20,000 historical customer profiles, 50,383 observed catalogue articles, 113,678 preceding purchase rows and 240,000 learned recommendations. Its reader supports known customers, reference-method comparison and an explicit popularity fallback for unknown IDs without raw data or MLflow access.

The next implementation increment is FastAPI around the packaged reader, followed by Streamlit, containers and synthetic CI. See [local pipeline instructions](local_pipeline.md) for the exact reproduction prerequisites and the scope of the completed DVC reuse check.
