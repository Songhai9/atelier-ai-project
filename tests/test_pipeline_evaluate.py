"""Full-reference V1 replay including retrieval misses, nonbuyers and pairing."""

from hashlib import sha256
import json
from pathlib import Path
import shutil

import duckdb
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

import atelier_ai.pipeline_dataset as dataset
import atelier_ai.pipeline_evaluate as evaluate
import atelier_ai.pipeline_features as features
import atelier_ai.pipeline_model as model
import atelier_ai.pipeline_predict as predict
from atelier_ai.preparation import prepare_sources
from atelier_ai.recommendation_features import FEATURE_COLUMNS
from atelier_ai.training_targets import make_training_schedule


@pytest.fixture
def evaluation_campaign(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")
    shutil.copytree(Path(evaluate.__file__).parent, tmp_path / "src/atelier_ai",
                    ignore=shutil.ignore_patterns("__pycache__"))
    raw = tmp_path / "data/raw/hm"
    raw.mkdir(parents=True)
    articles = "article_id,product_group_name,product_type_name,colour_group_name\n"
    articles += "".join(f"{i:010d},A,T{i % 3},Blue\n" for i in range(1, 304))
    purchases = "t_dat,customer_id,article_id,price,sales_channel_id\n"
    purchases += "".join(f"2020-07-21,outside,{i:010d},0.1,1\n" for i in range(1, 301))
    purchases += "2020-07-20,u,0000000001,0.1,1\n"
    purchases += "2020-07-22,u,0000000001,0.1,1\n" * 2
    purchases += "2020-07-23,u,0000000301,0.1,1\n"
    purchases += "2020-07-29,u,0000000002,0.1,1\n2020-07-30,u,0000000302,0.1,1\n"
    purchases += "2020-08-05,u,0000000001,0.1,1\n" * 2  # Included start; duplicate source purchase.
    purchases += "2020-08-06,u,0000000303,0.1,1\n2020-08-06,cold,0000000303,0.1,1\n"
    purchases += "2020-08-06,outside,0000000001,0.1,1\n"  # Outside the frozen cohort.
    purchases += "2020-08-12,quiet,0000000001,0.1,1\n"  # Reserved next window; excluded end.
    records = []
    for name, text in {"articles.csv": articles, "customers.csv": "customer_id\nu\ncold\nquiet\noutside\n",
                       "transactions_train.csv": purchases}.items():
        path = raw / name
        path.write_text(text)
        records.append({"path": path.relative_to(tmp_path).as_posix(),
                        "size_bytes": path.stat().st_size, "sha256": sha256(path.read_bytes()).hexdigest()})
    docs, configs = tmp_path / "docs", tmp_path / "configs"
    docs.mkdir()
    configs.mkdir()
    (docs / "data_manifest.json").write_text(json.dumps({"schema_version": 1, "files": records}))
    (configs / "temporal.yaml").write_text(
        'protocol_id: "fixture"\ntraining_weeks: 1\nhistory_days: 180\nhorizon_days: 7\n'
        'validation_start: "2020-07-29"\nv1_evaluation_start: "2020-08-05"\n'
        'confirmation_start: "2020-08-12"\nconfirmation_end_exclusive: "2020-08-19"\n')
    method = "lightgbm_leaves_63_min_20"
    evidence = tmp_path / "data/processed/weekly_v1/diagnostics" / method / "segmentation_manifest.json"
    evidence.parent.mkdir(parents=True)
    evidence.write_text(json.dumps({"baseline": "recent_purchases_180d"}))
    run = tmp_path / "data/processed/pipeline_runs/unit"
    prepared = prepare_sources(tmp_path, output_directory=run / "prepared")
    ids = pd.Index(["u", "cold", "quiet"], dtype="string")
    cohort_hash = sha256(("\n".join(ids) + "\n").encode()).hexdigest()
    context = {
        "customer_ids": ids, "cohort_sha256": cohort_hash,
        "schedule": make_training_schedule(validation_cutoff="2020-08-05", n_weeks=2),
        "prepared_directory": prepared,
        "prepared_manifest_sha256": model.file_sha256(prepared / "manifest.json"),
        "feature_columns": FEATURE_COLUMNS, "candidate_budget": 300,
    }
    parameters = {"objective": "lambdarank", "learning_rate": 0.05, "n_estimators": 8,
                  "num_leaves": 63, "min_child_samples": 20, "max_depth": 8, "reg_lambda": 1.0,
                  "random_state": 42, "n_jobs": 4, "deterministic": True, "force_col_wise": True,
                  "verbosity": -1, "metric": "None"}
    cutoffs = ["2020-07-22", "2020-07-29"]
    decision = {
        "method": method, "feature_columns": FEATURE_COLUMNS, "refit_parameters": parameters,
        "training": {"cohort_sha256": cohort_hash},
        "retrieval": {"history_limit": 30, "global_limit_7d": 100, "preferred_product_types": 3,
                      "per_preferred_type": 50, "fallback_days": 28, "candidate_budget": 300},
        "refit_protocol": {"training_cutoffs": cutoffs, "early_stopping": False, "uses_v1_targets": False},
        "v1_evaluation": {"start_inclusive": "2020-08-05", "end_exclusive": "2020-08-12", "k": 12,
                          "full_reference_targets": True, "target_metric_population": "all target buyers",
                          "prior_development_exposure": True},
        "confirmation_reserved": True,
        "evidence_sha256": {evidence.relative_to(tmp_path).as_posix(): model.file_sha256(evidence)},
    }
    (docs / "frozen.json").write_text(json.dumps({"decision": decision}))
    plan = {
        "run_id": "unit", "output_root": str(run), "method": method, "refit_parameters": parameters,
        "training_cutoffs": cutoffs, "v1_window": {"start_inclusive": "2020-08-05", "end_exclusive": "2020-08-12"},
        "inputs": {"frozen_selection": str(docs / "frozen.json")},
        "stages": [{"name": name, "planned_manifest": str(run / directory / "manifest.json")}
                   for name, directory in [("prepare_sources", "prepared"), ("build_training", "training"),
                                           ("fit_model", "model"), ("predict_v1", "predictions"),
                                           ("evaluate_v1", "evaluation")]],
    }
    for module in (features, dataset, predict):
        monkeypatch.setattr(module, "load_training_context", lambda root, plan: context)
    dataset.build_training_dataset(tmp_path, plan)
    model.fit_frozen_model(tmp_path, plan)
    predict.predict_v1(tmp_path, plan)
    return tmp_path, plan, run


def _unreachable(*args, **kwargs):
    raise AssertionError("A completed upstream stage must not be regenerated.")


def test_full_targets_zero_retrieval_buyer_and_nonbuyer_denominators(evaluation_campaign, monkeypatch):
    root, plan, run = evaluation_campaign
    upstream = {name: (run / name / "manifest.json").read_bytes() for name in ("training", "model", "predictions")}
    monkeypatch.setattr(predict, "_export_features", _unreachable)
    monkeypatch.setattr(predict, "_score_features", _unreachable)
    result = evaluate.evaluate_v1(root, plan)
    report = result["manifest"]
    assert report["recipients"] == 3 and report["target_buyers"] == 2
    assert report["excluded_nonbuyers"] == 1
    assert report["source_purchase_rows"] == 4 and report["target_pairs"] == 3
    assert report["candidate_target_pairs"] == 1
    assert report["baseline_comparator"] == "recent_purchases_180d"
    assert report["confirmation_evaluated"] is False
    bundle = run / "evaluation/bundle"
    targets = model._read_parquet(bundle / "targets.parquet")
    assert targets.loc[targets["article_id"].eq("0000000303"), "in_candidates"].eq(False).all()
    population = model._read_parquet(bundle / "population.parquet").set_index("customer_id")
    assert bool(population.loc["cold", "evaluated"]) is True
    assert population.loc["cold", "candidate_target_articles"] == 0
    assert bool(population.loc["quiet", "evaluated"]) is False
    scores = model._read_parquet(bundle / "customer_scores.parquet")
    assert set(scores["customer_id"]) == {"u", "cold"}
    assert scores.loc[scores["customer_id"].eq("cold"), ["ap_at_k", "recall_at_k", "ndcg_at_k"]].eq(0).all().all()
    comparison = model._read_parquet(bundle / "comparison.parquet")
    assert comparison["candidate_coverage"].eq(1 / 3).all()
    assert comparison["evaluated_customers"].eq(2).all()
    assert comparison["recommendation_customers"].eq(3).all()
    assert comparison["target_pairs"].eq(3).all()
    for method in report["methods"]:
        lists = evaluate._methods(run / "predictions/bundle", report)[method]
        rank = lists["u"].index("0000000001") + 1 if "0000000001" in lists["u"] else None
        actual = comparison.set_index("method").loc[method]
        assert actual["map_at_k"] == pytest.approx(1 / (4 * rank) if rank else 0)
        assert actual["recall_at_k"] == pytest.approx(0.25 if rank else 0)
    samples = model._read_parquet(bundle / "bootstrap_samples.parquet")
    assert len(samples) == 5000
    np.testing.assert_array_equal(samples["model_minus_baseline"],
                                  samples[report["method"]] - samples[report["baseline_comparator"]])
    assert evaluate.verify_v1_evaluation(root, plan) == report
    for name, content in upstream.items():
        assert (run / name / "manifest.json").read_bytes() == content
    monkeypatch.setattr(evaluate, "target_reference", _unreachable)
    second = evaluate.evaluate_v1(root, plan)
    assert second["reused"] is True and second["manifest"] == report


@pytest.mark.parametrize("name", ["targets.parquet", "bootstrap_samples.parquet", "source/pipeline_evaluate.py"])
def test_damaged_evaluation_is_rejected_and_preserved(evaluation_campaign, name):
    root, plan, run = evaluation_campaign
    evaluate.evaluate_v1(root, plan)
    path = run / "evaluation/bundle" / name
    damaged = path.read_bytes() + b"damaged"
    path.write_bytes(damaged)
    with pytest.raises(ValueError, match="checksum|snapshot differs"):
        evaluate.evaluate_v1(root, plan)
    assert path.read_bytes() == damaged


def test_comparator_evidence_cannot_change_after_freeze(evaluation_campaign):
    root, plan, run = evaluation_campaign
    evidence = root / "data/processed/weekly_v1/diagnostics/lightgbm_leaves_63_min_20/segmentation_manifest.json"
    evidence.write_text(json.dumps({"baseline": "recent_popularity_7d"}))
    with pytest.raises(ValueError, match="comparator evidence differs"):
        evaluate.evaluate_v1(root, plan)
    assert not (run / "evaluation").exists()


def test_interrupted_tracking_reuses_saved_evaluation(evaluation_campaign, monkeypatch):
    from mlflow.tracking import MlflowClient

    root, plan, run = evaluation_campaign
    log = MlflowClient.log_artifact

    def interrupt(*args, **kwargs):
        raise RuntimeError("Synthetic evaluation tracking interruption")

    monkeypatch.setattr(MlflowClient, "log_artifact", interrupt)
    with pytest.raises(RuntimeError, match="evaluation tracking interruption"):
        evaluate.evaluate_v1(root, plan)
    assert not (run / "evaluation/manifest.json").exists()
    before = (run / "evaluation/bundle/targets.parquet").read_bytes()
    monkeypatch.setattr(MlflowClient, "log_artifact", log)
    monkeypatch.setattr(evaluate, "target_reference", _unreachable)
    result = evaluate.evaluate_v1(root, plan)
    assert result["reused"] is True
    assert (run / "evaluation/bundle/targets.parquet").read_bytes() == before
    assert evaluate.verify_v1_evaluation(root, plan) == result["manifest"]


def test_changed_evaluation_code_preserves_previous_results(evaluation_campaign):
    root, plan, run = evaluation_campaign
    evaluate.evaluate_v1(root, plan)
    before = (run / "evaluation/manifest.json").read_bytes()
    source = root / "src/atelier_ai/pipeline_evaluate.py"
    source.write_text(source.read_text() + "\n# changed evaluation dependency\n")
    with pytest.raises(ValueError, match="Evaluation recipe differs"):
        evaluate.evaluate_v1(root, plan)
    assert (run / "evaluation/manifest.json").read_bytes() == before


def test_paired_bootstrap_matches_manual_draws_and_preserves_zero_buyers():
    methods = ["base", "model"]
    scores = pd.DataFrame({"customer_id": ["z", "a", "z", "a"],
                           "method": ["base", "base", "model", "model"],
                           "ap_at_k": [1.0, 0.0, 1.0, 0.0]})
    samples, summary = evaluate.paired_map_bootstrap(scores, methods, "model", "base",
                                                    resamples=20, seed=42, batch_size=3)
    indices = np.random.default_rng(42).integers(0, 2, size=(20, 2))
    expected = np.array([0.0, 1.0])[indices].mean(axis=1)
    np.testing.assert_array_equal(samples["base"], expected)
    np.testing.assert_array_equal(samples["model"], expected)
    assert samples["model_minus_baseline"].eq(0).all()
    assert summary["evaluated_buyers"].eq(2).all()
    assert summary.loc[summary["method"].eq("model_minus_baseline"), ["observed", "ci_lower", "ci_upper"]].eq(0).all().all()
    replay, _ = evaluate.paired_map_bootstrap(scores.iloc[::-1], methods, "model", "base",
                                             resamples=20, seed=42, batch_size=20)
    assert_frame_equal(samples, replay)


def test_bootstrap_rejects_different_buyer_populations():
    scores = pd.DataFrame({"customer_id": ["u", "v", "u"],
                           "method": ["base", "base", "model"], "ap_at_k": [0, 1, 1]})
    with pytest.raises(ValueError, match="same complete buyer population"):
        evaluate.paired_map_bootstrap(scores, ["base", "model"], "model", "base")
