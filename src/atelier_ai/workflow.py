"""Reproducible preparation and first ranking experiment used by the notebooks."""

from hashlib import sha256
from pathlib import Path
from time import perf_counter
import json
import platform
import os
import subprocess
import duckdb
import numpy as np
import pandas as pd

from atelier_ai.feature_export import export_feature_batches, file_sha256
from atelier_ai.historical_contexts import build_historical_context
from atelier_ai.preparation import open_prepared, select_customer_cohort
from atelier_ai.recommendation_features import (
    FEATURE_COLUMNS,
    EXPECTED_MISSING_FEATURES,
    retrieve_candidates,
    prepare_feature_tables,
    build_checked_feature_batch,
    feature_storage_schema,
)
from atelier_ai.temporal_protocol import load_temporal_protocol
from atelier_ai.training_targets import make_training_schedule
from atelier_ai.ranking_data import (
    inspect_training_groups,
    load_eligible_training_rows,
    prepare_ranker_inputs,
)
from atelier_ai.ranking_metrics import evaluate_recommendations


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def load_protocol(root):
    return load_temporal_protocol(Path(root) / "configs/temporal.yaml")


def cohort(root):
    with open_prepared(root) as con:
        ids = con.execute("SELECT customer_id FROM customers").df()["customer_id"]
    selected = select_customer_cohort(ids)
    path = Path(root) / "data/processed/weekly_v1/cohort.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(path, index=False)
    return pd.Index(selected["customer_id"], dtype="string")


def build_context(root, ids, cutoff, *, prepared_directory=None):
    with open_prepared(root, prepared_directory=prepared_directory) as con:
        metadata = con.execute(
            "SELECT article_id,product_group_name,product_type_name,colour_group_name FROM articles"
        ).df()
        return build_historical_context(con, ids, metadata, cutoff=cutoff)


def target_reference(root, ids, cutoff, *, prepared_directory=None):
    """Return all source rows and all distinct cohort targets for one target week."""
    cutoff = pd.Timestamp(cutoff)
    with open_prepared(root, prepared_directory=prepared_directory) as con:
        con.register("_target_customers", pd.DataFrame({"customer_id": ids}))
        rows = con.execute(
            "SELECT t.* FROM transactions t INNER JOIN _target_customers c "
            "USING(customer_id) WHERE t_dat>=? AND t_dat<?",
            [cutoff, cutoff + pd.Timedelta(days=7)],
        ).df()
    targets = (
        rows[["customer_id", "article_id"]]
        .drop_duplicates()
        .sort_values(["customer_id", "article_id"])
        .reset_index(drop=True)
    )
    return rows, targets


def reference_dict(targets):
    return targets.groupby("customer_id")["article_id"].agg(set).to_dict()


def provenance(root, ids, *, snapshots_directory=None, source_names=None):
    root = Path(root)
    snapshots = (
        root / "data/processed/weekly_v1/source_snapshots"
        if snapshots_directory is None else Path(snapshots_directory)
    )
    snapshots.mkdir(parents=True, exist_ok=True)
    source_hashes = {}
    sources = (
        sorted((root / "src/atelier_ai").glob("*.py"))
        if source_names is None
        else [root / "src/atelier_ai" / name for name in source_names]
    )
    for source in sources:
        contents = source.read_bytes()
        digest = sha256(contents).hexdigest()
        snapshot = snapshots / (digest + ".py")
        if not snapshot.exists():
            snapshot.write_bytes(contents)
        source_hashes[source.name] = digest
    return {
        "protocol": load_protocol(root)["protocol_id"],
        "temporal_config_sha256": file_sha256(root / "configs/temporal.yaml"),
        "source_manifest_sha256": file_sha256(root / "docs/data_manifest.json"),
        "cohort_sha256": sha256(("\n".join(ids) + "\n").encode()).hexdigest(),
        "source_sha256": source_hashes,
        "history_limit": 30,
        "global_limit_7d": 100,
        "preferred_product_types": 3,
        "per_preferred_type": 50,
        "fallback_days": 28,
        "candidate_budget": 300,
        "python": platform.python_version(),
    }


def export_cutoff(root, ids, cutoff, *, split, notebook_path, context=None):
    root = Path(root)
    cutoff = pd.Timestamp(cutoff)
    context = build_context(root, ids, cutoff) if context is None else context
    pools, memberships, baselines, audit = retrieve_candidates(ids, context)
    tables = prepare_feature_tables(ids, context, memberships)
    targets = None
    if split == "training":
        _, targets = target_reference(root, ids, cutoff)
        targets = targets.assign(label=1)

    def builder(pairs, tables):
        result = build_checked_feature_batch(pairs, tables)
        if targets is not None:
            result = result.merge(
                targets,
                on=["customer_id", "article_id"],
                how="left",
                validate="one_to_one",
            )
            result["label"] = result["label"].fillna(0).astype("int64")
        return result

    result = export_feature_batches(
        ids,
        pools,
        tables,
        builder,
        output_base=root / "data/processed/weekly_v1/features",
        feature_columns=FEATURE_COLUMNS,
        nullable_features=EXPECTED_MISSING_FEATURES,
        sql_schema=feature_storage_schema(include_label=split == "training"),
        provenance=provenance(root, ids),
        notebook_path=notebook_path,
        candidates_per_customer=300,
        batch_customers=500,
        dataset_split=split,
    )
    return result, context, pools, baselines, audit


def evaluate_methods(root, ids, targets, methods, context, pools):
    reference = reference_dict(targets)
    results = {}
    summary = []
    group_map = (
        context["popularity"].set_index("article_id")["product_group_name"].to_dict()
    )
    catalogue = set(context["popularity"]["article_id"])
    target_count = len(targets)
    candidate_hits = sum(
        len(set(pools[c]) & articles) for c, articles in reference.items()
    )
    for method, lists in methods.items():
        result = evaluate_recommendations(lists, reference, ids, k=12)
        all_articles = {article for values in lists.values() for article in values}
        diversity = np.mean(
            [len({group_map[a] for a in values}) for values in lists.values()]
        )
        result["summary"].update(
            {
                "candidate_coverage": candidate_hits / target_count,
                "catalogue_coverage": len(all_articles) / len(catalogue),
                "mean_recommended_product_groups": float(diversity),
            }
        )
        results[method] = result
        summary.append({"method": method, **result["summary"]})
    return pd.DataFrame(summary), results


def export_validation(root):
    root = Path(root)
    protocol = load_protocol(root)
    ids = cohort(root)
    result, context, pools, baseline_rows, audit = export_cutoff(
        root,
        ids,
        protocol["validation_start"],
        split="validation",
        notebook_path=root / "notebooks/03_candidates_and_features.ipynb",
    )
    rows, targets = target_reference(root, ids, protocol["validation_start"])
    methods = {
        name: {c: values[name] for c, values in baseline_rows.items()}
        for name in next(iter(baseline_rows.values()))
    }
    comparison, evaluations = evaluate_methods(
        root, ids, targets, methods, context, pools
    )
    output = root / "data/processed/weekly_v1/validation"
    output.mkdir(parents=True, exist_ok=True)
    targets.to_csv(output / "targets.csv", index=False)
    audit.to_csv(output / "candidate_audit.csv", index=False)
    comparison.to_csv(output / "baselines.csv", index=False)
    pd.concat(
        [
            value["customer_scores"].assign(method=name)
            for name, value in evaluations.items()
        ]
    ).to_csv(output / "baseline_customer_scores.csv", index=False)
    write_json(output / "baseline_recommendations.json", methods)
    index = {
        "directory": str(result["directory"].relative_to(root)),
        "fingerprint": result["manifest"]["fingerprint"],
        "cohort_sha256": provenance(root, ids)["cohort_sha256"],
        "cutoff": str(protocol["validation_start"].date()),
        "source_transaction_rows": len(rows),
        "target_pairs": len(targets),
        "buyers": targets["customer_id"].nunique(),
        "feature_count": len(FEATURE_COLUMNS),
        "candidate_rows": result["checks"]["rows"],
    }
    write_json(root / "data/processed/weekly_v1/validation.json", index)
    return index, comparison


def export_training(root):
    root = Path(root)
    protocol = load_protocol(root)
    ids = cohort(root)
    schedule = make_training_schedule(
        validation_cutoff=protocol["validation_start"],
        n_weeks=protocol["training_weeks"],
    )
    exports = []
    all_paths = []
    summaries = []
    output = root / "data/processed/weekly_v1/training"
    output.mkdir(parents=True, exist_ok=True)
    for cutoff in schedule["cutoff"]:
        print(f"Training cutoff {cutoff.date()}", flush=True)
        result, _, _, _, _ = export_cutoff(
            root,
            ids,
            cutoff,
            split="training",
            notebook_path=root / "notebooks/04_training_dataset.ipynb",
        )
        paths = [
            str(result["directory"] / p["file"]) for p in result["manifest"]["parts"]
        ]
        all_paths.extend(paths)
        rows, targets = target_reference(root, ids, cutoff)
        targets.assign(cutoff=cutoff).to_csv(
            output / f"targets_{cutoff:%Y-%m-%d}.csv", index=False
        )
        with duckdb.connect(config={"threads": 2, "memory_limit": "512MB"}) as con:
            n, positive = con.execute(
                "SELECT count(*),sum(label) FROM read_parquet(?)", [paths]
            ).fetchall()[0]
        exports.append(
            {
                "cutoff": str(cutoff.date()),
                "directory": str(result["directory"].relative_to(root)),
                "fingerprint": result["manifest"]["fingerprint"],
            }
        )
        summaries.append(
            {
                "cutoff": cutoff,
                "candidate_rows": n,
                "groups": len(ids),
                "target_transaction_rows": len(rows),
                "target_pairs": len(targets),
                "buyers": targets["customer_id"].nunique(),
                "positive_candidates": int(positive),
                "candidate_coverage": int(positive) / len(targets),
            }
        )
    audit = inspect_training_groups(
        all_paths, validation_cutoff=protocol["validation_start"]
    )
    selected = load_eligible_training_rows(all_paths, audit)
    inputs = prepare_ranker_inputs(
        selected, validation_cutoff=protocol["validation_start"]
    )
    with duckdb.connect() as con:
        for frame, name in [
            (selected, "rows"),
            (audit, "group_audit"),
            (inputs["group_index"], "groups"),
        ]:
            con.register("_export", frame)
            con.sql("SELECT * FROM _export").write_parquet(
                str(output / f"{name}.parquet"), compression="zstd"
            )
            con.unregister("_export")
    summary = pd.DataFrame(summaries)
    summary.to_csv(output / "summary.csv", index=False)
    index = {
        "protocol": protocol["protocol_id"],
        "cohort_sha256": provenance(root, ids)["cohort_sha256"],
        "exports": exports,
        "raw_candidate_rows": int(summary["candidate_rows"].sum()),
        "raw_groups": len(audit),
        "eligible_groups": len(inputs["groups"]),
        "training_rows": len(selected),
        "positive_training_rows": int(selected["label"].sum()),
        "feature_columns": FEATURE_COLUMNS,
        "row_sha256": file_sha256(output / "rows.parquet"),
    }
    write_json(root / "data/processed/weekly_v1/training.json", index)
    return index, summary


def full_reference_ndcg(target_counts, *, budget=300, k=12):
    """Build an early-stopping metric whose ideal ranking includes retrieval misses."""
    counts = np.asarray(target_counts, dtype=int)
    eligible = counts > 0
    discounts = 1 / np.log2(np.arange(2, k + 2))
    ideal = np.array([discounts[: min(k, n)].sum() for n in counts])

    def metric(y_true, y_pred):
        labels = np.asarray(y_true).reshape(-1, budget)
        predictions = np.asarray(y_pred).reshape(-1, budget)
        if len(labels) != len(counts):
            raise ValueError("Validation group alignment differs.")
        # Input rows are ordered by article ID; a stable sort fixes score ties.
        order = np.argsort(-predictions, axis=1, kind="stable")[:, :k]
        hits = np.take_along_axis(labels, order, axis=1)
        dcg = (hits * discounts).sum(axis=1)
        score = float((dcg[eligible] / ideal[eligible]).mean())
        return "ndcg_full_reference_at_12", score, True

    return metric


def fit_ranker_configuration(
    root,
    *,
    num_leaves,
    min_child_samples,
    method_name,
    output_subdirectory,
    feature_columns=None,
):
    """Fit one declared configuration using the common validation protocol."""
    import lightgbm as lgb
    import mlflow

    columns = list(
        FEATURE_COLUMNS
        if feature_columns is None
        else feature_columns
    )

    ordered_subset = [
        column
        for column in FEATURE_COLUMNS
        if column in columns
    ]

    if not columns or columns != ordered_subset:
        raise ValueError(
            "Use a nonempty ordered subset of FEATURE_COLUMNS."
        )

    root = Path(root)
    protocol = load_protocol(root)
    ids = cohort(root)
    output = root / "data/processed/weekly_v1" / output_subdirectory
    output.mkdir(parents=True, exist_ok=True)
    train_index = json.loads(
        (root / "data/processed/weekly_v1/training.json").read_text()
    )
    validation_index = json.loads(
        (root / "data/processed/weekly_v1/validation.json").read_text()
    )
    with duckdb.connect(config={"threads": 2, "memory_limit": "1GB"}) as con:
        rows = con.execute(
            "SELECT * FROM read_parquet(?)",
            [str(root / "data/processed/weekly_v1/training/rows.parquet")],
        ).df()
    inputs = prepare_ranker_inputs(rows, validation_cutoff=protocol["validation_start"])
    del rows
    directory = root / validation_index["directory"]
    manifest = json.loads((directory / "manifest.json").read_text())
    paths = [str(directory / part["file"]) for part in manifest["parts"]]
    validation_dir = root / "data/processed/weekly_v1/validation"
    targets = pd.read_csv(validation_dir / "targets.csv", dtype="string")
    target_pairs = targets.assign(label=1)
    projection = ",".join(
        '"' + c + '"' for c in ["customer_id", "article_id", *FEATURE_COLUMNS]
    )
    with duckdb.connect(config={"threads": 2, "memory_limit": "1GB"}) as con:
        validation = con.execute(
            f"SELECT {projection} FROM read_parquet(?) ORDER BY customer_id,article_id",
            [paths],
        ).df()
    validation = validation.merge(
        target_pairs,
        on=["customer_id", "article_id"],
        how="left",
        validate="one_to_one",
        sort=False,
    )
    validation["label"] = validation["label"].fillna(0).astype("int8")
    ordered_ids = validation["customer_id"].drop_duplicates().tolist()
    sizes = validation.groupby("customer_id", sort=False).size().to_numpy(dtype="int32")
    if len(sizes) != 20_000 or not (sizes == 300).all():
        raise ValueError("Validation must retain all complete recipient groups.")
    counts = (
        targets.groupby("customer_id")
        .size()
        .reindex(ordered_ids, fill_value=0)
        .to_numpy()
    )
    X_validation = validation[columns].astype("float32")
    y_validation = validation["label"].to_numpy()
    validation_keys = validation[["customer_id", "article_id"]].copy()
    del validation
    params = {
        "objective": "lambdarank",
        "learning_rate": 0.05,
        "n_estimators": 1000,
        "num_leaves": num_leaves,
        "min_child_samples": min_child_samples,
        "max_depth": 8,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": 4,
        "deterministic": True,
        "force_col_wise": True,
        "verbosity": -1,
        "metric": "None",
    }
    model = lgb.LGBMRanker(**params)
    learning = {}
    start = perf_counter()
    model.fit(
        inputs["X"][columns].astype("float32"),
        inputs["y"],
        group=inputs["groups"],
        eval_set=[(X_validation, y_validation)],
        eval_group=[sizes],
        eval_metric=full_reference_ndcg(counts),
        eval_at=[12],
        callbacks=[
            lgb.early_stopping(50, first_metric_only=True),
            lgb.record_evaluation(learning),
            lgb.log_evaluation(25),
        ],
    )
    fit_seconds = perf_counter() - start
    model.booster_.save_model(str(output / "model.txt"))

    reloaded = lgb.Booster(model_file=str(output / "model.txt"))

    if reloaded.feature_name() != columns:
        raise ValueError(
            "Reloaded model feature order differs from the selected columns."
        )

    probe = X_validation.iloc[:1000]

    if not np.allclose(
        model.predict(probe),
        reloaded.predict(probe),
        rtol=0,
        atol=1e-12,
    ):
        raise ValueError("Reloaded model predictions differ.")
    start = perf_counter()
    scores = reloaded.predict(X_validation, num_threads=4)
    score_seconds = perf_counter() - start
    validation_keys["score"] = scores
    top = (
        validation_keys.sort_values(
            ["customer_id", "score", "article_id"], ascending=[True, False, True]
        )
        .groupby("customer_id", sort=False)
        .head(12)
    )
    learned = top.groupby("customer_id")["article_id"].agg(list).to_dict()
    del X_validation, validation_keys, inputs
    baseline_methods = json.loads(
        (validation_dir / "baseline_recommendations.json").read_text()
    )
    methods = {**baseline_methods, method_name: learned}
    context = build_context(root, ids, protocol["validation_start"])
    pools, _, _, _ = retrieve_candidates(ids, context)
    comparison, evaluations = evaluate_methods(
        root, ids, targets, methods, context, pools
    )
    comparison.to_csv(output / "comparison.csv", index=False)
    top.to_csv(output / "recommendations.csv", index=False)
    pd.concat(
        [r["customer_scores"].assign(method=name) for name, r in evaluations.items()]
    ).to_csv(output / "customer_scores.csv", index=False)
    write_json(output / "learning_curve.json", learning)
    importance = pd.DataFrame(
        {
            "feature": columns,
            "gain": reloaded.feature_importance(importance_type="gain"),
        }
    ).sort_values("gain", ascending=False)
    importance.to_csv(output / "feature_importance.csv", index=False)
    best = comparison.sort_values(["map_at_k", "method"], ascending=[False, True]).iloc[
        0
    ]
    code_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    spec = {
        "method_name": method_name,
        "feature_columns": columns,
        "parameters": params,
        "early_stopping_rounds": 50,
        "best_iteration": int(model.best_iteration_),
        "training": train_index,
        "validation": validation_index,
        "provenance": provenance(root, ids),
        "git_head_at_run": code_commit,
        "working_tree_sources_hashed": True,
        "fit_seconds": fit_seconds,
        "score_seconds": score_seconds,
        "model_sha256": file_sha256(output / "model.txt"),
        "leading_method": str(best["method"]),
        "hyperparameter_search_complete": False,
        "v1_evaluated": False,
        "confirmation_evaluated": False,
        "final_refit_complete": False,
    }
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri((root / "mlruns").as_uri())
    mlflow.set_experiment("atelier-ai-weekly-v1")
    with mlflow.start_run(run_name=method_name) as run:
        mlflow.log_params(params)
        mlflow.log_params(
            {
                "protocol": protocol["protocol_id"],
                "feature_count": len(columns),
                "best_iteration": int(model.best_iteration_),
            }
        )
        for name, value in evaluations[method_name]["summary"].items():
            mlflow.log_metric(name, float(value))
        mlflow.log_metrics({"fit_seconds": fit_seconds, "score_seconds": score_seconds})
        for step, value in enumerate(
            learning["valid_0"]["ndcg_full_reference_at_12"], 1
        ):
            mlflow.log_metric("validation_ndcg_at_12", value, step=step)
        mlflow.log_dict(spec, "experiment_spec.json")
        for name in [
            "model.txt",
            "comparison.csv",
            "learning_curve.json",
            "feature_importance.csv",
        ]:
            mlflow.log_artifact(str(output / name))
        spec["mlflow_run_id"] = run.info.run_id
    write_json(output / "manifest.json", spec)
    return spec, comparison


def fit_first_ranker(root):
    """Run the original initial configuration and preserve its output contract."""
    return fit_ranker_configuration(
        root,
        num_leaves=31,
        min_child_samples=100,
        method_name="lightgbm_initial",
        output_subdirectory="first_ranker",
    )
