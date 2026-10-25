"""Portable, read-only access to a frozen historical recommendation bundle."""

from hashlib import sha256
import json
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd


TABLE_NAMES = (
    "recommendations", "customer_profiles", "customer_history", "article_aggregates",
    "affinity_group", "affinity_type", "affinity_colour", "customer_interactions",
    "catalogue", "fallback", "comparison", "bootstrap_summary", "candidate_audit",
)
PROVENANCE_NAMES = ("selection", "model", "predictions", "evaluation")
PAYLOAD_FILES = (
    *(f"{name}.parquet" for name in TABLE_NAMES), "model.txt", "reload_check.parquet",
    "baseline_recommendations.json", "cohort.csv", "feature_contract.json",
    *(f"provenance/{name}.json" for name in PROVENANCE_NAMES),
)


def file_identity(path):
    path = Path(path)
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"size_bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_table(directory, name):
    _require(name in TABLE_NAMES or name == "reload_check", "Unknown serving table.")
    with duckdb.connect(config={"threads": 2, "memory_limit": "1GB"}) as connection:
        return connection.execute("SELECT * FROM read_parquet(?)", [str(Path(directory) / f"{name}.parquet")]).df()


def verify_serving_bundle(bundle_directory):
    """Verify the package using only files inside it; never open upstream paths."""
    directory = Path(bundle_directory).resolve()
    report = _read(directory / "manifest.json")
    _require(report.get("schema_version") == 1 and report.get("stage") == "package_service"
             and report.get("status") == "complete"
             and report.get("serving_mode") == "precomputed_recommendations"
             and report.get("confirmation_evaluated") is False,
             "A complete historical serving package is required.")
    expected = set(PAYLOAD_FILES) | {f"source/{name}" for name in report["source_sha256"]}
    _require(set(report["artifacts"]) == expected, "Serving artifact list differs.")
    actual = {path.relative_to(directory).as_posix() for path in directory.rglob("*") if path.is_file()}
    _require(actual == expected | {"manifest.json"}, "Serving package contains unexpected or missing files.")
    for name, identity in report["artifacts"].items():
        path = directory / name
        _require(not Path(name).is_absolute() and ".." not in Path(name).parts
                 and path.resolve().is_relative_to(directory), "Serving artifact escapes its bundle.")
        _require(path.is_file() and file_identity(path) == identity, f"Serving checksum differs: {name}")
    for name, digest in report["source_sha256"].items():
        _require(report["artifacts"][f"source/{name}"]["sha256"] == digest,
                 f"Serving source snapshot differs: {name}")
    for name, digest in report["input_sha256"].items():
        _require(report["artifacts"][f"provenance/{name}.json"]["sha256"] == digest,
                 f"Serving input evidence differs: {name}")
    contract = _read(directory / "feature_contract.json")
    selection = _read(directory / "provenance/selection.json")["decision"]
    model = _read(directory / "provenance/model.json")
    prediction = _read(directory / "provenance/predictions.json")
    evaluation = _read(directory / "provenance/evaluation.json")
    _require(
        report["method"] == selection["method"] == model["method"] == prediction["method"] == evaluation["method"]
        and report["feature_columns"] == selection["feature_columns"] == model["feature_columns"] == contract["feature_columns"]
        and contract["input_dtype"] == "float32" and contract["history_days"] == 180
        and contract["customer_snapshot_attributes_used"] is False
        and contract["stock_availability_known"] is False
        and contract["learned_score_kind"] == "uncalibrated_lightgbm_ranking_score"
        and contract["fallback_score_kind"] == "historical_purchase_rows_7d"
        and contract["retrieval"] == selection["retrieval"]
        and model["selection_sha256"] == prediction["provenance"]["frozen_selection_sha256"] == evaluation["selection_sha256"] == report["input_sha256"]["selection"]
        and prediction["model_manifest_sha256"] == report["input_sha256"]["model"]
        and evaluation["prediction_manifest_sha256"] == report["input_sha256"]["predictions"]
        and report["model_sha256"] == model["artifacts"]["model.txt"]["sha256"] == prediction["model_sha256"] == evaluation["model_sha256"]
        and report["artifacts"]["model.txt"] == model["artifacts"]["model.txt"]
        and report["artifacts"]["reload_check.parquet"] == model["artifacts"]["reload_check.parquet"]
        and report["artifacts"]["recommendations.parquet"] == prediction["artifacts"]["recommendations.parquet"]
        and report["artifacts"]["baseline_recommendations.json"] == prediction["artifacts"]["baseline_recommendations.json"],
        "Serving inputs do not describe the same frozen system.",
    )
    _require(report["cohort_sha256"] == prediction["provenance"]["cohort_sha256"] == model["cohort_sha256"]
             and model["num_trees"] == report["num_trees"] == selection["refit_parameters"]["n_estimators"]
             and evaluation["confirmation_evaluated"] is False
             and prediction["target_labels_used_for_prediction"] is False,
             "Serving model or temporal lifecycle differs from the verified inputs.")
    cutoff = pd.Timestamp(report["historical_date"])
    _require(cutoff == pd.Timestamp(prediction["start_inclusive"])
             and cutoff == pd.Timestamp(evaluation["start_inclusive"]), "Serving reference date differs.")
    cohort = pd.read_csv(directory / "cohort.csv", dtype="string", keep_default_na=False)["customer_id"]
    _require(cohort.is_unique and len(cohort) == report["customers"] == prediction["recipients"]
             and sha256(("\n".join(cohort) + "\n").encode()).hexdigest() == report["cohort_sha256"],
             "Serving customer population differs.")
    ids = set(cohort)
    profiles, history = read_table(directory, "customer_profiles"), read_table(directory, "customer_history")
    catalogue, articles = read_table(directory, "catalogue"), read_table(directory, "article_aggregates")
    _require(profiles["customer_id"].is_unique and set(profiles["customer_id"]) == ids,
             "Profiles must cover the entire frozen cohort.")
    _require(catalogue["article_id"].is_unique and len(catalogue) == report["catalogue_articles"]
             and not catalogue[["article_id", "product_group_name", "product_type_name", "colour_group_name"]].isna().any().any(),
             "Serving catalogue keys or classifications differ.")
    catalogue_ids = set(catalogue["article_id"])
    _require(articles["article_id"].is_unique and set(articles["article_id"]) == catalogue_ids
             and articles["article_recency_days"].between(1, 180).all()
             and articles["article_recent_rows"].ge(0).all()
             and articles["article_rows_28d"].ge(articles["article_recent_rows"]).all(),
             "Historical article aggregates differ.")
    _require(len(history) == report["history_rows"] and history["customer_id"].isin(ids).all()
             and history["article_id"].isin(catalogue_ids).all()
             and history["t_dat"].ge(cutoff - pd.Timedelta(days=180)).all()
             and history["t_dat"].lt(cutoff).all(),
             "Profile history lies outside the historical serving scope.")
    indexed = profiles.set_index("customer_id")
    counts = history.groupby("customer_id").size().reindex(indexed.index, fill_value=0)
    distinct = history.groupby("customer_id")["article_id"].nunique().reindex(indexed.index, fill_value=0)
    _require(np.array_equal(indexed["customer_history_rows"], counts)
             and np.array_equal(indexed["customer_distinct_articles"], distinct)
             and np.array_equal(indexed["customer_has_history"], counts.gt(0)), "Profile history counts differ.")
    last = history.groupby("customer_id")["t_dat"].max().reindex(indexed.index)
    recency = (cutoff - last).dt.days.astype(float)
    _require(np.allclose(indexed["customer_recency_days"], recency, rtol=0, atol=0, equal_nan=True),
             "Profile recency differs from strictly earlier purchases.")
    for days in (7, 28, 90):
        recent = history.loc[history["t_dat"].ge(cutoff - pd.Timedelta(days=days))].groupby("customer_id").size()
        _require(np.array_equal(indexed[f"customer_rows_{days}d"], recent.reindex(indexed.index, fill_value=0)),
                 f"Profile {days}-day counts differ.")
    for prefix, category in (("group", "product_group_name"), ("type", "product_type_name"), ("colour", "colour_group_name")):
        affinity = read_table(directory, f"affinity_{prefix}")
        share = f"customer_{prefix}_share"
        _require(not affinity.duplicated(["customer_id", category]).any()
                 and affinity["customer_id"].isin(ids).all() and affinity[share].between(0, 1).all(),
                 "Historical affinity grain or shares differ.")
    interactions = read_table(directory, "customer_interactions")
    _require(not interactions.duplicated(["customer_id", "article_id"]).any()
             and interactions["customer_id"].isin(ids).all() and interactions["article_id"].isin(catalogue_ids).all()
             and interactions["customer_article_rows"].gt(0).all()
             and interactions["customer_article_recency_days"].between(1, 180).all()
             and int(interactions["customer_article_rows"].sum()) == len(history),
             "Historical interaction aggregates differ.")
    top, fallback = read_table(directory, "recommendations"), read_table(directory, "fallback")
    k = report["recommendations_per_customer"]
    _require(k == 12 and len(top) == report["recommendation_rows"] == len(ids) * k
             and set(top["customer_id"]) == ids and top["article_id"].isin(catalogue_ids).all()
             and not top.duplicated(["customer_id", "article_id"]).any()
             and top.groupby("customer_id").size().eq(k).all() and np.isfinite(top["score"]).all()
             and top.equals(top.sort_values(["customer_id", "score", "article_id"], ascending=[True, False, True]).reset_index(drop=True))
             and top["rank"].eq(top.groupby("customer_id", sort=False).cumcount() + 1).all(),
             "Serving top-k recommendations differ.")
    baselines = _read(directory / "baseline_recommendations.json")
    _require(set(baselines) == set(prediction["methods"]) - {report["method"]}, "Serving reference methods differ.")
    for lists in baselines.values():
        _require(set(lists) == ids and all(len(values) == len(set(values)) == k
                 and set(values) <= catalogue_ids for values in lists.values()), "Incomplete serving reference lists.")
    _require(len(fallback) == k and fallback["article_id"].is_unique
             and fallback["article_id"].isin(catalogue_ids).all()
             and fallback["rank"].tolist() == list(range(1, k + 1))
             and all(values == fallback["article_id"].tolist() for values in baselines["recent_popularity_7d"].values())
             and np.array_equal(fallback["score"], articles.set_index("article_id").loc[fallback["article_id"], "article_recent_rows"]),
             "Unknown-customer popularity fallback differs.")
    for name in ("comparison", "bootstrap_summary"):
        _require(report["artifacts"][f"{name}.parquet"] == evaluation["artifacts"][f"{name}.parquet"],
                 "Serving analysis differs from the verified V1 evaluation.")
    booster = lgb.Booster(model_file=str(directory / "model.txt"))
    _require(booster.feature_name() == report["feature_columns"] and booster.num_trees() == report["num_trees"],
             "Portable model feature/tree contract differs.")
    probes = read_table(directory, "reload_check")
    scores = booster.predict(probes[report["feature_columns"]].astype("float32"), num_threads=2, validate_features=True)
    _require(np.isfinite(scores).all() and np.allclose(scores, probes["score_after"], rtol=0, atol=1e-12),
             "Portable model reload predictions differ.")
    return report


def _clean(values):
    result = {}
    for key, value in values.items():
        if pd.isna(value):
            value = None
        elif isinstance(value, pd.Timestamp):
            value = value.isoformat()
        elif isinstance(value, np.generic):
            value = value.item()
        result[key] = value
    return result


class ServingBundle:
    """Small runtime adapter for known profiles and explicit unknown-ID fallback."""

    def __init__(self, bundle_directory):
        self.directory = Path(bundle_directory).resolve()
        self.manifest = verify_serving_bundle(self.directory)
        self.version = file_identity(self.directory / "manifest.json")["sha256"]
        self.profiles = read_table(self.directory, "customer_profiles").set_index("customer_id")
        self.catalogue = read_table(self.directory, "catalogue")
        self.top = read_table(self.directory, "recommendations")
        self.fallback = read_table(self.directory, "fallback")
        self.baselines = _read(self.directory / "baseline_recommendations.json")

    def customers(self, *, offset=0, limit=50):
        _require(type(offset) is int and offset >= 0 and type(limit) is int and 1 <= limit <= 100,
                 "Use a nonnegative offset and a page size between 1 and 100.")
        page = self.profiles.sort_index().iloc[offset:offset + limit].reset_index()
        columns = ["customer_id", "customer_has_history", "customer_history_rows", "customer_distinct_articles"]
        return [_clean(row) for row in page[columns].to_dict(orient="records")]

    def profile(self, customer_id):
        if customer_id not in self.profiles.index:
            return None
        with duckdb.connect() as connection:
            history = connection.execute(
                "SELECT * FROM read_parquet(?) WHERE customer_id=? "
                "ORDER BY t_dat DESC,article_id,sales_channel_id,price_source_units LIMIT 20",
                [str(self.directory / "customer_history.parquet"), customer_id],
            ).df()
        return {"customer_id": customer_id, "historical_date": self.manifest["historical_date"],
                "history_days": 180, "summary": _clean(self.profiles.loc[customer_id].to_dict()),
                "recent_purchase_rows": [_clean(row) for row in history.to_dict(orient="records")]}

    def recommendations(self, customer_id, *, method=None):
        method = self.manifest["method"] if method is None else method
        _require(method in {self.manifest["method"], *self.baselines}, "Unknown recommendation method.")
        known = customer_id in self.profiles.index
        if not known:
            rows = self.fallback[["article_id", "rank", "score"]]
            strategy, score_kind = "popularity_fallback_unknown_customer", "historical_purchase_rows_7d"
        elif method == self.manifest["method"]:
            rows = self.top.loc[self.top["customer_id"].eq(customer_id), ["article_id", "rank", "score"]]
            strategy, score_kind = method, "uncalibrated_lightgbm_ranking_score"
        else:
            articles = self.baselines[method][customer_id]
            rows = pd.DataFrame({"article_id": articles, "rank": range(1, len(articles) + 1), "score": None})
            strategy, score_kind = method, "baseline_rank_only"
        rows = rows.merge(self.catalogue, on="article_id", how="left", validate="many_to_one").sort_values("rank")
        return {"customer_id": customer_id,
                "strategy": strategy, "score_kind": score_kind,
                "historical_date": self.manifest["historical_date"], "bundle_version": self.version,
                "model_sha256": self.manifest["model_sha256"] if known and method == self.manifest["method"] else None,
                "items": [_clean(row) for row in rows.to_dict(orient="records")]}
