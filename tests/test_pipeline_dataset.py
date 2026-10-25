"""Synthetic end-to-end assembly, reuse, corruption and interruption checks."""

from hashlib import sha256
import json
from pathlib import Path
import shutil

import duckdb
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import atelier_ai.pipeline_dataset as dataset
import atelier_ai.pipeline_features as features
from atelier_ai.preparation import prepare_sources
from atelier_ai.ranking_data import prepare_ranker_inputs
from atelier_ai.recommendation_features import FEATURE_COLUMNS
from atelier_ai.training_targets import make_training_schedule


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    shutil.copytree(
        Path(features.__file__).parent, tmp_path / "src/atelier_ai",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    raw = tmp_path / "data/raw/hm"
    raw.mkdir(parents=True)
    articles = "article_id,product_group_name,product_type_name,colour_group_name\n"
    articles += "".join(f"{i:010d},A,T{i % 3},Blue\n" for i in range(1, 303))
    purchases = "t_dat,customer_id,article_id,price,sales_channel_id\n"
    purchases += "".join(
        f"2020-07-21,outside,{i:010d},0.1,1\n" for i in range(1, 301)
    )
    purchases += "2020-07-20,u,0000000001,0.1,1\n"
    purchases += "2020-07-22,u,0000000001,0.1,1\n" * 2
    purchases += "2020-07-23,u,0000000301,0.1,1\n"
    purchases += "2020-07-29,u,0000000002,0.1,1\n"
    purchases += "2020-07-30,u,0000000302,0.1,1\n"
    sources = {"articles.csv": articles,
               "customers.csv": "customer_id\nu\ncold\noutside\n",
               "transactions_train.csv": purchases}
    records = []
    for name, text in sources.items():
        path = raw / name
        path.write_text(text)
        records.append({"path": path.relative_to(tmp_path).as_posix(),
                        "size_bytes": path.stat().st_size,
                        "sha256": sha256(path.read_bytes()).hexdigest()})
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "data_manifest.json").write_text(json.dumps({"schema_version": 1, "files": records}))
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "temporal.yaml").write_text(
        'protocol_id: "fixture"\ntraining_weeks: 1\nhistory_days: 180\nhorizon_days: 7\n'
        'validation_start: "2020-07-29"\nv1_evaluation_start: "2020-08-05"\n'
        'confirmation_start: "2020-08-12"\nconfirmation_end_exclusive: "2020-08-19"\n'
    )
    (docs / "frozen.json").write_text(json.dumps({"decision": {"retrieval": {
        "history_limit": 30, "global_limit_7d": 100,
        "preferred_product_types": 3, "per_preferred_type": 50,
        "fallback_days": 28, "candidate_budget": 300,
    }}}))
    run = tmp_path / "data/processed/pipeline_runs/unit"
    prepared = prepare_sources(tmp_path, output_directory=run / "prepared")
    ids = pd.Index(["u", "cold"], dtype="string")
    context = {
        "customer_ids": ids,
        "cohort_sha256": sha256(("\n".join(ids) + "\n").encode()).hexdigest(),
        "schedule": make_training_schedule(validation_cutoff="2020-08-05", n_weeks=2),
        "prepared_directory": prepared,
        "prepared_manifest_sha256": sha256((prepared / "manifest.json").read_bytes()).hexdigest(),
        "feature_columns": FEATURE_COLUMNS, "candidate_budget": 300,
    }
    for module in (features, dataset):
        monkeypatch.setattr(module, "load_training_context", lambda root, plan: context)
    plan = {
        "inputs": {"frozen_selection": str(docs / "frozen.json")},
        "output_root": str(run),
        "stages": [
            {"name": "prepare_sources", "planned_manifest": str(prepared / "manifest.json")},
            {"name": "build_training", "planned_manifest": str(run / "training/manifest.json")},
        ],
    }
    return tmp_path, plan, run


def test_assembly_preserves_full_targets_groups_alignment_and_first_export(campaign, monkeypatch):
    root, plan, run = campaign
    first = features.export_training_cutoff(root, plan, "2020-07-22")
    first_part = first["feature_export"]["directory"] / first["feature_export"]["manifest"]["parts"][0]["file"]
    first_mtime = first_part.stat().st_mtime_ns
    result = dataset.build_training_dataset(root, plan)
    manifest = result["manifest"]
    assert result["reused"] is False
    assert manifest["raw_candidate_rows"] == 1200
    assert manifest["audited_groups"] == 4
    assert manifest["eligible_groups"] == 2
    assert manifest["training_rows"] == 600
    assert manifest["positive_training_rows"] == 2
    assert manifest["source_purchase_rows"] == 5
    assert manifest["target_pairs"] == 4
    assert manifest["uses_v1_targets"] is False
    assert manifest["feature_exports"][0]["fingerprint"] == first["summary"]["fingerprint"]
    assert first_part.stat().st_mtime_ns == first_mtime
    assert not (root / "data/processed/weekly_v1").exists()
    payload = run / "training/dataset"
    with duckdb.connect() as connection:
        targets = connection.execute("SELECT * FROM read_parquet(?)", [str(payload / "targets.parquet")]).df()
        rows = connection.execute("SELECT * FROM read_parquet(?)", [str(payload / "rows.parquet")]).df()
        groups = connection.execute("SELECT * FROM read_parquet(?)", [str(payload / "groups.parquet")]).df()
    assert {"0000000301", "0000000302"}.issubset(set(targets["article_id"]))
    assert rows["customer_id"].eq("u").all()
    inputs = prepare_ranker_inputs(rows, validation_cutoff="2020-08-05")
    assert_frame_equal(inputs["group_index"], groups)
    assert dataset.verify_training_dataset(root, plan) == manifest
    marker = result["manifest_path"]
    before = marker.read_bytes()

    def no_export(*args, **kwargs):
        raise AssertionError("A completed dataset must be verified without re-exporting.")

    monkeypatch.setattr(dataset, "export_training_cutoff", no_export)
    second = dataset.build_training_dataset(root, plan)
    assert second["reused"] is True
    assert marker.read_bytes() == before


@pytest.mark.parametrize("name", ["rows.parquet", "targets.parquet", "group_audit.parquet"])
def test_corrupted_assembled_artifact_is_preserved_and_rejected(campaign, name):
    root, plan, run = campaign
    result = dataset.build_training_dataset(root, plan)
    before = result["manifest_path"].read_bytes()
    path = run / "training/dataset" / name
    content = path.read_bytes()
    damaged = bytes([content[0] ^ 1]) + content[1:]
    path.write_bytes(damaged)
    with pytest.raises(ValueError, match="checksum"):
        dataset.build_training_dataset(root, plan)
    assert path.read_bytes() == damaged
    assert result["manifest_path"].read_bytes() == before


def test_interrupted_week_resumes_using_the_completed_first_export(campaign, monkeypatch):
    root, plan, run = campaign
    original = dataset.export_training_cutoff

    def interrupt(root, plan, cutoff):
        if pd.Timestamp(cutoff) == pd.Timestamp("2020-07-29"):
            raise RuntimeError("Synthetic cutoff interruption")
        return original(root, plan, cutoff)

    monkeypatch.setattr(dataset, "export_training_cutoff", interrupt)
    with pytest.raises(RuntimeError, match="cutoff interruption"):
        dataset.build_training_dataset(root, plan)
    assert not (run / "training/manifest.json").exists()
    assert not (run / "training/dataset").exists()
    part = next((run / "training/features").rglob("part-*.parquet"))
    before = part.stat().st_mtime_ns
    monkeypatch.setattr(dataset, "export_training_cutoff", original)
    result = dataset.build_training_dataset(root, plan)
    assert result["manifest"]["training_rows"] == 600
    assert part.stat().st_mtime_ns == before


def test_publication_can_resume_after_payload_rename(campaign, monkeypatch):
    root, plan, run = campaign
    publish = dataset._publish_marker

    def interrupt(*args):
        raise RuntimeError("Synthetic publication interruption")

    monkeypatch.setattr(dataset, "_publish_marker", interrupt)
    with pytest.raises(RuntimeError, match="publication interruption"):
        dataset.build_training_dataset(root, plan)
    assert (run / "training/dataset/manifest.json").is_file()
    assert not (run / "training/manifest.json").exists()
    monkeypatch.setattr(dataset, "_publish_marker", publish)
    result = dataset.build_training_dataset(root, plan)
    assert result["reused"] is True
    assert result["manifest_path"].is_file()


def test_changed_assembly_dependency_requires_a_new_campaign(campaign):
    root, plan, run = campaign
    dataset.build_training_dataset(root, plan)
    source = root / "src/atelier_ai/ranking_data.py"
    source.write_text(source.read_text() + "\n# changed dependency\n")
    with pytest.raises(ValueError, match="recipe differs"):
        dataset.build_training_dataset(root, plan)
