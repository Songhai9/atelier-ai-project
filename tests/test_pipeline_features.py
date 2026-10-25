import json
import shutil
from hashlib import sha256
from pathlib import Path

import duckdb
import pandas as pd
import pytest

import atelier_ai.pipeline_features as pipeline_features
from atelier_ai.feature_export import export_feature_batches
from atelier_ai.preparation import prepare_sources
from atelier_ai.recommendation_features import FEATURE_COLUMNS
from atelier_ai.training_targets import make_training_schedule


def test_python_source_snapshot_is_fingerprinted(tmp_path):
    source = tmp_path / "pipeline.py"
    source.write_text("# version one\n")
    kwargs = dict(
        customer_ids=pd.Index(["001"], dtype="string"),
        candidate_pools={"001": ["01"]},
        feature_tables={"cutoff": pd.Timestamp("2020-07-22")},
        build_batch=lambda pairs, tables: pairs.assign(x=1),
        output_base=tmp_path / "exports",
        feature_columns=["x"], nullable_features=[],
        sql_schema={"customer_id": "VARCHAR", "article_id": "VARCHAR",
                    "cutoff": "TIMESTAMP", "x": "BIGINT"},
        provenance={}, source_path=source, candidates_per_customer=1,
    )
    first = export_feature_batches(**kwargs)
    assert (first["directory"] / "pipeline_source.py").read_bytes() == source.read_bytes()
    assert first["manifest"]["source_snapshot_sha256"] == sha256(source.read_bytes()).hexdigest()
    assert export_feature_batches(**kwargs)["reused"] is True
    source.write_text("# version two\n")
    second = export_feature_batches(**kwargs)
    assert first["manifest"]["fingerprint"] != second["manifest"]["fingerprint"]


def test_cutoff_export_is_isolated_and_keeps_retrieval_misses(tmp_path, monkeypatch):
    package = Path(pipeline_features.__file__).parent
    shutil.copytree(
        package, tmp_path / "src/atelier_ai",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    raw = tmp_path / "data/raw/hm"
    raw.mkdir(parents=True)
    articles = "article_id,product_group_name,product_type_name,colour_group_name\n"
    articles += "".join(f"{i:010d},A,T{i % 3},Blue\n" for i in range(1, 302))
    transactions = "t_dat,customer_id,article_id,price,sales_channel_id\n"
    transactions += "".join(
        f"2020-07-21,outside,{i:010d},0.1,1\n" for i in range(1, 301)
    )
    transactions += "2020-07-20,u,0000000001,0.1,1\n"
    transactions += "2020-07-22,u,0000000001,0.1,1\n" * 2
    transactions += "2020-07-23,u,0000000301,0.1,1\n"
    files = {"articles.csv": articles,
             "customers.csv": "customer_id\nu\ncold\noutside\n",
             "transactions_train.csv": transactions}
    records = []
    for name, text in files.items():
        path = raw / name
        path.write_text(text)
        records.append({"path": str(path.relative_to(tmp_path)),
                        "size_bytes": path.stat().st_size,
                        "sha256": sha256(path.read_bytes()).hexdigest()})
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "data_manifest.json").write_text(json.dumps({"schema_version": 1, "files": records}))
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "temporal.yaml").write_text(
        'protocol_id: "fixture"\ntraining_weeks: 6\nhistory_days: 180\n'
        'horizon_days: 7\nvalidation_start: "2020-09-02"\n'
        'v1_evaluation_start: "2020-09-09"\n'
        'confirmation_start: "2020-09-16"\n'
        'confirmation_end_exclusive: "2020-09-23"\n'
    )
    (docs / "frozen.json").write_text(json.dumps({
        "decision": {"retrieval": {
            "history_limit": 30, "global_limit_7d": 100,
            "preferred_product_types": 3, "per_preferred_type": 50,
            "fallback_days": 28, "candidate_budget": 300,
        }}
    }))
    run = tmp_path / "data/processed/pipeline_runs/unit"
    prepared = prepare_sources(tmp_path, output_directory=run / "prepared")
    prepared_sha = sha256((prepared / "manifest.json").read_bytes()).hexdigest()
    ids = pd.Index(["u", "cold"], dtype="string")
    schedule = make_training_schedule(validation_cutoff="2020-07-29", n_weeks=1)
    context = {"customer_ids": ids, "schedule": schedule,
               "prepared_directory": prepared, "prepared_manifest_sha256": prepared_sha,
               "feature_columns": FEATURE_COLUMNS, "candidate_budget": 300}
    monkeypatch.setattr(pipeline_features, "load_training_context", lambda root, plan: context)
    plan = {"inputs": {"frozen_selection": str(docs / "frozen.json")},
            "output_root": str(run)}
    first = pipeline_features.export_training_cutoff(tmp_path, plan, "2020-07-22")
    summary = first["summary"]
    assert summary["candidate_rows"] == 600
    assert summary["audited_groups"] == 2 and summary["eligible_groups"] == 1
    assert summary["positive_candidate_pairs"] == 1
    assert summary["source_purchase_rows"] == 3 and summary["target_pairs"] == 2
    assert "0000000301" in set(first["targets"]["article_id"])
    export = first["feature_export"]
    parts = [str(export["directory"] / p["file"]) for p in export["manifest"]["parts"]]
    with duckdb.connect() as connection:
        assert connection.execute(
            "SELECT count(*) FROM read_parquet(?) WHERE article_id='0000000301'", [parts]
        ).fetchone()[0] == 0
    assert not (tmp_path / "data/processed/weekly_v1").exists()
    with pytest.raises(ValueError, match="frozen final-training calendar"):
        pipeline_features.export_training_cutoff(tmp_path, plan, "2020-09-16")
    second = pipeline_features.export_training_cutoff(tmp_path, plan, "2020-07-22")
    assert second["summary"]["reused"] is True
    part = Path(parts[0])
    content = part.read_bytes()
    corrupted = bytes([content[0] ^ 1]) + content[1:]
    part.write_bytes(corrupted)
    with pytest.raises(ValueError, match="checksum"):
        pipeline_features.export_training_cutoff(tmp_path, plan, "2020-07-22")
    assert part.read_bytes() == corrupted
