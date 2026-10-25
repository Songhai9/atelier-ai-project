"""Unified execution, read-only status, resume, failures and DVC graph checks."""

from hashlib import sha256
import json
from pathlib import Path
import shlex
import shutil
import sys

import pytest
import yaml

import atelier_ai.pipeline as pipeline
import atelier_ai.pipeline_control as control
from test_pipeline_evaluate import evaluation_campaign


def make_plan(root):
    return {"run_id": "unit", "stages": [
        {"name": name, "depends_on": list(dependencies),
         "planned_manifest": str(root / directory / "manifest.json")}
        for name, dependencies, directory in pipeline.STAGE_SPECS]}


def identities(directory):
    return {path.relative_to(directory).as_posix(): sha256(path.read_bytes()).hexdigest()
            for path in directory.rglob("*") if path.is_file()}


def test_status_is_read_only_and_never_claims_data_verification(tmp_path):
    plan = make_plan(tmp_path / "campaign")
    empty = control.pipeline_status(plan)
    assert empty["recorded_complete"] == 0 and empty["stage_count"] == 7
    assert empty["artifacts_verified"] is False
    assert not (tmp_path / "campaign").exists()
    for stage in plan["stages"][:3]:
        path = Path(stage["planned_manifest"])
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"stage": stage["name"], "status": "complete"}))
    before = identities(tmp_path)
    status = control.pipeline_status(plan)
    assert status["recorded_complete"] == 3
    assert status["mode"] == "manifest_status_only" and status["artifacts_verified"] is False
    assert identities(tmp_path) == before
    Path(plan["stages"][0]["planned_manifest"]).write_text('{"incomplete":')
    Path(plan["stages"][1]["planned_manifest"]).write_text(json.dumps({"stage": "wrong", "status": "complete"}))
    damaged = control.pipeline_status(plan)
    assert [stage["state"] for stage in damaged["stages"][:3]] == ["invalid", "invalid", "recorded_complete"]


def test_full_native_run_partial_resume_verified_reuse_and_source_corruption(evaluation_campaign, monkeypatch, capsys):
    root, plan, run = evaluation_campaign
    plan["stages"] = make_plan(run)["stages"]
    plan["inputs"]["source_manifest"] = str(root / "docs/data_manifest.json")
    # Rebuild from the synthetic CSVs, with the fixture's explicitly frozen tiny cohort.
    shutil.rmtree(run)
    prefix = control.run_pipeline(root, plan, until="prepare-sources")
    assert prefix["completed_through"] == "prepare_sources"
    assert len(prefix["stages"]) == 2 and all(stage["reused"] is False for stage in prefix["stages"])
    assert not (run / "training").exists()
    completed = control.run_pipeline(root, plan)
    assert len(completed["stages"]) == 7
    assert all(stage["status"] == "complete" for stage in completed["stages"])
    assert [stage["reused"] for stage in completed["stages"]] == [True, True, False, False, False, False, False]
    before = identities(run)
    replay = control.run_pipeline(root, plan)
    assert all(stage["reused"] is True for stage in replay["stages"])
    assert identities(run) == before
    verified = control.verify_pipeline(root, plan)
    assert verified["artifacts_verified"] is True and verified["verified_stages"] == 7
    assert verified["confirmation_evaluated"] is False
    assert identities(run) == before
    monkeypatch.setattr(pipeline, "build_plan", lambda project_root, run_id: plan)
    monkeypatch.setattr(sys, "argv", ["pipeline", "verify", "--project-root", str(root), "--run-id", "unit"])
    capsys.readouterr()
    pipeline.main()
    output = capsys.readouterr()
    assert json.loads(output.out)["verified_stages"] == 7
    assert "Checking SHA-256" in output.err
    assert identities(run) == before
    source = root / "data/raw/hm/transactions_train.csv"
    contents = source.read_bytes()
    source.write_bytes(contents.replace(b"0.1", b"0.2", 1))
    with pytest.raises(ValueError, match="SHA-256 differs"):
        control.verify_pipeline(root, plan)
    # Completion records alone are not proof that source files remain valid.
    assert control.pipeline_status(plan)["recorded_complete"] == 7
    assert identities(run) == before


def test_pipeline_stops_at_the_first_failure(tmp_path, monkeypatch):
    plan = make_plan(tmp_path)
    calls = []

    def execute(root, plan, command):
        calls.append(command)
        if command == "build-training":
            raise RuntimeError("Synthetic assembly failure")
        return {"stage": control.STAGE_COMMANDS[command], "status": "complete", "reused": True}

    monkeypatch.setattr(control, "execute_stage", execute)
    with pytest.raises(RuntimeError, match="assembly failure"):
        control.run_pipeline(tmp_path, plan)
    assert calls == ["validate-sources", "prepare-sources", "build-training"]


def test_individual_stage_requires_its_predecessor(tmp_path):
    with pytest.raises(FileNotFoundError):
        control.execute_stage(tmp_path, make_plan(tmp_path), "fit-model")
    assert list(tmp_path.iterdir()) == []


def test_cli_status_reports_json_without_touching_campaign(tmp_path, monkeypatch, capsys):
    plan = make_plan(tmp_path)
    monkeypatch.setattr(pipeline, "build_plan", lambda root, run_id: plan)
    monkeypatch.setattr(sys, "argv", ["pipeline", "status", "--run-id", "unit"])
    pipeline.main()
    report = json.loads(capsys.readouterr().out)
    assert report["mode"] == "manifest_status_only"
    assert report["artifacts_verified"] is False
    assert list(tmp_path.iterdir()) == []


def test_until_is_rejected_for_nonrun_commands(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["pipeline", "plan", "--run-id", "unit", "--until", "fit-model"])
    with pytest.raises(SystemExit) as exc:
        pipeline.main()
    assert exc.value.code == 2
    assert "only by run" in capsys.readouterr().err


def test_dvc_graph_has_persistent_nonoverlapping_outputs_and_a_final_integrity_check():
    root = Path(__file__).resolve().parents[1]
    definition = yaml.safe_load((root / "dvc.yaml").read_text())
    campaign = yaml.safe_load((root / "configs/pipeline.yaml").read_text())
    assert campaign["run_id"] == "local_v1"
    stages = definition["stages"]
    expected_names = [spec[0] for spec in pipeline.STAGE_SPECS]
    assert list(stages) == [*expected_names, "verify_pipeline"]
    owner, outputs = {}, []
    for name, stage in stages.items():
        assert stage["params"] == [{"configs/pipeline.yaml": ["run_id"]}]
        args = shlex.split(stage["cmd"])
        assert args[:3] == [".venv/bin/python", "-m", "atelier_ai.pipeline"]
        assert args[-2:] == ["--run-id", "${run_id}"]
        for record in stage.get("outs", []):
            for path, options in record.items():
                assert options == {"cache": False, "persist": True}
                assert path.startswith("data/processed/pipeline_runs/${run_id}/")
                assert path not in owner
                owner[path] = name
                outputs.append(Path(path))
    for left in outputs:
        for right in outputs:
            if left != right:
                assert not left.is_relative_to(right)
    directories = {name: directory for name, _, directory in pipeline.STAGE_SPECS}
    for name, dependencies, _ in pipeline.STAGE_SPECS:
        for dependency in dependencies:
            assert f"data/processed/pipeline_runs/${{run_id}}/{directories[dependency]}" in stages[name]["deps"]
    final = stages["verify_pipeline"]
    assert final["always_changed"] is True and not final.get("outs")
    assert set(owner) <= set(final["deps"])
    assert "**/__pycache__" in (root / ".dvcignore").read_text()
