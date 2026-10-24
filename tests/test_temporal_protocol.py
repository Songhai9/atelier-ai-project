"""Configuration contract and invalid temporal-boundary checks."""

from pathlib import Path

import pytest
import yaml

from atelier_ai.temporal_protocol import BOUNDARY_KEYS, load_temporal_protocol


CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "temporal.yaml"


def test_repository_protocol_matches_reconciled_calendar():
    protocol = load_temporal_protocol(CONFIG_PATH)

    assert protocol["training_weeks"] == 6
    assert protocol["history_days"] == 180
    assert protocol["horizon_days"] == 7

    assert [protocol[key].strftime("%Y-%m-%d") for key in BOUNDARY_KEYS] == [
        "2020-09-02",
        "2020-09-09",
        "2020-09-16",
        "2020-09-23",
    ]


@pytest.mark.parametrize(
    "field,value",
    [
        ("history_days", True),
        ("horizon_days", 0),
        ("v1_evaluation_start", "2020-09-10"),
        ("confirmation_start", "2020-09-15"),
        ("validation_start", "2020-02-30"),
    ],
)
def test_invalid_protocol_is_rejected(tmp_path, field, value):
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    config[field] = value

    candidate_path = tmp_path / "temporal.yaml"
    candidate_path.write_text(
        yaml.safe_dump(config),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_temporal_protocol(candidate_path)


def test_project_calendar_loader_has_the_same_validated_boundaries():
    from atelier_ai.temporal_protocol import load_protocol
    root = Path(__file__).resolve().parents[1]
    assert load_protocol(root) == load_temporal_protocol(root / "configs/temporal.yaml")
