"""Read and validate the temporal protocol."""

from pathlib import Path

import pandas as pd
import yaml


BOUNDARY_KEYS = (
    "validation_start",
    "v1_evaluation_start",
    "confirmation_start",
    "confirmation_end_exclusive",
)


def load_temporal_protocol(path):
    """Return validated parameters with boundaries converted to timestamps."""
    protocol = yaml.safe_load(Path(path).read_text(encoding="utf-8"))

    if not isinstance(protocol, dict):
        raise ValueError("Temporal configuration must be a mapping.")

    protocol_id = protocol.get("protocol_id")
    if not isinstance(protocol_id, str) or not protocol_id.strip():
        raise ValueError("protocol_id must be a nonempty string.")

    for key in ("training_weeks", "history_days", "horizon_days"):
        value = protocol.get(key)
        if type(value) is not int or value <= 0:
            raise ValueError(f"{key} must be a positive integer.")

    for key in BOUNDARY_KEYS:
        if not isinstance(protocol.get(key), str):
            raise ValueError(f"{key} must be a quoted YYYY-MM-DD string.")

    boundaries = pd.to_datetime(
        [protocol[key] for key in BOUNDARY_KEYS],
        format="%Y-%m-%d",
        errors="raise",
    )

    if boundaries.isna().any():
        raise ValueError("Temporal boundaries must not be missing.")

    horizon = pd.Timedelta(days=protocol["horizon_days"])

    for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        if end - start != horizon:
            raise ValueError(
                "Evaluation windows must be consecutive and match horizon_days."
            )

    for key, boundary in zip(BOUNDARY_KEYS, boundaries, strict=True):
        protocol[key] = boundary

    return protocol


def load_protocol(project_root):
    """Load the project calendar without importing modelling dependencies."""
    return load_temporal_protocol(Path(project_root) / "configs/temporal.yaml")
