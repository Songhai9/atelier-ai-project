"""Check the reconciled calendar without reading transaction data."""

from pathlib import Path

import pandas as pd
from atelier_ai.temporal_protocol import BOUNDARY_KEYS, load_temporal_protocol

from atelier_ai.training_targets import make_training_schedule


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "temporal.yaml"

config = load_temporal_protocol(CONFIG_PATH)
boundaries = [config[key] for key in BOUNDARY_KEYS]

training_schedule = make_training_schedule(
    validation_cutoff=boundaries[0],
    n_weeks=config["training_weeks"],
    history_days=config["history_days"],
    horizon_days=config["horizon_days"],
)

evaluation_schedule = pd.DataFrame(
    {
        "period": ["validation", "v1_evaluation", "confirmation"],
        "start_inclusive": boundaries[:-1],
        "end_exclusive": boundaries[1:],
    }
)

print(f"Protocol: {config['protocol_id']}")

print("\nHistorical training windows:")
print(
    training_schedule[["cutoff", "history_start", "target_end"]].to_string(index=False)
)

print("\nEvaluation windows:")
print(evaluation_schedule.to_string(index=False))

print("\nCalendar check passed.")
