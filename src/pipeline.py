from __future__ import annotations
from pathlib import Path
import pandas as pd
from .cohort import build_cohort_from_paths
from .feature_eng import load_and_build
from .train import train


def run(data_dir="data", model_dir="models", alert_rate_cap=0.05):
    data_dir=Path(data_dir)
    cohort=build_cohort_from_paths(data_dir)
    features=load_and_build(data_dir, cohort)
    features=features.merge(cohort[["patient_id","admit_ts","landmark_ts","horizon_ts","first_event_ts","actual_start_ts","surgery_scheduled_ts","discharge_disposition","label"]], on="patient_id", how="left")
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    features.to_parquet(Path(model_dir)/"feature_matrix.parquet", index=False)
    return train(features, model_dir, alert_rate_cap=alert_rate_cap)
