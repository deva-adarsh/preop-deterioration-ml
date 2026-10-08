from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd

DETERIORATION_EVENTS = {
    "emergency_icu_transfer",
    "vasopressor_initiation",
    "rapid_response_activation",
    "intubation",
    "cardiac_arrest",
    "death",
}

@dataclass(frozen=True)
class CohortConfig:
    horizon_hours: int = 72
    landmark_hours: int = 6


def derive_cohort(surgeries: pd.DataFrame, events: pd.DataFrame,
                  config: CohortConfig = CohortConfig()) -> pd.DataFrame:
    """Create the landmark cohort and outcome using only event timing.

    Cohort rule:
    - landmark = admit_ts + 6h
    - exclude patients whose first deterioration is <= landmark
    - eligible if actual_start_ts > landmark OR a deterioration is recorded after landmark
      (the event itself proves the patient remained in the pre-operative wait)
    - exclude no-event encounters with missing actual_start_ts because the extract has no
      discharge timestamp and therefore cannot prove that they were still waiting at 6h
    - positive = first deterioration in (landmark, admit_ts + 72h]

    `surgery_scheduled_ts` is deliberately NOT used as a feature or censor because the data
    dictionary says it is the most recent scheduling decision, which can reflect downstream
    information unavailable at the six-hour decision point.
    """
    s = surgeries.copy()
    e = events.copy()
    for col in ["admit_ts", "actual_start_ts", "surgery_scheduled_ts"]:
        if col not in s.columns:
            s[col] = pd.NaT
        s[col] = pd.to_datetime(s[col], utc=True, errors="coerce")
    e["ts"] = pd.to_datetime(e["ts"], utc=True, errors="coerce")
    e = e[e["event_type"].isin(DETERIORATION_EVENTS)].copy()

    first = e.groupby("patient_id", as_index=True)["ts"].min().rename("first_event_ts")
    s = s.join(first, on="patient_id")
    s["landmark_ts"] = s["admit_ts"] + pd.Timedelta(hours=config.landmark_hours)
    s["horizon_ts"] = s["admit_ts"] + pd.Timedelta(hours=config.horizon_hours)

    event_before_landmark = s["first_event_ts"].notna() & (s["first_event_ts"] <= s["landmark_ts"])
    proves_waiting = s["first_event_ts"].notna() & (s["first_event_ts"] > s["landmark_ts"])
    actual_waiting = s["actual_start_ts"].notna() & (s["actual_start_ts"] > s["landmark_ts"])
    no_start_no_event = s["actual_start_ts"].isna() & s["first_event_ts"].isna()

    eligible = (actual_waiting | proves_waiting) & ~event_before_landmark & ~no_start_no_event
    s = s.loc[eligible].copy()
    s["label"] = (
        s["first_event_ts"].notna()
        & (s["first_event_ts"] > s["landmark_ts"])
        & (s["first_event_ts"] <= s["horizon_ts"])
    ).astype("int8")

    # Safety assertions: all positive labels are genuinely post-landmark and <=72h.
    pos = s["label"].eq(1)
    assert (s.loc[pos, "first_event_ts"] > s.loc[pos, "landmark_ts"]).all()
    assert (s.loc[pos, "first_event_ts"] <= s.loc[pos, "horizon_ts"]).all()
    assert not (s["first_event_ts"].notna() & (s["first_event_ts"] <= s["landmark_ts"])).any()
    return s.reset_index(drop=True)


def build_cohort_from_paths(data_dir: str | Path) -> pd.DataFrame:
    data_dir = Path(data_dir)
    surgeries = pd.read_csv(data_dir / "surgeries.csv")
    events = pd.read_csv(data_dir / "events.csv")
    return derive_cohort(surgeries, events)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out", default="artifacts/cohort.csv")
    args = parser.parse_args()
    cohort = build_cohort_from_paths(args.data_dir)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(args.out, index=False)
    print(cohort.groupby("label").size())
