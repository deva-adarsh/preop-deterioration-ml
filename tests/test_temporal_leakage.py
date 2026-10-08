import pandas as pd
from src.cohort import derive_cohort


def test_event_at_or_before_landmark_excluded():
    surgeries=pd.DataFrame({
        "patient_id":["P1"],"surgery_id":["S1"],"admit_ts":["2024-01-01T00:00:00Z"],
        "actual_start_ts":["2024-01-02T00:00:00Z"]})
    events=pd.DataFrame({"patient_id":["P1"],"surgery_id":["S1"],"ts":["2024-01-01T06:00:00Z"],"event_type":["death"]})
    out=derive_cohort(surgeries,events)
    assert out.empty


def test_post_landmark_event_is_label():
    surgeries=pd.DataFrame({
        "patient_id":["P1"],"surgery_id":["S1"],"admit_ts":["2024-01-01T00:00:00Z"],
        "actual_start_ts":[None],"surgery_scheduled_ts":["2024-01-02T00:00:00Z"]})
    events=pd.DataFrame({"patient_id":["P1"],"surgery_id":["S1"],"ts":["2024-01-01T12:00:00Z"],"event_type":["rapid_response_activation"]})
    out=derive_cohort(surgeries,events)
    assert len(out)==1 and int(out.iloc[0].label)==1


def test_no_start_no_event_is_censored_out():
    surgeries=pd.DataFrame({
        "patient_id":["P1"],"surgery_id":["S1"],"admit_ts":["2024-01-01T00:00:00Z"],
        "actual_start_ts":[None],"surgery_scheduled_ts":["2024-01-02T00:00:00Z"]})
    events=pd.DataFrame({"patient_id":["P1"],"surgery_id":["S1"],"ts":[None],"event_type":["normal"]})
    out=derive_cohort(surgeries,events)
    assert out.empty
