# Pre-operative deterioration prediction at T0 + 6h

Production-oriented reference implementation for predicting severe pre-operative deterioration within 72h of admission among patients still waiting at the six-hour landmark.

## 1. Outcome definition

The supplied `events.csv` is the source of truth. A deterioration is any of:

- `emergency_icu_transfer`
- `vasopressor_initiation`
- `rapid_response_activation`
- `intubation`
- `cardiac_arrest`
- `death`

The event dictionary says deterioration patients have one or more timestamped rows, while uneventful waits have one `normal` row with no timestamp. The cascade rows describe one episode. Therefore the label is based on the **first** deterioration event.

### Landmark cohort

For each encounter, define `landmark_ts = admit_ts + 6h`.

1. Exclude patients whose first deterioration is at or before the landmark.
2. Include patients with a recorded `actual_start_ts` after the landmark.
3. If `actual_start_ts` is missing but a deterioration occurs after the landmark, include them: the event itself demonstrates that the patient remained in the pre-operative wait until deterioration. These are the positive cases in this extract.
4. Exclude encounters with neither a post-landmark actual start nor a post-landmark deterioration. The extract does not contain a discharge timestamp, so these encounters cannot prove that the patient was still waiting at six hours.
5. `label=1` when the first deterioration occurs strictly after 6h and no later than 72h after admission.

This conservative rule produced **79,962 evaluable encounters and 4,897 positives (6.12%)** in the supplied CSV event/surgery files during the audit. The supplied scenario's approximate one-in-eighteen prevalence is therefore directionally consistent.

`actual_start_ts`, `discharge_disposition`, `first_event_ts`, and any post-landmark fields are never model features.

## 2. Temporal leakage policy

Features are extracted using only timestamps `<= admit_ts + 6h`.

- Vitals/labs: last, mean, min, max, SD, range, first-to-last delta and slope.
- Monitor artifacts outside conservative physiologic data-quality bounds are set missing, not clipped.
- Medications: counts by class/drug and time since last administration, restricted to `<=6h`.
- Prior history: counts, prior deterioration count/recency and latest outpatient labs, restricted to records before admission. The source dictionary says prior history is already limited to the three months before admission.
- Admission/booked-case information: demographics, ASA, urgency, comorbidities, practice, procedure and surgeon-volume band.
- Scheduling timestamp is intentionally excluded from features because the dictionary says it represents the **most recent** scheduling decision, not necessarily the initial decision available at T0.

## 3. Evaluation

Primary split is chronological:

- Train: 2022-01-01 through 2023-12-31
- Validation: 2024-01-01 through 2024-06-30
- Locked test: 2024-07-01 through 2024-12-31

The training period also receives a six-fold practice-held-out robustness check. This is secondary to the chronological test because practice-held-out CV alone can still mix calendar time.

Report:

- ROC-AUC
- PR-AUC (primary discrimination metric because prevalence is low)
- Brier score
- calibration/threshold table
- sensitivity/recall, PPV, specificity, false-alarm rate
- alert rate
- decision-curve net benefit

## 4. Operating threshold

The repository does **not** pretend to know the hospital's true cost of a false alarm. The default operating rule is an explicit workload constraint: maximize validation recall subject to an alert-rate cap of 5%.

That 5% is an operational placeholder, not a clinical truth. Before production use, replace it with a capacity-derived number: e.g. the maximum additional same-day reviews or theatre-priority actions the ward can absorb. The final threshold is then locked and evaluated once on the held-out 2024 H2 test set.

A useful governance rule is: do not choose the threshold on the test set, and do not move it merely because the test-set PPV is disappointing.

## 5. Run

Place the five supplied files under `data/` with these names:

```text
data/surgeries.csv
data/observations.parquet
data/medications.csv
data/events.csv
data/prior_history.csv
```

Then:

```bash
pip install -r requirements.txt
python -m src.run_train --data-dir data --model-dir models --alert-rate-cap 0.05
streamlit run app/streamlit_app.py
pytest -q
```

## 6. Clinical interpretation and limitations

This is a **risk-ranking / decision-support model**, not an autonomous diagnosis or theatre-allocation engine. A high score should trigger human review, not an automatic cancellation or reprioritisation.

Important limitations from the supplied extract:

- There is no timestamped discharge field, so no-event/no-start encounters are conservatively censored out rather than assumed eligible.
- Recorded events are the outcome source and may reflect documentation practice as well as physiology.
- The event labels are severe operational events, not a complete clinical deterioration construct.
- `observations.parquet` explicitly contains monitor artifacts; data-quality handling is therefore part of the model, not an afterthought.
- Practice and time drift must be monitored after deployment.
- A threshold should be re-approved if theatre capacity, staffing, event ascertainment or case mix changes.
