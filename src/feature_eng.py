from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


VITALS = [
    "heart_rate",
    "sbp",
    "dbp",
    "map",
    "spo2",
    "resp_rate",
    "temp_c",
]

LABS = [
    "creatinine",
    "lactate",
    "hemoglobin",
    "wbc",
    "platelets",
    "glucose",
    "potassium",
    "crp",
    "troponin",
]

ALL_OBS = VITALS + LABS


# Conservative artifact filters.
# Values outside these ranges are treated as missing.
PLAUSIBLE = {
    "heart_rate": (20, 250),
    "sbp": (40, 250),
    "dbp": (20, 150),
    "map": (30, 180),
    "spo2": (50, 100),
    "resp_rate": (5, 60),
    "temp_c": (30, 43),
    "creatinine": (0.1, 20),
    "lactate": (0.1, 20),
    "hemoglobin": (3, 25),
    "wbc": (0.1, 100),
    "platelets": (5, 1500),
    "glucose": (20, 1000),
    "potassium": (1.5, 10),
    "crp": (0, 1000),
    "troponin": (0, 100),
}


BASE_NUMERIC = [
    "age",
    "weight_kg",
    "height_cm",
    "bmi",
    "asa_class",
]

BASE_CATEGORICAL = [
    "practice_id",
    "procedure_cpt",
    "urgency",
    "sex",
    "smoking_status",
    "surgeon_volume_indicator",
]

COMORBIDITIES = [
    "anticoagulated",
    "diabetes",
    "heart_failure",
    "ckd",
    "copd",
    "cad",
    "active_cancer",
    "cirrhosis",
    "prior_stroke",
    "immunosuppressed",
]


def _safe_name(s: str) -> str:
    return (
        str(s)
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
    )


def observation_features(
    obs: pd.DataFrame,
    cohort: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build observation features using only measurements available
    between T0 and T0 + 6 hours.

    This implementation is vectorized and avoids the expensive:

        for patient:
            for variable:
                g[g["variable"] == var]

    pattern used by the original implementation.
    """

    # ---------------------------------------------------------
    # 1. Prepare observation data
    # ---------------------------------------------------------

    o = obs[
        ["patient_id", "ts", "variable", "value"]
    ].copy()

    o["ts"] = pd.to_datetime(
        o["ts"],
        utc=True,
        errors="coerce",
    )

    o["value"] = pd.to_numeric(
        o["value"],
        errors="coerce",
    )

    # Keep only variables used by the model.
    o = o[o["variable"].isin(ALL_OBS)]

    # ---------------------------------------------------------
    # 2. Merge admission and 6-hour landmark
    # ---------------------------------------------------------

    c = cohort[
        ["patient_id", "admit_ts", "landmark_ts"]
    ].copy()

    c["admit_ts"] = pd.to_datetime(
        c["admit_ts"],
        utc=True,
        errors="coerce",
    )

    c["landmark_ts"] = pd.to_datetime(
        c["landmark_ts"],
        utc=True,
        errors="coerce",
    )

    o = o.merge(
        c,
        on="patient_id",
        how="inner",
    )

    # ---------------------------------------------------------
    # 3. Temporal filtering
    #
    # Only information available between:
    #
    # T0 <= timestamp <= T0 + 6h
    # ---------------------------------------------------------

    o = o[
        (o["ts"] >= o["admit_ts"])
        & (o["ts"] <= o["landmark_ts"])
    ].copy()

    if o.empty:
        return pd.DataFrame(columns=["patient_id"])

    # ---------------------------------------------------------
    # 4. Artifact filtering
    #
    # Apply plausible ranges using vectorized mapping.
    # ---------------------------------------------------------

    lower = o["variable"].map(
        {k: v[0] for k, v in PLAUSIBLE.items()}
    )

    upper = o["variable"].map(
        {k: v[1] for k, v in PLAUSIBLE.items()}
    )

    invalid = (
        o["value"].notna()
        & (
            (o["value"] < lower)
            | (o["value"] > upper)
        )
    )

    o.loc[invalid, "value"] = np.nan

    # ---------------------------------------------------------
    # 5. Create time variable for slope
    #
    # Hours since admission.
    # ---------------------------------------------------------

    o["hours_from_admit"] = (
        o["ts"] - o["admit_ts"]
    ).dt.total_seconds() / 3600.0

    # ---------------------------------------------------------
    # 6. Sort once
    #
    # This allows efficient first/last calculations.
    # ---------------------------------------------------------

    o = o.sort_values(
        ["patient_id", "variable", "ts"]
    )

    # ---------------------------------------------------------
    # 7. Basic statistics
    #
    # One groupby instead of patient × variable loops.
    # ---------------------------------------------------------

    grouped = (
        o.groupby(
            ["patient_id", "variable"],
            sort=False,
            observed=True,
        )["value"]
        .agg(
            n="count",
            last="last",
            mean="mean",
            min="min",
            max="max",
            std=lambda x: x.std(ddof=0),
            first="first",
        )
        .reset_index()
    )

    grouped["delta"] = (
        grouped["last"] - grouped["first"]
    )

    grouped["range"] = (
        grouped["max"] - grouped["min"]
    )

    # ---------------------------------------------------------
    # 8. Vectorized linear-regression slope
    #
    # slope =
    #
    # (n * sum(xy) - sum(x)*sum(y))
    # --------------------------------
    # (n * sum(x²) - sum(x)²)
    # ---------------------------------------------------------

    valid = o[
        o["value"].notna()
        & o["hours_from_admit"].notna()
    ].copy()

    if not valid.empty:

        valid["xy"] = (
            valid["hours_from_admit"]
            * valid["value"]
        )

        valid["x2"] = (
            valid["hours_from_admit"]
            ** 2
        )

        slope_stats = (
            valid.groupby(
                ["patient_id", "variable"],
                sort=False,
                observed=True,
            )
            .agg(
                n_slope=("value", "count"),
                sum_x=("hours_from_admit", "sum"),
                sum_y=("value", "sum"),
                sum_xy=("xy", "sum"),
                sum_x2=("x2", "sum"),
            )
            .reset_index()
        )

        numerator = (
            slope_stats["n_slope"]
            * slope_stats["sum_xy"]
            - slope_stats["sum_x"]
            * slope_stats["sum_y"]
        )

        denominator = (
            slope_stats["n_slope"]
            * slope_stats["sum_x2"]
            - slope_stats["sum_x"]
            ** 2
        )

        slope_stats["slope"] = np.where(
            denominator != 0,
            numerator / denominator,
            0.0,
        )

        grouped = grouped.merge(
            slope_stats[
                [
                    "patient_id",
                    "variable",
                    "slope",
                ]
            ],
            on=["patient_id", "variable"],
            how="left",
        )

    else:
        grouped["slope"] = np.nan

    # ---------------------------------------------------------
    # 9. Pivot statistics from long → wide
    # ---------------------------------------------------------

    stats = [
        "n",
        "last",
        "mean",
        "min",
        "max",
        "std",
        "delta",
        "slope",
        "range",
    ]

    wide_parts = []

    for stat in stats:

        x = grouped.pivot(
            index="patient_id",
            columns="variable",
            values=stat,
        )

        x.columns = [
            f"obs_{var}_{stat}"
            for var in x.columns
        ]

        wide_parts.append(x)

    result = pd.concat(
        wide_parts,
        axis=1,
    ).reset_index()

    # ---------------------------------------------------------
    # 10. Operational completeness features
    # ---------------------------------------------------------

    operational = (
        o.groupby("patient_id")
        .agg(
            obs_rows=("value", "size"),
            obs_unique_ts=("ts", "nunique"),
        )
    )

    result = result.merge(
        operational,
        left_on="patient_id",
        right_index=True,
        how="left",
    )

    return result.reset_index(drop=True)


def medication_features(
    meds: pd.DataFrame,
    cohort: pd.DataFrame,
) -> pd.DataFrame:

    m = meds[
        [
            "patient_id",
            "ts",
            "drug_class",
            "drug_name",
        ]
    ].copy()

    m["ts"] = pd.to_datetime(
        m["ts"],
        utc=True,
        errors="coerce",
    )

    c = cohort[
        [
            "patient_id",
            "admit_ts",
            "landmark_ts",
        ]
    ].copy()

    c["admit_ts"] = pd.to_datetime(
        c["admit_ts"],
        utc=True,
        errors="coerce",
    )

    c["landmark_ts"] = pd.to_datetime(
        c["landmark_ts"],
        utc=True,
        errors="coerce",
    )

    m = m.merge(
        c,
        on="patient_id",
        how="inner",
    )

    # Only medication administrations available by T0 + 6h.
    m = m[
        (m["ts"] >= m["admit_ts"])
        & (m["ts"] <= m["landmark_ts"])
    ].copy()

    if m.empty:
        return pd.DataFrame(columns=["patient_id"])

    # Total medication count.
    out = (
        m.groupby("patient_id")
        .size()
        .rename("med_total_count")
        .to_frame()
    )

    # Drug class counts.
    class_counts = pd.crosstab(
        m["patient_id"],
        m["drug_class"],
    )

    class_counts.columns = [
        f"medclass_{_safe_name(c)}_count"
        for c in class_counts.columns
    ]

    out = out.join(
        class_counts,
        how="outer",
    )

    # Drug name counts.
    drug_counts = pd.crosstab(
        m["patient_id"],
        m["drug_name"],
    )

    drug_counts.columns = [
        f"drug_{_safe_name(c)}_count"
        for c in drug_counts.columns
    ]

    out = out.join(
        drug_counts,
        how="outer",
    )

    # Time since most recent medication.
    recent = (
        m.groupby("patient_id")["ts"]
        .max()
    )

    landmark = (
        c.drop_duplicates("patient_id")
        .set_index("patient_id")["landmark_ts"]
    )

    out["med_hours_since_last"] = (
        landmark - recent
    ).dt.total_seconds() / 3600.0

    return out.reset_index()


def prior_history_features(
    prior: pd.DataFrame,
    cohort: pd.DataFrame,
) -> pd.DataFrame:

    p = prior[
        [
            "patient_id",
            "ts",
            "record_type",
            "detail",
            "value",
        ]
    ].copy()

    p["ts"] = pd.to_datetime(
        p["ts"],
        utc=True,
        errors="coerce",
    )

    c = cohort[
        ["patient_id", "admit_ts"]
    ].copy()

    c["admit_ts"] = pd.to_datetime(
        c["admit_ts"],
        utc=True,
        errors="coerce",
    )

    p = p.merge(
        c,
        on="patient_id",
        how="inner",
    )

    # Prior history must be strictly before admission.
    p = p[
        p["ts"] < p["admit_ts"]
    ].copy()

    if p.empty:
        return pd.DataFrame(columns=["patient_id"])

    out = (
        p.groupby("patient_id")
        .size()
        .rename("prior_total_records")
        .to_frame()
    )

    # Record-type counts.
    type_counts = pd.crosstab(
        p["patient_id"],
        p["record_type"],
    )

    type_counts.columns = [
        f"prior_type_{_safe_name(c)}_count"
        for c in type_counts.columns
    ]

    out = out.join(
        type_counts,
        how="outer",
    )

    # Detail counts.
    detail_counts = pd.crosstab(
        p["patient_id"],
        p["detail"],
    )

    detail_counts.columns = [
        f"prior_detail_{_safe_name(c)}_count"
        for c in detail_counts.columns
    ]

    out = out.join(
        detail_counts,
        how="outer",
    )

    # Latest outpatient lab values.
    labs = p[
        p["record_type"].eq("outpatient_lab")
        & p["value"].notna()
    ].copy()

    if not labs.empty:

        labs = labs.sort_values(
            ["patient_id", "detail", "ts"]
        )

        latest = (
            labs.groupby(
                ["patient_id", "detail"],
                sort=False,
            )
            .tail(1)
        )

        latest_wide = latest.pivot(
            index="patient_id",
            columns="detail",
            values="value",
        )

        latest_wide.columns = [
            f"prior_lab_{_safe_name(c)}_last"
            for c in latest_wide.columns
        ]

        out = out.join(
            latest_wide,
            how="outer",
        )

    # Recency of prior deterioration.
    deterioration = (
        p[
            p["record_type"]
            .eq("prior_deterioration")
        ]
        .groupby("patient_id")["ts"]
        .max()
    )

    admission = (
        c.drop_duplicates("patient_id")
        .set_index("patient_id")["admit_ts"]
    )

    out["prior_deterioration_days_since"] = (
        admission - deterioration
    ).dt.total_seconds() / 86400.0

    return out.reset_index()


def build_features(
    cohort: pd.DataFrame,
    observations: pd.DataFrame,
    medications: pd.DataFrame,
    prior_history: pd.DataFrame,
) -> pd.DataFrame:

    c = cohort.copy()

    c["admit_ts"] = pd.to_datetime(
        c["admit_ts"],
        utc=True,
        errors="coerce",
    )

    c["landmark_ts"] = pd.to_datetime(
        c["landmark_ts"],
        utc=True,
        errors="coerce",
    )

    # ---------------------------------------------------------
    # Static admission features.
    #
    # Do NOT include:
    # actual_start_ts
    # discharge_disposition
    # first_event_ts
    # horizon_ts
    # label
    # ---------------------------------------------------------

    keep = [
        "patient_id",
        "practice_id",
        "procedure_cpt",
        "urgency",
        "age",
        "sex",
        "weight_kg",
        "height_cm",
        "bmi",
        "asa_class",
        "smoking_status",
        "surgeon_volume_indicator",
    ] + COMORBIDITIES

    base = c[keep].copy()

    # Calendar/time features known at admission.
    base["admission_hour_utc"] = (
        c["admit_ts"].dt.hour
    )

    base["admission_dow_utc"] = (
        c["admit_ts"].dt.dayofweek
    )

    base["admission_month"] = (
        c["admit_ts"].dt.month
    )

    # ---------------------------------------------------------
    # Dynamic features.
    # ---------------------------------------------------------

    obs_features = observation_features(
        observations,
        cohort,
    )

    med_features = medication_features(
        medications,
        cohort,
    )

    prior_features = prior_history_features(
        prior_history,
        cohort,
    )

    base = base.merge(
        obs_features,
        on="patient_id",
        how="left",
    )

    base = base.merge(
        med_features,
        on="patient_id",
        how="left",
    )

    base = base.merge(
        prior_features,
        on="patient_id",
        how="left",
    )

    return base


def load_and_build(
    data_dir: str | Path,
    cohort: pd.DataFrame,
) -> pd.DataFrame:

    data_dir = Path(data_dir)

    print("Loading observations...")
    obs = pd.read_parquet(
        data_dir / "observations.parquet"
    )

    print(
        f"Observations loaded: {len(obs):,} rows"
    )

    print("Loading medications...")
    meds = pd.read_csv(
        data_dir / "medications.csv"
    )

    print(
        f"Medication rows loaded: {len(meds):,}"
    )

    print("Loading prior history...")
    prior = pd.read_csv(
        data_dir / "prior_history.csv"
    )

    print(
        f"Prior-history rows loaded: {len(prior):,}"
    )

    print("Building features...")

    features = build_features(
        cohort,
        obs,
        meds,
        prior,
    )

    print(
        f"Feature rows generated: {len(features):,}"
    )

    print(
        f"Feature columns generated: {len(features.columns):,}"
    )

    return features