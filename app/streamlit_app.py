from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    brier_score_loss,
)

from views.executive_summary import render as render_summary
from views.patient_risk_explorer import render as render_patients
from views.threshold_lab import render as render_threshold


# ============================================================
# PATHS
# ============================================================

# Current file:
#     project/app/streamlit_app.py
#
# parents[0] -> project/app
# parents[1] -> project
#
# Therefore this works both locally on Windows and on Render/Linux.

ROOT_DIR = Path(__file__).resolve().parents[1]

MODEL_DIR = ROOT_DIR / "models"

PIPELINE_PATH = MODEL_DIR / "pipeline.joblib"
METADATA_PATH = MODEL_DIR / "metadata.json"
FEATURES_PATH = MODEL_DIR / "features.parquet"
THRESHOLD_PATH = MODEL_DIR / "threshold_selection.csv"


# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="Pre-operative deterioration risk",
    layout="wide",
)

st.title(
    "Pre-operative Deterioration Risk"
)

st.caption(
    "T0 + 6h landmark • deterioration within T0 + 72h • "
    "human decision support"
)


# ============================================================
# ARTIFACT VALIDATION
# ============================================================

required_files = [
    METADATA_PATH,
    PIPELINE_PATH,
    FEATURES_PATH,
]

missing_files = [
    str(path)
    for path in required_files
    if not path.exists()
]

if missing_files:

    st.error(
        "Required model artifacts are missing:"
    )

    for file in missing_files:
        st.write(
            f"- `{file}`"
        )

    st.stop()


# ============================================================
# LOAD ARTIFACTS
# ============================================================

@st.cache_data
def load_metadata(path: Path):

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


@st.cache_resource
def load_model(path: Path):

    return joblib.load(path)


@st.cache_data
def load_features(path: Path):

    return pd.read_parquet(
        path
    )


@st.cache_data
def load_threshold_table(path: Path):

    if path.exists():

        return pd.read_csv(
            path
        )

    return pd.DataFrame()


meta = load_metadata(
    METADATA_PATH
)

model = load_model(
    PIPELINE_PATH
)

df = load_features(
    FEATURES_PATH
)

threshold_df = load_threshold_table(
    THRESHOLD_PATH
)


# ============================================================
# VALIDATE FEATURE COLUMNS
# ============================================================

feature_columns = meta.get(
    "feature_columns",
    []
)

if not feature_columns:

    st.error(
        "No feature_columns were found "
        "in metadata.json."
    )

    st.stop()


missing_features = [
    col
    for col in feature_columns
    if col not in df.columns
]

if missing_features:

    st.error(
        "The feature dataset is missing "
        "columns required by the model:"
    )

    st.write(
        missing_features
    )

    st.stop()


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

X = df[
    feature_columns
]

probabilities = model.predict_proba(
    X
)[:, 1]

df = df.copy()

df["risk_72h"] = probabilities


# ============================================================
# CALCULATE TEST METRICS
# ============================================================

if "admit_ts" in df.columns:

    df["admit_ts"] = pd.to_datetime(
        df["admit_ts"],
        utc=True,
        errors="coerce",
    )


TEST_START = pd.Timestamp(
    "2024-07-01",
    tz="UTC",
)


if "admit_ts" in df.columns:

    test_mask = (
        df["admit_ts"]
        >= TEST_START
    )

    test_df = df.loc[
        test_mask
    ].copy()

else:

    test_df = pd.DataFrame()


if (
    not test_df.empty
    and "label" in test_df.columns
    and test_df["label"].nunique() == 2
):

    y_test = test_df[
        "label"
    ].astype(int)

    p_test = test_df[
        "risk_72h"
    ].astype(float)

    test_metrics = {

        "pr_auc": float(
            average_precision_score(
                y_test,
                p_test,
            )
        ),

        "roc_auc": float(
            roc_auc_score(
                y_test,
                p_test,
            )
        ),

        "brier": float(
            brier_score_loss(
                y_test,
                p_test,
            )
        ),

        "prevalence": float(
            y_test.mean()
        ),
    }

else:

    test_metrics = {

        "pr_auc": None,

        "roc_auc": None,

        "brier": None,

        "prevalence": None,
    }


# ============================================================
# DETERMINE RECOMMENDED THRESHOLD
# ============================================================

# Fallback value.
# The actual value is preferably derived from
# threshold_selection.csv below.

recommended_threshold = float(
    meta.get(
        "recommended_threshold",
        0.121,
    )
)


# ============================================================
# DERIVE RECOMMENDED THRESHOLD
# FROM VALIDATION TABLE
# ============================================================

if not threshold_df.empty:

    required_threshold_columns = {

        "threshold",

        "alert_rate",

        "recall_sensitivity",

        "precision_ppv",
    }

    if required_threshold_columns.issubset(
        threshold_df.columns
    ):

        MAX_ALERT_RATE = 0.05

        feasible = threshold_df[
            threshold_df[
                "alert_rate"
            ] <= MAX_ALERT_RATE
        ].copy()

        if not feasible.empty:

            feasible = feasible.sort_values(

                by=[

                    "recall_sensitivity",

                    "precision_ppv",

                    "threshold",
                ],

                ascending=[

                    False,

                    False,

                    False,
                ],
            )

            recommended_threshold = float(
                feasible.iloc[0][
                    "threshold"
                ]
            )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "Model Operating point"
    )


    # ========================================================
    # RESET BUTTON
    # ========================================================

    if st.button(
        "Reset to recommended threshold",
        use_container_width=True,
    ):

        st.session_state[
            "alert_threshold"
        ] = recommended_threshold

        st.rerun()


    # ========================================================
    # INITIALIZE THRESHOLD
    # ========================================================

    if (
        "alert_threshold"
        not in st.session_state
    ):

        st.session_state[
            "alert_threshold"
        ] = recommended_threshold


    # ========================================================
    # ALERT THRESHOLD SLIDER
    # ========================================================

    threshold = st.slider(

        "Alert threshold",

        min_value=0.0,

        max_value=1.0,

        step=0.001,

        format="%.3f",

        key="alert_threshold",
    )


    st.caption(
        "Changing this control changes workload; "
        "it does not retrain the model."
    )


    st.divider()


    # ========================================================
    # RECOMMENDED THRESHOLD
    # ========================================================

    st.subheader(
        "Recommended threshold"
    )

    st.metric(

        "Validation-selected threshold",

        f"{recommended_threshold:.3f}",
    )

    st.caption(
        "Selected during validation using a 5% "
        "maximum alert-rate constraint."
    )


    # ========================================================
    # ESTIMATED ALERT RATE
    # ========================================================

    estimated_alert_rate = (
        df["risk_72h"]
        >= threshold
    ).mean()

    st.metric(

        "Estimated alert rate",

        f"{estimated_alert_rate:.1%}",
    )

    st.caption(
        "Estimated on the displayed historical cohort."
    )


    # ========================================================
    # CURRENT OPERATING POINT
    # ========================================================

    st.divider()

    st.subheader(
        "Current operating point"
    )

    if abs(
        threshold
        - recommended_threshold
    ) < 0.0005:

        st.success(
            "Using the recommended "
            "validation threshold."
        )

    else:

        st.info(
            f"Custom threshold selected: "
            f"{threshold:.3f}"
        )


# ============================================================
# EXECUTIVE SUMMARY
# ============================================================

render_summary(
    meta,
    test_metrics,
)


# ============================================================
# PATIENT RISK EXPLORER
# ============================================================

render_patients(
    df,
    threshold,
    model,
)


# ============================================================
# THRESHOLD LAB
# ============================================================

if not threshold_df.empty:

    render_threshold(
        threshold_df
    )


# ============================================================
# SAFETY NOTICE
# ============================================================

st.divider()

st.warning(
    "Clinical safety: a risk score is not a diagnosis. "
    "Any escalation, theatre reprioritisation, or "
    "critical-care review remains a clinician decision."
)