from __future__ import annotations

import pandas as pd
import streamlit as st
import shap


# ============================================================
# SHAP HELPERS
# ============================================================

def _get_feature_columns(model, df: pd.DataFrame) -> list[str]:
    """
    Determine the exact raw feature columns expected by the
    trained sklearn pipeline.

    Priority:
    1. model.feature_names_in_
    2. known columns from the dataframe after removing
       identifiers, targets, timestamps and downstream fields.
    """

    # ---------------------------------------------------------
    # Preferred: feature names stored by sklearn
    # ---------------------------------------------------------

    feature_columns = getattr(
        model,
        "feature_names_in_",
        None,
    )

    if feature_columns is not None:
        feature_columns = list(feature_columns)

        # Keep only columns actually available in the dataframe.
        feature_columns = [
            col
            for col in feature_columns
            if col in df.columns
        ]

        if feature_columns:
            return feature_columns

    # ---------------------------------------------------------
    # Fallback
    # ---------------------------------------------------------

    excluded_columns = {
        "patient_id",
        "label",
        "risk_72h",

        # Temporal / downstream columns
        "admit_ts",
        "landmark_ts",
        "horizon_ts",
        "first_event_ts",
        "actual_start_ts",
        "surgery_scheduled_ts",
        "discharge_disposition",
    }

    feature_columns = [
        col
        for col in df.columns
        if col not in excluded_columns
    ]

    return feature_columns


def _get_shap_values(
    model,
    patient_row: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate SHAP values for one patient.

    The trained model is expected to be:

        sklearn Pipeline
            |
            +-- preprocessing / ColumnTransformer
            |
            +-- XGBoost classifier

    We transform the raw patient data using the preprocessing
    part of the pipeline and then explain the XGBoost estimator.
    """

    # =========================================================
    # Validate model structure
    # =========================================================

    if not hasattr(model, "steps"):
        raise TypeError(
            "The supplied model is not a sklearn Pipeline."
        )

    if len(model.steps) < 2:
        raise ValueError(
            "Expected a preprocessing step followed by "
            "a final estimator."
        )

    # ---------------------------------------------------------
    # Separate preprocessing and estimator
    # ---------------------------------------------------------

    preprocessor = model[:-1]
    estimator = model.steps[-1][1]

    # =========================================================
    # Transform patient
    # =========================================================

    transformed_patient = preprocessor.transform(
        patient_row
    )

    # =========================================================
    # Get transformed feature names
    # =========================================================

    try:
        feature_names = (
            preprocessor
            .get_feature_names_out()
            .tolist()
        )

    except Exception:
        feature_names = [
            f"feature_{i}"
            for i in range(
                transformed_patient.shape[1]
            )
        ]

    # =========================================================
    # SHAP TreeExplainer
    # =========================================================

    explainer = shap.TreeExplainer(
        estimator
    )

    shap_output = explainer(
        transformed_patient
    )

    shap_values = shap_output.values

    # =========================================================
    # Handle SHAP output dimensions
    # =========================================================

    # Binary classification can sometimes return:

    # (n_samples, n_features, n_classes)

    if shap_values.ndim == 3:

        # Explain positive class = class 1
        shap_values = shap_values[:, :, 1]

    if shap_values.ndim == 2:

        # We only have one patient.
        shap_values = shap_values[0]

    elif shap_values.ndim == 1:

        # Already one-dimensional.
        shap_values = shap_values

    else:

        raise ValueError(
            "Unexpected SHAP output shape: "
            f"{shap_values.shape}"
        )

    # =========================================================
    # Validate dimensions
    # =========================================================

    if len(feature_names) != len(shap_values):

        raise ValueError(
            "Number of SHAP values does not match "
            "number of transformed features. "
            f"features={len(feature_names)}, "
            f"shap_values={len(shap_values)}"
        )

    # =========================================================
    # Create explanation dataframe
    # =========================================================

    explanation = pd.DataFrame(
        {
            "feature": feature_names,
            "shap_value": shap_values,
        }
    )

    explanation["abs_shap"] = (
        explanation["shap_value"]
        .abs()
    )

    # Keep the ten strongest contributors.
    explanation = (
        explanation
        .sort_values(
            "abs_shap",
            ascending=False,
        )
        .head(10)
        .reset_index(drop=True)
    )

    return explanation


def _clean_feature_name(name: str) -> str:
    """
    Convert sklearn transformed feature names into
    human-readable names.
    """

    name = str(name)

    # Remove transformer prefixes.
    name = name.replace(
        "num__",
        "",
    )

    name = name.replace(
        "cat__",
        "",
    )

    # Make one-hot categorical names easier to read.
    name = name.replace(
        "_",
        " ",
    )

    return name


# ============================================================
# MAIN VIEW
# ============================================================

def render(
    df: pd.DataFrame,
    threshold: float,
    model,
):
    """
    Render the patient risk explorer.

    Parameters
    ----------
    df:
        Historical feature dataframe containing model inputs
        and risk_72h predictions.

    threshold:
        Current alert threshold selected by the application.

    model:
        Trained sklearn Pipeline.
    """

    st.header(
        "Patient risk explorer"
    )

    st.write(
        "Enter a patient ID to inspect the model's "
        "predicted risk score and the main features "
        "contributing to that prediction."
    )

    # =========================================================
    # VALIDATE DATAFRAME
    # =========================================================

    if df is None or df.empty:

        st.warning(
            "No patient feature data is available."
        )

        return

    if "patient_id" not in df.columns:

        st.error(
            "The feature dataframe does not contain "
            "'patient_id'."
        )

        return

    if "risk_72h" not in df.columns:

        st.error(
            "The feature dataframe does not contain "
            "'risk_72h'."
        )

        return

    # =========================================================
    # PATIENT ID INPUT
    # =========================================================

    patient_ids = (
        df["patient_id"]
        .dropna()
        .astype(str)
        .str.strip()
    )

    if patient_ids.empty:

        st.warning(
            "No patient IDs are available."
        )

        return

    # ---------------------------------------------------------
    # Use the first patient as the initial example.
    #
    # This preserves the behavior shown in your current app
    # while avoiding an undefined default_patient_id variable.
    # ---------------------------------------------------------

    default_patient_id = patient_ids.iloc[0]

    patient_id = st.text_input(
        "Patient ID",
        value=default_patient_id,
        placeholder="Enter a patient ID, e.g. p-012373",
        key="patient_risk_explorer_patient_id",
    )

    patient_id = patient_id.strip()

    # =========================================================
    # NO PATIENT ID
    # =========================================================

    if not patient_id:

        st.info(
            "Enter a patient ID above to view the "
            "patient's risk assessment."
        )

        # -----------------------------------------------------
        # Show highest-risk patients as a useful fallback.
        # -----------------------------------------------------

        preview_columns = [
            "patient_id",
            "practice_id",
            "urgency",
            "asa_class",
            "age",
            "risk_72h",
        ]

        preview_columns = [
            col
            for col in preview_columns
            if col in df.columns
        ]

        if "risk_72h" in preview_columns:

            preview = (
                df[preview_columns]
                .sort_values(
                    "risk_72h",
                    ascending=False,
                )
                .head(10)
            )

            st.subheader(
                "Highest-risk patients"
            )

            st.dataframe(
                preview,
                use_container_width=True,
                hide_index=True,
            )

        return

    # =========================================================
    # FIND PATIENT
    # =========================================================

    patient_match = df[
        df["patient_id"]
        .astype(str)
        .str.strip()
        .str.upper()
        == patient_id.upper()
    ]

    # =========================================================
    # PATIENT NOT FOUND
    # =========================================================

    if patient_match.empty:

        st.error(
            f"Patient `{patient_id}` was not found."
        )

        st.caption(
            "Check the patient ID and try again."
        )

        return

    # =========================================================
    # GET PATIENT
    # =========================================================

    patient = patient_match.iloc[0]

    actual_patient_id = str(
        patient["patient_id"]
    )

    # =========================================================
    # RISK
    # =========================================================

    try:

        risk = float(
            patient["risk_72h"]
        )

    except (TypeError, ValueError):

        st.error(
            "The patient's risk score is invalid."
        )

        return

    # Protect against unexpected model output.
    risk = max(
        0.0,
        min(1.0, risk),
    )

    # Protect threshold as well.
    try:

        threshold = float(threshold)

    except (TypeError, ValueError):

        threshold = 0.121

    threshold = max(
        0.0,
        min(1.0, threshold),
    )

    alert = risk >= threshold

    # =========================================================
    # PATIENT HEADER
    # =========================================================

    st.subheader(
        f"Patient {actual_patient_id}"
    )

    # =========================================================
    # RISK SUMMARY
    # =========================================================

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Predicted risk score",
        f"{risk:.1%}",
    )

    col2.metric(
        "Alert threshold",
        f"{threshold:.1%}",
    )

    if alert:

        col3.error(
            "HIGH RISK"
        )

    else:

        col3.success(
            "BELOW THRESHOLD"
        )

    # =========================================================
    # DECISION SUPPORT MESSAGE
    # =========================================================

    if alert:

        st.warning(
            f"This patient's predicted risk score "
            f"({risk:.1%}) is above the selected "
            f"alert threshold ({threshold:.1%}). "
            "Human clinical review may be appropriate."
        )

    else:

        st.info(
            f"This patient's predicted risk score "
            f"({risk:.1%}) is below the selected "
            f"alert threshold ({threshold:.1%})."
        )

    # =========================================================
    # PATIENT CHARACTERISTICS
    # =========================================================

    st.subheader(
        "Patient characteristics"
    )

    characteristics = {}

    fields = {
        "practice_id": "Practice",
        "urgency": "Urgency",
        "asa_class": "ASA class",
        "age": "Age",
        "sex": "Sex",
        "weight_kg": "Weight (kg)",
        "height_cm": "Height (cm)",
        "bmi": "BMI",
        "smoking_status": "Smoking status",
        "procedure_cpt": "Procedure CPT",
    }

    for column, label in fields.items():

        if column not in patient.index:
            continue

        value = patient[column]

        if pd.notna(value):

            characteristics[label] = value

    if characteristics:

        characteristics_df = pd.DataFrame(
            characteristics.items(),
            columns=[
                "Attribute",
                "Value",
            ],
        )

        st.dataframe(
            characteristics_df,
            use_container_width=True,
            hide_index=True,
        )

    # =========================================================
    # SHAP EXPLANATION
    # =========================================================

    st.subheader(
        "Why did the model assign this risk?"
    )

    st.caption(
        "SHAP values show which model features contributed "
        "most strongly to this prediction. Positive values "
        "push the prediction higher; negative values push it "
        "lower. These are model contributions, not causal "
        "effects."
    )

    # =========================================================
    # PREPARE MODEL INPUT
    # =========================================================

    try:

        feature_columns = _get_feature_columns(
            model,
            df,
        )

        if not feature_columns:

            raise ValueError(
                "No model feature columns could be determined."
            )

        # -----------------------------------------------------
        # Ensure every required model column is available.
        # -----------------------------------------------------

        missing_columns = [
            col
            for col in feature_columns
            if col not in patient.index
        ]

        if missing_columns:

            raise ValueError(
                "The patient record is missing model "
                f"features: {missing_columns[:10]}"
            )

        patient_features = (
            patient[feature_columns]
            .to_frame()
            .T
        )

        # =====================================================
        # CALCULATE SHAP
        # =====================================================

        explanation = _get_shap_values(
            model,
            patient_features,
        )

        # =====================================================
        # HUMAN-READABLE NAMES
        # =====================================================

        explanation[
            "feature_display"
        ] = (
            explanation["feature"]
            .apply(_clean_feature_name)
        )

        # =====================================================
        # CONTRIBUTION DIRECTION
        # =====================================================

        explanation[
            "direction"
        ] = explanation[
            "shap_value"
        ].apply(
            lambda value:
                "↑ Higher risk"
                if value > 0
                else "↓ Lower risk"
        )

        # =====================================================
        # DISPLAY TABLE
        # =====================================================

        display_df = explanation[
            [
                "feature_display",
                "shap_value",
                "direction",
            ]
        ].copy()

        display_df.columns = [
            "Feature",
            "SHAP contribution",
            "Direction",
        ]

        display_df[
            "SHAP contribution"
        ] = (
            display_df[
                "SHAP contribution"
            ].round(4)
        )

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
        )

        # =====================================================
        # SHAP BAR CHART
        # =====================================================

        chart_df = (
            explanation
            .set_index(
                "feature_display"
            )["shap_value"]
            .sort_values()
        )

        st.bar_chart(
            chart_df,
            horizontal=True,
        )

    except Exception as exc:

        st.error(
            "Unable to calculate the SHAP explanation."
        )

        # Useful during development.
        # You can remove this before a clinical-facing deployment.
        with st.expander(
            "Show technical details"
        ):

            st.exception(exc)

    # =========================================================
    # RISK INTERPRETATION
    # =========================================================

    st.subheader(
        "Risk interpretation"
    )

    st.write(
        f"The model assigns this patient a predicted "
        f"risk score of {risk:.1%} for the defined severe "
        "deterioration outcome within the prediction window."
    )

    st.caption(
        "The risk score is a model prediction and should "
        "not be interpreted as a diagnosis or as evidence "
        "that any individual feature caused deterioration."
    )