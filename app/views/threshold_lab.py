import pandas as pd
import streamlit as st


# ============================================================
# THRESHOLD LAB
# ============================================================

def render(
    threshold_df: pd.DataFrame,
):
    """
    Render threshold operating-point analysis.

    Threshold selection:

        1. Maximize sensitivity / recall
        2. Maximize PPV as tie-breaker
        3. Prefer higher threshold as final tie-breaker

    Subject to the selected maximum alert-rate capacity.
    """

    st.header(
        "Threshold lab"
    )

    st.write(
        "Choose the maximum proportion of patients "
        "that the operational workflow can review."
    )


    # ========================================================
    # VALIDATE INPUT
    # ========================================================

    if (
        threshold_df is None
        or threshold_df.empty
    ):

        st.warning(
            "Threshold selection data is unavailable."
        )

        return


    required_columns = {
        "threshold",
        "alert_rate",
        "recall_sensitivity",
        "precision_ppv",
    }


    missing_columns = (
        required_columns
        - set(threshold_df.columns)
    )


    if missing_columns:

        st.error(
            "Threshold table is missing required columns:"
        )

        st.write(
            sorted(missing_columns)
        )

        return


    # ========================================================
    # CLEAN DATA
    # ========================================================

    df = threshold_df.copy()


    numeric_columns = [
        "threshold",
        "alert_rate",
        "recall_sensitivity",
        "precision_ppv",
    ]


    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )


    df = df.dropna(
        subset=numeric_columns
    )


    df = df[
        (df["threshold"] >= 0.0)
        & (df["threshold"] <= 1.0)
    ].copy()


    if df.empty:

        st.warning(
            "No valid threshold records are available."
        )

        return


    # ========================================================
    # MAXIMUM ALERT RATE
    # ========================================================

    st.subheader(
        "Maximum alert rate"
    )

    # Use integer percentage points instead of a decimal
    # slider. This avoids Streamlit displaying 0% for 0.05.

    capacity_percent = st.slider(
        "Maximum alert rate",
        min_value=1,
        max_value=20,
        value=5,
        step=1,
        format="%d%%",
    )


    max_alert_rate = (
        capacity_percent / 100.0
    )


    st.caption(
        f"Selected capacity: "
        f"{capacity_percent}%"
    )


    # ========================================================
    # FEASIBLE THRESHOLDS
    # ========================================================

    feasible = df[
        df["alert_rate"]
        <= max_alert_rate
    ].copy()


    if feasible.empty:

        st.error(
            "No threshold satisfies the selected "
            "maximum alert-rate capacity."
        )

        return


    # ========================================================
    # SELECT RECOMMENDED THRESHOLD
    # ========================================================

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


    recommended_row = (
        feasible.iloc[0]
    )


    recommended_threshold = float(
        recommended_row[
            "threshold"
        ]
    )


    recommended_alert_rate = float(
        recommended_row[
            "alert_rate"
        ]
    )


    recommended_sensitivity = float(
        recommended_row[
            "recall_sensitivity"
        ]
    )


    recommended_ppv = float(
        recommended_row[
            "precision_ppv"
        ]
    )


    # ========================================================
    # RECOMMENDED OPERATING POINT
    # ========================================================

    st.subheader(
        "Recommended validation threshold"
    )


    st.metric(
        "Threshold",
        f"{recommended_threshold:.4f}",
    )


    columns = st.columns(3)


    columns[0].metric(
        "Alert rate",
        f"{recommended_alert_rate:.1%}",
    )


    columns[1].metric(
        "Sensitivity",
        f"{recommended_sensitivity:.1%}",
    )


    columns[2].metric(
        "PPV",
        f"{recommended_ppv:.1%}",
    )


    st.caption(
        "This threshold is calculated from the validation "
        "set. It maximizes sensitivity subject to the "
        "selected alert-rate capacity."
    )


    # ========================================================
    # THRESHOLDS NEAR SELECTED OPERATING POINT
    # ========================================================

    st.subheader(
        "Thresholds near the selected operating point"
    )


    # --------------------------------------------------------
    # Remove exact duplicate thresholds
    # --------------------------------------------------------

    near_df = (
        df.sort_values(
            by="threshold"
        )
        .drop_duplicates(
            subset=["threshold"],
            keep="first",
        )
        .copy()
    )


    # --------------------------------------------------------
    # Distance from recommended threshold
    # --------------------------------------------------------

    near_df[
        "threshold_distance"
    ] = (
        near_df["threshold"]
        - recommended_threshold
    ).abs()


    # --------------------------------------------------------
    # Take nearest thresholds
    # --------------------------------------------------------

    near_df = (
        near_df.sort_values(
            by=[
                "threshold_distance",
                "threshold",
            ],
            ascending=[
                True,
                True,
            ],
        )
        .head(10)
        .copy()
    )


    # Sort for display
    near_df = near_df.sort_values(
        by="threshold"
    )


    # ========================================================
    # DISPLAY COLUMNS
    # ========================================================

    display_columns = [
        "threshold",
        "alert_rate",
        "recall_sensitivity",
        "precision_ppv",
    ]


    optional_columns = [
        "specificity",
        "fpr",
        "false_alarms_per_100",
    ]


    for column in optional_columns:

        if column in near_df.columns:

            display_columns.append(
                column
            )


    display_df = near_df[
        display_columns
    ].copy()


    # ========================================================
    # FORMAT DISPLAY
    # ========================================================

    formatted_df = (
        display_df.copy()
    )


    if "threshold" in formatted_df.columns:

        formatted_df[
            "threshold"
        ] = formatted_df[
            "threshold"
        ].map(
            lambda value:
                f"{value:.4f}"
        )


    percentage_columns = [
        "alert_rate",
        "recall_sensitivity",
        "precision_ppv",
        "specificity",
        "fpr",
    ]


    for column in percentage_columns:

        if column in formatted_df.columns:

            formatted_df[
                column
            ] = formatted_df[
                column
            ].map(
                lambda value:
                    f"{value:.1%}"
            )


    if (
        "false_alarms_per_100"
        in formatted_df.columns
    ):

        formatted_df[
            "false_alarms_per_100"
        ] = formatted_df[
            "false_alarms_per_100"
        ].map(
            lambda value:
                f"{value:.2f}"
        )


    # ========================================================
    # RENAME COLUMNS
    # ========================================================

    formatted_df = formatted_df.rename(
        columns={
            "threshold":
                "Threshold",

            "alert_rate":
                "Alert rate",

            "recall_sensitivity":
                "Sensitivity",

            "precision_ppv":
                "PPV",

            "specificity":
                "Specificity",

            "fpr":
                "FPR",

            "false_alarms_per_100":
                "False alarms / 100",
        }
    )


    # ========================================================
    # DISPLAY TABLE
    # ========================================================

    st.dataframe(
        formatted_df,
        use_container_width=True,
        hide_index=True,
    )


    # ========================================================
    # EXPLANATION
    # ========================================================

    st.info(
        f"With a maximum alert-rate capacity of "
        f"{max_alert_rate:.0%}, the selected validation "
        f"threshold is {recommended_threshold:.4f}. "
        f"It produces approximately "
        f"{recommended_alert_rate:.1%} alerts, "
        f"{recommended_sensitivity:.1%} sensitivity, "
        f"and {recommended_ppv:.1%} PPV on the validation "
        f"data."
    )