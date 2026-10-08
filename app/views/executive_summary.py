import streamlit as st


def render(meta, test_metrics=None):
    st.header("Executive summary")

    st.write(
        "The model estimates the probability of severe pre-operative "
        "deterioration after the six-hour landmark and within 72 hours "
        "of admission."
    )

    st.caption(
        "Use as decision support. It should trigger human review rather "
        "than automatically reordering theatre cases."
    )

    # ---------------------------------------------------------
    # Model performance metrics
    # ---------------------------------------------------------
    if test_metrics:
        cols = st.columns(4)

        pr_auc = test_metrics.get("pr_auc")
        roc_auc = test_metrics.get("roc_auc")
        brier = test_metrics.get("brier")
        prevalence = test_metrics.get("prevalence")

        # Safely display metrics
        cols[0].metric(
            "PR-AUC",
            f"{pr_auc:.3f}" if pr_auc is not None else "N/A"
        )

        cols[1].metric(
            "ROC-AUC",
            f"{roc_auc:.3f}" if roc_auc is not None else "N/A"
        )

        cols[2].metric(
            "Brier",
            f"{brier:.3f}" if brier is not None else "N/A"
        )

        cols[3].metric(
            "Test prevalence",
            f"{prevalence:.1%}" if prevalence is not None else "N/A"
        )

    # ---------------------------------------------------------
    # Operating contract
    # ---------------------------------------------------------
    st.subheader("Operating contract")

    st.write(
        meta.get(
            "threshold_rule",
            "Threshold rule unavailable"
        )
    )