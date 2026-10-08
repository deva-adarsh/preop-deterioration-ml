from __future__ import annotations

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
from .evaluate import choose_threshold, threshold_table

TARGET = "label"
ID_COLS = ["patient_id"]


def temporal_split(df: pd.DataFrame):
    d = df.copy()
    d["admit_ts"] = pd.to_datetime(d["admit_ts"], utc=True)
    train = d[d.admit_ts < "2024-01-01"].copy()
    valid = d[(d.admit_ts >= "2024-01-01") & (d.admit_ts < "2024-07-01")].copy()
    test = d[d.admit_ts >= "2024-07-01"].copy()
    return train, valid, test


def make_model(num_cols, cat_cols, seed=42):
    pre = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True))]), num_cols),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
    ])
    clf = XGBClassifier(
        n_estimators=500, max_depth=4, learning_rate=0.04,
        subsample=0.85, colsample_bytree=0.85,
        min_child_weight=5, reg_lambda=2.0, reg_alpha=0.1,
        objective="binary:logistic", eval_metric="logloss",
        tree_method="hist", random_state=seed, n_jobs=-1,
    )
    return Pipeline([("preprocess", pre), ("model", clf)])


def grouped_cv_score(X, y, groups, num_cols, cat_cols):
    scores = []
    for fold, (tr, va) in enumerate(GroupKFold(n_splits=6).split(X, y, groups)):
        model = make_model(num_cols, cat_cols, seed=100 + fold)
        model.fit(X.iloc[tr], y.iloc[tr])
        p = model.predict_proba(X.iloc[va])[:, 1]
        scores.append({"practice_holdout": int(fold), "pr_auc": average_precision_score(y.iloc[va], p), "roc_auc": roc_auc_score(y.iloc[va], p)})
    return pd.DataFrame(scores)


def train(features: pd.DataFrame, out_dir: str | Path, alert_rate_cap=0.05):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    df = features.copy()
    train_df, valid_df, test_df = temporal_split(df)
    drop = {TARGET, "admit_ts", "landmark_ts", "horizon_ts", "first_event_ts", "actual_start_ts", "surgery_scheduled_ts", "discharge_disposition"}
    drop |= set(ID_COLS)
    feature_cols = [c for c in df.columns if c not in drop]
    cat_cols = [c for c in ["practice_id", "procedure_cpt", "urgency", "sex", "smoking_status", "surgeon_volume_indicator"] if c in feature_cols]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    # Practice-held-out robustness check is descriptive; temporal split remains primary.
    cv = grouped_cv_score(train_df[feature_cols], train_df[TARGET], train_df["practice_id"], num_cols, cat_cols)
    cv.to_csv(out_dir / "practice_cv.csv", index=False)

    model = make_model(num_cols, cat_cols)
    model.fit(train_df[feature_cols], train_df[TARGET])
    p_valid = model.predict_proba(valid_df[feature_cols])[:, 1]
    threshold, selection = choose_threshold(valid_df[TARGET].to_numpy(), p_valid, max_alert_rate=alert_rate_cap)
    pd.DataFrame(selection).to_csv(out_dir / "threshold_selection.csv", index=False)

    p_test = model.predict_proba(test_df[feature_cols])[:, 1]
    test_metrics = threshold_table(test_df[TARGET].to_numpy(), p_test, [threshold])
    test_metrics.to_csv(out_dir / "test_metrics.csv", index=False)
    joblib.dump(model, out_dir / "pipeline.joblib")

    metadata = {
        "landmark_hours": 6, "horizon_hours": 72,
        "primary_split": "train <2024-01-01; validation 2024-01-01..2024-06-30; test >=2024-07-01",
        "threshold_rule": f"validation threshold maximizing recall subject to alert_rate <= {alert_rate_cap:.1%}",
        "feature_columns": feature_cols, "categorical_columns": cat_cols, "numeric_columns": num_cols,
        "positive_event_types": sorted(__import__("src.cohort", fromlist=["DETERIORATION_EVENTS"]).DETERIORATION_EVENTS),
        "test_metrics": test_metrics.to_dict(orient="records"),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    return model, metadata
