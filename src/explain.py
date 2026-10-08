from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd


def transformed_importance(model, out_path=None, top_n=30):
    """Global XGBoost gain/weight proxy mapped back to transformed feature names."""
    base = model
    if hasattr(model, "calibrated_classifiers_"):
        base = model.calibrated_classifiers_[0].estimator
    pre = base.named_steps["preprocess"]
    clf = base.named_steps["model"]
    names = pre.get_feature_names_out()
    imp = pd.Series(clf.feature_importances_, index=names).sort_values(ascending=False).head(top_n)
    out = imp.rename("importance").reset_index().rename(columns={"index": "feature"})
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_path, index=False)
    return out


def shap_local(model, feature_row: pd.DataFrame, top_n=10):
    """Return SHAP contributions for one patient row after the production preprocessing pipeline."""
    import shap
    base = model
    if hasattr(model, "calibrated_classifiers_"):
        base = model.calibrated_classifiers_[0].estimator
    pre = base.named_steps["preprocess"]
    clf = base.named_steps["model"]
    x = pre.transform(feature_row)
    names = pre.get_feature_names_out()
    explainer = shap.TreeExplainer(clf)
    sv = explainer.shap_values(x)
    values = sv[0] if isinstance(sv, list) else sv[0]
    return pd.DataFrame({"feature": names, "shap_value": values}).assign(abs_shap=lambda d: d.shap_value.abs()).sort_values("abs_shap", ascending=False).head(top_n)
