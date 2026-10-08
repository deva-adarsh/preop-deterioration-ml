from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, roc_auc_score


def threshold_table(y_true, prob, thresholds=None):
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob, dtype=float)
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)
    rows = []
    for t in thresholds:
        pred = (prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        rows.append({
            "threshold": float(t), "n": len(y_true), "alerts": int(pred.sum()),
            "alert_rate": float(pred.mean()), "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
            "precision_ppv": tp / (tp + fp) if tp + fp else np.nan,
            "recall_sensitivity": tp / (tp + fn) if tp + fn else np.nan,
            "specificity": tn / (tn + fp) if tn + fp else np.nan,
            "fpr": fp / (fp + tn) if fp + tn else np.nan,
            "false_alarms_per_100": 100 * fp / (fp + tn) if fp + tn else np.nan,
            "npv": tn / (tn + fn) if tn + fn else np.nan,
        })
    return pd.DataFrame(rows)


def choose_threshold(y_true, prob, max_alert_rate=0.05):
    """Choose the highest recall achievable under an explicit workload cap.

    This intentionally does not use the test set. If several thresholds have effectively
    identical recall, prefer the one with higher PPV and then the higher threshold.
    """
    candidate = np.unique(np.round(np.asarray(prob, dtype=float), 6))
    candidate = np.r_[0.0, candidate, 1.0]
    tab = threshold_table(y_true, prob, candidate)
    feasible = tab[tab["alert_rate"] <= max_alert_rate].copy()
    if feasible.empty:
        feasible = tab.iloc[[tab["alert_rate"].argmin()]].copy()
    best = feasible.sort_values(
        ["recall_sensitivity", "precision_ppv", "threshold"],
        ascending=[False, False, False],
        na_position="last",
    ).iloc[0]
    return float(best.threshold), tab


def discrimination_metrics(y_true, prob):
    return {
        "roc_auc": roc_auc_score(y_true, prob),
        "pr_auc": average_precision_score(y_true, prob),
        "brier": brier_score_loss(y_true, prob),
        "prevalence": float(np.mean(y_true)),
    }


def calibration_table(y_true, prob, n_bins=10):
    frac_pos, mean_pred = calibration_curve(y_true, prob, n_bins=n_bins, strategy="quantile")
    return pd.DataFrame({"mean_predicted": mean_pred, "observed_rate": frac_pos})


def decision_curve(y_true, prob, thresholds=None):
    y_true = np.asarray(y_true)
    prob = np.asarray(prob)
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.50, 100)
    n = len(y_true)
    rows = []
    for t in thresholds:
        pred = prob >= t
        tp = ((pred) & (y_true == 1)).sum()
        fp = ((pred) & (y_true == 0)).sum()
        nb = tp / n - fp / n * (t / (1 - t))
        rows.append({"threshold": t, "net_benefit": nb})
    return pd.DataFrame(rows)
