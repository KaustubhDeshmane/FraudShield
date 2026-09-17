"""Model evaluation module for Credit Card Fraud Detection.

Computes comprehensive metrics suitable for imbalanced classification:
Precision, Recall, F1, ROC-AUC, PR-AUC, Confusion Matrix, and curve coordinates.
"""

from typing import Dict, Any, Optional, List, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    classification_report,
)


def _downsample_curve(x_vals: np.ndarray, y_vals: np.ndarray, max_points: int = 150) -> Dict[str, List[float]]:
    """Downsample large curve arrays to a concise list of points for clean JSON storage and plotting."""
    n_points = len(x_vals)
    if n_points <= max_points:
        indices = np.arange(n_points)
    else:
        # Uniform sampling across indices, ensuring first and last points are included
        indices = np.unique(np.linspace(0, n_points - 1, max_points, dtype=int))

    return {
        "x": [round(float(v), 5) for v in x_vals[indices]],
        "y": [round(float(v), 5) for v in y_vals[indices]],
    }


def evaluate_model(
    model: Any,
    X_test: Union[pd.DataFrame, np.ndarray],
    y_test: Union[pd.Series, np.ndarray],
    feature_names: Optional[List[str]] = None,
    threshold: float = 0.50,
) -> Dict[str, Any]:
    """Calculate comprehensive evaluation metrics for a trained classifier."""
    y_test_arr = np.asarray(y_test).astype(int)

    # Calculate probabilities and predictions at threshold
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        # Fallback for models without predict_proba
        d_func = model.decision_function(X_test)
        y_proba = (d_func - d_func.min()) / (d_func.max() - d_func.min() + 1e-9)
    else:
        y_proba = model.predict(X_test).astype(float)

    y_pred = (y_proba >= threshold).astype(int)

    # Confusion matrix breakdown
    cm = confusion_matrix(y_test_arr, y_pred, labels=[0, 1])
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]

    # Standard metrics
    acc = float(accuracy_score(y_test_arr, y_pred))
    bal_acc = float(balanced_accuracy_score(y_test_arr, y_pred))
    p1 = float(precision_score(y_test_arr, y_pred, pos_label=1, zero_division=0))
    r1 = float(recall_score(y_test_arr, y_pred, pos_label=1, zero_division=0))
    f1_1 = float(f1_score(y_test_arr, y_pred, pos_label=1, zero_division=0))

    p0 = float(precision_score(y_test_arr, y_pred, pos_label=0, zero_division=0))
    r0 = float(recall_score(y_test_arr, y_pred, pos_label=0, zero_division=0))
    f1_0 = float(f1_score(y_test_arr, y_pred, pos_label=0, zero_division=0))

    macro_f1 = float(f1_score(y_test_arr, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_test_arr, y_pred, average="weighted", zero_division=0))

    roc_auc = float(roc_auc_score(y_test_arr, y_proba))
    pr_auc = float(average_precision_score(y_test_arr, y_proba))

    # Curve points for interactive plotting
    fpr, tpr, _ = roc_curve(y_test_arr, y_proba)
    roc_curve_data = _downsample_curve(fpr, tpr)

    prec_pts, rec_pts, _ = precision_recall_curve(y_test_arr, y_proba)
    pr_curve_data = _downsample_curve(rec_pts, prec_pts)

    metrics: Dict[str, Any] = {
        "threshold": float(threshold),
        "total_test_samples": int(len(y_test_arr)),
        "actual_fraud_count": int((y_test_arr == 1).sum()),
        "actual_legitimate_count": int((y_test_arr == 0).sum()),
        "accuracy": round(acc, 5),
        "balanced_accuracy": round(bal_acc, 5),
        "precision": round(p1, 5),
        "recall": round(r1, 5),
        "f1": round(f1_1, 5),
        "roc_auc": round(roc_auc, 5),
        "pr_auc": round(pr_auc, 5),
        "macro_f1": round(macro_f1, 5),
        "weighted_f1": round(weighted_f1, 5),
        "class_1": {
            "precision": round(p1, 5),
            "recall": round(r1, 5),
            "f1_score": round(f1_1, 5),
            "support": int((y_test_arr == 1).sum()),
        },
        "class_0": {
            "precision": round(p0, 5),
            "recall": round(r0, 5),
            "f1_score": round(f1_0, 5),
            "support": int((y_test_arr == 0).sum()),
        },
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "matrix": [[tn, fp], [fn, tp]],
        },
        "roc_curve": roc_curve_data,
        "pr_curve": pr_curve_data,
    }

    # Feature importances if available (e.g. Random Forest)
    if hasattr(model, "feature_importances_") and feature_names:
        importances = model.feature_importances_
        feat_imp = [
            {"feature": str(feat), "importance": round(float(imp), 5)}
            for feat, imp in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
        ]
        metrics["feature_importances"] = feat_imp

    return metrics


if __name__ == "__main__":
    print("evaluate module loaded successfully.")
