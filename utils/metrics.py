import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


def evaluate(y_true, y_pred, y_prob=None):
    """
    Returns dict of all evaluation metrics.
    y_true : list of true labels (0 or 1)
    y_pred : list of predicted labels (0 or 1)
    y_prob : list of probabilities for class 1 (optional)
    """
    results = {
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "f1_score":  round(f1_score(y_true, y_pred, average='weighted'), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
    }

    if y_prob is not None:
        try:
            results["roc_auc"] = round(
                roc_auc_score(y_true, y_prob), 4
            )
        except Exception:
            results["roc_auc"] = "N/A"

    results["report"] = classification_report(y_true, y_pred)
    return results


def print_metrics(results: dict):
    print("\n" + "="*40)
    print(f"  Accuracy  : {results['accuracy']*100:.2f}%")
    print(f"  F1 Score  : {results['f1_score']*100:.2f}%")
    if "roc_auc" in results:
        print(f"  ROC AUC   : {results['roc_auc']}")
    print("="*40)
    print(results["report"])