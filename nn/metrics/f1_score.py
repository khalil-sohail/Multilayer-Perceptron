
def f1_score(y_true, y_pred):
    """F1 score (harmonic mean of precision and recall).
    
    Provides a single score that balances precision and recall. Useful for
    imbalanced datasets where you care about both false positives and negatives.
    
    Formula:
        F1 = 2 * (Precision * Recall) / (Precision + Recall)
          = (2 * TP) / (2 * TP + FP + FN)
    
    Args:
        y_true (array-like): Ground truth binary labels.
        y_pred (array-like): Predicted binary labels.
    
    Returns:
        float: F1 score between 0 and 1. Higher is better.
    """
    TP = sum(1 for true, pred in zip(y_true, y_pred) if pred == 1 and true == 1)
    FP = sum(1 for true, pred in zip(y_true, y_pred) if pred == 1 and true == 0)
    FN = sum(1 for true, pred in zip(y_true, y_pred) if pred == 0 and true == 1)
    return (2 * TP) / ((2 * TP) + FP + FN)
