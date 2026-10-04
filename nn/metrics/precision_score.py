
def precision(y_true, y_pred):
    """Precision classification score.
    
    Proportion of predicted positive cases that were actually positive.
    Useful when false positives are costly.
    
    Equation:
        Precision = TP / (TP + FP)
    
    Args:
        y_true (array-like): Ground truth binary labels.
        y_pred (array-like): Predicted binary labels.
    
    Returns:
        float: Precision score between 0 and 1. Higher is better.
    """
    TP = sum(1 for true, pred in zip(y_true, y_pred) if pred == 1 and true == 1)
    FP = sum(1 for true, pred in zip(y_true, y_pred) if pred == 1 and true == 0)
    return TP / (TP + FP)
