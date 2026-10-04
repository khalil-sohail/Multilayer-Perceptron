def recall(y_true, y_pred):
    """Recall (sensitivity or true positive rate) classification score.
    
    Proportion of actual positive cases that were correctly identified.
    Useful when false negatives are costly.
    
    Equation:
        Recall = TP / (TP + FN)
    
    Args:
        y_true (array-like): Ground truth binary labels.
        y_pred (array-like): Predicted binary labels.
    
    Returns:
        float: Recall score between 0 and 1. Higher is better.
    """
    TP = sum(1 for true, pred in zip(y_true, y_pred) if pred == 1 and true == 1)
    FN = sum(1 for true, pred in zip(y_true, y_pred) if pred == 0 and true == 1)
    return TP / (TP + FN)
