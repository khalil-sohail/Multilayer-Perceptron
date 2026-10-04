
def accuracy(y_true, y_pred):
    """Accuracy classification score.
    
    Proportion of correct predictions among the total number of predictions.
    Useful for balanced datasets.
    
    Formula:
        Accuracy = (TP + TN) / (TP + TN + FP + FN)
    
    Args:
        y_true (array-like): Ground truth binary labels.
        y_pred (array-like): Predicted binary labels.
    
    Returns:
        float: Accuracy score between 0 and 1. Higher is better.
    """
    TP_AND_FN = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    return TP_AND_FN / len(y_true)
