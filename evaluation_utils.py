import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, jaccard_score


def calculate_metrics(predicted, ground_truth):
    predicted = np.array(predicted).flatten()
    ground_truth = np.array(ground_truth).flatten()
    precision, recall, f_score, _ = precision_recall_fscore_support(ground_truth, predicted, average='binary')
    accuracy = accuracy_score(ground_truth, predicted)
    jaccard = jaccard_score(ground_truth, predicted)
    return precision, recall, f_score, accuracy, jaccard


def cfmatrix(actual, predict, classlist=None, per=0, printout=True):
    """
    Calculates the confusion matrix and various classification metrics.
    
    Parameters:
    - actual: Numpy array of actual class labels
    - predict: Numpy array of predicted class labels
    - classlist: List of all classes
    - per: If 1, outputs percentages in the confusion matrix
    - printout: If True, prints the confusion matrix and metrics
    
    Returns:
    - A dictionary with confusion matrix and metrics
    """
    if classlist is None:
        classlist = np.unique(actual)
    
    n_class = len(classlist)
    confmatrix = np.zeros((n_class, n_class))
    
    # Calculate confusion matrix
    for i, class_i in enumerate(classlist):
        for j, class_j in enumerate(classlist):
            confmatrix[i, j] = np.sum((predict == class_i) & (actual == class_j))
    
    TP = np.diag(confmatrix)
    FP = np.sum(confmatrix, axis=0) - TP
    FN = np.sum(confmatrix, axis=1) - TP
    TN = np.sum(confmatrix) - (FP + FN + TP)
    
    print("True Positives:", TP)
    print("False Positives:", FP)
    print("False Negatives:", FN)
    print("True Negatives:", TN)
    
    
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    Specificity = TN / (TN + FP)
    ModelAccuracy = np.sum(TP) / np.sum(confmatrix)
    Jaccard_idx = TP / (TP + FN + FP)
    F1_score = 2 * (Precision * Recall) / (Precision + Recall)
    
    metrics = {
        'Precision': Precision,
        'Recall': Recall,
        'F1 Score': F1_score,
        'Specificity': Specificity,
        'Accuracy': ModelAccuracy,
        'Jaccard Index': Jaccard_idx,
        'Confusion Matrix': confmatrix
    }
    
    if per:
        confmatrix = (confmatrix / np.sum(confmatrix)) * 100
    
    if printout:
        print("Confusion Matrix:")
        print(confmatrix)
        print("Metrics:")
        for key, value in metrics.items():
            print(f"{key}: {value}")
    
    return metrics
