import numpy as np


def accuracy(y_true, y_pred):
    """Fraction of predictions equal to the true labels.

    Returns: a float between 0 and 1.
    """
    assert len(y_true) == len(y_pred), "Not same length"
    correct = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            correct += 1
    return correct / len(y_true)


def confusion_matrix(y_true, y_pred, n_classes):
    """Count the examples for each (true class, predicted class) pair.

    Returns: array (n_classes, n_classes), rows = true class, columns = predicted class.
    """
    assert len(y_true) == len(y_pred), "Not same length"
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for i in range(len(y_true)):
        cm[y_true[i], y_pred[i]] += 1
    return cm


def per_class_accuracy(cm):
    """Fraction of well classified examples for each true class (diagonal / row sum).

    Returns: array (n_classes,) of floats between 0 and 1.
    """
    arr = np.array([], dtype=int)
    for i in range(cm.shape[0]):
        mean = cm[i, i] / cm.sum(axis=1)[i]
        arr = np.append(arr, mean)
    return arr
