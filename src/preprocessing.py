import numpy as np


def fit_standardizer(X_train):
    """Compute the mean and the std of each feature, on the train set only.

    Returns: (mean, std), two arrays (d,).
    """
    return X_train.mean(axis=0), X_train.std(axis=0)


def standardize(X, mean, std):
    """Scale each feature to mean 0 and std 1 (z-score), using the mean and std of the train set.
    Features with std 0 are only centered, to avoid a division by zero.

    Returns: array with the same shape as X.
    """
    return (X - mean) / np.where(std == 0, 1, std)
