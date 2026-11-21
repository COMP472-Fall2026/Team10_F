from numpy import ndarray
from seeds import SEED
import numpy as np


def stratified_split(y: ndarray, val_frac=0.2, seed=SEED):
    """Split the positions of y into train and validation, keeping the same class proportions.
    For each class, val_frac of its examples go to validation. Same seed = same split.

    Returns: (train_idx, val_idx), two arrays of positions in y (grouped by class, not shuffled).
    """
    list_of_classes = np.unique(y)
    list_train = []
    list_val = []
    rng = np.random.default_rng(seed)
    for c in list_of_classes:
        positions = np.where(y == c)[0]
        positions = rng.permutation(positions)
        n_val = round(len(positions) * val_frac)
        list_val.append(positions[:n_val])
        list_train.append(positions[n_val:])
    train_idx = np.concatenate(list_train)
    val_idx = np.concatenate(list_val)

    return train_idx, val_idx
