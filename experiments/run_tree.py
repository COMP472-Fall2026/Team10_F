import time

import numpy as np

from src.data import load_dataset
from src.metrics import accuracy
from src.models.decision_tree import build_tree, predict
from src.split import stratified_split


def run_tree(name, max_depths=(1, 3, 6, 8, 10, 13, 15, 17, 20)):
    """Split the dataset, then build one tree per max_depth and compare train and validation accuracy.

    Returns: nothing, only prints the results.
    """
    X_train, y_train, _, _ = load_dataset(name)
    n_classes = len(np.unique(y_train))

    train_idx, val_idx = stratified_split(y_train)
    X_tr, y_tr = X_train[train_idx], y_train[train_idx]
    X_val, y_val = X_train[val_idx], y_train[val_idx]
    print(f"--- {name} : train {X_tr.shape}, val {X_val.shape}, {n_classes} classes")

    for max_depth in max_depths:
        start = time.time()
        tree = build_tree(X_tr, y_tr, n_classes, max_depth=max_depth)
        build_time = time.time() - start

        train_acc = accuracy(y_tr, predict(tree, X_tr))
        val_acc = accuracy(y_val, predict(tree, X_val))
        print(f"--- {name} : max_depth = {max_depth:2d} | train = {train_acc:.4f} | val = {val_acc:.4f} | {build_time:.1f} s")


if __name__ == "__main__":
    run_tree("sentiment", max_depths=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12))
    run_tree("image", max_depths=(10, 20, 30, 40, 60))
