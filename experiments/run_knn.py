import numpy as np

from src.data import load_dataset
from src.metrics import accuracy
from src.models.knn import find_neighbors, sweep, weighted_vote
from src.preprocessing import fit_standardizer, standardize
from src.split import stratified_split


def run_knn(name):
    """Split the dataset, then compare raw and standardized features for every k on the validation set.

    Returns: nothing, only prints the results.
    """
    X_train, y_train, _, _ = load_dataset(name)
    n_classes = len(np.unique(y_train))

    train_idx, val_idx = stratified_split(y_train)
    X_tr, y_tr = X_train[train_idx], y_train[train_idx]
    X_val, y_val = X_train[val_idx], y_train[val_idx]
    print(f"--- {name} : train {X_tr.shape}, val {X_val.shape}, {n_classes} classes")

    mean, std = fit_standardizer(X_tr)
    X_tr_normalized = standardize(X_tr, mean, std)
    X_val_normalized = standardize(X_val, mean, std)
    ks = [1, 3, 5, 11, 21, 31, 61, 101]
    neighbors, distances = find_neighbors(X_tr_normalized, X_tr_normalized[:200], 1)
    sanity_check = weighted_vote(neighbors, distances, y_tr, n_classes)
    print(f"--- {name} : accuracy for sanity check = {accuracy(sanity_check, y_tr[:200])}")

    sweep(X_tr, y_tr, X_val, y_val, ks, n_classes, f"{name}-raw")
    sweep(X_tr_normalized, y_tr, X_val_normalized, y_val, ks, n_classes, f"{name}-normalized")


if __name__ == "__main__":
    run_knn("image")
    run_knn("sentiment")
