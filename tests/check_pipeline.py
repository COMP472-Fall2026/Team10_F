import numpy as np

from seeds import SEED
from src.data import load_dataset
from src.metrics import accuracy, confusion_matrix, per_class_accuracy
from src.split import stratified_split


def test_pipeline(name):
    """Check that loading, split and metrics work together, using two fake models.

    Returns: nothing, prints the results and stops with an AssertionError if something is wrong.
    """
    # 1. Loading
    X_train, y_train, _, _ = load_dataset(name)
    n_classes = len(np.unique(y_train))

    # 2. Split: the same positions are used for X and y, so the rows stay aligned
    train_idx, val_idx = stratified_split(y_train)
    X_tr, _ = X_train[train_idx], y_train[train_idx]
    X_val, y_val = X_train[val_idx], y_train[val_idx]
    print(f"--- {name} : train {X_tr.shape}, val {X_val.shape}, {n_classes} classes")

    # 3. Fake model 1: always predicts class 0
    y_pred_zero = np.zeros(len(y_val), dtype=int)
    acc = accuracy(y_val, y_pred_zero)
    cm = confusion_matrix(y_val, y_pred_zero, n_classes)
    pca = per_class_accuracy(cm)
    print(f"Always 0    : accuracy = {acc:.4f} (expected {1 / n_classes:.4f})")
    print(f"              per-class = {pca[:5]} ...")

    # Checks: the exact result is known in advance
    assert np.isclose(acc, 1 / n_classes), "wrong accuracy"
    assert cm[:, 1:].sum() == 0, "predictions outside column 0"
    assert cm.sum() == len(y_val), "the matrix doesn't count every example"
    assert pca[0] == 1.0 and np.all(pca[1:] == 0), "wrong per-class accuracy"

    # 4. Fake model 2: predicts a random class
    rng = np.random.default_rng(SEED)
    y_pred_random = rng.integers(0, n_classes, size=len(y_val))
    acc = accuracy(y_val, y_pred_random)
    print(f"Random      : accuracy = {acc:.4f} (expected ≈ {1 / n_classes:.4f})")
    print()


if __name__ == '__main__':
    test_pipeline("image")
    test_pipeline("sentiment")
    print("Pipeline OK")
