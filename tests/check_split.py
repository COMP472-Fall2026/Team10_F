import numpy as np

from src.data import load_dataset
from src.split import stratified_split

def check_stratified_split(name):
    """Check that the split covers every example once, keeps the class balance and is reproducible."""
    X_train, y_train, X_test, ids_test = load_dataset(name)
    train_idx, val_idx = stratified_split(y_train)
    assert len(train_idx) + len(val_idx) == len(y_train), "Wrong split"
    assert np.intersect1d(train_idx, val_idx).size == 0, "Wrong split"
    bincount = np.bincount(y_train[val_idx])
    assert bincount.min() == bincount.max(), "Wrong split"
    train_idx2, val_idx2 = stratified_split(y_train)
    assert np.array_equal(train_idx, train_idx2) and np.array_equal(val_idx, val_idx2), "Wrong split"
    print("Split " + name + " ok")


check_stratified_split("image")
check_stratified_split("sentiment")