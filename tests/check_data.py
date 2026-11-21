import numpy as np

from src.data import load_dataset

## Check if load_dataset works

X_train, y_train, X_test, ids_train = load_dataset("image")
X_train2, y_train2, X_test2, ids_train2 = load_dataset("sentiment")

print(np.isnan(X_train).sum())
print(np.isnan(X_test).sum())

print(np.bincount(y_train))
print(np.bincount(y_train).size)

print(np.isnan(X_train2).sum())
print(np.isnan(X_test2).sum())

print(np.bincount(y_train2))
print(np.bincount(y_train2).size)
