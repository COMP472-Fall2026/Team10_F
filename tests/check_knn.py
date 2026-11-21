import numpy as np

from src.models.knn import distances_to_one, predict

X_ref = np.array([[1, 1],
                  [2, 1],
                  [4, 4],
                  [5, 4],
                  [4, 2]])
y_ref = np.array([0, 0, 1, 1, 1])
q = np.array([2, 2])
X_ref_copy = X_ref.copy()

dist = distances_to_one(X_ref, q)
assert np.array_equal(dist, [2, 1, 8, 13, 4]), f"distances : {dist}"
assert np.array_equal(X_ref, X_ref_copy), "distances_to_one modified X_ref"

for k, expected in [(1, 0), (3, 0), (5, 1)]:
    pred = predict(X_ref, y_ref, [q], k, 2, False)
    assert np.array_equal(pred, [expected]), f"k={k} : predicted {pred} instead of {[expected]}"

print("k-NN OK")
