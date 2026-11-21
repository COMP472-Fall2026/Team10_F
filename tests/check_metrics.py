import numpy as np

from src.metrics import accuracy, confusion_matrix, per_class_accuracy

y_true = np.array([0, 0, 1, 1, 2, 2])
y_pred = np.array([0, 1, 1, 1, 2, 0])

acc = accuracy(y_true, y_pred)
assert np.isclose(acc, 4 / 6), f"accuracy : {acc} instead of {4 / 6}"

cm = confusion_matrix(y_true, y_pred, 3)
expected_cm = np.array([[1, 1, 0],
                        [0, 2, 0],
                        [1, 0, 1]])
assert np.array_equal(cm, expected_cm), f"Wrong cm :\n{cm}"
assert cm.sum() == len(y_true), "metric doesn't count every example"

pca = per_class_accuracy(cm)
assert np.allclose(pca, [0.5, 1.0, 0.5]), f"per-class accuracy : {pca}"

print("Metrics OK")
