import numpy as np

from src.models.decision_tree import (
    best_split,
    build_tree,
    gini,
    predict,
    predict_one,
    split_score,
)

X = np.array([[1, 4], [2, 1], [3, 5], [5, 2], [6, 6], [7, 7]])
y = np.array([0, 0, 1, 0, 1, 1])

# gini / split_score
assert np.isclose(gini(np.array([0, 0, 0, 1, 1, 1]), 2), 0.5), "gini of a 50/50 group"
assert np.isclose(gini(np.array([1, 1]), 2), 0.0), "gini of a pure group"
assert np.isclose(split_score(np.array([0, 0]), np.array([1, 0, 1, 1]), 2), 0.25), "split_score"

# best_split
feature, threshold, score = best_split(X, y, 2, n_thresholds=4)
assert feature == 1 and 4 <= threshold < 5 and score == 0, f"best_split : {feature, threshold, score}"

X_const = np.array([[0, 4], [0, 1], [0, 5], [0, 2]])
y_const = np.array([0, 0, 1, 0])
assert best_split(X_const, y_const, 2, n_thresholds=4)[2] == 0, "best_split misses the perfect cut"
assert best_split(np.array([[0], [0]]), np.array([0, 1]), 2)[0] is None, "constant feature should give None"

# build_tree
tree = build_tree(X, y, 2, n_thresholds=4)
assert tree["leaf"] is False and tree["feature"] == 1, f"root : {tree}"
assert tree["left"] == {"leaf": True, "class": 0}, f"left : {tree['left']}"
assert tree["right"] == {"leaf": True, "class": 1}, f"right : {tree['right']}"
assert build_tree(X, np.ones(6, dtype=int), 2)["leaf"], "a pure group should be a leaf"
assert build_tree(X, y, 2, max_depth=0)["leaf"], "max_depth=0 should be a leaf"

# predict_one / predict
assert predict_one(tree, np.array([4, 3])) == 0, "predict_one (4, 3)"
assert predict_one(tree, np.array([4, 9])) == 1, "predict_one (4, 9)"
pred = predict(tree, X)
assert np.array_equal(pred, y), f"predict : {pred} instead of {y}"

print("Decision Tree OK")
