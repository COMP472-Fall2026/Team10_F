# ------------------------------------
# COMP 472 Project - Team 10_F
# Part: Random Forest (tests)
# Written by: Zarin Haque 40317494
# ------------------------------------
# Tests of the Random Forest on small made-up data where the right answer is known.
# Prints "Random Forest OK" if every test passes, otherwise stops at the failing test.

import numpy as np

from src.models import decision_tree
from src.models.random_forest import (
    bootstrap_sample,
    build_forest,
    get_all_votes,
    get_k,
    majority_vote,
    oob_score_curve,
    predict,
    predict_tree,
    score_curve,
)


# Made-up data: class = 1 when feature 0 + feature 1 > 0, the other features are noise
def toy_data(n=300, n_feats=6, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_feats))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


def check_all():
    X, y = toy_data()

    # Testing decision_tree with n_features
    assert decision_tree.build_tree(X, y, 2, max_depth=4) == decision_tree.build_tree(X, y, 2, max_depth=4, n_features=None), \
        "n_features=None should not change the Decision Tree"
    tree_1 = decision_tree.build_tree(X, y, 2, max_depth=1, n_features=1, rng=np.random.default_rng(0))
    assert tree_1["feature"] == np.random.default_rng(0).choice(6, size=1, replace=False)[0], \
        "the root should use the one feature drawn by the rng"

    # Testing get_k
    assert get_k("sqrt", 500) == 22 and get_k("sqrt", 768) == 27, "sqrt"
    assert get_k("log2", 768) == 9, "log2"
    assert get_k(0.1, 500) == 50, "fraction"
    assert get_k(None, 500) == 500 and get_k(1000, 500) == 500, "None / too big"

    # Testing bootstrap_sample: about 63% of the rows are used, 37% are out-of-bag
    X_big = np.arange(10000).reshape(-1, 1)
    X_boot, y_boot, oob_idx = bootstrap_sample(X_big, X_big[:, 0], np.random.default_rng(0))
    assert len(X_boot) == 10000 and len(np.unique(X_boot)) < 10000, "the bootstrap must draw with replacement"
    assert np.array_equal(X_boot[:, 0], y_boot), "X_boot and y_boot must use the same rows"
    assert len(np.intersect1d(oob_idx, X_boot[:, 0])) == 0, "an OOB row can't be in the sample"
    assert len(oob_idx) + len(np.unique(X_boot)) == 10000, "every row is either drawn or OOB"
    assert 0.34 < len(oob_idx) / 10000 < 0.40, f"OOB fraction {len(oob_idx) / 10000:.3f} should be close to 1/e = 0.368"

    # Testing predict_tree: same result as decision_tree.predict()
    tree = decision_tree.build_tree(X, y, 2, max_depth=6)
    assert np.array_equal(predict_tree(tree, X), decision_tree.predict(tree, X)), "predict_tree != decision_tree.predict"

    # Testing majority_vote (ties go to the smallest class)
    all_votes = np.array([[0, 1, 2], [0, 1, 1], [1, 2, 0]])
    assert np.array_equal(majority_vote(all_votes, 3), [0, 1, 0]), f"majority_vote : {majority_vote(all_votes, 3)}"

    # 1 tree, no bootstrap, all features: the forest is exactly the Decision Tree
    forest_1 = build_forest(X, y, 2, n_trees=1, max_depth=6, n_features=None, bootstrap=False, n_jobs=1, verbose=False)
    assert forest_1["trees"][0] == tree, "1 tree, no bootstrap, all features should be the Decision Tree"

    # A real forest learns the made-up problem
    forest = build_forest(X, y, 2, n_trees=20, max_depth=None, n_jobs=1, verbose=False, seed=1)
    X_new, y_new = toy_data(seed=1)
    acc = np.mean(predict(forest, X_new) == y_new)
    assert acc > 0.85, f"forest accuracy on new data is only {acc:.3f}"

    # Same seed = same forest
    again = build_forest(X, y, 2, n_trees=20, max_depth=None, n_jobs=1, verbose=False, seed=1)
    assert np.array_equal(get_all_votes(forest, X_new), get_all_votes(again, X_new)), "same seed = same forest"

    # The trees are all different (bootstrap + random features)
    assert len({str(t) for t in forest["trees"]}) == 20, "all the trees should be different"

    # Parallel training gives exactly the same forest
    parallel = build_forest(X, y, 2, n_trees=20, max_depth=None, n_jobs=2, verbose=False, seed=1)
    assert np.array_equal(get_all_votes(forest, X_new), get_all_votes(parallel, X_new)), "n_jobs changed the result"

    # The first 5 trees of a 20-tree forest are the 5-tree forest
    small = build_forest(X, y, 2, n_trees=5, max_depth=None, n_jobs=1, verbose=False, seed=1)
    assert np.array_equal(get_all_votes(small, X_new), get_all_votes(forest, X_new)[:5]), "tree t depends on n_trees"

    # Testing score_curve: the last value is the accuracy of the whole forest
    scores = score_curve(get_all_votes(forest, X_new), y_new, [1, 5, 20], 2)
    assert len(scores) == 3 and np.isclose(scores[-1], acc), "score_curve"

    # Testing oob_score_curve: lower than the train accuracy, close to the accuracy on new data
    oob_scores, coverage = oob_score_curve(forest, X, y, [5, 20])
    train_acc = np.mean(predict(forest, X) == y)
    assert coverage[-1] > 0.99, f"after 20 trees almost every row should have an OOB vote ({coverage[-1]:.3f})"
    assert oob_scores[-1] < train_acc and abs(oob_scores[-1] - acc) < 0.08, \
        f"OOB {oob_scores[-1]:.3f}, train {train_acc:.3f}, new data {acc:.3f}"

    print("Random Forest OK")


if __name__ == "__main__":  # Runs the tests
    check_all()
