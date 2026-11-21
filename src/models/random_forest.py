# ------------------------------------
# COMP 472 Project - Team 10_F
# Part: Random Forest
# Written by: Zarin Haque 40317494
# ------------------------------------
# Random Forest built on our Decision Tree (src/models/decision_tree.py).
# Each tree is trained on a bootstrap sample of the training data and only looks at
# n_features random features at each split. The forest predicts with a majority vote of its trees.

import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from seeds import SEED
from src.models.decision_tree import build_tree


# Number of features k tried at each split ('sqrt', 'log2', a fraction like 0.1, an int, or None for all)
def get_k(n_features, n_feats):
    if n_features is None:
        k = n_feats
    elif n_features == "sqrt":
        k = int(np.sqrt(n_feats))
    elif n_features == "log2":
        k = int(np.log2(n_feats))
    elif isinstance(n_features, float):
        k = int(n_features * n_feats)  # Fraction of the features
    else:
        k = int(n_features)
    return max(1, min(n_feats, k))  # Keeping k between 1 and n_feats


# Draw n rows WITH replacement (some rows are drawn twice, some never)
# The rows never drawn are the out-of-bag (OOB) rows of the tree
def bootstrap_sample(X, y, rng, max_samples=1.0):
    n = len(X)
    idx = rng.integers(0, n, size=round(max_samples * n))
    oob_idx = np.setdiff1d(np.arange(n), idx)  # Rows not in idx
    return X[idx], y[idx], oob_idx


# Build one tree of the forest from its own seed
# (tree number t always gets the same seed, whatever the number of trees or processes)
def build_one_tree(X, y, n_classes, seed, max_depth, min_samples_split, n_thresholds, n_features,
                   bootstrap, max_samples):
    rng = np.random.default_rng(seed)
    if bootstrap:
        X_boot, y_boot, oob_idx = bootstrap_sample(X, y, rng, max_samples)
    else:  # No bootstrap: the tree sees every row, so it has no OOB rows
        X_boot, y_boot, oob_idx = X, y, np.array([], dtype=int)
    tree = build_tree(X_boot, y_boot, n_classes, max_depth=max_depth, min_samples_split=min_samples_split,
                      n_thresholds=n_thresholds, n_features=n_features, rng=rng)
    return tree, oob_idx


# ----- Parallel training -----
# Each worker process receives X and y only once, then builds trees from their seeds
_X, _y = None, None


def _init_worker(X, y):
    global _X, _y
    _X, _y = X, y


def _build_in_worker(job):
    seed, params = job
    return build_one_tree(_X, _y, seed=seed, **params)


# Train n_trees trees and return the forest as a dict
# n_jobs: number of trees trained at the same time (None = all CPU cores but one, 1 = one tree at a time)
def build_forest(X, y, n_classes, n_trees=50, max_depth=None, min_samples_split=2, n_thresholds=10,
                 n_features="sqrt", bootstrap=True, max_samples=1.0, seed=SEED, n_jobs=None, verbose=True):
    n_feats = X.shape[1]
    k = get_k(n_features, n_feats)
    params = {"max_depth": max_depth, "min_samples_split": min_samples_split, "n_thresholds": n_thresholds,
              "n_features": None if k == n_feats else k,  # All features = normal tree, no random features
              "bootstrap": bootstrap, "max_samples": max_samples}

    # One seed per tree, made from the forest's seed
    tree_seeds = np.random.SeedSequence(seed).spawn(n_trees)
    jobs = [(tree_seed, {"n_classes": n_classes, **params}) for tree_seed in tree_seeds]
    if n_jobs is None:
        n_jobs = max(1, (os.cpu_count() or 2) - 1)  # All CPU cores but one

    start = time.time()
    results = []
    if n_jobs == 1:
        _init_worker(X, y)
        outputs = map(_build_in_worker, jobs)
    else:
        pool = ProcessPoolExecutor(max_workers=n_jobs, initializer=_init_worker, initargs=(X, y))
        outputs = pool.map(_build_in_worker, jobs)  # Results come back in the order of the trees
    for t, result in enumerate(outputs, start=1):
        results.append(result)
        if verbose and (t % 10 == 0 or t == n_trees):
            print(f"      {t}/{n_trees} trees, {time.time() - start:.1f} s")
    if n_jobs != 1:
        pool.shutdown()

    return {"trees": [tree for tree, _ in results],
            "oob_idx": [oob_idx for _, oob_idx in results],  # OOB rows of every tree
            "n_classes": n_classes,
            "params": params}


# Predict every row of X with one tree
# Same result as decision_tree.predict(), but faster: all the rows go down the tree together
def predict_tree(tree, X):
    y_pred = np.empty(len(X), dtype=int)

    def send_down(node, rows):
        if len(rows) == 0:  # No rows reach this node
            return
        if node["leaf"]:
            y_pred[rows] = node["class"]
            return
        go_left = X[rows, node["feature"]] <= node["threshold"]
        send_down(node["left"], rows[go_left])
        send_down(node["right"], rows[~go_left])

    send_down(tree, np.arange(len(X)))
    return y_pred


# Prediction of every tree for every row: array (n_trees, number of rows)
def get_all_votes(forest, X):
    return np.array([predict_tree(tree, X) for tree in forest["trees"]])


# Majority vote of the trees for every row (ties go to the smallest class)
def majority_vote(all_votes, n_classes):
    n_rows = all_votes.shape[1]
    counts = np.zeros((n_rows, n_classes), dtype=int)  # Votes of every row for every class
    for votes in all_votes:  # One tree at a time
        counts[np.arange(n_rows), votes] += 1
    return np.argmax(counts, axis=1)


# Predict every row of X with the whole forest
def predict(forest, X):
    return majority_vote(get_all_votes(forest, X), forest["n_classes"])


# Accuracy of the forest made of its first n trees, for every n in ns
# (the trees are independent, so the first 10 trees of a 100-tree forest ARE a 10-tree forest)
def score_curve(all_votes, y, ns, n_classes):
    n_rows = all_votes.shape[1]
    counts = np.zeros((n_rows, n_classes), dtype=int)
    scores = []
    for t, votes in enumerate(all_votes, start=1):  # Adding the votes one tree at a time
        counts[np.arange(n_rows), votes] += 1
        if t in ns:
            scores.append(np.mean(np.argmax(counts, axis=1) == y))
    return scores


# OOB accuracy of the forest made of its first n trees, for every n in ns
# Each training row is only predicted by the trees that did NOT see it (about 37% of the trees)
# X and y must be the same data that was given to build_forest()
# Also returns coverage: the fraction of rows that already have at least one OOB vote
def oob_score_curve(forest, X, y, ns):
    counts = np.zeros((len(y), forest["n_classes"]), dtype=int)  # OOB votes of every row for every class
    scores, coverage = [], []
    for t, (tree, oob_idx) in enumerate(zip(forest["trees"], forest["oob_idx"]), start=1):
        counts[oob_idx, predict_tree(tree, X[oob_idx])] += 1  # Each tree only votes for its OOB rows
        if t in ns:
            has_vote = counts.sum(axis=1) > 0  # Rows with at least one OOB vote
            scores.append(np.mean(np.argmax(counts[has_vote], axis=1) == y[has_vote]))
            coverage.append(has_vote.mean())
    return scores, coverage
