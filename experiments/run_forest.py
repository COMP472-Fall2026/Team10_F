# ------------------------------------
# COMP 472 Project - Team 10_F
# Part: Random Forest (experiments)
# Written by: Zarin Haque 40317494
# ------------------------------------
# Runs the Random Forest experiments on one or both tasks:
#   1. accuracy against the number of trees (compared with the single Decision Tree)
#   2. accuracy against max_depth
#   3. accuracy against n_features (features tried at each split)
# Prints the results, saves the figures in figures/ and every result in results/forest_<task>.csv
# Usage: python -m experiments.run_forest            (both tasks)
#        python -m experiments.run_forest sentiment  (one task)

import os
import sys
import time

import matplotlib

matplotlib.use("Agg")  # Save figures to files without opening a window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd

from experiments.plot_hyperparams import BLUE, CHOSEN_DEPTH, NOISE, ORANGE, load_split, mark_choice
from src.metrics import accuracy
from src.models import decision_tree, random_forest
from src.models.random_forest import get_k

GREEN = "#3a9e5c"

# Experiment 1: one big forest, scored with its first n trees for every n (trained only once)
N_TREES = {"sentiment": [1, 5, 10, 25, 50, 75, 100],
           "image": [1, 5, 10, 25, 50, 100, 200, 300]}  # Image still improves at 100 trees, so we go further
# Experiments 2 and 3: one forest of SWEEP_TREES trees per value, the other settings at their default
SWEEP_TREES = 30
DEPTHS = {"sentiment": [5, 10, 15, 20, None], "image": [10, 20, 30, None]}
N_FEATURES = {"sentiment": ["log2", "sqrt", 0.1, 0.2], "image": ["log2", "sqrt", 0.1]}
# Default settings (the usual ones for classification: full-depth trees, sqrt of the features at each split)
DEFAULT_DEPTH = None
DEFAULT_N_FEATURES = "sqrt"
N_JOBS = None  # None = all CPU cores but one
FIGURE_NAMES = {"max_depth": "depth", "n_features": "features"}  # figures/forest_<name>_<task>.png


# Position of the first value (simplest/cheapest first) whose validation accuracy is within NOISE of the best
# (same rule as for k-NN and the Decision Tree: a smaller difference than the noise is not a real improvement)
def simplest_within_noise(values, accs):
    best = max(accs)
    for i, acc in enumerate(accs):
        if acc >= best - NOISE:
            return i


# Forest made of the first n trees of a bigger forest (same trees, nothing is retrained)
def first_trees(forest, n):
    return {"trees": forest["trees"][:n], "oob_idx": forest["oob_idx"][:n], "n_classes": forest["n_classes"],
            "params": forest["params"]}


# Train, validation and OOB accuracy of a whole forest, and its prediction time on the validation set
def evaluate(forest, X_tr, y_tr, X_val, y_val):
    start = time.time()
    val = accuracy(y_val, random_forest.predict(forest, X_val))
    predict_s = time.time() - start
    train = accuracy(y_tr, random_forest.predict(forest, X_tr))
    oob = random_forest.oob_score_curve(forest, X_tr, y_tr, [len(forest["trees"])])[0][0]
    return {"train": train, "val": val, "oob": oob, "predict_s": predict_s}


# Experiment 1: accuracy against the number of trees, compared with the single Decision Tree
# Returns the big forest (reused by the other experiments), its build time and the result rows
def n_trees_experiment(name, X_tr, y_tr, X_val, y_val, n_classes, ns):
    print(f"--- {name} : forest of {ns[-1]} trees (max_depth = {DEFAULT_DEPTH}, n_features = {DEFAULT_N_FEATURES})")
    start = time.time()
    forest = random_forest.build_forest(X_tr, y_tr, n_classes, n_trees=ns[-1], max_depth=DEFAULT_DEPTH,
                                        n_features=DEFAULT_N_FEATURES, n_jobs=N_JOBS)
    build_s = time.time() - start

    train_votes = random_forest.get_all_votes(forest, X_tr)
    val_votes = random_forest.get_all_votes(forest, X_val)
    train_accs = random_forest.score_curve(train_votes, y_tr, ns, n_classes)
    val_accs = random_forest.score_curve(val_votes, y_val, ns, n_classes)
    oob_accs, coverage = random_forest.oob_score_curve(forest, X_tr, y_tr, ns)

    rows = []
    for n, tr, va, oob, cov in zip(ns, train_accs, val_accs, oob_accs, coverage):
        print(f"--- {name} : n_trees = {n:3d} | train = {tr:.4f} | val = {va:.4f} | oob = {oob:.4f} (covers {100 * cov:.0f} % of rows)")
        rows.append({"experiment": "n_trees", "n_trees": n, "max_depth": DEFAULT_DEPTH, "n_features": DEFAULT_N_FEATURES,
                     "train": tr, "val": va, "oob": oob, "build_s": build_s * n / ns[-1]})

    # Single Decision Tree with the depth chosen in run_tree.py, for the comparison
    depth = CHOSEN_DEPTH[name]
    start = time.time()
    tree = decision_tree.build_tree(X_tr, y_tr, n_classes, max_depth=depth)
    tree_s = time.time() - start
    tree_train = accuracy(y_tr, random_forest.predict_tree(tree, X_tr))
    tree_val = accuracy(y_val, random_forest.predict_tree(tree, X_val))
    print(f"--- {name} : single Decision Tree (max_depth = {depth}) | train = {tree_train:.4f} | val = {tree_val:.4f} | {tree_s:.1f} s")
    rows.append({"experiment": "single_tree", "n_trees": 1, "max_depth": depth, "n_features": None,
                 "train": tree_train, "val": tree_val, "oob": np.nan, "build_s": tree_s})

    # Validation predictions of the whole forest, for the error analysis
    np.save(f"results/forest_val_preds_{name}.npy", random_forest.majority_vote(val_votes, n_classes))

    plot_n_trees(name, ns, train_accs, val_accs, oob_accs, tree_train, tree_val, depth)
    return forest, build_s, rows


# Experiments 2 and 3: one forest of SWEEP_TREES trees for each value of param ("max_depth" or "n_features")
# The forest with the default settings is not retrained: it is the first SWEEP_TREES trees of the big forest
def sweep_experiment(name, X_tr, y_tr, X_val, y_val, n_classes, big_forest, big_build_s, param, values):
    rows = []
    for value in values:
        depth = value if param == "max_depth" else DEFAULT_DEPTH
        n_features = value if param == "n_features" else DEFAULT_N_FEATURES
        if depth == DEFAULT_DEPTH and n_features == DEFAULT_N_FEATURES:
            forest = first_trees(big_forest, SWEEP_TREES)
            build_s = big_build_s * SWEEP_TREES / len(big_forest["trees"])  # Its share of the build time
        else:
            start = time.time()
            forest = random_forest.build_forest(X_tr, y_tr, n_classes, n_trees=SWEEP_TREES, max_depth=depth,
                                                n_features=n_features, n_jobs=N_JOBS, verbose=False)
            build_s = time.time() - start
        result = evaluate(forest, X_tr, y_tr, X_val, y_val)
        print(f"--- {name} : {param} = {str(value):5s} | train = {result['train']:.4f} | val = {result['val']:.4f} "
              f"| oob = {result['oob']:.4f} | build {build_s:.1f} s")
        rows.append({"experiment": param, "n_trees": SWEEP_TREES, "max_depth": depth, "n_features": n_features,
                     "build_s": build_s, **result})
    return rows


# Figure: train, validation and OOB accuracy against the number of trees,
# with the single Decision Tree as horizontal lines
def plot_n_trees(name, ns, train_accs, val_accs, oob_accs, tree_train, tree_val, depth):
    plt.figure(figsize=(8, 5))
    plt.plot(ns, train_accs, color=BLUE, linewidth=2, marker="o", markersize=5, label="forest - train")
    plt.plot(ns, val_accs, color=ORANGE, linewidth=2, marker="o", markersize=5, label="forest - validation")
    plt.plot(ns, oob_accs, "--", color=GREEN, linewidth=2, marker="o", markersize=5, label="forest - out-of-bag")
    plt.axhline(tree_train, color=BLUE, linestyle=":", linewidth=1.5, label=f"single tree (depth {depth}) - train")
    plt.axhline(tree_val, color=ORANGE, linestyle=":", linewidth=1.5, label=f"single tree (depth {depth}) - validation")

    plt.xscale("log")
    plt.xticks(ns, [str(n) for n in ns])
    i = simplest_within_noise(ns, val_accs)
    mark_choice(ns[i], val_accs[i], f"chosen : {ns[i]} trees ({100 * val_accs[i]:.1f} %)", max(val_accs))

    plt.xlabel("number of trees (log scale)")
    plt.ylabel("accuracy")
    plt.title(f"Random Forest - {name} : accuracy against the number of trees")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"figures/forest_ntrees_{name}.png", dpi=120)
    plt.close()


# Figure: train, validation and OOB accuracy for every value of param, with the chosen value circled
# (the x axis shows the values as text, since None and "sqrt" are not numbers, with the build time under each)
def plot_sweep(name, param, rows, n_feats):
    values = [row[param] for row in rows]
    labels = []
    for row in rows:
        value = row[param]
        if param == "n_features":
            value = f"{value} ({get_k(value, n_feats)})"  # e.g. "sqrt (22)"
        labels.append(f"{value}\n{row['build_s']:.0f} s")
    xs = list(range(len(rows)))
    val_accs = [row["val"] for row in rows]

    plt.figure(figsize=(8, 5))
    plt.plot(xs, [row["train"] for row in rows], color=BLUE, linewidth=2, marker="o", markersize=5, label="train")
    plt.plot(xs, val_accs, color=ORANGE, linewidth=2, marker="o", markersize=5, label="validation")
    plt.plot(xs, [row["oob"] for row in rows], "--", color=GREEN, linewidth=2, marker="o", markersize=5, label="out-of-bag")
    i = simplest_within_noise(values, val_accs)
    mark_choice(xs[i], val_accs[i], f"chosen : {param} = {values[i]} ({100 * val_accs[i]:.1f} %)", max(val_accs))

    plt.xticks(xs, labels)
    xlabel = "max_depth (None = grow until the leaves are pure)" if param == "max_depth" else "n_features (features tried at each split)"
    plt.xlabel(f"{xlabel}, with the build time of {SWEEP_TREES} trees")
    plt.ylabel("accuracy")
    plt.title(f"Random Forest - {name} : {param} ({SWEEP_TREES} trees)")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(f"figures/forest_{FIGURE_NAMES[param]}_{name}.png", dpi=120)
    plt.close()


# Runs the 3 experiments on one task
def run_forest(name):
    X_tr, y_tr, X_val, y_val, n_classes = load_split(name)
    n_feats = X_tr.shape[1]
    print(f"--- {name} : train {X_tr.shape}, val {X_val.shape}, {n_classes} classes")

    start = time.time()
    big_forest, big_build_s, rows = n_trees_experiment(name, X_tr, y_tr, X_val, y_val, n_classes, N_TREES[name])
    print(f"--- {name} : experiment 1 done in {time.time() - start:.0f} s\n")

    for param, values in [("max_depth", DEPTHS[name]), ("n_features", N_FEATURES[name])]:
        sweep_rows = sweep_experiment(name, X_tr, y_tr, X_val, y_val, n_classes, big_forest, big_build_s, param, values)
        plot_sweep(name, param, sweep_rows, n_feats)
        rows += sweep_rows
        print()

    pd.DataFrame(rows).to_csv(f"results/forest_{name}.csv", index=False)
    print(f"--- {name} : results saved in results/forest_{name}.csv, figures in figures/forest_*_{name}.png\n")


if __name__ == "__main__":  # Runs the experiments
    os.makedirs("figures", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    names = sys.argv[1:] or ["sentiment", "image"]  # e.g. python -m experiments.run_forest sentiment
    for name in names:
        run_forest(name)
