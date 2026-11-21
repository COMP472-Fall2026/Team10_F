import os

import matplotlib.pyplot as plt
import numpy as np

from src.data import load_dataset
from src.metrics import accuracy
from src.models import decision_tree, knn
from src.preprocessing import fit_standardizer, standardize
from src.split import stratified_split

BLUE = "#2a78d6"
ORANGE = "#eb6834"
GRAY = "#52514e"
NOISE = 0.01  # validation noise: about +/- 1 point of accuracy

# hyperparameters chosen on the validation set (see run_knn.py and run_tree.py)
CHOSEN_KNN = {"image": {"k": 1, "features": "raw", "vote": "weighted"},
              "sentiment": {"k": 61, "features": "raw", "vote": "weighted"}}
CHOSEN_DEPTH = {"image": 40, "sentiment": 5}


def load_split(name):
    """Load the dataset and split it with the project seed.

    Returns: (X_tr, y_tr, X_val, y_val, n_classes)
    """
    X_train, y_train, _, _ = load_dataset(name)
    n_classes = len(np.unique(y_train))
    train_idx, val_idx = stratified_split(y_train)
    return X_train[train_idx], y_train[train_idx], X_train[val_idx], y_train[val_idx], n_classes


def knn_curves(name, ks):
    """Validation accuracy of k-NN for every k, with raw or standardized features and simple or weighted vote.
    The neighbors are computed only once per feature version, for the largest k.

    Returns: dict {(features, vote): list of accuracies, one per k}
    """
    X_tr, y_tr, X_val, y_val, n_classes = load_split(name)
    mean, std = fit_standardizer(X_tr)
    versions = {"raw": (X_tr, X_val),
                "standardized": (standardize(X_tr, mean, std), standardize(X_val, mean, std))}

    curves = {}
    for features, (X_ref, X_query) in versions.items():
        neighbors, distances = knn.find_neighbors(X_ref, X_query, max(ks))
        curves[(features, "vote")] = [accuracy(y_val, knn.vote(neighbors[:, :k], y_tr, n_classes)) for k in ks]
        curves[(features, "weighted")] = [
            accuracy(y_val, knn.weighted_vote(neighbors[:, :k], distances[:, :k], y_tr, n_classes)) for k in ks]
    return curves


def tree_curves(name, depths):
    """Train and validation accuracy of one decision tree per max_depth.

    Returns: (train_accs, val_accs), two lists with one accuracy per depth
    """
    X_tr, y_tr, X_val, y_val, n_classes = load_split(name)
    train_accs, val_accs = [], []
    for depth in depths:
        tree = decision_tree.build_tree(X_tr, y_tr, n_classes, max_depth=depth)
        train_accs.append(accuracy(y_tr, decision_tree.predict(tree, X_tr)))
        val_accs.append(accuracy(y_val, decision_tree.predict(tree, X_val)))
        print(f"   {name} : max_depth = {depth} done")
    return train_accs, val_accs


def mark_choice(x, y, text, best):
    """Draw the noise band around the best validation accuracy and highlight the chosen point.

    Returns: nothing
    """
    plt.axhspan(best - NOISE, best + NOISE, color=GRAY, alpha=0.12, label="best val ± 1 pt (noise)")
    plt.scatter([x], [y], s=160, facecolors="none", edgecolors="#0b0b0b", linewidths=2, zorder=5)
    ax = plt.gca()
    x_frac = ax.transAxes.inverted().transform(ax.transData.transform((x, y)))[0]
    on_right = x_frac > 0.6  # write the text on the left of the point when it is close to the right border
    plt.annotate(text, (x, y), textcoords="offset points", xytext=(-12 if on_right else 12, -22),
                 ha="right" if on_right else "left", fontsize=10, color="#0b0b0b")


def plot_knn(name, ks, curves, path):
    """Plot validation accuracy against k for the 4 k-NN versions and highlight the chosen k.

    Returns: nothing, saves the figure as a PNG
    """
    plt.figure(figsize=(8, 5))
    colors = {"vote": BLUE, "weighted": ORANGE}
    styles = {"raw": "-", "standardized": "--"}
    for (features, vote), accs in curves.items():
        plt.plot(ks, accs, styles[features], color=colors[vote], linewidth=2, marker="o", markersize=5,
                 label=f"{features} features, {'simple' if vote == 'vote' else 'weighted'} vote")

    plt.xscale("log")  # set the scale before mark_choice, which needs the final position of the point
    plt.xticks(ks, [str(k) for k in ks])

    chosen = CHOSEN_KNN[name]
    accs = curves[(chosen["features"], chosen["vote"])]
    best = max(max(a) for a in curves.values())
    y = accs[ks.index(chosen["k"])]
    mark_choice(chosen["k"], y, f"chosen : k = {chosen['k']} ({100 * y:.1f} %)", best)

    plt.xlabel("k (number of neighbors, log scale)")
    plt.ylabel("validation accuracy")
    plt.title(f"k-NN - {name} : validation accuracy for each k")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def plot_tree(name, depths, train_accs, val_accs, path):
    """Plot train and validation accuracy against max_depth and highlight the chosen depth.

    Returns: nothing, saves the figure as a PNG
    """
    plt.figure(figsize=(8, 5))
    plt.plot(depths, train_accs, color=BLUE, linewidth=2, marker="o", markersize=5, label="train")
    plt.plot(depths, val_accs, color=ORANGE, linewidth=2, marker="o", markersize=5, label="validation")

    depth = CHOSEN_DEPTH[name]
    y = val_accs[depths.index(depth)]
    mark_choice(depth, y, f"chosen : max_depth = {depth} ({100 * y:.1f} %)", max(val_accs))

    plt.xticks(depths)
    plt.xlabel("max_depth")
    plt.ylabel("accuracy")
    plt.title(f"Decision Tree - {name} : train vs validation accuracy")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    ks = [1, 3, 5, 11, 21, 31, 61, 101]
    for name in ["sentiment", "image"]:
        print(f"k-NN {name} ...")
        plot_knn(name, ks, knn_curves(name, ks), f"figures/knn_{name}.png")

    tree_depths = {"sentiment": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], "image": [10, 20, 30, 40, 60]}
    for name, depths in tree_depths.items():
        print(f"Decision Tree {name} (slow) ...")
        train_accs, val_accs = tree_curves(name, depths)
        plot_tree(name, depths, train_accs, val_accs, f"figures/tree_{name}.png")
