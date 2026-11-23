import os

import matplotlib

matplotlib.use("Agg")  # Save figures to files without opening a window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import torch

from experiments.run_nn import scale_features
from make_submission import NN_SETTINGS
from src.data import load_dataset
from src.metrics import accuracy, confusion_matrix, per_class_accuracy
from src.models import decision_tree, knn, neural_network, random_forest
from src.split import stratified_split


def predict_val(name, model, **params):
    """Split the dataset, train the model on the train part only and predict the validation set.

    model  : "knn" (params k, weighted), "tree" (param max_depth),
             "forest" (params n_trees, max_depth, n_features)
             or "mlp" (param scaling + the settings of neural_network.fit)
    Returns: (y_val, y_pred), two arrays of length n_val
    """
    X_train, y_train, _, _ = load_dataset(name)
    n_classes = len(np.unique(y_train))

    train_idx, val_idx = stratified_split(y_train)
    X_tr, y_tr = X_train[train_idx], y_train[train_idx]
    X_val, y_val = X_train[val_idx], y_train[val_idx]

    if model == "knn":
        y_pred = knn.predict(X_tr, y_tr, X_val, k=params["k"], n_classes=n_classes, weighted=params["weighted"])
    elif model == "tree":
        tree = decision_tree.build_tree(X_tr, y_tr, n_classes, max_depth=params["max_depth"])
        y_pred = decision_tree.predict(tree, X_val)
    elif model == "forest":
        forest = random_forest.build_forest(X_tr, y_tr, n_classes, n_trees=params["n_trees"], max_depth=params["max_depth"],
                                            n_features=params["n_features"], verbose=False)
        y_pred = random_forest.predict(forest, X_val)
    elif model == "mlp":
        X_tr_scaled, X_val_scaled = scale_features(X_tr, X_val, params.pop("scaling"))
        X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
        network, _ = neural_network.fit(torch.tensor(X_tr_scaled, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.long),
                                        X_val_tensor, torch.tensor(y_val, dtype=torch.long), **params)
        y_pred = neural_network.predict(network, X_val_tensor)
    else:
        raise ValueError(f"unknown model : {model} (expected 'knn', 'tree', 'forest' or 'mlp')")
    return y_val, y_pred


def top_confusions(cm, n):
    """Find the n largest off-diagonal cells of the confusion matrix.

    Returns: list of n tuples (true_class, predicted_class, count), largest first
    """
    off = cm.copy()
    np.fill_diagonal(off, -1)  # -1 so that diagonal cells always come last and can be removed
    flat = np.argsort(off, axis=None)[::-1]
    rows, cols = np.unravel_index(flat, off.shape)
    confusions = [(int(r), int(c), int(off[r, c])) for r, c in zip(rows, cols) if r != c]
    return confusions[:n]


def save_confusion_plot(cm, title, path):
    """Draw the confusion matrix as a colored image and save it as a PNG.
    Each row is converted to percentages of its true class, so colors can be compared between classes.
    Small matrices (<= 10 classes) also show the count and the percentage in every cell.

    Returns: nothing
    """
    n_classes = cm.shape[0]
    percent = 100 * cm / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(6, 5) if n_classes <= 10 else (9, 8))
    # big matrices: the scale stops at the largest cell, otherwise every cell looks almost white
    plt.imshow(percent, cmap="Blues", vmin=0, vmax=100 if n_classes <= 10 else percent.max())
    plt.colorbar(label="% of the true class")

    if n_classes <= 10:
        for i in range(n_classes):
            for j in range(n_classes):
                color = "white" if percent[i, j] > 50 else "#0b0b0b"
                plt.text(j, i, f"{cm[i, j]}\n{percent[i, j]:.0f} %", ha="center", va="center", color=color, fontsize=11)
        plt.xticks(range(n_classes))
        plt.yticks(range(n_classes))
    else:
        ticks = range(0, n_classes, 5)
        plt.xticks(ticks)
        plt.yticks(ticks)

    plt.xlabel("predicted class")
    plt.ylabel("true class")
    plt.title(title + "\n(diagonal = correct predictions)")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def analyze(name, model, n_show=5, **params):
    """Print the accuracy, the per-class accuracy and the main confusions of a model on the validation set.

    model  : "knn" (params k, weighted), "tree" (param max_depth),
             "forest" (params n_trees, max_depth, n_features)
             or "mlp" (param scaling + the settings of neural_network.fit)
    Returns: nothing, prints the results and saves the confusion matrix plot in figures/
    """
    y_val, y_pred = predict_val(name, model, **params)
    n_classes = len(np.unique(y_val))
    cm = confusion_matrix(y_val, y_pred, n_classes)
    pca = per_class_accuracy(cm)

    print(f"===== {name} : {model} {params} | val accuracy = {accuracy(y_val, y_pred):.4f}")

    if n_classes <= 5:
        print("confusion matrix (rows = true class, columns = predicted class) :")
        print(cm)
        print("per-class accuracy :", np.round(pca, 3))
    else:
        order = np.argsort(pca)
        print(f"{n_show} worst classes :", [(int(c), round(float(pca[c]), 3)) for c in order[:n_show]])
        print(f"{n_show} best classes  :", [(int(c), round(float(pca[c]), 3)) for c in order[::-1][:n_show]])

    print("most frequent confusions (true -> predicted : count) :")
    for true_c, pred_c, count in top_confusions(cm, 2 * n_show):
        print(f"   {true_c:2d} -> {pred_c:2d} : {count}")

    os.makedirs("figures", exist_ok=True)
    save_confusion_plot(cm, f"{model} confusion matrix - {name}", f"figures/confusion_{model}_{name}.png")
    print()


if __name__ == "__main__":
    analyze("sentiment", "knn", k=61, weighted=True)
    analyze("image", "knn", k=1, weighted=True)
    analyze("sentiment", "tree", max_depth=5)
    analyze("image", "tree", max_depth=40)
    # Random Forest: settings chosen in run_forest.py (a few minutes each)
    analyze("sentiment", "forest", n_trees=100, max_depth=None, n_features="sqrt")
    analyze("image", "forest", n_trees=300, max_depth=None, n_features="sqrt")
    # MLP: settings chosen in run_nn.py, shared with make_submission.py
    analyze("sentiment", "mlp", **NN_SETTINGS["sentiment"])
    analyze("image", "mlp", **NN_SETTINGS["image"])
