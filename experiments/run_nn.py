import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.data import load_dataset
from src.models.neural_network import fit
from src.preprocessing import fit_standardizer, standardize
from src.split import stratified_split


def scale_features(train_features, other_features, scaling):
    """
    Scale two sets of features the same way, using only train_features to compute the scaling.

    Args:
        train_features (np.ndarray): The features the model learns from (n, d).
        other_features (np.ndarray): The features it is evaluated on: validation or test (m, d).
        scaling (str): How to scale the features:
            "standardize": mean 0 and std 1 per feature, computed on train_features only.
            "scale": multiply every value by sqrt(d), the same constant for all values.

    Returns:
        tuple: (train_features, other_features), scaled.
    """
    if scaling == "standardize":
        mean, std = fit_standardizer(train_features)  # train only: no data leakage
        return standardize(train_features, mean, std), standardize(other_features, mean, std)
    if scaling == "scale":
        factor = np.sqrt(train_features.shape[1])
        return train_features * factor, other_features * factor
    raise ValueError(f'unknown scaling "{scaling}", expected "standardize" or "scale"')


def prepare_split(features, labels, scaling):
    """
    Split the dataset into training and validation sets, scale the features and convert everything to tensors.

    Args:
        features (np.ndarray): All the training features, as returned by load_dataset (n, d).
        labels (np.ndarray): The true label of each row of features (n,).
        scaling (str): "standardize" or "scale", see scale_features.

    Returns:
        tuple: (train_features, train_labels, validation_features, validation_labels) as tensors,
        in the order expected by fit, so it can be called as fit(*data).
    """
    train_idx, validation_idx = stratified_split(labels)
    train_features, validation_features = features[train_idx], features[validation_idx]
    train_labels, validation_labels = labels[train_idx], labels[validation_idx]

    train_features, validation_features = scale_features(train_features, validation_features, scaling)

    return (
        torch.tensor(train_features, dtype=torch.float32),
        torch.tensor(train_labels, dtype=torch.long),
        torch.tensor(validation_features, dtype=torch.float32),
        torch.tensor(validation_labels, dtype=torch.long),
    )


def run_experiment(runs, name, data, **settings):
    """
    Train one network with the given settings and store the result in runs[name].

    Args:
        runs (dict): Where the experiments of one task are stored, name -> {"settings", "history"}.
        name (str): A short name for the experiment, used in the table and the figure.
        data (tuple): (train_features, train_labels, validation_features, validation_labels), from prepare_split.
        **settings: The hyperparameters passed to fit (hidden_sizes, dropout, lr, weight_decay, epochs...).

    Returns:
        None, runs is updated in place.
    """
    _, history = fit(*data, **settings)
    runs[name] = {"settings": settings, "history": history}


def summarize(runs):
    """
    Build a table with one row per experiment: its settings and its best validation accuracy.

    Returns:
        pd.DataFrame: columns experiment, settings, best_validation_acc, best_epoch, train_acc_at_best.
    """
    rows = []
    for name, run in runs.items():
        validation_acc = run["history"]["validation_acc"]
        best = int(np.argmax(validation_acc))  # position of the best epoch (0 = first epoch)
        rows.append({
            "experiment": name,
            "settings": ", ".join(f"{key}={value}" for key, value in run["settings"].items()),
            "best_validation_acc": round(validation_acc[best], 4),
            "best_epoch": best + 1,
            "train_acc_at_best": round(run["history"]["train_acc"][best], 4),
        })
    return pd.DataFrame(rows)


def plot_validation_curves(runs, title, path):
    """
    Draw the validation accuracy of every experiment on the same figure and save it to path.

    Returns:
        None, the figure is written to path.
    """
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for name, run in runs.items():
        validation_acc = run["history"]["validation_acc"]
        ax.plot(range(1, len(validation_acc) + 1), validation_acc, label=name)
    ax.set(title=title, xlabel="epoch", ylabel="validation accuracy")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def report(runs, title, path):
    """Print the table of results and save the figure of the validation curves."""
    print(f"--- {title}")
    print(summarize(runs).to_string(index=False))
    plot_validation_curves(runs, title, path)
    print(f"--- figure saved to {path}\n")


def run_sentiment():
    """
    Sentiment experiments: effect of the learning rate, then of regularization and of the network size.

    Returns:
        None, prints two tables and saves two figures.
    """
    features, labels, _, _ = load_dataset("sentiment")
    data = prepare_split(features, labels, "standardize")

    lr_runs = {}
    for lr in [1e-1, 1e-3, 1e-5]:
        run_experiment(lr_runs, f"lr={lr:g}", data, lr=lr)
    report(lr_runs, "sentiment: learning rate", "figures/nn_sentiment_lr.png")

    runs = {}
    run_experiment(runs, "baseline", data, lr=1e-3)
    run_experiment(runs, "lr 1e-4", data, lr=1e-4)
    run_experiment(runs, "lr 1e-4 + dropout 0.3", data, lr=1e-4, dropout=0.3)
    run_experiment(runs, "lr 1e-4 + dropout 0.5", data, lr=1e-4, dropout=0.5)
    run_experiment(runs, "lr 1e-4 + wd 1e-3", data, lr=1e-4, weight_decay=1e-3)
    run_experiment(runs, "lr 1e-4 + wd 1e-2", data, lr=1e-4, weight_decay=1e-2)
    run_experiment(runs, "lr 1e-4 + dropout 0.5 + wd 1e-2", data, lr=1e-4, dropout=0.5, weight_decay=1e-2)
    run_experiment(runs, "lr 1e-4 + wd 1e-2, [512, 128]", data, hidden_sizes=[512, 128], lr=1e-4, weight_decay=1e-2)
    run_experiment(runs, "lr 1e-4 + wd 1e-2, [64]", data, hidden_sizes=[64], lr=1e-4, weight_decay=1e-2)
    report(runs, "sentiment: regularization and network size", "figures/nn_sentiment.png")


def run_image():
    """
    Image experiments: standardized vs scaled features, then bigger networks.

    Returns:
        None, prints a table and saves a figure.
    """
    features, labels, _, _ = load_dataset("image")
    standardized = prepare_split(features, labels, "standardize")
    scaled = prepare_split(features, labels, "scale")

    runs = {}
    run_experiment(runs, "standardize [256]", standardized, hidden_sizes=[256], lr=1e-3)
    run_experiment(runs, "standardize [1024]", standardized,
                   hidden_sizes=[1024], dropout=0.5, lr=1e-3, weight_decay=1e-3, epochs=60)
    run_experiment(runs, "scale [1024]", scaled,
                   hidden_sizes=[1024], dropout=0.5, lr=1e-3, weight_decay=1e-3, epochs=60)
    run_experiment(runs, "scale [2048]", scaled,
                   hidden_sizes=[2048], dropout=0.6, lr=1e-3, weight_decay=1e-3, epochs=60)
    run_experiment(runs, "scale [4096]", scaled,
                   hidden_sizes=[4096], dropout=0.6, lr=1e-3, weight_decay=1e-3, epochs=60)
    report(runs, "image: scaling and network size", "figures/nn_image.png")


if __name__ == "__main__":
    run_sentiment()
    run_image()
