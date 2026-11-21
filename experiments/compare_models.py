"""4x2 table: validation accuracy of the 4 chosen models on the 2 datasets (same 80/20 split for all of them).
Run from the repo root: python -m experiments.compare_models   (~5 min, the Random Forest is the slowest)
"""
import os
import time

import pandas as pd

from experiments.analyze_errors import predict_val
from make_submission import NN_SETTINGS, validate_nn
from src.data import load_dataset
from src.metrics import accuracy

# Settings chosen on the validation set (run_knn, run_tree, run_forest, run_nn)
CHOSEN = {
    "image": {
        "k-NN": {"k": 1, "weighted": True},
        "Decision Tree": {"max_depth": 40},
        "Random Forest": {"n_trees": 300, "max_depth": None, "n_features": "sqrt"},
    },
    "sentiment": {
        "k-NN": {"k": 61, "weighted": True},
        "Decision Tree": {"max_depth": 5},
        "Random Forest": {"n_trees": 100, "max_depth": None, "n_features": "sqrt"},
    },
}
MODEL_KEYS = {"k-NN": "knn", "Decision Tree": "tree", "Random Forest": "forest"}


def validation_accuracies(name):
    """Train each chosen model on the train part and measure it on the validation part.

    Returns: dict model name -> (validation accuracy, seconds)
    """
    results = {}
    for model, params in CHOSEN[name].items():
        start = time.time()
        y_val, y_pred = predict_val(name, MODEL_KEYS[model], **params)
        results[model] = (accuracy(y_val, y_pred), time.time() - start)
        print(f"--- {name} : {model:13s} | val = {results[model][0]:.4f} | {results[model][1]:.0f} s")

    X_train, y_train, _, _ = load_dataset(name)
    nn_settings = dict(NN_SETTINGS[name])
    scaling = nn_settings.pop("scaling")
    start = time.time()
    acc, best_epoch = validate_nn(X_train, y_train, scaling, **nn_settings)
    results["MLP"] = (acc, time.time() - start)
    print(f"--- {name} : {'MLP':13s} | val = {acc:.4f} | {results['MLP'][1]:.0f} s (best epoch {best_epoch})")
    return results


if __name__ == "__main__":
    results = {name: validation_accuracies(name) for name in ["image", "sentiment"]}
    table = pd.DataFrame({name: {model: round(acc, 4) for model, (acc, _) in res.items()}
                          for name, res in results.items()})
    print("\nValidation accuracy (rows = model, columns = dataset):")
    print(table.to_string())
    os.makedirs("results", exist_ok=True)
    table.to_csv("results/compare_models.csv", index_label="model")
    print("--- saved to results/compare_models.csv")
