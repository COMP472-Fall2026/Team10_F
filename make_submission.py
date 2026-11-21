import numpy as np
import pandas as pd
import torch

from experiments.run_nn import prepare_split, scale_features
from src.data import load_dataset
from src.metrics import accuracy
from src.models import knn
from src.models import neural_network
from src.split import stratified_split

# Best settings of each model, chosen on the validation set with experiments/run_knn.py and experiments/run_nn.py
KNN_SETTINGS = {
    "image": {"k": 1, "weighted": True},
    "sentiment": {"k": 61, "weighted": True},
}
NN_SETTINGS = {
    "image": {"scaling": "scale", "hidden_sizes": [4096], "dropout": 0.6, "lr": 1e-3, "weight_decay": 1e-3,
              "epochs": 60},
    "sentiment": {"scaling": "standardize", "hidden_sizes": [256], "lr": 1e-4, "weight_decay": 1e-2},
}


def validate_knn(X_train, y_train, k, weighted):
    """Accuracy of k-NN on the shared validation split (the train part is used as reference).

    Returns: a float between 0 and 1.
    """
    n_classes = len(np.unique(y_train))
    train_idx, val_idx = stratified_split(y_train)
    y_pred = knn.predict(X_train[train_idx], y_train[train_idx], X_train[val_idx], k, n_classes, weighted)
    return accuracy(y_train[val_idx], y_pred)


def validate_nn(X_train, y_train, scaling, **settings):
    """Train the neural network on the shared split and measure it on the validation part.

    Returns: (accuracy, best_epoch), the best validation accuracy and the epoch where it was reached.
    """
    _, history = neural_network.fit(*prepare_split(X_train, y_train, scaling), **settings)
    best = int(np.argmax(history["validation_acc"]))
    return history["validation_acc"][best], best + 1


def predict_test_knn(X_train, y_train, X_test, k, weighted):
    """Use the full train set as reference and predict the test set with k-NN.

    Returns: array (n_test,) of predicted labels.
    """
    n_classes = len(np.unique(y_train))
    return knn.predict(X_train, y_train, X_test, k, n_classes, weighted)


def predict_test_nn(X_train, y_train, X_test, epochs, scaling, **settings):
    """Train the neural network on the full train set for a fixed number of epochs and predict the test set.

    Returns: array (n_test,) of predicted labels.
    """
    train_features, test_features = scale_features(X_train, X_test, scaling)
    model = neural_network.train_final(torch.tensor(train_features, dtype=torch.float32),
                                       torch.tensor(y_train, dtype=torch.long), epochs, **settings)
    return neural_network.predict(model, torch.tensor(test_features, dtype=torch.float32))


def best_predictions(name):
    """Compare k-NN and the neural network on the validation split, then predict the test set with the best one.

    Returns: (predictions, validation_accuracy), predictions = DataFrame with columns id and label.
    """
    X_train, y_train, X_test, ids_test = load_dataset(name)
    knn_settings = KNN_SETTINGS[name]
    nn_settings = dict(NN_SETTINGS[name])        # copy, so the global settings are not modified
    scaling = nn_settings.pop("scaling")

    knn_acc = validate_knn(X_train, y_train, **knn_settings)
    nn_acc, best_epoch = validate_nn(X_train, y_train, scaling, **nn_settings)
    print(f"--- {name} : validation accuracy k-NN = {knn_acc:.4f} | neural network = {nn_acc:.4f}")

    if knn_acc >= nn_acc:
        print(f"--- {name} : k-NN kept {knn_settings}")
        y_pred = predict_test_knn(X_train, y_train, X_test, **knn_settings)
        best_acc = knn_acc
    else:
        print(f"--- {name} : neural network kept, trained for {best_epoch} epochs on the full train set")
        nn_settings.pop("epochs", None)          # the number of epochs is now the best one found above
        y_pred = predict_test_nn(X_train, y_train, X_test, best_epoch, scaling, **nn_settings)
        best_acc = nn_acc
    return pd.DataFrame({"id": ids_test, "label": y_pred}), best_acc


def check_submission(df):
    """Check the submission format: columns, number of rows, ids and label ranges.

    Returns: nothing, stops with an AssertionError if something is wrong
    """
    sample = pd.read_csv("dataset/sample_submission.csv")
    assert list(df.columns) == ['id', 'label'], f"wrong columns : {list(df.columns)} instead of ['id', 'label']"
    assert df.shape[0] == 10000, f"wrong number of rows : {df.shape[0]} instead of 10000"
    assert df["id"].size == len(np.unique(df["id"])), "some ids appear more than once"
    assert set(df["id"]) == set(sample["id"]), "ids are not the same as in sample_submission.csv"
    mask = df["id"].str.startswith("img")
    labels_img = df[mask]["label"]
    labels_sentiment = df[~mask]["label"]
    assert labels_img.min() >= 0 and labels_img.max() <= 49, f"image labels out of range [0, 49] : min {labels_img.min()}, max {labels_img.max()}"
    assert labels_sentiment.min() >= 0 and labels_sentiment.max() <= 2, f"sentiment labels out of range [0, 2] : min {labels_sentiment.min()}, max {labels_sentiment.max()}"


if __name__ == '__main__':
    df_img, acc_img = best_predictions("image")
    df_stm, acc_stm = best_predictions("sentiment")
    df = pd.concat([df_img, df_stm])
    check_submission(df)
    df.to_csv("submission.csv", index=False)
    print(f"--- submission.csv written, expected score ≈ {(acc_img + acc_stm) / 2:.4f} (mean of the two validation accuracies)")
