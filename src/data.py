import pandas as pd
import numpy as np


def load_dataset(name):
    """Load dataset/<name>_train.csv and dataset/<name>_test.csv ("image" or "sentiment").

    Returns: (X_train, y_train, X_test, ids_test)
    X as float32 arrays (n, d), y_train the labels (n,), ids_test the ids of the test rows.
    """
    df_train = pd.read_csv("dataset/" + name + "_train.csv")
    df_test = pd.read_csv("dataset/" + name + "_test.csv")

    list_of_feature = [f for f in df_train.columns if f.startswith("feature")]
    X_train = df_train[list_of_feature].to_numpy(np.float32)
    X_test = df_test[list_of_feature].to_numpy(np.float32)
    y_train = df_train["label"].to_numpy()
    ids_test = df_test["id"].to_numpy()
    return X_train, y_train, X_test, ids_test
