import time

import numpy as np

from src.data import load_dataset
from src.models import kmeans
from src.preprocessing import fit_standardizer, standardize


def run_kmeans(name, n_show=5):
    """Cluster the whole training set with k = number of classes, with raw and standardized features,
    then compare the clusters with the true labels (the labels are only used for this comparison).

    Returns: nothing, only prints the results.
    """
    X_train, y_train, _, _ = load_dataset(name)
    n_classes = len(np.unique(y_train))
    print(f"--- {name} : {X_train.shape}, k = {n_classes}, purity of random clusters ~ {1 / n_classes:.4f}")

    mean, std = fit_standardizer(X_train)
    for label, X in [("raw", X_train), ("standardized", standardize(X_train, mean, std))]:
        start = time.time()
        _, clusters, inertia, n_iter = kmeans.fit(X, n_classes)
        table = kmeans.cluster_class_table(clusters, y_train, n_classes, n_classes)
        print(f"--- {name}-{label} : purity = {kmeans.purity(table):.4f}, inertia = {inertia:.1f}, "
              f"{n_iter} iterations, {time.time() - start:.1f} seconds")

        if n_classes <= 5:
            print("rows = clusters, columns = true classes :")
            print(table)
        else:
            sizes = table.sum(axis=1)
            cluster_purity = table.max(axis=1) / np.maximum(sizes, 1)
            order = np.argsort(cluster_purity)
            print("cluster sizes : min", sizes.min(), "max", sizes.max())
            print(f"{n_show} purest clusters (cluster, main class, purity, size) :",
                  [(int(c), int(table[c].argmax()), round(float(cluster_purity[c]), 3), int(sizes[c]))
                   for c in order[::-1][:n_show]])
            print(f"{n_show} least pure clusters (cluster, main class, purity, size) :",
                  [(int(c), int(table[c].argmax()), round(float(cluster_purity[c]), 3), int(sizes[c]))
                   for c in order[:n_show]])
        print()


if __name__ == "__main__":
    run_kmeans("sentiment")
    run_kmeans("image")
