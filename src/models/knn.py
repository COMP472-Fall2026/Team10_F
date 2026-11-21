import time

import numpy as np

from src.metrics import accuracy


def distances_to_one(X_ref, q):
    """Compute the squared euclidean distance from one point q (d,) to every point of X_ref (n, d).

    Returns: array (n,) of squared distances.
    """
    arr = X_ref - q
    arr *= arr
    arr = arr.sum(axis=1)
    return arr


def find_neighbors(X_ref, X_query, k_max):
    """Find the k_max closest points of X_ref for each point of X_query (m, d).

    Returns: (neighbors, distances), two arrays (m, k_max) sorted from closest to farthest:
    the positions of the neighbors in X_ref and their squared distances.
    """
    neighbors = []
    distances = []
    for q in X_query:
        dist = distances_to_one(X_ref, q)
        nearest_position = np.argsort(dist)
        nearest_position_idx = nearest_position[:k_max]
        neighbors.append(nearest_position_idx)
        distances.append(dist[nearest_position_idx])
    return np.array(neighbors), np.array(distances)


def vote(neighbors, y_ref, n_classes):
    """Majority vote: each neighbor gives one vote to its class (ties go to the smallest class).

    Returns: array (m,) with the predicted class of each query point.
    """
    vote_list = []
    for n in neighbors:
        bincount = np.bincount(y_ref[n], minlength=n_classes)
        vote_list.append(np.argmax(bincount))
    return np.array(vote_list)


def weighted_vote(neighbors, distances, y_ref, n_classes):
    """Weighted vote: each neighbor votes with a weight 1/distance, so closer neighbors count more.

    Returns: array (m,) with the predicted class of each query point.
    """
    vote_list = []
    for n, d in zip(neighbors, distances):
        bincount = np.bincount(y_ref[n], minlength=n_classes, weights=1 / (d + 1e-12))
        vote_list.append(np.argmax(bincount))
    return np.array(vote_list)


def sweep(X_ref, y_ref, X_query, y_query, ks, n_classes, label):
    """Print the accuracy of the simple and weighted vote for every k in ks (sorted list).

    The neighbors are computed only once for the largest k, then reused for every smaller k.
    Returns: nothing, only prints the results and the total time.
    """
    start = time.time()
    neighbors, distances = find_neighbors(X_ref, X_query, ks[-1])
    for k in ks:
        y_pred_vote = vote(neighbors[:, :k], y_ref, n_classes)
        y_pred_weighted = weighted_vote(neighbors[:, :k], distances[:, :k], y_ref, n_classes)
        print(
            f"--- {label} : accuracy for {k}: vote =  {accuracy(y_query, y_pred_vote):.4f}, weighted = {accuracy(y_query, y_pred_weighted):.4f}")
    end = time.time()
    print(f"--- {label} : {end - start:.2f} seconds")


def predict(X_ref, y_ref, X_query, k, n_classes, weighted=True):
    """Classify each point of X_query (m, d) with k-NN, using X_ref and y_ref as reference.

    Returns: array (m,) with the predicted class of each query point.
    """
    neighbors, distances = find_neighbors(X_ref, X_query, k_max=k)
    if weighted:
        return weighted_vote(neighbors, distances, y_ref, n_classes)
    else:
        return vote(neighbors, y_ref, n_classes)
