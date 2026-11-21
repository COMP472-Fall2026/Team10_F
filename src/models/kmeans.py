import numpy as np

from seeds import SEED
from src.models.knn import distances_to_one


def init_centroids(X, k, rng):
    """Pick k different points of X at random as the first centroids.

    Returns: array (k, d).
    """
    positions = rng.choice(len(X), size=k, replace=False)
    return X[positions].copy()


def assign_clusters(X, centroids):
    """Give each point of X (n, d) to its closest centroid.

    Returns: (clusters, distances), two arrays (n,): the cluster of each point
    and its squared distance to the centroid of that cluster.
    """
    all_distances = np.array([distances_to_one(X, c) for c in centroids])
    clusters = np.argmin(all_distances, axis=0)
    distances = all_distances[clusters, np.arange(len(X))]
    return clusters, distances


def update_centroids(X, clusters, old_centroids):
    """Move each centroid to the mean of the points of its cluster.
    A cluster with no point keeps its old centroid.

    Returns: array (k, d).
    """
    centroids = old_centroids.copy()
    for c in range(len(centroids)):
        points = X[clusters == c]
        if len(points) > 0:
            centroids[c] = points.mean(axis=0)
    return centroids


def fit(X, k, max_iter=100, seed=SEED):
    """Run k-Means on X (n, d): assign every point to its closest centroid, move every centroid
    to the mean of its points, and repeat until no point changes cluster (or max_iter).
    The labels are never used.

    Returns: (centroids, clusters, inertia, n_iter)
    centroids (k, d), clusters (n,), inertia = sum of the squared distances of the points to their centroid.
    """
    rng = np.random.default_rng(seed)
    centroids = init_centroids(X, k, rng)
    clusters, distances = assign_clusters(X, centroids)
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        centroids = update_centroids(X, clusters, centroids)
        new_clusters, distances = assign_clusters(X, centroids)
        converged = np.array_equal(new_clusters, clusters)
        clusters = new_clusters
        if converged:
            break
    return centroids, clusters, distances.sum(), n_iter


def cluster_class_table(clusters, y, k, n_classes):
    """Count the examples for each (cluster, true class) pair, to compare the clusters with the labels.

    Returns: array (k, n_classes), rows = cluster, columns = true class.
    """
    assert len(clusters) == len(y), "Not same length"
    table = np.zeros((k, n_classes), dtype=int)
    for i in range(len(y)):
        table[clusters[i], y[i]] += 1
    return table


def purity(table):
    """Fraction of the points that belong to the majority class of their cluster
    (1 = every cluster contains only one class).

    Returns: a float between 0 and 1.
    """
    return table.max(axis=1).sum() / table.sum()
