import numpy as np

from src.models.kmeans import assign_clusters, cluster_class_table, fit, purity, update_centroids

# 1D toy example: two obvious groups {1, 2, 3} and {10, 11, 12}
X = np.array([[1.0], [2.0], [3.0], [10.0], [11.0], [12.0]])

# assign_clusters: with centroids 1 and 3, the points 1, 2 go to cluster 0 (2 is at the same distance: argmin keeps 0)
clusters, distances = assign_clusters(X, np.array([[1.0], [3.0]]))
assert np.array_equal(clusters, [0, 0, 1, 1, 1, 1]), f"clusters : {clusters}"
assert np.array_equal(distances, [0, 1, 0, 49, 64, 81]), f"distances : {distances}"

# update_centroids: means of {1, 2} and {3, 10, 11, 12}
centroids = update_centroids(X, clusters, np.array([[1.0], [3.0]]))
assert np.allclose(centroids, [[1.5], [9.0]]), f"centroids : {centroids}"

# an empty cluster keeps its old centroid
centroids = update_centroids(X, np.zeros(6, dtype=int), np.array([[0.0], [100.0]]))
assert np.allclose(centroids, [[6.5], [100.0]]), f"empty cluster : {centroids}"

# fit: the two groups are found (the cluster numbers can be swapped)
centroids, clusters, inertia, n_iter = fit(X, 2)
assert clusters[0] == clusters[1] == clusters[2] != clusters[3] == clusters[4] == clusters[5], f"fit : {clusters}"
assert np.allclose(sorted(centroids[:, 0]), [2, 11]), f"fit centroids : {centroids}"
assert np.isclose(inertia, 4), f"inertia : {inertia} instead of 4 (1 + 0 + 1 for each group)"

# cluster_class_table and purity on a hand-made example
table = cluster_class_table(np.array([0, 0, 0, 1, 1, 1]), np.array([0, 0, 1, 1, 1, 1]), 2, 2)
assert np.array_equal(table, [[2, 1], [0, 3]]), f"table :\n{table}"
assert np.isclose(purity(table), 5 / 6), f"purity : {purity(table)} instead of {5 / 6}"

print("k-Means OK")
