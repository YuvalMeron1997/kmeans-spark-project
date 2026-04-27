from scipy.spatial.distance import euclidean
import numpy as np


def assign_to_cluster(point, centroids):
    """
    Assign a data point to the nearest centroid.

    Returns:
        (cluster_index, point)
    """
    point = np.asarray(point)
    distances = [euclidean(point, np.asarray(c)) for c in centroids]
    cluster_idx = int(np.argmin(distances))
    return cluster_idx, point


def run_kmeans(rdd, k, max_iter=30, tol=1e-4):
    """
    Run K-Means clustering on a Spark RDD.

    Parameters
    ----------
    rdd : pyspark RDD
        Dataset of points
    k : int
        Number of clusters
    max_iter : int
        Maximum iterations
    tol : float
        Convergence threshold

    Returns
    -------
    list
        Final centroids
    """

    centroids = rdd.takeSample(False, k)

    for _ in range(max_iter):

        # Assign points
        mapped = rdd.map(lambda x: assign_to_cluster(x, centroids))

        # Update centroids
        new_centroids = (
            mapped
            .map(lambda x: (x[0], (x[1], 1)))
            .reduceByKey(lambda a, b: (
                tuple(a[0][i] + b[0][i] for i in range(len(a[0]))),
                a[1] + b[1]
            ))
            .mapValues(lambda x: tuple(val / x[1] for val in x[0]))
            .collect()
        )

        new_centroids = [c[1] for c in new_centroids]

        # Convergence check
        converged = True
        for i in range(k):
            if euclidean(np.array(new_centroids[i]), np.array(centroids[i])) > tol:
                converged = False
                break

        centroids = new_centroids

        if converged:
            break

    return centroids
