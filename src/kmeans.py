from scipy.spatial.distance import euclidean
import numpy as np

def assign_to_cluster(point, centroids):
    """
    Assign a point to the closest centroid.

    Parameters
    ----------
    point : array-like
    centroids : list of array-like

    Returns
    -------
    tuple (cluster_index, point)
    """
    point = np.asarray(point)
    distances = [euclidean(point, np.asarray(c)) for c in centroids]
    cluster_idx = int(np.argmin(distances))
    return cluster_idx, point
