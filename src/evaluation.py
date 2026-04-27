import numpy as np
from sklearn.metrics import calinski_harabasz_score, adjusted_rand_score
from .kmeans import assign_to_cluster


def compute_calinski(rdd, centroids):
    """
    Compute Calinski-Harabasz score.
    """

    assigned = rdd.map(lambda x: assign_to_cluster(x, centroids)).collect()

    labels = [x[0] for x in assigned]
    data = np.array([x[1] for x in assigned])

    return calinski_harabasz_score(data, labels)


def compute_ari(class_labels, rdd, centroids):
    """
    Compute Adjusted Rand Index (ARI).
    """

    assigned = rdd.map(lambda x: assign_to_cluster(x, centroids)).collect()
    predicted = [x[0] for x in assigned]

    return adjusted_rand_score(class_labels, predicted)
