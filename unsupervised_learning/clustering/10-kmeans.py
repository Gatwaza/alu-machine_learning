#!/usr/bin/env python3
"""Module for K-means clustering using sklearn."""
import sklearn.cluster


def kmeans(X, k):
    """Perform K-means on a dataset using sklearn.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        k: number of clusters

    Returns:
        C: numpy.ndarray of shape (k, d) with centroid means
        clss: numpy.ndarray of shape (n,) with cluster indices
    """
    model = sklearn.cluster.KMeans(n_clusters=k)
    model.fit(X)
    C = model.cluster_centers_
    clss = model.labels_
    return C, clss
