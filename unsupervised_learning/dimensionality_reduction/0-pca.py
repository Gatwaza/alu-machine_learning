#!/usr/bin/env python3
"""Module for performing PCA on a dataset."""
import numpy as np


def pca(X, var=0.95):
    """Perform PCA on a dataset.

    Args:
        X: numpy.ndarray of shape (n, d) with zero mean across all dimensions
        var: fraction of variance the PCA transformation should maintain

    Returns:
        W: numpy.ndarray of shape (d, nd) - the weights matrix
    """
    _, s, Vt = np.linalg.svd(X, full_matrices=False)
    cumvar = np.cumsum(s ** 2) / np.sum(s ** 2)
    nd = np.argmax(cumvar >= var) + 1
    W = Vt[:nd + 1].T
    return W
