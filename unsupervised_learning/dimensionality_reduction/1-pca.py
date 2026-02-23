#!/usr/bin/env python3
"""Module for performing PCA on a dataset with a fixed number of dimensions."""
import numpy as np


def pca(X, ndim):
    """Perform PCA on a dataset reducing to ndim dimensions.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        ndim: new dimensionality of the transformed X

    Returns:
        T: numpy.ndarray of shape (n, ndim) containing the transformed X
    """
    X_m = X - np.mean(X, axis=0)
    _, _, Vt = np.linalg.svd(X_m, full_matrices=False)
    W = Vt[:ndim].T
    T = np.matmul(X_m, W)
    return T
