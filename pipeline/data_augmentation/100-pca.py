#!/usr/bin/env python3

import tensorflow as tf
import numpy as np

def pca_color(image, alphas):
    """
    Performs PCA color augmentation as described in the AlexNet paper.

    Args:
        image: A 3D tf.Tensor containing the image to change.
        alphas: A tuple of length 3 containing the amount that each channel should change.

    Returns:
        The augmented image.
    """
    # Convert image to numpy array
    img_array = tf.image.convert_image_dtype(image, tf.float32).numpy()

    # Flatten image to a 2D array (pixels, channels)
    pixels = np.reshape(img_array, (-1, 3))

    # Compute mean of each channel
    mean = np.mean(pixels, axis=0)

    # Center the pixel values around the mean
    centered_pixels = pixels - mean

    # Compute covariance matrix
    cov_matrix = np.cov(centered_pixels, rowvar=False)

    # Perform eigenvalue decomposition
    eig_values, eig_vectors = np.linalg.eigh(cov_matrix)

    # Generate random coefficients
    random_coefficients = np.random.randn(3) * alphas

    # Scale principal components by random coefficients
    scaled_components = eig_vectors @ np.diag(random_coefficients)

    # Repeat scaled components for each pixel
    scaled_components = np.tile(scaled_components, (pixels.shape[0], 1))

    # Add scaled components to centered pixels
    augmented_pixels = centered_pixels + scaled_components

    # Add mean back to the augmented pixels
    augmented_pixels += mean

    # Reshape augmented pixels to original image shape
    augmented_image = np.reshape(augmented_pixels, image.shape)

    # Clip values to be within [0, 1]
    augmented_image = np.clip(augmented_image, 0, 1)

    # Convert back to tf.Tensor
    augmented_image = tf.convert_to_tensor(augmented_image, dtype=tf.float32)

    return augmented_image
