#!/usr/bin/env python3

import tensorflow as tf

def shear_image(image, intensity):
    """
    Randomly shears an image.

    Args:
        image: A 3D tf.Tensor containing the image to shear.
        intensity: The intensity with which the image should be sheared.

    Returns:
        The sheared image.
    """
    # Convert the image to numpy array
    image_np = image.numpy()
    
    # Generate random shearing parameters
    shear_params = tf.random.uniform(
        (2,), minval=-intensity, maxval=intensity)
    
    # Apply shear transformation to the image
    sheared_image = tf.keras.preprocessing.image.apply_affine_transform(
        image_np, shear=shear_params)
    
    return sheared_image
