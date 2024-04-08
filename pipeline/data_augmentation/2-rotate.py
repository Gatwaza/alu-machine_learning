#!/usr/bin/env python3

import tensorflow as tf

def rotate_image(image):
    """
    Rotates an image by 90 degrees counter-clockwise.

    Args:
        image: A 3D tf.Tensor containing the image to rotate.

    Returns:
        The rotated image.
    """
    # Rotate the image by 90 degrees counter-clockwise
    rotated_image = tf.image.rot90(image, k=1)

    return rotated_image
