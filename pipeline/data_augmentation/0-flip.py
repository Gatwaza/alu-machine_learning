#!/usr/bin/env python3
"""flip image horizontary"""

import tensorflow as tf

def flip_image(image):
    """
    Flips an image horizontally.

    Args:
        image: A 3D tf.Tensor containing the image to flip.

    Returns:
        The flipped image.
    """
    # Flip the image horizontally
    flipped_image = tf.image.flip_left_right(image)

    return flipped_image
