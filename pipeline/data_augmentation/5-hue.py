#!/usr/bin/env python3

import tensorflow as tf

def change_hue(image, delta):
    """
    Changes the hue of an image.

    Args:
        image: A 3D tf.Tensor containing the image to change.
        delta: The amount the hue should change.

    Returns:
        The altered image.
    """
    # Apply hue adjustment to the image
    hue_adjusted_image = tf.image.adjust_hue(image, delta)
    
    return hue_adjusted_image
