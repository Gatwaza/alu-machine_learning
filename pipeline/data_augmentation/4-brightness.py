#!/usr/bin/env python3

import tensorflow as tf

def change_brightness(image, max_delta):
    """
    Randomly changes the brightness of an image.

    Args:
        image: A 3D tf.Tensor containing the image to change.
        max_delta: The maximum amount the image should be brightened (or darkened).

    Returns:
        The altered image.
    """
    # Generate random brightness delta within the specified range
    brightness_delta = tf.random.uniform((), minval=-max_delta, maxval=max_delta)
    
    # Apply brightness adjustment to the image
    brightened_image = tf.image.adjust_brightness(image, brightness_delta)
    
    return brightened_image
