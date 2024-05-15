#!/usr/bin/env python3

"""
This module defines the NST class for performing tasks related to neural style transfer.
"""

import numpy as np
import tensorflow as tf


class NST:
    """
    The NST class performs tasks for neural style transfer.
    """

    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        Initialize the NST object.

        Parameters:
        - style_image (numpy.ndarray): The image used as a style reference.
        - content_image (numpy.ndarray): The image used as a content reference.
        - alpha (float): The weight for content cost.
        - beta (float): The weight for style cost.

        Raises:
        - TypeError: If style_image or content_image is not a numpy.ndarray with shape (h, w, 3).
        - TypeError: If alpha or beta is not a non-negative number.

        Sets Tensorflow to execute eagerly and initializes instance attributes.
        """
        if not isinstance(style_image, np.ndarray) or style_image.shape[-1] != 3:
            raise TypeError("style_image must be a numpy.ndarray with shape (h, w, 3)")

        if not isinstance(content_image, np.ndarray) or content_image.shape[-1] != 3:
            raise TypeError("content_image must be a numpy.ndarray with shape (h, w, 3)")

        if alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        if beta < 0:
            raise TypeError("beta must be a non-negative number")

        tf.compat.v1.enable_eager_execution()

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta

    @staticmethod
    def scale_image(image):
        """
        Rescale an image such that its pixels values are between 0 and 1 and its largest side is 512 pixels.

        Parameters:
        - image (numpy.ndarray): The image to be scaled.

        Raises:
        - TypeError: If image is not a numpy.ndarray with shape (h, w, 3).

        Returns:
        - tf.tensor: The scaled image.
        """
        if not isinstance(image, np.ndarray) or image.shape[-1] != 3:
            raise TypeError("image must be a numpy.ndarray with shape (h, w, 3)")

        h, w = image.shape[0], image.shape[1]
        max_dim = max(h, w)
        scale_factor = 512 / max_dim
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)

        resized_image = tf.image.resize(image, (new_h, new_w), method=tf.image.ResizeMethod.BICUBIC)
        scaled_image = resized_image / 255.0
        scaled_image = tf.expand_dims(scaled_image, axis=0)

        return scaled_image
