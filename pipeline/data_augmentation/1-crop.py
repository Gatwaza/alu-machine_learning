#!/usr/bin/env python3

import tensorflow as tf

def crop_image(image, size):
    """
    Performs a random crop of an image.

    Args:
        image: A 3D tf.Tensor containing the image to crop.
        size: A tuple containing the size of
		 the crop (height, width, channels).

    Returns:
        The cropped image.
    """
    # Extracting the dimensions of the image
    img_height, img_width, _ = image.shape.as_list()

    # Extracting the desired crop size
    crop_height, crop_width, _ = size

    # Ensure that crop size is not larger than the image size
    crop_height = min(crop_height, img_height)
    crop_width = min(crop_width, img_width)

    # Generate random coordinates for the top-left corner of the crop
    offset_height = tf.random.uniform(
        (),
        maxval=img_height - crop_height,
        dtype=tf.int32
    )
    offset_width = tf.random.uniform(
        (),
        maxval=img_width - crop_width,
        dtype=tf.int32
    )

    # Crop the image using random coordinates
    cropped_image = tf.image.crop_to_bounding_box(
        image,
        offset_height,
        offset_width,
        crop_height,
        crop_width
    )

    return cropped_image
