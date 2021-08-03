
import tensorflow as tf
import tensorflow_addons as tfa
from skimage.transform import warp, AffineTransform
import numpy as np
import pandas as pd
import math
tf.random.set_seed(42)

@tf.function
def apply_random_scaling(image, minval=-.02, maxval=.02):
    param = tf.random.uniform([], minval=minval, maxval=maxval)
    source_size = image.shape
    target_size = tf.cast(source_size[0] * (1.0 + param), tf.int32), tf.cast(source_size[1] * (1.0 + param), tf.int32)
    output = tf.image.resize_with_crop_or_pad(tf.image.resize(image, target_size), source_size[0], source_size[1])
    return output, param


def apply_random_shearing(image, minval=-5., maxval=5.):
    # param = tf.random.uniform([], minval=tf.math.atan(minval/image.shape[1]), maxval=tf.math.atan(maxval/image.shape[1]))
    # param = tf.random.uniform([], minval=tf.math.atan(), maxval=tf.math.atan(maxval/image.shape[1]))
    param = np.random.uniform(low=minval, high=maxval)
    output = warp(np.array(image), AffineTransform(shear=np.arctan(param / image.shape[1])).inverse)
    return output, param


@tf.function
def apply_random_rotation(image, minval=-7, maxval=7):
    param = tf.random.uniform([], minval=minval, maxval=maxval)
    output = tfa.image.rotate(image, param * math.pi / 180.0, interpolation='BILINEAR')
    return output, param


def apply_test_time_augmentation(image, labels, image_id):
    '''Implements TTA, https://arxiv.org/pdf/1911.06475.pdf pag13:

    (...) for each test CXR, we applied a random
    transformation (amongst horizontal flipping,
    rotating ±7 degrees, scaling±2%,and shearing±5 pixels) 10 times (...)

    :param image - the input image
    :param labels - the labels associated with the image
    :param image_id - an ordinal or id associated with the image

    :returns - a DataFrame containing one row for each generated image (+1 for the original one), a list of generated images and labels.
    The dataframe contains the augmentation method used, the parameter and the image/label filenames.
    '''
    dataframe = pd.DataFrame()
    image_list = list()
    image_list.append((image, labels))

    dataframe = dataframe.append({'image_id': image_id,
                                  'tta_id': 0,
                                  'image_fn': '{}_{}_image.npy'.format(image_id, 0),
                                  'labels_fn': '{}_label.npy'.format(image_id),
                                  'method': 'ORIGINAL',
                                  'param': 0.0}, ignore_index=True)

    for i in range(1, 11):

        random_function = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)

        output = image
        param = tf.constant(0.0)

        if tf.equal(random_function, 0):
            output = tf.image.flip_left_right(image)
            param = tf.constant(0.0)
            method = 'FLIP'
        if tf.equal(random_function, 1):
            output, param = apply_random_rotation(image)
            method = 'ROTATION'
        if tf.equal(random_function, 2):
            output, param = apply_random_scaling(image)
            method = 'SCALING'
        if tf.equal(random_function, 3):
            output, param = apply_random_shearing(image)
            method = 'SHEAR'
        image_list.append((output, labels))

        dataframe = dataframe.append({'image_id': image_id,
                                      'tta_id': int(i),
                                      'image_fn': '{}_{}_image.npy'.format(image_id, i),
                                      'labels_fn': '{}_label.npy'.format(image_id),
                                      'method': method,
                                      'param': float(param)}, ignore_index=True)
    return dataframe, image_list
