import os
import numpy as np
import tensorflow as tf


def create_folder_if_not_exist(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)
        

def extract_patches(image, patch_size, stride):
    '''extract patches from given image according to given stride and patch_size.'''
    image = image[np.newaxis, ...]
    if len(image.shape) == 3:
        image = image[..., np.newaxis]
    image_tiles = tf.image.extract_patches(image, sizes=(1, patch_size, patch_size, 1), strides=(1, stride, stride, 1),
                                           padding='SAME', rates=(1, 1, 1, 1))
    image_tiles = np.reshape(image_tiles,
                             (image_tiles.shape[1] * image_tiles.shape[2], patch_size, patch_size, image.shape[3]))
    return image_tiles