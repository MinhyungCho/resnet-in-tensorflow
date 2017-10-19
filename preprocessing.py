# Coder: Wenxin Xu
# Github: https://github.com/wenxinxu/resnet_in_tensorflow
# ==============================================================================
# This code was modified from the code in the link above.

#from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

def horizontal_flip(image):
    '''
    Flip an image at 50% possibility
    :param image: a 3 dimensional numpy array representing an image
    :param axis: 0 for vertical flip and 1 for horizontal flip
    :return: 3D image after flip
    '''
    flip_prop = np.random.randint(low=0, high=2)
    if flip_prop == 0:
      image = np.fliplr(image)
    return image


def pad_images(data, padding_size):
    pad_width = ((0, 0), (padding_size, padding_size), (padding_size, padding_size), (0, 0))
    return np.pad(data, pad_width=pad_width, mode='reflect')


def random_crop_and_flip(batch_data, padding_size):
    '''
    Helper to random crop and random flip a batch of images
    :param padding_size: int. how many layers of 0 padding was added to each side
    :param batch_data: a 4D batch array
    :return: randomly cropped and flipped image
    '''
    height = batch_data.shape[1]-2*padding_size
    width = batch_data.shape[2]-2*padding_size
    depth = batch_data.shape[3]

    cropped_batch = np.zeros(len(batch_data) * height * width * depth).reshape(
        len(batch_data), height, width, depth)

    for i in range(len(batch_data)):
        x_offset = np.random.randint(low=0, high=2 * padding_size + 1, size=1)[0]
        y_offset = np.random.randint(low=0, high=2 * padding_size + 1, size=1)[0]
        cropped_batch[i, ...] = batch_data[i, ...][x_offset:x_offset+height,
                      y_offset:y_offset+width, :]

        cropped_batch[i, ...] = horizontal_flip(image=cropped_batch[i, ...])

    return cropped_batch

