# Coder: Wenxin Xu
# Github: https://github.com/wenxinxu/resnet_in_tensorflow
# ==============================================================================
# This code was modified from the code in the link above.

#from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import tarfile
from six.moves import urllib
import numpy as np

from hyperparameter import FLAGS

data_dir = FLAGS.data_dir
full_data_dir = os.path.join(data_dir, 'cifar-10-batches-py/data_batch_')
vali_dir = os.path.join(data_dir, 'cifar-10-batches-py/test_batch')
DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'

IMG_WIDTH = 32
IMG_HEIGHT = 32
IMG_DEPTH = 3
NUM_CLASS = 10

NUM_TRAIN_BATCH = 5 # How many batches of files you want to read in, from 0 to 5)

def maybe_download_and_extract():
    '''
    Will download and extract the cifar10 data automatically
    :return: nothing
    '''
    dest_directory = data_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename, float(count * block_size)
                                                             / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def _read_one_batch(path):
    '''
    The training data contains five data batches in total. The validation data has only one
    batch. This function takes the directory of one batch of data and returns the images and
    corresponding labels as numpy arrays

    :param path: the directory of one batch of data
    :param is_random_label: do you want to use random labels?
    :return: image numpy arrays and label numpy arrays
    '''
    fo = open(path, 'rb')
    dicts = np.load(fo,encoding='latin1')
    fo.close()

    data = dicts['data']
    label = np.array(dicts['labels'])

    return data, label


def read_in_all_images(address_list):
    """
    This function reads all training or validation data, shuffles them if needed, and returns the
    images and the corresponding labels as numpy arrays

    :param address_list: a list of paths of cPickle files
    :return: concatenated numpy array of data and labels. Data are in 4D arrays: [num_images,
    image_height, image_width, image_depth] and labels are in 1D arrays: [num_images]
    """
    data = np.array([]).reshape([0, IMG_WIDTH * IMG_HEIGHT * IMG_DEPTH])
    label = np.array([])

    for address in address_list:
        print('Reading images from ' + address)
        batch_data, batch_label = _read_one_batch(address)
        # Concatenate along axis 0 by default
        data = np.concatenate((data, batch_data))
        label = np.concatenate((label, batch_label))

    num_data = len(label)

    # This reshape order is really important. Don't change
    # Reshape is correct. Double checked
    data = data.reshape((num_data, IMG_HEIGHT * IMG_WIDTH, IMG_DEPTH), order='F')
    data = data.reshape((num_data, IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH))

    data = data.astype(np.float32)
    return data, label

def read_train_data():
    '''
    Read all the train data into numpy array and add padding_size of 0 paddings on each side of the
    image
    :param padding_size: int. how many layers of zero pads to add on each side?
    :return: all the train data and corresponding labels
    '''
    path_list = []
    for i in range(1, NUM_TRAIN_BATCH+1):
        path_list.append(full_data_dir + str(i))
    data, label = read_in_all_images(path_list)
   
    return data, label

def read_validation_data():
    '''
    Read in validation data. Whitening at the same time
    :return: Validation image data as 4D numpy array. Validation labels as 1D numpy array
    '''
    validation_array, validation_labels = read_in_all_images([vali_dir])

    return validation_array, validation_labels

