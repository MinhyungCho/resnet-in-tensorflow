#from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import tarfile
from six.moves import urllib
import numpy as np
import scipy.io as sio

from hyperparameter import FLAGS

data_dir = FLAGS.data_dir
full_data_dir = os.path.join(data_dir, 'svhn/')
DATA_URL = 'http://ufldl.stanford.edu/housenumbers/'

def maybe_download():
  def _progress(count, block_size, total_size):
    sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename, float(count * block_size)
                                                     / float(total_size) * 100.0))
    sys.stdout.flush()

  dest_directory = full_data_dir
  if not os.path.exists(dest_directory):
      os.makedirs(dest_directory)

  filelist=['train_32x32.mat','test_32x32.mat','extra_32x32.mat']
  for filename in filelist:
    filepath = os.path.join(dest_directory,filename)
    if not os.path.exists(filepath):
      urllib.request.urlretrieve(DATA_URL+filename, filepath, _progress)
      print()
      statinfo = os.stat(filepath)
      print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')

  return


def read_images(address):
  train_data = sio.loadmat(address)

  data = train_data['X']
  label = train_data['y']

  data = np.swapaxes(data, 2, 3)
  data = np.swapaxes(data, 1, 2)
  data = np.swapaxes(data, 0, 1)

  label = np.squeeze(label)
  return data, label


def read_test_data():
    """
    test_data.shape = (26032, 32, 32, 3)
    test_label.shape = (26032,)
    """
    test_data, test_label = read_images(full_data_dir+'test_32x32.mat')

    return test_data, test_label 


def read_train_data():
    """
    train_data.shape = (73257, 32, 32, 3)
    train_label.shape = (73257,)
    extra_data.shape = (531131, 32, 32, 3)
    extra_label.shape = (531131,)
    data.shape = (604388, 32, 32, 3)
    labels.shape = (604388,)
    """

    train_data, train_label = read_images(full_data_dir+'train_32x32.mat')
    extra_data, extra_label = read_images(full_data_dir+'extra_32x32.mat')

    data = np.concatenate( (train_data, extra_data) )
    label = np.concatenate( (train_label, extra_label) )

    return data, label
