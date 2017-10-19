#from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from layers import _variable_on_device

def batch_norm(inputs, bn_param, scale=True, momentum=0.99, epsilon=1e-5, name='batch_norm', device=None):
  with tf.variable_scope(name):
    beta = _variable_on_device('beta', [inputs.get_shape()[-1]],
               initializer=tf.zeros_initializer(), 
               trainable=True,
               device=device)

    if scale:
      gamma = _variable_on_device('gamma', [inputs.get_shape()[-1]],
                  initializer=tf.ones_initializer(), 
                  trainable=True,
                  device=device)
    else:
      gamma = None

    reduced_dim = [i for i in range(len(inputs.get_shape())-1)] 
    batch_mean, batch_var = tf.nn.moments(inputs,reduced_dim,keep_dims=False)

    # moving average of the populations
    pop_mean = _variable_on_device('pop_mean', 
                   shape=[inputs.get_shape()[-1]],
                   initializer=tf.zeros_initializer(), 
                   trainable=False, 
                   device=device)
    pop_var = _variable_on_device('pop_var', 
                  shape=[inputs.get_shape()[-1]],
                  initializer=tf.ones_initializer(), 
                  trainable=False, 
                  device=device)

    pop_mean_op = tf.assign(pop_mean, pop_mean * momentum + batch_mean * (1 - momentum))
    pop_var_op  = tf.assign(pop_var, pop_var * momentum + batch_var * (1 - momentum))

    tf.add_to_collection('batch_norm_update', pop_mean_op)
    tf.add_to_collection('batch_norm_update', pop_var_op)

    # for training, bn_param[0]=0
    # for evaluation, bn_param[0]=1
    mean = bn_param[0]*pop_mean + (1-bn_param[0])*batch_mean
    var = bn_param[0]*pop_var + (1-bn_param[0])*batch_var

    return tf.nn.batch_normalization(inputs, mean, var, beta, gamma, epsilon)
