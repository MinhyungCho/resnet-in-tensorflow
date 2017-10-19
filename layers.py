#from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def _variable_on_device(name, shape, initializer, trainable,  device):
  if device==None:
    var = tf.get_variable(name, shape, initializer=initializer, trainable=trainable)
  else:
    with tf.device(device):
      var = tf.get_variable(name, shape, initializer=initializer, trainable=trainable)
  return var

def base_layer(input_tensor, shape, F, bias, layer_name, device):
  with tf.variable_scope(layer_name) as scope:
    weight = _variable_on_device('weight', 
                 shape=shape,
                 initializer=tf.contrib.layers.variance_scaling_initializer(),
                 trainable=True,
                 device=device)

    if bias:
      b = _variable_on_device('bias', 
              shape=shape[-1],
              initializer=tf.zeros_initializer(), 
              trainable=True,
              device=device)
      preactivation = F(input_tensor, weight) + b
    else:
      preactivation = F(input_tensor, weight)

    return preactivation

def convolution_layer(input_tensor, shape, strides, bias, layer_name, device):
  def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=strides, padding='SAME', use_cudnn_on_gpu=True)

  return base_layer(input_tensor, shape, conv2d, bias, layer_name, device)

def full_connection_layer(input_tensor, shape, bias, layer_name, device):
  return base_layer(input_tensor, shape, tf.matmul, bias, layer_name, device)
