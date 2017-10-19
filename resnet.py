# Coder: Wenxin Xu
# Github: https://github.com/wenxinxu/resnet_in_tensorflow
# ==============================================================================
# This code was modified from the original code in the link above.

#from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from gutils import unit_initializer
from layers import convolution_layer
from layers import full_connection_layer
from batch_norm import batch_norm

from hyperparameter import FLAGS

def bn_relu_conv(input_layer, filter_shape, strides, bn_param, device):
    bn = batch_norm(input_layer, bn_param, device=device)
    relu = tf.nn.relu(bn)
    conv = convolution_layer(relu, shape=filter_shape, strides=strides, bias=False, layer_name='conv', device=device)
    return conv, relu


def bn_relu_dropout_conv(input_layer, filter_shape, strides, bn_param, keep_prob, device):
    layer = [batch_norm(input_layer, bn_param, device=device)]
    layer.append(tf.nn.relu(layer[-1]))

    if FLAGS.keep_prob!=None:
      layer.append(tf.nn.dropout(layer[-1], keep_prob[0]))

    layer.append(convolution_layer(layer[-1], shape=filter_shape, strides=strides, bias=False, layer_name='conv', device=device))
    return layer[-1]


def first_residual_block(input_layer, output_channel, bn_param, keep_prob, down_sample=False, device=None):
    input_channel = input_layer.get_shape().as_list()[-1]
    assert input_channel!=output_channel

    if down_sample: strides=[1,2,2,1]
    else: strides=[1,1,1,1]

    with tf.variable_scope('layer1_in_block'):
        conv1, relu = bn_relu_conv(input_layer, [3, 3, input_channel, output_channel], strides=strides, bn_param=bn_param, device=device)
    with tf.variable_scope('layer2_in_block'):
        conv2 = bn_relu_dropout_conv(conv1, [3, 3, output_channel, output_channel], strides=[1,1,1,1], bn_param=bn_param, keep_prob=keep_prob, device=device)

    projection = convolution_layer(relu, shape=[1,1,input_channel,output_channel], strides=strides, bias=False, layer_name='projection', device=device)

    return conv2 + projection


def residual_block(input_layer, output_channel, bn_param, keep_prob, device=None):
    input_channel = input_layer.get_shape().as_list()[-1]
    assert input_channel==output_channel

    with tf.variable_scope('layer1_in_block'):
        conv1, _ = bn_relu_conv(input_layer, [3, 3, input_channel, output_channel], strides=[1,1,1,1], bn_param=bn_param, device=device)
    with tf.variable_scope('layer2_in_block'):
        conv2 = bn_relu_dropout_conv(conv1, [3, 3, output_channel, output_channel], strides=[1,1,1,1], bn_param=bn_param, keep_prob=keep_prob, device=device)

    output = conv2 + input_layer

    return output


def inference(input_tensor_batch, bn_param, keep_prob, n, k, num_classes, device):
    layers = []
    with tf.variable_scope('group1'):
        conv0 = convolution_layer(input_tensor_batch, shape=[3, 3, 3, 16], strides=[1, 1, 1, 1], bias=False, layer_name='conv0', device=device)
        layers.append(conv0)

    for i in range(n):
        with tf.variable_scope('group2_block%d' %i):
            if i == 0 and k!=1:
                conv1 = first_residual_block(layers[-1], 16*k, bn_param, keep_prob, down_sample=False, device=device)
            else:
                conv1 = residual_block(layers[-1], 16*k, bn_param, keep_prob, device=device)
            layers.append(conv1)

    for i in range(n):
        with tf.variable_scope('group3_block%d' %i):
            if i==0:
                conv2 = first_residual_block(layers[-1], 32*k, bn_param, keep_prob, down_sample=True, device=device)
            else:
                conv2 = residual_block(layers[-1], 32*k, bn_param, keep_prob, device=device)
            layers.append(conv2)

    for i in range(n):
        with tf.variable_scope('group4_block%d' %i):
            if i==0:
                conv3 = first_residual_block(layers[-1], 64*k, bn_param, keep_prob, down_sample=True, device=device)
            else:
                conv3 = residual_block(layers[-1], 64*k, bn_param, keep_prob, device=device)
            layers.append(conv3)
        assert conv3.get_shape().as_list()[1:] == [8, 8, 64*k]

    with tf.variable_scope('fc'):
        bn_layer = batch_norm(layers[-1], bn_param, device=device)
        relu_layer = tf.nn.relu(bn_layer)
        global_pool = tf.reduce_mean(relu_layer, [1, 2])

        assert global_pool.get_shape().as_list()[-1:] == [64*k]

        shape=[global_pool.get_shape().as_list()[-1], num_classes]
        output = full_connection_layer(global_pool, shape=shape, bias=True, layer_name='output', device=device)

        layers.append(output)

    return layers[-1]
