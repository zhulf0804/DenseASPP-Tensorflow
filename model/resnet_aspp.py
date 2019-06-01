#coding=utf-8
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
import math

slim = tf.contrib.slim
from tensorflow.contrib.slim.nets import resnet_v2
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import layers

#Used for BN
_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5


# for denseASPP
denseASPP_layers_num = 5
denseASPP_rates = [3, 6, 12, 18, 24]

def weight_variable(shape, stddev=None, name='weight'):
    if stddev == None:
        if len(shape) == 4:
            stddev = math.sqrt(2. / (shape[0] * shape[1] * shape[2]))
        else:
            stddev = math.sqrt(2. / shape[0])
    else:
        stddev = 0.1
    initial = tf.truncated_normal(shape, stddev=stddev)
    W = tf.Variable(initial, name=name)

    return W


def bias_variable(shape, name='bias'):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def batch_norm(inputs, training):

  return tf.layers.batch_normalization(
      inputs=inputs, axis=-1,
      momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
      scale=True, training=training, fused=True)

def denseASPP_layer(input, rate, n, train, keep_prob):
    '''

    :param input: input feature map
    :param rate: dilatation rate
    :param n0: input channels
    :param n: output channels
    :return:
    '''
    with tf.name_scope("denseASPP_layer"):
        input_shape = input.get_shape().as_list()
        n = tf.cast(n, tf.int32)
        weight_3 = weight_variable([3, 3, input_shape[-1], n])

        ####
        input = batch_norm(input, train)
        input = tf.nn.relu(input)
        ####

        input = tf.nn.atrous_conv2d(input, weight_3, rate=rate, padding='SAME')
        input = tf.nn.dropout(input, keep_prob=keep_prob)

    return input

def denseASPP_block(input, train, keep_prob):

    with tf.contrib.slim.arg_scope(resnet_v2.resnet_arg_scope(batch_norm_decay=_BATCH_NORM_DECAY)):
        input_shape = input.get_shape().as_list()

        c0 = input_shape[-1]
        n = tf.cast(c0 / 8, tf.int32) # output feature maps of denseASPP layer
        n0 = tf.cast(c0 / 4, tf.int32) # input feature maps of denseASPP layer

        input_0 = input

        for layer in range(1, denseASPP_layers_num + 1):

            with tf.name_scope('denseASPP_layer_%d'%layer):

                input_shape = input.get_shape().as_list()

                ####
                input = batch_norm(input, train)
                input = tf.nn.relu(input)
                ####

                weight_1 = weight_variable([1, 1, input_shape[-1], n0])
                input_compress = tf.nn.conv2d(input, weight_1, [1, 1, 1, 1], padding='SAME')


                output = denseASPP_layer(input_compress, denseASPP_rates[layer-1], n, train, keep_prob)

                input_0 = tf.concat([input_0, output], axis=-1)
                input = input_0

    return input


def denseASPP(inputs, is_training, output_stride, pre_trained_model, classes, keep_prob=1.0):

    with tf.contrib.slim.arg_scope(resnet_v2.resnet_arg_scope(batch_norm_decay=_BATCH_NORM_DECAY)):
        logits, end_points = resnet_v2.resnet_v2_101(inputs, num_classes=None, is_training=is_training,
                                                     global_pool=False, output_stride=output_stride)

    if is_training:
        exclude = ['resnet_v2_101' + '/logits', 'global_step']
        variables_to_restore = tf.contrib.slim.get_variables_to_restore(exclude=exclude)
        tf.train.init_from_checkpoint(pre_trained_model, {v.name.split(':')[0]: v for v in variables_to_restore})

    net = end_points['resnet_v2_101' + '/block4']



    with tf.name_scope("denseASPP"):

        input = denseASPP_block(net, is_training, keep_prob)

    with tf.contrib.slim.arg_scope(resnet_v2.resnet_arg_scope(batch_norm_decay=_BATCH_NORM_DECAY)):

        with tf.name_scope("segmentation"):

            input_shape = input.get_shape().as_list()
            input = tf.nn.dropout(input, keep_prob=keep_prob)
            weight_1 = weight_variable([1, 1, input_shape[-1], classes])
            bias = bias_variable([classes])
            input = tf.nn.conv2d(input, weight_1, [1, 1, 1, 1], padding='SAME') + bias

        with tf.name_scope("upsamling"):
            input_shape = input.get_shape().as_list()
            input = tf.image.resize_bilinear(input, tf.shape(inputs)[1:3])


    output = input
    return output
