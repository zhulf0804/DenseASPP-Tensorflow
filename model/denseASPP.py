#coding=utf-8
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
import math
import densenet

denseASPP_layers_num = 5
denseASPP_rates = [3, 6, 12, 18, 24]

CLASSES = densenet.CLASSES

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
        weight_3 = densenet.weight_variable([3, 3, input_shape[-1], n])

        ####
        input = densenet.bn_layer(input, train)
        input = tf.nn.relu(input)
        ####

        input = tf.nn.atrous_conv2d(input, weight_3, rate=rate, padding='SAME')
        input = tf.nn.dropout(input, keep_prob=keep_prob)

    return input

def denseASPP_block(input, train, keep_prob):

    input_shape = input.get_shape().as_list()

    c0 = input_shape[-1]
    n = tf.cast(c0 / 8, tf.int32) # output feature maps of denseASPP layer
    n0 = tf.cast(c0 / 4, tf.int32) # input feature maps of denseASPP layer

    input_0 = input

    for layer in range(1, denseASPP_layers_num + 1):

        with tf.name_scope('denseASPP_layer_%d'%layer):

            input_shape = input.get_shape().as_list()

            ####
            input = densenet.bn_layer(input, train)
            input = tf.nn.relu(input)
            ####

            weight_1 = densenet.weight_variable([1, 1, input_shape[-1], n0])
            input_compress = tf.nn.conv2d(input, weight_1, [1, 1, 1, 1], padding='SAME')


            output = denseASPP_layer(input_compress, denseASPP_rates[layer-1], n, train, keep_prob)

            input_0 = tf.concat([input_0, output], axis=-1)
            input = input_0

    return input


def denseASPP(input, keep_prob, train=True):

    with tf.name_scope("densenet_121"):
        input = densenet.densenet_121(input, keep_prob, train)

    with tf.name_scope("denseASPP"):
        #input = densenet.bn_layer(input, train)
        input = denseASPP_block(input, train, keep_prob)

    with tf.name_scope("segmentation"):
        input_shape = input.get_shape().as_list()

        ####
        #input = densenet.bn_layer(input, train)
        #input = tf.nn.relu(input)
        ####
        input = tf.nn.dropout(input, keep_prob=keep_prob)
        weight_1 = densenet.weight_variable([1, 1, input_shape[-1], CLASSES])
        input = tf.nn.conv2d(input, weight_1, [1, 1, 1, 1], padding='SAME')

    with tf.name_scope("upsamling"):
        input_shape = input.get_shape().as_list()
        input = tf.image.resize_bilinear(input, [8*input_shape[1], 8*input_shape[2]])


    '''
    with tf.name_scope("upsampling"):

        input_shape = input.get_shape().as_list()

        ####
        input = densenet.bn_layer(input, train)
        #input = tf.nn.relu(input)
        ####

        weight_2_1 = densenet.weight_variable([2, 2, CLASSES, CLASSES])
        input = tf.nn.conv2d_transpose(input, weight_2_1, [input_shape[0], input_shape[1] * 2, input_shape[2] * 2, input_shape[3]], [1, 2, 2, 1], padding='SAME')


        input_shape = input.get_shape().as_list()

        ####
        input = densenet.bn_layer(input, train)
        # input = tf.nn.relu(input)
        ####

        weight_2_2 = densenet.weight_variable([2, 2, CLASSES, CLASSES])
        input = tf.nn.conv2d_transpose(input, weight_2_2,
                                       [input_shape[0], input_shape[1] * 2, input_shape[2] * 2, input_shape[3]],
                                       [1, 2, 2, 1], padding='SAME')

        input_shape = input.get_shape().as_list()

        ####
        # input = densenet.bn_layer(input, train)
        # input = tf.nn.relu(input)
        ####

        weight_2_3 = densenet.weight_variable([2, 2, CLASSES, CLASSES])
        input = tf.nn.conv2d_transpose(input, weight_2_3,
                                       [input_shape[0], input_shape[1] * 2, input_shape[2] * 2, input_shape[3]],
                                       [1, 2, 2, 1], padding='SAME')
    '''

    output = input
    return output

if __name__ == '__main__':
    input = tf.constant(0.1, shape=[4, 128, 128, 3], dtype=tf.float32)
    denseASPP_output = denseASPP(input, 1.0, train=True)
    print(denseASPP_output)
