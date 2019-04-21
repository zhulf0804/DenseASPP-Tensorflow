# coding=utf-8
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
import os
import math
import sys
import cv2
import random
from PIL import Image

dataset = '/Users/zhulf/data/CamVid'
tfrecord_file = os.path.join(dataset, 'tfrecord')

_NUM_SHARDS = 2
HEIGHT = 360
WIDTH = 480

def flip_random_left_right(image, anno):
    '''
    :param image: [height, width, channel]
    :return:
    '''
    flag = random.randint(0, 1)
    if flag:
        return cv2.flip(image, 1), cv2.flip(anno, 1)
    return image, anno

def flip_batch_random_left_right(batch_image, batch_anno):
    '''
    data augmentation: random flip horizontally
    :param batch_image: [batch, height, width, channel]
    :return:
    '''
    for i in range(len(batch_image)):
        batch_image[i, ...], batch_anno[i, ...] = flip_random_left_right(batch_image[i, ...], batch_anno[i, ...])

    return batch_image, batch_anno

def random_brightness(image, max_delta):
    '''
    :param image: [height, width, channel]
    :param max_delta: [-max_delta, max_delta]
    :return:
    '''
    #flag = random.randint(0, 1)
    flag = 1
    if flag:
        delta = random.randint(-max_delta, max_delta)
        return image + delta
    return image

def random_batch_brightness(batch_image, max_delta=32):
    '''
    data augmentation: random brightness
    :param batch_image: [batch, height, width, channel]
    :return:
    '''
    for i in range(len(batch_image)):
        batch_image[i, ...] = random_brightness(batch_image[i, ...], max_delta)

    return batch_image

def random_contrast(image, lower=0.8, upper=1.2):
    '''
    random contrast factor from [lower, upper]
    :param image: [height, width, channel]
    :param lower:
    :param upper:
    :return: (x - mean) * contrast_factor + mean
    '''
    mmean = np.mean(image)
    contrast_factor = random.uniform(lower, upper)
    #flag = random.randint(0, 1)
    flag = 1
    if flag:
        return (image - mmean) * contrast_factor + mmean
    return image

def random_batch_contrast(batch_image, lower=0.8, upper=1.2):
    '''
    :param batch_image: [batch, height, width, channel]
    :param lower:
    :param upper:
    :return:
    '''
    for i in range(len(batch_image)):
        batch_image[i, ...] = random_contrast(batch_image[i, ...], lower, upper)

    return batch_image

def random_resize_pad_crop(image, anno, mmin, mmax):

    flag = random.randint(0, 1)

    if flag:

        height, width = anno.shape
        random_scale = random.uniform(mmin, mmax)
        image = cv2.resize(image, (int(random_scale*WIDTH), int(random_scale*HEIGHT)), interpolation=cv2.INTER_LINEAR)
        anno = cv2.resize(anno, (int(random_scale*WIDTH), int(random_scale*HEIGHT)), interpolation=cv2.INTER_NEAREST)


        max_h = int(random_scale*HEIGHT) - HEIGHT
        max_w = int(random_scale*WIDTH) - WIDTH
        x_st = np.random.randint(low=0, high=max_h)
        y_st = np.random.randint(low=0, high=max_w)

        if random_scale >= 1.0:
            image = image[x_st: x_st + HEIGHT, y_st : y_st + WIDTH, :]
            anno = anno[x_st: x_st + HEIGHT, y_st: y_st + WIDTH]

    return image, anno

def random_batch_resize_pad_crop(batch_image, batch_anno, mmin=1.2, mmax=2.0):
    '''
    :param batch_image: [batch, height, width, channel]
    :param
    :param
    :return:
    '''
    for i in range(len(batch_image)):
        batch_image[i, ...], batch_anno[i, ...] = random_resize_pad_crop(batch_image[i, ...], batch_anno[i, ...], mmin, mmax)

    return batch_image, batch_anno



def image_standardization(image):
    '''
    (image - mean) / adjusted_stddev, adjusted_stddev = max(stddev, 1.0/sqrt(image.NumElements()))
    :param image: [height, width, channel]
    :return:
    '''
    mmean = np.mean(image)
    stddev = np.std(image)
    num_elements = np.size(image)
    adjusted_stddev = np.maximum(stddev, 1.0/math.sqrt(num_elements))

    return (image - mmean) / adjusted_stddev

def image_standardization_batch(batch_image):

    for i in range(len(batch_image)):
        batch_image[i, ...] = image_standardization(batch_image[i, ...])

    return batch_image


def aug_std(batch_image, batch_anno, type='test'):
    if type == 'train':
        batch_image, batch_anno = flip_batch_random_left_right(batch_image, batch_anno)
        batch_image, batch_anno = random_batch_resize_pad_crop(batch_image, batch_anno)
        batch_image = batch_image.astype(np.float32)
        batch_image = random_batch_brightness(batch_image, max_delta=63)
        batch_image = random_batch_contrast(batch_image, lower=0.8, upper=1.2)
    else:
        batch_image = batch_image.astype(np.float32)

    batch_image = image_standardization_batch(batch_image)

    return batch_image, batch_anno


def read_and_decode(filelist):
    filename_queue = tf.train.string_input_producer(filelist)
    reader = tf.TFRecordReader()
    _, serialized_exampe = reader.read(filename_queue)

    features = tf.parse_single_example(serialized_exampe,
                                       features={
                                           'image/encoded': tf.FixedLenFeature([], tf.string),
                                           'image/anno': tf.FixedLenFeature([], tf.string),
                                           'image/filename': tf.FixedLenFeature([], tf.string),
                                           'image/height': tf.FixedLenFeature([], tf.int64),
                                           'image/width': tf.FixedLenFeature([], tf.int64),
                                       })

    image = tf.decode_raw(features['image/encoded'], tf.uint8)
    anno = tf.decode_raw(features['image/anno'], tf.uint8)
    filename =features['image/filename']
    height = tf.cast(features['image/height'], tf.int32)
    width = tf.cast(features['image/width'], tf.int32)

    image = tf.reshape(image, [HEIGHT, WIDTH, 3])
    anno = tf.reshape(anno, [HEIGHT, WIDTH])

    #image = tf.cast(image, tf.float32)
    #image = image / 255

    return image, anno, filename

def read_batch(batch_size, type = 'train'):
    filelist_train = [os.path.join(tfrecord_file, 'image_%s_%05d-of-%05d.tfrecord' % ('train', shard_id, _NUM_SHARDS - 1)) for shard_id in range(_NUM_SHARDS)]
    filelist_val = [os.path.join(tfrecord_file, 'image_%s_%05d-of-%05d.tfrecord' % ('val', shard_id, _NUM_SHARDS - 1)) for shard_id in range(_NUM_SHARDS)]
    filelist_test = [os.path.join(tfrecord_file, 'image_%s_%05d-of-%05d.tfrecord' % ('test', shard_id, _NUM_SHARDS - 1)) for shard_id in range(_NUM_SHARDS)]

    filelist = []
    if type == 'train':
        filelist = filelist + filelist_train
    elif type == 'val':
        filelist = filelist + filelist_val
    elif type == 'test':
        filelist = filelist + filelist_test
    elif type == 'trainval':
        filelist = filelist + filelist_train + filelist_val
    else:
        raise Exception('data set name not exits')


    print(filelist)
    image, anno, filename = read_and_decode(filelist)

    image_batch, anno_batch, filename = tf.train.shuffle_batch([image, anno, filename], batch_size=batch_size, capacity=128, min_after_dequeue=64, num_threads=2)

    # print(image_batch, anno_batch)

    return image_batch, anno_batch, filename

if __name__ == '__main__':
    BATCH_SIZE = 4
    image_batch, anno_batch, filename = read_batch(BATCH_SIZE, type='test')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        b_image, b_anno, b_filename = sess.run([image_batch, anno_batch, filename])
        print(b_image.shape)
        print(b_anno.shape)
        print(b_filename)

        b_image, b_anno, b_filename = sess.run([image_batch, anno_batch, filename])
        print(b_image.shape)
        print(b_anno.shape)
        print(b_filename)

        coord.request_stop()
        # 其他所有线程关闭后，这一函数才能返回
        coord.join(threads)