# coding=utf-8
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
import os
import math
import sys
from PIL import Image
import cv2
import random
import to_tfrecord as TFRecord

tfrecord_file = TFRecord.tfrecord_file
_NUM_SHARDS = TFRecord._NUM_SHARDS
HEIGHT = 1024
WIDTH = 2048
CROP_HEIGHT = 512
CROP_WIDTH = 512


def flip_randomly_left_right_image_with_annotation(image_0_tensor, image_tensor, annotation_tensor):
    # Reference https://github.com/warmspringwinds/tf-image-segmentation/blob/master/tf_image_segmentation/utils/augmentation.py
    # Random variable: two possible outcomes (0 or 1)
    # with a 1 in 2 chance
    random_var = tf.random_uniform(maxval=2, dtype=tf.int32, shape=[])

    randomly_flipped_img_0 = tf.cond(pred=tf.equal(random_var, 0),
                                   fn1=lambda: tf.image.flip_left_right(image_0_tensor),
                                   fn2=lambda: image_0_tensor)

    randomly_flipped_img = tf.cond(pred=tf.equal(random_var, 0),
                                                 fn1=lambda: tf.image.flip_left_right(image_tensor),
                                                 fn2=lambda: image_tensor)

    randomly_flipped_annotation = tf.cond(pred=tf.equal(random_var, 0),
                                                        fn1=lambda: tf.image.flip_left_right(tf.expand_dims(annotation_tensor, -1)),
                                                        fn2=lambda: tf.expand_dims(annotation_tensor, -1))
    randomly_flipped_annotation = tf.squeeze(randomly_flipped_annotation, -1)
    return randomly_flipped_img_0, randomly_flipped_img, randomly_flipped_annotation

def random_resize(batch_image_0, batch_image, batch_anno, mmin=0.5, mmax=2):
    rand_var = tf.random_uniform(shape=[],
                                 minval=mmin,
                                 maxval=mmax)
    scaled_shape = [tf.cast(tf.round(rand_var * HEIGHT), tf.int32), tf.cast(tf.round(rand_var * WIDTH), tf.int32)]

    batch_image_0 = tf.image.resize_bilinear(batch_image_0, scaled_shape)
    batch_image_0 = tf.cast(batch_image_0, tf.uint8)
    batch_image = tf.image.resize_bilinear(batch_image, scaled_shape)

    batch_anno = tf.expand_dims(batch_anno, -1)
    batch_anno = tf.image.resize_nearest_neighbor(batch_anno, scaled_shape)
    batch_anno = tf.squeeze(batch_anno, -1)

    return batch_image_0, batch_image, batch_anno

def random_crop(batch_image_0, batch_image, batch_anno):

    '''
    seed = random.randint(0, 1e10)
    input_shape = batch_image.get_shape().as_list()
    batch_image_0 = tf.random_crop(batch_image_0, [input_shape[0], CROP_HEIGHT, CROP_WIDTH, 3], seed=seed)
    batch_image = tf.random_crop(batch_image, [input_shape[0], CROP_HEIGHT, CROP_WIDTH, 3], seed=seed)
    batch_anno = tf.random_crop(batch_anno, [input_shape[0], CROP_HEIGHT, CROP_WIDTH], seed=seed)
    return batch_image_0, batch_image, batch_anno
    '''
    x_st = np.random.randint(low=0, high=512)
    y_st = np.random.randint(low=0, high=1536)

    input_shape = batch_image.get_shape().as_list()
    batch_image_0 = tf.slice(batch_image_0, [0, x_st, y_st, 0], [input_shape[0], CROP_HEIGHT, CROP_WIDTH, 3])
    batch_image = tf.slice(batch_image, [0, x_st, y_st, 0], [input_shape[0], CROP_HEIGHT, CROP_WIDTH, 3])
    batch_anno = tf.slice(batch_anno, [0, x_st, y_st], [input_shape[0], CROP_HEIGHT, CROP_WIDTH])

    return batch_image_0, batch_image, batch_anno

def augmentation_standardization(image_0, image, anno, type):

    image = tf.cast(image, tf.float32)

    #if type == 'train' or type == 'val':

        #image_0, image, anno = flip_randomly_left_right_image_with_annotation(image_0, image, anno)
        #image = tf.image.random_brightness(image, max_delta=10)

    image = tf.image.per_image_standardization(image)
    #image /= 255
    #image -= 0.5
    image_0 = tf.reshape(image_0, [HEIGHT, WIDTH, 3])
    image = tf.reshape(image, [HEIGHT, WIDTH, 3])
    anno = tf.reshape(anno, [HEIGHT, WIDTH])

    return image_0, image, anno

def augmentation_scale(image_0, image, anno, mmin, mmax, type):

    #if type == 'train' or type == 'val':
    #    image_0, image, anno = random_resize(image_0, image, anno, mmin, mmax)

    ''' for resize to small size
    scaled_shape = [tf.cast(tf.round(0.25 * HEIGHT), tf.int32), tf.cast(tf.round(0.25 * WIDTH), tf.int32)]
    image_0 = tf.image.resize_bilinear(image_0, scaled_shape)
    image = tf.image.resize_bilinear(image, scaled_shape)
    anno = tf.expand_dims(anno, -1)
    anno = tf.image.resize_nearest_neighbor(anno, scaled_shape)
    anno = tf.squeeze(anno, axis=-1)
    '''

    image_0, image, anno = random_crop(image_0, image, anno)

    return image_0, image, anno

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
    filename = features['image/filename']
    height = tf.cast(features['image/height'], tf.int32)
    width = tf.cast(features['image/width'], tf.int32)

    image = tf.reshape(image, [HEIGHT, WIDTH, 3])
    anno = tf.reshape(anno, [HEIGHT, WIDTH])

    return image, anno, filename

def read_batch(batch_size, type='train'):
    filelist_train = [ os.path.join(tfrecord_file, 'image_%s_%05d-of-%05d.tfrecord' % ('train', shard_id, _NUM_SHARDS - 1)) for
        shard_id in range(_NUM_SHARDS)]
    filelist_val = [os.path.join(tfrecord_file, 'image_%s_%05d-of-%05d.tfrecord' % ('val', shard_id, _NUM_SHARDS - 1))
                    for shard_id in range(_NUM_SHARDS)]
    filelist_test = [os.path.join(tfrecord_file, 'image_%s_%05d-of-%05d.tfrecord' % ('test', shard_id, _NUM_SHARDS - 1))
                     for shard_id in range(_NUM_SHARDS)]

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
    image_0 = image
    ## data augmentation and standardation

    image_0, image, anno = augmentation_standardization(image_0, image, anno, type)

    image_0_batch, image_batch, anno_batch, filename = tf.train.shuffle_batch([image_0, image, anno, filename], batch_size=batch_size,
                                                               capacity=128, min_after_dequeue=64, num_threads=4)

    # print(image_batch, anno_batch)
    image_0_batch, image_batch, anno_batch = augmentation_scale(image_0_batch, image_batch, anno_batch, mmin=0.5, mmax=2.0, type=type)
    return image_0_batch, image_batch, anno_batch, filename

if __name__ == '__main__':
    BATCH_SIZE = 4
    image_0, image_batch, anno_batch, filename = read_batch(BATCH_SIZE, type='val')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        b_image_0, b_image, b_anno, b_filename = sess.run([image_0, image_batch, anno_batch, filename])
        '''
        print(b_image_0.shape)
        print(b_image.shape)
        print(b_anno.shape)
        print(b_filename)

        print(b_image_0)
        print(b_image)
        print(b_anno)
        print(np.unique(b_anno))
        '''
        print(np.unique(b_anno))
        for i in range(BATCH_SIZE):
            cv2.imwrite('test/%d_img.png'%i, b_image_0[i])
            cv2.imwrite('test/%d_img_2.png' % i, 255 * (0.5 + b_image[i]))
            cv2.imwrite('test/%d_anno.png' % i, 10*b_anno[i])

        coord.request_stop()

        coord.join(threads)