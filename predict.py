# coding=utf-8
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
import os
from PIL import Image
import datetime

import matplotlib.pyplot as plt
import model.denseASPP as denseASPP
import cv2
import input_data
import utils.utils as Utils


flags = tf.app.flags
FLAGS = flags.FLAGS

# for dataset
flags.DEFINE_integer('height', 1024, 'The height of raw image.')
flags.DEFINE_integer('width', 2048, 'The width of raw image.')
flags.DEFINE_integer('batch_size', 1, 'Batch size.')
flags.DEFINE_integer('classes', 19, 'The number of classes')
flags.DEFINE_integer('ignore_label', 255, 'The ignore label value.')
flags.DEFINE_integer('crop_height', 1024, 'The height of cropped image used for training.')
flags.DEFINE_integer('crop_width', 2048, 'The width of cropped image used for training.')
flags.DEFINE_integer('channels', 3, 'The channels of input image.')

#flags.DEFINE_multi_float('rgb_mean', [123.15,115.90,103.06], 'RGB mean value of ImageNet.')
flags.DEFINE_multi_float('rgb_mean', [72.39239876,82.90891754,73.15835921], 'RGB mean value of ImageNet.')

flags.DEFINE_multi_float('scales', [0.5,0.75,1.0,1.25,1.5,1.75,2.0], 'Scales for random scale.')

# for prediction
flags.DEFINE_string('file_path', '', 'The file path to be predicted.')

# for checkpoint
flags.DEFINE_string('pretrained_model_path', './resnet_v2_101_2017_04_14/resnet_v2_101.ckpt', 'Path to save pretrained model.')
flags.DEFINE_string('saved_ckpt_path', './checkpoint/', 'Path to load training checkpoint.')

# for network configration
flags.DEFINE_integer('output_stride', 8, 'output stride in the resnet model.')

# for saved configration
flags.DEFINE_enum('prediction_on', 'val', ['train', 'val', 'test'], 'Which dataset to predict.')
flags.DEFINE_string('saved_prediction', './pred/', 'Path to save predictions.')

cmap = input_data.label_colours

def color_gray(image):
    height, width = image.shape

    return_img = np.zeros([height, width, 3], np.uint8)
    for i in range(height):
        for j in range(width):
            if image[i, j] == FLAGS.ignore_label:
                return_img[i, j, :] = (0, 0, 0)
            else:
                return_img[i, j, :] = cmap[image[i, j]]

    return return_img

#val_data = input_data.read_val_data()
val_data = input_data.read_val_data(rgb_mean=FLAGS.rgb_mean, crop_height = FLAGS.crop_height, crop_width = FLAGS.crop_width, classes = FLAGS.classes, ignore_label = FLAGS.ignore_label, scales = FLAGS.scales)

with tf.name_scope("input"):

    x = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.height, FLAGS.width, 3], name='x_input')
    y = tf.placeholder(tf.int32, [FLAGS.batch_size, FLAGS.height, FLAGS.width], name='ground_truth')

logits = denseASPP.denseASPP(x, 1.0, train=False)


with tf.name_scope('prediction_and_miou'):

    prediction = tf.argmax(logits, axis=-1, name='predictions')

with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    #saver.restore(sess, './checkpoint/pspnet.model-2000')

    ckpt = tf.train.get_checkpoint_state(FLAGS.saved_ckpt_path)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")


    for i in range(1):

        if FLAGS.file_path == '':
            print("predicting on %s set..." % FLAGS.prediction_on)
            b_image_0, b_image, _, b_filename = val_data.next_batch(FLAGS.batch_size, is_training=False)
        else:
            b_image_0 = cv2.imread(FLAGS.file_path)

            b_image = b_image_0[:, :, ::-1]
            b_image = b_image.astype(np.float32)
            b_image = input_data.mean_substraction(b_image, FLAGS.rgb_mean)

            b_image_0 = np.expand_dims(b_image_0, 0)
            b_image = np.expand_dims(b_image, 0)
            b_filename = os.path.basename(FLAGS.file_path)

        pred = sess.run(prediction, feed_dict={x: b_image})
        print(pred.shape)


        # save raw image, annotation, and prediction
        pred = pred.astype(np.uint8)

        pred_color = color_gray(pred[0, :, :])


        b_image_0 = b_image_0.astype(np.uint8)

        img = Image.fromarray(b_image_0[0])

        pred = Image.fromarray(pred_color)

        basename = b_filename.split('.')[0]
        #print(basename)

        if not os.path.exists(FLAGS.saved_prediction):
            os.mkdir(FLAGS.saved_prediction)
        img.save(os.path.join(FLAGS.saved_prediction, basename + '.png'))

        pred.save(os.path.join(FLAGS.saved_prediction, basename + '_pred.png'))

        print("%s.png: prediction saved in %s" % (basename, FLAGS.saved_prediction))


# python predict.py --prediction_on test
# python predict.py --prediction_on val
# python predict.py --prediction_on train

# python predict.py --file_path /Volumes/Samsung_T5/datasets/Cityscape/leftImg8bit_trainvaltest/leftImg8bit/test/berlin/berlin_000270_000019_leftImg8bit.png