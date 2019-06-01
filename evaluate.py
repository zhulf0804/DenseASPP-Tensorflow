# coding=utf-8
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
import os
from PIL import Image
import cv2
import datetime
import argparse
import glob
import matplotlib.pyplot as plt

import model.denseASPP as denseASPP
import model.resnet_aspp as resnet_aspp

import input_data
import utils.utils as Utils

flags = tf.app.flags
FLAGS = flags.FLAGS

CITYSCAPE_ANNO_DIR = input_data.CITYSCAPE_ANNO_DIR

# for dataset
flags.DEFINE_integer('height', 1024, 'The height of raw image.')
flags.DEFINE_integer('width', 2048, 'The width of raw image.')
flags.DEFINE_integer('batch_size', 1, 'Batch size.')
flags.DEFINE_integer('classes', 19, 'The number of classes')
flags.DEFINE_integer('ignore_label', 255, 'The ignore label value.')

flags.DEFINE_integer('val_num', 500, 'the number of val set.')
flags.DEFINE_integer('test_num', 1525, 'the number of test set.')

flags.DEFINE_string('dataset', 'val', 'which dataset to select to predict.(val or test).')


flags.DEFINE_integer('crop_height', 1024, 'The height of cropped image used for training.')
flags.DEFINE_integer('crop_width', 2048, 'The width of cropped image used for training.')
flags.DEFINE_integer('channels', 3, 'The channels of input image.')
#flags.DEFINE_multi_float('rgb_mean', [123.15,115.90,103.06], 'RGB mean value of ImageNet.')
flags.DEFINE_multi_float('rgb_mean', [72.39239876,82.90891754,73.15835921], 'RGB mean value of ImageNet.')

flags.DEFINE_multi_float('scales', [0.5,0.75,1.0,1.25,1.5,1.75,2.0], 'Scales for random scale.')

# for checkpoint
flags.DEFINE_string('pretrained_model_path', './resnet_v2_101_2017_04_14/resnet_v2_101.ckpt', 'Path to save pretrained model.')
flags.DEFINE_string('saved_ckpt_path', './checkpoint/', 'Path to load training checkpoint.')

# for network configration
flags.DEFINE_integer('output_stride', 8, 'output stride in the resnet model.')

# for saved configration

flags.DEFINE_string('saved_prediction', './pred/', 'Path to save predictions.')
flags.DEFINE_string('saved_prediction_val_color', './pred/val_color', 'Path to save val set color predictions.')
flags.DEFINE_string('saved_prediction_val_gray', './pred/val_gray', 'Path to save val set gray predictions.')
flags.DEFINE_string('saved_prediction_test_color', './pred/test_color', 'Path to save test set color predictions.')
flags.DEFINE_string('saved_prediction_test_gray', './pred/test_gray', 'Path to save test set gray predictions.')
flags.DEFINE_string('saved_submit_test_gray', './pred/submit_gray', 'Path to save test set gray submit version.')



#VAL_LIST = input_data.VAL_LIST
#ANNOTATION_PATH = input_data.ANNO_VAL_LIST

cityscapes_trainIds2labelIds = np.array([7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33], dtype=np.uint8)


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

if not os.path.exists(FLAGS.saved_prediction):
    os.mkdir(FLAGS.saved_prediction)

val_data = input_data.read_val_data(rgb_mean=FLAGS.rgb_mean, crop_height = FLAGS.crop_height, crop_width = FLAGS.crop_width, classes = FLAGS.classes, ignore_label = FLAGS.ignore_label, scales = FLAGS.scales)
test_data = input_data.read_test_data(rgb_mean=FLAGS.rgb_mean, crop_height = FLAGS.crop_height, crop_width = FLAGS.crop_width, classes = FLAGS.classes, ignore_label = FLAGS.ignore_label, scales = FLAGS.scales)

with tf.name_scope("input"):

    x = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.height, FLAGS.width, 3], name='x_input')
    y = tf.placeholder(tf.int32, [FLAGS.batch_size, FLAGS.height, FLAGS.width], name='ground_truth')

#logits = denseASPP.denseASPP(x, 1.0, train=False)
logits = resnet_aspp.denseASPP(x, is_training=False, output_stride=FLAGS.output_stride, pre_trained_model=FLAGS.pretrained_model_path, classes=FLAGS.classes)


with tf.name_scope('prediction_and_miou'):

    prediction = tf.argmax(logits, axis=-1, name='predictions')


def get_val_predictions():


    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        #saver.restore(sess, './checkpoint/deeplabv3plus.model-55000')

        ckpt = tf.train.get_checkpoint_state(FLAGS.saved_ckpt_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("Model restored...")

        print("predicting on val set...")

        for i in range(FLAGS.val_num):
            b_image_0, b_image, b_anno, b_filename = val_data.next_batch(FLAGS.batch_size, is_training=False, Shuffle=False)

            pred = sess.run(prediction, feed_dict={x: b_image})

            basename = b_filename.split('.')[0]

            if i % 100 == 0:
                print(i, pred.shape)
                print(basename)

            # save raw image, annotation, and prediction
            pred = pred.astype(np.uint8)
            b_anno = b_anno.astype(np.uint8)

            pred_color = color_gray(pred[0, :, :])
            b_anno_color = color_gray(b_anno[0, :, :])

            b_image_0 = b_image_0.astype(np.uint8)

            pred_gray = Image.fromarray(pred[0])
            #pred_gray = map(lambda x: cityscapes_trainIds2labelIds[x], np.arry(pred_gray))
            img = Image.fromarray(b_image_0[0])
            anno = Image.fromarray(b_anno_color)
            pred = Image.fromarray(pred_color)

            if not os.path.exists(FLAGS.saved_prediction_val_gray):
                os.mkdir(FLAGS.saved_prediction_val_gray)
            pred_gray.save(os.path.join(FLAGS.saved_prediction_val_gray, basename + '.png'))

            if not os.path.exists(FLAGS.saved_prediction_val_color):
                os.mkdir(FLAGS.saved_prediction_val_color)
            img.save(os.path.join(FLAGS.saved_prediction_val_color, basename + '_raw.png'))
            anno.save(os.path.join(FLAGS.saved_prediction_val_color, basename + '_anno.png'))
            pred.save(os.path.join(FLAGS.saved_prediction_val_color, basename + '_pred.png'))

    print("predicting on val set finished")

def get_test_predictions():

    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        #saver.restore(sess, './checkpoint/deeplabv3plus.model-55000')

        ckpt = tf.train.get_checkpoint_state(FLAGS.saved_ckpt_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("Model restored...")

        print("predicting on test set...")

        for i in range(FLAGS.test_num):
            b_image_0, b_image, b_anno, b_filename = test_data.next_batch(FLAGS.batch_size, is_training=False, Shuffle=False)

            pred = sess.run(prediction, feed_dict={x: b_image})

            basename = b_filename.split('.')[0]


            if i % 100 == 0:
                print(i, pred.shape)
                print(basename)

            # save raw image, annotation, and prediction
            pred = pred.astype(np.uint8)
            pred_color = color_gray(pred[0, :, :])
            b_anno_color = color_gray(b_anno[0, :, :])

            pred_gray = Image.fromarray(pred[0])

            b_image_0 = b_image_0.astype(np.uint8)

            img = Image.fromarray(b_image_0[0])
            pred = Image.fromarray(pred_color)


            if not os.path.exists(FLAGS.saved_prediction_test_gray):
                os.mkdir(FLAGS.saved_prediction_test_gray)
            pred_gray.save(os.path.join(FLAGS.saved_prediction_test_gray, basename + '.png'))
            pred_gray = map(lambda x: cityscapes_trainIds2labelIds[x], np.array(pred_gray))

            if not os.path.exists(FLAGS.saved_submit_test_gray):
                os.mkdir(FLAGS.saved_submit_test_gray)
            
            pred_gray.save(os.path.join(FLAGS.saved_submit_test_gray, basename + '.png'))

            if not os.path.exists(FLAGS.saved_prediction_test_color):
                os.mkdir(FLAGS.saved_prediction_test_color)
            img.save(os.path.join(FLAGS.saved_prediction_test_color, basename + '_raw.png'))
            pred.save(os.path.join(FLAGS.saved_prediction_test_color, basename + '_pred.png'))

    print("predicting on test set finished")


def get_val_mIoU():

    print("Start to get mIoU on val set...")

    #prediction_files = glob.glob(os.path.join(FLAGS.saved_prediction_val_gray, '*.png'))


    #f = open(VAL_LIST)
    #lines = f.readlines()
    annotation_files = glob.glob(os.path.join(CITYSCAPE_ANNO_DIR, 'val/*/*_gtFine_labelTrainIds.png'))
    prediction_files = [os.path.join(FLAGS.saved_prediction_val_gray, os.path.join(os.path.basename(filename))) for filename in annotation_files]
    prediction_files = [filename.replace('_gtFine_labelTrainIds.png', '_leftImg8bit.png') for filename in prediction_files]


    for i, annotation_file in enumerate(annotation_files):

        annotation_data = cv2.imread(annotation_file, cv2.IMREAD_GRAYSCALE)
        annotation_data = annotation_data.reshape(-1)
        if i == 0:
            annotations_data = annotation_data
        else:
            annotations_data = np.concatenate((annotations_data, annotation_data))

    print(annotations_data.shape)
    for i, prediction_file in enumerate(prediction_files):
        prediction_data = cv2.imread(prediction_file, cv2.IMREAD_GRAYSCALE)
        prediction_data = prediction_data.reshape(-1)
        if i == 0:
            predictions_data = prediction_data
        else:

            predictions_data = np.concatenate((predictions_data, prediction_data))

    print(predictions_data.shape)

    mIoU, IoU = Utils.cal_batch_mIoU(predictions_data, annotations_data, FLAGS.classes)

    print(mIoU)
    print(IoU)

if __name__ == '__main__':

    if FLAGS.dataset == 'val':
        get_val_predictions()
        get_val_mIoU()
    elif FLAGS.dataset == 'test':
        get_test_predictions()


