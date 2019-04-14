# coding=utf-8
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import model.denseASPP as denseASPP
import Cityscape.labels as Labels
import input_data

BATCH_SIZE = 1
HEIGHT = 360
WIDTH = 480
CLASSES = denseASPP.CLASSES
saved_ckpt_path = './checkpoint/'
saved_prediction = './pred/'
prediction_on = 'test' # 'train', 'val' or 'test'

cmap = Labels.trainId2Color
cmap[19] = (0, 0, 0) # add ignore class color
cmap[255] = (0, 0, 0)

def color_gray(image):
    height, width = image.shape

    return_img = np.zeros([height, width, 3], np.uint8)
    for i in range(height):
        for j in range(width):
            return_img[i, j, :] = cmap[image[i, j]]

    return return_img


image_batch, anno_batch, filename = input_data.read_batch(BATCH_SIZE, type=prediction_on)


with tf.name_scope("input"):

    x = tf.placeholder(tf.float32, [BATCH_SIZE, HEIGHT, WIDTH, 3], name='x_input')
    y = tf.placeholder(tf.int32, [BATCH_SIZE, HEIGHT, WIDTH], name='ground_truth')

logits = denseASPP.denseASPP(x, keep_prob=1.0, train=False)


with tf.name_scope('prediction_and_miou'):

    prediction = tf.argmax(logits, axis=-1, name='predictions')



with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    saver.restore(sess, './checkpoint/denseASPP.model-12000')

    #ckpt = tf.train.get_checkpoint_state(saved_ckpt_path)
    #if ckpt and ckpt.model_checkpoint_path:
    #    saver.restore(sess, ckpt.model_checkpoint_path)
    #    print("Model restored...")
    print("predicting on %s set..." % prediction_on)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for i in range(2):
        b_image_0, b_image, b_anno, b_filename = sess.run([image_batch, anno_batch, filename])

        pred = sess.run(prediction, feed_dict={x: b_image, y: b_anno})

        # save raw image, annotation, and prediction
        pred = pred.astype(np.uint8)
        b_anno = b_anno.astype(np.uint8)
        pred_color = color_gray(pred[0, :, :])
        b_anno_color = color_gray(b_anno[0, :, :])

        b_image_0 = b_image_0.astype(np.uint8)

        img = Image.fromarray(b_image_0[0])
        anno = Image.fromarray(b_anno_color)
        pred = Image.fromarray(pred_color)

        basename = b_filename[0].split('.')[0]
        #print(basename)

        if not os.path.exists(saved_prediction):
            os.mkdir(saved_prediction)
        img.save(os.path.join(saved_prediction, basename + '.png'))
        anno.save(os.path.join(saved_prediction, basename + '_anno.png'))
        pred.save(os.path.join(saved_prediction, basename + '_pred.png'))

        print("%s.png: prediction saved in %s" % (basename, saved_prediction))


    coord.request_stop()

    coord.join(threads)