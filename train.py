# coding=utf-8
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
import os
import model.denseASPP as denseASPP
import model.densenet as DenseNet
import input_data

BATCH_SIZE = 8
HEIGHT = 512
WIDTH = 512
CHANNELS = 3
MAX_STEPS = 80*3000
# 6000 steps for one epoch
KEEP_PROB = 1.0

initial_lr = 3e-4
weight_decay = 1e-5

saved_ckpt_path = './checkpoint/'
saved_summary_train_path = './summary/train/'
saved_summary_test_path = './summary/test/'



with tf.name_scope('input'):
    x = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, HEIGHT, WIDTH, CHANNELS], name='x_input')
    y = tf.placeholder(dtype=tf.int32, shape=[BATCH_SIZE, HEIGHT, WIDTH], name='ground_truth')
    keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')

logits = denseASPP.denseASPP(x, keep_prob, train=True)

with tf.name_scope('regularization'):
    regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    reg_term = tf.contrib.layers.apply_regularization(regularizer)

with tf.name_scope('loss'):
    #reshaped_logits = tf.reshape(logits, [BATCH_SIZE, -1])
    #reshape_y = tf.reshape(y, [BATCH_SIZE, -1])
    #loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=reshape_y, logits=reshaped_logits), name='loss')
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits), name='loss')
    tf.summary.scalar('loss', loss)
    #loss_all = loss + reg_term
    loss_all = loss
    tf.summary.scalar('loss_all', loss_all)

with tf.name_scope('learning_rate'):
    lr = tf.Variable(initial_lr, dtype=tf.float32)
    tf.summary.scalar('learning_rate', lr)

optimizer = tf.train.AdamOptimizer(lr).minimize(loss_all)

merged = tf.summary.merge_all()

_, image_batch, anno_batch, filename = input_data.read_batch(BATCH_SIZE, type = 'train')
_, image_batch_test, anno_batch_test, filename_test = input_data.read_batch(BATCH_SIZE, type = 'val')

with tf.Session() as sess:

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()

    # if os.path.exists(saved_ckpt_path):
    ckpt = tf.train.get_checkpoint_state(saved_ckpt_path)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")

    # saver.restore(sess, './checkpoint/denseASPP.model-30000')

    train_summary_writer = tf.summary.FileWriter(saved_summary_train_path, sess.graph)
    test_summary_writer = tf.summary.FileWriter(saved_summary_test_path, sess.graph)

    for i in range(0, MAX_STEPS + 1):

        b_image, b_anno, b_filename = sess.run([image_batch, anno_batch, filename])

        b_image_test, b_anno_test, b_filename_test = sess.run([image_batch_test, anno_batch_test, filename_test])


        _ = sess.run(optimizer, feed_dict={x: b_image, y: b_anno, keep_prob: KEEP_PROB})

        train_summary = sess.run(merged, feed_dict={x: b_image, y: b_anno, keep_prob: 1.0})
        train_summary_writer.add_summary(train_summary, i)
        test_summary = sess.run(merged, feed_dict={x: b_image_test, y: b_anno_test, keep_prob: 1.0})
        test_summary_writer.add_summary(test_summary, i)

        train_loss_val_all, train_loss_val = sess.run([loss_all, loss], feed_dict={x: b_image, y: b_anno, keep_prob: 1.0})
        test_loss_val_all, test_loss_val = sess.run([loss_all, loss], feed_dict={x: b_image_test, y: b_anno_test, keep_prob: 1.0})

        learning_rate = sess.run(lr)

        if i % 10 == 0:
            print(
                "train step: %d, learning rate: %f, train loss all: %f, train loss: %f, test loss all: %f, test loss: %f" % (
                i, learning_rate, train_loss_val_all, train_loss_val, test_loss_val_all, test_loss_val))

        if i % 6000 == 0:
            saver.save(sess, os.path.join(saved_ckpt_path, 'denseASPP.model'), global_step=i)

        if i != 0 and i % 3000 == 0:
            sess.run(tf.assign(lr, pow((1 - 1.0*i/MAX_STEPS), 0.9) * lr))

    coord.request_stop()
    coord.join(threads)
