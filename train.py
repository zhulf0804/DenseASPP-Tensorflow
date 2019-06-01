#coding=utf-8

from __future__ import print_function
from __future__ import division
from __future__ import  absolute_import

import tensorflow as tf
import numpy as np
import os
import cv2
import datetime
slim = tf.contrib.slim
import model.denseASPP as denseASPP
import model.resnet_aspp as resnet_aspp

import input_data
import utils.utils as Utils

flags = tf.app.flags
FLAGS = flags.FLAGS

# for dataset
flags.DEFINE_integer('height', 1024, 'The height of raw image.')
flags.DEFINE_integer('width', 2048, 'The width of raw image.')
flags.DEFINE_integer('crop_height', 512, 'The height of cropped image used for training.')
flags.DEFINE_integer('crop_width', 512, 'The width of cropped image used for training.')
flags.DEFINE_integer('channels', 3, 'The channels of input image.')
flags.DEFINE_integer('ignore_label', 255, 'The ignore label value.')
flags.DEFINE_integer('classes', 19, 'The ignore label value.')
#flags.DEFINE_multi_float('rgb_mean', [123.15,115.90,103.06], 'RGB mean value of ImageNet.')
flags.DEFINE_multi_float('rgb_mean', [72.39239876,82.90891754,73.15835921], 'RGB mean value of ImageNet.')

flags.DEFINE_string('dataset', 'train', 'which dataset to select to train.(train or trainval).')

# for augmentation
flags.DEFINE_boolean('train_random_scales', True, 'whether to random scale.')
flags.DEFINE_multi_float('scales', [0.5,0.75,1.0,1.25,1.5,1.75,2.0], 'Scales for random scale.')
flags.DEFINE_boolean('train_random_mirror', True, 'whether to random mirror.')

flags.DEFINE_boolean('val_random_scales', False, 'whether to random scale.')
flags.DEFINE_boolean('val_random_mirror', False, 'whether to random mirror.')

# for training configuration
flags.DEFINE_integer('batch_size', 4, 'The number of images in each batch during training.')
flags.DEFINE_integer('max_epoches',120, 'The max epoches to train the model.')
#flags.DEFINE_integer('samples', 2975, 'The number of images in train set used to train.')
flags.DEFINE_integer('train_samples', 2975, 'The number of images in train set used to train.')
flags.DEFINE_integer('trainval_samples', 3475, 'The number of images in trainval set used to train.')

# for network configration
flags.DEFINE_integer('output_stride', 8, 'output stride in the resnet model.')

# network hyper-parameters
flags.DEFINE_float('initial_lr', 1e-2, 'The initial learning rate.')
flags.DEFINE_float('end_lr', 1e-6, 'The end learning rate.')
flags.DEFINE_float('keep_prob', 0.9, 'Keep probability.')

flags.DEFINE_integer('decay_steps', 70000, 'Used for poly learning rate.')
flags.DEFINE_float('weight_decay', 1e-4, 'The weight decay value for l2 regularization.')
flags.DEFINE_float('power', 0.9, 'Used for poly learning rate.')

# for saved configration
flags.DEFINE_string('saved_ckpt_path', './checkpoint/', 'Path to save training checkpoint.')
flags.DEFINE_string('saved_summary_train_path', './summary/train/', 'Path to save training summary.')
flags.DEFINE_string('saved_summary_test_path', './summary/test/', 'Path to save test summary.')
flags.DEFINE_integer('print_steps', 200, 'Used for print training information.')
flags.DEFINE_integer('saved_steps', 5000, 'Used for saving model.')
flags.DEFINE_integer('iou_steps', 1000, 'Used for print mIoU information.')
flags.DEFINE_string('pretrained_model_path', './resnet_v2_101_2017_04_14/resnet_v2_101.ckpt', 'Path to save pretrained model.')


def cal_loss(logits, y, loss_weight=1.0):
    '''
    raw_prediction = tf.reshape(logits, [-1, CLASSES])
    raw_gt = tf.reshape(y, [-1])
    indices = tf.squeeze(tf.where(tf.less_equal(raw_gt, CLASSES - 1)), 1)
    gt = tf.cast(tf.gather(raw_gt, indices), tf.int32)
    prediction = tf.gather(raw_prediction, indices)
    # Pixel-wise softmax loss.
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=gt)
    '''

    y = tf.reshape(y, shape=[-1])
    not_ignore_mask = tf.to_float(tf.not_equal(y,
                                               FLAGS.ignore_label)) * loss_weight
    one_hot_labels = tf.one_hot(
        y, FLAGS.classes, on_value=1.0, off_value=0.0)
    logits = tf.reshape(logits, shape=[-1, FLAGS.classes])
    loss = tf.losses.softmax_cross_entropy(onehot_labels=one_hot_labels, logits=logits, weights=not_ignore_mask)

    return tf.reduce_mean(loss)

if FLAGS.dataset == 'train':
    print('training on train set')
    MAX_STEPS = FLAGS.max_epoches * FLAGS.train_samples // FLAGS.batch_size
    train_data = input_data.read_train_data(rgb_mean=FLAGS.rgb_mean, crop_height = FLAGS.crop_height, crop_width = FLAGS.crop_width, classes = FLAGS.classes, ignore_label = FLAGS.ignore_label, scales = FLAGS.scales)
    val_data = input_data.read_val_data(rgb_mean=FLAGS.rgb_mean, crop_height = FLAGS.crop_height, crop_width = FLAGS.crop_width, classes = FLAGS.classes, ignore_label = FLAGS.ignore_label, scales = FLAGS.scales)
elif FLAGS.dataset == 'trainval':
    print('training on trainval set')
    MAX_STEPS = FLAGS.max_epoches * FLAGS.trainval_samples // FLAGS.batch_size
    trainval_data = input_data.read_trainval_data(rgb_mean=FLAGS.rgb_mean, crop_height = FLAGS.crop_height, crop_width = FLAGS.crop_width, classes = FLAGS.classes, ignore_label = FLAGS.ignore_label, scales = FLAGS.scales)
else:
    raise Exception('train or trainval is needed')

with tf.name_scope('input'):
    x = tf.placeholder(dtype=tf.float32, shape=[FLAGS.batch_size, FLAGS.crop_height, FLAGS.crop_width, FLAGS.channels], name='x_input')
    y = tf.placeholder(dtype=tf.int32, shape=[FLAGS.batch_size, FLAGS.crop_height, FLAGS.crop_width], name='ground_truth')
    keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')

logits = resnet_aspp.denseASPP(x, is_training=True, output_stride=FLAGS.output_stride, pre_trained_model=FLAGS.pretrained_model_path, classes=FLAGS.classes)


with tf.name_scope('regularization'):
    train_var_list = [v for v in tf.trainable_variables()
                      if 'beta' not in v.name and 'gamma' not in v.name]
    # Add weight decay to the loss.
    with tf.variable_scope("total_loss"):
        l2_loss = FLAGS.weight_decay * tf.add_n(
            [tf.nn.l2_loss(v) for v in train_var_list])

with tf.name_scope('loss'):
    #reshaped_logits = tf.reshape(logits, [BATCH_SIZE, -1])
    #reshape_y = tf.reshape(y, [BATCH_SIZE, -1])
    #loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=reshape_y, logits=reshaped_logits), name='loss')
    #loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits), name='loss')
    loss = cal_loss(logits, y)
    tf.summary.scalar('loss', loss)
    loss_all = loss + l2_loss
    #loss_all = loss
    tf.summary.scalar('loss_all', loss_all)

with tf.name_scope('learning_rate'):
    global_step = tf.Variable(0, trainable=False)
    lr = tf.train.polynomial_decay(
        learning_rate=FLAGS.initial_lr,
        global_step=global_step,
        decay_steps=FLAGS.decay_steps,
        end_learning_rate=FLAGS.end_lr,
        power=FLAGS.power,
        cycle=False,
        name=None
    )
    tf.summary.scalar('learning_rate', lr)

with tf.name_scope("opt"):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9).minimize(loss_all, var_list=train_var_list, global_step=global_step)


with tf.name_scope("mIoU"):
    softmax = tf.nn.softmax(logits, axis=-1)
    predictions = tf.argmax(softmax, axis=-1, name='predictions')

    train_mIoU = tf.Variable(0, dtype=tf.float32, trainable=False)
    tf.summary.scalar('train_mIoU', train_mIoU)
    test_mIoU = tf.Variable(0, dtype=tf.float32, trainable=False)
    tf.summary.scalar('test_mIoU',test_mIoU)

merged = tf.summary.merge_all()


with tf.Session() as sess:


    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()

    # if os.path.exists(saved_ckpt_path):
    ckpt = tf.train.get_checkpoint_state(FLAGS.saved_ckpt_path)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")

    # saver.restore(sess, './checkpoint/PSPNet.model-30000')

    train_summary_writer = tf.summary.FileWriter(FLAGS.saved_summary_train_path, sess.graph)
    test_summary_writer = tf.summary.FileWriter(FLAGS.saved_summary_test_path, sess.graph)

    for i in range(0, MAX_STEPS + 1):


        if FLAGS.dataset == 'train':
            b_image_0, b_image, b_anno, b_filename = train_data.next_batch(FLAGS.batch_size, FLAGS.train_random_scales, FLAGS.train_random_mirror, is_training=True)
            b_image_test_0, b_image_test, b_anno_test, b_filename_test = val_data.next_batch(FLAGS.batch_size, FLAGS.val_random_scales, FLAGS.val_random_mirror, is_training=True)
        elif FLAGS.dataset == 'trainval':
            b_image_0, b_image, b_anno, b_filename = trainval_data.next_batch(FLAGS.batch_size, FLAGS.train_random_scales,
                                                                           FLAGS.train_random_mirror, is_training=True)
        _ = sess.run(optimizer, feed_dict={x: b_image, y: b_anno, keep_prob: FLAGS.keep_prob})

        train_summary = sess.run(merged, feed_dict={x: b_image, y: b_anno, keep_prob: 1.0})
        train_summary_writer.add_summary(train_summary, i)

        if FLAGS.dataset == 'train':
            test_summary = sess.run(merged, feed_dict={x: b_image_test, y: b_anno_test, keep_prob: 1.0})
            test_summary_writer.add_summary(test_summary, i)

        pred_train, train_loss_val_all, train_loss_val = sess.run([predictions, loss_all, loss], feed_dict={x: b_image, y: b_anno, keep_prob: 1.0})

        if FLAGS.dataset == 'train':
            pred_test, test_loss_val_all, test_loss_val = sess.run([predictions, loss_all, loss], feed_dict={x: b_image_test, y: b_anno_test, keep_prob: 1.0})



        learning_rate = sess.run(lr)

        if i % FLAGS.print_steps == 0 and FLAGS.dataset == 'train':
            print(datetime.datetime.now().strftime("%Y.%m.%d-%H:%M:%S"), " | Step: %d | Train loss all: %f" % (i, train_loss_val_all))

        if i % FLAGS.print_steps == 0 and FLAGS.dataset == 'trainval':
            print(datetime.datetime.now().strftime("%Y.%m.%d-%H:%M:%S"), " | Step: %d | Train loss all: %f" % (i, train_loss_val_all))


        if i % FLAGS.iou_steps == 0 and FLAGS.dataset == 'train':

            train_mIoU_val, train_IoU_val = Utils.cal_batch_mIoU(pred_train, b_anno, FLAGS.classes)
            test_mIoU_val, test_IoU_val = Utils.cal_batch_mIoU(pred_test, b_anno_test, FLAGS.classes)

            sess.run(tf.assign(train_mIoU, train_mIoU_val))
            sess.run(tf.assign(test_mIoU, test_mIoU_val))

            print('------------------------------')

            print(
                "Step: %d | Lr: %f | Train loss all: %f | Train loss: %f | Train mIoU: %f | Test loss all: %f | Test loss: %f | Test mIoU: %f" % (
                i, learning_rate, train_loss_val_all, train_loss_val, train_mIoU_val, test_loss_val_all, test_loss_val, test_mIoU_val))
            print('------------------------------')
            print(train_IoU_val)
            print(test_IoU_val)
            print('------------------------------')
            #prediction = tf.argmax(logits, axis=-1, name='predictions')


        if i % FLAGS.saved_steps == 0:
            saver.save(sess, os.path.join(FLAGS.saved_ckpt_path, 'pspnet.model'), global_step=i)





# python train.py --dataset trainval --samples 3475
# python train.py