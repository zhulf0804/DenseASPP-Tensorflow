# coding=utf-8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import numpy as np
from sklearn.utils import shuffle
import cv2
import random
from random import choice
import math
import cityscape


CITYSCAPE_IMG_DIR = cityscape.CITYSCAPE_IMG_DIR
CITYSCAPE_ANNO_DIR = cityscape.CITYSCAPE_ANNO_DIR

# colour map
label_colours = [(128, 64,128),  # 0=background
                 # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                 (244, 35,232), (70, 70, 70), (102,102,156), (190,153,153), (153,153,153),
                 # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                 (250,170, 30), (220,220,  0), (107,142, 35), (152,251,152), (70,130,180),
                 # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
                 (220, 20, 60), (255,  0,  0), (0,  0,142), (0,  0, 70), (0, 60,100),
                 # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
                 (0, 80,100), (0,  0,230), (119, 11, 32)]



dataset = cityscape.CITYSCAPE_DIR # Select your path

IMG_TRAIN_LIST = os.path.join(dataset, 'img_train.txt')
IMG_VAL_LIST = os.path.join(dataset, 'img_val.txt')
IMG_TEST_LIST = os.path.join(dataset, 'img_test.txt')

ANNO_TRAIN_LIST = os.path.join(dataset, 'anno_train.txt')
ANNO_VAL_LIST = os.path.join(dataset, 'anno_val.txt')
ANNO_TEST_LIST = os.path.join(dataset, 'anno_test.txt')


def flip_random_left_right(image, anno):
    '''
    :param image: [height, width, channel]
    :return:
    '''
    flag = random.randint(0, 1)
    if flag:
        return cv2.flip(image, 1), cv2.flip(anno, 1)
    return image, anno


def random_pad_crop(image, anno, crop_height, crop_width, ignore_label, rgb_mean):

    image = image.astype(np.float32)

    height, width = anno.shape

    #padded_image = np.pad(image, ((0, np.maximum(height, HEIGHT) - height), (0, np.maximum(width, WIDTH) - width), (0, 0)), mode='constant', constant_values=_MEAN_RGB)

    padded_image_r = np.pad(image[:, :, 0], ((0, np.maximum(height, crop_height) - height), (0, np.maximum(width, crop_width) - width)), mode='constant', constant_values=rgb_mean[0])
    padded_image_g = np.pad(image[:, :, 1], ((0, np.maximum(height, crop_height) - height), (0, np.maximum(width, crop_width) - width)), mode='constant', constant_values=rgb_mean[1])
    padded_image_b = np.pad(image[:, :, 2], ((0, np.maximum(height, crop_height) - height), (0, np.maximum(width, crop_width) - width)), mode='constant', constant_values=rgb_mean[2])
    padded_image = np.zeros(shape=[np.maximum(height, crop_height), np.maximum(width, crop_width), 3], dtype=np.float32)
    padded_image[:, :, 0] = padded_image_r
    padded_image[:, :, 1] = padded_image_g
    padded_image[:, :, 2] = padded_image_b

    padded_anno = np.pad(anno, ((0, np.maximum(height, crop_height) - height), (0, np.maximum(width, crop_width) - width)), mode='constant', constant_values=ignore_label)

    y = random.randint(0, np.maximum(height, crop_height) - crop_height)
    x = random.randint(0, np.maximum(width, crop_width) - crop_width)

    cropped_image = padded_image[y:y+crop_height, x:x+crop_width, :]
    cropped_anno = padded_anno[y:y+crop_height, x:x+crop_width]

    return cropped_image, cropped_anno


def random_resize(image, anno, scales):
    height, width = anno.shape

    scale = choice(scales)
    scale_image = cv2.resize(image, (int(scale * width), int(scale * height)), interpolation=cv2.INTER_LINEAR)
    scale_anno = cv2.resize(anno, (int(scale * width), int(scale * height)), interpolation=cv2.INTER_NEAREST)

    return scale_image, scale_anno


def mean_substraction(image, rgb_mean):
    substraction_mean_image = np.zeros_like(image, dtype=np.float32)
    substraction_mean_image[:, :, 0] = image[:, :, 0] - rgb_mean[0]
    substraction_mean_image[:, :, 1] = image[:, :, 1] - rgb_mean[1]
    substraction_mean_image[:, :, 2] = image[:, :, 2] - rgb_mean[2]

    return substraction_mean_image


def augment(img, anno, crop_height, crop_width, ignore_label, random_scales, scales, random_mirror,  rgb_mean):

    if random_scales:
        scale_img, scale_anno = random_resize(img, anno, scales)
    else:
        scale_img, scale_anno = img, anno

    scale_img = scale_img.astype(np.float32)

    cropped_image, cropped_anno = random_pad_crop(scale_img, scale_anno, crop_height, crop_width, ignore_label, rgb_mean)

    if random_mirror:
        cropped_image, cropped_anno = flip_random_left_right(cropped_image, cropped_anno)

    substracted_img = mean_substraction(cropped_image, rgb_mean)

    return substracted_img, cropped_anno


class Dataset(object):

    def __init__(self, img_filenames, anno_filenames, rgb_mean, crop_height, crop_width, classes, ignore_label, scales):
        self._num_examples = len(anno_filenames)
        self._image_data = img_filenames
        self._labels = anno_filenames
        self._epochs_done = 0
        self._index_in_epoch = 0
        self._flag = 0
        self._rgb_mean = rgb_mean
        self._crop_height = crop_height
        self._crop_width = crop_width
        self._classes = classes
        self._ignore_label = ignore_label
        self._scales = scales

    def next_batch(self, batch_size, random_scales=False, random_mirror=False, is_training=False, Shuffle=True):

        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            self._epochs_done += 1
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples

            if Shuffle:
                self._image_data, self._labels = shuffle(self._image_data, self._labels)

        end = self._index_in_epoch

        batch_img_raw = np.zeros([batch_size, self._crop_height, self._crop_width, 3], dtype=np.float32)
        batch_img = np.zeros([batch_size, self._crop_height, self._crop_width, 3], dtype=np.float32)
        batch_anno = np.zeros([batch_size, self._crop_height, self._crop_width], dtype=np.uint8)
        filenames = []
        for i in range(start, end):
            img = cv2.imread(self._image_data[i])
            img = img[:,:,::-1]
            anno = cv2.imread(self._labels[i], cv2.IMREAD_GRAYSCALE)

            if is_training:
                aug_img, aug_anno = augment(img, anno, self._crop_height, self._crop_width, self._ignore_label, random_scales, self._scales, random_mirror, self._rgb_mean)

                height, width, _ = img.shape
                batch_img_raw[i-start, 0:np.minimum(height, self._crop_height), 0:np.minimum(width, self._crop_width), :] = img[0:np.minimum(height, self._crop_height), 0:np.minimum(width, self._crop_width), :]
                batch_img[i-start, ...] = aug_img
                batch_anno[i-start, ...] = aug_anno
                filenames.append(os.path.basename(self._image_data[i]))
            #print(os.path.basename(self._image_data[i]), os.path.basename(self._labels[i]))
        if is_training:
            return batch_img_raw, batch_img, batch_anno, filenames
        else:
            inference_image = mean_substraction(img, self._rgb_mean)
            #inference_image, anno = random_pad_crop(inference_image, anno, self._crop_height, self._crop_width, self._ignore_label)
            #print(os.path.basename(self._image_data[start]))
            return np.expand_dims(img, 0), np.expand_dims(inference_image, 0), np.expand_dims(anno, 0), os.path.basename(self._image_data[start])


def read_train_data(rgb_mean, crop_height, crop_width, classes, ignore_label, scales, Shuffle=True):
    f = open(IMG_TRAIN_LIST)
    lines = f.readlines()
    img_filenames = [line.strip() for line in lines]

    anno_filenames = [filename.replace(CITYSCAPE_IMG_DIR, CITYSCAPE_ANNO_DIR) for filename in img_filenames]
    anno_filenames = [filename.replace('_leftImg8bit.png', '_gtFine_labelTrainIds.png') for filename in anno_filenames]

    if Shuffle:
        img_filenames, anno_filenames = shuffle(img_filenames, anno_filenames)

    train_data = Dataset(img_filenames, anno_filenames, rgb_mean, crop_height, crop_width, classes, ignore_label, scales)

    return train_data

def read_trainval_data(rgb_mean, crop_height, crop_width, classes, ignore_label, scales, Shuffle=True):
    train_f = open(IMG_TRAIN_LIST)
    train_lines = train_f.readlines()
    train_img_filenames = [line.strip() for line in train_lines]

    train_anno_filenames = [filename.replace(CITYSCAPE_IMG_DIR, CITYSCAPE_ANNO_DIR) for filename in train_img_filenames]
    train_anno_filenames = [filename.replace('_leftImg8bit.png', '_gtFine_labelTrainIds.png') for filename in train_anno_filenames]

    val_f = open(IMG_VAL_LIST)
    val_lines = val_f.readlines()
    val_img_filenames = [line.strip() for line in val_lines]


    val_anno_filenames = [filename.replace(CITYSCAPE_IMG_DIR, CITYSCAPE_ANNO_DIR) for filename in val_img_filenames]
    val_anno_filenames = [filename.replace('_leftImg8bit.png', '_gtFine_labelTrainIds.png') for filename in val_anno_filenames]

    img_filenames = train_img_filenames + val_img_filenames
    anno_filenames = train_anno_filenames +  val_anno_filenames

    if Shuffle:
        img_filenames, anno_filenames = shuffle(img_filenames, anno_filenames)

    trainval_data = Dataset(img_filenames, anno_filenames, rgb_mean, crop_height, crop_width, classes, ignore_label, scales)

    return trainval_data


def read_val_data(rgb_mean, crop_height, crop_width, classes, ignore_label, scales, Shuffle=True):
    f = open(IMG_VAL_LIST)
    lines = f.readlines()
    img_filenames = [line.strip() for line in lines]
    #f_anno = open(ANNO_VAL_LIST)
    #anno_filenames = f_anno.readlines()

    anno_filenames = [filename.replace(CITYSCAPE_IMG_DIR, CITYSCAPE_ANNO_DIR) for filename in img_filenames]
    anno_filenames = [filename.replace('_leftImg8bit.png', '_gtFine_labelTrainIds.png') for filename in anno_filenames]

    if Shuffle:
        img_filenames, anno_filenames = shuffle(img_filenames, anno_filenames)

    val_data = Dataset(img_filenames, anno_filenames, rgb_mean, crop_height, crop_width, classes, ignore_label, scales)

    return val_data

def read_test_data(rgb_mean, crop_height, crop_width, classes, ignore_label, scales, Shuffle=True):
    f = open(IMG_TEST_LIST)
    lines = f.readlines()
    img_filenames = [line.strip() for line in lines]

    anno_filenames = [filename.replace(CITYSCAPE_IMG_DIR, CITYSCAPE_ANNO_DIR) for filename in img_filenames]
    anno_filenames = [filename.replace('_leftImg8bit.png', '_gtFine_labelTrainIds.png') for filename in anno_filenames]

    if Shuffle:
        img_filenames, anno_filenames = shuffle(img_filenames, anno_filenames)

    test_data = Dataset(img_filenames, anno_filenames, rgb_mean, crop_height, crop_width, classes, ignore_label, scales)

    return test_data


if __name__ == '__main__':
    train_data = read_train_data()
    test_data = read_val_data()
    train_img_raw, train_img_data, train_lables, train_filenames = train_data.next_batch(4, True)
    test_img_raw, test_img_data, test_labels, test_filenames = test_data.next_batch(1)

    for i in range(4):
        cv2.imwrite('test/trainraw_%d.png' % i, train_img_raw[i])
        cv2.imwrite('test/train_%d.png'%i, train_img_data[i])
        cv2.imwrite('test/train_labels_%d.png'%i, train_lables[i])
        print(train_filenames[i])

    print("===============")

    for i in range(1):
        cv2.imwrite('test/testraw_%d.png' % i, test_img_raw[i])

        cv2.imwrite('test/test_%d.png' % i, test_img_data[i])
        cv2.imwrite('test/test_labels_%d.png' % i, test_labels[i])
        print(test_filenames)
