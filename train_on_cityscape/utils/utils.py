# coding=utf-8
from __future__ import print_function
from __future__ import division

import cv2
import numpy as np
import sys
sys.path.append("..")
import Cityscape.labels as Labels

ANNO_TRAIN_FILE = '../Cityscape/anno_train.txt'

CLASS_NAMES = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']


def cal_classes_weight():
    '''
    f = open(ANNO_TRAIN_FILE, 'r')
    CLASSES_NUM = 20
    CLASSES = [0 for i in range(CLASSES_NUM)]
    lines = f.readlines()
    for line in lines:
        img = cv2.imread(line.strip(), cv2.IMREAD_GRAYSCALE)
        img = Labels.id_to_trainId_map_func(img)
        for i in range(CLASSES_NUM):
            CLASSES[i] += np.sum(img == i)

    print(CLASSES)

    '''

    CLASSES = [2036416525, 336090793, 1260636120, 36199498, 48454166, 67789506, 11477088, 30448193, 879783988, 63949536, 221979646, 67326424, 7463162, 386328286, 14772328, 12990290, 12863955, 5449152, 22861233, 715747311]
    CLASSES = np.array(CLASSES)
    CLASSES = CLASSES / np.sum(CLASSES)
    CLASSES = 1 / CLASSES
    CLASSES = CLASSES / np.sum(CLASSES)
    CLASSES *= 20
    print(CLASSES)
    '''
    loss_weight = [0.0121096, 0.07337357, 0.0195617, 0.68122994, 0.50893832, 0.3637758, 2.14864449, \
                   0.80990625, 0.02802981, 0.3856194, 0.11109209, 0.36627791, 3.30425387, 0.06383219, \
                   1.66934974, 1.89835499, 1.91699846, 4.52550817, 1.07868993, 0.03445375]
    '''

def cal_batch_mIoU(pred, gt, classes_num):
    """

    :param pred: [batch, height, width]
    :param gt: [batch, height, width]
    :param classes_num:
    :return:
    """
    IoU_0 = []
    IoU = []
    eps = 1e-6
    for i in range(classes_num):
        a = np.sum(pred == i)
        b = np.sum(gt == i)
        c = [pred == i, gt == i]
        c = np.sum(np.all(c, 0))
        iou = c / (a + b - c + eps)
        if b != 0:
            IoU.append(iou)
        IoU_0.append(round(iou, 2))

    IoU_0 = dict(zip(CLASS_NAMES, IoU_0))
    mIoU = np.mean(IoU)
    return mIoU, IoU_0

if __name__ == '__main__':
    cal_classes_weight()
