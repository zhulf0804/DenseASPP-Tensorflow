# coding=utf-8
from __future__ import print_function
from __future__ import division

import numpy as np
from PIL import Image
import os
import glob

CLASSES_NAME = ['Sky', 'Building', 'Pole', 'Road', 'Pavement', 'Tree', 'SignSymbol', 'Fence', 'Car', 'Pedestrian', 'Bicyclist']


def classes_core(dir):
    CLASSES = 33
    images = glob.glob(os.path.join(dir, '*g'))

    nums = np.zeros([CLASSES], np.uint8)

    for image in images:
        img = Image.open(image)
        img_np = np.array(img)
        mmax = np.max(img_np)
        for i in range(mmax + 1):
            num = np.sum(img_np == i)
            nums[i] += num

    print("%s information" %dir)

    for i in range(CLASSES):
        print("%d: %d" %(i, nums[i]))

def classes(dataset_dir = '../CamVid'):
    # stas the classes in the annotation files
    train = 'trainannot'
    test = 'testannot'
    val = 'valannot'
    train = os.path.join(dataset_dir, train)
    test = os.path.join(dataset_dir, test)
    val = os.path.join(dataset_dir, val)

    classes_core(train)
    classes_core(val)
    classes_core(test)


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

    IoU_0 = dict(zip(CLASSES_NAME, IoU_0))
    mIoU = np.mean(IoU)
    return mIoU, IoU_0


if __name__ == '__main__':
    #classes()
    pred = np.array([[[[1, 1],
                      [2, 2]],
                     [[1, 1],
                      [2, 2]]
                     ],[[[1, 1],
                      [2, 2]],
                     [[1, 1],
                      [2, 2]]
                     ]])

    anno = np.array([[[[1, 0],
                      [2, 2]],
                     [[1, 1],
                      [2, 2]]
                     ],[[[1, 1],
                      [2, 2]],
                     [[1, 1],
                      [2, 2]]
                     ]])

    mIoU, IoU = cal_batch_mIoU(pred, anno, 3)
    print(mIoU)
    print(IoU)