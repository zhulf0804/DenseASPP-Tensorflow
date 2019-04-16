# coding=utf-8
from __future__ import print_function
from __future__ import division

import numpy as np
from PIL import Image
import os
import glob


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




if __name__ == '__main__':
    classes()