#coding=utf-8

from __future__ import print_function
from __future__ import division

import os
import glob
import numpy as np
from PIL import Image
import argparse
from collections import namedtuple


CITYSCAPE_DIR = '/Volumes/Samsung_T5/Cityscape' # cofig your data path


CITYSCAPE_IMG_DIR = os.path.join(CITYSCAPE_DIR, 'leftImg8bit')
CITYSCAPE_ANNO_DIR = os.path.join(CITYSCAPE_DIR, 'gtFine')

types = ['train', 'val', 'test']

SAVED_DIR = CITYSCAPE_DIR
SAVED_IMG_TRAIN_FILE = os.path.join(SAVED_DIR, 'img_train.txt')
SAVED_IMG_VAL_FILE = os.path.join(SAVED_DIR, 'img_val.txt')
SAVED_IMG_TEST_FILE = os.path.join(SAVED_DIR, 'img_test.txt')
SAVED_IMG_FILES = [SAVED_IMG_TRAIN_FILE, SAVED_IMG_VAL_FILE, SAVED_IMG_TEST_FILE]

SAVED_ANNO_TRAIN_FILE = os.path.join(SAVED_DIR, 'anno_train.txt')
SAVED_ANNO_VAL_FILE = os.path.join(SAVED_DIR, 'anno_val.txt')
SAVED_ANNO_TEST_FILE = os.path.join(SAVED_DIR, 'anno_test.txt')
SAVED_ANNO_FILES = [SAVED_ANNO_TRAIN_FILE, SAVED_ANNO_VAL_FILE, SAVED_ANNO_TEST_FILE]

def get_anno_file_list():
    for i in range(len(types)):
        CITYSCAPE_ANNO_DIR_type = os.path.join(CITYSCAPE_ANNO_DIR, types[i])
        dirs = os.listdir(CITYSCAPE_ANNO_DIR_type)

        f = open(SAVED_ANNO_FILES[i], 'w')

        length = 0
        for dir in dirs:
            CITYSCAPE_ANNO_DIR_type_dir = os.path.join(CITYSCAPE_ANNO_DIR_type, dir)
            img_files_path = glob.glob(os.path.join(CITYSCAPE_ANNO_DIR_type_dir, '*_labelTrainIds.png'))
            length += len(img_files_path)
            for img_file_path in img_files_path:
                f.write(img_file_path + '\n')
        print("image %s files: %d" %(types[i], length))

def get_img_file_list():
    for i in range(len(types)):
        CITYSCAPE_IMG_DIR_type = os.path.join(CITYSCAPE_IMG_DIR, types[i])
        dirs = os.listdir(CITYSCAPE_IMG_DIR_type)

        f = open(SAVED_IMG_FILES[i], 'w')

        length = 0
        for dir in dirs:
            CITYSCAPE_IMG_DIR_type_dir = os.path.join(CITYSCAPE_IMG_DIR_type, dir)
            img_files_path = glob.glob(os.path.join(CITYSCAPE_IMG_DIR_type_dir, '*.png'))
            length += len(img_files_path)
            for img_file_path in img_files_path:
                f.write(img_file_path + '\n')
        print("image %s files: %d" %(types[i], length))

if __name__ == '__main__':

    get_img_file_list()
    get_anno_file_list()





